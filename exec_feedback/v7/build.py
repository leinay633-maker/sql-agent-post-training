"""Frozen v0.7 training sets. T and R14 use the v0.5 rule unchanged (frontier tasks first, one execution-verified
trajectory per task, error turns and corrected wrong finishes unsupervised); TR adds the teacher's pick on every pool
task the 4B never solved in v0.5 round 1; T_raw never looks at gold."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from ..core import digest, read_jsonl, write_jsonl
from ..storage import atomic_json, fingerprint, read_json, recover_records
from ..rft.selection import supervised_turns, task_order, tokenize_trajectory
from ..rft.study import rank
from .freeze import imported, load


def load_samples(run: Path, kind: str, pool_ids: set[str]) -> list[dict[str, Any]]:
    folder = run / "samples" / kind / "samples"
    statuses = [read_json(p) for p in sorted(folder.glob("shard*.status.json"))]
    if not statuses or len({s["identity"] for s in statuses}) != 1:
        raise ValueError(f"No {kind} samples, or shards sampled different policies")
    rows, seen = [], set()
    for path in sorted(folder.glob("shard*.jsonl")):
        for row in recover_records(path):
            if row["key"] in seen or row["task_id"] not in pool_ids:
                raise ValueError("Duplicate or unregistered sample")
            seen.add(row["key"])
            rows.append(row)
    return rows


def raw_turns(row: dict[str, Any]) -> list[bool] | None:
    """Gold-free mask: a normally submitted trajectory, every assistant turn kept unless its feedback was an error."""
    if row["status"] != "submitted" or not row["steps"]:
        return None
    flags = []
    for step in row["steps"]:
        observation = step.get("observation")
        flags.append("action" in step and not (isinstance(observation, dict) and observation.get("status", "ok") != "ok"))
    return flags if flags[-1] else None


def example(row: dict[str, Any], flags: list[bool], tokenizer: Any, max_length: int, task: Any,
            origin: str, frontier: bool) -> dict[str, Any] | None:
    tokens = tokenize_trajectory(row["messages"], flags, tokenizer, max_length)
    if tokens is None:
        return None
    return {"task_id": task.id, "db_id": task.db_id, "split": task.split, "source": task.source,
            "fixture": task.fixture, "key": row["key"], "origin": origin, "frontier": frontier, "clean": all(flags),
            "masked_turns": flags.count(False), "turns": len(flags),
            "supervised_tokens": sum(x != -100 for x in tokens["labels"]), **tokens}


def verified_picks(by_task: dict[str, list[dict[str, Any]]], order: list[str], frontier: set[str], tasks: dict,
                   tokenizer: Any, settings: dict[str, Any], tag: str, origin: str,
                   want: Callable[[str], bool], limit: int) -> tuple[list[dict[str, Any]], int]:
    """v0.5 pick: per task in `order`, the first (fixed-rank) correct trajectory that fits; stop at `limit`."""
    chosen, overlong = [], 0
    for task_id in order:
        if len(chosen) >= limit:
            break
        if not want(task_id):
            continue
        candidates = [(r, f) for r in by_task[task_id] if (f := supervised_turns(r)) is not None]
        candidates.sort(key=lambda item: rank(f"{settings['salt']}:pick:{tag}", item[0]["key"]))
        for row, flags in candidates:
            ex = example(row, flags, tokenizer, settings["max_length"], tasks[task_id], origin, task_id in frontier)
            if ex is None:
                overlong += 1
                continue
            chosen.append(ex)
            break
    return chosen, overlong


def stats(rows: list[dict[str, Any]], overlong: int, path: Path) -> dict[str, Any]:
    return {"examples": len(rows), "origins": dict(Counter(r.get("origin", "student") for r in rows)),
            "frontier_tasks": sum(bool(r["frontier"]) for r in rows),
            "repair_trajectories": sum(not r["clean"] for r in rows),
            "masked_turns": sum(r["masked_turns"] for r in rows), "turns": dict(Counter(r["turns"] for r in rows)),
            "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "input_tokens": sum(len(r["input_ids"]) for r in rows), "databases": len({r["db_id"] for r in rows}),
            "skipped_overlong": overlong, "sha256": digest(path)}


def sampling_summary(rows: list[dict[str, Any]], by_task: dict[str, list[dict[str, Any]]], histogram: Counter,
                     frontier: list[str], easy: list[str]) -> dict[str, Any]:
    return {"trajectories": len(rows), "sampled_tasks": len(by_task),
            "attempts_per_task": dict(Counter(len(v) for v in by_task.values())),
            "sampled_accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
            "pass_at_attempts": sum(any(r["correct"] for r in v) for v in by_task.values()) / max(1, len(by_task)),
            "pass_histogram": dict(histogram), "frontier_tasks": len(frontier), "easy_tasks": len(easy),
            "unsolved_tasks": len(by_task) - len(frontier) - len(easy),
            "statuses": dict(Counter(r["status"] for r in rows)), "verdicts": dict(Counter(r["verdict"] for r in rows)),
            "protocol_error_turns": sum(1 for r in rows for st in r["steps"]
                                        if isinstance(st.get("observation"), dict)
                                        and st["observation"].get("status") == "protocol_error"),
            "truncated_turns": sum(r.get("truncated_turns", 0) for r in rows)}


def build_teacher(run: Path, *, tokenizer: Any = None) -> dict[str, Any]:
    from ..models import load_tokenizer

    frozen, pool = load(run, "pool")
    s, salt = frozen["settings"], frozen["settings"]["salt"]
    out = run / "data"
    out.mkdir(exist_ok=True)
    if (out / "teacher.json").exists():
        raise ValueError("Teacher datasets already frozen")
    tasks = {t.id: t for t in pool}
    rows = load_samples(run, "teacher", set(tasks))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tokenizer = tokenizer or load_tokenizer(frozen["models"]["student"]["spec"])
    order, frontier, easy, histogram = task_order(by_task, salt, "teacher")
    n = frozen["n_examples"]
    # T: the v0.5 rule on the teacher's own outcomes, truncated to N = |R|.
    t_rows, t_overlong = verified_picks(by_task, order, set(frontier), tasks, tokenizer, s, "teacher", "teacher",
                                        lambda _: True, n)
    write_jsonl(out / "sft_T.jsonl", t_rows)
    # TR: every v0.5 R example on this pool, plus the same teacher pick on tasks the 4B never solved in round 1.
    outcomes = read_json(imported(run, frozen, "student_round1.json"))["tasks"]
    solved = {t for t, (_, c) in outcomes.items() if c > 0}
    r_rows = [{**r, "origin": "student"} for r in read_jsonl(imported(run, frozen, "sft_R.jsonl")) if r["task_id"] in tasks]
    gap, gap_overlong = verified_picks(by_task, order, set(frontier), tasks, tokenizer, s, "teacher", "teacher",
                                       lambda t: t not in solved, s["tr_max_examples"] - len(r_rows))
    if {r["task_id"] for r in r_rows} & {g["task_id"] for g in gap}:
        raise ValueError("TR gap examples overlap R tasks")
    write_jsonl(out / "sft_TR.jsonl", r_rows + gap)
    # T_raw: no gold at all - any normally submitted trajectory, random task order, error turns masked.
    raw, raw_overlong = [], 0
    raw_order = sorted((t for t in by_task if any(raw_turns(r) is not None for r in by_task[t])),
                       key=lambda t: rank(f"{salt}:raw-order", t))
    for task_id in raw_order:
        if len(raw) >= n:
            break
        candidates = [(r, f) for r in by_task[task_id] if (f := raw_turns(r)) is not None]
        candidates.sort(key=lambda item: rank(f"{salt}:raw-pick", item[0]["key"]))
        for row, flags in candidates:
            ex = example(row, flags, tokenizer, s["max_length"], tasks[task_id], "teacher_raw", False)
            if ex is None:
                raw_overlong += 1
                continue
            raw.append(ex)
            break
    write_jsonl(out / "sft_Traw.jsonl", raw)
    if not t_rows or not raw:
        raise ValueError("No usable teacher trajectories")
    summary = {"run": frozen["identity"], "kind": "teacher", "n_examples": n,
               "sampling": sampling_summary(rows, by_task, histogram, frontier, easy),
               "student_round1_solved_tasks": len(solved & set(tasks)),
               "gap_tasks_with_teacher_sample": sum(1 for t in by_task if t not in solved),
               "datasets": {"T": stats(t_rows, t_overlong, out / "sft_T.jsonl"),
                            "TR": {**stats(r_rows + gap, gap_overlong, out / "sft_TR.jsonl"),
                                   "student_examples": len(r_rows), "teacher_gap_examples": len(gap)},
                            "Traw": {**stats(raw, raw_overlong, out / "sft_Traw.jsonl"),
                                     "wrong_final_answers": sum(1 for r in raw for row in by_task[r["task_id"]]
                                                                if row["key"] == r["key"] and not row["correct"])}},
               "tokenizer_chat_template_sha256": fingerprint(tokenizer.chat_template)}
    summary["identity"] = fingerprint(summary)
    atomic_json(out / "teacher.json", summary)
    return summary


def build_scale(run: Path, *, tokenizer: Any = None) -> dict[str, Any]:
    from ..models import load_tokenizer

    frozen, pool = load(run, "pool")
    s, salt = frozen["settings"], frozen["settings"]["salt"]
    out = run / "data"
    out.mkdir(exist_ok=True)
    if (out / "scale.json").exists():
        raise ValueError("14B dataset already frozen")
    tasks = {t.id: t for t in pool}
    rows = load_samples(run, "scale", set(tasks))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tokenizer = tokenizer or load_tokenizer(frozen["models"]["scale"]["spec"])
    order, frontier, easy, histogram = task_order(by_task, salt, "scale")
    chosen, overlong = verified_picks(by_task, order, set(frontier), tasks, tokenizer, s, "scale", "scale",
                                      lambda _: True, frozen["n_examples"])
    if not chosen:
        raise ValueError("No usable 14B trajectories")
    write_jsonl(out / "sft_R14.jsonl", chosen)
    summary = {"run": frozen["identity"], "kind": "scale", "n_examples": frozen["n_examples"],
               "sampling": sampling_summary(rows, by_task, histogram, frontier, easy),
               "datasets": {"R14": stats(chosen, overlong, out / "sft_R14.jsonl")},
               "tokenizer_chat_template_sha256": fingerprint(tokenizer.chat_template)}
    summary["identity"] = fingerprint(summary)
    atomic_json(out / "scale.json", summary)
    return summary


DATASETS = {"T": "teacher", "TR": "teacher", "Traw": "teacher", "R14": "scale"}


def dataset(run: Path, frozen: dict[str, Any], family: str) -> list[dict[str, Any]]:
    summary = read_json(run / "data" / f"{DATASETS[family]}.json")
    path = run / "data" / f"sft_{family}.jsonl"
    if summary["run"] != frozen["identity"] or digest(path) != summary["datasets"][family]["sha256"]:
        raise ValueError(f"Frozen {family} dataset changed")
    rows = read_jsonl(path)
    if any(r["split"] != "train" or r["fixture"] or r["source"] != "bird-train" for r in rows):
        raise ValueError("Only audited BIRD train trajectories may be trained on")
    return rows
