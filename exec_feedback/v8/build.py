"""Frozen v0.8 training sets: every v0.5 R example (the tasks the 4B solved in round 1) plus one execution-verified
trajectory per task the 4B never solved, picked by the v0.5 rule (frontier tasks first, error turns and corrected
wrong finishes unsupervised).

TR takes those trajectories from the teacher. HR takes them from the 4B itself, sampled with the gold SQL as a private
hint; the hint fields are removed from the prompt before tokenization, so HR trains on the ordinary v0.5 prompt."""
from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import digest, read_jsonl, write_jsonl
from ..storage import atomic_json, fingerprint, read_json, recover_records
from ..rft.selection import supervised_turns, task_order, tokenize_trajectory
from ..rft.study import rank
from . import HINT_KEYS
from .freeze import imported, load

FAMILY_OF = {"teacher": "TR", "hint": "HR"}
SOURCE_OF = {v: k for k, v in FAMILY_OF.items()}


def load_samples(run: Path, sampler: str, pool_ids: set[str]) -> list[dict[str, Any]]:
    folder = run / "samples" / sampler / "samples"
    statuses = [read_json(p) for p in sorted(folder.glob("shard*.status.json"))]
    if not statuses or len({s["identity"] for s in statuses}) != 1:
        raise ValueError(f"No {sampler} samples, or shards sampled different policies")
    rows, seen = [], set()
    for path in sorted(folder.glob("shard*.jsonl")):
        for row in recover_records(path):
            if row["key"] in seen or row["task_id"] not in pool_ids:
                raise ValueError("Duplicate or unregistered sample")
            seen.add(row["key"])
            rows.append(row)
    return rows


def strip_hint(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The hinted prompt minus the hint: json.dumps of the same public dict without the two trailing keys, i.e. the
    exact text messages_for() writes without `extra`."""
    public = json.loads(messages[1]["content"])
    if list(public) != ["question", "evidence", "schema", *HINT_KEYS]:
        raise ValueError("Hinted prompt is not the v0.5 prompt plus the hint fields")
    for key in HINT_KEYS:
        del public[key]
    clean = copy.deepcopy(messages)
    clean[1]["content"] = json.dumps(public, ensure_ascii=False)
    return clean


def example(row: dict[str, Any], flags: list[bool], tokenizer: Any, max_length: int, task: Any, origin: str,
            frontier: bool) -> dict[str, Any] | None:
    messages = strip_hint(row["messages"]) if row.get("hinted") else row["messages"]
    tokens = tokenize_trajectory(messages, flags, tokenizer, max_length)
    if tokens is None:
        return None
    return {"task_id": task.id, "db_id": task.db_id, "split": task.split, "source": task.source,
            "fixture": task.fixture, "key": row["key"], "origin": origin, "frontier": frontier, "clean": all(flags),
            "masked_turns": flags.count(False), "turns": len(flags),
            "supervised_tokens": sum(x != -100 for x in tokens["labels"]), **tokens}


def stats(rows: list[dict[str, Any]], overlong: int, path: Path) -> dict[str, Any]:
    return {"examples": len(rows), "origins": dict(Counter(r.get("origin", "student") for r in rows)),
            "frontier_tasks": sum(bool(r["frontier"]) for r in rows),
            "repair_trajectories": sum(not r["clean"] for r in rows),
            "masked_turns": sum(r["masked_turns"] for r in rows), "turns": dict(Counter(r["turns"] for r in rows)),
            "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "input_tokens": sum(len(r["input_ids"]) for r in rows), "databases": len({r["db_id"] for r in rows}),
            "skipped_overlong": overlong, "sha256": digest(path)}


def first_actions(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter((r["steps"][0].get("action") or {}).get("action", "invalid") if r["steps"] else "none"
                        for r in rows))


def build(run: Path, *, sampler: str, tokenizer: Any = None) -> dict[str, Any]:
    from ..models import load_tokenizer

    family = FAMILY_OF[sampler]
    frozen, pool = load(run, "pool")
    s, salt = frozen["settings"], frozen["settings"]["salt"]
    out = run / "data"
    out.mkdir(exist_ok=True)
    if (out / f"{sampler}.json").exists():
        raise ValueError(f"{family} dataset already frozen")
    tasks = {t.id: t for t in pool}
    unsolved = set(frozen["unsolved_tasks"])
    rows = load_samples(run, sampler, tasks.keys() & unsolved)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tokenizer = tokenizer or load_tokenizer(frozen["models"]["student"]["spec"])
    order, frontier, easy, histogram = task_order(by_task, salt, sampler)
    r_rows = [{**r, "origin": "student"} for r in read_jsonl(imported(run, frozen, "sft_R.jsonl")) if r["task_id"] in tasks]
    if {r["task_id"] for r in r_rows} & unsolved:
        raise ValueError("A v0.5 R example sits on a task the 4B never solved")
    gap, overlong = [], 0
    for task_id in order:
        if len(gap) >= s["gap_max_examples"]:
            break
        candidates = [(r, f) for r in by_task[task_id] if (f := supervised_turns(r)) is not None]
        candidates.sort(key=lambda item: rank(f"{salt}:pick:{sampler}", item[0]["key"]))
        for row, flags in candidates:
            ex = example(row, flags, tokenizer, s["max_length"], tasks[task_id], sampler, task_id in set(frontier))
            if ex is None:
                overlong += 1
                continue
            gap.append(ex)
            break
    if not gap and not frozen["smoke_limit"]:  # a smoke subset may hold no solvable gap task at all
        raise ValueError(f"No usable {sampler} trajectories on the unsolved tasks")
    write_jsonl(out / f"sft_{family}.jsonl", r_rows + gap)
    picked = {g["key"] for g in gap}
    summary = {"run": frozen["identity"], "sampler": sampler, "family": family, "n_examples": frozen["n_examples"],
               "unsolved_tasks": len(unsolved & tasks.keys()), "sampled_tasks": len(by_task),
               "sampling": {"trajectories": len(rows),
                            "sampled_accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
                            "pass_at_k": sum(any(r["correct"] for r in v) for v in by_task.values()) / max(1, len(by_task)),
                            "pass_histogram": dict(histogram), "frontier_tasks": len(frontier),
                            "easy_tasks": len(easy), "statuses": dict(Counter(r["status"] for r in rows)),
                            "verdicts": dict(Counter(r["verdict"] for r in rows)),
                            "first_action": first_actions(rows),
                            "first_action_of_picks": first_actions([r for r in rows if r["key"] in picked]),
                            "truncated_turns": sum(r.get("truncated_turns", 0) for r in rows),
                            "protocol_error_turns": sum(1 for r in rows for st in r["steps"]
                                                        if isinstance(st.get("observation"), dict)
                                                        and st["observation"].get("status") == "protocol_error")},
               "datasets": {family: {**stats(r_rows + gap, overlong, out / f"sft_{family}.jsonl"),
                                     "student_examples": len(r_rows), "gap_examples": len(gap)}},
               "tokenizer_chat_template_sha256": fingerprint(tokenizer.chat_template)}
    summary["identity"] = fingerprint(summary)
    atomic_json(out / f"{sampler}.json", summary)
    return summary


def dataset(run: Path, frozen: dict[str, Any], family: str) -> list[dict[str, Any]]:
    sampler = SOURCE_OF[family]
    summary = read_json(run / "data" / f"{sampler}.json")
    path = run / "data" / f"sft_{family}.jsonl"
    if summary["run"] != frozen["identity"] or digest(path) != summary["datasets"][family]["sha256"]:
        raise ValueError(f"Frozen {family} dataset changed")
    rows = read_jsonl(path)
    if any(r["split"] != "train" or r["fixture"] or r["source"] != "bird-train" for r in rows):
        raise ValueError("Only audited BIRD train trajectories may be trained on")
    return rows
