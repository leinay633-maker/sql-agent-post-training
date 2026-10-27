"""Pick one execution-verified trajectory per task and build multi-turn loss masks."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import digest, write_jsonl
from ..storage import atomic_json, fingerprint, read_json, recover_records
from .study import load_run, rank

VARIANTS = ("R", "Rclean")


def supervised_turns(row: dict[str, Any]) -> list[bool] | None:
    """Per assistant turn: train on it unless its feedback was an error or it was a corrected wrong finish."""
    if not row["correct"] or row["status"] != "submitted":
        return None
    wrong = {check["step"] for check in row["finish_checks"] if not check["correct"]}
    flags = []
    for index, step in enumerate(row["steps"]):
        observation = step.get("observation")
        bad = ("action" not in step or index in wrong
               or (isinstance(observation, dict) and observation.get("status", "ok") != "ok"))
        flags.append(not bad)
    return flags if flags and flags[-1] else None


def tokenize_trajectory(messages: list[dict[str, str]], supervised: list[bool], tokenizer: Any,
                        max_length: int) -> dict[str, list[int]] | None:
    """Labels cover each kept assistant turn's content and its closing end-of-turn token."""
    positions = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(positions) != len(supervised) or positions[-1] != len(messages) - 1:
        raise ValueError("Assistant turns and step records disagree")
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    labels = [-100] * len(full)
    for turn, position in enumerate(positions):
        prefix = tokenizer.apply_chat_template(messages[:position], tokenize=True, add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(messages[:position + 1], tokenize=True, add_generation_prompt=False)
        if upto[:len(prefix)] != prefix or full[:len(upto)] != upto or len(upto) <= len(prefix):
            raise ValueError("Chat template is not prefix-consistent; refuse an incorrect loss mask")
        stop = next((i for i in range(len(prefix), len(upto)) if full[i] == end), None)
        if stop is None:
            raise ValueError("Assistant turn has no end-of-turn token")
        if supervised[turn]:
            labels[len(prefix):stop + 1] = full[len(prefix):stop + 1]
    if len(full) > max_length:
        return None
    return {"input_ids": full, "labels": labels}


def task_order(by_task: dict[str, list[dict[str, Any]]], salt: str, round_: int
               ) -> tuple[list[str], list[str], list[str], Counter]:
    """Frontier tasks (some but not all samples correct) first, then always-correct tasks."""
    frontier, easy, histogram = [], [], Counter()
    for task_id, samples in by_task.items():
        correct = sum(r["correct"] for r in samples)
        histogram[f"{correct}/{len(samples)}"] += 1
        if 0 < correct < len(samples):
            frontier.append(task_id)
        elif correct:
            easy.append(task_id)
    order = (sorted(frontier, key=lambda t: rank(f"{salt}:frontier:{round_}", t))
             + sorted(easy, key=lambda t: rank(f"{salt}:easy:{round_}", t)))
    return order, frontier, easy, histogram


def round_samples(run: Path, round_: int, pool_ids: set[str]) -> list[dict[str, Any]]:
    folder = run / f"round{round_}" / "samples"
    statuses = [read_json(p) for p in sorted(folder.glob("shard*.status.json"))]
    if not statuses or len({s["identity"] for s in statuses}) != 1:
        raise ValueError("Missing shards or shards sampled different policies")
    rows, seen = [], set()
    for path in sorted(folder.glob("shard*.jsonl")):
        for row in recover_records(path):
            if row["key"] in seen or row["task_id"] not in pool_ids:
                raise ValueError("Duplicate or unregistered sample")
            seen.add(row["key"])
            rows.append(row)
    return rows


def build(run: Path, round_: int) -> dict[str, Any]:
    from ..models import load_tokenizer

    frozen, pool = load_run(run, "pool")
    settings, salt = frozen["settings"], frozen["settings"]["salt"]
    out = run / f"round{round_}"
    if (out / "selection.json").exists():
        raise ValueError("Selection already frozen for this round")
    rows = round_samples(run, round_, {t.id for t in pool})
    tasks = {t.id: t for t in pool}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tokenizer = load_tokenizer(frozen["model"])
    order, frontier, easy, histogram = task_order(by_task, salt, round_)
    frontier_set = set(frontier)
    variants = VARIANTS if round_ == 1 else ("R",)
    stats: dict[str, Any] = {}
    for variant in variants:
        chosen, overlong = [], 0
        for task_id in order:
            candidates = []
            for row in by_task[task_id]:
                flags = supervised_turns(row)
                if flags is not None and (variant == "R" or all(flags)):
                    candidates.append((row, flags))
            candidates.sort(key=lambda item: rank(f"{salt}:pick:{round_}", item[0]["key"]))
            for row, flags in candidates:
                tokens = tokenize_trajectory(row["messages"], flags, tokenizer, settings["max_length"])
                if tokens is None:
                    overlong += 1
                    continue
                task = tasks[task_id]
                chosen.append({"task_id": task_id, "db_id": task.db_id, "split": task.split,
                               "source": task.source, "fixture": task.fixture, "key": row["key"],
                               "frontier": task_id in frontier_set, "clean": all(flags),
                               "masked_turns": flags.count(False), "turns": len(flags),
                               "supervised_tokens": sum(x != -100 for x in tokens["labels"]), **tokens})
                break
            if len(chosen) == settings["train_examples"]:
                break
        if not chosen:
            raise ValueError(f"No usable trajectories for {variant}")
        write_jsonl(out / f"sft_{variant}.jsonl", chosen)
        stats[variant] = {"examples": len(chosen), "registered_examples": settings["train_examples"],
                          "frontier_tasks": sum(r["frontier"] for r in chosen),
                          "repair_trajectories": sum(not r["clean"] for r in chosen),
                          "masked_turns": sum(r["masked_turns"] for r in chosen),
                          "turns": dict(Counter(r["turns"] for r in chosen)),
                          "supervised_tokens": sum(r["supervised_tokens"] for r in chosen),
                          "input_tokens": sum(len(r["input_ids"]) for r in chosen),
                          "databases": len({r["db_id"] for r in chosen}),
                          "skipped_overlong": overlong, "sha256": digest(out / f"sft_{variant}.jsonl")}
    attempts = Counter(len(samples) for samples in by_task.values())
    summary = {
        "round": round_, "run": frozen["identity"], "pool_tasks": len(pool), "sampled_tasks": len(by_task),
        "trajectories": len(rows), "attempts_per_task": dict(attempts),
        "sampled_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "pass_at_attempts": sum(any(r["correct"] for r in s) for s in by_task.values()) / len(by_task),
        "pass_histogram": dict(histogram), "frontier_tasks": len(frontier), "easy_tasks": len(easy),
        "unsolved_tasks": len(by_task) - len(frontier) - len(easy),
        "statuses": dict(Counter(r["status"] for r in rows)), "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "datasets": stats, "tokenizer_chat_template_sha256": fingerprint(tokenizer.chat_template),
    }
    summary["identity"] = fingerprint(summary)
    atomic_json(out / "selection.json", summary)
    return summary
