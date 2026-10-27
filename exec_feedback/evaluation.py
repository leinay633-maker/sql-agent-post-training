"""Resumable evaluation: fixed task denominator, identity checks, durable records."""
from __future__ import annotations

import dataclasses
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import Limits, Task, audit, digest
from .datasets import dataset_id
from .pipeline import episode
from .storage import append_record, atomic_json, code_fingerprint, exclusive_lock, fingerprint, read_json, recover_records


def summarize(records: list[dict[str, Any]], total: int, identity: dict[str, Any]) -> dict[str, Any]:
    fixture = any(r.get("fixture") for r in records)
    statuses = dict(Counter(r["status"] for r in records))
    if len(records) < total:
        statuses["not_attempted"] = total - len(records)
    usage: dict[str, int] = defaultdict(int)
    for row in records:
        for key, value in row.get("usage", {}).items():
            usage[key] += value
    return {"run_kind": "engineering_fixture" if fixture else "model_evaluation",
            "complete": len(records) == total, "tasks": total, "completed": len(records),
            "correct": None if fixture else sum(bool(r["correct"]) for r in records),
            "strict_success_rate": None if fixture or not total else sum(bool(r["correct"]) for r in records) / total,
            "statuses": statuses, "usage": dict(usage),
            "usage_complete": all(all(k in r.get("usage", {}) for k in ("prompt_tokens", "completion_tokens")) for r in records),
            "recorded_model_steps": sum(len(r["steps"]) for r in records),
            "elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in records),
            "provider": identity, "denominator_policy": "all registered tasks; failures and unattempted retained"}


def evaluate_run(tasks: list[Task], root: Path, policy: Any, out: Path, *, max_steps: int = 4,
                 feedback: bool = True, limits: Limits = Limits(), condition: str = "baseline",
                 resume: bool = True, batch_size: int = 1, review_success: bool = False) -> dict[str, Any]:
    audit(tasks, root)
    if not tasks:
        raise ValueError("No evaluation tasks")
    existed = out.exists()
    if existed and not resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = policy.metadata.get("identity", {k: v for k, v in policy.metadata.items() if k not in {"calls", "usage"}})
    immutable = {"protocol": "efa-v3", "dataset_id": dataset_id(tasks), "code": code_fingerprint(),
                 "task_ids": [t.id for t in tasks], "max_steps": max_steps, "feedback": feedback,
                 "limits": dataclasses.asdict(limits), "condition": condition,
                 "batch_size": batch_size, "review_success": review_success if feedback else False, "provider": identity}
    signature = fingerprint(immutable)
    with exclusive_lock(out / ".lock"):
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path)["signature"] != signature:
                raise ValueError("Evaluation model/data/protocol/code changed; cannot resume")
        else:
            atomic_json(manifest_path, {**immutable, "signature": signature, "created_unix": time.time()})
        if (out / "completed.json").exists():
            done = read_json(out / "completed.json")
            if (done["signature"] != signature or
                digest(out / "episodes.jsonl") != done["episodes_sha256"] or
                digest(out / "summary.json") != done["summary_sha256"]):
                raise ValueError("Completed evaluation artifacts were modified")
            return read_json(out / "summary.json")
        records = recover_records(out / "episodes.jsonl")
        completed = {r["task_id"] for r in records}
        if len(completed) != len(records) or not completed <= set(immutable["task_ids"]):
            raise ValueError("Duplicate/unknown task in evaluation journal")
        def scalar_rows():
            for task in tasks:
                if task.id in completed:
                    continue
                if hasattr(policy, "set_task"):
                    policy.set_task(task.id)
                yield episode(task, root, policy, max_steps=max_steps, feedback=feedback,
                              limits=limits, review_success=review_success)
        from .batching import batched_episodes
        pending = [task for task in tasks if task.id not in completed]
        rows = scalar_rows() if batch_size == 1 else batched_episodes(pending, root, policy,
                    batch_size=batch_size, max_steps=max_steps, feedback=feedback,
                    limits=limits, review_success=review_success)
        batch_started = time.monotonic()
        try:
            for row in rows:
                row["condition"] = condition
                row["trace_sha256"] = fingerprint({"messages": row["messages"], "steps": row["steps"]})
                append_record(out / "episodes.jsonl", row)
                records.append(row)
                atomic_json(out / "summary.json", summarize(records, len(tasks), identity))
        finally:
            summary = summarize(records, len(tasks), identity)
            # Per-dialogue elapsed time overlaps during batching; it is not wall time.
            old_timing = read_json(out / "timing.json") if (out / "timing.json").exists() else {"active_seconds": 0.0}
            old_timing["active_seconds"] += time.monotonic() - batch_started
            atomic_json(out / "timing.json", old_timing)
            summary["active_wall_seconds"] = old_timing["active_seconds"]
            summary["measured_runtime"] = {k: v for k, v in policy.metadata.items()
                                          if k in {"calls", "usage", "generate_batches", "generate_seconds", "packages"}}
            summary["batch_size"] = batch_size
            atomic_json(out / "summary.json", summary)
        atomic_json(out / "completed.json", {"signature": signature, "completed": len(records),
                    "episodes_sha256": digest(out / "episodes.jsonl"), "summary_sha256": digest(out / "summary.json")})
        return summary
