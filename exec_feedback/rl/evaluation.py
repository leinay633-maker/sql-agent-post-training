"""v0.4-only evaluation: frozen models first, independent grading afterwards."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core import Limits, digest
from ..storage import (CallBudget, append_record, atomic_json, exclusive_lock,
                       fingerprint, read_json, recover_records)
from .backend import LocalPolicy
from .rollout import rollouts
from .study import load_run


def frozen_adapters(run: Path, frozen: dict[str, Any]) -> dict[str, Path]:
    from ..models import model_identity

    if frozen["settings"]["phase"] != "study":
        raise ValueError("Pilot cannot evaluate validation data")
    adapters = {"S4": Path(frozen["warm_adapter"])}
    for seed in frozen["settings"]["seeds"]:
        root = run / f"E4_seed_{seed}"
        result = read_json(root / "result.json")
        expected = fingerprint({"run": frozen["identity"], "seed": seed, "method": "online-rloo"})
        if not result.get("training_completed") or not result.get("adapter_reload_verified"):
            raise ValueError("Freeze every registered seed before validation scoring")
        if result["identity"] != expected:
            raise ValueError("RL result belongs to a different protocol/run")
        adapter = root / "adapter"
        if model_identity(frozen["model"], adapter)["adapter_hash"] != result["adapter_hash"]:
            raise ValueError("Frozen RL adapter changed")
        adapters[f"E4_seed_{seed}"] = adapter
    return adapters


def evaluate_condition(run: Path, condition: str) -> dict[str, Any]:
    from ..datasets import validate_gold
    from ..evaluation import summarize
    from ..models import load_model, model_identity
    from ..pipeline import grade_episode

    frozen, tasks = load_run(run, training=False)
    settings = frozen["settings"]
    adapters = frozen_adapters(run, frozen)
    if condition not in adapters:
        raise ValueError("Unknown v0.4 condition")
    limits = Limits(**settings["limits"])
    # Independent condition processes share one durable gold preflight. Wait
    # only here; each model/trajectory directory still has its own writer lock.
    checked = validate_gold(tasks, Path(frozen["raw_root"]), run / "gold_validation", limits,
                            lock_timeout=settings["eval_seconds"])
    if checked["invalid_ids"]:
        raise ValueError("Validation references invalid; no task removed or training restarted")
    out = run / f"eval_{condition}"
    provider = {"protocol": frozen["protocol"], "run_identity": frozen["identity"],
                "condition": condition, "task_ids": [t.id for t in tasks],
                "adapter_hash": model_identity(frozen["model"], adapters[condition])["adapter_hash"],
                "backend": "hf-v04-bucketed", "batch": settings["generation_batch"],
                "slots": settings["slots"], "max_steps": 2,
                "max_tokens": settings["max_tokens"], "greedy": True}
    identity = fingerprint(provider)
    with exclusive_lock(out / ".lock"):
        meta = out / "manifest.json"
        if meta.exists() and read_json(meta)["identity"] != identity:
            raise ValueError("Evaluation identity changed")
        atomic_json(meta, {**provider, "identity": identity})
        if (out / "done.json").exists():
            done = read_json(out / "done.json")
            if done["identity"] != identity:
                raise ValueError("Completed evaluation identity changed")
            for name, value in done["files"].items():
                if digest(out / name) != value:
                    raise ValueError("Completed evaluation artifact changed")
            return read_json(out / "summary.json")
        records = recover_records(out / "episodes.jsonl")
        seen = {r["task_id"] for r in records}
        if len(seen) != len(records) or not seen <= set(provider["task_ids"]):
            raise ValueError("Duplicate/foreign evaluation task id")
        model, tokenizer = load_model(frozen["model"], adapters[condition])
        budget = CallBudget(out / "budget.json", settings["eval_calls"], settings["eval_seconds"])
        policy = LocalPolicy(model, tokenizer, batch_size=settings["generation_batch"],
                             max_context=settings["max_context"], max_tokens=settings["max_tokens"],
                             sample=False, budget=budget)
        pending = [t for t in tasks if t.id not in seen]
        lookup = {t.id: t for t in tasks}
        started = time.perf_counter()
        previous = 0.0
        if (out / "summary.json").exists():
            previous = read_json(out / "summary.json").get("active_wall_seconds", 0.0)
        try:
            for episode in rollouts(pending, Path(frozen["raw_root"]), policy,
                                    slots=settings["slots"], limits=limits):
                task = lookup[episode["task_id"]]
                row = episode["record"]
                if episode["generation_failed"]:
                    # A length-truncated generation never earns an answer credit.
                    row.pop("result", None)
                    row["final_sql"] = None
                row = grade_episode(task, task.database(Path(frozen["raw_root"])),
                                    "model", row, limits, 2, True)
                row["condition"] = condition
                append_record(out / "episodes.jsonl", row)
                records.append(row)
        finally:
            summary = summarize(records, len(tasks), provider)
            summary.update(protocol=frozen["protocol"], identity=identity,
                           active_wall_seconds=previous + time.perf_counter() - started,
                           measured_runtime=policy.metadata)
            atomic_json(out / "summary.json", summary)
        atomic_json(out / "done.json", {"identity": identity, "files": {
            name: digest(out / name) for name in ("episodes.jsonl", "summary.json")}})
        return summary


def benchmark(run: Path) -> dict[str, Any]:
    """Same train questions/model/token budget; batch4 FIFO vs batch16 bucketing."""
    import torch
    from ..models import load_model

    frozen, tasks = load_run(run, training=True)
    if frozen["settings"]["phase"] != "pilot":
        raise ValueError("Throughput tuning is train-pilot only")
    settings = frozen["settings"]
    out = run / "throughput"
    out.mkdir(exist_ok=False)
    model, tokenizer = load_model(frozen["model"], Path(frozen["warm_adapter"]))
    results = {}
    for name, batch, slots, bucket in (("batch4_fifo", 4, 4, False),
                                      ("bucketed", settings["generation_batch"], settings["slots"], True)):
        budget = CallBudget(out / f"{name}-budget.json",
                            2 * (len(tasks) + min(4, len(tasks))),
                            settings["train_seconds"])
        policy = LocalPolicy(model, tokenizer, batch_size=batch,
                             max_context=settings["max_context"], max_tokens=settings["max_tokens"],
                             sample=False, bucket=bucket, budget=budget)
        # Identical train warmup excluded from timed counters; no gold grading.
        list(rollouts(tasks[:4], Path(frozen["raw_root"]), policy,
                      slots=slots, limits=Limits(**settings["limits"])))
        policy.reset_metrics()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows = []
        for row in rollouts(tasks, Path(frozen["raw_root"]), policy,
                            slots=slots, limits=Limits(**settings["limits"])):
            append_record(out / f"{name}.jsonl", row)
            rows.append(row)
        elapsed = time.perf_counter() - started
        results[name] = {**policy.metadata, "wall_seconds": elapsed,
                         "seconds_per_question": elapsed / len(tasks),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "questions": len(tasks), "batch": batch, "slots": slots,
                         "final_sql": {r["task_id"]: r["record"]["final_sql"] for r in rows},
                         "failed": sum(r["generation_failed"] or
                                       r["record"]["status"] != "submitted"
                                       for r in rows)}
        atomic_json(out / f"{name}.json", results[name])
    old, new = results["batch4_fifo"], results["bucketed"]
    report = {"protocol": frozen["protocol"], "run_identity": frozen["identity"],
              "split": "train", "gold_used": False, "quality_claimed": False,
              "runs": results, "wall_speed_ratio": old["wall_seconds"] / new["wall_seconds"],
              "different_sql": sum(old["final_sql"][t.id] != new["final_sql"][t.id] for t in tasks)}
    atomic_json(out / "comparison.json", report)
    return report
