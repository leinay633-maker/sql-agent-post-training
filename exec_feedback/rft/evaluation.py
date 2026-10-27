"""Resumable greedy evaluation of frozen v0.5 conditions on a registered task role."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..core import Limits, digest, execute
from ..storage import (append_record, atomic_json, code_fingerprint, exclusive_lock, fingerprint,
                       read_json, recover_records)
from . import PROTOCOL
from .study import load_run
from .training import model_dir

CONDITION = re.compile(r"(R|Rclean|P)_r([12])_seed(\d+)")


def frozen_adapter(run: Path, frozen: dict[str, Any], condition: str) -> Path | None:
    from ..models import model_identity

    if condition == "B":
        return None
    match = CONDITION.fullmatch(condition)
    if match is None:
        raise ValueError(f"Unknown condition: {condition}")
    variant, round_, seed = match.group(1), int(match.group(2)), int(match.group(3))
    out = model_dir(run, round_, variant, seed)
    result, manifest = read_json(out / "result.json"), read_json(out / "manifest.json")
    if not result.get("training_completed") or not result.get("adapter_reload_verified"):
        raise ValueError("Freeze the trained model before evaluating it")
    if result["identity"] != manifest["identity"] or manifest["run"] != frozen["identity"]:
        raise ValueError("Model belongs to a different run")
    if model_identity(frozen["model"], out / "adapter")["adapter_hash"] != result["adapter_hash"]:
        raise ValueError("Frozen adapter changed")
    return out / "adapter"


def evaluate(run: Path, condition: str, role: str) -> dict[str, Any]:
    import torch
    from ..evaluation import summarize
    from ..models import load_model, model_identity
    from ..pipeline import grade_episode
    from .runner import Generator, run_dialogues

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load_run(run, role)
    settings = frozen["settings"]
    if not tasks:
        raise ValueError(f"No registered {role} tasks in this run")
    adapter = frozen_adapter(run, frozen, condition)
    limits, root = Limits(**settings["limits"]), Path(frozen["raw_root"])
    identity = {"protocol": PROTOCOL, "run": frozen["identity"], "role": role, "condition": condition,
                "task_ids": [t.id for t in tasks], "base": frozen["model_identity"],
                "adapter_hash": model_identity(frozen["model"], adapter)["adapter_hash"] if adapter else None,
                "max_steps": settings["max_steps"], "review_success": True, "decoding": "greedy",
                "max_tokens": settings["max_tokens"], "max_context": settings["max_context"],
                "batch": settings["eval_batch"], "slots": settings["eval_slots"],
                "limits": settings["limits"], "code": code_fingerprint()}
    signature = fingerprint(identity)
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".lock"):
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json")["signature"] != signature:
                raise ValueError("Model/data/protocol/code changed; cannot resume this evaluation")
        else:
            atomic_json(out / "manifest.json", {**identity, "signature": signature, "created_unix": time.time()})
        if (out / "completed.json").exists():
            done = read_json(out / "completed.json")
            if (done["signature"] != signature or digest(out / "episodes.jsonl") != done["episodes_sha256"]
                    or digest(out / "summary.json") != done["summary_sha256"]):
                raise ValueError("Completed evaluation artifacts were modified")
            return read_json(out / "summary.json")
        records = recover_records(out / "episodes.jsonl")
        seen = {r["task_id"] for r in records}
        if len(seen) != len(records) or not seen <= set(identity["task_ids"]):
            raise ValueError("Duplicate/unknown task in evaluation journal")
        pending = [(t.id, t) for t in tasks if t.id not in seen]
        started = time.monotonic()
        model, tokenizer = load_model(frozen["model"], adapter)
        model.eval()
        generator = Generator(model, tokenizer, batch_size=settings["eval_batch"],
                              max_context=settings["max_context"], max_tokens=settings["max_tokens"])

        def finish(key: str, task: Any, record: dict[str, Any]) -> dict[str, Any]:
            row = grade_episode(task, task.database(root), "model", record, limits, settings["max_steps"], True)
            row["condition"] = condition
            if row["status"] == "incorrect":
                # Analysis only, after grading: wrong column count vs right shape with wrong rows.
                gold = execute(task.database(root), task.gold_sql, limits)
                predicted = (row.get("answer_preview") or {}).get("columns")
                if gold["status"] == "ok" and predicted is not None:
                    row["mismatch"] = "columns" if len(predicted) != len(gold["columns"]) else "rows"
            return row

        for row in run_dialogues(pending, root, generator, slots=settings["eval_slots"],
                                 max_steps=settings["max_steps"], review_success=True, limits=limits,
                                 workers=settings["sql_workers"], finish=finish):
            append_record(out / "episodes.jsonl", row)
            records.append(row)
        summary = summarize(records, len(tasks), identity)
        summary["generator"] = generator.metadata
        summary["wall_seconds_this_invocation"] = time.monotonic() - started
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "completed.json", {"signature": signature,
                                             "episodes_sha256": digest(out / "episodes.jsonl"),
                                             "summary_sha256": digest(out / "summary.json")})
        del model
        torch.cuda.empty_cache()
        return summary
