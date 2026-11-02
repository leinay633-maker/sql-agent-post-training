"""Greedy evaluation of the three deployments (the v0.6-v0.11 path: vLLM's offline engine, run_dialogues, the project
grader against the frozen gold cache): `lora` is the base 4B with the adapter, `merged` the merged bf16 weights, `fp8`
the merged weights quantized to FP8 by vLLM as it loads them."""
from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from ..v6.rollouts import deadline_from, versions
from . import CONDITIONS
from .freeze import adapter, gold, load, merged_model, model_ready


def engine_for(run: Path, frozen: dict[str, Any], condition: str, *, ready: bool = True
               ) -> tuple[dict[str, Any], Path | None, str | None, dict[str, Any]]:
    """(model spec, adapter or None, quantization or None, what was loaded) of a deployment, each part verified; `ready`
    also checks the downloaded 4B against its pin (CPU tests have none)."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown deployment: {condition}")
    base = model_ready(frozen) if ready else frozen["models"]["student"]["spec"]
    folder, adapter_hash, step = adapter(run, frozen)
    if condition == "lora":
        return base, folder, None, {"adapter_hash": adapter_hash, "step": step}
    merged, record = merged_model(run, frozen)
    return ({**base, "local_path": str(merged)}, None, "fp8" if condition == "fp8" else None,
            {"adapter_hash": adapter_hash, "step": step, "merged": record["identity"]})


def vllm_factory(spec: dict[str, Any], settings: dict[str, Any], *, adapter: Path | None, quantization: str | None,
                 name: str) -> Any:
    from ..v6.engine import lora_request
    from ..v8.engine import generator, make_engine

    llm, tokenizer = make_engine(spec, settings, "student", mem=settings["eval_gpu_memory"], seed=20261029,
                                 lora=adapter is not None, max_num_seqs=settings["eval_max_num_seqs"],
                                 quantization=quantization)
    return generator(llm, tokenizer, spec, settings, sampling=None,
                     lora=lora_request(name, adapter) if adapter is not None else None)


def evaluate(run: Path, *, condition: str, role: str, stop: str | None,
             factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    frozen, tasks = load(run, role)
    s = frozen["settings"]
    spec, folder, quantization, loaded = engine_for(run, frozen, condition, ready=factory is vllm_factory)
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    answers = gold(run, frozen, role)
    # one order for all three deployments, so their batches start out alike and differ only where their answers do
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{role}:{t.id}".encode()).digest())
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "model": frozen["models"]["student"]["identity"], **loaded, "quantization": quantization,
                "code": frozen["code"]}
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "episodes.jsonl"
    if (out / "summary.json").exists():
        return read_json(out / "summary.json")
    if journal.exists():  # an interrupted attempt: keep it for forensics, run again
        journal.rename(journal.with_name(journal.name + f".interrupted-{time.time_ns()}"))
    atomic_json(out / "meta.json", {**identity, "signature": fingerprint(identity)})
    started_load = time.monotonic()
    generator = factory(spec, s, adapter=folder, quantization=quantization, name=f"deploy-{condition}")
    load_seconds = time.monotonic() - started_load

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        row = grade_episode(task, task.database(root), "model", record, limits, s["max_steps"], True,
                            gold=answers[task.id])
        row["condition"] = condition
        return row

    statuses, started = Counter(), time.monotonic()
    for row in run_dialogues([(t.id, t) for t in tasks], root, generator, slots=s["eval_slots"],
                             max_steps=s["max_steps"], review_success=True, limits=limits, workers=s["sql_workers"],
                             finish=finish, deadline=deadline_from(stop), fenced_json=True):
        append_record(journal, row)
        statuses[row["status"]] += 1
    done = sum(statuses.values())
    summary = {**identity, "registered_tasks": len(tasks), "episodes": done, "complete": done == len(tasks),
               "statuses": dict(statuses), "correct": statuses.get("correct", 0),
               "wall_seconds": time.monotonic() - started, "engine_load_seconds": load_seconds,
               "generator": getattr(generator, "metadata", {}), "packages": versions(),
               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "episodes_sha256": digest(journal) if journal.exists() else None}
    atomic_json(out / ("summary.json" if summary["complete"] else "partial.json"), summary)
    return summary


def rows_of(run: Path, role: str, condition: str) -> list[dict[str, Any]] | None:
    """A complete evaluation's rows, re-checked against its summary; None if there is none."""
    out = run / "eval" / role / condition
    if not (out / "summary.json").exists():
        return None
    summary = read_json(out / "summary.json")
    if digest(out / "episodes.jsonl") != summary["episodes_sha256"]:
        raise ValueError(f"Evaluation changed: {role}/{condition}")
    return recover_records(out / "episodes.jsonl")
