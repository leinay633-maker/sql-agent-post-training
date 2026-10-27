"""v0.9 greedy evaluation (vLLM, the v0.6-v0.8 path): the base 4B, the imported PTR adapters and the online runs at
their registered milestone, on dev, validation or test; and the common milestone K the online families are compared at.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from ..v6.rollouts import deadline_from, versions
from ..v7.freeze import _adapter_hash
from . import METHODS, START
from .freeze import load, model_ready, start_adapter
from .online import gold_for, milestones, registered, run_name

CONDITION = re.compile(rf"(?P<base>B)|(?P<family>{START}|{'|'.join(METHODS)})_seed(?P<seed>\d+)(@(?P<step>\d+))?")
MAIN = ("RL", "RAFT")


def parse(condition: str) -> tuple[str | None, int | None, int | None]:
    """'B' -> (None, None, None); 'RL_seed42' -> ('RL', 42, None); 'RL_seed42@60' -> ('RL', 42, 60)."""
    match = CONDITION.fullmatch(condition)
    if match is None or (match.group("family") == START and match.group("step")):
        raise ValueError(f"Unknown condition: {condition}")
    if match.group("base"):
        return None, None, None
    return match.group("family"), int(match.group("seed")), int(match.group("step")) if match.group("step") else None


def common_steps(run: Path) -> dict[str, Any]:
    """Preregistered: every online family is evaluated at K, the largest milestone every main run (RL and RAFT, all
    seeds) reached. A main run whose last milestone is below half of the best run's is reported as short and does not
    pull K down; it is evaluated at its own last milestone. RLhi runs at min(K, their own last milestone)."""
    frozen, _ = load(run)
    s = frozen["settings"]
    last = {}
    for family in METHODS:
        for seed in registered(s, family):
            marks = milestones(run / "train" / run_name(family, seed))
            last[run_name(family, seed)] = max(marks) if marks else 0
    main = [run_name(f, seed) for f in MAIN for seed in s["seeds"]]
    top = max((last[n] for n in main), default=0)
    kept = [n for n in main if last[n] > 0 and 2 * last[n] >= top]
    k = min((last[n] for n in kept), default=0)
    chosen = {}
    for name, reached in last.items():
        if not reached:
            continue
        if name in main:
            step = k if name in kept else reached
        else:
            step = min(k, reached) if k else reached
        chosen[name] = {"step": step, "adapter_hash": milestones(run / "train" / name)[step],
                        "short": name in main and name not in kept}
    result = {"run": frozen["identity"], "k": k, "last_milestones": last, "short": [n for n in main if n not in kept],
              "evaluate": chosen, "rule": "K = min last milestone over main runs reaching >= half of the best one"}
    atomic_json(run / "steps.json", result)
    return result


def adapter_for(run: Path, frozen: dict[str, Any], condition: str) -> tuple[Path | None, str | None, int | None]:
    family, seed, step = parse(condition)
    if family is None:
        return None, None, None
    if seed not in (frozen["settings"]["seeds"] if family == START else registered(frozen["settings"], family)):
        raise ValueError(f"Unregistered seed in {condition}")
    if family == START:
        folder, expected = start_adapter(run, frozen, seed)
        return folder, expected, 0
    name = run_name(family, seed)
    if step is None:
        chosen = read_json(run / "steps.json")
        if chosen["run"] != frozen["identity"] or name not in chosen["evaluate"]:
            raise ValueError(f"No registered milestone for {name}")
        step, expected = chosen["evaluate"][name]["step"], chosen["evaluate"][name]["adapter_hash"]
    else:
        marks = milestones(run / "train" / name)
        if step not in marks:
            raise ValueError(f"{name} has no milestone at step {step}")
        expected = marks[step]
    manifest = read_json(run / "train" / name / "manifest.json")
    if manifest["run"] != frozen["identity"]:
        raise ValueError(f"{name} belongs to another run")
    folder = run / "train" / name / "milestones" / f"step-{step:04d}" / "adapter"
    if _adapter_hash(folder) != expected:
        raise ValueError(f"Milestone adapter changed: {name}@{step}")
    return folder, expected, step


def vllm_factory(spec: dict[str, Any], settings: dict[str, Any], *, adapter: Path | None, mem: float, seed: int,
                 name: str) -> Any:
    from ..v6.engine import lora_request
    from ..v8.engine import generator, make_engine

    llm, tokenizer = make_engine(spec, settings, "student", mem=mem, seed=seed, lora=adapter is not None,
                                 max_num_seqs=settings["eval_max_num_seqs"])
    return generator(llm, tokenizer, spec, settings, sampling=None,
                     lora=lora_request(name, adapter) if adapter is not None else None)


def evaluate(run: Path, *, condition: str, role: str, stop: str | None, mem: float | None = None,
             factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load(run, role)
    s = frozen["settings"]
    spec = model_ready(frozen) if factory is vllm_factory else frozen["models"]["student"]["spec"]
    adapter, adapter_hash, step = adapter_for(run, frozen, condition)
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    gold = gold_for(run, frozen, role)
    # Per-condition order so concurrent evaluations do not hit one large database together.
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{condition}:{role}:{t.id}".encode()).digest())
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "model": frozen["models"]["student"]["identity"], "adapter_hash": adapter_hash, "step": step,
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
    generator = factory(spec, s, adapter=adapter, mem=mem if mem is not None else s["eval_gpu_memory"],
                        seed=20261014, name=condition.replace("@", "-at-"))
    load_seconds = time.monotonic() - started_load

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        row = grade_episode(task, task.database(root), "model", record, limits, s["max_steps"], True,
                            gold=gold.get(task.id))
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
               "generator": getattr(generator, "metadata", {}), "gold_cache_entries": len(gold),
               "packages": versions(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "episodes_sha256": digest(journal) if journal.exists() else None}
    atomic_json(out / ("summary.json" if summary["complete"] else "partial.json"), summary)
    return summary


def merged(run: Path, role: str, condition: str) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """A complete evaluation -> rows; otherwise (None, info)."""
    out = run / "eval" / role / condition
    if not (out / "summary.json").exists():
        return None, {"missing": True}
    summary = read_json(out / "summary.json")
    if digest(out / "episodes.jsonl") != summary["episodes_sha256"]:
        raise ValueError(f"Evaluation changed: {role}/{condition}")
    return recover_records(out / "episodes.jsonl"), {
        "wall_seconds": summary["wall_seconds"], "engine_load_seconds": summary["engine_load_seconds"],
        "step": summary.get("step"),
        "generator": {k: summary["generator"].get(k, 0) for k in (
            "calls", "generate_seconds", "prompt_tokens", "cached_prompt_tokens", "completion_tokens",
            "length_stops", "context_rejections")}}
