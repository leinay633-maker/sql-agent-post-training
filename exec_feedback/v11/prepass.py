"""The pre-pass: every pool task sampled `prepass_samples` times with a seed's start (its PTR adapter in vLLM), with the
training protocol, budget and temperature, and scored by the train-only oracle. The tasks solved at least once and not
every time are that seed's boundary tasks, the only ones whose RLOO group has a non-zero advantage. A task whose gold
reference cannot be scored is left out; a dialogue rejected for its context length counts as wrong, as in training.

Resumable: tasks run in fixed chunks and a chunk's rows are journalled only once all of its dialogues finished.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from ..v6.rollouts import deadline_from, versions
from ..v9.online import TracingGenerator, vllm_engine
from . import START
from .freeze import load, model_ready, start_adapter


def prepass_dir(run: Path, seed: int) -> Path:
    return run / "prepass" / f"{START}_seed{seed}"


def chunk_order(pool: list[Task], salt: str, seed: int) -> list[Task]:
    return sorted(pool, key=lambda t: hashlib.sha256(f"{salt}:prepass:{seed}:{t.id}".encode()).digest())


def sample(engine: Any, tokenizer: Any, tasks: list[Task], root: Path, oracle: Any, settings: dict[str, Any],
           stop_ids: list[int], group: int, deadline: float | None) -> list[dict[str, Any]] | None:
    """G dialogues per task exactly as an online step samples them, without token traces; one summary row per
    dialogue, or None if the deadline cut the chunk short (nothing of it is kept)."""
    from ..rft.runner import run_dialogues

    generator = TracingGenerator(engine, tokenizer, max_context=settings["max_context"],
                                 max_tokens=settings["max_tokens"], temperature=settings["temperature"],
                                 stop_ids=stop_ids, trace=False)
    items = [(f"{t.id}#{k}", t) for t in tasks for k in range(group)]

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        verdict = oracle.score(task, record)["verdict"]
        return {"key": key, "task_id": task.id, "k": int(key.rsplit("#", 1)[1]), "verdict": verdict["status"],
                "correct": bool(verdict["correct"]), "valid": verdict["status"] != "oracle_error",
                "turns": len(record.get("steps", []))}

    rows = {row["key"]: row for row in run_dialogues(
        items, root, generator, slots=settings["eval_slots"], max_steps=settings["max_steps"], review_success=True,
        limits=Limits(**settings["limits"]), workers=settings["sql_workers"], finish=finish, deadline=deadline,
        fenced_json=True)}
    if set(rows) != {k for k, _ in items}:
        return None
    return [rows[k] for k, _ in items]


def summarise(rows: list[dict[str, Any]], pool: list[Task], samples: int) -> dict[str, Any]:
    """Per task: correct dialogues out of `samples`; boundary = scorable tasks solved 1..samples-1 times."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    counts, invalid = {}, []
    for task in pool:
        mine = by_task.get(task.id, [])
        if len(mine) != samples or len({r["k"] for r in mine}) != samples:
            raise ValueError(f"Pre-pass rows for {task.id}: expected {samples}, found {len(mine)}")
        if not all(r["valid"] for r in mine):
            invalid.append(task.id)
            continue
        counts[task.id] = sum(r["correct"] for r in mine)
    boundary = sorted(t for t, c in counts.items() if 0 < c < samples)
    return {"tasks": len(pool), "scorable": len(counts), "invalid": sorted(invalid),
            "histogram": {str(c): n for c, n in sorted(Counter(counts.values()).items())},
            "boundary": boundary, "boundary_share": len(boundary) / len(counts) if counts else 0.0,
            "boundary_sha256": hashlib.sha256(json.dumps(boundary).encode()).hexdigest(),
            "statuses": dict(Counter(r["verdict"] for r in rows)),
            "mean_turns": sum(r["turns"] for r in rows) / len(rows) if rows else 0.0}


def prepass(run: Path, *, seed: int, until: str | None = None,
            engine_factory: Callable[..., Any] = vllm_engine, tokenizer: Any = None) -> dict[str, Any]:
    from ..rft.sampling import Oracle
    from ..v7.engine import stop_ids as model_stop_ids

    frozen, pool = load(run, "pool")
    s = frozen["settings"]
    if seed not in s["seeds"]:
        raise ValueError("Unregistered seed")
    spec = model_ready(frozen) if engine_factory is vllm_engine else frozen["models"]["student"]["spec"]
    start, start_hash = start_adapter(run, frozen, seed)
    samples = s["prepass_samples"]
    identity = fingerprint({"run": frozen["identity"], "seed": seed, "start": start_hash, "samples": samples,
                            "temperature": s["temperature"], "model": frozen["models"]["student"]["identity"]})
    out = prepass_dir(run, seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
        raise ValueError("Pre-pass inputs changed; use a new run")
    if (out / "result.json").exists():
        return read_json(out / "result.json")  # a failed gate stays failed: the trainings read it and do not start
    atomic_json(out / "manifest.json", {"identity": identity, "run": frozen["identity"], "seed": seed,
                                        "start_adapter": str(start), "start_hash": start_hash, "samples": samples,
                                        "packages": versions()})
    journal = out / "rows.jsonl"
    rows = recover_records(journal) if journal.exists() else []
    done = Counter(r["task_id"] for r in rows)
    if any(n < samples for n in done.values()):  # a crash while journalling a chunk: keep the rows aside, redo it
        journal.rename(journal.with_name(f"rows.partial-{time.time_ns()}.jsonl"))
        rows = [r for r in rows if done[r["task_id"]] == samples]
        for row in rows:
            append_record(journal, row)
        done = Counter(r["task_id"] for r in rows)
    order = chunk_order(pool, s["salt"], seed)
    todo = [t for t in order if done.get(t.id, 0) < samples]
    deadline = deadline_from(until)
    root, limits = Path(frozen["raw_root"]), Limits(**s["limits"])
    started = time.monotonic()
    meta: dict[str, Any] = {}
    if todo and deadline is not None and started >= deadline:  # past the deadline: no engine started for nothing
        append_record(out / "attempts.jsonl", {"stopped_by": "deadline", "remaining": len(todo), "wall_seconds": 0.0})
        return {"finished": False, "stopped_by": "deadline", "remaining": len(todo)}
    if todo:
        if tokenizer is None:
            from ..models import load_tokenizer
            tokenizer = load_tokenizer(spec)
        # the evaluations' engine budget: a pre-pass shares its GPU with another pre-pass or evaluation
        engine = engine_factory(spec, {**s, "rl_gpu_memory": s["eval_gpu_memory"],
                                       "rl_max_num_seqs": s["eval_max_num_seqs"]}, seed=20261021 + 1009 * seed)
        engine.set_policy(start, 0)
        oracle, stops = Oracle(pool, root, limits), model_stop_ids(spec, tokenizer)
        chunk = s["prepass_chunk"]
        for i in range(0, len(todo), chunk):
            part = sample(engine, tokenizer, todo[i:i + chunk], root, oracle, s, stops, samples, deadline)
            if part is None:
                append_record(out / "attempts.jsonl", {"stopped_by": "deadline", "remaining": len(todo) - i,
                                                       "wall_seconds": time.monotonic() - started})
                return {"finished": False, "stopped_by": "deadline", "remaining": len(todo) - i}
            for row in part:
                append_record(journal, row)
            rows.extend(part)
        meta = {"wall_seconds": time.monotonic() - started, "dialogues": len(todo) * samples}
    summary = summarise(rows, pool, samples)
    low, high = s["boundary_gate"]
    passed = s["phase"] != "study" or low <= summary["boundary_share"] <= high
    result = {"finished": True, "identity": identity, "seed": seed, "start_hash": start_hash, "samples": samples,
              **summary, "gate": {"range": [low, high], "passed": passed}, "last_attempt": meta}
    atomic_json(out / "result.json", result)
    return result


def gate_failed(run: Path, seed: int) -> bool:
    path = prepass_dir(run, seed) / "result.json"
    return path.exists() and not read_json(path)["gate"]["passed"]


def boundary_tasks(run: Path, frozen: dict[str, Any], pool: list[Task], seed: int) -> tuple[list[Task], str]:
    """The seed's boundary tasks from its finished, gated pre-pass, re-checked against its record. A smoke run whose
    tiny pool has no boundary task trains on the whole smoke pool (its numbers are never results)."""
    out = prepass_dir(run, seed)
    if not (out / "result.json").exists():
        raise ValueError(f"No finished pre-pass for seed {seed}")
    result = read_json(out / "result.json")
    manifest = read_json(out / "manifest.json")
    if manifest["run"] != frozen["identity"] or result["identity"] != manifest["identity"]:
        raise ValueError("The pre-pass belongs to another run")
    if not result["gate"]["passed"]:
        raise ValueError(f"The pre-pass of seed {seed} failed its gate; nothing trains")
    ids = result["boundary"]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != result["boundary_sha256"]:
        raise ValueError("Boundary list changed")
    by_id = {t.id: t for t in pool}
    if not ids and frozen["settings"]["phase"] == "smoke":
        return list(pool), result["boundary_sha256"]
    if not ids or any(i not in by_id for i in ids):
        raise ValueError("Boundary tasks are not pool tasks")
    return [by_id[i] for i in ids], result["boundary_sha256"]
