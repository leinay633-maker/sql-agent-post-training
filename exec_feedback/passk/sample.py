"""Sampling: every test task `samples` times with one model, the online runs' agent protocol and temperature, graded by
the project's grader against the frozen gold cache (the gold never enters a prompt). A dialogue rejected for its context
length counts as wrong, as in training.

Resumable: tasks run in fixed chunks and a chunk's rows are journalled only once all of its dialogues finished; a
deadline drops the chunk it cut short.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from ..v6.rollouts import deadline_from, versions
from .freeze import adapter_for, control_tasks, gold, load, model_ready, parse


def sample_dir(run: Path, condition: str) -> Path:
    return run / "samples" / condition


def chunk_order(tasks: list[Task], salt: str, condition: str) -> list[Task]:
    """One order per model, so two samplers on a GPU do not hit one large database together."""
    return sorted(tasks, key=lambda t: hashlib.sha256(f"{salt}:passk:{condition}:{t.id}".encode()).digest())


def engine_seed(salt: str, condition: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:engine:{condition}".encode()).digest()[:4], "big")


def decoding(settings: dict[str, Any], condition: str) -> dict[str, Any]:
    """The online runs' sampling, or the model card's recommended decoding for a control condition."""
    if parse(condition)[2]:
        return dict(settings["control"]["sampling"])
    return {"temperature": settings["temperature"], "top_p": settings["top_p"], "top_k": 0}


def vllm_sampler(spec: dict[str, Any], settings: dict[str, Any], *, adapter: Path | None, seed: int, name: str,
                 sampling: dict[str, Any]) -> Any:
    from ..v6.engine import lora_request
    from ..v8.engine import generator, make_engine

    llm, tokenizer = make_engine(spec, settings, "student", mem=settings["gpu_memory"], seed=seed,
                                 lora=adapter is not None, max_num_seqs=settings["max_num_seqs"],
                                 kv_cache_bytes=settings["kv_cache_gib"] << 30)
    # top_k 0 is no top-k cut (the generator passes top_k only when it is set)
    return generator(llm, tokenizer, spec, settings, sampling=sampling,
                     lora=lora_request(name, adapter) if adapter is not None else None)


def chunk(generator: Any, tasks: list[Task], root: Path, answers: dict[str, Any], settings: dict[str, Any],
          samples: int, deadline: float | None) -> list[dict[str, Any]] | None:
    """`samples` graded dialogues per task, one compact row each; None if the deadline cut the chunk short."""
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    limits = Limits(**settings["limits"])
    items = [(f"{t.id}#{k}", t) for t in tasks for k in range(samples)]

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        row = grade_episode(task, task.database(root), "model", record, limits, settings["max_steps"], True,
                            gold=answers[task.id])
        return {"key": key, "task_id": task.id, "db_id": task.db_id, "k": int(key.rsplit("#", 1)[1]),
                "status": row["status"], "correct": row["status"] == "correct", "turns": len(row.get("steps", [])),
                "final_sql": row.get("final_sql")}

    rows = {row["key"]: row for row in run_dialogues(
        items, root, generator, slots=settings["slots"], max_steps=settings["max_steps"], review_success=True,
        limits=limits, workers=settings["sql_workers"], finish=finish, deadline=deadline, fenced_json=True)}
    if set(rows) != {k for k, _ in items}:
        return None
    return [rows[k] for k, _ in items]


def summed(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """The generator's counters over every attempt (a resumed model's tokens are not only its last attempt's): numbers
    added up, the largest batch kept, other fields (engine, stop ids) from the last attempt that had them."""
    out: dict[str, Any] = {}
    for a in attempts:
        for key, value in (a.get("generator") or {}).items():
            if key == "max_batch":
                out[key] = max(out.get(key, 0), value)
            elif type(value) in (int, float):
                out[key] = out.get(key, 0) + value
            else:
                out[key] = value
    return out


def check_rows(rows: list[dict[str, Any]], tasks: list[Task], samples: int) -> None:
    by_task: dict[str, set[int]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], set()).add(row["k"])
    for task in tasks:
        if by_task.get(task.id) != set(range(samples)):
            raise ValueError(f"Rows for {task.id}: expected samples 0..{samples - 1}")
    if len(rows) != len(tasks) * samples or set(by_task) != {t.id for t in tasks}:
        raise ValueError("Sampled rows do not match the registered tasks")


def sample(run: Path, *, condition: str, until: str | None = None,
           factory: Callable[..., Any] = vllm_sampler) -> dict[str, Any]:
    frozen, tasks = load(run, tasks=True)
    s = frozen["settings"]
    adapter, adapter_hash = adapter_for(run, frozen, condition)
    spec = model_ready(frozen) if factory is vllm_sampler else frozen["models"]["student"]["spec"]
    samples, sampling = s["samples"], decoding(s, condition)
    if parse(condition)[2]:
        tasks = control_tasks(tasks, s["salt"], s["control"]["tasks"])
    identity = fingerprint({"run": frozen["identity"], "condition": condition, "adapter_hash": adapter_hash,
                            "samples": samples, "sampling": sampling, "tasks": sorted(t.id for t in tasks),
                            "model": frozen["models"]["student"]["identity"]})
    out = sample_dir(run, condition)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
        raise ValueError("Sampling inputs changed; use a new run")
    if (out / "result.json").exists():
        return read_json(out / "result.json")
    atomic_json(out / "manifest.json", {"identity": identity, "run": frozen["identity"], "condition": condition,
                                        "adapter": str(adapter) if adapter else None, "adapter_hash": adapter_hash,
                                        "samples": samples, "sampling": sampling, "tasks": len(tasks),
                                        "packages": versions()})
    journal = out / "rows.jsonl"
    rows = recover_records(journal) if journal.exists() else []
    done = Counter(r["task_id"] for r in rows)
    if any(n != samples for n in done.values()):  # a crash while journalling a chunk: keep the rows aside, redo it
        journal.rename(journal.with_name(f"rows.partial-{time.time_ns()}.jsonl"))
        rows = [r for r in rows if done[r["task_id"]] == samples]
        for row in rows:
            append_record(journal, row)
        done = Counter(r["task_id"] for r in rows)
    todo = [t for t in chunk_order(tasks, s["salt"], condition) if done.get(t.id, 0) < samples]
    deadline = deadline_from(until)
    started = time.monotonic()

    def attempt(stopped_by: str, sampled: int, load_seconds: float, generator: Any = None) -> None:
        append_record(out / "attempts.jsonl", {
            "stopped_by": stopped_by, "tasks_sampled": sampled, "remaining": len(todo) - sampled,
            "wall_seconds": time.monotonic() - started, "load_seconds": load_seconds,
            "generator": dict(getattr(generator, "metadata", {}) or {}), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})

    if todo and deadline is not None and started >= deadline:  # past the deadline: no engine started for nothing
        attempt("deadline", 0, 0.0)
        return {"finished": False, "stopped_by": "deadline", "remaining": len(todo)}
    if todo:
        answers, root = gold(run, frozen), Path(frozen["raw_root"])
        generator, load_seconds, sampled, stopped_by = None, None, 0, "finished"
        try:
            generator = factory(spec, s, adapter=adapter, seed=engine_seed(s["salt"], condition),
                                name=condition.replace("+", "-"), sampling=sampling)
            load_seconds = time.monotonic() - started
            step = s["chunk_tasks"]
            for i in range(0, len(todo), step):
                part = chunk(generator, todo[i:i + step], root, answers, s, samples, deadline)
                if part is None:
                    stopped_by = "deadline"
                    break
                for row in part:
                    append_record(journal, row)
                rows.extend(part)
                sampled += len(todo[i:i + step])
                elapsed = time.monotonic() - started - load_seconds
                atomic_json(out / "progress.json", {
                    "tasks": len(tasks), "done": len(tasks) - len(todo) + sampled,
                    "seconds_per_task": elapsed / sampled, "load_seconds": load_seconds,
                    "updated": dt.datetime.now(dt.timezone.utc).isoformat()})
        except BaseException as exc:  # an attempt that dies is recorded too: its time and tokens count in the result
            stopped_by = f"error: {type(exc).__name__}"
            raise
        finally:  # an engine that never came up spent the whole attempt loading
            attempt(stopped_by, sampled, time.monotonic() - started if load_seconds is None else load_seconds,
                    generator)
        if stopped_by == "deadline":
            return {"finished": False, "stopped_by": "deadline", "remaining": len(todo) - sampled}
    check_rows(rows, tasks, samples)
    attempts = recover_records(out / "attempts.jsonl") if (out / "attempts.jsonl").exists() else []
    statuses = Counter(r["status"] for r in rows)
    solved = Counter(r["task_id"] for r in rows if r["correct"])
    result = {"finished": True, "identity": identity, "condition": condition, "adapter_hash": adapter_hash,
              "samples": samples, "sampling": sampling, "tasks": len(tasks), "dialogues": len(rows),
              "rows_sha256": digest(journal),
              "statuses": dict(statuses), "correct_dialogues": statuses.get("correct", 0), "solved_tasks": len(solved),
              "sampling_seconds": sum(a["wall_seconds"] - a["load_seconds"] for a in attempts),
              "attempts": len(attempts), "generator": summed(attempts), "packages": versions()}
    atomic_json(out / "result.json", result)
    return result


def finished(run: Path, frozen: dict[str, Any], condition: str) -> list[dict[str, Any]] | None:
    """A finished model's rows, re-checked against its result and this run; None if it did not finish."""
    out = sample_dir(run, condition)
    if not (out / "result.json").exists():
        return None
    result, manifest = read_json(out / "result.json"), read_json(out / "manifest.json")
    if manifest["run"] != frozen["identity"] or result["identity"] != manifest["identity"]:
        raise ValueError(f"The samples of {condition} belong to another run")
    if digest(out / "rows.jsonl") != result["rows_sha256"]:
        raise ValueError(f"The samples of {condition} changed")
    return recover_records(out / "rows.jsonl")
