"""v0.8 GPU work (vLLM): claim-based sampling on the 4B's unsolved pool tasks, by the teacher or by the 4B with the
gold SQL as a private hint, and greedy evaluation."""
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
from . import HINT_NOTE
from .freeze import _adapter_hash, load, model_ready

CONDITION = re.compile(r"(?P<base>B|B235)|(?P<family>R|P|TR|HR|PTR|PHR)_seed(?P<seed>\d+)")
IMPORTED_FAMILIES = {"R", "P"}
KIND_OF_BASE = {"B": "student", "B235": "teacher"}
SAMPLERS = {"teacher": "teacher", "hint": "student"}  # sampler -> model kind


def parse(condition: str) -> tuple[str, str | None]:
    """'TR_seed42' -> ('student', 'TR_seed42'); 'B235' -> ('teacher', None)."""
    match = CONDITION.fullmatch(condition)
    if match is None:
        raise ValueError(f"Unknown condition: {condition}")
    if match.group("base"):
        return KIND_OF_BASE[condition], None
    return "student", condition


def trained_adapter(run: Path, frozen: dict[str, Any], name: str) -> tuple[Path, str]:
    """Adapter directory and hash of a frozen model of this run (trained here or imported from v0.5)."""
    family = name.split("_seed")[0]
    if family in IMPORTED_FAMILIES:
        folder = run / "imports" / "adapters" / name
        expected = frozen["imports"]["adapters"].get(name)
        if expected is None:
            raise ValueError(f"{name} was not imported")
    else:
        folder = run / "train" / name
        result = read_json(folder / "result.json")
        if not (result.get("training_completed") and result.get("adapter_reload_verified")):
            raise ValueError(f"{name} is not a frozen, reload-verified model")
        if read_json(folder / "manifest.json")["run"] != frozen["identity"]:
            raise ValueError(f"{name} belongs to another run")
        expected = result["adapter_hash"]
    if _adapter_hash(folder / "adapter") != expected:
        raise ValueError(f"Frozen adapter changed: {name}")
    return folder / "adapter", expected


def gold_for(run: Path, frozen: dict[str, Any], role: str) -> dict[str, Any]:
    path = run / "gold" / f"{role}.json"
    if role not in frozen["gold"] or digest(path) != frozen["gold"][role]["sha256"]:
        raise ValueError(f"Gold cache for {role} missing or changed")
    return read_json(path)["results"]


def vllm_factory(spec: dict[str, Any], settings: dict[str, Any], kind: str, *, sampling: dict[str, Any] | None,
                 adapter: Path | None, mem: float, seed: int, name: str) -> Any:
    from .engine import generator, make_engine
    from ..v6.engine import lora_request

    llm, tokenizer = make_engine(spec, settings, kind, mem=mem, seed=seed, lora=adapter is not None,
                                 max_num_seqs=settings["max_num_seqs"][kind])
    return generator(llm, tokenizer, spec, settings, sampling=sampling,
                     lora=lora_request(name, adapter) if adapter is not None else None)


def hint_for(task: Task) -> dict[str, str]:
    return {"reference_sql": task.gold_sql, "reference_note": HINT_NOTE}


def sample(run: Path, *, sampler: str, stop: str | None, mem: float | None = None, seed: int = 20261013,
           factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    from ..rft.runner import run_dialogues
    from ..rft.sampling import Oracle

    if sampler not in SAMPLERS:
        raise ValueError("Samplers are teacher and hint")
    kind = SAMPLERS[sampler]
    frozen, pool = load(run, "pool")
    s = frozen["settings"]
    spec = model_ready(frozen, kind) if factory is vllm_factory else frozen["models"][kind]["spec"]
    unsolved = set(frozen["unsolved_tasks"])
    tasks = [t for t in pool if t.id in unsolved]
    samples, shards = s[f"samples_{sampler}"], s["sample_shards"]
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    order = sorted(tasks, key=lambda t: hashlib.sha256(f"{s['salt']}:sample:{sampler}:{t.id}".encode()).digest())
    base = run / "samples" / sampler
    out, claims = base / "samples", base / "claims"
    out.mkdir(parents=True, exist_ok=True)
    claims.mkdir(parents=True, exist_ok=True)
    sampling = {k: s[k] for k in ("temperature", "top_p", "top_k")}
    identity = fingerprint({"run": frozen["identity"], "sampler": sampler, "shards": shards, "samples": samples,
                            "model": frozen["models"][kind]["identity"], "sampling": sampling,
                            "hint": HINT_NOTE if sampler == "hint" else None})
    oracle = Oracle(pool, root, limits)
    deadline = deadline_from(stop)
    extra_for = hint_for if sampler == "hint" else None
    memory = mem if mem is not None else s["gpu_memory"]["teacher"] if kind == "teacher" else s["sample_gpu_memory"]
    generator, load_seconds, worked = None, 0.0, []
    for shard in range(shards):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            (claims / f"{shard:02d}").mkdir()
        except FileExistsError:
            continue
        mine = [t for i, t in enumerate(order) if i % shards == shard]
        registered = [(f"{t.id}#{k}", t) for t in mine for k in range(samples)]  # same task back to back: prefix cache
        journal, status_path = out / f"shard{shard:02d}.jsonl", out / f"shard{shard:02d}.status.json"
        if status_path.exists() and read_json(status_path)["identity"] != identity:
            raise ValueError("Sampling policy/run changed; use a new run")
        done = {r["key"] for r in recover_records(journal)}
        items = [item for item in registered if item[0] not in done]
        if generator is None and items:
            started = time.monotonic()
            generator = factory(spec, s, kind, sampling=sampling, adapter=None, mem=memory, seed=seed, name=sampler)
            load_seconds = time.monotonic() - started
        counts = Counter({"completed": len(done)})
        prior = read_json(status_path).get("active_seconds", 0.0) if status_path.exists() else 0.0
        t0, fresh = time.monotonic(), 0

        def finish(key: str, task: Task, record: dict[str, Any], shard: int = shard) -> dict[str, Any]:
            scored = oracle.score(task, record)
            return {**record, "key": key, "task_id": task.id, "db_id": task.db_id, "k": int(key.rsplit("#", 1)[1]),
                    "sampler": sampler, "hinted": sampler == "hint", "shard": shard,
                    "correct": bool(scored["verdict"]["correct"]), "verdict": scored["verdict"]["status"],
                    "finish_checks": scored["finish_checks"], "engine": "vllm"}

        def save(final: bool, shard: int = shard, registered: list = registered, counts: Counter = counts,
                 t0: float = t0, prior: float = prior, status_path: Path = status_path) -> None:
            atomic_json(status_path, {
                "identity": identity, "sampler": sampler, "final": final, "shard": shard, "shards": shards,
                "samples_per_task": samples, "registered_items": len(registered), **counts,
                "complete": counts["completed"] >= len(registered), "active_seconds": prior + time.monotonic() - t0,
                "engine_load_seconds": load_seconds, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "sampling": sampling, "generator": getattr(generator, "metadata", {}), "packages": versions()})

        try:
            if items:
                for row in run_dialogues(items, root, generator, slots=s["slots"][kind], max_steps=s["max_steps"],
                                         review_success=True, limits=limits, workers=s["sql_workers"], finish=finish,
                                         deadline=deadline, extra_for=extra_for, fenced_json=True):
                    append_record(journal, row)
                    counts["completed"] += 1
                    counts["correct"] += row["correct"]
                    counts[f"status:{row['status']}"] += 1
                    fresh += 1
                    if fresh >= 64 and counts["status:schema_error"] == fresh:
                        raise RuntimeError("Every dialogue failed at the schema query; SQL sandbox is broken")
                    if counts["completed"] % 400 == 0:
                        save(False)
        finally:
            save(True)
            if counts["completed"] < len(registered):
                # Deadline or crash mid-shard: release the claim so a later sampler (or a resume) can finish it.
                (claims / f"{shard:02d}").rmdir()
        worked.append(shard)
    return {"sampler": sampler, "shards_worked": worked}


def sampling_state(run: Path, sampler: str, shards: int) -> dict[str, Any]:
    folder = run / "samples" / sampler / "samples"
    statuses = [read_json(p) for p in sorted(folder.glob("shard*.status.json"))] if folder.exists() else []
    claims = run / "samples" / sampler / "claims"
    claimed = len(list(claims.iterdir())) if claims.exists() else 0
    return {"shards": shards, "claimed": claimed, "complete": sum(bool(x.get("complete")) for x in statuses),
            "trajectories": sum(x.get("completed", 0) for x in statuses),
            "correct": sum(x.get("correct", 0) for x in statuses),
            "active_seconds": sum(x.get("active_seconds", 0.0) for x in statuses)}


def evaluate(run: Path, *, condition: str, role: str, stop: str | None, mem: float | None = None,
             factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    """Greedy evaluation of B | B235 | <family>_seed<n> on dev, validation or test (one shard per condition)."""
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load(run, role)
    s = frozen["settings"]
    kind, name = parse(condition)
    spec = model_ready(frozen, kind) if factory is vllm_factory else frozen["models"][kind]["spec"]
    adapter, adapter_hash = trained_adapter(run, frozen, name) if name else (None, None)
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    gold = gold_for(run, frozen, role)
    # Per-condition order so concurrent evaluations do not hit one large database together.
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{condition}:{role}:{t.id}".encode()).digest())
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "model": frozen["models"][kind]["identity"], "adapter_hash": adapter_hash, "code": frozen["code"]}
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "episodes.jsonl"
    if (out / "summary.json").exists():
        return read_json(out / "summary.json")
    if journal.exists():  # an interrupted attempt: keep it for forensics, run again
        journal.rename(journal.with_name(journal.name + f".interrupted-{time.time_ns()}"))
    atomic_json(out / "meta.json", {**identity, "signature": fingerprint(identity)})
    started_load = time.monotonic()
    generator = factory(spec, s, kind, sampling=None, adapter=adapter,
                        mem=mem if mem is not None else s["gpu_memory"][kind], seed=20261013, name=name or condition)
    load_seconds = time.monotonic() - started_load

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        row = grade_episode(task, task.database(root), "model", record, limits, s["max_steps"], True,
                            gold=gold.get(task.id))
        row["condition"] = condition
        return row

    statuses, started = Counter(), time.monotonic()
    for row in run_dialogues([(t.id, t) for t in tasks], root, generator, slots=s["slots"][kind],
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
        "generator": {k: summary["generator"].get(k, 0) for k in (
            "calls", "generate_seconds", "prompt_tokens", "cached_prompt_tokens", "completion_tokens",
            "length_stops", "context_rejections")}}


def eval_done(run: Path, role: str, condition: str) -> bool:
    return (run / "eval" / role / condition / "summary.json").exists()
