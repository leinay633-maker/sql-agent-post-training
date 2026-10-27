"""v0.7 GPU work: claim-based pool sampling (teacher / 14B), greedy and execution-vote evaluation, all vLLM."""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from ..v6.rollouts import decide, deadline_from, result_key, versions
from .freeze import load, model_ready

CONDITION = re.compile(r"(?P<base>B|B14|B32)|(?P<family>R|P|T|TR|Traw|PT|PTR|R14)_seed(?P<seed>\d+)")
IMPORTED_FAMILIES = {"R", "P"}
KIND_OF_BASE = {"B": "student", "B14": "scale", "B32": "teacher"}
SAMPLE_KINDS = ("teacher", "scale")


def parse(condition: str) -> tuple[str, str | None, bool]:
    """'T_seed42+vote' -> (model kind, family_seed or None, vote)."""
    name, vote = (condition[:-5], True) if condition.endswith("+vote") else (condition, False)
    match = CONDITION.fullmatch(name)
    if match is None:
        raise ValueError(f"Unknown condition: {condition}")
    if match.group("base"):
        return KIND_OF_BASE[name], None, vote
    return ("scale" if match.group("family") == "R14" else "student"), name, vote


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
    from .freeze import _adapter_hash
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

    llm, tokenizer = make_engine(spec, settings, mem=mem, seed=seed, lora=adapter is not None,
                                 max_num_seqs=settings["max_num_seqs"][kind])
    return generator(llm, tokenizer, spec, settings, sampling=sampling,
                     lora=lora_request(name, adapter) if adapter is not None else None)


def sample(run: Path, *, kind: str, stop: str | None, mem: float | None = None, seed: int = 20261012,
           factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    from ..rft.runner import run_dialogues
    from ..rft.sampling import Oracle

    if kind not in SAMPLE_KINDS:
        raise ValueError("Pool sampling is for the teacher and the 14B")
    frozen, pool = load(run, "pool")
    s = frozen["settings"]
    spec = model_ready(frozen, kind) if factory is vllm_factory else frozen["models"][kind]["spec"]
    samples, shards = s[f"samples_{kind}"], s["sample_shards"]
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    order = sorted(pool, key=lambda t: hashlib.sha256(f"{s['salt']}:sample:{kind}:{t.id}".encode()).digest())
    base = run / "samples" / kind
    out, claims = base / "samples", base / "claims"
    out.mkdir(parents=True, exist_ok=True)
    claims.mkdir(parents=True, exist_ok=True)
    sampling = {k: s[k] for k in ("temperature", "top_p", "top_k")}
    identity = fingerprint({"run": frozen["identity"], "kind": kind, "shards": shards, "samples": samples,
                            "model": frozen["models"][kind]["identity"], "sampling": sampling})
    oracle = Oracle(pool, root, limits)
    deadline = deadline_from(stop)
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
            generator = factory(spec, s, kind, sampling=sampling, adapter=None,
                                mem=mem if mem is not None else s["gpu_memory"][kind], seed=seed, name=kind)
            load_seconds = time.monotonic() - started
        counts = Counter({"completed": len(done)})
        prior = read_json(status_path).get("active_seconds", 0.0) if status_path.exists() else 0.0
        t0, fresh = time.monotonic(), 0

        def finish(key: str, task: Task, record: dict[str, Any], shard: int = shard) -> dict[str, Any]:
            scored = oracle.score(task, record)
            return {**record, "key": key, "task_id": task.id, "db_id": task.db_id, "k": int(key.rsplit("#", 1)[1]),
                    "kind": kind, "shard": shard, "correct": bool(scored["verdict"]["correct"]),
                    "verdict": scored["verdict"]["status"], "finish_checks": scored["finish_checks"], "engine": "vllm"}

        def save(final: bool, shard: int = shard, registered: list = registered, counts: Counter = counts,
                 t0: float = t0, prior: float = prior, status_path: Path = status_path) -> None:
            atomic_json(status_path, {
                "identity": identity, "kind": kind, "final": final, "shard": shard, "shards": shards,
                "samples_per_task": samples, "registered_items": len(registered), **counts,
                "complete": counts["completed"] >= len(registered), "active_seconds": prior + time.monotonic() - t0,
                "engine_load_seconds": load_seconds, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "sampling": sampling, "generator": getattr(generator, "metadata", {}), "packages": versions()})

        try:
            if items:
                for row in run_dialogues(items, root, generator, slots=s["slots"][kind], max_steps=s["max_steps"],
                                         review_success=True, limits=limits, workers=s["sql_workers"], finish=finish,
                                         deadline=deadline, fenced_json=True):
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
    return {"kind": kind, "shards_worked": worked}


def sampling_state(run: Path, kind: str, shards: int) -> dict[str, Any]:
    folder = run / "samples" / kind / "samples"
    statuses = [read_json(p) for p in sorted(folder.glob("shard*.status.json"))] if folder.exists() else []
    claimed = len(list((run / "samples" / kind / "claims").iterdir())) if (run / "samples" / kind / "claims").exists() else 0
    complete = sum(bool(x.get("complete")) for x in statuses)
    return {"shards": shards, "claimed": claimed, "complete": complete,
            "trajectories": sum(x.get("completed", 0) for x in statuses),
            "correct": sum(x.get("correct", 0) for x in statuses),
            "active_seconds": sum(x.get("active_seconds", 0.0) for x in statuses)}


def evaluate(run: Path, *, condition: str, role: str, stop: str | None, mem: float | None = None, shard: int = 0,
             factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    """condition: B | B14 | B32 | <family>_seed<n> ; suffix '+vote' = vote_n sampled dialogues, execution vote."""
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load(run, role)
    s = frozen["settings"]
    kind, name, vote = parse(condition)
    spec = model_ready(frozen, kind) if factory is vllm_factory else frozen["models"][kind]["spec"]
    adapter, adapter_hash = trained_adapter(run, frozen, name) if name else (None, None)
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    gold = gold_for(run, frozen, role)
    registered = len(tasks)
    # Per-condition order so concurrent evaluations do not hit one large database together.
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{condition}:{role}:{t.id}".encode()).digest())
    shards = s["vote_shards"] if vote else s["eval_shards"][kind]
    if not 0 <= shard < shards:
        raise ValueError("Invalid evaluation shard")
    tasks = [t for i, t in enumerate(tasks) if i % shards == shard]
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "model": frozen["models"][kind]["identity"], "adapter_hash": adapter_hash,
                "vote": s["vote_n"] if vote else 0, "shard": shard, "shards": shards, "code": frozen["code"]}
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    journal, samples_log = out / f"shard{shard:02d}.jsonl", out / f"shard{shard:02d}.samples.jsonl"
    if (out / f"shard{shard:02d}.summary.json").exists():
        return read_json(out / f"shard{shard:02d}.summary.json")
    for stale in (journal, samples_log):
        if stale.exists():  # an interrupted attempt: keep it for forensics, run the shard again
            stale.rename(stale.with_name(stale.name + f".interrupted-{time.time_ns()}"))
    atomic_json(out / f"shard{shard:02d}.meta.json", {**identity, "signature": fingerprint(identity)})
    sampling = {"temperature": s["vote_temperature"], "top_p": s["vote_top_p"], "top_k": 0} if vote else None
    started_load = time.monotonic()
    generator = factory(spec, s, kind, sampling=sampling, adapter=adapter,
                        mem=mem if mem is not None else s["gpu_memory"][kind], seed=20261012 + shard,
                        name=(name or condition).split("+")[0])
    load_seconds = time.monotonic() - started_load

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        answer_key = result_key(record.get("result"))
        row = grade_episode(task, task.database(root), "model", record, limits, s["max_steps"], True,
                            gold=gold.get(task.id))
        row["condition"] = condition
        if vote:
            row.update(_key=answer_key, _k=int(key.rsplit("#", 1)[1]))
        return row

    items = [(f"{t.id}#{k}", t) for t in tasks for k in range(s["vote_n"])] if vote else [(t.id, t) for t in tasks]
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    statuses, started, interrupted = Counter(), time.monotonic(), 0
    try:
        for row in run_dialogues(items, root, generator, slots=s["slots"][kind], max_steps=s["max_steps"],
                                 review_success=True, limits=limits, workers=s["sql_workers"], finish=finish,
                                 deadline=deadline_from(stop), fenced_json=True):
            if not vote:
                append_record(journal, row)
                statuses[row["status"]] += 1
                continue
            append_record(samples_log, {k: row.get(k) for k in ("task_id", "_k", "_key", "status", "correct", "final_sql")})
            pending[row["task_id"]].append(row)
            if len(pending[row["task_id"]]) == s["vote_n"]:
                chosen = decide(pending.pop(row["task_id"]))
                append_record(journal, chosen)
                statuses[chosen["status"]] += 1
    finally:
        for task_id in list(pending):
            chosen = decide(pending.pop(task_id))
            chosen["vote"]["incomplete"] = True
            append_record(journal, chosen)
            statuses[chosen["status"]] += 1
            interrupted += 1
    done = sum(statuses.values())
    summary = {**identity, "registered_tasks": registered, "shard_tasks": len(tasks), "episodes": done,
               "complete": done == len(tasks) and not interrupted, "statuses": dict(statuses),
               "correct": statuses.get("correct", 0), "incomplete_votes": interrupted,
               "wall_seconds": time.monotonic() - started, "engine_load_seconds": load_seconds,
               "generator": getattr(generator, "metadata", {}), "gold_cache_entries": len(gold),
               "packages": versions(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "episodes_sha256": digest(journal) if journal.exists() else None}
    atomic_json(out / f"shard{shard:02d}.{'summary' if summary['complete'] else 'partial'}.json", summary)
    return summary


def eval_shards(settings: dict[str, Any], condition: str) -> int:
    kind, _, vote = parse(condition)
    return settings["vote_shards"] if vote else settings["eval_shards"][kind]


def merged(run: Path, role: str, condition: str) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """All shards complete -> rows; otherwise (None, info)."""
    out = run / "eval" / role / condition
    settings = read_json(run / "manifest.json")["settings"]
    rows, summaries = [], []
    for shard in range(eval_shards(settings, condition)):
        p = out / f"shard{shard:02d}.summary.json"
        if not p.exists():
            return None, {"missing_shard": shard}
        summary = read_json(p)
        if digest(out / f"shard{shard:02d}.jsonl") != summary["episodes_sha256"]:
            raise ValueError(f"Evaluation shard changed: {role}/{condition}/{shard}")
        summaries.append(summary)
        rows += recover_records(out / f"shard{shard:02d}.jsonl")
    return rows, {"wall_seconds": max(x["wall_seconds"] for x in summaries),
                  "engine_load_seconds": max(x["engine_load_seconds"] for x in summaries),
                  "generator": {k: sum(x["generator"].get(k, 0) for x in summaries)
                                for k in ("calls", "generate_seconds", "prompt_tokens", "cached_prompt_tokens",
                                          "completion_tokens", "length_stops", "context_rejections")}}


def eval_done(run: Path, role: str, condition: str, settings: dict[str, Any]) -> bool:
    out = run / "eval" / role / condition
    return all((out / f"shard{i:02d}.summary.json").exists() for i in range(eval_shards(settings, condition)))
