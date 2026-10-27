"""v0.6 GPU work: claim-based count sampling, greedy evaluation and execution-vote evaluation (all vLLM)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from .freeze import gold_for, load, notes_lookup


def versions() -> dict[str, str]:
    out = {}
    for p in ("vllm", "torch", "transformers"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = "missing"
    return out


def deadline_from(stop: str | None) -> float | None:
    if not stop:
        return None
    import datetime as dt
    left = (dt.datetime.fromisoformat(stop) - dt.datetime.now(dt.timezone.utc)).total_seconds()
    return time.monotonic() + max(0.0, left)


def adapter_for(run: Path, frozen: dict[str, Any], condition: str) -> Path | None:
    from ..rft.evaluation import frozen_adapter
    return frozen_adapter(run, frozen, condition.split("+")[0])


def sample(run: Path, *, round_: int, policy: str, stop: str | None, mem: float, seed: int,
           max_num_seqs: int = 256) -> dict[str, Any]:
    from ..rft.runner import run_dialogues
    from ..rft.sampling import Oracle
    from .engine import VLLMGenerator, lora_request, make_engine

    frozen, pool = load(run, "pool")
    s = frozen["settings"]
    samples = s["samples_round1"] if round_ == 1 else s["samples_round2"]
    if (round_ == 1) != (policy == "B"):
        raise ValueError("Round 1 samples the base model; later rounds sample a frozen adapter")
    adapter = adapter_for(run, frozen, policy)
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    shards = s["sample_shards"]
    order = sorted(pool, key=lambda t: hashlib.sha256(f"{s['salt']}:sample:{round_}:{t.id}".encode()).digest())
    out, claims = run / f"round{round_}" / "samples", run / f"round{round_}" / "claims"
    out.mkdir(parents=True, exist_ok=True)
    claims.mkdir(parents=True, exist_ok=True)
    from ..models import model_identity
    policy_id = {"base": frozen["model_identity"],
                 "adapter_hash": model_identity(frozen["model"], adapter)["adapter_hash"] if adapter else None}
    identity = fingerprint({"run": frozen["identity"], "round": round_, "shards": shards, "samples": samples,
                            "policy": policy_id})
    oracle, extra_for = Oracle(pool, root, limits), notes_lookup(run)
    started_load = time.monotonic()
    llm, tokenizer = make_engine(frozen, mem=mem, seed=seed, lora=adapter is not None, max_num_seqs=max_num_seqs)
    lora = lora_request(policy, adapter) if adapter else None
    load_seconds = time.monotonic() - started_load
    sampling = {k: s[k] for k in ("temperature", "top_p", "top_k")}
    deadline = deadline_from(stop)
    done_shards = []
    for shard in range(shards):
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            (claims / f"{shard:02d}").mkdir()
        except FileExistsError:
            continue
        mine = [t for i, t in enumerate(order) if i % shards == shard]
        registered = [(f"{t.id}#{k}", t) for t in mine for k in range(samples)]
        journal, status_path = out / f"shard{shard:02d}.jsonl", out / f"shard{shard:02d}.status.json"
        if status_path.exists() and read_json(status_path)["identity"] != identity:
            raise ValueError("Sampling policy/run changed; use a new round directory")
        done = {r["key"] for r in recover_records(journal)}
        items = [item for item in registered if item[0] not in done]
        generator = VLLMGenerator(llm, tokenizer, max_context=s["max_context"], max_tokens=s["max_tokens"],
                                  sampling=sampling, lora=lora)
        counts = Counter({"completed": len(done)})
        prior = read_json(status_path).get("active_seconds", 0.0) if status_path.exists() else 0.0
        t0 = time.monotonic()

        def finish(key: str, task: Task, record: dict[str, Any], shard: int = shard) -> dict[str, Any]:
            scored = oracle.score(task, record)
            return {**record, "key": key, "task_id": task.id, "db_id": task.db_id, "k": int(key.rsplit("#", 1)[1]),
                    "round": round_, "shard": shard, "correct": bool(scored["verdict"]["correct"]),
                    "verdict": scored["verdict"]["status"], "finish_checks": scored["finish_checks"], "engine": "vllm"}

        def save(final: bool, shard: int = shard, registered: list = registered, generator: Any = generator,
                 counts: Counter = counts, t0: float = t0, prior: float = prior, status_path: Path = status_path) -> None:
            atomic_json(status_path, {
                "identity": identity, "policy": policy, "policy_identity": policy_id, "final": final, "shard": shard,
                "shards": shards, "samples_per_task": samples, "registered_items": len(registered), **counts,
                "complete": counts["completed"] >= len(registered), "active_seconds": prior + time.monotonic() - t0,
                "engine_load_seconds": load_seconds, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "sampling": sampling, "generator": generator.metadata, "packages": versions()})

        fresh = 0
        try:
            for row in run_dialogues(items, root, generator, slots=s["sample_slots"], max_steps=s["max_steps"],
                                     review_success=True, limits=limits, workers=s["sql_workers"], finish=finish,
                                     deadline=deadline, extra_for=extra_for, fenced_json=True):
                append_record(journal, row)
                counts["completed"] += 1
                counts["correct"] += row["correct"]
                counts[f"status:{row['status']}"] += 1
                fresh += 1
                if fresh >= 128 and counts["status:schema_error"] == fresh:
                    raise RuntimeError("Every dialogue failed at the schema query; SQL sandbox is broken")
                if counts["completed"] % 400 == 0:
                    save(False)
        finally:
            save(True)
            if counts["completed"] < len(registered):
                # Deadline or crash mid-shard: release the claim so another sampler (or a resume) finishes it.
                (claims / f"{shard:02d}").rmdir()
        done_shards.append(shard)
    return {"round": round_, "policy": policy, "shards_worked": done_shards}


VOTE_KEY_VERSION = 1


def result_key(answer: dict[str, Any] | None) -> str | None:
    """Gold-free identity of an executed answer: column count + multiset of rows."""
    if not answer or answer.get("status") != "ok":
        return None
    rows = sorted(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in answer["rows"])
    return hashlib.sha256(json.dumps([len(answer["columns"]), rows]).encode()).hexdigest()


def decide(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Largest group of identical execution results; ties -> group containing the earliest sample."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in samples:
        if x["_key"] is not None:
            groups[x["_key"]].append(x)
    if groups:
        best = max(groups.values(), key=lambda g: (len(g), -min(x["_k"] for x in g)))
        chosen, votes = min(best, key=lambda x: x["_k"]), len(best)
    else:
        chosen, votes = min(samples, key=lambda x: x["_k"]), 0
    row = {k: v for k, v in chosen.items() if not k.startswith("_")}
    row["vote"] = {"samples": len(samples), "groups": len(groups), "votes": votes, "chosen_k": chosen["_k"],
                   "any_correct": any(x["correct"] for x in samples),
                   "sample_correct": sum(bool(x["correct"]) for x in samples)}
    return row


def evaluate(run: Path, *, condition: str, role: str, stop: str | None, mem: float, shard: int = 0,
             max_num_seqs: int = 256) -> dict[str, Any]:
    """condition: B | R_r1_seed42 | ... ; suffix '+vote' = vote_n sampled trajectories, execution vote."""
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues
    from .engine import VLLMGenerator, lora_request, make_engine

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load(run, role)
    s = frozen["settings"]
    vote = condition.endswith("+vote")
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    adapter = adapter_for(run, frozen, condition)
    gold = gold_for(run, frozen, role) if role in frozen["gold"] else {}
    registered = len(tasks)
    # Per-condition order so concurrent evaluations do not hit one large database together.
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{condition}:{role}:{t.id}".encode()).digest())
    shards = s["vote_shards"] if vote else 1
    tasks = [t for i, t in enumerate(tasks) if i % shards == shard]
    from ..models import model_identity
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "adapter_hash": model_identity(frozen["model"], adapter)["adapter_hash"] if adapter else None,
                "vote": s["vote_n"] if vote else 0, "shard": shard, "shards": shards, "code": frozen["code"]}
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    journal = out / f"shard{shard:02d}.jsonl"
    samples_log = out / f"shard{shard:02d}.samples.jsonl"
    meta_path = out / f"shard{shard:02d}.meta.json"
    if (out / f"shard{shard:02d}.summary.json").exists():
        return read_json(out / f"shard{shard:02d}.summary.json")
    for stale in (journal, samples_log):
        if stale.exists():  # an interrupted attempt: keep it for forensics, start the shard again
            stale.rename(stale.with_name(stale.name + f".interrupted-{time.time_ns()}"))
    atomic_json(meta_path, {**identity, "signature": fingerprint(identity)})
    started_load = time.monotonic()
    llm, tokenizer = make_engine(frozen, mem=mem, seed=20261012 + shard, lora=adapter is not None,
                                 max_num_seqs=max_num_seqs)
    load_seconds = time.monotonic() - started_load
    sampling = {"temperature": s["vote_temperature"], "top_p": s["vote_top_p"], "top_k": 0} if vote else None
    generator = VLLMGenerator(llm, tokenizer, max_context=s["max_context"], max_tokens=s["max_tokens"],
                              sampling=sampling, lora=lora_request(condition.split("+")[0], adapter) if adapter else None)

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
    statuses, started = Counter(), time.monotonic()
    interrupted = 0
    try:
        for row in run_dialogues(items, root, generator, slots=s["eval_slots"], max_steps=s["max_steps"],
                                 review_success=True, limits=limits, workers=s["sql_workers"], finish=finish,
                                 deadline=deadline_from(stop), extra_for=notes_lookup(run), fenced_json=True):
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
               "generator": generator.metadata, "gold_cache_entries": len(gold), "packages": versions(),
               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "episodes_sha256": digest(journal) if journal.exists() else None}
    if summary["complete"]:
        atomic_json(out / f"shard{shard:02d}.summary.json", summary)
    else:
        atomic_json(out / f"shard{shard:02d}.partial.json", summary)
    return summary


def merged(run: Path, role: str, condition: str) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """All shards complete -> rows; otherwise (None, info)."""
    out = run / "eval" / role / condition
    frozen = read_json(run / "manifest.json")
    shards = frozen["settings"]["vote_shards"] if condition.endswith("+vote") else 1
    rows, summaries = [], []
    for shard in range(shards):
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
