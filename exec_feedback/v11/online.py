"""v0.11 online runs: v0.9's RL loop (one GPU, vLLM samples with the current LoRA, the train-only oracle scores the final
SQL, PEFT takes one on-policy RLOO step), drawing its tasks from the seed's boundary tasks only.

A run trains to a stop step: first to the registered milestone (planned_steps), where it pauses until the evaluations
at K are done; then, exploratory, from that checkpoint on to explore_steps. Every record of v0.9's loop is kept, plus
the pass over the boundary tasks and the three degenerate behaviours v0.10's diagnosis tracked.
"""
from __future__ import annotations

import gc
import math
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits
from ..storage import append_record, atomic_json, exclusive_lock, fingerprint, read_json
from ..v6.rollouts import deadline_from, versions
from ..v9.online import (_set_aside, _version, advantages, backward_step, curve_eval, gold_for, group_counts,
                         lora_check, milestones, peft_policy, restore_checkpoint, rollouts, save_adapter,
                         save_checkpoint, step_tasks, task_order, vllm_engine)
from ..v9.online import LORA_CHECK_TRAJECTORIES
from . import FAMILY
from .freeze import load, model_ready, start_adapter
from .prepass import boundary_tasks


def run_name(seed: int) -> str:
    return f"{FAMILY}_seed{seed}"


def plan_of(settings: dict[str, Any], boundary_sha: str, pool_size: int) -> dict[str, Any]:
    keys = ("tasks_per_step", "group_size", "planned_steps", "explore_steps", "temperature", "learning_rate",
            "warmup_steps", "max_grad_norm", "loss_tokens", "tis_cap", "micro_tokens", "save_steps", "curve_every")
    return {"method": "rloo", "pool": "boundary", "boundary_sha256": boundary_sha, "boundary_tasks": pool_size,
            **{k: settings[k] for k in keys}}


def kind(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    return action.get("action") if isinstance(action, dict) else None  # None: the output did not parse


def shape(steps: list[dict[str, Any]]) -> dict[str, bool]:
    """v0.10's three behaviours (as scripts/v10_diagnose.py): unparsed, the first output is no valid action; no_query,
    submitted before any query (or never queried); repeat, the same SQL queried twice in a row. A dialogue rejected
    for its context length has no steps: all False."""
    kinds = [kind(s) for s in steps]
    before = kinds[:kinds.index("finish")] if "finish" in kinds else kinds
    sqls = [s["action"].get("sql") for s in steps if kind(s) == "query"]
    return {"unparsed": bool(kinds) and kinds[0] is None, "no_query": bool(kinds) and "query" not in before,
            "repeat": any(a == b for a, b in zip(sqls, sqls[1:]))}


def behaviour(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if row.get("steps"):
            counts["dialogues"] += 1
            counts.update(k for k, v in shape(row["steps"]).items() if v)
    return {k: counts.get(k, 0) for k in ("dialogues", "unparsed", "no_query", "repeat")}


def paused(out: Path) -> dict[str, Any] | None:
    path = out / "paused.json"
    return read_json(path) if path.exists() else None


def ended(out: Path) -> dict[str, Any] | None:
    """The record of a run that can train no further: it reached explore_steps, or a deadline stopped it."""
    path = out / "result.json"
    record = read_json(path) if path.exists() else None
    return record if record and record.get("finished") else None


def trained_to(out: Path) -> int:
    """The step a run last stopped at on purpose: its end, else the registered milestone it paused at; 0 before."""
    record = ended(out) or paused(out)
    return int(record["steps_completed"]) if record else 0


def train_online(run: Path, *, seed: int, until: str | None, stop_step: int | None = None,
                 engine_factory: Callable[..., Any] = vllm_engine,
                 policy_loader: Callable[..., tuple[Any, Any]] = peft_policy) -> dict[str, Any]:
    """Train to stop_step (planned_steps or explore_steps; default the latter). Below explore_steps the run pauses:
    paused.json, no result. A deadline ends the run for good (result with stopped_by=deadline), as in v0.9."""
    import torch
    from ..models import model_identity
    from ..rft.sampling import Oracle
    from ..v7.engine import stop_ids as model_stop_ids

    frozen, pool = load(run, "pool")
    _, dev = load(run, "dev")
    s = frozen["settings"]
    if seed not in s["seeds"]:
        raise ValueError("Unregistered seed")
    target = s["explore_steps"] if stop_step is None else stop_step
    if target not in (s["planned_steps"], s["explore_steps"]):
        raise ValueError("A run stops at the registered milestone or at the end of the exploratory continuation")
    spec = model_ready(frozen) if engine_factory is vllm_engine else frozen["models"]["student"]["spec"]
    start, start_hash = start_adapter(run, frozen, seed)
    tasks_pool, boundary_sha = boundary_tasks(run, frozen, pool, seed)
    plan = plan_of(s, boundary_sha, len(tasks_pool))
    identity = fingerprint({"run": frozen["identity"], "family": FAMILY, "seed": seed, "start": start_hash,
                            "plan": plan, "model": frozen["models"]["student"]["identity"]})
    out = run / "train" / run_name(seed)
    out.mkdir(parents=True, exist_ok=True)
    root, limits = Path(frozen["raw_root"]), Limits(**s["limits"])
    with exclusive_lock(out / ".train.lock"):
        if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
            raise ValueError("Online run inputs changed; use a new run")
        if (record := ended(out)) is not None:
            return record
        held = paused(out)
        if held is not None and held["step"] >= target:
            return {**held, "finished": False}
        deadline = deadline_from(until)
        attempt = uuid.uuid4().hex[:12]
        started = time.monotonic()
        # vLLM first: its start-up memory profiling must not see the trainer's allocations appear underneath it.
        engine = engine_factory(spec, s, seed=20261014 + 1009 * seed)
        engine_seconds = time.monotonic() - started
        torch.manual_seed(seed)
        model, tokenizer = policy_loader(spec, start)
        for module in model.modules():  # sampling has no dropout, so teacher forcing must not have it either
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        base = model_identity(spec)
        stops = model_stop_ids(spec, tokenizer)
        pad_id = tokenizer.pad_token_id
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable LoRA parameters")
        optimizer = torch.optim.AdamW(params, lr=plan["learning_rate"], weight_decay=0.0)
        state = restore_checkpoint(out, model, optimizer, identity)
        fresh = state is None
        if fresh:
            state = {"identity": identity, "step": 0, "optimizer_updates": 0, "active_seconds": 0.0,
                     "curve_steps": [], "lora_check": None}
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "family": FAMILY, "seed": seed,
            "start_adapter": str(start), "start_hash": start_hash, "plan": plan,
            "packages": {**versions(), "peft": _version("peft")}})
        append_record(out / "attempts.jsonl", {"attempt": attempt, "resumed_step": state["step"], "target": target,
                                               "engine_load_seconds": engine_seconds,
                                               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        _set_aside(out, state["step"])
        prior, clock = state["active_seconds"], time.monotonic()
        order = task_order(tasks_pool, s["salt"], seed)
        oracle = Oracle(pool, root, limits)
        gold_dev = gold_for(run, frozen, "dev")
        policy_root = out / "policy"
        current = policy_root / f"step-{state['step']:04d}"
        save_adapter(model, current)
        engine.set_policy(current, state["step"])
        recent: list[float] = []
        stopped_by = "planned"
        normaliser = float(s["tasks_per_step"] * s["group_size"] * s["loss_tokens"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def left() -> float:
            return math.inf if deadline is None else deadline - time.monotonic()

        def curve(step: int) -> None:
            if step % s["curve_every"] or step in state["curve_steps"]:
                return
            estimate = (read_json(out / "curve-estimate.json")["seconds"] * 1.3
                        if (out / "curve-estimate.json").exists() else 600.0)
            if left() < estimate + 120:
                append_record(out / "curve.jsonl", {"step": step, "skipped": "deadline", "attempt": attempt})
                state["curve_steps"].append(step)
                return
            rows, summary = curve_eval(engine, tokenizer, dev, root, gold_dev, s, stops, deadline)
            path = out / "curve" / f"step-{step:04d}.jsonl"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                path.rename(path.with_name(path.name + f".interrupted-{time.time_ns()}"))
            for row in rows:
                append_record(path, row)
            append_record(out / "curve.jsonl", {"step": step, "attempt": attempt, **summary,
                                                "behaviour": behaviour(rows)})
            atomic_json(out / "curve-estimate.json", {"seconds": summary["wall_seconds"]})
            state["curve_steps"].append(step)

        while state["step"] < target:
            curve(state["step"])
            estimate = 1.3 * max(recent[-3:]) if recent else 600.0
            if left() < estimate + 120:
                stopped_by = "deadline"
                break
            t0 = time.monotonic()
            index = state["step"]
            tasks = step_tasks(order, index, s["tasks_per_step"])
            rows, examples, gen = rollouts(engine, tokenizer, tasks, root, oracle, s, stops, s["group_size"])
            t_gen = time.monotonic() - t0
            rewards = [1.0 if r["correct"] else 0.0 for r in rows]
            valid = [r["valid"] for r in rows]
            weights = advantages(rewards, s["group_size"], plan["method"])
            for g in range(0, len(rows), s["group_size"]):  # a group with an unscorable reference teaches nothing
                if not all(valid[g:g + s["group_size"]]):
                    weights[g:g + s["group_size"]] = [0.0] * s["group_size"]
            items = []
            for row, example, reward, weight in zip(rows, examples, rewards, weights):
                row["reward"], row["weight"] = reward, weight
                row["trained"] = bool(weight and example and example["usable"])
                row["unusable"] = example["reason"] if example and not example["usable"] else None
                if row["trained"]:
                    items.append({**example, "weight": weight})
            check = None
            if fresh and index == 0 and state["lora_check"] is None:
                usable = [e for e in examples if e and e["usable"]][:LORA_CHECK_TRAJECTORIES]
                model.eval()
                check = lora_check(model, usable, pad_id=pad_id, budget=s["micro_tokens"])
                state["lora_check"] = check
                if not check["passed"]:
                    raise RuntimeError(f"vLLM did not sample from the start LoRA: {check}")
            t1 = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = backward_step(model, items, pad_id=pad_id, normaliser=normaliser, cap=plan["tis_cap"],
                                  budget=s["micro_tokens"]) if items else {}
            lr = plan["learning_rate"] * min(1.0, (state["optimizer_updates"] + 1) / plan["warmup_steps"])
            norm = 0.0
            if items:
                norm = float(torch.nn.utils.clip_grad_norm_(params, plan["max_grad_norm"], error_if_nonfinite=True))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                state["optimizer_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():  # micro-batch shapes change every step; hand cached blocks back
                torch.cuda.empty_cache()
            t_train = time.monotonic() - t1
            state["step"] += 1
            nxt = policy_root / f"step-{state['step']:04d}"
            save_adapter(model, nxt)
            engine.set_policy(nxt, state["step"])
            if state["step"] % s["save_steps"] == 0:
                saved = save_adapter(model, out / "milestones" / f"step-{state['step']:04d}" / "adapter", base)
                marks = milestones(out)
                marks[state["step"]] = saved
                atomic_json(out / "milestones.json", {str(k): v for k, v in sorted(marks.items())})
            state["active_seconds"] = prior + time.monotonic() - clock
            journal = out / "rollouts" / f"step-{index:04d}.jsonl"
            journal.parent.mkdir(exist_ok=True)
            for row in rows:
                append_record(journal, row)
            step_seconds = time.monotonic() - t0
            recent.append(step_seconds)
            targets = max(1, stats.get("targets", 0))
            append_record(out / "metrics.jsonl", {
                "step": state["step"], "attempt": attempt, "tasks": [t.id for t in tasks],
                "boundary_pass": index * s["tasks_per_step"] // len(order),
                "reward_mean": sum(rewards) / len(rewards), **group_counts(rewards, valid, s["group_size"]),
                "behaviour": behaviour(rows),
                "statuses": dict(Counter(r["status"] for r in rows)), "trained_trajectories": len(items),
                "unusable": dict(Counter(r["unusable"] for r in rows if r["unusable"])),
                "mean_turns": sum(len(r["steps"]) for r in rows) / len(rows),
                "mean_completion_tokens": gen["completion_tokens"] / len(rows),
                "length_stops": gen["length_stops"], "context_rejections": gen["context_rejections"],
                "optimizer_updated": bool(items), "optimizer_updates": state["optimizer_updates"],
                "lr": lr if items else 0.0, "grad_norm": norm, "loss": stats.get("loss", 0.0),
                "targets": stats.get("targets", 0), "train_tokens": stats.get("tokens", 0),
                "micro_batches": stats.get("micro_batches", 0),
                "mean_abs_logp_gap": stats.get("abs_gap_sum", 0.0) / max(1, stats.get("known_targets", 0)),
                "max_abs_logp_gap": stats.get("max_abs_gap", 0.0),
                "tis_clipped_fraction": stats.get("tis_clipped", 0) / targets,
                "entropy": stats.get("entropy_sum", 0.0) / targets, "mean_logp": stats.get("logp_sum", 0.0) / targets,
                "lora_check": check, "generate_seconds": t_gen, "train_seconds": t_train, "step_seconds": step_seconds,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "active_seconds": state["active_seconds"]})
            # durable last: a crash before this line replays the step, and _set_aside keeps the replayed records
            save_checkpoint(out, model, optimizer, state)
            for old in sorted(policy_root.glob("step-*"))[:-2]:
                shutil.rmtree(old, ignore_errors=True)
            gc.collect()
        curve(state["step"])
        state["active_seconds"] = prior + time.monotonic() - clock
        save_checkpoint(out, model, optimizer, state)
        marks = milestones(out)
        if s["phase"] == "smoke" and stopped_by == "planned" and not state["optimizer_updates"]:
            raise RuntimeError("Smoke run never updated the policy; the gradient path is unexercised")
        common = {"identity": identity, "family": FAMILY, "seed": seed, "steps_completed": state["step"],
                  "planned_steps": s["planned_steps"], "explore_steps": s["explore_steps"],
                  "optimizer_updates": state["optimizer_updates"], "milestones": {str(k): v for k, v in marks.items()},
                  "last_milestone": max(marks) if marks else None, "lora_check": state["lora_check"],
                  "curve_steps": state["curve_steps"], "active_seconds": state["active_seconds"], "attempt": attempt}
        if stopped_by == "planned" and state["step"] < s["explore_steps"]:
            held = {**common, "step": state["step"], "stopped_by": "paused"}
            atomic_json(out / "paused.json", held)
            return {**held, "finished": False}
        result = {"finished": True, **common, "stopped_by": stopped_by}
        atomic_json(out / "result.json", result)
        return result
