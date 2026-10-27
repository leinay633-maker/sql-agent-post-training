"""v0.12 OPD runs: v0.9's online loop (one GPU, vLLM samples with the current LoRA, PEFT takes one on-policy step) with
the teacher's per-token signal in place of the execution reward.

Every usable sampled dialogue goes to the teacher as one token sequence (v0.9's trajectory: the last turn's prompt holds
every earlier turn as the template re-rendered it). The teacher returns log p_T of each token given its prefix; on the
sampled assistant tokens the signal is A_t = log p_T(y_t) - log p_vLLM(y_t), the negative single-sample reverse KL with
discount zero, and the loss is v0.9's: -sum w_t A_t log p_theta(y_t) / constant, w_t = min(p_theta / p_vLLM, cap).
The oracle still scores the final SQL against the train gold, but only for the record: no gold enters the gradient.

OPD starts from a fresh LoRA on the base 4B (its B is zero, so step 0 samples the base model); TROPD (exploratory)
continues v0.8's TR LoRA. A run trains to planned_steps or to its deadline; smoke settings may stop it earlier and resume
it (pause/resume path). Records as v0.9 and v0.11, plus the teacher's timing and the per-token KL statistics.
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
from ..v9.online import (_set_aside, _version, curve_eval, entropy_sum, gold_for, lora_check, micro_batches,
                         milestones, restore_checkpoint, rollouts, save_adapter, save_checkpoint, sequence_logprobs,
                         step_tasks, task_order, vllm_engine)
from ..v9.online import LORA_CHECK_TRAJECTORIES
from ..v11.online import behaviour
from . import EXPLORE, EXPLORE_START, FAMILY
from .freeze import imported_adapter, load, model_ready
from .teacher import TeacherClient, TeacherUnavailable, identity_of

FAMILIES = (FAMILY, EXPLORE)
ADVANTAGE_FLOOR = -5.0   # only counted (share of tokens the teacher finds e^5 times less likely), never clipped


def run_name(family: str, seed: int) -> str:
    return f"{family}_seed{seed}"


def registered(settings: dict[str, Any], family: str) -> list[int]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown online family: {family}")
    return settings["explore_seeds"] if family == EXPLORE else settings["seeds"]


def plan_of(settings: dict[str, Any], family: str) -> dict[str, Any]:
    keys = ("tasks_per_step", "group_size", "planned_steps", "temperature", "learning_rate", "warmup_steps",
            "max_grad_norm", "loss_tokens", "tis_cap", "micro_tokens", "save_steps", "curve_every", "lora_r",
            "lora_alpha", "lora_check_step")
    return {"method": "opd", "start": EXPLORE_START if family == EXPLORE else "fresh-lora",
            "teacher_priority": 1 if family == EXPLORE else 0, **{k: settings[k] for k in keys}}


def fresh_lora(spec: dict[str, Any], start: Path | None, *, settings: dict[str, Any], seed: int) -> tuple[Any, Any]:
    """The base 4B (gradient checkpointing) with a new LoRA shaped as v0.8's R/TR (r, alpha, all-linear; dropout set to
    zero for training below), or with an imported adapter to continue (TROPD)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from ..models import load_model

    model, tokenizer = load_model(spec, training=True)
    if start is not None:
        return PeftModel.from_pretrained(model, str(start), is_trainable=True), tokenizer
    torch.manual_seed(seed)
    return get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=settings["lora_r"],
                                            lora_alpha=settings["lora_alpha"], lora_dropout=0.05,
                                            target_modules="all-linear", bias="none")), tokenizer


def teacher_client(home: Path, frozen: dict[str, Any], family: str, seed: int, tag: str,
                   until: str | None = None) -> TeacherClient:
    """The run's client: no call waits past the run's deadline. The low-priority run (TROPD) has no other limit: the
    main runs may keep the teacher busy for long stretches, and that wait is the preregistered price, not a failure."""
    plan = plan_of(frozen["settings"], family)
    deadline = deadline_from(until)
    wait = None if plan["teacher_priority"] and deadline is not None else frozen["settings"]["teacher_wait_seconds"]
    return TeacherClient(home, identity_of(frozen), wait=wait, until=deadline, priority=plan["teacher_priority"],
                         caller=f"{tag}:{run_name(family, seed)}")


# ------------------------------------------------------------------------------------------------ objective
def with_teacher(examples: list[dict[str, Any] | None], teacher: Any) -> tuple[list[dict[str, Any] | None],
                                                                             list[str | None], dict[str, Any]]:
    """Score every usable trajectory once; attach the teacher's log-probs at its trained positions. A trajectory whose
    positions lack a finite teacher value is kept out of the gradient ('teacher_missing')."""
    keep = [i for i, e in enumerate(examples) if e and e["usable"]]
    started = time.monotonic()
    rows = teacher.score([examples[i]["input_ids"] for i in keep]) if keep else []
    seconds = time.monotonic() - started
    if len(rows) != len(keep):
        raise ValueError("The teacher returned a different number of trajectories")
    out: list[dict[str, Any] | None] = [None] * len(examples)
    missing: list[str | None] = [None] * len(examples)
    for i, row in zip(keep, rows):
        example = examples[i]
        if len(row) != len(example["input_ids"]):
            raise ValueError("Teacher row length differs from the trajectory")
        values = [row[p] for p in example["positions"]]
        if any(v is None or not math.isfinite(v) for v in values):
            missing[i] = "teacher_missing"
            continue
        out[i] = {**example, "teacher_logprobs": values}
    last = getattr(teacher, "last", {}) or {}
    return out, missing, {"teacher_seconds": seconds, "teacher_sequences": len(keep),
                          "teacher_tokens": sum(len(examples[i]["input_ids"]) for i in keep),
                          "teacher_score_seconds": last.get("score_seconds"),
                          "teacher_wait_seconds": last.get("wait_seconds")}


def dialogue_kl(item: dict[str, Any]) -> float | None:
    """Mean log p_vLLM - log p_T over the trajectory's tokens with a sampler log-prob (the per-token reverse KL
    estimate the run is pushed down on)."""
    pairs = [(o, t) for o, t in zip(item["old_logprobs"], item["teacher_logprobs"]) if o is not None]
    return sum(o - t for o, t in pairs) / len(pairs) if pairs else None


def opd_backward(model: Any, items: list[dict[str, Any]], *, pad_id: int, normaliser: float, cap: float,
                 budget: int) -> dict[str, Any]:
    """Accumulate -sum(A_t * min(ratio, cap) * logp) / normaliser, A_t = log p_T - log p_sampler per token (the
    trainer's own, detached, where vLLM gave no log-prob); the caller steps the optimizer."""
    import torch

    stats = Counter()
    for batch in micro_batches(items, budget):
        logp, picked = sequence_logprobs(model, batch, pad_id)
        device = logp.device
        old_values = [v for x in batch for v in x["old_logprobs"]]
        known = torch.tensor([v is not None for v in old_values], device=device)
        old = torch.tensor([v if v is not None else 0.0 for v in old_values], device=device, dtype=torch.float32)
        teacher = torch.tensor([v for x in batch for v in x["teacher_logprobs"]], device=device, dtype=torch.float32)
        if teacher.shape != logp.shape:
            raise ValueError("Teacher log-probs do not line up with the trained positions")
        sampler = torch.where(known, old, logp.detach())
        advantage = teacher - sampler
        gap = (logp.detach() - old).masked_fill(~known, 0.0)
        ratio = torch.where(known, gap.exp(), torch.ones_like(gap))
        tis = ratio.clamp(max=cap)
        loss = -(advantage * tis * logp).sum() / normaliser
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite OPD loss")
        loss.backward()
        stats["loss"] += float(loss.detach())
        stats["targets"] += int(logp.numel())
        stats["known_targets"] += int(known.sum())
        stats["abs_gap_sum"] += float(gap.abs().sum())
        stats["max_abs_gap"] = max(stats["max_abs_gap"], float(gap.abs().max()) if gap.numel() else 0.0)
        stats["tis_clipped"] += int((ratio > cap).sum())
        stats["entropy_sum"] += entropy_sum(picked)
        stats["logp_sum"] += float(logp.detach().sum())
        stats["reverse_kl_sum"] += float((sampler - teacher).sum())
        stats["teacher_logp_sum"] += float(teacher.sum())
        stats["sampler_logp_sum"] += float(sampler.sum())
        stats["advantage_abs_sum"] += float(advantage.abs().sum())
        stats["advantage_low"] += int((advantage < ADVANTAGE_FLOOR).sum())
        stats["tokens"] += sum(len(x["input_ids"]) for x in batch)
        stats["micro_batches"] += 1
        del logp, picked, loss
    return dict(stats)


# ------------------------------------------------------------------------------------------------ run state
def paused(out: Path) -> dict[str, Any] | None:
    path = out / "paused.json"
    return read_json(path) if path.exists() else None


def ended(out: Path) -> dict[str, Any] | None:
    """The record of a run that trains no further: it reached planned_steps, or its deadline stopped it."""
    path = out / "result.json"
    record = read_json(path) if path.exists() else None
    return record if record and record.get("finished") else None


def start_of(run: Path, frozen: dict[str, Any], family: str, seed: int) -> tuple[Path | None, str]:
    if family == EXPLORE:
        return imported_adapter(run, frozen, f"{EXPLORE_START}_seed{seed}")
    return None, f"fresh-lora:{seed}"


def train_online(run: Path, *, family: str, seed: int, until: str | None, teacher: Any, stop_step: int | None = None,
                 engine_factory: Callable[..., Any] = vllm_engine,
                 policy_loader: Callable[..., tuple[Any, Any]] = fresh_lora) -> dict[str, Any]:
    """Train to planned_steps, the deadline or the end of the teacher service (result.json). In smoke settings stop_step
    may stop it earlier: it pauses (paused.json, no result) and a later call resumes from the checkpoint."""
    import torch
    from ..models import model_identity
    from ..rft.sampling import Oracle
    from ..v7.engine import stop_ids as model_stop_ids

    frozen, pool = load(run, "pool")
    _, dev = load(run, "dev")
    s = frozen["settings"]
    if seed not in registered(s, family):
        raise ValueError("Unregistered family/seed")
    target = s["planned_steps"] if stop_step is None else stop_step
    if target != s["planned_steps"] and (s["phase"] != "smoke" or not 0 < target < s["planned_steps"]):
        raise ValueError("Only smoke runs pause before planned_steps")
    spec = model_ready(frozen) if engine_factory is vllm_engine else frozen["models"]["student"]["spec"]
    start, start_hash = start_of(run, frozen, family, seed)
    plan = plan_of(s, family)
    identity = fingerprint({"run": frozen["identity"], "family": family, "seed": seed, "start": start_hash,
                            "plan": plan, "model": frozen["models"]["student"]["identity"],
                            "teacher": frozen["models"]["teacher"]["identity"]})
    out = run / "train" / run_name(family, seed)
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
        model, tokenizer = policy_loader(spec, start, settings=s, seed=seed)
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
        if state is None:
            state = {"identity": identity, "step": 0, "optimizer_updates": 0, "active_seconds": 0.0,
                     "curve_steps": [], "lora_check": None}
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "family": family, "seed": seed,
            "start_adapter": str(start) if start else None, "start_hash": start_hash, "plan": plan,
            "teacher": frozen["models"]["teacher"]["identity"],
            "packages": {**versions(), "peft": _version("peft")}})
        append_record(out / "attempts.jsonl", {"attempt": attempt, "resumed_step": state["step"], "target": target,
                                               "engine_load_seconds": engine_seconds,
                                               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        _set_aside(out, state["step"])
        prior, clock = state["active_seconds"], time.monotonic()
        order = task_order(pool, s["salt"], seed)
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
            try:
                scored, missing, timing = with_teacher(examples, teacher)
            except TeacherUnavailable:
                # no teacher before the run's own deadline, or none ever again: the run ends here and this step's
                # samples are dropped; a teacher that is only away fails the attempt (retried from the checkpoint)
                if left() > 0 and not getattr(teacher, "gone", lambda: False)():
                    raise
                stopped_by = "deadline" if left() <= 0 else "teacher_gone"
                break
            rewards = [1.0 if r["correct"] else 0.0 for r in rows]
            items = []
            for row, example, item, lost in zip(rows, examples, scored, missing):
                row["reward"] = 1.0 if row["correct"] else 0.0
                row["trained"] = item is not None
                row["unusable"] = example["reason"] if example and not example["usable"] else lost
                row["teacher_kl"] = dialogue_kl(item) if item is not None else None
                if item is not None:
                    items.append(item)
            check = None
            if index == s["lora_check_step"] and state["lora_check"] is None:
                usable = [e for e in examples if e and e["usable"]][:LORA_CHECK_TRAJECTORIES]
                model.eval()
                check = lora_check(model, usable, pad_id=pad_id, budget=s["micro_tokens"])
                state["lora_check"] = {**check, "step": index}
                if not check["passed"]:
                    raise RuntimeError(f"vLLM did not sample from the trained LoRA: {check}")
            t1 = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = opd_backward(model, items, pad_id=pad_id, normaliser=normaliser, cap=plan["tis_cap"],
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
                "pass": index * s["tasks_per_step"] // len(order),
                "reward_mean": sum(rewards) / len(rewards), "behaviour": behaviour(rows),
                "statuses": dict(Counter(r["status"] for r in rows)), "trained_trajectories": len(items),
                "unusable": dict(Counter(r["unusable"] for r in rows if r["unusable"])),
                "mean_turns": sum(len(r["steps"]) for r in rows) / len(rows),
                "mean_completion_tokens": gen["completion_tokens"] / len(rows),
                "length_stops": gen["length_stops"], "context_rejections": gen["context_rejections"],
                "optimizer_updated": bool(items), "optimizer_updates": state["optimizer_updates"],
                "lr": lr if items else 0.0, "grad_norm": norm, "loss": stats.get("loss", 0.0),
                "targets": stats.get("targets", 0), "train_tokens": stats.get("tokens", 0),
                "micro_batches": stats.get("micro_batches", 0),
                "reverse_kl": stats.get("reverse_kl_sum", 0.0) / targets,
                "teacher_logp": stats.get("teacher_logp_sum", 0.0) / targets,
                "sampler_logp": stats.get("sampler_logp_sum", 0.0) / targets,
                "advantage_abs": stats.get("advantage_abs_sum", 0.0) / targets,
                "advantage_low_fraction": stats.get("advantage_low", 0) / targets,
                "mean_abs_logp_gap": stats.get("abs_gap_sum", 0.0) / max(1, stats.get("known_targets", 0)),
                "max_abs_logp_gap": stats.get("max_abs_gap", 0.0),
                "tis_clipped_fraction": stats.get("tis_clipped", 0) / targets,
                "entropy": stats.get("entropy_sum", 0.0) / targets, "mean_logp": stats.get("logp_sum", 0.0) / targets,
                "lora_check": check, "generate_seconds": t_gen, **timing, "train_seconds": t_train,
                "step_seconds": step_seconds,
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
        common = {"identity": identity, "family": family, "seed": seed, "steps_completed": state["step"],
                  "planned_steps": s["planned_steps"], "optimizer_updates": state["optimizer_updates"],
                  "milestones": {str(k): v for k, v in marks.items()},
                  "last_milestone": max(marks) if marks else None, "lora_check": state["lora_check"],
                  "curve_steps": state["curve_steps"], "active_seconds": state["active_seconds"], "attempt": attempt}
        if stopped_by == "planned" and state["step"] < s["planned_steps"]:
            held = {**common, "step": state["step"], "stopped_by": "paused"}
            atomic_json(out / "paused.json", held)
            return {**held, "finished": False}
        result = {"finished": True, **common, "stopped_by": stopped_by}
        atomic_json(out / "result.json", result)
        return result
