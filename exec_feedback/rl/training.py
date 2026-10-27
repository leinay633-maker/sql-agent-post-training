"""Online two-turn RLOO: fresh groups, one update, train-only terminal rewards."""
from __future__ import annotations

import gc
import importlib.metadata
import random
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import Limits, Task
from ..storage import (CallBudget, append_record, atomic_json, exclusive_lock,
                       fingerprint, read_json, recover_records)
from .backend import LocalPolicy
from .checkpoint import restore_checkpoint, save_checkpoint, trainable
from .objective import backward_batch
from .rollout import TrainingOracle, rollouts


def draw_tasks(tasks: list[Task], cursor: int, count: int, seed: int) -> list[Task]:
    selected = []
    while len(selected) < count:
        epoch, offset = divmod(cursor, len(tasks))
        order = list(range(len(tasks)))
        random.Random(seed + epoch).shuffle(order)
        take = min(count - len(selected), len(tasks) - offset)
        selected.extend(tasks[i] for i in order[offset:offset + take])
        cursor += take
    return selected


def train_loop(model: Any, policy: Any, oracle: TrainingOracle,
               tasks: list[Task], root: Path, settings: dict[str, Any],
               out: Path, *, seed: int, identity: str,
               stop_after: int | None = None) -> dict[str, Any]:
    """CPU-testable optimizer loop. Validation has no entry point here."""
    import torch

    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("Online training requires train tasks only")
    parameters = list(trainable(model).values())
    if not parameters:
        raise ValueError("No trainable policy parameters")
    cuda = parameters[0].device.type == "cuda"
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0.0)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("rollouts.jsonl", "metrics.jsonl", "attempts.jsonl"):
        recover_records(out / name)  # preserves a torn suffix; never drops complete lines
    state = restore_checkpoint(out / "checkpoints", model, optimizer, identity)
    if state is None:
        torch.manual_seed(seed)
        random.seed(seed)
        state = {"identity": identity, "update": 0, "cursor": 0,
                 "optimizer_updates": 0, "history": [], "active_seconds": 0.0}
        save_checkpoint(out / "checkpoints", model, optimizer, state)
    # Sampling and teacher forcing must not use different dropout distributions.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    status_path = out / "status.json"
    prior = read_json(status_path)["active_seconds"] if status_path.exists() else 0.0
    prior = max(prior, state["active_seconds"])
    started = time.monotonic()
    attempt = uuid.uuid4().hex
    last_saved = state["update"]
    append_record(out / "attempts.jsonl", {"attempt": attempt, "resumed_update": last_saved})
    try:
        while state["update"] < settings["updates"]:
            elapsed = prior + time.monotonic() - started
            if elapsed >= settings["train_seconds"]:
                break
            if stop_after is not None and state["update"] >= stop_after:
                break
            before = time.perf_counter()
            prompts = draw_tasks(tasks, state["cursor"], settings["prompt_batch"], seed)
            repeated = [task for task in prompts for _ in range(settings["group_size"])]
            append_record(out / "attempts.jsonl", {
                "attempt": attempt, "update": state["update"] + 1,
                "status": "rollout_started", "prompt_ids": [t.id for t in prompts]})
            episodes = []
            for episode in rollouts(repeated, root, policy, slots=settings["slots"],
                                    limits=oracle.limits):
                row = oracle.score(episode)
                row.update(attempt=attempt, update=state["update"] + 1)
                append_record(out / "rollouts.jsonl", row)
                episodes.append(row)
            episodes.sort(key=lambda row: row["request_index"])
            if [r["request_index"] for r in episodes] != list(range(len(repeated))):
                raise ValueError("Missing/duplicate rollout; do not train an incomplete group")
            if any(row["task_id"] != task.id for row, task in zip(episodes, repeated)):
                raise ValueError("Rollout task changed its registered group position")
            rollout_seconds = time.perf_counter() - before
            model.train()
            optimizer.zero_grad(set_to_none=True)
            metrics = backward_batch(model, episodes, settings["group_size"])
            if metrics["has_gradient"]:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                state["optimizer_updates"] += 1
                metrics["grad_norm"] = float(norm)
            else:
                metrics["grad_norm"] = 0.0
            if cuda:
                torch.cuda.synchronize()
            state["update"] += 1
            state["cursor"] += len(prompts)
            state["active_seconds"] = prior + time.monotonic() - started
            metrics.update(
                update=state["update"], attempt=attempt,
                prompt_ids=[task.id for task in prompts],
                rollout_seconds=rollout_seconds,
                update_seconds=time.perf_counter() - before,
                optimizer_updates=state["optimizer_updates"],
                natural_trajectories=len(episodes), rule_trajectories=0,
                generation_failures=sum(r.get("generation_failed", False) for r in episodes),
                reward_statuses=dict(Counter(r.get("reward_status", "fixture")
                                             for r in episodes)),
                active_seconds=state["active_seconds"])
            state["history"].append(metrics)
            append_record(out / "metrics.jsonl", metrics)
            if state["update"] % settings["save_steps"] == 0:
                save_checkpoint(out / "checkpoints", model, optimizer, state)
                last_saved = state["update"]
        if state["update"] != last_saved:
            save_checkpoint(out / "checkpoints", model, optimizer, state)
            last_saved = state["update"]
    finally:
        # On an interrupted backward, only the last complete checkpoint is
        # resumable. Extra journal attempts stay visible, not counted twice.
        atomic_json(status_path, {"identity": identity, "durable_update": last_saved,
                    "active_seconds": prior + time.monotonic() - started})
    return {**state, "completed": state["update"] == settings["updates"],
            "active_seconds": prior + time.monotonic() - started}


def train_seed(run: Path, seed: int) -> dict[str, Any]:
    import torch
    from ..models import load_model, model_identity
    from .study import load_run

    frozen, tasks = load_run(run, training=True)
    settings = frozen["settings"]
    if seed not in settings["seeds"]:
        raise ValueError("Seed was not registered before evaluation")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H20 BF16 smoke required; no implicit CPU training")
    out = run / f"E4_seed_{seed}"
    identity = fingerprint({"run": frozen["identity"], "seed": seed, "method": "online-rloo"})
    with exclusive_lock(out / ".lock"):
        if (out / "result.json").exists():
            result = read_json(out / "result.json")
            actual = model_identity(frozen["model"], out / "adapter")["adapter_hash"]
            if result["identity"] != identity or result["adapter_hash"] != actual:
                raise ValueError("Completed RL artifacts changed")
            return result
        stage_started = time.perf_counter()
        torch.manual_seed(seed)
        model, tokenizer = load_model(frozen["model"], Path(frozen["warm_adapter"]), training=True)
        budget = CallBudget(out / "budget.json", settings["train_calls"], settings["train_seconds"])
        policy = LocalPolicy(model, tokenizer, batch_size=settings["generation_batch"],
                             max_context=settings["max_context"], max_tokens=settings["max_tokens"],
                             sample=True, budget=budget)
        oracle = TrainingOracle(tasks, Path(frozen["raw_root"]), Limits(**settings["limits"]))
        torch.cuda.reset_peak_memory_stats()
        state = train_loop(model, policy, oracle, tasks, Path(frozen["raw_root"]),
                           settings, out, seed=seed, identity=identity)
        if not state["completed"]:
            raise RuntimeError("RL budget exhausted; checkpoint retained, no completed model claimed")
        adapter = out / "adapter"
        if adapter.exists():
            adapter.rename(out / f"adapter.interrupted-{time.time_ns()}")
        model.eval()
        samples = recover_records(out / "rollouts.jsonl")
        tokens = next((r["sampled_turns"][0]["prompt_ids"][-128:]
                       for r in samples if r["sampled_turns"]), None)
        if not tokens:
            raise RuntimeError("No generated action available for reload verification")
        probe = torch.tensor([tokens], device=next(model.parameters()).device)
        with torch.inference_mode():
            expected = model(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        model.save_pretrained(str(adapter), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter))
        atomic_json(adapter / "efa-base.json", model_identity(frozen["model"]))
        peak = torch.cuda.max_memory_allocated()
        del policy, model, probe
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, _ = load_model(frozen["model"], adapter)
        reloaded.eval()
        probe = torch.tensor([tokens], device=next(reloaded.parameters()).device)
        with torch.inference_mode():
            actual = reloaded(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        verified = bool(torch.allclose(actual, expected, atol=0.05, rtol=0.05))
        verification = {"adapter_reload_verified": verified,
                        "reload_max_abs_delta": float((actual - expected).abs().max()),
                        "probe_tokens": tokens}
        atomic_json(out / "reload.json", verification)
        if not verified:
            raise RuntimeError("RL adapter reload failed; no completed result written")
        result = {
            "identity": identity, "run_identity": frozen["identity"], "seed": seed,
            "training_completed": True, **verification,
            "adapter_hash": model_identity(frozen["model"], adapter)["adapter_hash"],
            "peak_allocated_bytes": peak, "durable_training": state,
            "invocation_wall_seconds": time.perf_counter() - stage_started,
            "invocation_includes": "load, train reward preflight, remaining updates, save, reload",
            "packages": {p: importlib.metadata.version(p)
                         for p in ("torch", "transformers", "peft", "accelerate")},
            "unique_train_tasks": len({t for row in state["history"] for t in row["prompt_ids"]}),
            "validation_used_in_training": False, "kl_beta": 0.0,
        }
        atomic_json(out / "result.json", result)
        return result
