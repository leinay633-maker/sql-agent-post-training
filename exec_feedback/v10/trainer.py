"""The v0.10 trainer: one process per trainer GPU (torchrun), FSDP2 over all of them.

Parameters, gradients and AdamW state are fully sharded in fp32; forward and backward run in bf16 with activation
checkpointing per decoder layer. Rank 0 also drives the run: it posts the rollout requests, merges the workers' groups,
computes the RLOO advantages, splits the trained trajectories over the ranks and writes every record. Each rank then runs
the same number of forward/backward passes (short ranks get zero-weight dummy micro-batches): FSDP's collectives would
hang otherwise. The loss, the truncated importance weights and the token selection are v0.9's; the loss is multiplied by
the world size because FSDP averages gradients over ranks while v0.9's normaliser is a constant per step.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import math
import os
import shutil
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import digest
from ..storage import append_record, atomic_json, fingerprint, read_json, recover_records
from ..v6.rollouts import deadline_from, versions
from ..v9.online import advantages, entropy_sum, group_counts, micro_batches, sequence_logprobs, step_tasks, task_order
from . import FAMILIES
from .freeze import load, model_ready
from .merge import merged_start
from .protocol import Channel, wait_for
from .rollout import assignment, curve_assignment
from .weights import WEIGHTS, export_full, model_dir, same_checksum, write_weights

H20_BF16_DENSE = 148e12        # FLOP/s per GPU, NVIDIA's H20 datasheet (dense BF16); only for the MFU column
WEIGHTS_CHECK_TRAJECTORIES = 32
HEARTBEAT_LIMIT = 180.0        # a worker silent this long is dead
BENCH_STEPS = 3                # the trained items of the first steps are kept for the trainer-only benchmark
STATS = ("loss", "targets", "known_targets", "abs_gap_sum", "tis_clipped", "entropy_sum", "logp_sum", "tokens",
         "micro_batches")


def run_name(family: str, seed: int) -> str:
    return f"{family}_seed{seed}"


def plan_of(settings: dict[str, Any], family: str) -> dict[str, Any]:
    keys = ("tasks_per_step", "group_size", "planned_steps", "temperature", "learning_rate", "warmup_steps",
            "max_grad_norm", "loss_tokens", "tis_cap", "micro_tokens", "save_steps", "curve_every",
            "checkpoint_every", "reshard_after_forward", "trainer_gpus", "rollout_gpus")
    return {"method": "rloo", "staleness": FAMILIES[family], **{k: settings[k] for k in keys}}


def milestones(out: Path) -> dict[int, str]:
    """Saved milestone model directories of a run: step -> directory identity."""
    path = out / "milestones.json"
    return {int(k): v for k, v in read_json(path).items()} if path.exists() else {}


# ------------------------------------------------------------------------------------------------ collectives
class Dist:
    """Rank, world size and the few collectives a run needs (rank 0's objects, per-rank objects, sums, maxima)."""

    def __init__(self, device: Any) -> None:
        import torch.distributed as dist

        self.dist, self.device = dist, device
        on = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if on else 0
        self.world = dist.get_world_size() if on else 1

    def share(self, value: Any) -> Any:
        if self.world == 1:
            return value
        box = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(box, src=0)
        return box[0]

    def scatter(self, values: list[Any] | None) -> Any:
        """values[r] (given on rank 0) to rank r."""
        if self.world == 1:
            return values[0]
        box = [None]
        self.dist.scatter_object_list(box, values if self.rank == 0 else None, src=0)
        return box[0]

    def reduce(self, values: list[float], op: str) -> list[float]:
        if self.world == 1:
            return list(values)
        import torch

        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM if op == "sum" else self.dist.ReduceOp.MAX)
        return tensor.tolist()

    def barrier(self) -> None:
        if self.world > 1:
            self.dist.barrier()


# ------------------------------------------------------------------------------------------------ model
def load_policy(start: Path, device: Any) -> Any:
    """The merged start weights as fp32 master weights, with non-reentrant activation checkpointing."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(start), torch_dtype=torch.float32, trust_remote_code=False,
                                                 local_files_only=True, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


def shard(model: Any, mesh: Any, reshard_after_forward: bool) -> Any:
    """FSDP2 per decoder layer plus the root (embeddings, final norm, tied head). reshard_after_forward=True frees each
    gathered layer after its forward and gathers it again for the backward (ZeRO-3 like); False keeps it gathered until
    its backward (ZeRO-2 like: one all-gather per layer and micro-batch, more memory)."""
    import torch
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    for module in model.modules():  # sampling has no dropout, so teacher forcing must not have it either
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    return model


def rank_shards(items: list[dict[str, Any]], world: int, budget: int, pad_id: int) -> list[list[list[dict[str, Any]]]]:
    """Trajectories over ranks by token load (longest first, ties by key), micro-batched within each rank; every rank
    gets the same number of micro-batches, padded with zero-weight dummies."""
    loads, shards = [0] * world, [[] for _ in range(world)]
    for item in sorted(items, key=lambda x: (-len(x["input_ids"]), x.get("key", ""))):
        rank = min(range(world), key=lambda r: (loads[r], r))
        shards[rank].append(item)
        loads[rank] += len(item["input_ids"])
    batches = [micro_batches(part, budget) if part else [] for part in shards]
    count = max(len(b) for b in batches)
    dummy = {"key": "", "input_ids": [pad_id, pad_id], "positions": [1], "old_logprobs": [None], "weight": 0.0,
             "dummy": True}
    return [b + [[dict(dummy)] for _ in range(count - len(b))] for b in batches]


def backward_rank(model: Any, batches: list[list[dict[str, Any]]], *, pad_id: int, normaliser: float, cap: float,
                  scale: float, grad: bool = True) -> dict[str, float]:
    """v0.9's -sum(w * min(ratio, cap) * logp) / normaliser over this rank's micro-batches, times `scale`; with
    grad=False only the statistics (the step-0 weights check). Dummy micro-batches run forward and backward but count
    nowhere."""
    import torch

    stats: Counter = Counter()
    for batch in batches:
        real = not all(x.get("dummy") for x in batch)
        with contextlib.nullcontext() if grad else torch.no_grad():
            logp, picked = sequence_logprobs(model, batch, pad_id)
        device = logp.device
        weights = torch.tensor([x["weight"] for x in batch for _ in x["positions"]], device=device, dtype=torch.float32)
        old_values = [v for x in batch for v in x["old_logprobs"]]
        known = torch.tensor([v is not None for v in old_values], device=device)
        old = torch.tensor([v if v is not None else 0.0 for v in old_values], device=device, dtype=torch.float32)
        gap = (logp.detach() - old).masked_fill(~known, 0.0)
        ratio = torch.where(known, gap.exp(), torch.ones_like(gap))
        if grad:
            loss = -(weights * ratio.clamp(max=cap) * logp).sum() / normaliser * scale
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite policy loss")
            loss.backward()
            if real:
                stats["loss"] += float(loss.detach()) / scale
            del loss
        if real:
            stats["targets"] += int(logp.numel())
            stats["known_targets"] += int(known.sum())
            stats["abs_gap_sum"] += float(gap.abs().sum())
            stats["max_abs_gap"] = max(stats["max_abs_gap"], float(gap.abs().max()) if gap.numel() else 0.0)
            stats["tis_clipped"] += int((ratio > cap).sum())
            stats["entropy_sum"] += entropy_sum(picked)
            stats["logp_sum"] += float(logp.detach().sum())
            stats["tokens"] += sum(len(x["input_ids"]) for x in batch)
            stats["micro_batches"] += 1
        del logp, picked
    return dict(stats)


def reduce_stats(d: Dist, stats: dict[str, float]) -> dict[str, float]:
    out = dict(zip(STATS, d.reduce([float(stats.get(k, 0.0)) for k in STATS], "sum")))
    out["max_abs_gap"] = d.reduce([float(stats.get("max_abs_gap", 0.0))], "max")[0]
    return out


def clip_and_norm(model: Any, max_norm: float) -> float:
    """Global gradient norm over all shards, then clipping (TorchTitan's FSDP2 pattern: the norm is a partial DTensor
    until full_tensor() reduces it over the ranks)."""
    import torch

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total = torch.nn.utils.get_total_norm(grads, 2.0, error_if_nonfinite=False, foreach=True)
    if hasattr(total, "full_tensor"):
        total = total.full_tensor()
    norm = float(total)
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm")
    torch.nn.utils.clip_grads_with_norm_(model.parameters(), max_norm, total, foreach=True)
    return norm


def make_optimizer(model: Any, lr: float) -> Any:
    import torch

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, foreach=True)


# ------------------------------------------------------------------------------------------------ checkpoints
def save_checkpoint(out: Path, model: Any, optimizer: Any, state: dict[str, Any], d: Dist) -> None:
    """Sharded fp32 parameters and AdamW state (torch.distributed.checkpoint), then state.json, then the swap."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    tmp, final, old = out / "ckpt.tmp", out / "ckpt", out / "ckpt.old"
    if d.rank == 0 and tmp.exists():
        shutil.rmtree(tmp)
    d.barrier()
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optim": optim_state}, checkpoint_id=str(tmp))
    d.barrier()
    if d.rank == 0:
        atomic_json(tmp / "state.json", state)
        if final.exists():
            if old.exists():
                shutil.rmtree(old)
            os.replace(final, old)
        os.replace(tmp, final)
        if old.exists():
            shutil.rmtree(old)
    d.barrier()


def restore_checkpoint(out: Path, model: Any, optimizer: Any, identity: str) -> dict[str, Any] | None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    folder = next((f for f in (out / "ckpt", out / "ckpt.old") if (f / "state.json").exists()), None)
    if folder is None:
        return None
    state = read_json(folder / "state.json")
    if state["identity"] != identity:
        raise ValueError("Checkpoint belongs to another online run")
    model_state, optim_state = get_state_dict(model, optimizer)
    box = {"model": model_state, "optim": optim_state}
    dcp.load(box, checkpoint_id=str(folder))
    set_state_dict(model, optimizer, model_state_dict=box["model"], optim_state_dict=box["optim"])
    return state


def _set_aside(out: Path, step: int) -> None:
    """Records of steps after the last checkpoint were not durable: keep them beside the journal for forensics, never
    count them twice (the step is sampled again). Curve points after it were measured on weights that no longer
    exist; they stay in curve.jsonl under their attempt, and the report keeps the durable attempt's point per step."""
    stamp = time.time_ns()
    path = out / "metrics.jsonl"
    rows = recover_records(path)
    durable = [r for r in rows if r["step"] <= step]
    if len(durable) != len(rows):
        path.rename(out / f"metrics.rolled-back-{stamp}.jsonl")
        for row in durable:
            append_record(path, row)
    for journal in sorted((out / "rollouts").glob("step-*.jsonl")) if (out / "rollouts").exists() else []:
        if int(journal.stem.split("-")[1]) >= step:
            journal.rename(journal.with_name(journal.name + f".rolled-back-{stamp}"))


# ------------------------------------------------------------------------------------------------ the run
def train_full(run: Path, *, family: str, seed: int, until: str | None, root: Path, mesh: Any = None,
               device: Any = None, policy_loader: Callable[[Path, Any], Any] = load_policy,
               pad_id: int | None = None, stop_after: int | None = None) -> dict[str, Any]:
    import torch

    frozen, pool = load(run, "pool")
    _, dev = load(run, "dev")
    s = frozen["settings"]
    if family not in FAMILIES or seed not in s["seeds"]:
        raise ValueError("Unregistered family/seed")
    device = device or (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
                        else torch.device("cpu"))
    d = Dist(device)
    if d.world != s["trainer_gpus"]:
        raise ValueError(f"The run registers {s['trainer_gpus']} trainer ranks, {d.world} started")
    plan = plan_of(s, family)
    start, record = merged_start(run, frozen, seed)
    out = run / "train" / run_name(family, seed)
    identity = fingerprint({"run": frozen["identity"], "family": family, "seed": seed, "start": record["identity"],
                            "start_checksum": record["checksum"], "plan": plan,
                            "model": frozen["models"]["student"]["identity"]})
    status = None
    if d.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
            status = "changed"
        elif (out / "result.json").exists() and read_json(out / "result.json").get("finished"):
            status = "finished"
    status = d.share(status)
    if status == "changed":
        raise ValueError("Online run inputs changed; use a new run")
    if status == "finished":
        return read_json(out / "result.json")
    if mesh is None:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh(device.type, (d.world,))
    torch.manual_seed(seed)
    t_load = time.monotonic()
    model = shard(policy_loader(start, device), mesh, s["reshard_after_forward"])
    optimizer = make_optimizer(model, plan["learning_rate"])
    n_params = int(sum(p.numel() for p in model.parameters()))
    state = restore_checkpoint(out, model, optimizer, identity)
    if state is None:
        state = {"identity": identity, "step": 0, "optimizer_updates": 0, "active_seconds": 0.0, "curve_steps": [],
                 "weights_check": None}
        if stop_after is None and s["phase"] == "smoke":
            stop_after = s["planned_steps"] - 1  # the smoke stops once and resumes: restore runs on the GPUs too
    load_seconds = time.monotonic() - t_load
    if pad_id is None:
        from ..models import load_tokenizer
        pad_id = load_tokenizer(model_ready(frozen)).pad_token_id
    cuda = device.type == "cuda"
    attempt = d.share(uuid.uuid4().hex[:12] if d.rank == 0 else None)
    channel = Channel(root) if d.rank == 0 else None
    workers, group, staleness = s["rollout_gpus"], s["group_size"], plan["staleness"]
    if d.rank == 0:
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "family": family, "seed": seed, "start": str(start),
            "start_identity": record["identity"], "start_checksum": record["checksum"], "plan": plan,
            "parameters": n_params, "packages": versions()})
        append_record(out / "attempts.jsonl", {"attempt": attempt, "resumed_step": state["step"],
                                               "load_seconds": load_seconds, "world": d.world, "root": str(root),
                                               "gpus": os.environ.get("CUDA_VISIBLE_DEVICES")})
        _set_aside(out, state["step"])
    order = task_order(pool, s["salt"], seed)
    normaliser = float(s["tasks_per_step"] * group * s["loss_tokens"])
    deadline = deadline_from(until)
    clock, prior = time.monotonic(), state["active_seconds"]
    queue: dict[str, Any] = {"seq": 0, "sample": {}, "curve": {}}
    published: dict[int, list[float]] = {}
    started_wall = time.time()

    def left() -> float:
        return math.inf if deadline is None else deadline - time.monotonic()

    def post(kind: str, step: int, version: int) -> None:  # rank 0
        tasks = step_tasks(order, step, s["tasks_per_step"]) if kind == "sample" else dev
        assign = assignment(tasks, workers) if kind == "sample" else curve_assignment(tasks, workers)
        channel.post(queue["seq"], {"kind": kind, "step": step, "version": version, "assign": assign})
        queue[kind][step] = queue["seq"]
        queue["seq"] += 1

    def requests(step: int, sample: int | None) -> None:  # rank 0: the next sample first, then the curve point
        if sample is not None and sample < s["planned_steps"]:
            post("sample", sample, step)
        if step % s["curve_every"] == 0 and step not in state["curve_steps"] and step not in queue["curve"]:
            post("curve", step, step)

    def alive() -> None:  # rank 0, while waiting
        silent = channel.silent(workers, HEARTBEAT_LIMIT, since=started_wall)
        if silent:
            raise RuntimeError(f"Rollout workers {silent} stopped beating")

    def publish(version: int) -> dict[str, Any] | None:  # every rank joins the all-gathers; rank 0 writes the file
        t0 = time.monotonic()
        weights, sums = export_full(model, d.rank)
        gathered = time.monotonic() - t0
        ready = None
        if d.rank == 0:
            ready = write_weights(channel.weights(version), weights, {"version": version, "attempt": attempt}, sums)
            ready["gather_seconds"], ready["write_seconds"] = gathered, time.monotonic() - t0 - gathered
            published[version] = ready["checksum"]
        del weights
        return ready

    def curves(block: bool) -> None:  # rank 0: record finished learning-curve evaluations
        for step, seq in sorted(queue["curve"].items()):
            get = lambda seq=seq: channel.results(seq, workers)
            parts = wait_for(get, timeout=max(60.0, min(left(), 60.0 * s["wait_minutes"])),
                             what=f"curve at step {step}", channel=channel, check=alive) if block else get()
            if parts is None:
                continue
            rows = [r for p in parts for r in p["rows"]]
            path = out / "curve" / f"step-{step:04d}.jsonl"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                path.rename(path.with_name(path.name + f".interrupted-{time.time_ns()}"))
            for row in rows:
                append_record(path, row)
            statuses = Counter(r["status"] for r in rows)
            append_record(out / "curve.jsonl", {
                "step": step, "attempt": attempt, "version": parts[0]["version"], "tasks": len(dev),
                "episodes": len(rows), "complete": all(p["summary"]["complete"] for p in parts),
                "correct": statuses.get("correct", 0), "statuses": dict(statuses),
                "wall_seconds": max(p["seconds"] for p in parts), "episodes_sha256": digest(path)})
            state["curve_steps"].append(step)
            del queue["curve"][step]
            channel.drop_results(seq)

    publish(state["step"])
    if d.rank == 0:
        requests(state["step"], state["step"])
        if staleness:
            requests(state["step"], state["step"] + 1)
    marks = milestones(out) if d.rank == 0 else {}
    recent: list[float] = []
    stopped_by = "planned"
    while state["step"] < s["planned_steps"]:
        index = state["step"]
        go = None
        if d.rank == 0:
            estimate = 1.3 * max(recent[-3:]) if recent else 600.0
            go = "stop_after" if stop_after is not None and index >= stop_after else \
                "deadline" if left() < estimate + 120 else ""
        go = d.share(go)
        if go:
            stopped_by = go
            break
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        payloads, info = None, None
        if d.rank == 0:
            seq = queue["sample"].pop(index)
            parts = wait_for(lambda: channel.results(seq, workers), timeout=60.0 * s["wait_minutes"],
                             what=f"rollouts of step {index}", channel=channel, check=alive)
            t_wait = time.monotonic() - t0
            for part in parts:
                if not same_checksum(part["checksum"], published.get(part["version"], [math.nan, math.nan])):
                    raise RuntimeError(f"Worker {part['worker']} served step {index} with weights that are not "
                                       f"version {part['version']}")
            rows = [r for p in parts for r in p["rows"]]
            examples = [e for p in parts for e in p["examples"]]
            rewards = [1.0 if r["correct"] else 0.0 for r in rows]
            valid = [r["valid"] for r in rows]
            weights = advantages(rewards, group, "rloo")
            for g in range(0, len(rows), group):  # a group with an unscorable reference teaches nothing
                if not all(valid[g:g + group]):
                    weights[g:g + group] = [0.0] * group
            items, usable = [], []
            for row, example, reward, weight in zip(rows, examples, rewards, weights):
                row["reward"], row["weight"] = reward, weight
                row["trained"] = bool(weight and example and example["usable"])
                row["unusable"] = example["reason"] if example and not example["usable"] else None
                if example and example["usable"]:
                    usable.append({**example, "weight": 0.0, "key": row["key"]})
                if row["trained"]:
                    items.append({**example, "weight": weight, "key": row["key"]})
            check = index == 0 and state["weights_check"] is None
            train_shards = rank_shards(items, d.world, s["micro_tokens"], pad_id)
            check_shards = rank_shards(usable[:WEIGHTS_CHECK_TRAJECTORIES], d.world, s["micro_tokens"], pad_id) \
                if check else [None] * d.world
            payloads = [{"train": train_shards[r], "check": check_shards[r], "items": len(items)}
                        for r in range(d.world)]
            info = {"seq": seq, "rows": rows, "parts": parts, "rewards": rewards, "valid": valid, "t_wait": t_wait,
                    "items": items, "used": sorted({p["version"] for p in parts})}
        mine = d.scatter(payloads)
        model.train()
        check = None
        if mine["check"] is not None:
            totals = reduce_stats(d, backward_rank(model, mine["check"], pad_id=pad_id, normaliser=normaliser,
                                                   cap=plan["tis_cap"], scale=d.world, grad=False))
            gap = totals["abs_gap_sum"] / max(1.0, totals["known_targets"])
            check = {"targets": int(totals["known_targets"]), "mean_abs_gap": gap, "max_abs_gap": totals["max_abs_gap"],
                     "limit": s["gap_limit"], "passed": totals["known_targets"] > 0 and gap < s["gap_limit"]}
            state["weights_check"] = check
            if not check["passed"]:
                raise RuntimeError(f"vLLM and the trainer disagree on the start weights: {check}")
        trained = mine["items"] > 0
        optimizer.zero_grad(set_to_none=True)
        t1 = time.monotonic()
        stats = backward_rank(model, mine["train"], pad_id=pad_id, normaliser=normaliser, cap=plan["tis_cap"],
                              scale=d.world) if trained else {}
        totals = reduce_stats(d, stats)
        t_train = time.monotonic() - t1
        lr = plan["learning_rate"] * min(1.0, (state["optimizer_updates"] + 1) / plan["warmup_steps"])
        norm = 0.0
        t2 = time.monotonic()
        if trained:
            norm = clip_and_norm(model, plan["max_grad_norm"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            state["optimizer_updates"] += 1
        optimizer.zero_grad(set_to_none=True)
        t_optim = time.monotonic() - t2
        state["step"] += 1
        step = state["step"]
        ready = publish(step)
        peak = d.reduce([float(torch.cuda.max_memory_allocated()) if cuda else 0.0], "max")[0]
        if cuda:
            torch.cuda.empty_cache()  # micro-batch shapes change every step; hand cached blocks back
        if d.rank == 0:
            channel.prune(keep_from=step - 1)
            requests(step, step + 1 if staleness else step)
            t3 = time.monotonic()
            if step % s["save_steps"] == 0 or step == s["planned_steps"]:
                marks[step] = model_dir(out / "milestones" / f"step-{step:04d}", channel.weights(step) / WEIGHTS, start)
                atomic_json(out / "milestones.json", {str(k): v for k, v in sorted(marks.items())})
            t_milestone = time.monotonic() - t3
            journal = out / "rollouts" / f"step-{index:04d}.jsonl"
            journal.parent.mkdir(exist_ok=True)
            for row in info["rows"]:
                append_record(journal, row)
            if index < BENCH_STEPS and info["items"]:
                (out / "bench-items").mkdir(exist_ok=True)
                atomic_json(out / "bench-items" / f"step-{index:04d}.json", {"step": index, "items": info["items"]})
            parts, rows, rewards = info["parts"], info["rows"], info["rewards"]
            targets = max(1.0, totals["targets"])
            gen: Counter = Counter()
            for p in parts:
                gen.update({k: v for k, v in p["generator"].items() if k in ("completion_tokens", "length_stops",
                                                                             "context_rejections")})
            state["active_seconds"] = prior + time.monotonic() - clock
            append_record(out / "metrics.jsonl", {
                "step": step, "attempt": attempt, "tasks": [r["task_id"] for r in rows[::group]],
                "versions_used": info["used"], "staleness": index - min(info["used"]),
                "reward_mean": sum(rewards) / len(rewards), **group_counts(rewards, info["valid"], group),
                "statuses": dict(Counter(r["status"] for r in rows)), "trained_trajectories": len(info["items"]),
                "unusable": dict(Counter(r["unusable"] for r in rows if r["unusable"])),
                "mean_turns": sum(len(r["steps"]) for r in rows) / len(rows),
                "mean_completion_tokens": gen["completion_tokens"] / len(rows),
                "length_stops": gen["length_stops"], "context_rejections": gen["context_rejections"],
                "optimizer_updated": trained, "optimizer_updates": state["optimizer_updates"],
                "lr": lr if trained else 0.0, "grad_norm": norm, "loss": totals["loss"],
                "targets": int(totals["targets"]), "train_tokens": int(totals["tokens"]),
                "micro_batches": int(totals["micro_batches"]),
                "mean_abs_logp_gap": totals["abs_gap_sum"] / max(1.0, totals["known_targets"]),
                "max_abs_logp_gap": totals["max_abs_gap"], "tis_clipped_fraction": totals["tis_clipped"] / targets,
                "entropy": totals["entropy_sum"] / targets, "mean_logp": totals["logp_sum"] / targets,
                "weights_check": check, "weights_verified": True,
                # the system: where the step's wall time went
                "wait_seconds": info["t_wait"], "train_seconds": t_train, "optimizer_seconds": t_optim,
                "gather_seconds": ready["gather_seconds"], "write_seconds": ready["write_seconds"],
                "milestone_seconds": t_milestone, "step_seconds": time.monotonic() - t0,
                "rollout_seconds": [p["seconds"] for p in parts], "load_seconds": [p["load_seconds"] for p in parts],
                "rollout_window": [[p["started"], p["finished"]] for p in parts],
                "train_tokens_per_second": totals["tokens"] / t_train if t_train > 0 else 0.0,
                "mfu": 6.0 * n_params * totals["tokens"] / t_train / (d.world * H20_BF16_DENSE) if t_train > 0 else 0.0,
                "peak_trainer_gpu_bytes": peak, "active_seconds": state["active_seconds"]})
            channel.drop_results(info["seq"])
            curves(block=False)
        if step % s["checkpoint_every"] == 0 and step < s["planned_steps"]:
            t4 = time.monotonic()
            save_checkpoint(out, model, optimizer, d.share(state), d)
            if d.rank == 0:
                append_record(out / "checkpoints.jsonl", {"step": step, "attempt": attempt,
                                                          "seconds": time.monotonic() - t4})
        recent.append(time.monotonic() - t0)
        gc.collect()
    finished = stopped_by == "planned"
    if d.rank == 0:
        try:
            curves(block=True)
        except (TimeoutError, RuntimeError) as exc:  # a missing curve point is reported, never fatal to the run
            append_record(out / "curve.jsonl", {"step": None, "attempt": attempt, "skipped": str(exc)[:200]})
        channel.post(queue["seq"], {"kind": "stop", "step": state["step"], "version": state["step"], "assign": {}})
        state["active_seconds"] = prior + time.monotonic() - clock
    state = d.share(state)
    if not finished:  # resumable in a later window
        save_checkpoint(out, model, optimizer, state, d)
    result = None
    if d.rank == 0:
        marks = milestones(out)
        if s["phase"] == "smoke" and finished and not state["optimizer_updates"]:
            raise RuntimeError("Smoke run never updated the policy; the gradient path is unexercised")
        result = {"finished": finished, "identity": identity, "family": family, "seed": seed, "method": plan["method"],
                  "staleness": staleness, "stopped_by": stopped_by, "steps_completed": state["step"],
                  "planned_steps": s["planned_steps"], "optimizer_updates": state["optimizer_updates"],
                  "milestones": {str(k): v for k, v in marks.items()}, "last_milestone": max(marks) if marks else None,
                  "weights_check": state["weights_check"], "curve_steps": state["curve_steps"],
                  "active_seconds": state["active_seconds"], "attempt": attempt, "parameters": n_params,
                  "world": d.world}
        atomic_json(out / ("result.json" if finished else "stopped.json"), result)
        if finished:  # 48 GB of fp32 state that nothing reads any more; the final weights are the last milestone
            for folder in (out / "ckpt", out / "ckpt.old", out / "ckpt.tmp"):
                shutil.rmtree(folder, ignore_errors=True)
    return d.share(result)


def main() -> int:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    ap = argparse.ArgumentParser(prog="exec_feedback.v10.trainer")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--until", default=None)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    wait = read_json(args.run / "manifest.json")["settings"]["wait_minutes"]
    # rank 0 waits on the rollout workers while the other ranks sit in a collective: the timeout covers that wait
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=wait + 20), device_id=torch.device("cuda", local))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    try:
        result = train_full(args.run, family=args.family, seed=args.seed, until=args.until, root=args.root, mesh=mesh,
                            device=torch.device("cuda", local))
        if dist.get_rank() == 0:
            print({k: result.get(k) for k in ("finished", "stopped_by", "steps_completed", "optimizer_updates",
                                              "last_milestone", "weights_check")}, flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except BaseException:  # noqa: BLE001 - torchrun sees the exit code and stops the other ranks
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    from ..v9.shutdown import hard_exit  # vLLM-era processes can hang at interpreter exit (see v9/shutdown.py)

    hard_exit(main())
