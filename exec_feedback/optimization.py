"""Token-normalized PyTorch optimization with restartable adapter checkpoints.

This engine is shared by real PEFT models and the CPU integration tests; there
is no dummy optimizer or fabricated loss path.
"""
from __future__ import annotations

import contextlib
import inspect
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .core import digest
from .storage import append_record, atomic_json, fingerprint, read_json, recover_records


def batch_tensors(features: list[dict[str, list[int]]], pad_id: int, device: Any) -> dict[str, Any]:
    import torch
    width = max(len(f["input_ids"]) for f in features)
    return {key: torch.tensor([f[key] + [pad] * (width - len(f[key])) for f in features], device=device, dtype=torch.long)
            for key, pad in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))}


def completion_loss(model: Any, batch: dict[str, Any]) -> tuple[Any, int]:
    import torch.nn.functional as F
    import torch
    labels = batch["labels"]
    params = inspect.signature((model.get_base_model() if hasattr(model, "get_base_model") else model).forward).parameters
    kwargs = {}
    if "logits_to_keep" in params:
        active_columns = (labels != -100).nonzero()[:, 1]
        keep = labels.shape[1] - max(0, int(active_columns.min()) - 1)
        kwargs["logits_to_keep"] = keep
    result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, **kwargs)
    logits = result.logits[:, :-1, :]
    targets = labels[:, -(logits.shape[1]):]
    mask = targets != -100
    count = int(mask.sum())
    if count == 0:
        raise ValueError("No supervised next-token targets in batch")
    # Only selected target logits are promoted to fp32; context logits are not supervised.
    loss = F.cross_entropy(logits[mask].float(), targets[mask], reduction="sum")
    return loss, count


def train_engine(model: Any, examples: list[dict[str, list[int]]], settings: dict[str, Any],
                 out: Path, *, pad_id: int, seed: int, identity: str,
                 stop_after_steps: int | None = None) -> dict[str, Any]:
    import torch
    if not examples:
        raise ValueError("No tokenized training examples")
    for row in examples:
        if not (len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])):
            raise ValueError("Training tensor lengths disagree")
        if row["labels"][0] != -100 or not any(x != -100 for x in row["labels"][1:]):
            raise ValueError("Training examples must mask the prefix and supervise next tokens")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    device = trainable[0][1].device
    cuda = device.type == "cuda"
    bf16 = cuda and torch.cuda.is_bf16_supported()
    optimizer = torch.optim.AdamW([p for _,p in trainable], lr=settings["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and not bf16)
    out.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"identity": identity, "settings": settings, "seed": seed,
                             "examples": examples, "trainable": [(n, list(p.shape)) for n,p in trainable]})
    manifest_file = out / "engine.json"
    if manifest_file.exists() and read_json(manifest_file)["signature"] != signature:
        raise ValueError("Training inputs/config changed; cannot resume checkpoint")
    atomic_json(manifest_file, {"signature": signature, "seed": seed})
    checkpoints = sorted(p for p in out.glob("checkpoint-*") if p.is_dir() and re.fullmatch(r"checkpoint-\d{8}", p.name) and (p / "complete.json").exists())
    step, cursor, prior_seconds = 0, 0, 0.0
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    if checkpoints:
        checkpoint = checkpoints[-1]
        saved = read_json(checkpoint / "complete.json")
        if saved["signature"] != signature:
            raise ValueError("Checkpoint identity mismatch")
        if any(digest(checkpoint / name) != value for name, value in saved["files"].items()):
            raise ValueError("Checkpoint content hash mismatch")
        weights = torch.load(checkpoint / "trainable.pt", map_location="cpu", weights_only=True)
        if set(weights) != {n for n,_ in trainable}:
            raise ValueError("Checkpoint trainable names differ")
        with torch.no_grad():
            for name, parameter in trainable:
                parameter.copy_(weights[name].to(parameter.device))
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=True))
        scaler.load_state_dict(torch.load(checkpoint / "scaler.pt", weights_only=True))
        rng = torch.load(checkpoint / "rng.pt", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        if cuda:
            torch.cuda.set_rng_state_all(rng["cuda"])
        step, cursor, prior_seconds = saved["step"], saved["cursor"], saved["active_seconds"]
    records = recover_records(out / "metrics.jsonl")
    durable = [r for r in records if r["step"] <= step]
    if [r["step"] for r in durable] != list(range(1, step + 1)):
        raise ValueError("Checkpoint and durable metrics disagree")
    if len(durable) != len(records):
        # Preserve rolled-back optimizer steps for audit, but do not plot them
        # twice as if they were additional completed training steps.
        (out / "metrics.jsonl").rename(out / f"metrics.rolled-back-{time.time_ns()}.jsonl")
        for row in durable: append_record(out / "metrics.jsonl", row)
    state_file = out / "state.json"
    if state_file.exists():
        prior_seconds = max(prior_seconds, read_json(state_file).get("active_seconds", 0))
    started = time.monotonic()
    def save() -> None:
        destination = out / f"checkpoint-{step:08d}"
        if destination.exists():
            return
        temporary = out / f"checkpoint-{step:08d}.tmp-{os.getpid()}"
        temporary.mkdir(exist_ok=False)
        torch.save({n: p.detach().cpu() for n,p in trainable}, temporary / "trainable.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scaler.state_dict(), temporary / "scaler.pt")
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if cuda else []}, temporary / "rng.pt")
        atomic_json(temporary / "complete.json", {"signature": signature, "step": step, "cursor": cursor,
                    "active_seconds": prior_seconds + time.monotonic() - started,
                    "files": {p.name: digest(p) for p in temporary.glob("*.pt")}})
        os.replace(temporary, destination)
        complete = sorted(p for p in out.glob("checkpoint-*") if p.is_dir() and re.fullmatch(r"checkpoint-\d{8}", p.name) and (p / "complete.json").exists())
        for old in complete[:-2]:
            shutil.rmtree(old)
    order_cache: dict[int, list[int]] = {}
    def next_batch() -> list[dict[str, list[int]]]:
        nonlocal cursor
        batch = []
        for _ in range(settings.get("batch_size", 1)):
            epoch, offset = divmod(cursor, len(examples))
            if epoch not in order_cache:
                order = list(range(len(examples)))
                random.Random(seed + epoch).shuffle(order)
                order_cache.clear()
                order_cache[epoch] = order
            batch.append(examples[order_cache[epoch][offset]])
            cursor += 1
        return batch
    model.train()
    completed = False
    last_loss = None
    try:
        while step < settings["max_steps"]:
            if prior_seconds + time.monotonic() - started >= settings["max_seconds"]:
                break
            if stop_after_steps is not None and step >= stop_after_steps:
                break
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            micro = [next_batch() for _ in range(settings["gradient_accumulation_steps"])]
            expected_tokens = sum(sum(x != -100 for x in row["labels"][1:]) for group in micro for row in group)
            loss_sum, input_tokens = 0.0, 0
            for group in micro:
                batch = batch_tensors(group, pad_id, device)
                ctx = torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16) if cuda else contextlib.nullcontext()
                with ctx:
                    loss, count = completion_loss(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss / expected_tokens).backward()
                loss_sum += float(loss.detach())
                input_tokens += int(batch["attention_mask"].sum())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for _,p in trainable], max_norm=1.0, error_if_nonfinite=True)
            warmup = int(settings["max_steps"] * settings.get("warmup_ratio", 0.05))
            scale = (step + 1) / warmup if warmup and step < warmup else (settings["max_steps"] - step) / max(1, settings["max_steps"] - warmup)
            for group in optimizer.param_groups:
                group["lr"] = settings["learning_rate"] * scale
            scaler.step(optimizer)
            scaler.update()
            step += 1
            last_loss = loss_sum / expected_tokens
            append_record(out / "metrics.jsonl", {"step": step, "loss": last_loss,
                          "supervised_tokens": expected_tokens, "input_tokens": input_tokens,
                          "grad_norm": float(norm), "lr": optimizer.param_groups[0]["lr"],
                          "step_seconds": time.perf_counter() - step_started,
                          "active_seconds": prior_seconds + time.monotonic() - started})
            if step % settings["save_steps"] == 0:
                save()
        completed = step == settings["max_steps"]
        save()
    finally:
        # Only completed optimizer steps are durable checkpoints. An interrupted
        # accumulation is restarted from the latest complete checkpoint.
        atomic_json(out / "state.json", {"signature": signature, "completed": completed, "step": step,
                    "last_loss": last_loss, "active_seconds": prior_seconds + time.monotonic() - started})
    result = {"completed": completed, "step": step, "loss": last_loss,
              "active_seconds": prior_seconds + time.monotonic() - started,
              "trainable_parameters": sum(p.numel() for _,p in trainable),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None}
    return result
