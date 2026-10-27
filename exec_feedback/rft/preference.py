"""Second stage: preference optimization on execution-verified correct/incorrect trajectory pairs.

Pairs come only from round-1 frontier train tasks (some samples right, some
wrong). Each R seed continues from its own frozen adapter; that adapter, before
any update, is also the reference policy.
"""
from __future__ import annotations

import gc
import importlib.metadata
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..core import digest, read_jsonl, write_jsonl
from ..storage import append_record, atomic_json, exclusive_lock, fingerprint, read_json
from .selection import round_samples, supervised_turns, tokenize_trajectory
from .study import load_run, rank
from .training import model_dir

REJECTED = {"incorrect", "execution_failed", "budget_exhausted"}


def rejected_turns(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[bool]] | None:
    """A wrong trajectory, trimmed to its last assistant turn; every assistant turn counts."""
    if row["correct"] or row["verdict"] not in REJECTED or not row["steps"]:
        return None
    messages = list(row["messages"])
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    turns = sum(m["role"] == "assistant" for m in messages)
    return (messages, [True] * turns) if turns else None


def build_pairs(run: Path) -> dict[str, Any]:
    from ..models import load_tokenizer

    frozen, pool = load_run(run, "pool")
    settings, salt = frozen["settings"], frozen["settings"]["salt"]
    out = run / "round1"
    if (out / "pairs.json").exists():
        raise ValueError("Preference pairs already frozen")
    tasks = {t.id: t for t in pool}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in round_samples(run, 1, set(tasks)):
        by_task[row["task_id"]].append(row)
    tokenizer = load_tokenizer(frozen["model"])
    pairs, skipped = [], Counter()
    for task_id in sorted(by_task, key=lambda t: rank(f"{salt}:pairs", t)):
        good = [(r, f) for r in by_task[task_id] if (f := supervised_turns(r)) is not None]
        bad = [(r, x) for r in by_task[task_id] if (x := rejected_turns(r)) is not None]
        if not good or not bad:
            continue
        good.sort(key=lambda item: rank(f"{salt}:chosen", item[0]["key"]))
        bad.sort(key=lambda item: rank(f"{salt}:rejected", item[0]["key"]))
        for (row, flags), (wrong, (messages, all_turns)) in list(zip(good, bad))[:settings["pairs_per_task"]]:
            chosen = tokenize_trajectory(row["messages"], flags, tokenizer, settings["max_length"])
            rejected = tokenize_trajectory(messages, all_turns, tokenizer, settings["max_length"])
            if chosen is None or rejected is None:
                skipped["overlong"] += 1
                continue
            task = tasks[task_id]
            pairs.append({"task_id": task_id, "db_id": task.db_id, "split": task.split, "source": task.source,
                          "fixture": task.fixture, "chosen_key": row["key"], "rejected_key": wrong["key"],
                          "rejected_verdict": wrong["verdict"], "chosen": chosen, "rejected": rejected})
    if not pairs:
        raise ValueError("No frontier tasks with both a correct and a wrong trajectory")
    write_jsonl(out / "pairs.jsonl", pairs)
    summary = {"run": frozen["identity"], "pairs": len(pairs), "tasks": len({p["task_id"] for p in pairs}),
               "rejected_verdicts": dict(Counter(p["rejected_verdict"] for p in pairs)),
               "skipped": dict(skipped), "sha256": digest(out / "pairs.jsonl")}
    summary["identity"] = fingerprint(summary)
    atomic_json(out / "pairs.json", summary)
    return summary


def sequence_logprob(model: Any, example: dict[str, list[int]]) -> Any:
    """Sum of log p over the example's supervised tokens (its loss-mask labels)."""
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    labels = example["labels"]
    first = next(i for i, x in enumerate(labels) if x != -100)
    ids = torch.tensor([example["input_ids"]], device=device)
    keep = len(labels) - first + 1
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=keep)
    logits = output.logits[0, :-1].float()
    targets = torch.tensor(labels[-(keep - 1):], device=device)
    mask = targets != -100
    picked = -functional.cross_entropy(logits[mask], targets[mask], reduction="none")
    return picked.sum(), int(mask.sum())


def pair_loss(policy: tuple[Any, Any], reference: tuple[float, float], chosen_tokens: int,
              beta: float, sft_weight: float) -> Any:
    """DPO margin plus token-mean NLL on the chosen trajectory (RPO form)."""
    import torch.nn.functional as functional

    chosen, rejected = policy
    margin = (chosen - reference[0]) - (rejected - reference[1])
    return -functional.logsigmoid(beta * margin) - sft_weight * chosen / max(1, chosen_tokens)


def train_preference(run: Path, *, seed: int) -> dict[str, Any]:
    import torch
    from ..models import load_model, model_identity
    from .evaluation import frozen_adapter

    frozen, _ = load_run(run)
    settings = frozen["settings"]
    if seed not in settings["seeds"]:
        raise ValueError("Unregistered seed")
    pairs_meta = read_json(run / "round1" / "pairs.json")
    data = run / "round1" / "pairs.jsonl"
    if pairs_meta["run"] != frozen["identity"] or digest(data) != pairs_meta["sha256"]:
        raise ValueError("Frozen preference pairs changed")
    pairs = read_jsonl(data)
    if any(p["split"] != "train" or p["fixture"] or p["source"] != "bird-train" for p in pairs):
        raise ValueError("Only audited BIRD train trajectories may be trained on")
    start = frozen_adapter(run, frozen, f"R_r1_seed{seed}")
    spec = frozen["model"]
    per_step = settings["pair_batch"]
    plan = {"steps": math.ceil(len(pairs) / per_step), "pairs_per_step": per_step,
            "learning_rate": settings["pair_learning_rate"], "beta": settings["pair_beta"],
            "sft_weight": settings["pair_sft_weight"], "warmup_ratio": 0.1}
    identity = fingerprint({"run": frozen["identity"], "seed": seed, "pairs": pairs_meta["sha256"],
                            "start": model_identity(spec, start)["adapter_hash"], "plan": plan})
    out = model_dir(run, 1, "P", seed)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".train.lock"):
        if (out / "result.json").exists():
            saved = read_json(out / "result.json")
            if saved.get("identity") != identity:
                raise ValueError("Preference inputs changed; use a new run")
            if saved.get("training_completed") and saved.get("adapter_reload_verified"):
                return saved
        torch.manual_seed(seed)
        model, tokenizer = load_model(spec, start, training=True)
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "seed": seed, "start_adapter": str(start),
            "pairs": len(pairs), "plan": plan,
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}})
        # Reference and policy must score under identical numerics: no dropout, same autocast.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        # The untouched start adapter is the reference; score it once before any update.
        model.eval()
        references = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for pair in pairs:
                references.append((float(sequence_logprob(model, pair["chosen"])[0]),
                                   float(sequence_logprob(model, pair["rejected"])[0])))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=plan["learning_rate"], weight_decay=0.0)
        order = list(range(len(pairs)))
        random.Random(seed).shuffle(order)
        warmup = max(1, int(plan["steps"] * plan["warmup_ratio"]))
        model.train()
        for step in range(plan["steps"]):
            batch = order[step * per_step:(step + 1) * per_step]
            if not batch:
                break
            scale = (step + 1) / warmup if step < warmup else (plan["steps"] - step) / max(1, plan["steps"] - warmup)
            for group in optimizer.param_groups:
                group["lr"] = plan["learning_rate"] * scale
            optimizer.zero_grad(set_to_none=True)
            totals = Counter()
            for index in batch:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    chosen, chosen_tokens = sequence_logprob(model, pairs[index]["chosen"])
                    rejected, _ = sequence_logprob(model, pairs[index]["rejected"])
                loss = pair_loss((chosen, rejected), references[index], chosen_tokens,
                                 plan["beta"], plan["sft_weight"]) / len(batch)
                loss.backward()
                margin = (float(chosen) - references[index][0]) - (float(rejected) - references[index][1])
                totals["loss"] += float(loss) * len(batch)
                totals["margin"] += margin
                totals["accuracy"] += margin > 0
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            append_record(out / "metrics.jsonl", {"step": step + 1, "pairs": len(batch), "grad_norm": float(norm),
                                                  "lr": optimizer.param_groups[0]["lr"],
                                                  **{k: v / len(batch) for k, v in totals.items()}})
        model.eval()
        tail = pairs[0]["chosen"]["input_ids"][-128:]
        probe = torch.tensor([tail], device=next(model.parameters()).device)
        with torch.inference_mode():
            expected = model(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        model.save_pretrained(str(out / "adapter"), safe_serialization=True)
        tokenizer.save_pretrained(str(out / "adapter"))
        atomic_json(out / "adapter" / "efa-base.json", model_identity(spec))
        del model, probe, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, _ = load_model(spec, out / "adapter")
        reloaded.eval()
        probe = torch.tensor([tail], device=next(reloaded.parameters()).device)
        with torch.inference_mode():
            actual = reloaded(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        verified = bool(torch.allclose(actual, expected, rtol=0.05, atol=0.05))
        result = {"training_completed": True, "identity": identity, "steps": plan["steps"], "pairs": len(pairs),
                  "adapter_reload_verified": verified, "reload_max_abs_delta": float((actual - expected).abs().max()),
                  "adapter_hash": model_identity(spec, out / "adapter")["adapter_hash"]}
        atomic_json(out / "result.json", result)
        if not verified:
            raise RuntimeError("Saved adapter did not reproduce logits within tolerance")
        return result
