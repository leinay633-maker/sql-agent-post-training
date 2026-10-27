"""v0.7 LoRA training: the v0.5 SFT trainer and preference stage, unchanged, on v0.7's frozen datasets."""
from __future__ import annotations

import gc
import importlib.metadata
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import digest, read_jsonl
from ..storage import append_record, atomic_json, exclusive_lock, fingerprint, read_json
from .build import dataset
from .freeze import imported, load, model_ready
from .rollouts import trained_adapter

SFT_FAMILIES = {"T": "student", "TR": "student", "Traw": "student", "R14": "scale"}
PREFERENCE_FAMILIES = {"PT": "T", "PTR": "TR"}


def registered(settings: dict[str, Any], family: str) -> list[int]:
    if family == "Traw":
        return settings["raw_seeds"]
    if family == "R14":
        return settings["scale_seeds"]
    return settings["seeds"]


def _packages() -> dict[str, str]:
    return {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}


def _save_adapter(model: Any, tokenizer: Any, spec: dict[str, Any], out: Path, probe_ids: list[int]) -> tuple[list[int], Any]:
    """Save the adapter and return the probe and its last-position logits, to compare after a reload from disk."""
    import torch
    from ..models import model_identity

    model.eval()
    tail = probe_ids[-128:]
    probe = torch.tensor([tail], device=next(model.parameters()).device)
    with torch.inference_mode():
        expected = model(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
    model.save_pretrained(str(out / "adapter"), safe_serialization=True)
    tokenizer.save_pretrained(str(out / "adapter"))
    atomic_json(out / "adapter" / "efa-base.json", model_identity(spec))
    return tail, expected


def _reload_check(spec: dict[str, Any], out: Path, tail: list[int], expected: Any) -> dict[str, Any]:
    """The v0.5 acceptance check; the caller has already released the training model."""
    import torch
    from ..models import load_model, model_identity

    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_model(spec, out / "adapter")
    reloaded.eval()
    probe = torch.tensor([tail], device=next(reloaded.parameters()).device)
    with torch.inference_mode():
        actual = reloaded(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
    return {"adapter_reload_verified": bool(torch.allclose(actual, expected, rtol=0.05, atol=0.05)),
            "reload_max_abs_delta": float((actual - expected).abs().max()),
            "adapter_hash": model_identity(spec, out / "adapter")["adapter_hash"]}


def train_sft(run: Path, *, family: str, seed: int) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from ..models import load_model
    from ..optimization import train_engine

    frozen, _ = load(run)
    s = frozen["settings"]
    if family not in SFT_FAMILIES or seed not in registered(s, family):
        raise ValueError("Unregistered family/seed")
    kind = SFT_FAMILIES[family]
    spec = model_ready(frozen, kind)
    rows = dataset(run, frozen, family)
    data = run / "data" / f"sft_{family}.jsonl"
    examples = [{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]), "labels": r["labels"]}
                for r in rows]
    per_step = s["micro_batch"][kind] * s["accumulation"][kind]
    engine = {"max_steps": math.ceil(len(examples) / per_step), "batch_size": s["micro_batch"][kind],
              "gradient_accumulation_steps": s["accumulation"][kind], "learning_rate": s["learning_rate"],
              "warmup_ratio": s["warmup_ratio"], "save_steps": s["save_steps"], "max_seconds": s["train_seconds"][kind]}
    lora = {"r": s["lora_r"], "lora_alpha": s["lora_alpha"], "lora_dropout": 0.05, "target_modules": "all-linear",
            "bias": "none"}
    identity = fingerprint({"run": frozen["identity"], "family": family, "seed": seed, "data": digest(data),
                            "engine": engine, "lora": lora, "model": frozen["models"][kind]["identity"]})
    out = run / "train" / f"{family}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".train.lock"):
        if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
            raise ValueError("Training inputs changed; use a new run")
        if (out / "result.json").exists():
            saved = read_json(out / "result.json")
            if saved.get("training_completed") and saved.get("adapter_reload_verified"):
                return saved
        torch.manual_seed(seed)
        model, tokenizer = load_model(spec, training=True)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "family": family, "seed": seed, "kind": kind,
            "data_sha256": digest(data), "rows": len(rows), "origins": dict(Counter(r.get("origin") for r in rows)),
            "engine": engine, "lora": lora, "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "input_tokens": sum(len(r["input_ids"]) for r in rows), "packages": _packages()})
        metrics = train_engine(model, examples, engine, out / "checkpoints", pad_id=tokenizer.pad_token_id,
                               seed=seed, identity=identity)
        if not metrics["completed"]:
            atomic_json(out / "result.json", {"training_completed": False, "metrics": metrics, "identity": identity})
            raise RuntimeError("Training stopped at its time budget; checkpoints kept, no finished model claimed")
        tail, expected = _save_adapter(model, tokenizer, spec, out, examples[0]["input_ids"])
        del model
        check = _reload_check(spec, out, tail, expected)
        result = {"training_completed": True, "identity": identity, "metrics": metrics, **check}
        atomic_json(out / "result.json", result)
        if not check["adapter_reload_verified"]:
            raise RuntimeError("Saved adapter did not reproduce logits within tolerance")
        return result


def train_preference(run: Path, *, family: str, seed: int) -> dict[str, Any]:
    """v0.5 P on top of the selected distilled family X: same 547 pairs, plan and loss; reference = untouched X."""
    import torch
    from ..models import load_model
    from ..rft.preference import pair_loss, sequence_logprob

    frozen, _ = load(run)
    s = frozen["settings"]
    if family not in PREFERENCE_FAMILIES or seed not in s["seeds"]:
        raise ValueError("Unregistered preference family/seed")
    choice = read_json(run / "select_x.json")
    if choice.get("run") != frozen["identity"] or choice.get("x") != PREFERENCE_FAMILIES[family]:
        raise ValueError("P is only trained on the dev-selected family")
    start, start_hash = trained_adapter(run, frozen, f"{PREFERENCE_FAMILIES[family]}_seed{seed}")
    spec = model_ready(frozen, "student")
    pairs = read_jsonl(imported(run, frozen, "pairs.jsonl"))
    if s["pairs_limit"]:
        pairs = pairs[:s["pairs_limit"]]
    if any(p["split"] != "train" or p["fixture"] or p["source"] != "bird-train" for p in pairs):
        raise ValueError("Only audited BIRD train trajectories may be trained on")
    per_step = s["pair_batch"]
    plan = {"steps": math.ceil(len(pairs) / per_step), "pairs_per_step": per_step,
            "learning_rate": s["pair_learning_rate"], "beta": s["pair_beta"], "sft_weight": s["pair_sft_weight"],
            "warmup_ratio": 0.1}
    identity = fingerprint({"run": frozen["identity"], "family": family, "seed": seed, "start": start_hash,
                            "pairs": frozen["imports"]["pairs_sha256"], "pairs_used": len(pairs), "plan": plan})
    out = run / "train" / f"{family}_seed{seed}"
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
        atomic_json(out / "manifest.json", {"identity": identity, "run": frozen["identity"], "family": family,
                                            "seed": seed, "start_adapter": str(start), "pairs": len(pairs),
                                            "plan": plan, "packages": _packages()})
        for module in model.modules():  # reference and policy must score under identical numerics
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
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
                loss = pair_loss((chosen, rejected), references[index], chosen_tokens, plan["beta"],
                                 plan["sft_weight"]) / len(batch)
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
        tail, expected = _save_adapter(model, tokenizer, spec, out, pairs[0]["chosen"]["input_ids"])
        del model, optimizer, parameters
        check = _reload_check(spec, out, tail, expected)
        result = {"training_completed": True, "identity": identity, "steps": plan["steps"], "pairs": len(pairs), **check}
        atomic_json(out / "result.json", result)
        if not check["adapter_reload_verified"]:
            raise RuntimeError("Saved adapter did not reproduce logits within tolerance")
        return result
