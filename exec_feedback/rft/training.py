"""LoRA on frozen multi-turn RFT examples with the shared token-normalized engine."""
from __future__ import annotations

import gc
import importlib.metadata
import math
from pathlib import Path
from typing import Any

from ..core import digest, read_jsonl
from ..storage import atomic_json, exclusive_lock, fingerprint, read_json
from .study import load_run


def model_dir(run: Path, round_: int, variant: str, seed: int) -> Path:
    return run / f"round{round_}" / f"train_{variant}_seed{seed}"


def train_model(run: Path, *, round_: int, variant: str, seed: int) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from ..models import load_model, model_identity
    from ..optimization import train_engine

    frozen, _ = load_run(run)
    settings = frozen["settings"]
    registered = settings["clean_seeds"] if variant == "Rclean" else settings["seeds"]
    if variant not in {"R", "Rclean"} or seed not in registered or (variant == "Rclean" and round_ != 1):
        raise ValueError("Unregistered variant/seed/round")
    selection = read_json(run / f"round{round_}" / "selection.json")
    data = run / f"round{round_}" / f"sft_{variant}.jsonl"
    if selection["run"] != frozen["identity"] or digest(data) != selection["datasets"][variant]["sha256"]:
        raise ValueError("Frozen RFT dataset changed")
    rows = read_jsonl(data)
    if any(r["split"] != "train" or r["fixture"] or r["source"] != "bird-train" for r in rows):
        raise ValueError("Only audited BIRD train trajectories may be trained on")
    examples = [{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]), "labels": r["labels"]}
                for r in rows]
    per_step = settings["micro_batch"] * settings["accumulation"]
    engine = {"max_steps": math.ceil(len(examples) / per_step), "batch_size": settings["micro_batch"],
              "gradient_accumulation_steps": settings["accumulation"], "learning_rate": settings["learning_rate"],
              "warmup_ratio": settings["warmup_ratio"], "save_steps": settings["save_steps"],
              "max_seconds": settings["train_seconds"]}
    lora = {"r": settings["lora_r"], "lora_alpha": settings["lora_alpha"], "lora_dropout": 0.05,
            "target_modules": "all-linear", "bias": "none"}
    spec = frozen["model"]
    identity = fingerprint({"run": frozen["identity"], "round": round_, "variant": variant, "seed": seed,
                            "data": digest(data), "engine": engine, "lora": lora})
    out = model_dir(run, round_, variant, seed)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".train.lock"):
        if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
            raise ValueError("Training inputs changed; use a new run")
        if (out / "result.json").exists():
            saved = read_json(out / "result.json")
            if saved.get("training_completed") and saved.get("adapter_reload_verified"):
                if model_identity(spec, out / "adapter")["adapter_hash"] != saved["adapter_hash"]:
                    raise ValueError("Completed adapter was modified")
                return saved
        torch.manual_seed(seed)
        model, tokenizer = load_model(spec, training=True)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "round": round_, "variant": variant, "seed": seed,
            "data_sha256": digest(data), "rows": len(rows), "engine": engine, "lora": lora,
            "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "input_tokens": sum(len(r["input_ids"]) for r in rows),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}})
        metrics = train_engine(model, examples, engine, out / "checkpoints", pad_id=tokenizer.pad_token_id,
                               seed=seed, identity=identity)
        if not metrics["completed"]:
            atomic_json(out / "result.json", {"training_completed": False, "metrics": metrics, "identity": identity})
            raise RuntimeError("Training stopped at its time budget; checkpoints kept, no finished model claimed")
        model.eval()
        tail = examples[0]["input_ids"][-128:]
        probe = torch.tensor([tail], device=next(model.parameters()).device)
        with torch.inference_mode():
            expected = model(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        model.save_pretrained(str(out / "adapter"), safe_serialization=True)
        tokenizer.save_pretrained(str(out / "adapter"))
        atomic_json(out / "adapter" / "efa-base.json", model_identity(spec))
        del model, probe
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, _ = load_model(spec, out / "adapter")
        reloaded.eval()
        probe = torch.tensor([tail], device=next(reloaded.parameters()).device)
        with torch.inference_mode():
            actual = reloaded(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        verified = bool(torch.allclose(actual, expected, rtol=0.05, atol=0.05))
        result = {"training_completed": True, "identity": identity, "metrics": metrics,
                  "adapter_reload_verified": verified, "reload_max_abs_delta": float((actual - expected).abs().max()),
                  "adapter_hash": model_identity(spec, out / "adapter")["adapter_hash"]}
        atomic_json(out / "result.json", result)
        if not verified:
            raise RuntimeError("Saved adapter did not reproduce logits within tolerance")
        return result
