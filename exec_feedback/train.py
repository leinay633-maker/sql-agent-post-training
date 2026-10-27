"""Explicit opt-in single-GPU QLoRA entry. Planning uses only the standard library."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import time
from pathlib import Path
from typing import Any

from .core import digest, read_jsonl
from .pipeline import code_revision, parse_action


def validate_rows(rows: list[dict[str, Any]], allow_fixture: bool = False) -> None:
    if not rows:
        raise ValueError("No training examples")
    seen = set()
    for row in rows:
        if row.get("split") != "train" or "mini-dev" in row.get("source", "").lower():
            raise ValueError("Non-training or held-out data cannot enter SFT")
        if type(row.get("fixture")) is not bool:
            raise ValueError("Explicit fixture provenance is required")
        if row["fixture"] and not allow_fixture:
            raise ValueError("Fixture training requires --allow-fixture and is not research")
        if row["task_id"] in seen:
            raise ValueError("Duplicate training task id")
        seen.add(row["task_id"])
        if not isinstance(row.get("prompt"), list) or len(row["prompt"]) < 2:
            raise ValueError("Missing prompt messages")
        for message in row["prompt"]:
            if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
                raise ValueError("Invalid training message")
        if row["prompt"][-1]["role"] != "user":
            raise ValueError("Prompt must end before the target assistant answer")
        if parse_action(row["completion"])["action"] != "finish":
            raise ValueError("Stage-1 SFT targets the corrected final action only")


def tokenize_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["prompt"] + [{"role": "assistant", "content": row["completion"]}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Chat template/token boundary mismatch; refuse an incorrect loss mask")
    if len(full) > max_length:
        raise ValueError("Overlength example; curate it explicitly, do not silently truncate")
    # All context, including earlier failed assistant actions, is masked.
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}


def collate(features: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch
    width = max(len(f["input_ids"]) for f in features)
    return {key: torch.tensor([f[key] + [pad] * (width - len(f[key])) for f in features], dtype=torch.long)
            for key, pad in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))}


def fit(data: Path, spec: dict[str, Any], settings: dict[str, Any], out: Path, *,
        seed: int = 42, allow_fixture: bool = False) -> dict[str, Any]:
    from .models import load_model, model_identity
    from .optimization import train_engine
    from .storage import atomic_json, code_fingerprint, exclusive_lock, fingerprint, read_json
    import gc
    import torch
    from peft import LoraConfig, get_peft_model

    rows = read_jsonl(data)
    validate_rows(rows, allow_fixture)
    identity = fingerprint({"data": digest(data), "model": model_identity(spec), "settings": settings,
                            "seed": seed, "code": code_fingerprint()})
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".train.lock"):
        manifest_path = out / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
            raise ValueError("Training data/model/config/code changed; use a new run")
        if (out / "result.json").exists():
            saved = read_json(out / "result.json")
            if saved.get("training_completed") and saved.get("adapter_reload_verified"):
                current = model_identity(spec, out / "adapter")["adapter_hash"]
                if current != saved["adapter_hash"]:
                    raise ValueError("Completed adapter was modified")
                return saved
        torch.manual_seed(seed)
        model, tokenizer = load_model(spec, training=True)
        tokenized = [tokenize_row(row, tokenizer, settings["max_length"]) for row in rows]
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=settings["lora_r"],
                   lora_alpha=settings["lora_alpha"], lora_dropout=0.05, target_modules="all-linear", bias="none"))
        manifest = {"identity": identity, "model": model_identity(spec), "settings": settings,
                    "seed": seed, "rows": len(rows), "data_sha256": digest(data),
                    "supervised_tokens_per_epoch": sum(sum(t != -100 for t in r["labels"]) for r in tokenized),
                    "input_tokens_per_epoch": sum(len(r["input_ids"]) for r in tokenized),
                    "origins": dict(__import__('collections').Counter(r.get("origin", "legacy") for r in rows)),
                    "fixture": any(r["fixture"] for r in rows), "cuda": torch.version.cuda,
                    "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                    **code_revision()}
        atomic_json(manifest_path, manifest)
        metrics = train_engine(model, tokenized, settings, out / "checkpoints", pad_id=tokenizer.pad_token_id,
                               seed=seed, identity=identity)
        if not metrics["completed"]:
            atomic_json(out / "result.json", {"training_completed": False, "metrics": metrics})
            raise RuntimeError("Training stopped at its time budget; complete checkpoints retained, no finished model claimed")
        model.eval()
        probe = torch.tensor([tokenized[0]["input_ids"][-min(128, len(tokenized[0]["input_ids"])):]], device=model.device)
        with torch.inference_mode():
            expected = model(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        model.save_pretrained(str(out / "adapter"), safe_serialization=True)
        tokenizer.save_pretrained(str(out / "adapter"))
        atomic_json(out / "adapter" / "efa-base.json", model_identity(spec))
        del model, probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # This is a fresh base model + saved adapter, not a check of file existence.
        reloaded, _ = load_model(spec, out / "adapter", training=False)
        reloaded.eval()
        probe = torch.tensor([tokenized[0]["input_ids"][-min(128, len(tokenized[0]["input_ids"])):]], device=reloaded.device)
        with torch.inference_mode():
            actual = reloaded(input_ids=probe, use_cache=False).logits[0, -1].float().cpu()
        delta = float((actual - expected).abs().max())
        tolerance = 0.05 if spec["device"] == "cuda" else 1e-5
        verified = bool(torch.allclose(actual, expected, rtol=tolerance, atol=tolerance))
        del reloaded, probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = {"training_completed": True, "fixture": manifest["fixture"], "metrics": metrics,
                  "adapter_reload_verified": verified, "reload_max_abs_delta": delta,
                  "adapter_hash": model_identity(spec, out / "adapter")["adapter_hash"], "identity": identity}
        atomic_json(out / "result.json", result)
        if not verified:
            raise RuntimeError("Saved adapter did not reproduce logits within configured precision tolerance")
        return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.data)
    validate_rows(rows, args.allow_fixture)
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pin the model to a full 40-character upstream commit")
    plan = {"mode": "plan_only", "samples": len(rows), "fixture": any(r["fixture"] for r in rows),
            "data_sha256": digest(args.data), "config_sha256": digest(args.config), "config": config,
            "weights_downloaded": False, "training_started": False,
            "note": "GPU memory/throughput unmeasured until execution. Planning does not load a model."}
    if not args.execute:
        return plan
    from .configuration import DEFAULT
    spec = {**DEFAULT["model"], "id": config["model"], "revision": config["revision"],
            "allow_download": args.allow_download}
    settings = {**DEFAULT["training"], **{k:v for k,v in config.items() if k in DEFAULT["training"]}}
    return fit(args.data, spec, settings, args.out, seed=config["seed"], allow_fixture=args.allow_fixture)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/qlora-smoke.json"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/qlora-smoke"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--allow-fixture", action="store_true")
    args = p.parse_args()
    try:
        print(json.dumps(train(args), indent=2))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        p.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
