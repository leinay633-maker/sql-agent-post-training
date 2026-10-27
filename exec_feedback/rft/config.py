"""Frozen v0.5 settings; see docs/RESEARCH_V05.md for why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261010",
    "dev_databases": 6, "pool_limit": 0, "dev_limit": 0, "validation_limit": 0,
    "samples_per_task": 6, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "max_steps": 4, "max_tokens": 512, "max_context": 8192,
    "sample_batch": 32, "sample_slots": 128, "sample_seconds": 2700, "sql_workers": 16,
    "train_examples": 4096, "learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 32,
    "micro_batch": 4, "accumulation": 4, "warmup_ratio": 0.03, "max_length": 8192,
    "save_steps": 32, "train_seconds": 7200, "seeds": [42, 43, 44], "clean_seeds": [42],
    "eval_batch": 16, "eval_slots": 64, "quantization": "none",
    "pairs_per_task": 1, "pair_batch": 16, "pair_learning_rate": 0.00002, "pair_beta": 0.1,
    "pair_sft_weight": 1.0,
    "limits": {"seconds": 10.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.5 setting or protocol mismatch")
    if value["phase"] not in {"pilot", "study"}:
        raise ValueError("Expected pilot or study phase")
    if not isinstance(value["salt"], str) or not value["salt"]:
        raise ValueError("A fixed selection salt is required")
    for key in ("seeds", "clean_seeds"):
        seeds = value[key]
        if not isinstance(seeds, list) or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid {key}")
    if not value["seeds"] or not set(value["clean_seeds"]) <= set(value["seeds"]):
        raise ValueError("Register main seeds; ablation seeds must be a subset")
    positive = {"dev_databases", "samples_per_task", "max_steps", "max_tokens", "max_context",
                "sample_batch", "sample_slots", "sample_seconds", "sql_workers", "train_examples",
                "lora_r", "lora_alpha", "micro_batch", "accumulation", "max_length", "save_steps",
                "train_seconds", "eval_batch", "eval_slots", "pairs_per_task", "pair_batch"}
    for key in positive | {"pool_limit", "dev_limit", "validation_limit"}:
        if type(value[key]) is not int or value[key] < (1 if key in positive else 0):
            raise ValueError(f"Invalid integer setting: {key}")
    if not 2 <= value["samples_per_task"] <= 16 or not 1 <= value["max_steps"] <= 16:
        raise ValueError("Invalid sample count or agent step budget")
    if value["max_tokens"] >= value["max_context"] or value["max_context"] > 8192 or value["max_length"] > 8192:
        raise ValueError("Invalid token budget")
    if not value["sample_batch"] <= value["sample_slots"] <= 512 or not value["eval_batch"] <= value["eval_slots"] <= 512:
        raise ValueError("Slots must cover a generation batch")
    for key, low, high in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0), ("learning_rate", 0.0, 0.001),
                           ("warmup_ratio", -1e-12, 0.5), ("pair_learning_rate", 0.0, 0.001),
                           ("pair_beta", 0.0, 10.0), ("pair_sft_weight", -1e-12, 10.0)):
        number = value[key]
        if type(number) not in {int, float} or not math.isfinite(number) or not low < number <= high:
            raise ValueError(f"Invalid setting: {key}")
    if type(value["top_k"]) is not int or value["top_k"] < 0:
        raise ValueError("Invalid top_k")
    if value["quantization"] not in {"none", "nf4"}:
        raise ValueError("quantization must be none or nf4")
    if value["phase"] == "study" and (value["quantization"] != "none" or value["pool_limit"] or value["dev_limit"]
                                      or value["validation_limit"]):
        raise ValueError("The registered study uses the full BF16 protocol; limits are pilot-only")
    if value["phase"] == "pilot" and value["validation_limit"]:
        raise ValueError("Pilot never scores validation or test data")
    Limits(**value["limits"])


def plan(value: dict[str, Any]) -> dict[str, Any]:
    validate_settings(value)
    steps = math.ceil(value["train_examples"] / (value["micro_batch"] * value["accumulation"]))
    conditions = len(value["seeds"]) + len(value["clean_seeds"])
    return {
        "protocol": PROTOCOL, "phase": value["phase"],
        "samples_per_task": value["samples_per_task"], "sample_wall_budget_seconds": value["sample_seconds"],
        "train_examples": value["train_examples"], "optimizer_steps_per_model": steps,
        "trained_models": conditions, "evaluated_conditions": conditions + 1,
        "note": "Upper bounds and registered sizes, not measured throughput or results.",
    }
