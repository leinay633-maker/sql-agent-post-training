"""Frozen v0.6 settings. Keys shared with v0.5 keep their meaning so the v0.5 build/trainer run unchanged."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "salt": "20261012",
    # rollouts (vLLM)
    "samples_round1": 16, "samples_round2": 16, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "round2_policy_seed": 42, "sample_shards": 64, "sample_slots": 256, "sample_minutes": 80,
    # agent protocol
    "max_steps": 4, "max_tokens": 768, "max_context": 16384, "note_budget": 10000, "sql_workers": 16,
    # evaluation
    "eval_slots": 256, "vote_n": 8, "vote_temperature": 0.8, "vote_top_p": 0.95, "vote_shards": 4,
    # SFT (consumed by exec_feedback.rft.selection / exec_feedback.rft.training)
    "train_examples": 8192, "learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 32, "micro_batch": 2,
    "accumulation": 8, "warmup_ratio": 0.03, "max_length": 12288, "save_steps": 32, "train_seconds": 9000,
    "seeds": [42, 43, 44], "clean_seeds": [42],
    # Deterministic SQL budget: the instruction cap (vm_steps, ~6 s CPU) binds; the 30 s wall clock is only a
    # backstop, so heavy concurrent load cannot turn a passing query into a timeout (v0.5/v0.6-pilot defect).
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.6 setting or protocol mismatch")
    ints = [k for k, v in DEFAULT.items() if type(v) is int]
    for key in ints:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"Invalid integer setting: {key}")
    if not 2 <= value["samples_round1"] <= 64 or not 0 <= value["samples_round2"] <= 64:
        raise ValueError("Invalid sample counts")
    if not value["max_tokens"] < value["max_context"] <= 32768 or value["max_length"] > value["max_context"]:
        raise ValueError("Invalid token budget")
    if not value["note_budget"] + value["max_tokens"] < value["max_context"]:
        raise ValueError("Column-note budget leaves no room for dialogue")
    for key in ("seeds", "clean_seeds"):
        seeds = value[key]
        if not isinstance(seeds, list) or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid {key}")
    if not value["seeds"] or not set(value["clean_seeds"]) <= set(value["seeds"]):
        raise ValueError("Register main seeds; ablation seeds must be a subset")
    if value["round2_policy_seed"] not in value["seeds"]:
        raise ValueError("Round-2 policy must be a registered seed")
    for key, low, high in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0), ("vote_temperature", 0.0, 2.0),
                           ("vote_top_p", 0.0, 1.0), ("learning_rate", 0.0, 0.001), ("warmup_ratio", -1e-12, 0.5)):
        number = value[key]
        if type(number) not in {int, float} or not math.isfinite(number) or not low < number <= high:
            raise ValueError(f"Invalid setting: {key}")
    Limits(**value["limits"])
