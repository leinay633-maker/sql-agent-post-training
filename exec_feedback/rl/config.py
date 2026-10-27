"""Frozen v0.4 settings. Paths are explicit CLI arguments, not hidden overrides."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT = {
    "protocol": PROTOCOL, "phase": "study", "train_tasks": 1024,
    "validation_tasks": 512, "seeds": [42, 43, 44], "selection_seed": 42,
    "updates": 128, "prompt_batch": 8, "group_size": 4,
    "generation_batch": 16, "slots": 64, "max_tokens": 384,
    "max_context": 8192, "learning_rate": 0.00001, "save_steps": 8,
    "train_seconds": 14400, "train_calls": 9000,
    "eval_seconds": 7200, "eval_calls": 1200,
    "limits": {"seconds": 10.0, "max_rows": 10000,
               "max_bytes": 8388608, "vm_steps": 50000000},
}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.4 setting or protocol mismatch")
    if value["phase"] not in {"pilot", "study"}:
        raise ValueError("Expected pilot or study phase")
    if not isinstance(value["seeds"], list) or not value["seeds"]:
        raise ValueError("Register seeds before any validation scoring")
    if any(type(s) is not int or s < 0 for s in value["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(value["seeds"])) != len(value["seeds"]):
        raise ValueError("Duplicate seeds")
    integer_keys = set(DEFAULT) - {"protocol", "phase", "seeds", "limits", "learning_rate"}
    for key in integer_keys:
        minimum = 0 if key in {"validation_tasks", "selection_seed"} else 1
        if type(value[key]) is not int or value[key] < minimum:
            raise ValueError(f"Invalid integer setting: {key}")
    if not 2 <= value["group_size"] <= 16 or not 1 <= value["generation_batch"] <= 64:
        raise ValueError("Invalid group/generation batch size")
    if not value["generation_batch"] <= value["slots"] <= 256:
        raise ValueError("Slots must cover a generation batch and be at most 256")
    if value["max_tokens"] >= value["max_context"] or value["max_context"] > 8192:
        raise ValueError("Invalid token budget")
    lr = value["learning_rate"]
    if type(lr) not in {float, int} or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError("Invalid learning rate")
    if value["phase"] == "pilot" and value["validation_tasks"] != 0:
        raise ValueError("Pilot must not register validation scoring")
    if value["phase"] == "study" and value["validation_tasks"] < 1:
        raise ValueError("Study needs a frozen evaluation denominator")
    if value["train_calls"] < value["updates"] * value["prompt_batch"] * value["group_size"] * 2:
        raise ValueError("Training call cap cannot cover the registered schedule")
    if value["eval_calls"] < value["validation_tasks"] * 2:
        raise ValueError("Evaluation call cap is below the registered denominator")
    from ..core import Limits
    Limits(**value["limits"])


def plan(value: dict[str, Any]) -> dict[str, Any]:
    validate_settings(value)
    trajectories = value["updates"] * value["prompt_batch"] * value["group_size"]
    train_calls = len(value["seeds"]) * trajectories * 2
    eval_calls = (len(value["seeds"]) + 1) * value["validation_tasks"] * 2
    return {
        "protocol": PROTOCOL, "phase": value["phase"],
        "train_candidates": value["train_tasks"],
        "prompt_draws_per_seed": value["updates"] * value["prompt_batch"],
        "trajectories_per_seed": trajectories,
        "maximum_train_calls": train_calls, "maximum_eval_calls": eval_calls,
        "maximum_generated_tokens": (train_calls + eval_calls) * value["max_tokens"],
        "maximum_model_calls": train_calls + eval_calls,
        "gpu_seconds": None, "gpu_memory_bytes": None, "quality": None,
        "note": "Counts are configured upper bounds, not measured throughput/results.",
    }
