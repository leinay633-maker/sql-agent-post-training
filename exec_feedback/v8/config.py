"""Frozen v0.8 settings; docs/RESEARCH_V08.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

KINDS = ("student", "teacher")
DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261013",
    # rollouts only on the pool tasks the 4B never solved in v0.5 round 1; both models' cards recommend 0.7/0.8/20
    "samples_teacher": 4, "samples_hint": 2, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "sample_shards": 64, "teacher_minutes": 75, "hint_minutes": 40,
    # gate: teacher greedy dev score - 4B greedy dev score (project scoring) >= gate_delta, bootstrap lower bound > 0
    "gate_delta": 0.05,
    # agent protocol (v0.5 state machine; fenced JSON accepted)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "slots": {"student": 256, "teacher": 256},
    "max_num_seqs": {"student": 256, "teacher": 256},
    "gpu_memory": {"student": 0.40, "teacher": 0.85},
    "sample_gpu_memory": 0.85, "teacher_gpus": 4,
    # data: N = number of v0.5 R examples; each new set adds at most gap_max_examples gap tasks
    "train_examples": 3435, "gap_max_examples": 4096, "pairs_limit": 0,
    # SFT, identical to v0.5 R (16 examples per optimizer step)
    "learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 32, "warmup_ratio": 0.03, "max_length": 8192,
    "save_steps": 32, "micro_batch": 4, "accumulation": 4, "train_seconds": 9000,
    "seeds": [42, 43, 44],
    # second stage on each new family, identical to v0.5 P
    "pair_batch": 16, "pair_learning_rate": 0.00002, "pair_beta": 0.1, "pair_sft_weight": 1.0,
    # instruction cap binds (~6 s CPU); the wall clock is only a backstop against host load
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def _per_kind(value: dict[str, Any], key: str, low: float, high: float, integer: bool) -> None:
    table = value[key]
    if not isinstance(table, dict) or set(table) != set(KINDS):
        raise ValueError(f"{key} must give exactly {KINDS}")
    for kind, number in table.items():
        if integer and type(number) is not int or type(number) not in {int, float}:
            raise ValueError(f"Invalid {key}.{kind}")
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"Invalid {key}.{kind}")


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.8 setting or protocol mismatch")
    if value["phase"] not in {"study", "smoke"}:
        raise ValueError("phase is study or smoke")
    if not isinstance(value["salt"], str) or not value["salt"]:
        raise ValueError("A fixed salt is required")
    for key in [k for k, v in DEFAULT.items() if type(v) is int]:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"Invalid integer setting: {key}")
    if not 1 <= value["samples_teacher"] <= 16 or not 1 <= value["samples_hint"] <= 16:
        raise ValueError("Invalid sample counts")
    if not 1 <= value["max_steps"] <= 16 or not 1 <= value["sample_shards"] <= 256:
        raise ValueError("Invalid step budget or shard count")
    if not value["max_tokens"] < value["max_context"] <= 32768 or value["max_length"] > value["max_context"]:
        raise ValueError("Invalid token budget")
    if not 1 <= value["train_examples"] or not 1 <= value["gap_max_examples"] <= 16384:
        raise ValueError("Invalid example counts")
    if value["phase"] == "study" and (value["train_examples"] != 3435 or value["pairs_limit"]):
        raise ValueError("The registered study uses N = 3435 (v0.5 R) and every v0.5 pair")
    if value["teacher_gpus"] not in {1, 2, 4, 8}:
        raise ValueError("teacher_gpus is a tensor-parallel width")
    for key, integer, low, high in (("slots", True, 1, 512), ("max_num_seqs", True, 1, 1024),
                                    ("gpu_memory", False, 0.05, 0.95)):
        _per_kind(value, key, low, high, integer)
    if not 1 <= value["micro_batch"] <= 64 or not 1 <= value["accumulation"] <= 64 \
            or not 60 <= value["train_seconds"] <= 86400:
        raise ValueError("Invalid training budget")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
            or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    for key, low, high in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0), ("learning_rate", 0.0, 0.001),
                           ("warmup_ratio", -1e-12, 0.5), ("pair_learning_rate", 0.0, 0.001), ("pair_beta", 0.0, 10.0),
                           ("pair_sft_weight", -1e-12, 10.0), ("gate_delta", -1e-12, 0.5),
                           ("sample_gpu_memory", 0.05, 0.95)):
        number = value[key]
        if type(number) not in {int, float} or not math.isfinite(number) or not low < number <= high:
            raise ValueError(f"Invalid setting: {key}")
    if type(value["top_k"]) is not int or value["top_k"] < 0:
        raise ValueError("Invalid top_k")
    Limits(**value["limits"])
