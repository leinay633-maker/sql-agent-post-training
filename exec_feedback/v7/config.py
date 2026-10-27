"""Frozen v0.7 settings; docs/RESEARCH_V07.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

KINDS = ("student", "scale", "teacher")
DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261012",
    # rollouts (vLLM); K per pool task matches v0.5's ~2.04 self samples per task
    "samples_teacher": 2, "samples_scale": 2, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "sample_shards": 64, "teacher_minutes": 90, "scale_minutes": 45,
    # agent protocol (v0.5 state machine; fenced JSON accepted in every condition)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "slots": {"student": 256, "scale": 192, "teacher": 96},
    "max_num_seqs": {"student": 256, "scale": 128, "teacher": 64},
    "gpu_memory": {"student": 0.40, "scale": 0.88, "teacher": 0.92},
    "eval_shards": {"student": 1, "scale": 2, "teacher": 4},
    "vote_n": 8, "vote_temperature": 0.8, "vote_top_p": 0.95, "vote_shards": 4,
    # data: N = number of v0.5 R examples (verified against the import); TR keeps every gap example up to the cap
    "train_examples": 3435, "tr_max_examples": 8192, "pairs_limit": 0,
    # SFT, identical to v0.5 R; the 14B uses micro 2 x accumulation 8 (same 16 examples per step)
    "learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 32, "warmup_ratio": 0.03, "max_length": 8192,
    "save_steps": 32, "micro_batch": {"student": 4, "scale": 2}, "accumulation": {"student": 4, "scale": 8},
    "train_seconds": {"student": 7200, "scale": 14400},
    "seeds": [42, 43, 44], "raw_seeds": [42], "scale_seeds": [42, 43, 44],
    # second stage on the selected distilled family, identical to v0.5 P
    "pair_batch": 16, "pair_learning_rate": 0.00002, "pair_beta": 0.1, "pair_sft_weight": 1.0,
    # instruction cap binds (~6 s CPU); the wall clock is only a backstop against host load
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def _per_kind(value: dict[str, Any], key: str, kinds: tuple[str, ...], low: float, high: float, integer: bool) -> None:
    table = value[key]
    if not isinstance(table, dict) or set(table) != set(kinds):
        raise ValueError(f"{key} must give exactly {kinds}")
    for kind, number in table.items():
        if integer and type(number) is not int or type(number) not in {int, float}:
            raise ValueError(f"Invalid {key}.{kind}")
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"Invalid {key}.{kind}")


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.7 setting or protocol mismatch")
    if value["phase"] not in {"study", "smoke"}:
        raise ValueError("phase is study or smoke")
    if not isinstance(value["salt"], str) or not value["salt"]:
        raise ValueError("A fixed salt is required")
    ints = [k for k, v in DEFAULT.items() if type(v) is int]
    for key in ints:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"Invalid integer setting: {key}")
    if not 1 <= value["samples_teacher"] <= 16 or not 1 <= value["samples_scale"] <= 16:
        raise ValueError("Invalid sample counts")
    if not 1 <= value["max_steps"] <= 16 or not 1 <= value["sample_shards"] <= 256:
        raise ValueError("Invalid step budget or shard count")
    if not value["max_tokens"] < value["max_context"] <= 32768 or value["max_length"] > value["max_context"]:
        raise ValueError("Invalid token budget")
    if not 1 <= value["train_examples"] <= value["tr_max_examples"]:
        raise ValueError("Invalid example counts")
    if value["phase"] == "study" and (value["train_examples"] != 3435 or value["pairs_limit"]):
        raise ValueError("The registered study uses N = 3435 (v0.5 R) and every v0.5 pair")
    for key, integer, low, high in (("slots", True, 1, 512), ("max_num_seqs", True, 1, 1024),
                                    ("gpu_memory", False, 0.05, 0.95), ("eval_shards", True, 1, 16)):
        _per_kind(value, key, KINDS, low, high, integer)
    for key, low, high in (("micro_batch", 1, 64), ("accumulation", 1, 64), ("train_seconds", 60, 86400)):
        _per_kind(value, key, ("student", "scale"), low, high, True)
    per_step = {k: value["micro_batch"][k] * value["accumulation"][k] for k in ("student", "scale")}
    if len(set(per_step.values())) != 1:
        raise ValueError("Student and 14B must see the same examples per optimizer step")
    for key in ("seeds", "raw_seeds", "scale_seeds"):
        seeds = value[key]
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
                or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid {key}")
    if not set(value["raw_seeds"]) <= set(value["seeds"]) or value["scale_seeds"][0] != value["seeds"][0]:
        raise ValueError("Ablation seeds must be registered main seeds; the 14B arm starts with the first seed")
    if not 1 <= value["vote_n"] <= 32 or not 1 <= value["vote_shards"] <= 16:
        raise ValueError("Invalid vote settings")
    for key, low, high in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0), ("vote_temperature", 0.0, 2.0),
                           ("vote_top_p", 0.0, 1.0), ("learning_rate", 0.0, 0.001), ("warmup_ratio", -1e-12, 0.5),
                           ("pair_learning_rate", 0.0, 0.001), ("pair_beta", 0.0, 10.0),
                           ("pair_sft_weight", -1e-12, 10.0)):
        number = value[key]
        if type(number) not in {int, float} or not math.isfinite(number) or not low < number <= high:
            raise ValueError(f"Invalid setting: {key}")
    Limits(**value["limits"])
