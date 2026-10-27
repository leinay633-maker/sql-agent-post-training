"""Frozen v0.9 settings; docs/RESEARCH_V09.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261014",
    # agent protocol, unchanged since v0.5 (fenced JSON accepted as in v0.6-v0.8)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
    # greedy evaluation engines, as v0.8 (two per GPU)
    "eval_slots": 256, "eval_max_num_seqs": 256, "eval_gpu_memory": 0.40,
    # online runs: every step draws tasks_per_step pool tasks and samples group_size dialogues for each
    "seeds": [42, 43, 44], "hi_seeds": [42, 43],
    "tasks_per_step": 16, "group_size": 8, "planned_steps": 200, "save_steps": 10, "curve_every": 20,
    "temperature": 1.0,
    "learning_rate": 0.00002, "hi_learning_rate": 0.00005, "warmup_steps": 5, "max_grad_norm": 1.0,
    # loss = -sum(weight * logp) / (tasks_per_step * group_size * loss_tokens): a constant, so neither long nor
    # short trajectories are re-weighted by their own length
    "loss_tokens": 512,
    # truncated importance weight min(pi_train / pi_vllm, tis_cap) per sampled token
    "tis_cap": 2.0,
    "micro_tokens": 16384,
    "rl_gpu_memory": 0.40, "rl_slots": 128, "rl_max_num_seqs": 128,
    "lora_r": 16,
}
FLOATS = {"temperature": (0.0, 2.0), "learning_rate": (0.0, 0.001), "hi_learning_rate": (0.0, 0.001),
          "max_grad_norm": (0.0, 100.0), "tis_cap": (1.0, 100.0), "rl_gpu_memory": (0.05, 0.95),
          "eval_gpu_memory": (0.05, 0.95)}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.9 setting or protocol mismatch")
    if value["phase"] not in {"study", "smoke"}:
        raise ValueError("phase is study or smoke")
    if not isinstance(value["salt"], str) or not value["salt"]:
        raise ValueError("A fixed salt is required")
    for key in [k for k, v in DEFAULT.items() if type(v) is int]:
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"Invalid integer setting: {key}")
    for key, (low, high) in FLOATS.items():
        number = value[key]
        if type(number) not in {int, float} or not math.isfinite(number) or not low < number <= high:
            raise ValueError(f"Invalid setting: {key}")
    if not 1 <= value["max_steps"] <= 16 or not value["max_tokens"] < value["max_context"] <= 32768:
        raise ValueError("Invalid agent budget")
    if value["group_size"] < 2:
        raise ValueError("RLOO needs at least two dialogues per task")
    if value["curve_every"] % value["save_steps"]:
        raise ValueError("Curve evaluations sit on saved milestones")
    if value["rl_slots"] < value["tasks_per_step"] * value["group_size"] or value["rl_slots"] > 512:
        raise ValueError("Every dialogue of a step runs at once: rl_slots >= tasks_per_step * group_size")
    for key in ("seeds", "hi_seeds"):
        seeds = value[key]
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
                or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid {key}")
    if not set(value["hi_seeds"]) <= set(value["seeds"]):
        raise ValueError("hi_seeds start from PTR seeds that the main families also use")
    if value["phase"] == "study" and (value["max_steps"], value["max_tokens"], value["max_context"]) != (4, 512, 8192):
        raise ValueError("The registered study keeps the v0.5-v0.8 agent protocol")
    Limits(**value["limits"])
