"""Frozen v0.10 settings; docs/RESEARCH_V10.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261014",  # v0.9's salt: every seed sees v0.9's task stream
    # agent protocol, unchanged since v0.5 (fenced JSON accepted as in v0.6-v0.9)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
    # greedy evaluation engines, as v0.8 and v0.9 (two per GPU)
    "eval_slots": 256, "eval_max_num_seqs": 256, "eval_gpu_memory": 0.40,
    # online runs: v0.9's sampling, objective and step count (= v0.9's common milestone K); only the learning rate
    # differs (full parameters: v0.9's LoRA rate / 10)
    "seeds": [42, 43, 44],
    # a full-weight milestone is 8 GB, so every 20 steps instead of v0.9's 10
    "tasks_per_step": 16, "group_size": 8, "planned_steps": 200, "save_steps": 20, "curve_every": 20,
    "temperature": 1.0, "learning_rate": 0.000002, "warmup_steps": 5, "max_grad_norm": 1.0,
    "loss_tokens": 512, "tis_cap": 2.0, "micro_tokens": 16384,
    # the system: trainer GPUs first, rollout GPUs after them; every rollout GPU samples whole groups
    "trainer_gpus": 4, "rollout_gpus": 4, "rollout_gpu_memory": 0.85, "rollout_slots": 128,
    "rollout_max_num_seqs": 128, "checkpoint_every": 20, "reshard_after_forward": True,
    # step 0: mean |log p_vLLM - log p_trainer| over the sampled tokens must stay below this (same weights, two kernels)
    "gap_limit": 0.15,
    # a rollout result or a weight file that takes longer than this stops the run
    "wait_minutes": 30,
}
FLOATS = {"temperature": (0.0, 2.0), "learning_rate": (0.0, 0.001), "max_grad_norm": (0.0, 100.0),
          "tis_cap": (1.0, 100.0), "rollout_gpu_memory": (0.05, 0.95), "eval_gpu_memory": (0.05, 0.95),
          "gap_limit": (0.0, 10.0)}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def per_worker(settings: dict[str, Any]) -> int:
    """Pool tasks every rollout worker samples per step (whole groups only)."""
    return settings["tasks_per_step"] // settings["rollout_gpus"]


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.10 setting or protocol mismatch")
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
    if type(value["reshard_after_forward"]) is not bool:
        raise ValueError("reshard_after_forward is true (full sharding) or false (gradient/optimizer sharding)")
    if not 1 <= value["max_steps"] <= 16 or not value["max_tokens"] < value["max_context"] <= 32768:
        raise ValueError("Invalid agent budget")
    if value["group_size"] < 2:
        raise ValueError("RLOO needs at least two dialogues per task")
    if value["curve_every"] % value["save_steps"]:
        raise ValueError("Curve evaluations sit on saved milestones")
    if value["tasks_per_step"] % value["rollout_gpus"]:
        raise ValueError("Every rollout GPU samples the same number of whole groups")
    if value["rollout_slots"] < per_worker(value) * value["group_size"] or value["rollout_slots"] > 512:
        raise ValueError("A worker runs its share of a step at once: rollout_slots >= tasks per worker * group_size")
    if value["trainer_gpus"] + value["rollout_gpus"] > 8:
        raise ValueError("One 8-GPU host")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
            or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    if value["phase"] == "study":
        if (value["max_steps"], value["max_tokens"], value["max_context"]) != (4, 512, 8192):
            raise ValueError("The registered study keeps the v0.5-v0.9 agent protocol")
        if (value["tasks_per_step"], value["group_size"], value["temperature"], value["salt"]) != (16, 8, 1.0, "20261014"):
            raise ValueError("The registered study samples exactly as v0.9 did")
        if (value["trainer_gpus"], value["rollout_gpus"]) != (4, 4):
            raise ValueError("The registered study trains on 4 GPUs and samples on 4")
    Limits(**value["limits"])
