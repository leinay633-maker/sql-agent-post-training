"""Frozen v0.11 settings; docs/RESEARCH_V11.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261014",
    # agent protocol, unchanged since v0.5 (fenced JSON accepted as in v0.6-v0.9)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
    # greedy evaluation engines and the pre-pass samplers, two per GPU as in v0.8 and v0.9
    "eval_slots": 256, "eval_max_num_seqs": 256, "eval_gpu_memory": 0.40,
    "seeds": [42, 43, 44],
    # v0.9's RL, unchanged: every step draws tasks_per_step tasks and samples group_size dialogues for each
    "tasks_per_step": 16, "group_size": 8, "planned_steps": 200, "save_steps": 10, "curve_every": 20,
    "temperature": 1.0, "learning_rate": 0.00002, "warmup_steps": 5, "max_grad_norm": 1.0,
    "loss_tokens": 512, "tis_cap": 2.0, "micro_tokens": 16384,
    "rl_gpu_memory": 0.40, "rl_slots": 128, "rl_max_num_seqs": 128, "lora_r": 16,
    # exploratory: the runs continue from the registered milestone to here after its evaluations
    "explore_steps": 400,
    # pre-pass: every pool task sampled prepass_samples times with the seed's start; boundary = solved 1..samples-1
    # times. prepass_chunk tasks are journalled together: their dialogues stream through the evaluations' slots (a
    # chunk no larger than the slots would leave the batch half empty in its last turns), a crash loses one chunk
    "prepass_samples": 8, "prepass_chunk": 256,
    # the boundary share of the scorable pool tasks outside which the pre-pass is taken for broken and nothing trains
    "boundary_gate": [0.05, 0.60],
}
FLOATS = {"temperature": (0.0, 2.0), "learning_rate": (0.0, 0.001), "max_grad_norm": (0.0, 100.0),
          "tis_cap": (1.0, 100.0), "rl_gpu_memory": (0.05, 0.95), "eval_gpu_memory": (0.05, 0.95)}
# what v0.11 must share with v0.9's RL (checked against v0.9's frozen settings at the freeze)
SAME_AS_V09_RL = ("salt", "max_steps", "max_tokens", "max_context", "limits", "tasks_per_step", "group_size",
                  "planned_steps", "save_steps", "curve_every", "temperature", "learning_rate", "warmup_steps",
                  "max_grad_norm", "loss_tokens", "tis_cap", "micro_tokens", "lora_r")


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.11 setting or protocol mismatch")
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
    if value["group_size"] < 2 or value["prepass_samples"] < 2:
        raise ValueError("RLOO and the boundary rule need at least two dialogues per task")
    if value["curve_every"] % value["save_steps"] or value["planned_steps"] % value["save_steps"] \
            or value["explore_steps"] % value["save_steps"]:
        raise ValueError("Curve evaluations and both comparison points sit on saved milestones")
    if value["explore_steps"] <= value["planned_steps"]:
        raise ValueError("The exploratory continuation goes past the registered milestone")
    if value["rl_slots"] < value["tasks_per_step"] * value["group_size"] or value["rl_slots"] > 512:
        raise ValueError("Every dialogue of a step runs at once: rl_slots >= tasks_per_step * group_size")
    gate = value["boundary_gate"]
    if not (isinstance(gate, list) and len(gate) == 2 and all(type(x) in {int, float} for x in gate)
            and 0.0 <= gate[0] < gate[1] <= 1.0):
        raise ValueError("boundary_gate is [low, high] within [0, 1]")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
            or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    if value["phase"] == "study":
        if (value["max_steps"], value["max_tokens"], value["max_context"]) != (4, 512, 8192):
            raise ValueError("The registered study keeps the v0.5-v0.9 agent protocol")
        if value["prepass_samples"] != value["group_size"]:
            raise ValueError("A boundary task is one whose group of G dialogues has a non-zero advantage")
    Limits(**value["limits"])
