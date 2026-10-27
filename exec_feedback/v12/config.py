"""Frozen v0.12 settings; docs/RESEARCH_V12.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261014",
    # agent protocol, unchanged since v0.5 (fenced JSON accepted as in v0.6-v0.11)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
    # greedy evaluation engines of the 4B, two per GPU as in v0.8-v0.11
    "eval_slots": 256, "eval_max_num_seqs": 256, "eval_gpu_memory": 0.40,
    "seeds": [42, 43, 44],
    # exploratory TROPD (OPD from TR's LoRA): these seeds, low priority with the teacher
    "explore_seeds": [42],
    # OPD runs in v0.9's loop: every step draws tasks_per_step pool tasks (v0.9's per-seed order) and samples
    # group_size dialogues for each; no group baseline, so 32 x 4 covers twice v0.9's tasks with its 128 dialogues
    "tasks_per_step": 32, "group_size": 4, "planned_steps": 400, "save_steps": 10, "curve_every": 20,
    "temperature": 1.0, "learning_rate": 0.0001, "warmup_steps": 5, "max_grad_norm": 1.0,
    "loss_tokens": 512, "tis_cap": 2.0, "micro_tokens": 16384,
    "rl_gpu_memory": 0.40, "rl_slots": 128, "rl_max_num_seqs": 128,
    # a fresh LoRA shaped as R/TR's; its B starts at zero, so v0.9's start-of-run check moves to this step
    "lora_r": 16, "lora_alpha": 32, "lora_check_step": 10,
    # the teacher: v0.8's FP8 235B over teacher_gpus whole GPUs. Evaluation as v0.8 ran it; scoring with prompt
    # log-probs, whose full-vocabulary rows per prefill chunk are not in vLLM's memory profile, so a smaller chunk
    # and a little more headroom, and one position more than the student's context (vLLM needs prompt < max_model_len)
    "teacher_gpus": 4,
    "teacher_eval_gpu_memory": 0.85, "teacher_eval_slots": 256, "teacher_eval_max_num_seqs": 256,
    "teacher_score_gpu_memory": 0.80, "teacher_score_max_num_seqs": 128, "teacher_score_batched_tokens": 4096,
    "teacher_score_context": 8208,
    # scoring gate: the teacher samples these many continuations and scores them again; mean |difference| below
    # this (nat per token) or nothing trains (the same bound as v0.9's LoRA check: misalignment, not kernel noise)
    "teacher_gate_samples": 32, "teacher_gate_max_gap": 0.15,
    # how long a main run waits for the teacher (down, restarting, busy) before its step fails: queueing behind the two
    # other main runs plus a restart (a reload of the 235B) fit; TROPD waits up to its own deadline instead
    "teacher_wait_seconds": 1800,
}
FLOATS = {"temperature": (0.0, 2.0), "learning_rate": (0.0, 0.001), "max_grad_norm": (0.0, 100.0),
          "tis_cap": (1.0, 100.0), "rl_gpu_memory": (0.05, 0.95), "eval_gpu_memory": (0.05, 0.95),
          "teacher_eval_gpu_memory": (0.05, 0.95), "teacher_score_gpu_memory": (0.05, 0.95),
          "teacher_gate_max_gap": (0.0, 10.0)}
# Values the registered study cannot run without (docs/RESEARCH_V12.md); smoke settings may shrink everything else.
REGISTERED = {"max_steps": 4, "max_tokens": 512, "max_context": 8192, "seeds": [42, 43, 44], "explore_seeds": [42],
              "tasks_per_step": 32,
              "group_size": 4, "planned_steps": 400, "save_steps": 10, "curve_every": 20, "temperature": 1.0,
              "learning_rate": 0.0001, "warmup_steps": 5, "max_grad_norm": 1.0, "loss_tokens": 512, "tis_cap": 2.0,
              "lora_r": 16, "lora_alpha": 32, "lora_check_step": 10, "teacher_gpus": 4, "teacher_gate_max_gap": 0.15}


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing v0.12 setting or protocol mismatch")
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
    if value["teacher_score_context"] <= value["max_context"]:
        raise ValueError("The teacher scores whole dialogues: its context must exceed the student's")
    if value["curve_every"] % value["save_steps"] or value["planned_steps"] % value["save_steps"]:
        raise ValueError("Curve evaluations and the step cap sit on saved milestones")
    if not value["lora_check_step"] < value["planned_steps"]:
        raise ValueError("The LoRA check happens before the last step")
    if value["rl_slots"] < value["tasks_per_step"] * value["group_size"] or value["rl_slots"] > 512:
        raise ValueError("Every dialogue of a step runs at once: rl_slots >= tasks_per_step * group_size")
    if value["teacher_score_batched_tokens"] < 512:
        raise ValueError("teacher_score_batched_tokens is a prefill chunk, at least 512 tokens")
    for key in ("seeds", "explore_seeds"):
        seeds = value[key]
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
                or len(set(seeds)) != len(seeds):
            raise ValueError(f"Invalid {key}")
    if not set(value["explore_seeds"]) <= set(value["seeds"]):
        raise ValueError("TROPD starts from TR adapters of registered seeds")
    if value["phase"] == "study":
        differ = [k for k, v in REGISTERED.items() if value[k] != v]
        if differ:
            raise ValueError(f"The registered study keeps docs/RESEARCH_V12.md's values; differs in {differ}")
    Limits(**value["limits"])
