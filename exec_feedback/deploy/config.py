"""Frozen deployment-study settings; docs/RESEARCH_DEPLOY.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261029",
    # agent protocol of v0.5-v0.12 (checked against v0.11's frozen settings)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
    # greedy evaluations, one engine on the whole GPU at a time
    "roles": ["test", "validation"], "eval_slots": 256, "eval_max_num_seqs": 256, "eval_gpu_memory": 0.85,
    "lora_r": 16,
    # load test: bench_tasks fixed test tasks at each concurrency, against vLLM's OpenAI-compatible server; its token
    # budget per step is the offline engine's default on a >= 70 GB GPU in vLLM 0.11.2 (the server's own is 8192)
    "bench_tasks": 256, "concurrency": [1, 4, 16, 64], "warmup_tasks": 8, "serve_gpu_memory": 0.85,
    "serve_max_num_seqs": 256, "serve_batched_tokens": 16384, "request_timeout": 300, "vllm_port": 8000,
    "service_port": 8090,
    # merge-loss and FP8-loss intervals: v0.6's bootstrap over databases (one model, so no seed resampling)
    "bootstrap_draws": 2000, "bootstrap_seed": 20261012,
}
FLOATS = {"eval_gpu_memory": (0.05, 0.95), "serve_gpu_memory": (0.05, 0.95)}
SAME_AS_V11 = ("max_steps", "max_tokens", "max_context", "limits")
ROLES = ("dev", "validation", "test")


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing deployment setting or protocol mismatch")
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
    if value["eval_slots"] > 512 or value["sql_workers"] > 64 or value["request_timeout"] > 300:
        raise ValueError("At most 512 slots, 64 SQL workers and a 300 s request timeout")
    roles = value["roles"]
    if not (isinstance(roles, list) and "test" in roles and set(roles) <= set(ROLES) and len(set(roles)) == len(roles)):
        raise ValueError("roles are evaluation roles and include test (the load test draws from it)")
    levels = value["concurrency"]
    if not (isinstance(levels, list) and levels and all(type(c) is int and 1 <= c <= 256 for c in levels)
            and all(a < b for a, b in zip(levels, levels[1:])) and levels[-1] <= value["serve_max_num_seqs"]):
        raise ValueError("concurrency rises within 1..serve_max_num_seqs")
    if value["vllm_port"] == value["service_port"] or not all(1024 <= value[p] <= 65535 for p in ("vllm_port", "service_port")):
        raise ValueError("Two different unprivileged ports")
    if value["phase"] == "study" and (value["max_steps"], value["max_tokens"], value["max_context"]) != (4, 512, 8192):
        raise ValueError("The registered study keeps the v0.5-v0.12 agent protocol")
    Limits(**value["limits"])
