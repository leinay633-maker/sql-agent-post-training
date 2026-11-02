"""Frozen pass@k settings; docs/RESEARCH_PASSK.md says why each value was chosen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import PROTOCOL

DEFAULT: dict[str, Any] = {
    "protocol": PROTOCOL, "phase": "study", "salt": "20261029",
    # agent protocol and sampling of the online runs (v0.9-v0.12; checked against v0.11's and v0.12's frozen settings)
    "max_steps": 4, "max_tokens": 512, "max_context": 8192, "sql_workers": 16,
    "limits": {"seconds": 30.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 1000000000},
    "temperature": 1.0, "top_p": 1.0,
    "seeds": [42, 43, 44],
    # dialogues per task and model, and the k reported (the largest is the sample count)
    "samples": 64, "ks": [1, 2, 4, 8, 16, 32, 64],
    # samplers: two per GPU, each with 40% of the memory and a fixed 28 GiB KV cache (no start-up profiling, which
    # failed when the other engine on the GPU exited meanwhile); chunk_tasks x samples dialogues are journalled together
    "slots": 256, "max_num_seqs": 256, "gpu_memory": 0.40, "kv_cache_gib": 28, "lora_r": 16, "chunk_tasks": 32,
    # report: v0.6's cluster bootstrap; the lucky-match audit list
    "bootstrap_draws": 2000, "bootstrap_seed": 20261012, "audit_per_direction": 30, "audit_seed": 20261029,
    # decoding control (added before any run): B and PTR never trained on their own temperature-1 samples, so the
    # first seed's PTR and RLb sample a fixed subset again with the model card's recommended decoding, which cuts the
    # low-probability tail
    "control": {"conditions": ["PTR_seed42", "RLb_seed42"], "tasks": 500,
                "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}},
}
FLOATS = {"temperature": (0.0, 2.0), "top_p": (0.0, 1.0), "gpu_memory": (0.05, 0.95)}
# what the samplers share with the online runs (checked against v0.11's and v0.12's frozen settings at the freeze)
SAME_AS_ONLINE = ("max_steps", "max_tokens", "max_context", "limits", "temperature")


def load_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_settings(value)
    return value


def validate_settings(value: dict[str, Any]) -> None:
    from ..core import Limits

    if set(value) != set(DEFAULT) or value["protocol"] != PROTOCOL:
        raise ValueError("Unknown/missing pass@k setting or protocol mismatch")
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
    if value["slots"] > 512 or value["sql_workers"] > 64:
        raise ValueError("At most 512 dialogue slots and 64 SQL workers")
    ks = value["ks"]
    if not (isinstance(ks, list) and ks and all(type(k) is int for k in ks) and ks[0] == 1
            and all(a < b for a, b in zip(ks, ks[1:])) and ks[-1] == value["samples"]):
        raise ValueError("ks rise from 1 to the sample count")
    if value["samples"] < 2:
        raise ValueError("pass@k needs at least two samples per task")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) \
            or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    control = value["control"]
    if not isinstance(control, dict) or set(control) != set(DEFAULT["control"]) \
            or set(control["sampling"]) != set(DEFAULT["control"]["sampling"]):
        raise ValueError("control has conditions, tasks and sampling (temperature, top_p, top_k)")
    allowed = {f"{family}_seed{seed}" for family in ("PTR", "RLb", "OPD") for seed in seeds} | {"B"}
    if not (isinstance(control["conditions"], list) and control["conditions"]
            and set(control["conditions"]) <= allowed and len(set(control["conditions"])) == len(control["conditions"])):
        raise ValueError("Control conditions are sampled models of this run")
    decoding = control["sampling"]
    if type(control["tasks"]) is not int or control["tasks"] < 1 or type(decoding["top_k"]) is not int \
            or decoding["top_k"] < 0 or any(type(decoding[k]) not in {int, float} or not low < decoding[k] <= high
                                            for k, (low, high) in (("temperature", (0.0, 2.0)), ("top_p", (0.0, 1.0)))):
        raise ValueError("Invalid control decoding or task count")
    if value["phase"] == "study":
        if (value["max_steps"], value["max_tokens"], value["max_context"]) != (4, 512, 8192):
            raise ValueError("The registered study keeps the v0.5-v0.12 agent protocol")
        if (value["temperature"], value["top_p"]) != (1.0, 1.0):
            raise ValueError("The registered study samples as the online runs did: temperature 1, top_p 1")
        if value["chunk_tasks"] * value["samples"] < 2 * value["slots"]:
            raise ValueError("A chunk smaller than twice the slots leaves the batch half empty in its last turns")
    Limits(**value["limits"])
