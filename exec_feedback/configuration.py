"""One user-edited JSON configuration, validated before any model is loaded."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .core import Limits
from .storage import atomic_json, read_json

DEFAULT = {
    "version": 3,
    "name": "qwen4b-sql-feedback",
    "data": {"raw_root": "data/raw", "train_json": "", "holdout_json": "",
             "terms_accepted": False, "validation_fraction": 0.2, "seed": 42},
    "model": {"backend": "transformers", "id": "Qwen/Qwen3-4B-Instruct-2507",
              "revision": "cdbee75f17c01a7cc42f958dc650907174af0554", "local_path": "",
              "device": "cuda", "quantization": "nf4", "allow_download": False,
              "endpoint": "http://127.0.0.1:8000/v1", "allow_remote": False,
              "batch_size": 1, "max_context_tokens": 8192, "max_tokens": 512, "temperature": 0.0, "seed": 42},
    "experiment": {"output_root": "outputs/studies", "train_limit": 256,
                   "validation_limit": 64, "test_limit": 0, "max_steps": 4,
                   "seeds": [42], "collection": "mixed", "min_pairs": 32,
                   "final_test": False, "review_success": False, "semantic_augmentation": False,
                   "stratified_sampling": False, "gold_screen": False},
    "training": {"max_length": 8192, "max_steps": 100, "gradient_accumulation_steps": 8,
                 "batch_size": 1, "learning_rate": 0.0002, "lora_r": 16, "lora_alpha": 32,
                 "save_steps": 10, "max_seconds": 7200, "warmup_ratio": 0.05},
    "limits": {"seconds": 10.0, "max_rows": 10000, "max_bytes": 8388608, "vm_steps": 50000000},
    "resources": {"max_model_calls": 2000, "max_active_seconds": 21600,
                  "workers": 2, "gpus": 1, "allow_fixture": False, "export_stages": False, "export_root": "exports"},
    "service": {"host": "127.0.0.1", "port": 8787, "adapter": "",
                "database_root": "data/raw", "results_root": "outputs/service", "max_steps": 4}
}


def load_config(path: Path) -> dict[str, Any]:
    supplied = read_json(path)
    if not isinstance(supplied, dict):
        raise ValueError("Config must be a JSON object")
    def merge(base: dict[str, Any], incoming: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        if not isinstance(incoming, dict):
            raise ValueError(f"{prefix} must be an object")
        unknown = set(incoming) - set(base)
        if unknown:
            raise ValueError(f"Unknown config keys at {prefix}: {sorted(unknown)}")
        for key, value in incoming.items():
            base[key] = merge(base[key], value, prefix + key + ".") if isinstance(base[key], dict) else value
        return base
    config = merge(copy.deepcopy(DEFAULT), supplied)
    validate_config(config)
    base = path.resolve().parent
    for section, keys in (("data", ("raw_root", "train_json", "holdout_json")),
                          ("model", ("local_path",)), ("experiment", ("output_root",)),
                          ("service", ("adapter", "database_root", "results_root")), ("resources", ("export_root",))):
        for key in keys:
            if config[section][key]:
                p = Path(config[section][key]).expanduser()
                config[section][key] = str((base / p).resolve() if not p.is_absolute() else p.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["version"] not in {2, 3, 4} or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", config["name"]):
        raise ValueError("Invalid config version or study name")
    # Config version 4 is protocol v0.3.1: v0.3 plus the pre-selection reference screen.
    if config["experiment"].get("gold_screen") is not (config["version"] == 4):
        raise ValueError("experiment.gold_screen=true is protocol v0.3.1 and requires config version 4, and only it")
    m, e, t, r = (config[x] for x in ("model", "experiment", "training", "resources"))
    if m["backend"] not in {"transformers", "http"} or m["device"] not in {"cpu", "cuda"} or m["quantization"] not in {"none", "nf4"}:
        raise ValueError("Unsupported model backend/device/quantization")
    if not m["id"] or (not m["local_path"] and not re.fullmatch(r"[0-9a-f]{40}", m["revision"])):
        raise ValueError("A remote model must have a fixed 40-character revision")
    if m["device"] == "cpu" and m["quantization"] != "none":
        raise ValueError("CPU mode requires quantization=none")
    if e["final_test"]:
        raise ValueError("Training studies use validation only; use final-test --study after freezing the trained adapters")
    if e["collection"] not in {"natural", "injected", "mixed"}:
        raise ValueError("collection must be natural, injected, or mixed")
    if not e["seeds"] or len(set(e["seeds"])) != len(e["seeds"]) or any(type(x) is not int or x < 0 for x in e["seeds"]):
        raise ValueError("seeds must be distinct nonnegative integers")
    for value, low, high, name in (
        (m["batch_size"], 1, 64, "inference batch size"), (e["max_steps"], 2, 16, "max_steps"), (e["min_pairs"], 1, 100000, "min_pairs"),
        (m["max_tokens"], 16, 8192, "max_tokens"), (m["max_context_tokens"], 256, 131072, "max_context_tokens"),
        (t["max_length"], 128, 131072, "max_length"), (t["max_steps"], 1, 1000000, "train steps"),
        (t["gradient_accumulation_steps"], 1, 1024, "gradient accumulation"),
        (t["batch_size"], 1, 64, "batch_size"), (t["lora_r"], 1, 512, "LoRA rank"),
        (t["save_steps"], 1, 100000, "save_steps"), (r["workers"], 1, 8, "workers"), (r["gpus"], 1, 8, "gpus"),
        (r["max_model_calls"], 1, 1000000, "max_model_calls")):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid {name}")
    for key in ("train_limit", "validation_limit", "test_limit"):
        if type(e[key]) is not int or e[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < t["learning_rate"] < 0.1 or not 0 <= t["warmup_ratio"] < 1:
        raise ValueError("Invalid learning rate/warmup")
    if not 0 <= m["temperature"] <= 2 or not 0 < t["max_seconds"] <= 604800 or not 0 < r["max_active_seconds"] <= 604800:
        raise ValueError("Invalid time/temperature limits")
    for section, key in (("data", "terms_accepted"), ("model", "allow_download"), ("model", "allow_remote"),
                         ("experiment", "final_test"), ("resources", "allow_fixture")):
        if type(config[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be a boolean")
    if type(config["service"]["port"]) is not int or not 1 <= config["service"]["port"] <= 65535:
        raise ValueError("Invalid service port")
    if type(config["service"]["max_steps"]) is not int or not 1 <= config["service"]["max_steps"] <= 16:
        raise ValueError("Invalid service step limit")
    if m["batch_size"] > 1 and (m["backend"] != "transformers" or m["temperature"] != 0):
        raise ValueError("Batched inference requires transformers + temperature=0")
    for key in ("review_success", "semantic_augmentation", "stratified_sampling", "gold_screen"):
        if type(e[key]) is not bool: raise ValueError(f"experiment.{key} must be boolean")
    if type(r["export_stages"]) is not bool: raise ValueError("export_stages must be boolean")
    if m["max_tokens"] >= m["max_context_tokens"]:
        raise ValueError("Output budget must be smaller than context budget")
    Limits(**config["limits"])


def initialize(path: Path) -> dict[str, Any]:
    if path.exists():
        raise ValueError("Config already exists; it was not overwritten")
    atomic_json(path, DEFAULT)
    return {"config": str(path), "next": "Set data paths/terms and model download or local cache, then run plan"}
