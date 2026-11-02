"""Full-weight models made from LoRA adapters: the 4B with a PTR adapter (the start point of the online runs) or with
v0.9's RL adapter at its registered milestone (the LoRA arm), merged in fp32 and rounded once to bf16. Every condition is
then trained and evaluated as a plain full-weight model, so the comparison with the LoRA arm sees one inference path.
The record next to each directory names its inputs and its weight checksum; directories are shared by every run on the
host whose freeze has the same inputs."""
from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..storage import atomic_json, exclusive_lock, read_json
from . import LORA, START
from .freeze import load, lora_adapter, model_ready, start_adapter
from .weights import COPY, WEIGHTS, checksum, dir_identity, weight_keys

PROBE = list(range(1000, 1128))  # any fixed ids: the check compares the same model before and after merging
MAX_FP32_DELTA = 1e-3
NAME = re.compile(rf"(?P<family>{START}|{LORA})_seed(?P<seed>\d+)")


def merged_dir(models_root: Path, name: str) -> Path:
    return Path(models_root) / "merged" / name


def record_of(folder: Path) -> Path:
    return folder.with_name(folder.name + ".json")


def adapter_of(run: Path, frozen: dict[str, Any], name: str) -> tuple[Path, str]:
    match = NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Not a mergeable condition: {name}")
    seed = int(match.group("seed"))
    if match.group("family") == START:
        return start_adapter(run, frozen, seed)
    folder, adapter_hash, _ = lora_adapter(run, frozen, seed)
    return folder, adapter_hash


def merged_model(run: Path, frozen: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    """The verified merged directory of a condition (raises if missing, built from other inputs or changed on disk)."""
    _, adapter_hash = adapter_of(run, frozen, name)
    folder = merged_dir(Path(frozen["models_root"]), name)
    path = record_of(folder)
    if not path.exists():
        raise ValueError(f"{name} has not been merged yet: {folder}")
    record = read_json(path)
    if record["inputs"]["base"] != frozen["models"]["student"]["identity"]["local_snapshot"] \
            or record["inputs"]["adapter"] != adapter_hash:
        raise ValueError(f"Merged {name} was built from other inputs")
    if dir_identity(folder) != record["identity"]:
        raise ValueError(f"Merged {name} changed on disk")
    return folder, record


def merged_start(run: Path, frozen: dict[str, Any], seed: int) -> tuple[Path, dict[str, Any]]:
    return merged_model(run, frozen, f"{START}_seed{seed}")


def merge_adapter(base: Path, adapter: Path, out: Path, *, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """The base model with a LoRA merged in fp32 (checked against the unmerged model on fixed probe ids), rounded once
    to bf16 and saved as one checkpoint with the base's config and tokenizer files; the record, naming `inputs`, goes
    next to the directory. The caller holds the directory's lock."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.float32, trust_remote_code=False,
                                                 local_files_only=True, attn_implementation="sdpa")
    model.eval()
    probe = torch.tensor([PROBE])
    policy = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    policy.eval()
    with torch.no_grad():
        unmerged = policy(input_ids=probe, use_cache=False).logits[0, -16:].float()
    merged = policy.merge_and_unload()
    with torch.no_grad():
        exact = merged(input_ids=probe, use_cache=False).logits[0, -16:].float()
    fp32_delta = float((exact - unmerged).abs().max())
    if fp32_delta > MAX_FP32_DELTA:
        raise RuntimeError(f"Merged weights do not reproduce the LoRA model in fp32 (max logit delta {fp32_delta})")
    merged = merged.to(torch.bfloat16)
    with torch.no_grad():
        rounded = merged(input_ids=probe, use_cache=False).logits[0, -16:].float()
    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    merged.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="64GB")
    if not (tmp / WEIGHTS).exists():
        raise RuntimeError("Expected one weight file from save_pretrained")
    for file in COPY:
        if (base / file).exists() and not (tmp / file).exists():
            shutil.copyfile(base / file, tmp / file)
    if weight_keys(tmp) != weight_keys(base):
        raise RuntimeError("Merged checkpoint names differ from the base checkpoint's")
    from safetensors.torch import load_file
    sums = checksum(load_file(str(tmp / WEIGHTS)).values())
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    record = {"name": name, "inputs": {**inputs, "peft": importlib.metadata.version("peft")},
              "adapter": str(adapter), "fp32_merge_max_logit_delta": fp32_delta,
              "bf16_rounding_max_logit_delta": float((rounded - exact).abs().max()),
              "keys": len(weight_keys(out)), "checksum": sums, "identity": dir_identity(out)}
    atomic_json(record_of(out), record)
    return record


def merge(run: Path, *, name: str, threads: int = 16) -> dict[str, Any]:
    import torch

    frozen, _ = load(run)
    match = NAME.fullmatch(name)
    if match is None or int(match.group("seed")) not in frozen["settings"]["seeds"]:
        raise ValueError(f"Unregistered condition: {name}")
    spec = model_ready(frozen)
    base = Path(spec["local_path"])
    adapter, adapter_hash = adapter_of(run, frozen, name)
    out = merged_dir(Path(frozen["models_root"]), name)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    with exclusive_lock(out.parent / f".{out.name}.lock", blocking=True):
        if record_of(out).exists():
            try:
                return merged_model(run, frozen, name)[1]
            except ValueError:
                stale = out.with_name(out.name + f".stale-{os.getpid()}")
                if out.exists():
                    os.replace(out, stale)
                os.replace(record_of(out), record_of(stale))
        return merge_adapter(base, adapter, out, name=name,
                             inputs={"base": frozen["models"]["student"]["identity"]["local_snapshot"],
                                     "adapter": adapter_hash})
