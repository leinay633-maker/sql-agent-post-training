"""Full-weight files shared by the trainer, the rollout engines and the evaluator.

The trainer streams each FSDP parameter to rank 0 as bf16 and writes one safetensors file per version; an engine loads
it in place and both sides compute the same checksum (float64 sum and sum of squares over every parameter). vLLM stacks
q/k/v and gate/up into single parameters, and both sums are additive, so the two sides agree exactly when the engine
holds the trainer's weights.
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..core import digest
from ..storage import atomic_json, fingerprint, read_json

WEIGHTS = "model.safetensors"
COPY = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
MARKER = ".efa-dir.json"


def checksum(tensors: Iterable[Any]) -> list[float]:
    import torch

    total = squares = 0.0
    for tensor in tensors:
        flat = tensor.detach().reshape(-1)
        for part in flat.split(1 << 24):
            x = part.to(torch.float64)
            total += float(x.sum())
            squares += float((x * x).sum())
    return [total, squares]


def same_checksum(a: list[float], b: list[float]) -> bool:
    """Equal up to float64 summation order (the sums run over billions of bf16 values in different orders)."""
    scale = math.sqrt(max(abs(a[1]), abs(b[1]), 1.0))
    return abs(a[0] - b[0]) <= 1e-6 * scale and abs(a[1] - b[1]) <= 1e-9 * max(abs(a[1]), 1.0)


def export_full(model: Any, rank: int) -> tuple[dict[str, Any] | None, list[float] | None]:
    """Every parameter in bf16 on rank 0's CPU, one all-gather at a time (all ranks must call this together), and the
    checksum of exactly those bf16 values, computed on the GPU before the copy. named_parameters() lists a tied
    embedding once, as the checkpoint stores it."""
    import torch

    out: dict[str, Any] | None = {} if rank == 0 else None
    total = squares = 0.0
    for name, param in model.named_parameters():
        full = param.full_tensor() if hasattr(param, "full_tensor") else param.detach()
        if out is not None:
            rounded = full.detach().to(torch.bfloat16)
            part = checksum([rounded])
            total, squares = total + part[0], squares + part[1]
            out[name] = rounded.cpu().contiguous()
            del rounded
        del full
    return out, ([total, squares] if out is not None else None)


def write_weights(folder: Path, state: dict[str, Any], meta: dict[str, Any],
                  sums: list[float] | None = None) -> dict[str, Any]:
    """Write <folder>/model.safetensors and <folder>.ready.json last (readers wait for the ready file)."""
    from safetensors.torch import save_file

    tmp = folder.with_name(folder.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    save_file(state, str(tmp / WEIGHTS), metadata={"format": "pt"})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    ready = {**meta, "path": str(folder / WEIGHTS), "bytes": (folder / WEIGHTS).stat().st_size, "keys": len(state),
             "checksum": sums if sums is not None else checksum(state.values())}
    atomic_json(folder.with_name(folder.name + ".ready.json"), ready)
    return ready


def model_dir(dest: Path, weights: Path, start: Path) -> str:
    """A loadable model directory: the weight file plus the start directory's config and tokenizer files."""
    tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copyfile(weights, tmp / WEIGHTS)
    for name in COPY:
        if (start / name).exists():
            shutil.copyfile(start / name, tmp / name)
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return dir_identity(dest)


def dir_identity(folder: Path) -> str:
    """sha256 fingerprint of every file of a model directory. The full hash runs once; later calls compare the recorded
    size/inode/mtime of each file (8 GB per directory)."""
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != MARKER)
    if not files:
        raise ValueError(f"Empty model directory: {folder}")
    stats = {p.relative_to(folder).as_posix(): [p.stat().st_size, p.stat().st_ino, p.stat().st_mtime_ns] for p in files}
    marker = folder / MARKER
    if marker.exists():
        saved = read_json(marker)
        if saved.get("stat") == stats:
            return saved["identity"]
    identity = fingerprint({p.relative_to(folder).as_posix(): digest(p) for p in files})
    atomic_json(marker, {"identity": identity, "stat": stats})
    return identity


def weight_keys(folder: Path) -> set[str]:
    """Tensor names of a Hugging Face checkpoint directory (sharded with an index, or one file)."""
    from safetensors import safe_open

    index = folder / "model.safetensors.index.json"
    if index.exists():
        return set(read_json(index)["weight_map"])
    with safe_open(str(folder / WEIGHTS), framework="pt") as handle:
        return set(handle.keys())
