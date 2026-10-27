"""Small training-state checkpoints; raw rollout journals are never rewritten."""
from __future__ import annotations

import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from ..core import digest
from ..storage import atomic_json, read_json


def trainable(model: Any) -> dict[str, Any]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(out: Path, model: Any, optimizer: Any,
                    state: dict[str, Any]) -> Path:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"checkpoint-{state['update']:08d}"
    if destination.exists():
        raise FileExistsError("A completed checkpoint must not be overwritten")
    temporary = out / f".checkpoint-{time.time_ns()}"
    temporary.mkdir()
    torch.save({n: p.detach().cpu() for n, p in trainable(model).items()},
               temporary / "trainable.pt")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save({"cpu": torch.get_rng_state(), "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               temporary / "rng.pt")
    atomic_json(temporary / "complete.json", {
        **state, "files": {p.name: digest(p) for p in temporary.glob("*.pt")}})
    os.replace(temporary, destination)
    # Training state only: keep two complete checkpoints, never delete journals.
    complete = sorted(p for p in out.glob("checkpoint-*")
                      if (p / "complete.json").exists())
    for old in complete[:-2]:
        shutil.rmtree(old)
    return destination


def restore_checkpoint(out: Path, model: Any, optimizer: Any,
                       identity: str) -> dict[str, Any] | None:
    import torch

    complete = sorted(p for p in out.glob("checkpoint-*")
                      if (p / "complete.json").exists())
    if not complete:
        return None
    checkpoint = complete[-1]
    state = read_json(checkpoint / "complete.json")
    if state["identity"] != identity:
        raise ValueError("RL checkpoint identity changed")
    if set(state["files"]) != {"trainable.pt", "optimizer.pt", "rng.pt"}:
        raise ValueError("RL checkpoint file manifest is incomplete")
    if checkpoint.name != f"checkpoint-{state['update']:08d}":
        raise ValueError("RL checkpoint update does not match its directory")
    if any(digest(checkpoint / name) != value for name, value in state["files"].items()):
        raise ValueError("RL checkpoint hash mismatch")
    parameters = trainable(model)
    weights = torch.load(checkpoint / "trainable.pt", map_location="cpu", weights_only=True)
    if set(weights) != set(parameters):
        raise ValueError("RL trainable parameter names changed")
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].device))
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=True))
    rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu"].cpu())
    random.setstate(rng["python"])
    if rng["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Cannot resume a CUDA RNG checkpoint on CPU")
        torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
    return state
