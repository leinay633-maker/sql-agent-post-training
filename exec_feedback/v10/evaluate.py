"""v0.10 greedy evaluation (vLLM, the v0.6-v0.9 agent path) of full-weight models only: the base 4B, the merged PTR
start points, v0.9's RL merged at its registered milestone (the LoRA arm) and the online runs at their final milestone K.
A run that has not finished all K steps is not evaluated; '@step' conditions (other milestones) are exploratory."""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, fingerprint, read_json
from ..v6.rollouts import deadline_from, versions
from ..v9.evaluate import merged  # noqa: F401  (same eval/<role>/<condition> layout; re-exported)
from ..v9.online import gold_for
from . import FAMILIES, LORA, START
from .freeze import load, model_ready
from .merge import merged_model
from .trainer import milestones, run_name
from .weights import dir_identity

CONDITION = re.compile(rf"(?P<base>B)|(?P<family>{START}|{LORA}|{'|'.join(FAMILIES)})_seed(?P<seed>\d+)"
                       r"(@(?P<step>\d+))?")


def parse(condition: str) -> tuple[str | None, int | None, int | None]:
    """'B' -> (None, None, None); 'RLF_seed42' -> ('RLF', 42, None); 'RLFa_seed42@60' -> ('RLFa', 42, 60)."""
    match = CONDITION.fullmatch(condition)
    if match is None or (match.group("family") in (START, LORA) and match.group("step")):
        raise ValueError(f"Unknown condition: {condition}")
    if match.group("base"):
        return None, None, None
    return match.group("family"), int(match.group("seed")), int(match.group("step")) if match.group("step") else None


def model_for(run: Path, frozen: dict[str, Any], condition: str) -> tuple[Path, str, int | None]:
    """(model directory, its identity, online step) of a condition, verified on disk."""
    family, seed, step = parse(condition)
    if family is None:
        path = Path(model_ready(frozen)["local_path"])
        return path, frozen["models"]["student"]["identity"]["local_snapshot"], None
    if seed not in frozen["settings"]["seeds"]:
        raise ValueError(f"Unregistered seed in {condition}")
    if family in (START, LORA):
        folder, record = merged_model(run, frozen, f"{family}_seed{seed}")
        return folder, record["identity"], (0 if family == START else frozen["imports"]["v09"]["rl"][
            f"{LORA}_seed{seed}"]["step"])
    name = run_name(family, seed)
    out = run / "train" / name
    if step is None:
        result = read_json(out / "result.json") if (out / "result.json").exists() else {}
        if not result.get("finished") or result.get("identity") != read_json(out / "manifest.json")["identity"]:
            raise ValueError(f"{name} has not finished its {frozen['settings']['planned_steps']} steps")
        step = frozen["settings"]["planned_steps"]
    if read_json(out / "manifest.json")["run"] != frozen["identity"]:
        raise ValueError(f"{name} belongs to another run")
    marks = milestones(out)
    if step not in marks:
        raise ValueError(f"{name} has no milestone at step {step}")
    folder = out / "milestones" / f"step-{step:04d}"
    if dir_identity(folder) != marks[step]:
        raise ValueError(f"Milestone changed on disk: {name}@{step}")
    return folder, marks[step], step


def vllm_factory(spec: dict[str, Any], settings: dict[str, Any], *, model_dir: Path, mem: float, seed: int) -> Any:
    """The v0.8 student engine on another weight directory; tokenizer, chat template and stop ids stay the base's."""
    from vllm import LLM
    from ..models import load_tokenizer
    from ..v8.engine import generator

    llm = LLM(model=str(model_dir), tokenizer=spec["local_path"], dtype="bfloat16",
              max_model_len=settings["max_context"], gpu_memory_utilization=mem, enable_prefix_caching=True,
              seed=seed, generation_config="vllm", max_num_seqs=settings["eval_max_num_seqs"],
              trust_remote_code=False)
    return generator(llm, load_tokenizer(spec), spec, settings, sampling=None, lora=None)


def evaluate(run: Path, *, condition: str, role: str, stop: str | None, mem: float | None = None,
             factory: Callable[..., Any] = vllm_factory) -> dict[str, Any]:
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load(run, role)
    s = frozen["settings"]
    spec = model_ready(frozen) if factory is vllm_factory else frozen["models"]["student"]["spec"]
    model_dir, model_identity, step = model_for(run, frozen, condition)
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    gold = gold_for(run, frozen, role)
    # Per-condition order so concurrent evaluations do not hit one large database together.
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{condition}:{role}:{t.id}".encode()).digest())
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "model": frozen["models"]["student"]["identity"], "weights": model_identity, "step": step,
                "code": frozen["code"]}
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "episodes.jsonl"
    if (out / "summary.json").exists():
        return read_json(out / "summary.json")
    if journal.exists():  # an interrupted attempt: keep it for forensics, run again
        journal.rename(journal.with_name(journal.name + f".interrupted-{time.time_ns()}"))
    atomic_json(out / "meta.json", {**identity, "signature": fingerprint(identity), "model_dir": str(model_dir)})
    started_load = time.monotonic()
    generator = factory(spec, s, model_dir=model_dir, mem=mem if mem is not None else s["eval_gpu_memory"],
                        seed=20261014)
    load_seconds = time.monotonic() - started_load

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        row = grade_episode(task, task.database(root), "model", record, limits, s["max_steps"], True,
                            gold=gold.get(task.id))
        row["condition"] = condition
        return row

    statuses, started = Counter(), time.monotonic()
    for row in run_dialogues([(t.id, t) for t in tasks], root, generator, slots=s["eval_slots"],
                             max_steps=s["max_steps"], review_success=True, limits=limits, workers=s["sql_workers"],
                             finish=finish, deadline=deadline_from(stop), fenced_json=True):
        append_record(journal, row)
        statuses[row["status"]] += 1
    done = sum(statuses.values())
    summary = {**identity, "registered_tasks": len(tasks), "episodes": done, "complete": done == len(tasks),
               "statuses": dict(statuses), "correct": statuses.get("correct", 0),
               "wall_seconds": time.monotonic() - started, "engine_load_seconds": load_seconds,
               "generator": getattr(generator, "metadata", {}), "gold_cache_entries": len(gold),
               "packages": versions(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "episodes_sha256": digest(journal) if journal.exists() else None}
    atomic_json(out / ("summary.json" if summary["complete"] else "partial.json"), summary)
    return summary
