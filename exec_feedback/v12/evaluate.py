"""v0.12 greedy evaluation (vLLM, the v0.6-v0.11 path): the base 4B (B), v0.8's TR and PTR (imported), the OPD runs at
the registered milestone K, TROPD at min(K, its last milestone), any run at another milestone ('OPD_seed42@100', ad
hoc), and the teacher (B235) as v0.8 evaluated it: one engine load for several (run, role) pairs.
"""
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
from ..v7.freeze import _adapter_hash
from ..v9.evaluate import merged, vllm_factory  # noqa: F401  (same eval/<role>/<condition> layout; re-exported)
from ..v9.online import gold_for, milestones
from . import ARMS, EXPLORE, FAMILY, TEACHER
from .freeze import imported_adapter, load, model_ready
from .online import run_name

CONDITION = re.compile(rf"(?P<base>B|{TEACHER})|(?P<family>{'|'.join((*ARMS, FAMILY, EXPLORE))})_seed(?P<seed>\d+)"
                       r"(@(?P<step>\d+))?")


def parse(condition: str) -> tuple[str | None, int | None, int | None]:
    """'B' -> ('B', None, None); 'B235' -> ('B235', None, None); 'TR_seed42' -> ('TR', 42, None);
    'OPD_seed44@100' -> ('OPD', 44, 100). Imported arms have no other milestone."""
    match = CONDITION.fullmatch(condition)
    if match is None or (match.group("family") in ARMS and match.group("step")):
        raise ValueError(f"Unknown condition: {condition}")
    if match.group("base"):
        return condition, None, None
    return match.group("family"), int(match.group("seed")), int(match.group("step")) if match.group("step") else None


def common_steps(run: Path) -> dict[str, Any]:
    """Preregistered: the OPD runs are evaluated at K, the largest milestone up to planned_steps that every one of them
    reached; a run whose last milestone is below half of the best run's is reported as short, does not pull K down and
    is evaluated at its own last milestone (v0.9's rule). TROPD (exploratory) at min(K, its last milestone), as v0.9's
    RLhi."""
    frozen, _ = load(run)
    s = frozen["settings"]
    last = {}
    for family, seeds in ((FAMILY, s["seeds"]), (EXPLORE, s["explore_seeds"])):
        for seed in seeds:
            marks = [m for m in milestones(run / "train" / run_name(family, seed)) if m <= s["planned_steps"]]
            last[run_name(family, seed)] = max(marks) if marks else 0
    main = [run_name(FAMILY, seed) for seed in s["seeds"]]
    top = max((last[n] for n in main), default=0)
    kept = [n for n in main if last[n] > 0 and 2 * last[n] >= top]
    k = min((last[n] for n in kept), default=0)
    chosen = {}
    for name, reached in last.items():
        if not reached:
            continue
        if name in main:
            step = k if name in kept else reached
        else:
            step = min(k, reached) if k else reached
        chosen[name] = {"step": step, "adapter_hash": milestones(run / "train" / name)[step],
                        "short": name in main and name not in kept}
    result = {"run": frozen["identity"], "k": k, "last_milestones": last, "short": [n for n in main if n not in kept],
              "evaluate": chosen, "rule": "K = min last milestone <= planned_steps over OPD runs reaching >= half of "
                                          "the best one; TROPD at min(K, its last)"}
    atomic_json(run / "steps.json", result)
    return result


def adapter_for(run: Path, frozen: dict[str, Any], condition: str) -> tuple[Path | None, str | None, int | None]:
    family, seed, step = parse(condition)
    if family in ("B", TEACHER):
        return None, None, None
    if seed not in frozen["settings"]["seeds"]:
        raise ValueError(f"Unregistered seed in {condition}")
    if family in ARMS:
        folder, expected = imported_adapter(run, frozen, f"{family}_seed{seed}")
        return folder, expected, None
    name = run_name(family, seed)
    out = run / "train" / name
    if step is None:
        chosen = read_json(run / "steps.json")
        if chosen["run"] != frozen["identity"] or name not in chosen["evaluate"]:
            raise ValueError(f"No registered milestone for {name}")
        step, expected = chosen["evaluate"][name]["step"], chosen["evaluate"][name]["adapter_hash"]
    else:
        marks = milestones(out)
        if step not in marks:
            raise ValueError(f"{name} has no milestone at step {step}")
        expected = marks[step]
    if read_json(out / "manifest.json")["run"] != frozen["identity"]:
        raise ValueError(f"{name} belongs to another run")
    folder = out / "milestones" / f"step-{step:04d}" / "adapter"
    if _adapter_hash(folder) != expected:
        raise ValueError(f"Milestone adapter changed: {name}@{step}")
    return folder, expected, step


def teacher_factory(spec: dict[str, Any], settings: dict[str, Any], *, mem: float, seed: int) -> Any:
    """v0.8's teacher evaluation engine: 4-way tensor parallel, the student's context, greedy."""
    from ..v8.engine import generator, make_engine

    llm, tokenizer = make_engine(spec, settings, "teacher", mem=mem, seed=seed, lora=False,
                                 max_num_seqs=settings["teacher_eval_max_num_seqs"])
    return generator(llm, tokenizer, spec, settings, sampling=None, lora=None)


def evaluate(run: Path, *, condition: str, role: str, stop: str | None, mem: float | None = None,
             factory: Callable[..., Any] = vllm_factory, generator: Any = None,
             shared_load_seconds: float | None = None) -> dict[str, Any]:
    """One condition on one role; the teacher only with a generator built by teacher_factory (evaluate_teacher), whose
    counters then cover this role only."""
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    if role not in {"dev", "validation", "test"}:
        raise ValueError("Evaluation roles are dev, validation and test")
    frozen, tasks = load(run, role)
    s = frozen["settings"]
    kind = "teacher" if condition == TEACHER else "student"
    if kind == "teacher" and generator is None:
        raise ValueError("The teacher is evaluated through evaluate_teacher (one engine for several roles)")
    adapter, adapter_hash, step = adapter_for(run, frozen, condition)
    if generator is None and factory is vllm_factory:
        model_ready(frozen)
    spec = frozen["models"][kind]["spec"]
    limits, root = Limits(**s["limits"]), Path(frozen["raw_root"])
    gold = gold_for(run, frozen, role)
    # Per-condition order so concurrent evaluations do not hit one large database together.
    tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"{condition}:{role}:{t.id}".encode()).digest())
    identity = {"protocol": frozen["protocol"], "run": frozen["identity"], "role": role, "condition": condition,
                "model": frozen["models"][kind]["identity"], "adapter_hash": adapter_hash, "step": step,
                "code": frozen["code"]}
    out = run / "eval" / role / condition
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "episodes.jsonl"
    if (out / "summary.json").exists():
        return read_json(out / "summary.json")
    if journal.exists():  # an interrupted attempt: keep it for forensics, run again
        journal.rename(journal.with_name(journal.name + f".interrupted-{time.time_ns()}"))
    atomic_json(out / "meta.json", {**identity, "signature": fingerprint(identity)})
    started_load = time.monotonic()
    if generator is None:
        generator = factory(spec, s, adapter=adapter, mem=mem if mem is not None else s["eval_gpu_memory"],
                            seed=20261014, name=condition.replace("@", "-at-"))
    load_seconds = time.monotonic() - started_load if shared_load_seconds is None else shared_load_seconds
    before = dict(getattr(generator, "metadata", {}))
    slots = s["teacher_eval_slots"] if kind == "teacher" else s["eval_slots"]

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        row = grade_episode(task, task.database(root), "model", record, limits, s["max_steps"], True,
                            gold=gold.get(task.id))
        row["condition"] = condition
        return row

    statuses, started = Counter(), time.monotonic()
    for row in run_dialogues([(t.id, t) for t in tasks], root, generator, slots=slots, max_steps=s["max_steps"],
                             review_success=True, limits=limits, workers=s["sql_workers"], finish=finish,
                             deadline=deadline_from(stop), fenced_json=True):
        append_record(journal, row)
        statuses[row["status"]] += 1
    done = sum(statuses.values())
    counters = {k: v - before.get(k, 0) if type(v) in (int, float) and k != "max_batch" else v
                for k, v in dict(getattr(generator, "metadata", {})).items()}
    summary = {**identity, "registered_tasks": len(tasks), "episodes": done, "complete": done == len(tasks),
               "statuses": dict(statuses), "correct": statuses.get("correct", 0),
               "wall_seconds": time.monotonic() - started, "engine_load_seconds": load_seconds,
               "engine_shared": shared_load_seconds is not None, "generator": counters,
               "gold_cache_entries": len(gold),
               "packages": versions(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "episodes_sha256": digest(journal) if journal.exists() else None}
    atomic_json(out / ("summary.json" if summary["complete"] else "partial.json"), summary)
    return summary


def evaluate_teacher(jobs: list[tuple[Path, str]], *, stop: str | None,
                     factory: Callable[..., Any] = teacher_factory) -> list[dict[str, Any]]:
    """The teacher on several (run, role) pairs with one engine load, in the given order (the smoke subset first: a
    failure there stops before the full sets). Every run must pin the same teacher."""
    frozen = [load(run)[0] for run, _ in jobs]
    if len({fingerprint(f["models"]["teacher"]["identity"]) for f in frozen}) != 1:
        raise ValueError("The runs pin different teachers")
    pending = [(run, role) for run, role in jobs if not (run / "eval" / role / TEACHER / "summary.json").exists()]
    if not pending:
        return [read_json(run / "eval" / role / TEACHER / "summary.json") for run, role in jobs]
    first = frozen[0]
    spec = model_ready(first, "teacher") if factory is teacher_factory else first["models"]["teacher"]["spec"]
    started = time.monotonic()
    generator = factory(spec, first["settings"], mem=first["settings"]["teacher_eval_gpu_memory"], seed=20261013)
    load_seconds = time.monotonic() - started
    out = []
    for run, role in jobs:
        summary = evaluate(run, condition=TEACHER, role=role, stop=stop, generator=generator,
                           shared_load_seconds=load_seconds)
        out.append(summary)
        if not summary["complete"]:
            break  # a deadline cut this role short; the later ones would be cut too
    return out
