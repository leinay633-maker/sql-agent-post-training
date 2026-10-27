"""v0.9 scheduler for one 8-GPU host (the v0.8 Scheduler, a new plan).

Smoke run S (16 tasks per role, two online steps per method) and main run M share the scheduler; a main code path
starts only after the smoke run has completed the same path. The online runs take one whole GPU each and train until
`until` = stop - FINAL_RESERVE, so that the common milestone, its evaluations and the report fit before the stop.
Plan: docs/RESEARCH_V09.md; operation: docs/H20_V09_RUNBOOK.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..v8 import orchestrate as base
from ..v8.orchestrate import Job, Scheduler, every, exists, iso, read
from . import METHODS, START

KILL_BEFORE_STOP = 10 * 60           # every job is killed by stop - 10 min; the report runs in that tail
JOB_STOP_BEFORE_STOP = 12 * 60       # evaluations start no new dialogue after this
FINAL_RESERVE = 45 * 60              # online runs end this long before the stop: K, 24 evaluations, report, export
SCHEDULER_RESERVE = 15 * 60          # a job must fit before stop - this
EXPORT_EVERY = 30 * 60               # results package during the long online phase (metrics, curves, STATUS)
ROLES = ("dev", "validation", "test")
ESTIMATES = {"dev": 300.0, "validation": 600.0, "test": 900.0}
ORDER = [("RL", 0), ("RAFT", 0), ("RL", 1), ("RAFT", 1), ("RL", 2), ("RAFT", 2)]


class Plan:
    """Jobs for the smoke run S and the main run M, recomputed from disk on every scheduler pass."""

    def __init__(self, *, run: Path, smoke_run: Path, models_root: Path, main_py: str, vllm_py: str,
                 stop_epoch: float, gpus: list[int]) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.settings = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8"))["settings"]
                         for tag, r in self.runs.items()}
        self.models_root, self.main_py, self.vllm_py, self.stop_epoch = models_root.resolve(), main_py, vllm_py, stop_epoch
        self.gpus = gpus
        self.smoke_gpus = set(gpus[:2])
        self.until = stop_epoch - FINAL_RESERVE
        self.scheduler: Scheduler | None = None

    def model(self) -> Callable[[], bool]:
        return exists(self.models_root / "student" / ".efa-verified.json")

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.v9", *a]

    def dead(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.dead(name)

    def running(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.is_running(name)

    def trained(self, tag: str, name: str) -> Callable[[], bool]:
        return lambda: bool((read(self.runs[tag] / "train" / name / "result.json") or {}).get("finished"))

    def evaluated(self, tag: str, role: str, condition: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "eval" / role / condition / "summary.json")

    def online(self, tag: str) -> list[str]:
        s = self.settings[tag]
        if tag == "S":
            return [f"RL_seed{s['seeds'][0]}", f"RAFT_seed{s['seeds'][0]}"]
        names = [f"{family}_seed{s['seeds'][i]}" for family, i in ORDER if i < len(s["seeds"])]
        return names + [f"RLhi_seed{seed}" for seed in s["hi_seeds"]]

    def settled(self, tag: str) -> Callable[[], bool]:
        """Every online run of the tag finished, gave up, or can no longer start (its training window is over)."""
        def check() -> bool:
            names = self.online(tag)
            if any(self.running(f"{tag}:train-{n}") for n in names):
                return False
            late = time.time() > self.until - 30 * 60
            return all(self.trained(tag, n)() or self.dead(f"{tag}:train-{n}") or (tag == "M" and late) for n in names)
        return check

    def smoke_training_pending(self) -> bool:
        return not all(self.trained("S", n)() or self.dead(f"S:train-{n}") for n in self.online("S"))

    def eval_avoid(self) -> set[int]:
        return set(self.smoke_gpus) if self.smoke_training_pending() else set()

    def __call__(self) -> tuple[list[Job], list[Any]]:
        jobs: list[Job] = []
        for tag in ("S", "M"):
            jobs += self.for_run(tag)
        return jobs, []

    def for_run(self, tag: str) -> list[Job]:
        run, s = self.runs[tag], self.settings[tag]
        smoke = tag == "S"
        r = str(run)
        jobs: list[Job] = []
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        steps = run / "steps.json"
        smoke_first = s["seeds"][0]
        gate_eval = self.evaluated("S", "dev", f"{START}_seed{self.settings['S']['seeds'][0]}")
        gate_train = lambda: all(self.trained("S", n)() for n in self.online("S"))

        evals: list[Job] = []

        def evaluation(condition: str, role: str, deps: Callable[[], bool], priority: int, stage: str) -> None:
            evals.append(Job(f"{tag}:eval-{condition}-{role}",
                            self.py("vllm", "evaluate", "--run", r, "--condition", condition, "--role", role,
                                    "--stop", stop_iso),
                            1, every(deps, self.model(), (lambda: True) if smoke else gate_eval),
                            self.evaluated(tag, role, condition), priority, 600.0 if smoke else ESTIMATES[role],
                            stage=f"{tag}:{stage}", retries=2, kind="" if smoke else f"eval-{role}",
                            avoid=self.eval_avoid))

        # ---- online runs (one whole GPU each; smoke runs on the first two GPUs)
        until = iso(min(self.until, time.time() + 3 * 3600)) if smoke else iso(self.until)
        for i, name in enumerate(self.online(tag)):
            family, seed = name.split("_seed")
            enough = (lambda: True) if smoke else (lambda: self.until - time.time() > 30 * 60)
            jobs.append(Job(f"{tag}:train-{name}",
                            self.py("vllm", "train", "--run", r, "--family", family, "--seed", seed, "--until", until),
                            2, every(self.model(), enough, (lambda: True) if smoke else gate_train),
                            self.trained(tag, name), (10 if smoke else 120) + i, 600.0 if smoke else 1200.0,
                            stage=f"{tag}:train", retries=2,
                            avoid=(lambda: set(self.gpus) - self.smoke_gpus) if smoke else base.NO_GPUS))
        jobs.append(Job(f"{tag}:steps", self.py("main", "steps", "--run", r), 0, self.settled(tag), exists(steps),
                        (30 if smoke else 140), 120.0, stage=f"{tag}:steps", retries=1))

        # ---- evaluations: the start points first, the online families at their registered milestone
        roles = ("dev",) if smoke else ROLES
        if smoke:
            for role in ROLES:
                evaluation("B", role, lambda: True, 5, "eval-base")
            evaluation(f"{START}_seed{smoke_first}", "dev", lambda: True, 4, "eval-base")
        else:
            for role in ("test", "validation", "dev"):
                evaluation("B", role, lambda: True, 100, "eval-base")
                for seed in s["seeds"]:
                    evaluation(f"{START}_seed{seed}", role, lambda: True, 100, "eval-base")
        chosen = (read(steps) or {}).get("evaluate", {})  # runs without any milestone get no evaluation job
        for name in self.online(tag):
            if name in chosen:
                for role in (("test", "validation", "dev") if not smoke else roles):
                    evaluation(name, role, lambda: True, 40 if smoke else 150, "eval-online")
        jobs += evals
        report_ready = every(exists(steps), lambda: all(j.done() or self.dead(j.name) for j in evals))
        jobs.append(Job(f"{tag}:report", self.py("main", "report", "--run", r, "--procs", "8" if smoke else "96"),
                        0, report_ready, exists(run / "report" / "report.json"), 90 if smoke else 190, 600.0,
                        stage=f"{tag}:report", retries=1))
        return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--smoke-run", type=Path, required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--stop", required=True, help="absolute compute stop, ISO with offset")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--main-py", required=True)
    ap.add_argument("--vllm-py", required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--export-hook", default="")
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()
    stop = dt.datetime.fromisoformat(args.stop).timestamp()
    gpus = [int(g) for g in args.gpus.split(",")]
    plan = Plan(run=args.run, smoke_run=args.smoke_run, models_root=args.models_root, main_py=args.main_py,
                vllm_py=args.vllm_py, stop_epoch=stop, gpus=gpus)

    def export(label: str) -> None:
        if args.export_hook:
            subprocess.run(["bash", args.export_hook, label], capture_output=True, timeout=1800)

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        if tag == "S" and stage != "S:report":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        subprocess.run([args.main_py, "-m", "exec_feedback.v9", "progress", "--run", str(plan.runs[tag]), "--stage",
                        stage, "--commit", args.commit, "--file", str(args.progress)], cwd=args.repo,
                       capture_output=True, timeout=600)
        export(stage.replace(":", "-"))

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=60s", str(int(seconds)), *cmd]

    os.chdir(args.repo)
    scheduler = Scheduler(gpus=gpus, stop=stop, status=args.status, log_dir=args.log_dir, plan=plan,
                          external=lambda: not plan.model()(), reserve=SCHEDULER_RESERVE, wrap=wrap,
                          milestone=milestone, kill_before=KILL_BEFORE_STOP)
    plan.scheduler = scheduler
    scheduler.note(f"v0.9 scheduler: GPUs {gpus}, smoke GPUs {sorted(plan.smoke_gpus)}, online runs until "
                   f"{dt.datetime.fromtimestamp(plan.until, base.CST):%m-%d %H:%M}, stop {args.stop}")
    last_export = time.time()
    while scheduler.step():
        if time.time() - last_export > EXPORT_EVERY and any(
                j["name"].startswith("M:train-") for j in scheduler.running.values()):
            last_export = time.time()
            export(f"progress-{dt.datetime.now(base.CST):%H%M}")
            scheduler.note("exported a progress package (online metrics and dev curves so far)")
        time.sleep(scheduler.poll)
    scheduler.note("scheduler: nothing running, nothing left that can start")
    if not (plan.runs["M"] / "report" / "report.json").exists():
        subprocess.run([args.main_py, "-m", "exec_feedback.v9", "report", "--run", str(plan.runs["M"]), "--procs", "96"],
                       cwd=args.repo, timeout=KILL_BEFORE_STOP - 60)
        milestone("M:report")
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
