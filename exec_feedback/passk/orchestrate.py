"""pass@k scheduler for one H20 (the v0.8 Scheduler, a new plan).

In priority order on the GPU: the whole smoke run S (16 test tasks, 16 dialogues each, the first seed's models and the
decoding control, then its report); then the main run M: every model's sampling in the frozen order (seed 42's PTR, RLb,
B and OPD, the decoding control, then seed 43's and seed 44's PTR and RLb, their OPD last), half a GPU each, two at a
time, then the report. Main sampling starts once the smoke run is reported with every smoke model sampled; a sampler
that would not finish before the stop does not start (its estimate comes from this host's measured seconds per task),
and a main-comparison pair starts together or not at all: a pair's PTR only if its RLb can finish too, the RLb only
once its PTR runs. With --only, this host samples just the named main-run models (in the same order; whole pairs
only) and the rest are sampled on another host from the same bundle, merged on the PC afterwards.
Plan: docs/RESEARCH_PASSK.md; operation: docs/H20_PASSK_RUNBOOK.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..v8 import orchestrate as base
from ..v8.orchestrate import Job, Scheduler, every, exists, iso, read
from . import RL, START
from .freeze import CONTROL, parse

KILL_BEFORE_STOP = 10 * 60           # every job is killed by stop - 10 min
JOB_STOP_BEFORE_STOP = 15 * 60       # samplers start no new dialogue after this; the report and export use the tail
SCHEDULER_RESERVE = 15 * 60          # a job must fit before stop - this
EXPORT_EVERY = 60 * 60               # results package and a STATUS line per running sampler while sampling
# Seconds per test task (64 dialogues) on half an H20 before this host measured one: v0.11's pre-pass sampled 53,456
# dialogues in 172 min per half GPU (5.2 per second, 12.3 s per 64); test tasks come from BIRD dev, so 1.15 x that.
SECONDS_PER_TASK = 14.0
LOAD_SECONDS = 600.0                 # engine start, gold cache, the last chunk's tail


def pair_of(condition: str, conditions: list[str]) -> tuple[str, str] | None:
    """The main comparison's same-seed (PTR, RLb) pair a condition belongs to (decoding-control pairs too), if both
    are sampled in this run."""
    family, seed, control = parse(condition)
    if family not in (START, RL):
        return None
    suffix = CONTROL if control else ""
    pair = (f"{START}_seed{seed}{suffix}", f"{RL}_seed{seed}{suffix}")
    return pair if all(c in conditions for c in pair) else None


class Plan:
    """Jobs for the smoke run S and the main run M, recomputed from disk on every scheduler pass."""

    def __init__(self, *, run: Path, smoke_run: Path, models_root: Path, main_py: str, vllm_py: str,
                 stop_epoch: float, only: list[str] | None = None) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.frozen = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8")) for tag, r in self.runs.items()}
        self.models_root, self.main_py, self.vllm_py, self.stop_epoch = models_root.resolve(), main_py, vllm_py, stop_epoch
        self.scheduler: Scheduler | None = None
        self.only: set[str] | None = None
        if only is not None:  # given, so never "all": an empty list or an empty name is an error
            names = self.frozen["M"]["conditions"]
            unknown = sorted(set(only) - set(names))
            split = [pair for c in names if (pair := pair_of(c, names)) and len(set(pair) & set(only)) == 1]
            if not only or unknown or split:
                raise ValueError(f"--only must name main-run models, and pairs whole: given {only}, unknown "
                                 f"{unknown}, split {sorted(set(split))}")
            self.only = set(only)

    def model(self) -> Callable[[], bool]:
        return exists(self.models_root / "student" / ".efa-verified.json")

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.passk", *a]

    def dead(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.dead(name)

    def running(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.is_running(name)

    def conditions(self, tag: str) -> list[str]:
        """The models this host samples for run `tag`, in the frozen order: all of them, or the --only ones."""
        names = self.frozen[tag]["conditions"]
        return names if tag == "S" or self.only is None else [c for c in names if c in self.only]

    def folder(self, tag: str, condition: str) -> Path:
        return self.runs[tag] / "samples" / condition

    def sampled(self, tag: str, condition: str) -> Callable[[], bool]:
        return exists(self.folder(tag, condition) / "result.json")

    def reported(self, tag: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "report" / "report.json")

    def smoke_over(self) -> bool:
        return (self.reported("S")() and not self.running("S:report")) or self.dead("S:report")

    def smoke_ok(self) -> bool:
        """The smoke run is reported (its report job ended) and every one of its models sampled: the main samplers may
        start."""
        return self.reported("S")() and not self.running("S:report") \
            and all(self.sampled("S", c)() for c in self.conditions("S"))

    def seconds_per_task(self) -> float:
        """Measured on this host: finished main models first, else the running ones' progress, else v0.11's rate."""
        done = [read(self.folder("M", c) / "result.json") or {} for c in self.conditions("M")]
        rates = [r["sampling_seconds"] / r["tasks"] for r in done if r.get("sampling_seconds") and r.get("tasks")]
        if not rates:
            rates = [p["seconds_per_task"] for c in self.conditions("M")
                     if (p := read(self.folder("M", c) / "progress.json")) and p.get("done", 0) >= 64]
        return 1.05 * statistics.fmean(rates) if rates else SECONDS_PER_TASK

    def tasks_left(self, tag: str, condition: str) -> int:
        total = self.frozen[tag]["control"]["tasks"] if condition.endswith("+rec") else self.frozen[tag]["counts"]["test"]
        progress = read(self.folder(tag, condition) / "progress.json") or {}
        return max(0, total - int(progress.get("done", 0)))

    def seconds_left(self, job: dict[str, Any], per_task: float) -> float:
        """A running sampler's remaining time: its tasks left, plus the engine start until this try reports progress (a
        retry finds the last try's progress file before writing its own)."""
        tag, _, condition = job["name"].partition(":sample-")
        progress = self.folder(tag, condition) / "progress.json"
        began = time.time() - (time.monotonic() - job["started"])
        loading = not progress.exists() or progress.stat().st_mtime < began
        return per_task * self.tasks_left(tag, condition) + (LOAD_SECONDS if loading else 0.0)

    def startable(self, tag: str, condition: str, per_task: float) -> bool:
        """Main-comparison pairs start together: a pair's PTR only if its RLb can finish before the stop reserve too,
        starting on the second free half GPU, or on the first half a running sampler frees, and never before the
        scheduler's stagger after the PTR; the RLb only once its PTR runs, finished or failed for good. A refused PTR
        is noted once. Other samplers always."""
        pair = pair_of(condition, self.conditions(tag))
        s = self.scheduler
        if pair is None or s is None:
            return True
        ptr, rlb = pair
        if condition == rlb:
            return self.running(f"{tag}:sample-{ptr}") or self.sampled(tag, ptr)() or self.dead(f"{tag}:sample-{ptr}")
        if self.running(f"{tag}:sample-{rlb}") or self.sampled(tag, rlb)():
            return True  # a retry of the PTR: its RLb already runs or finished
        halves = [j for j in s.running.values() if j["slots"]]
        free = 2 * len(s.gpus) - sum(j["slots"] for j in halves)
        ends = sorted(self.seconds_left(j, per_task) for j in halves) + [0.0, 0.0]
        if free >= 2:
            wait = s.stagger
        elif free == 1:
            wait = max(s.stagger, ends[0])
        else:  # the PTR waits for the first half to free, its RLb for the second
            wait = max(ends[0] + s.stagger, ends[1])
        need = wait + LOAD_SECONDS + per_task * self.tasks_left(tag, rlb)
        if time.time() + need <= s.stop - s.reserve:
            return True
        name = f"{tag}:sample-{condition}"
        if name not in s.reported_late:
            s.reported_late.add(name)
            s.note(f"not starting {name}: its pair {rlb} would need ~{need / 60:.0f} min, "
                   f"{max(0.0, s.stop - s.reserve - time.time()) / 60:.0f} min left before the stop reserve")
        return False

    def given_up(self, job: Job, per_task: float) -> bool:
        """Done, dead, or no longer started at all (its estimate ends past the stop reserve, or its main-comparison pair
        cannot start): the report does not wait."""
        s = self.scheduler
        tag, _, condition = job.name.partition(":sample-")
        return job.done() or self.dead(job.name) or (
            s is not None and not self.running(job.name)
            and (time.time() + s.estimate(job) > s.stop - s.reserve or not self.startable(tag, condition, per_task)))

    def __call__(self) -> tuple[list[Job], list[Any]]:
        return self.for_run("S") + self.for_run("M"), []

    def for_run(self, tag: str) -> list[Job]:
        run = self.runs[tag]
        smoke = tag == "S"
        r = str(run)
        offset = 0 if smoke else 100
        until = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        if not smoke and self.smoke_over() and not self.smoke_ok():
            return []  # the smoke run failed a path: nothing samples for hours on the full set (STATUS says so)
        gate = (lambda: True) if smoke else self.smoke_ok
        per_task = 30.0 if smoke else self.seconds_per_task()
        samplers = []
        for i, condition in enumerate(self.conditions(tag)):
            samplers.append(Job(f"{tag}:sample-{condition}",
                                self.py("vllm", "sample", "--run", r, "--condition", condition, "--until", until),
                                1, every(self.model(), gate, lambda c=condition: self.startable(tag, c, per_task)),
                                self.sampled(tag, condition), offset + i,
                                LOAD_SECONDS + per_task * self.tasks_left(tag, condition), stage=f"{tag}:sample",
                                retries=2))
        report = Job(f"{tag}:report", self.py("main", "report", "--run", r), 0,
                     lambda: all(self.given_up(j, per_task) for j in samplers), self.reported(tag), offset + 90, 600.0,
                     stage=f"{tag}:report", retries=1)
        return [*samplers, report]

    def progress_lines(self) -> list[str]:
        lines = []
        for condition in self.conditions("M"):
            if self.running(f"M:sample-{condition}"):
                p = read(self.folder("M", condition) / "progress.json") or {}
                if p:
                    lines.append(f"{condition} {p['done']}/{p['tasks']} tasks, {p['seconds_per_task']:.1f} s per task")
        return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--smoke-run", type=Path, required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--stop", required=True, help="absolute compute stop, ISO with offset")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--main-py", required=True)
    ap.add_argument("--vllm-py", required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--export-hook", default="")
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--only", default=None, help="comma-separated main-run models to sample on this host (default all)")
    args = ap.parse_args()
    stop = dt.datetime.fromisoformat(args.stop).timestamp()
    gpus = [int(g) for g in args.gpus.split(",")]
    plan = Plan(run=args.run, smoke_run=args.smoke_run, models_root=args.models_root, main_py=args.main_py,
                vllm_py=args.vllm_py, stop_epoch=stop, only=None if args.only is None else args.only.split(","))

    def export(label: str) -> None:
        if args.export_hook:
            done = subprocess.run(["bash", args.export_hook, label], capture_output=True, text=True, timeout=1800)
            if done.returncode != 0:
                scheduler.note(f"export {label} failed (exit {done.returncode}): {done.stderr.strip()[-200:]}")

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        if stage == "S:sample":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        if stage == "S:report" and not plan.smoke_ok():
            scheduler.note("the smoke run did not sample every model: the main run does not start (see the S: logs)")
        subprocess.run([args.main_py, "-m", "exec_feedback.passk", "progress", "--run", str(plan.runs[tag]), "--stage",
                        stage, "--commit", args.commit, "--file", str(args.progress)], cwd=args.repo,
                       capture_output=True, timeout=600)
        export(stage.replace(":", "-"))

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=120s", str(int(seconds)), *cmd]

    os.chdir(args.repo)
    scheduler = Scheduler(gpus=gpus, stop=stop, status=args.status, log_dir=args.log_dir, plan=plan,
                          external=lambda: not plan.model()(), reserve=SCHEDULER_RESERVE, wrap=wrap,
                          milestone=milestone, kill_before=KILL_BEFORE_STOP)
    plan.scheduler = scheduler
    scheduler.note(f"pass@k scheduler: GPUs {gpus}, samplers until "
                   f"{dt.datetime.fromtimestamp(stop - JOB_STOP_BEFORE_STOP, base.CST):%m-%d %H:%M}, stop {args.stop}, "
                   f"main-run models {'all' if plan.only is None else ','.join(plan.conditions('M'))}")
    last_export = time.time()
    while scheduler.step():
        if time.time() - last_export > EXPORT_EVERY and any(
                j["name"].startswith("M:sample-") for j in scheduler.running.values()):
            last_export = time.time()
            for line in plan.progress_lines():
                scheduler.note(f"sampling {line}")
            export(f"progress-{dt.datetime.now(base.CST):%H%M}")
        time.sleep(scheduler.poll)
    scheduler.note("scheduler: nothing running, nothing left that can start")
    if not plan.reported("M")():  # a report the schedule never got to: whatever time is left, up to the stop
        try:
            done = subprocess.run([args.main_py, "-m", "exec_feedback.passk", "report", "--run", str(plan.runs["M"])],
                                  cwd=args.repo, capture_output=True, text=True, timeout=max(300.0, stop - time.time()))
            if done.returncode != 0:
                scheduler.note(f"the report failed (exit {done.returncode}): {done.stderr.strip()[-300:]}")
        except subprocess.TimeoutExpired:
            scheduler.note("the report did not finish before the stop; the package has the samples")
        milestone("M:report")
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
