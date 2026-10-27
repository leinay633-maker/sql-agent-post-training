"""v0.12 scheduler for one 8-GPU host (the v0.8 Scheduler, a new plan). Plan: docs/RESEARCH_V12.md; operation:
docs/H20_V12_RUNBOOK.md.

GPUs: the first teacher_gpus are the teacher's block; of the rest, one per OPD seed (in seed order), then one for TROPD.
1. T:teacher-eval on the block: the teacher (B235) on the smoke dev subset, then the main dev and test sets, one load.
   Meanwhile the student GPUs except the smoke one evaluate B, TR and PTR.
2. T:serve on the block once the teacher evaluation is over: the scoring gate, then the service until every run that
   needs it has settled (the plan writes <teacher home>/stop) or the training cutoff passes.
3. Smoke OPD_seed42 (its GPU) and TROPD_seed42 (its GPU, low priority with the teacher): two steps, pause, resume to
   three; then their dev evaluations and the smoke report.
4. Main OPD runs once the smoke OPD resumed and ended; TROPD once its smoke did and the comparator evaluations are
   settled (from the moment the service is up those evaluations keep to TROPD's GPU). Training ends TRAIN_RESERVE before
   the stop; K, the evaluations at K on every free GPU (the block too once the service has stopped) and the report.
A main path whose smoke failed never starts; a job that no longer fits before the stop is reported once.
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
from . import ARMS, EXPLORE, FAMILY, TEACHER
from .online import ended, paused, run_name

KILL_BEFORE_STOP = 8 * 60            # every job is killed by stop - 8 min; a missing report runs in that tail
JOB_STOP_BEFORE_STOP = 10 * 60       # evaluations start no new dialogue after this
TRAIN_RESERVE = 45 * 60              # training ends this long before the stop (docs/RESEARCH_V12.md)
LATE = 30 * 60                       # a main run that has not started this close to its cutoff never starts
SCHEDULER_RESERVE = 10 * 60          # a job must fit before stop - this
EXPORT_EVERY = 30 * 60
ROLES = ("test", "validation", "dev")
ESTIMATES = {"dev": 300.0, "validation": 600.0, "test": 900.0}
TEACHER_EVAL_SECONDS = 40 * 60       # one load plus 16 + 615 + 1501 dialogues (v0.8: dev 5.2, validation 9.3 min)
SERVE_SECONDS = 15 * 60              # only for the fit check; the service runs until it is told to stop
TRAIN_SECONDS = 15 * 60


class Plan:
    """Jobs for the teacher (T), the smoke run S and the main run M, recomputed from disk on every pass."""

    def __init__(self, *, run: Path, smoke_run: Path, teacher_home: Path, models_root: Path, main_py: str,
                 vllm_py: str, stop_epoch: float, gpus: list[int]) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.settings = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8"))["settings"]
                         for tag, r in self.runs.items()}
        self.home, self.models_root = teacher_home.resolve(), models_root.resolve()
        self.main_py, self.vllm_py, self.stop_epoch, self.gpus = main_py, vllm_py, stop_epoch, gpus
        width = self.settings["M"]["teacher_gpus"]
        if len(gpus) <= width:
            raise ValueError(f"Need more than {width} GPUs: the teacher block plus at least one student GPU")
        self.block, self.rest = gpus[:width], gpus[width:]
        self.train_until = stop_epoch - TRAIN_RESERVE
        self.scheduler: Scheduler | None = None
        self.noted: set[str] = set()

    # ---- state
    def ready(self, kind: str) -> Callable[[], bool]:
        return exists(self.models_root / kind / ".efa-verified.json")

    def downloads_pending(self) -> bool:
        return any(not (self.models_root / k / ".efa-verified.json").exists()
                   and not (self.models_root / f"{k}.download.rc").exists() for k in ("student", "teacher"))

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.v12", *a]

    def dead(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.dead(name)

    def running(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.is_running(name)

    def serving(self) -> bool:
        return (self.home / "serving.json").exists()

    def gate_failed(self) -> bool:
        gate = read(self.home / "gate.json")
        return gate is not None and not gate.get("passed")

    def service_over(self) -> bool:
        """The service ended for good: stopped, its gate failed, or it died for the last time."""
        return (self.home / "served.json").exists() or self.gate_failed() or self.dead("T:serve")

    def teacher_jobs(self) -> list[tuple[Path, str]]:
        return [(self.runs["S"], "dev"), (self.runs["M"], "dev"), (self.runs["M"], "test")]

    def teacher_evaluated(self) -> bool:
        return all((run / "eval" / role / TEACHER / "summary.json").exists() for run, role in self.teacher_jobs())

    def teacher_eval_over(self) -> bool:
        return self.teacher_evaluated() or self.dead("T:teacher-eval")

    def evaluated(self, tag: str, role: str, condition: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "eval" / role / condition / "summary.json")

    def out(self, tag: str, family: str, seed: int) -> Path:
        return self.runs[tag] / "train" / run_name(family, seed)

    def members(self, tag: str) -> list[tuple[str, int]]:
        s = self.settings[tag]
        if tag == "S":
            return [(FAMILY, s["seeds"][0]), (EXPLORE, s["explore_seeds"][0])]
        return [(FAMILY, seed) for seed in s["seeds"]] + [(EXPLORE, seed) for seed in s["explore_seeds"]]

    def gpu_for(self, family: str, seed: int) -> int | None:
        """The OPD seeds take the first student GPUs in seed order, TROPD the next; None when there are too few."""
        m = self.settings["M"]
        index = m["seeds"].index(seed) if family == FAMILY else len(m["seeds"]) + m["explore_seeds"].index(seed)
        return self.rest[index] if index < len(self.rest) else None

    def only(self, gpu: int | None) -> set[int]:
        """avoid-set that pins a job to one GPU (or, without one, keeps it off the teacher's block)."""
        return set(self.gpus) - {gpu} if gpu is not None else set(self.block)

    def job_names(self, tag: str, family: str, seed: int) -> list[str]:
        name = run_name(family, seed)
        return [f"S:train-{name}-pause", f"S:train-{name}"] if tag == "S" else [f"M:train-{name}"]

    def smoke_trained(self, family: str) -> bool:
        """The smoke run of this family paused, resumed in a fresh process and reached its planned steps."""
        seed = self.settings["S"]["seeds"][0] if family == FAMILY else self.settings["S"]["explore_seeds"][0]
        record = ended(self.out("S", family, seed))
        return bool(record) and record["stopped_by"] == "planned" and bool(record.get("lora_check"))

    def smoke_failed(self, family: str) -> bool:
        seed = self.settings["S"]["seeds"][0] if family == FAMILY else self.settings["S"]["explore_seeds"][0]
        return not self.smoke_trained(family) and (
            any(self.dead(n) for n in self.job_names("S", family, seed)) or self.service_over()
            or ended(self.out("S", family, seed)) is not None)

    def run_settled(self, tag: str, family: str, seed: int) -> bool:
        """Will this run never (again) need the teacher?"""
        names = self.job_names(tag, family, seed)
        if any(self.running(n) for n in names):
            return False
        if ended(self.out(tag, family, seed)) is not None or self.service_over():
            return True
        if any(self.dead(n) for n in names):
            return True
        # a main run that is not running now never starts again this close to its cutoff (its job's deps say so),
        # whether it never started or a retry is due; its milestones so far are what it has
        if tag == "M" and (self.smoke_failed(family) or time.time() > self.train_until - LATE):
            return True
        return tag == "M" and family == EXPLORE and self.gpu_for(family, seed) is None

    def training_settled(self, tag: str) -> Callable[[], bool]:
        return lambda: all(self.run_settled(tag, f, s) for f, s in self.members(tag))

    def comparators(self, tag: str) -> list[str]:
        seeds = self.settings[tag]["seeds"][:1] if tag == "S" else self.settings[tag]["seeds"]
        return ["B", *(f"{arm}_seed{seed}" for arm in ARMS for seed in seeds)]

    def comparator_names(self, tag: str) -> list[str]:
        roles = ("dev",) if tag == "S" else ROLES
        return [f"{tag}:eval-{c}-{role}" for role in roles for c in self.comparators(tag)]

    def given_up(self, name: str, done: Callable[[], bool], estimate: float) -> bool:
        """Done, dead, or no longer startable before the stop reserve: nothing waits for it."""
        s = self.scheduler
        return done() or self.dead(name) or (
            s is not None and not self.running(name) and time.time() + estimate > s.stop - s.reserve)

    def comparators_settled(self) -> bool:
        for tag in ("S", "M"):
            roles = ("dev",) if tag == "S" else ROLES
            for role in roles:
                for c in self.comparators(tag):
                    name = f"{tag}:eval-{c}-{role}"
                    if self.running(name) or not self.given_up(name, self.evaluated(tag, role, c),
                                                               600.0 if tag == "S" else ESTIMATES[role]):
                        return False
        return True

    def eval_avoid(self) -> set[int]:
        """Before the service is up: the student GPUs but the smoke OPD's; while runs need the teacher: TROPD's GPU
        only (it starts when these are done); afterwards every GPU, the block once the service has stopped."""
        if self.training_settled("S")() and self.training_settled("M")():
            return set() if self.service_over() or not (self.serving() or self.running("T:serve")) else set(self.block)
        if not self.serving() and not self.service_over():
            return set(self.gpus) - set(self.rest[1:]) if len(self.rest) > 1 else set(self.block)
        tropd = self.gpu_for(EXPLORE, self.settings["M"]["explore_seeds"][0])
        return self.only(tropd if tropd is not None else self.rest[-1])

    def note_once(self, key: str, text: str) -> None:
        if key not in self.noted and self.scheduler is not None:
            self.noted.add(key)
            self.scheduler.note(text)

    # ---- the plan
    def __call__(self) -> tuple[list[Job], list[Any]]:
        gate = read(self.home / "gate.json")
        if gate is not None:  # each service attempt runs the gate again and rewrites the file
            self.note_once(f"gate-{gate.get('time')}",
                           f"teacher scoring gate {'passed' if gate.get('passed') else 'FAILED'}: mean |gap| "
                           f"{gate.get('mean_abs_gap')}, max {gate.get('max_abs_gap')} nat over "
                           f"{gate.get('sampled_tokens')} tokens, {gate.get('missing')} missing; load "
                           f"{(gate.get('engine_load_seconds') or 0) / 60:.1f} min, scoring {gate.get('score_seconds')} s")
        if self.dead("T:serve") and not (self.home / "over").exists():  # runs waiting for it stop at once
            self.home.mkdir(parents=True, exist_ok=True)
            (self.home / "over").write_text(dt.datetime.now(base.CST).isoformat(), encoding="utf-8")
            self.note_once("over", "the teacher service failed for the last time: the runs stop waiting for it")
        if self.training_settled("S")() and self.training_settled("M")() and not (self.home / "stop").exists() \
                and (self.serving() or self.running("T:serve")):
            self.home.mkdir(parents=True, exist_ok=True)
            (self.home / "stop").write_text(dt.datetime.now(base.CST).isoformat(), encoding="utf-8")
            self.note_once("stop", "every run that needs the teacher has settled: the service is told to stop")
        return self.teacher() + self.for_run("S") + self.for_run("M"), []

    def teacher(self) -> list[Job]:
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        width = self.settings["M"]["teacher_gpus"]
        avoid = lambda: set(self.gpus) - set(self.block)
        jobs = [Job("T:teacher-eval",
                    self.py("vllm", "teacher-eval", *(x for run, role in self.teacher_jobs()
                                                      for x in ("--job", str(run), role)), "--stop", stop_iso),
                    2, self.ready("teacher"), self.teacher_evaluated, 1, TEACHER_EVAL_SECONDS, stage="T:teacher-eval",
                    retries=1, width=width, avoid=avoid)]
        # never (re)started once nothing needs it: a service that died after the runs settled would hold the block
        stop = lambda: (self.home / "served.json").exists() or (self.home / "stop").exists() or self.gate_failed() \
            or (self.training_settled("S")() and self.training_settled("M")())
        jobs.append(Job("T:serve", self.py("vllm", "serve-teacher", "--run", str(self.runs["M"]), "--home",
                                           str(self.home), "--until", iso(self.train_until + 10 * 60)),
                        2, every(self.ready("teacher"), self.teacher_eval_over), stop, 2, SERVE_SECONDS,
                        stage="T:serve", retries=3, width=width, avoid=avoid))
        return jobs

    def for_run(self, tag: str) -> list[Job]:
        run, s = self.runs[tag], self.settings[tag]
        smoke = tag == "S"
        r = str(run)
        offset = 0 if smoke else 100
        jobs: list[Job] = []
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        until = iso(min(self.train_until, time.time() + 2 * 3600)) if smoke else iso(self.train_until)
        roles = ("dev",) if smoke else ROLES

        def evaluation(condition: str, role: str, deps: Callable[[], bool], priority: int, stage: str,
                       avoid: Callable[[], set[int]]) -> Job:
            return Job(f"{tag}:eval-{condition}-{role}",
                       self.py("vllm", "evaluate", "--run", r, "--condition", condition, "--role", role, "--stop",
                               stop_iso),
                       1, every(deps, self.ready("student")), self.evaluated(tag, role, condition), offset + priority,
                       600.0 if smoke else ESTIMATES[role], stage=f"{tag}:{stage}", retries=2,
                       kind="" if smoke else f"eval-{role}", avoid=avoid)

        # ---- B, TR and PTR (no training); the main ones once the smoke went through the evaluation path
        path_ok = (lambda: True) if smoke else every(*(self.evaluated("S", "dev", c) for c in self.comparators("S")))
        for role in roles:
            for condition in self.comparators(tag):
                jobs.append(evaluation(condition, role, path_ok, 3 if smoke else 30, "eval-arms", self.eval_avoid))

        # ---- training: smoke pause and resume, then the main runs
        for i, (family, seed) in enumerate(self.members(tag)):
            name, out = run_name(family, seed), self.out(tag, family, seed)
            gpu = self.gpu_for(family, seed)
            if gpu is None and family == EXPLORE:
                self.note_once(f"no-gpu-{tag}-{name}", f"{tag}:train-{name} not planned: no GPU left for it")
                continue
            train = ["train", "--run", r, "--family", family, "--seed", str(seed), "--until", until,
                     "--teacher-home", str(self.home), "--tag", tag]
            deps = [self.ready("student"), self.serving]
            if smoke:
                jobs.append(Job(f"S:train-{name}-pause",
                                self.py("vllm", *train, "--stop-step", str(s["planned_steps"] - 1)), 2, every(*deps),
                                lambda out=out: paused(out) is not None or ended(out) is not None, 10 + i,
                                TRAIN_SECONDS, stage="S:train", retries=1, avoid=lambda gpu=gpu: self.only(gpu)))
                deps.append(lambda out=out: paused(out) is not None)
                done = lambda out=out: ended(out) is not None
            else:
                deps += [lambda family=family: self.smoke_trained(family),
                         lambda: time.time() < self.train_until - LATE]
                if family == EXPLORE:
                    deps.append(self.comparators_settled)
                done = lambda out=out: ended(out) is not None
            jobs.append(Job(f"{tag}:train-{name}", self.py("vllm", *train), 2, every(*deps), done,
                            offset + 20 + i, TRAIN_SECONDS, stage=f"{tag}:train", retries=1 if smoke else 2,
                            avoid=lambda gpu=gpu: self.only(gpu)))

        # ---- K, the evaluations at K, the report
        steps = run / "steps.json"
        jobs.append(Job(f"{tag}:steps", self.py("main", "steps", "--run", r), 0, self.training_settled(tag),
                        exists(steps), offset + 30, 120.0, stage=f"{tag}:steps", retries=1))
        chosen = (read(steps) or {}).get("evaluate", {})
        k_evals = [evaluation(name, role, exists(steps), 40, "eval-k", self.eval_avoid)
                   for name in chosen for role in roles]
        jobs += k_evals
        waits = [(j.name, j.done, j.estimate) for j in jobs if j.name.startswith(f"{tag}:eval-")]
        over = lambda: self.teacher_eval_over() and all(self.given_up(n, d, e) for n, d, e in waits)
        jobs.append(Job(f"{tag}:report",
                        self.py("main", "report", "--run", r, "--procs", "8" if smoke else "96", "--teacher-home",
                                str(self.home)),
                        0, every(exists(steps), over), exists(run / "report" / "report.json"), offset + 50, 600.0,
                        stage=f"{tag}:report", retries=1))
        return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--smoke-run", type=Path, required=True)
    ap.add_argument("--teacher-home", type=Path, required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--stop", required=True, help="absolute compute stop, ISO with offset")
    ap.add_argument("--gpus", required=True)
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
    plan = Plan(run=args.run, smoke_run=args.smoke_run, teacher_home=args.teacher_home, models_root=args.models_root,
                main_py=args.main_py, vllm_py=args.vllm_py, stop_epoch=stop, gpus=gpus)

    def export(label: str) -> None:
        if args.export_hook:
            done = subprocess.run(["bash", args.export_hook, label], capture_output=True, text=True, timeout=1800)
            if done.returncode != 0:
                scheduler.note(f"export {label} failed (exit {done.returncode}): {done.stderr.strip()[-200:]}")

    def progress(stage: str) -> None:
        subprocess.run([args.main_py, "-m", "exec_feedback.v12", "progress", "--run", str(plan.runs["M"]), "--stage",
                        stage, "--commit", args.commit, "--file", str(args.progress), "--teacher-home",
                        str(plan.home)], cwd=args.repo, capture_output=True, timeout=600)

    def milestone(stage: str) -> None:
        if stage.startswith("S:") and stage != "S:report":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        if stage == "T:serve":
            gate = read(plan.home / "gate.json") or {}
            scheduler.note(f"teacher service over: gate {'passed' if gate.get('passed') else 'FAILED'} "
                           f"(mean |gap| {gate.get('mean_abs_gap')}), served: {read(plan.home / 'served.json')}")
        if stage.startswith(("M:", "T:")):
            progress(stage)
        export(stage.replace(":", "-"))

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=120s", str(int(seconds)), *cmd]

    os.chdir(args.repo)
    scheduler = Scheduler(gpus=gpus, stop=stop, status=args.status, log_dir=args.log_dir, plan=plan,
                          external=plan.downloads_pending, reserve=SCHEDULER_RESERVE, wrap=wrap, milestone=milestone,
                          kill_before=KILL_BEFORE_STOP)
    plan.scheduler = scheduler
    layout = {run_name(f, s): plan.gpu_for(f, s) for f, s in plan.members("M")}
    scheduler.note(f"v0.12 scheduler: teacher on GPUs {plan.block}, runs on {layout}; training until "
                   f"{dt.datetime.fromtimestamp(plan.train_until, base.CST):%m-%d %H:%M}, stop {args.stop}")
    last_export = time.time()
    while scheduler.step():
        if time.time() - last_export > EXPORT_EVERY and any(
                j["name"].startswith(("M:train-", "T:serve")) for j in scheduler.running.values()):
            last_export = time.time()
            export(f"progress-{dt.datetime.now(base.CST):%H%M}")
            scheduler.note("exported a progress package (online metrics, dev curves, teacher requests so far)")
        time.sleep(scheduler.poll)
    scheduler.note("scheduler: nothing running, nothing left that can start")
    if not (plan.runs["M"] / "report" / "report.json").exists():
        try:  # whatever time is left, up to 5 minutes past the stop (the booking ends 25 minutes after it)
            done = subprocess.run([args.main_py, "-m", "exec_feedback.v12", "report", "--run", str(plan.runs["M"]),
                                   "--procs", "96", "--teacher-home", str(plan.home)], cwd=args.repo,
                                  capture_output=True, text=True, timeout=max(300.0, stop + 300 - time.time()))
            if done.returncode != 0:
                scheduler.note(f"the report failed (exit {done.returncode}): {done.stderr.strip()[-300:]}")
        except subprocess.TimeoutExpired:
            scheduler.note("the report did not finish before the stop; the package has the evaluations")
        milestone("M:report")
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
