"""v0.11 scheduler for one H20 (the v0.8 Scheduler, a new plan).

In priority order on the one GPU: the whole smoke run S (16 tasks per role: pre-pass, two RLb steps, pause, resume to
three, every evaluation, both reports); then the main run M: B, PTR and RL9 evaluated, the three pre-passes, the three
boundary runs to the registered milestone K one after another, K, the evaluations at K and the preregistered report;
the v0.10 4% replay (optional, one GPU for minutes); then the exploratory continuation to explore_steps, its
evaluations and the final report. Evaluations and pre-passes take half the GPU (two at a time), training all of it.
Main work starts after the whole smoke run, on each code path only if the smoke went through it; the continuation
starts after the preregistered report and only if it fits, so a short booking keeps the preregistered part.
Plan: docs/RESEARCH_V11.md; operation: docs/H20_V11_RUNBOOK.md.
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

from ..storage import recover_records
from ..v8 import orchestrate as base
from ..v8.orchestrate import Job, Scheduler, every, exists, iso, read
from . import ARM, FAMILY, START
from .online import ended, paused, run_name, trained_to

KILL_BEFORE_STOP = 10 * 60           # every job is killed by stop - 10 min; a missing report runs in that tail
JOB_STOP_BEFORE_STOP = 12 * 60       # evaluations start no new dialogue after this
# Runs to K end this long before the stop: K, then nine evaluations two at a time (about 55 min at the measured
# 1.15 x duration + 1 min the scheduler plans with, 150 s apart on the GPU), the report, its export, and the
# scheduler's own reserve below; continuations end a little earlier than strictly needed for the same reason.
MAIN_RESERVE = 120 * 60
EXPLORE_RESERVE = 110 * 60
SCHEDULER_RESERVE = 15 * 60          # a job must fit before stop - this
EXPORT_EVERY = 30 * 60               # results package during the long phases (pre-pass rows, metrics, curves, STATUS)
ROLES = ("test", "validation", "dev")
ESTIMATES = {"dev": 300.0, "validation": 600.0, "test": 900.0}
# Seconds per online step before this host measured one: v0.9's RL took 182-185 min for 200 steps on one H20,
# dev curve points included.
STEP_SECONDS = 56.0
RUN_OVERHEAD = 900.0                 # engine and trainer start, curve points, milestones
REGROUP_SECONDS = 1800.0


class Plan:
    """Jobs for the smoke run S, the main run M and the optional 4% replay R, recomputed from disk on every pass."""

    def __init__(self, *, run: Path, smoke_run: Path, models_root: Path, main_py: str, vllm_py: str,
                 stop_epoch: float, gpus: list[int], regroup: dict[str, Path] | None = None) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.settings = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8"))["settings"]
                         for tag, r in self.runs.items()}
        self.models_root, self.main_py, self.vllm_py, self.stop_epoch = models_root.resolve(), main_py, vllm_py, stop_epoch
        self.gpus, self.regroup = gpus, regroup
        self.main_until = stop_epoch - MAIN_RESERVE
        self.explore_until = stop_epoch - EXPLORE_RESERVE
        self.scheduler: Scheduler | None = None
        self.unfit: set[str] = set()

    def model(self) -> Callable[[], bool]:
        return exists(self.models_root / "student" / ".efa-verified.json")

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.v11", *a]

    def dead(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.dead(name)

    def running(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.is_running(name)

    def seeds(self, tag: str) -> list[int]:
        return self.settings[tag]["seeds"][:1] if tag == "S" else self.settings[tag]["seeds"]

    def out(self, tag: str, seed: int) -> Path:
        return self.runs[tag] / "train" / run_name(seed)

    def prepass(self, tag: str, seed: int) -> dict[str, Any] | None:
        return read(self.runs[tag] / "prepass" / f"{START}_seed{seed}" / "result.json")

    def gate_ok(self, tag: str, seed: int) -> Callable[[], bool]:
        return lambda: bool(((self.prepass(tag, seed) or {}).get("gate") or {}).get("passed"))

    def gate_failed(self, tag: str, seed: int) -> bool:
        result = self.prepass(tag, seed)
        return result is not None and not result["gate"]["passed"]

    def at_k(self, tag: str, seed: int) -> Callable[[], bool]:
        """The run is done with the registered part: paused at K, or ended (a deadline also ends it)."""
        return lambda: ended(self.out(tag, seed)) is not None or \
            trained_to(self.out(tag, seed)) >= self.settings[tag]["planned_steps"]

    def held(self, tag: str, seed: int) -> bool:
        """Paused at K, waiting for its continuation."""
        return ended(self.out(tag, seed)) is None and paused(self.out(tag, seed)) is not None

    def explored(self, tag: str, seed: int) -> bool:
        """The continuation reached explore_steps (not cut short by a deadline)."""
        record = ended(self.out(tag, seed))
        return bool(record) and record["steps_completed"] == self.settings[tag]["explore_steps"] \
            and record["stopped_by"] == "planned"

    def evaluated(self, tag: str, role: str, condition: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "eval" / role / condition / "summary.json")

    def reported(self, tag: str, name: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "report" / f"{name}.json")

    def smoke_over(self) -> bool:
        """The smoke run went all the way through: its final report written, or given up on."""
        return self.reported("S", "final")() or self.dead("S:report-final")

    def smoke(self, path: str) -> Callable[[], bool]:
        """Main work on a code path starts once the smoke run is over and went through that path itself: a path that
        failed on 16 tasks is not tried again for hours on the full sets (the STATUS line alerts the operator)."""
        seed = self.seeds("S")[0]
        name = run_name(seed)
        checks = {
            "eval": every(*(self.evaluated("S", "dev", c) for c in ("B", f"{START}_seed{seed}", f"{ARM}_seed{seed}"))),
            "prepass": exists(self.runs["S"] / "prepass" / f"{START}_seed{seed}" / "result.json"),
            "train": lambda: paused(self.out("S", seed)) is not None,
            "eval-k": self.evaluated("S", "dev", name),
            "explore": lambda: self.explored("S", seed),
            "eval-explore": self.evaluated("S", "dev", f"{name}@{self.settings['S']['explore_steps']}")}
        return every(self.smoke_over, checks[path])

    def smoke_failed(self, path: str) -> bool:
        """The smoke run is over without having gone through this path: main work on it never starts."""
        return self.smoke_over() and not self.smoke(path)()

    def late(self) -> bool:
        return time.time() > self.main_until - 30 * 60

    def settled(self, tag: str) -> Callable[[], bool]:
        """Every run of the tag reached K, gave up, cannot train (its pre-pass failed its gate or gave up) or, main
        only, can no longer start before the cutoff or never will (the smoke run failed that path)."""
        def check() -> bool:
            blocked = tag == "M" and (self.late() or self.smoke_failed("prepass") or self.smoke_failed("train"))
            for seed in self.seeds(tag):
                train, pre = f"{tag}:train-{run_name(seed)}", f"{tag}:prepass-{START}_seed{seed}"
                if self.running(train) or self.running(pre):
                    return False
                if not (self.at_k(tag, seed)() or self.dead(train) or self.dead(pre) or self.gate_failed(tag, seed)
                        or blocked):
                    return False
            return True
        return check

    def step_seconds(self) -> float:
        """1.1 x the mean step of the main runs on this host (the mean: a run lasts the sum of its steps), else v0.9's."""
        rows = [r for seed in self.seeds("M") if (self.out("M", seed) / "metrics.jsonl").exists()
                for r in recover_records(self.out("M", seed) / "metrics.jsonl") if r.get("step_seconds")]
        return 1.1 * statistics.fmean(r["step_seconds"] for r in rows) if rows else STEP_SECONDS

    def resume_step(self, tag: str, seed: int) -> int:
        out = self.out(tag, seed)
        state = next((read(out / d / "state.json") for d in ("ckpt", "ckpt.old") if (out / d / "state.json").exists()),
                     None)
        return max(int((state or {}).get("step", 0)), trained_to(out))

    def explore_need(self, seed: int) -> float:
        """Seconds a continuation still needs: its steps left from its newest checkpoint, and the start-up."""
        left_steps = self.settings["M"]["explore_steps"] - self.resume_step("M", seed)
        return max(0, left_steps) * self.step_seconds() + RUN_OVERHEAD

    def queued_need(self, seed: int) -> float:
        """Its own need plus that of the continuations queued before it on the GPUs (the running one included) that
        will run: the continuations take a whole GPU each and start in seed order."""
        order = self.seeds("M")
        total = self.explore_need(seed)
        for other in order[:order.index(seed)]:
            job = f"M:explore-{run_name(other)}"
            if self.held("M", other) and not self.dead(job) and (
                    self.running(job) or self.queued_need(other) < self.explore_until - time.time()):
                total += self.explore_need(other) / len(self.gpus)
        return total

    def fits(self, seed: int) -> Callable[[], bool]:
        """The continuation would end before its cutoff after those queued before it; one that no longer does is
        noted once in STATUS."""
        def check() -> bool:
            need, left = self.queued_need(seed), self.explore_until - time.time()
            if need < left:
                return True
            name = run_name(seed)
            if name not in self.unfit and self.scheduler is not None and not any(
                    self.running(f"M:explore-{run_name(s)}") for s in self.seeds("M")):
                self.unfit.add(name)
                self.scheduler.note(f"not starting M:explore-{name}: needs ~{need / 60:.0f} min, "
                                    f"{max(0.0, left) / 60:.0f} min left before the exploratory cutoff")
            return False
        return check

    def given_up(self, job: Job) -> bool:
        """Done, dead, or no longer started at all (its estimate would end past the stop reserve): a report does not
        wait for it inside the schedule."""
        s = self.scheduler
        return job.done() or self.dead(job.name) or (
            s is not None and not self.running(job.name) and time.time() + s.estimate(job) > s.stop - s.reserve)

    def explore_settled(self, tag: str) -> Callable[[], bool]:
        """Every continuation ended, gave up, cannot start (no run paused at K, the smoke's continuation failed) or no
        longer fits."""
        def check() -> bool:
            for seed in self.seeds(tag):
                job = f"{tag}:explore-{run_name(seed)}"
                if self.running(job) or self.running(f"{tag}:train-{run_name(seed)}"):
                    return False
                if self.held(tag, seed) and not self.dead(job) and (
                        tag == "S" or (not self.smoke_failed("explore") and self.fits(seed)())):
                    return False
            return True
        return check

    def regroup_command(self) -> list[str]:
        """v0.10's first kept step of RLF_seed42 replayed on this GPU as 4, 8, 2 and 16 ranks would split it."""
        g = self.regroup or {}
        return [self.main_py, "-m", "torch.distributed.run", "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
                "-m", "exec_feedback.v10.bench", "--run", str(g.get("run")), "--family", "RLF", "--seed", "42",
                "--regroup", "4,8,2,16", "--start", str(g.get("start")), "--out", str(g.get("out"))]

    def __call__(self) -> tuple[list[Job], list[Any]]:
        jobs: list[Job] = []
        for tag in ("S", "M"):
            jobs += self.for_run(tag)
        return jobs, []

    def for_run(self, tag: str) -> list[Job]:
        run, s = self.runs[tag], self.settings[tag]
        smoke = tag == "S"
        r = str(run)
        offset = 0 if smoke else 100
        jobs: list[Job] = []
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        gate = (lambda path: lambda: True) if smoke else self.smoke
        main_until = iso(min(self.main_until, time.time() + 2 * 3600)) if smoke else iso(self.main_until)
        explore_until = iso(min(self.explore_until, time.time() + 2 * 3600)) if smoke else iso(self.explore_until)
        enough = (lambda: True) if smoke else (lambda: not self.late())
        steps = run / "steps.json"
        evals: list[Job] = []

        def evaluation(condition: str, role: str, path: str, priority: int, stage: str, into: list[Job]) -> None:
            if not smoke and self.smoke_failed(path):
                return  # the smoke run failed this path: no report waits for an evaluation that never starts
            into.append(Job(f"{tag}:eval-{condition}-{role}",
                            self.py("vllm", "evaluate", "--run", r, "--condition", condition, "--role", role,
                                    "--stop", stop_iso),
                            1, every(gate(path), self.model()), self.evaluated(tag, role, condition), priority,
                            600.0 if smoke else ESTIMATES[role], stage=f"{tag}:{stage}", retries=2,
                            kind="" if smoke else f"eval-{role}"))

        # ---- the arms that need no training: B, the PTR start points and v0.9's RL (RL9)
        for role in (("dev", "validation", "test") if smoke else ROLES):
            evaluation("B", role, "eval", offset + 3, "eval-start", evals)
            if smoke and role != "dev":
                continue
            for seed in self.seeds(tag):
                evaluation(f"{START}_seed{seed}", role, "eval", offset + 3, "eval-start", evals)
                evaluation(f"{ARM}_seed{seed}", role, "eval", offset + 3, "eval-start", evals)

        # ---- pre-passes (half a GPU each), then the runs to K one at a time on the whole GPU
        for i, seed in enumerate(self.seeds(tag)):
            name = f"{START}_seed{seed}"
            jobs.append(Job(f"{tag}:prepass-{name}",
                            self.py("vllm", "prepass", "--run", r, "--seed", str(seed), "--until", main_until),
                            1, every(self.model(), gate("prepass"), enough),
                            exists(run / "prepass" / name / "result.json"), offset + 10 + i,
                            600.0 if smoke else 1200.0, stage=f"{tag}:prepass", retries=2))
        for i, seed in enumerate(self.seeds(tag)):
            name = run_name(seed)
            jobs.append(Job(f"{tag}:train-{name}",
                            self.py("vllm", "train", "--run", r, "--seed", str(seed), "--leg", "main", "--until",
                                    main_until),
                            2, every(self.model(), gate("train"), enough, self.gate_ok(tag, seed)), self.at_k(tag, seed),
                            offset + 20 + i, 600.0 if smoke else 1200.0, stage=f"{tag}:train", retries=2))
        over = (lambda: True) if smoke else self.smoke_over
        jobs.append(Job(f"{tag}:steps", self.py("main", "steps", "--run", r), 0, every(over, self.settled(tag)),
                        exists(steps), offset + 30, 120.0, stage=f"{tag}:steps", retries=1))

        # ---- the boundary runs at K (runs without a milestone get no evaluation job), the preregistered report
        chosen = (read(steps) or {}).get("evaluate", {})
        for seed in self.seeds(tag):
            if run_name(seed) in chosen:
                for role in (("dev",) if smoke else ROLES):
                    evaluation(run_name(seed), role, "eval-k", offset + 40, "eval-k", evals)
        jobs += evals
        main_done = self.reported(tag, "main")
        # what comes after the preregistered report also comes if that report gave up (it can be rebuilt from the
        # exported evaluations; the continuation cannot)
        main_over = lambda: main_done() or self.dead(f"{tag}:report-main")
        jobs.append(Job(f"{tag}:report-main",
                        self.py("main", "report", "--run", r, "--name", "main", "--procs", "8" if smoke else "96"),
                        0, every(over, exists(steps), lambda: all(self.given_up(j) for j in evals)),
                        main_done, offset + 50, 600.0, stage=f"{tag}:report-main", retries=1))
        if not smoke and self.regroup:
            g = self.regroup
            jobs.append(Job("R:regroup", self.regroup_command(), 2,
                            every(main_over, exists(g["start"].with_name(g["start"].name + ".json"))),
                            exists(g["out"]), offset + 60, REGROUP_SECONDS, stage="R:regroup", retries=1))

        # ---- exploratory: each run paused at K continues to explore_steps (if it fits), then its evaluations
        late_evals: list[Job] = []
        for i, seed in enumerate(self.seeds(tag)):
            name = run_name(seed)
            deps = [self.model(), main_over, gate("explore"), lambda seed=seed: self.held(tag, seed)]
            if not smoke:
                deps.append(self.fits(seed))
            jobs.append(Job(f"{tag}:explore-{name}",
                            self.py("vllm", "train", "--run", r, "--seed", str(seed), "--leg", "explore", "--until",
                                    explore_until),
                            2, every(*deps), lambda seed=seed: ended(self.out(tag, seed)) is not None, offset + 70 + i,
                            600.0 if smoke else 1200.0, stage=f"{tag}:explore", retries=2))
            if self.explored(tag, seed):
                for role in (("dev",) if smoke else ROLES):
                    evaluation(f"{name}@{s['explore_steps']}", role, "eval-explore", offset + 80,
                               "eval-explore", late_evals)
        jobs += late_evals
        jobs.append(Job(f"{tag}:report-final",
                        self.py("main", "report", "--run", r, "--name", "final", "--procs", "8" if smoke else "96"),
                        0, every(main_over, self.explore_settled(tag),
                                 lambda: all(self.given_up(j) for j in late_evals)),
                        self.reported(tag, "final"), offset + 90, 600.0, stage=f"{tag}:report-final", retries=1))
        return jobs


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
    ap.add_argument("--regroup-run", type=Path, default=None, help="v0.10's final run package (the 4% replay)")
    ap.add_argument("--regroup-start", type=Path, default=None, help="merged PTR_seed42 (v0.10 merge)")
    ap.add_argument("--regroup-out", type=Path, default=None)
    args = ap.parse_args()
    stop = dt.datetime.fromisoformat(args.stop).timestamp()
    gpus = [int(g) for g in args.gpus.split(",")]
    regroup = None
    if args.regroup_run and args.regroup_start and args.regroup_out:
        regroup = {"run": args.regroup_run.resolve(), "start": args.regroup_start.resolve(),
                   "out": args.regroup_out.resolve()}
    plan = Plan(run=args.run, smoke_run=args.smoke_run, models_root=args.models_root, main_py=args.main_py,
                vllm_py=args.vllm_py, stop_epoch=stop, gpus=gpus, regroup=regroup)

    def export(label: str) -> None:
        if args.export_hook:
            done = subprocess.run(["bash", args.export_hook, label], capture_output=True, text=True, timeout=1800)
            if done.returncode != 0:
                scheduler.note(f"export {label} failed (exit {done.returncode}): {done.stderr.strip()[-200:]}")

    def progress(tag: str, stage: str) -> None:
        subprocess.run([args.main_py, "-m", "exec_feedback.v11", "progress", "--run", str(plan.runs[tag]), "--stage",
                        stage, "--commit", args.commit, "--file", str(args.progress)], cwd=args.repo,
                       capture_output=True, timeout=600)

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        if tag == "S" and stage != "S:report-final":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        if tag in plan.runs:
            progress(tag, stage)
        if stage == "R:regroup":
            regroup_note()
        export(stage.replace(":", "-"))

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=120s", str(int(seconds)), *cmd]

    def report(name: str) -> None:
        """Whatever time is left, up to 5 minutes past the stop (the booking ends 25 minutes after it)."""
        try:
            done = subprocess.run([args.main_py, "-m", "exec_feedback.v11", "report", "--run", str(plan.runs["M"]),
                                   "--name", name, "--procs", "96"], cwd=args.repo, capture_output=True, text=True,
                                  timeout=max(300.0, stop + 300 - time.time()))
            if done.returncode != 0:
                scheduler.note(f"the {name} report failed (exit {done.returncode}): {done.stderr.strip()[-300:]}")
        except subprocess.TimeoutExpired:
            scheduler.note(f"the {name} report did not finish before the stop; the package has the evaluations")
        milestone(f"M:report-{name}")

    def regroup_note() -> None:
        """The 4% replay in one STATUS line: whether the start matched v0.10's, and the norms by split."""
        result = read(regroup["out"]) if regroup else None
        if not result:
            return
        start = result.get("start") or {}
        norms = ", ".join(f"{r['world']} ranks {'/'.join(f'{n:.4f}' for n in r['grad_norms'])}"
                          + (f" (layout {r['layout_first_grad_norm']:.4f})" if r.get("layout_first_grad_norm") else "")
                          for r in result.get("rows", []))
        scheduler.note(f"4% replay: start {'matches' if start.get('same') else 'DIFFERS from'} v0.10's (sum delta "
                       f"{start.get('sum_delta', 0):.3g}, squares relative delta "
                       f"{start.get('squares_relative_delta', 0):.3g}); first-step grad norm by split: {norms}")

    os.chdir(args.repo)
    scheduler = Scheduler(gpus=gpus, stop=stop, status=args.status, log_dir=args.log_dir, plan=plan,
                          external=lambda: not plan.model()(), reserve=SCHEDULER_RESERVE, wrap=wrap,
                          milestone=milestone, kill_before=KILL_BEFORE_STOP)
    plan.scheduler = scheduler
    scheduler.note(f"v0.11 scheduler: GPUs {gpus}, runs to K until "
                   f"{dt.datetime.fromtimestamp(plan.main_until, base.CST):%m-%d %H:%M}, continuations until "
                   f"{dt.datetime.fromtimestamp(plan.explore_until, base.CST):%m-%d %H:%M}, stop {args.stop}"
                   f"{', with the 4% replay' if regroup else ''}")
    last_export = time.time()
    long_phase = ("M:prepass-", "M:train-", "M:explore-")
    while scheduler.step():
        if time.time() - last_export > EXPORT_EVERY and any(
                j["name"].startswith(long_phase) for j in scheduler.running.values()):
            last_export = time.time()
            export(f"progress-{dt.datetime.now(base.CST):%H%M}")
            scheduler.note("exported a progress package (pre-pass rows, online metrics and dev curves so far)")
        time.sleep(scheduler.poll)
    scheduler.note("scheduler: nothing running, nothing left that can start")
    report_dir = plan.runs["M"] / "report"
    for name in ("main", "final"):  # the final one also when no continuation ran: report.* is then the fuller one
        if not (report_dir / f"{name}.json").exists():
            report(name)
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
