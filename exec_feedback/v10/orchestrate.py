"""v0.10 scheduler for one 8-GPU host (the v0.8 Scheduler, a new plan).

Each booking trains the seeds given on the command line (all three in one 13-hour booking: every seed's RLF first, then
RLFa while time allows, so a late async run never takes the place of a sync one). Order: merge the
start points and the LoRA arm into full weights (CPU), the weights gate (one GPU), the smoke run S (tiny subset; RLF and
RLFa side by side on two 4-GPU blocks, each stopping once and resuming), then the main online runs on all eight GPUs,
then the evaluations at K, the layout benchmark (if it still fits) and the report; the smoke's own evaluations and report
last, on what is left. Main runs start only after both smoke runs completed, and only if their estimate (the steps still to run) fits
before `until` (stop - FINAL_RESERVE): a run stopped at the deadline cannot be resumed on another host. Each finished
main run's model at K is packed into the exports as soon as the run ends (CPU), so it can be fetched while the rest
still runs.
Plan: docs/RESEARCH_V10.md; operation: docs/H20_V10_RUNBOOK.md.
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
from . import FAMILIES, LORA, START
from .pack import pack_stem

KILL_BEFORE_STOP = 10 * 60           # every job is killed by stop - 10 min; the report runs in that tail
JOB_STOP_BEFORE_STOP = 12 * 60       # evaluations start no new dialogue after this
FINAL_RESERVE = 40 * 60              # online runs end this long before the stop: evaluations at K, report, export
SEED_RESERVE = 15 * 60               # and this much earlier per further seed: four more conditions in three roles
SCHEDULER_RESERVE = 15 * 60          # a job must fit before stop - this
EXPORT_EVERY = 30 * 60               # results package during the online phase (metrics, curves, STATUS)
ROLES = ("dev", "validation", "test")
ESTIMATES = {"dev": 300.0, "validation": 600.0, "test": 900.0}
REPORT_SECONDS = 600.0
BENCH_SECONDS = 1500.0
# Seconds per online step before anything was measured on this host: v0.9 measured 37-44 s per RL step on one H20
# (19 s sampling 128 dialogues, 15 s LoRA training); see docs/RESEARCH_V10.md for the v0.10 estimate.
STEP_SECONDS = {"RLF": 45.0, "RLFa": 35.0}
RUN_OVERHEAD = 900.0                 # engine and trainer start, curve points, milestones


class Plan:
    """Jobs for the smoke run S and the main run M, recomputed from disk on every scheduler pass."""

    def __init__(self, *, run: Path, smoke_run: Path, models_root: Path, main_py: str, vllm_py: str,
                 stop_epoch: float, gpus: list[int], seeds: list[int], exports: Path | None = None) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        manifests = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8")) for tag, r in self.runs.items()}
        self.settings = {tag: m["settings"] for tag, m in manifests.items()}
        self.identity = manifests["M"]["identity"]
        self.exports = exports.resolve() if exports else None
        if not set(seeds) <= set(self.settings["M"]["seeds"]):
            raise ValueError(f"Seeds {seeds} are not all registered in the main run")
        self.models_root, self.main_py, self.vllm_py, self.stop_epoch = models_root.resolve(), main_py, vllm_py, stop_epoch
        self.gpus, self.seeds = gpus, seeds
        self.until = stop_epoch - FINAL_RESERVE - SEED_RESERVE * (len(seeds) - 1)
        self.scheduler: Scheduler | None = None
        self.unfit: set[str] = set()

    def model(self) -> Callable[[], bool]:
        return exists(self.models_root / "student" / ".efa-verified.json")

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.v10", *a]

    def dead(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.dead(name)

    def running(self, name: str) -> bool:
        return self.scheduler is not None and self.scheduler.is_running(name)

    def merged(self, name: str) -> Callable[[], bool]:
        return exists(self.models_root / "merged" / f"{name}.json")

    def trained(self, tag: str, name: str) -> Callable[[], bool]:
        return lambda: bool((read(self.runs[tag] / "train" / name / "result.json") or {}).get("finished"))

    def evaluated(self, tag: str, role: str, condition: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "eval" / role / condition / "summary.json")

    def gate_passed(self) -> bool:
        return bool((read(self.runs["S"] / "gate" / "weights.json") or {}).get("passed"))

    def smoke_seed(self) -> int:
        return self.settings["S"]["seeds"][0]

    def online(self, tag: str) -> list[str]:
        seeds = [self.smoke_seed()] if tag == "S" else self.seeds
        return [f"{family}_seed{seed}" for family in FAMILIES for seed in seeds]

    def step_seconds(self, family: str) -> float:
        """1.1 x the measured mean step of this family's finished main runs on this host. The mean, since a run lasts
        the sum of its steps and sampling has a long tail (v0.9 RL_seed42: median step 36 s, mean 45 s). Before the
        async family has one, its steps are estimated from the sync runs' with sampling and training overlapped: the
        longer of the wait for samples and the rest of the step (the sync step's own length would keep RLFa from
        starting after a slow RLF although it needs much less). Before anything was measured, the default."""
        def rows(fam: str) -> list[dict[str, Any]]:
            return [r for folder in (self.runs["M"] / "train").glob(f"{fam}_seed*") if (folder / "result.json").exists()
                    for r in recover_records(folder / "metrics.jsonl") if r.get("step_seconds")]

        if own := rows(family):
            return 1.1 * statistics.fmean(r["step_seconds"] for r in own)
        if family != "RLF" and (sync := rows("RLF")):
            waits = [r.get("wait_seconds") or 0.0 for r in sync]
            return 1.1 * statistics.fmean(max(w, r["step_seconds"] - w) for w, r in zip(waits, sync))
        return STEP_SECONDS[family]

    def resume_step(self, name: str) -> int:
        """The step a restarted main run resumes from: its newest checkpoint (as restore picks it), 0 before one."""
        out = self.runs["M"] / "train" / name
        state = next((read(out / d / "state.json") for d in ("ckpt", "ckpt.old") if (out / d / "state.json").exists()),
                     None)
        return int((state or {}).get("step", 0))

    def run_estimate(self, name: str) -> float:
        """The steps still to run (a retry after a crash resumes from the checkpoint) plus engine start, curves and
        milestones."""
        family = name.split("_seed")[0]
        left = self.settings["M"]["planned_steps"] - self.resume_step(name)
        return left * self.step_seconds(family) + RUN_OVERHEAD

    def fits(self, name: str) -> Callable[[], bool]:
        """The run would end before `until`. A run that no longer does is noted once in STATUS: the scheduler's own
        "not starting" note never comes for it, since its dependencies are never all met. Only while no main run is
        running: until the one running ends, its steps are not in the estimate, which may still come down."""
        def check() -> bool:
            need, left = self.run_estimate(name), self.until - time.time()
            if need < left:
                return True
            if name not in self.unfit and self.scheduler is not None and not any(
                    self.running(f"M:run-{other}") for other in self.online("M")):
                self.unfit.add(name)
                self.scheduler.note(f"not starting M:run-{name}: needs ~{need / 60:.0f} min, "
                                    f"{max(0.0, left) / 60:.0f} min left before the online cutoff")
            return False
        return check

    def late(self, job: Job) -> bool:
        """The scheduler no longer starts this job: its estimate would end past the stop reserve."""
        s = self.scheduler
        return s is not None and time.time() + s.estimate(job) > s.stop - s.reserve

    def blocked(self) -> bool:
        """The main runs can never start here: the weights gate failed, or a smoke run gave up."""
        gate = read(self.runs["S"] / "gate" / "weights.json")
        return (gate is not None and not gate.get("passed")) or any(self.dead(f"S:run-{n}") for n in self.online("S"))

    def settled(self) -> bool:
        """Every main online run of this booking finished, gave up, can no longer start before `until`, or never can
        (then the evaluations of B, PTR and RL use the GPUs at once instead of waiting for `until`)."""
        for name in self.online("M"):
            job = f"M:run-{name}"
            if self.running(job):
                return False
            if not (self.trained("M", name)() or self.dead(job) or not self.fits(name)() or self.blocked()):
                return False
        return True

    def __call__(self) -> tuple[list[Job], list[Any]]:
        jobs: list[Job] = self.merges()
        for tag in ("S", "M"):
            jobs += self.for_run(tag)
        return jobs, []

    def merges(self) -> list[Job]:
        names = [f"{START}_seed{self.smoke_seed()}"] + [f"{f}_seed{s}" for s in self.seeds for f in (START, LORA)]
        m = str(self.runs["M"])
        return [Job(f"M:merge-{name}", self.py("main", "merge", "--run", m, "--name", name), 0, self.model(),
                    self.merged(name), 1, 900.0, stage="M:merge", retries=1)
                for name in dict.fromkeys(names)]

    def for_run(self, tag: str) -> list[Job]:
        run, s = self.runs[tag], self.settings[tag]
        smoke = tag == "S"
        r = str(run)
        jobs: list[Job] = []
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        first = self.smoke_seed()
        width = s["trainer_gpus"] + s["rollout_gpus"]
        smoke_runs = lambda: all(self.trained("S", n)() for n in self.online("S"))
        if smoke:
            jobs.append(Job("S:gate-weights", self.py("vllm", "gate", "--run", r), 2,
                            every(self.model(), self.merged(f"{START}_seed{first}")), exists(run / "gate" / "weights.json"),
                            5, 600.0, stage="S:gate", retries=1))

        # ---- online runs: the smoke pair side by side on two blocks, the main runs one at a time on all GPUs
        until = iso(min(self.until, time.time() + 2 * 3600)) if smoke else iso(self.until)
        for i, name in enumerate(self.online(tag)):
            family, seed = name.split("_seed")
            deps = [self.model(), self.merged(f"{START}_seed{seed}"), self.gate_passed]
            if not smoke:
                deps += [smoke_runs, self.fits(name)]
            # no `kind`: the scheduler would take a run's wall time as the next estimate, also for a run stopped at
            # the deadline or a retry that resumes halfway; run_estimate measures steps instead
            jobs.append(Job(f"{tag}:run-{name}",
                            self.py("main", "run", "--run", r, "--family", family, "--seed", seed, "--until", until,
                                    "--worker-python", self.vllm_py),
                            2, every(*deps), self.trained(tag, name), (10 if smoke else 100) + i,
                            1500.0 if smoke else self.run_estimate(name), stage=f"{tag}:run", retries=2, width=width))
            if not smoke and self.exports is not None:  # the host is recycled at the end of the booking
                manifest = self.exports / f"{pack_stem(self.identity, name, s['planned_steps'])}.json"
                jobs.append(Job(f"M:pack-{name}",
                                self.py("main", "pack", "--run", r, "--name", name, "--exports", str(self.exports)),
                                0, self.trained(tag, name), exists(manifest), 5, 900.0, retries=1))

        # ---- evaluations of full-weight models, after the online runs of this booking (they need every GPU). The
        # smoke's are engineering checks nobody can act on during the booking: they come last, on GPUs left idle,
        # and never hold up a main run
        evals: list[Job] = []

        def evaluation(condition: str, role: str, deps: Callable[[], bool]) -> None:
            evals.append(Job(f"{tag}:eval-{condition}-{role}",
                             self.py("vllm", "evaluate", "--run", r, "--condition", condition, "--role", role,
                                     "--stop", stop_iso),
                             1, every(deps, self.model(), self.settled), self.evaluated(tag, role, condition),
                             200 if smoke else 150, 600.0 if smoke else ESTIMATES[role], stage=f"{tag}:eval",
                             retries=1 if smoke else 2, kind="" if smoke else f"eval-{role}"))

        seeds = [first] if smoke else self.seeds
        # once the main runs are settled, a run that did not train in this booking never will: planning its
        # evaluations would keep the report (and the stage's export) waiting for jobs that can never start
        settled = not smoke and self.settled()
        for role in (("dev", "validation", "test") if smoke else ("test", "validation", "dev")):
            evaluation("B", role, lambda: True)
            if smoke and role != "dev":
                continue
            for seed in seeds:
                for fam in (START, LORA):
                    evaluation(f"{fam}_seed{seed}", role, self.merged(f"{fam}_seed{seed}"))
                for fam in FAMILIES:
                    name = f"{fam}_seed{seed}"
                    if not settled or self.trained(tag, name)():
                        evaluation(name, role, self.trained(tag, name))
        jobs += evals
        finished = lambda: all(j.done() or self.dead(j.name) for j in evals)

        # ---- the trainer-only layout benchmark on the first seed's main sync run, then the report. The evaluations
        # come first; after a main run that ended close to `until` they leave the benchmark too little time (a later
        # booking measures it then), and the report waits for it only while it can still start. The benchmark's
        # estimate counts the report after it: one that fits alone could still push the report past the stop reserve
        if not smoke:
            sync = self.trained(tag, f"RLF_seed{seeds[0]}")
            bench = Job("M:bench", self.py("main", "bench", "--run", r, "--family", "RLF", "--seed", str(seeds[0])),
                        2, every(sync, self.settled), exists(run / "bench" / "summary.json"), 160,
                        BENCH_SECONDS + REPORT_SECONDS, stage="M:bench", retries=1,
                        width=min(len(self.gpus), 2 * s["trainer_gpus"]))
            jobs.append(bench)
            done_evals = finished
            finished = lambda: done_evals() and (bench.done() or self.dead(bench.name) or (
                not self.running(bench.name) and (not sync() or self.late(bench))))
        report = self.py("main", "report", "--run", r, "--procs", "8" if smoke else "96")
        if not smoke:
            report += ["--smoke-run", str(self.runs["S"])]
        jobs.append(Job(f"{tag}:report", report, 0, every(self.settled, finished), exists(run / "report" / "report.json"),
                        210 if smoke else 190, REPORT_SECONDS, stage=f"{tag}:report", retries=1))
        return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--smoke-run", type=Path, required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--stop", required=True, help="absolute compute stop, ISO with offset")
    ap.add_argument("--seeds", required=True, help="seeds this booking trains, e.g. 42")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--main-py", required=True)
    ap.add_argument("--vllm-py", required=True)
    ap.add_argument("--status", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--export-hook", default="")
    ap.add_argument("--exports", type=Path, default=None, help="where finished main runs' models at K are packed")
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()
    stop = dt.datetime.fromisoformat(args.stop).timestamp()
    gpus = [int(g) for g in args.gpus.split(",")]
    plan = Plan(run=args.run, smoke_run=args.smoke_run, models_root=args.models_root, main_py=args.main_py,
                vllm_py=args.vllm_py, stop_epoch=stop, gpus=gpus, seeds=[int(x) for x in args.seeds.split(",")],
                exports=args.exports)

    def export(label: str) -> None:
        if args.export_hook:
            done = subprocess.run(["bash", args.export_hook, label], capture_output=True, text=True, timeout=1800)
            if done.returncode != 0:
                scheduler.note(f"export {label} failed (exit {done.returncode}): {done.stderr.strip()[-200:]}")

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        if tag == "S" and stage != "S:report":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        subprocess.run([args.main_py, "-m", "exec_feedback.v10", "progress", "--run", str(plan.runs[tag]), "--stage",
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
    scheduler.note(f"v0.10 scheduler: GPUs {gpus}, seeds {plan.seeds}, online runs until "
                   f"{dt.datetime.fromtimestamp(plan.until, base.CST):%m-%d %H:%M}, stop {args.stop}")
    last_export = time.time()
    while scheduler.step():
        if time.time() - last_export > EXPORT_EVERY and any(
                j["name"].startswith("M:run-") for j in scheduler.running.values()):
            last_export = time.time()
            export(f"progress-{dt.datetime.now(base.CST):%H%M}")
            scheduler.note("exported a progress package (online metrics and dev curves so far)")
        time.sleep(scheduler.poll)
    scheduler.note("scheduler: nothing running, nothing left that can start")
    if not (plan.runs["M"] / "report" / "report.json").exists():
        # whatever time is left, up to 5 minutes past the stop (the booking ends 25 minutes after it); a report that
        # still does not finish must not turn the whole booking into STOPPED without its final package
        try:
            subprocess.run([args.main_py, "-m", "exec_feedback.v10", "report", "--run", str(plan.runs["M"]), "--procs",
                            "96", "--smoke-run", str(plan.runs["S"])], cwd=args.repo,
                           timeout=max(300.0, stop + 300 - time.time()))
        except subprocess.TimeoutExpired:
            scheduler.note("the report did not finish before the stop; the final package has the evaluations")
        milestone("M:report")
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
