"""v0.8 scheduler for one 8-GPU host.

A smoke run (tiny subset, every job type, numbers never results) and the main run share one scheduler; main work on a
code path waits until the smoke run has exercised that path. Jobs start by priority when their dependencies hold, GPUs
are free and their estimated duration fits before the absolute stop.

Changes from v0.7: jobs can take several whole GPUs (the tensor-parallel teacher); a ready multi-GPU job holds its
block so smaller jobs stop refilling it; long student jobs keep off the teacher block while the teacher still has work;
estimates come from measured durations of finished jobs of the same kind (v0.7 left a 3 h estimate in place and idled
the host), and a job that no longer fits is reported once instead of silently skipped.
Plan: docs/RESEARCH_V08.md; operation: docs/H20_V08_RUNBOOK.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

CST = dt.timezone(dt.timedelta(hours=8))
ROLES = ("dev", "validation", "test")
KILL_BEFORE_STOP = 20 * 60      # every job is killed by then; the final report runs in this tail
JOB_STOP_BEFORE_STOP = 30 * 60  # samplers/evaluators start no new dialogues after this
NO_GPUS: Callable[[], set[int]] = lambda: set()


@dataclasses.dataclass
class Job:
    name: str
    cmd: list[str]
    slots: int                      # 0 = CPU, 1 = half GPU, 2 = full GPU (per GPU)
    deps: Callable[[], bool]
    done: Callable[[], bool]
    priority: int
    estimate: float                 # seconds; replaced by measured durations of the same kind when available
    stage: str = ""
    retries: int = 2
    width: int = 1                  # whole GPUs (tensor parallel); width > 1 implies slots == 2
    kind: str = ""
    avoid: Callable[[], set[int]] = NO_GPUS


@dataclasses.dataclass
class Group:
    """Claim-based samplers: one worker per free GPU (or GPU block) until every shard is claimed or the window ends."""
    name: str
    cmd: Callable[[int, str], list[str]]   # (worker number, sampler stop ISO) -> command
    deps: Callable[[], bool]
    claims_left: Callable[[], bool]
    finished: Callable[[], bool]           # DONE marker written
    close: Callable[[], None]              # writes the DONE marker
    priority: int
    minutes: float
    post_reserve: float                    # seconds the downstream work needs after sampling
    stage: str = ""
    width: int = 1
    max_failures: int = 0                  # 0 = twice the GPU count
    avoid: Callable[[], set[int]] = NO_GPUS


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


class Scheduler:
    def __init__(self, *, gpus: list[int], stop: float, status: Path, log_dir: Path,
                 plan: Callable[[], tuple[list[Job], list[Group]]], external: Callable[[], bool] = lambda: False,
                 reserve: float = 30 * 60, wrap: Callable[[list[str], float], list[str]] | None = None,
                 milestone: Callable[[str], None] | None = None, poll: float = 10.0, stagger: float = 150.0,
                 retry_delay: float = 60.0, kill_before: float = KILL_BEFORE_STOP) -> None:
        self.gpus, self.stop, self.status, self.log_dir, self.plan = gpus, stop, status, log_dir, plan
        self.kill_before = kill_before
        self.external, self.reserve, self.poll, self.stagger, self.retry_delay = external, reserve, poll, stagger, retry_delay
        self.wrap = wrap or (lambda cmd, seconds: cmd)
        self.milestone = milestone or (lambda stage: None)
        self.used = {g: 0 for g in gpus}
        self.last_start = {g: -1e9 for g in gpus}
        self.held: set[int] = set()
        self.running: dict[int, dict[str, Any]] = {}
        self.failures: dict[str, int] = {}
        self.retries: dict[str, int] = {}
        self.retry_after: dict[str, float] = {}
        self.window: dict[str, float] = {}
        self.group_workers: dict[str, int] = {}
        self.group_failures: dict[str, int] = {}
        self.broken_groups: set[str] = set()
        self.durations: dict[str, list[float]] = {}
        self.reported_late: set[str] = set()
        self.reached: set[str] = set()
        self.plan_errors, self.last_plan_error = 0, ""
        log_dir.mkdir(parents=True, exist_ok=True)

    def note(self, text: str) -> None:
        line = f"{dt.datetime.now(CST).strftime('%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        with self.status.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def left(self) -> float:
        return self.stop - time.time()

    def is_running(self, name: str) -> bool:
        return any(j["name"] == name for j in self.running.values())

    def dead(self, name: str) -> bool:
        return self.failures.get(name, 0) > self.retries.get(name, 2)

    def estimate(self, job: Job) -> float:
        seen = self.durations.get(job.kind) if job.kind else None
        return 1.15 * max(seen) + 60 if seen else job.estimate

    def free_gpus(self, slots: int, width: int, avoid: set[int]) -> list[int] | None:
        now = time.monotonic()
        ready = lambda g: g not in avoid and g not in self.held and (
            self.used[g] == 0 or now - self.last_start[g] >= self.stagger)
        if width == 1:  # from the high end, so the low block (the teacher's) stays free longest
            for g in reversed(self.gpus):
                if ready(g) and self.used[g] + slots <= 2:
                    return [g]
            return None
        for block in self.blocks(width):
            if all(ready(g) and self.used[g] == 0 for g in block):
                return block
        return None

    def blocks(self, width: int) -> list[list[int]]:
        return [self.gpus[i:i + width] for i in range(0, len(self.gpus) - width + 1, width)]

    def hold(self, width: int, avoid: set[int]) -> None:
        """Keep the least busy allowed block for a waiting multi-GPU job: nothing new starts there this pass."""
        candidates = [b for b in self.blocks(width) if not set(b) & avoid and not set(b) & self.held]
        if candidates:
            self.held |= set(min(candidates, key=lambda b: sum(self.used[g] for g in b)))

    def launch(self, name: str, cmd: list[str], slots: int, timeout: float, *, width: int = 1, group: str = "",
               kind: str = "", avoid: set[int] | None = None) -> bool:
        gpus = self.free_gpus(slots, width, avoid or set()) if slots else []
        if slots and gpus is None:
            return False
        env = {**os.environ}
        if slots:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        tries = self.failures.get(name, 0) + 1
        log = self.log_dir / f"{name.replace(':', '_')}.try{tries}.log"
        fh = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(self.wrap(cmd, max(60.0, timeout)), stdout=fh, stderr=subprocess.STDOUT, env=env)
        self.running[proc.pid] = {"name": name, "proc": proc, "gpus": gpus, "slots": slots, "fh": fh, "group": group,
                                  "kind": kind, "started": time.monotonic()}
        for g in gpus:
            self.used[g] += slots
            self.last_start[g] = time.monotonic()
        where = "CPU" if not slots else (f"GPU {gpus[0]} ({'full' if slots == 2 else 'half'})" if width == 1
                                         else f"GPUs {','.join(map(str, gpus))}")
        self.note(f"start {name} on {where}, try {tries}")
        return True

    def reap(self) -> None:
        for pid, job in list(self.running.items()):
            rc = job["proc"].poll()
            if rc is None:
                continue
            job["fh"].close()
            for g in job["gpus"]:
                self.used[g] -= job["slots"]
            del self.running[pid]
            minutes = (time.monotonic() - job["started"]) / 60
            self.note(f"done {job['name']} exit {rc} ({minutes:.1f} min)")
            if rc == 0:
                if job["kind"] and job["name"].startswith("M:"):
                    self.durations.setdefault(job["kind"], []).append(minutes * 60)
                continue
            if job["group"]:
                self.group_failures[job["group"]] = self.group_failures.get(job["group"], 0) + 1
            else:
                self.failures[job["name"]] = self.failures.get(job["name"], 0) + 1
                self.retry_after[job["name"]] = time.monotonic() + self.retry_delay

    def step(self) -> bool:
        """Start what can start; True while anything runs or may still become runnable."""
        self.reap()
        self.held = set()
        try:
            jobs, groups = self.plan()
            self.plan_errors = 0
        except Exception as exc:  # noqa: BLE001 - a bad artifact must not kill a night's schedule; retried next pass
            self.plan_errors += 1
            message = f"plan error ({type(exc).__name__}: {str(exc)[:200]}); retrying"
            if message != self.last_plan_error:
                self.last_plan_error = message
                self.note(message)
            return bool(self.running) or self.plan_errors < 30
        alive = bool(self.running) or self.external()
        for job in sorted([*jobs, *groups], key=lambda j: j.priority):  # stable: equal priorities keep plan order
            if isinstance(job, Group):
                alive |= self.run_group(job)
                continue
            self.retries[job.name] = job.retries
            if job.done() or self.is_running(job.name) or self.dead(job.name):
                continue
            if time.monotonic() < self.retry_after.get(job.name, 0):
                alive = True
                continue
            if not job.deps():
                continue  # becomes ready only through something running (counted above) or external downloads
            estimate = self.estimate(job)
            if time.time() + estimate > self.stop - self.reserve:
                if job.name not in self.reported_late:
                    self.reported_late.add(job.name)
                    self.note(f"not starting {job.name}: needs ~{estimate / 60:.0f} min, "
                              f"{max(0.0, self.left() - self.reserve) / 60:.0f} min left before the stop reserve")
                continue
            avoid = job.avoid()
            alive = True
            if not self.launch(job.name, job.cmd, job.slots, self.left() - self.kill_before, width=job.width,
                               kind=job.kind, avoid=avoid) and job.width > 1:
                self.hold(job.width, avoid)
        self.milestones(jobs)
        return alive

    def run_group(self, group: Group) -> bool:
        if group.finished():
            return False
        workers = [j for j in self.running.values() if j["group"] == group.name]
        if not group.deps():
            return bool(workers)
        now = time.time()
        if group.name not in self.window:
            end = min(now + group.minutes * 60, self.stop - group.post_reserve)
            self.window[group.name] = min(max(end, now + 15 * 60), self.stop - JOB_STOP_BEFORE_STOP)
            self.note(f"{group.name}: sampling window until {dt.datetime.fromtimestamp(self.window[group.name], CST):%H:%M}")
        window = self.window[group.name]
        limit = group.max_failures or 2 * len(self.gpus)
        broken = self.group_failures.get(group.name, 0) >= limit
        if broken:
            self.broken_groups.add(group.name)
        if group.claims_left() and now < window - 120 and not broken:
            avoid = group.avoid()
            while self.free_gpus(2, group.width, avoid) is not None:
                n = self.group_workers.get(group.name, 0) + 1
                self.group_workers[group.name] = n
                self.launch(f"{group.name}-w{n}", group.cmd(n, iso(window)), 2,
                            min(window - now + 1800, self.left() - self.kill_before), width=group.width,
                            group=group.name, avoid=avoid)
            if group.width > 1 and not [j for j in self.running.values() if j["group"] == group.name]:
                self.hold(group.width, avoid)
            return True
        if workers:
            return True
        group.close()
        self.note(f"{group.name}: sampling closed" + (" after repeated sampler failures" if broken else ""))
        self.milestone(group.stage or group.name)
        return True

    def milestones(self, jobs: list[Job]) -> None:
        stages: dict[str, list[Job]] = {}
        for job in jobs:
            if job.stage:
                stages.setdefault(job.stage, []).append(job)
        for stage, members in stages.items():
            if stage in self.reached or any(self.is_running(j.name) for j in members):
                continue
            if all(j.done() or self.dead(j.name) for j in members):
                self.reached.add(stage)
                self.milestone(stage)

    def loop(self) -> None:
        while self.step():
            time.sleep(self.poll)
        self.note("scheduler: nothing running, nothing left that can start")


# ------------------------------------------------------------------------------------------------ v0.8 plan
def exists(path: Path) -> Callable[[], bool]:
    return lambda: path.exists()


def every(*checks: Callable[[], bool]) -> Callable[[], bool]:
    return lambda: all(c() for c in checks)


def stage_of(condition: str) -> str:
    """Milestones per model family, so a path the gate closes (TR, PTR) never holds back the others' exports."""
    if condition == "B235":
        return "teacher"
    family = condition.split("_seed")[0]
    return "base" if family in ("B", "R", "P") else family


def read(path: Path) -> dict[str, Any] | None:
    """A JSON artifact, or None while it is missing or not readable (never crash the scheduler on a bad file)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gate_passed(run: Path) -> Callable[[], bool]:
    return lambda: bool((read(run / "gate.json") or {}).get("passed"))


def gate_failed(run: Path) -> bool:
    gate = read(run / "gate.json")
    return gate is not None and not gate.get("passed")


class Plan:
    """Jobs for the smoke run S and the main run M, recomputed from disk on every scheduler pass."""

    def __init__(self, *, run: Path, smoke_run: Path, models_root: Path, main_py: str, vllm_py: str,
                 stop_epoch: float, gpus: list[int]) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.settings = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8"))["settings"]
                         for tag, r in self.runs.items()}
        self.models_root, self.main_py, self.vllm_py, self.stop_epoch = models_root.resolve(), main_py, vllm_py, stop_epoch
        self.block = set(gpus[:self.settings["M"]["teacher_gpus"]])
        self.scheduler: Scheduler | None = None

    def model(self, kind: str) -> Callable[[], bool]:
        return exists(self.models_root / kind / ".efa-verified.json")

    def download_failed(self, kind: str) -> bool:
        rc = self.models_root / f"{kind}.download.rc"
        return rc.exists() and rc.read_text().strip() != "0"

    def downloads_pending(self) -> bool:
        return any(not (self.models_root / k / ".efa-verified.json").exists()
                   and not (self.models_root / f"{k}.download.rc").exists() for k in ("student", "teacher"))

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.v8", *a]

    def dead(self, name: str) -> bool:
        s = self.scheduler
        return s is not None and (s.dead(name) or name in s.broken_groups)

    def trained(self, tag: str, name: str) -> Callable[[], bool]:
        result = self.runs[tag] / "train" / name / "result.json"

        def check() -> bool:
            r = read(result) or {}
            return bool(r.get("training_completed") and r.get("adapter_reload_verified"))
        return check

    def evaluated(self, tag: str, role: str, condition: str) -> Callable[[], bool]:
        return exists(self.runs[tag] / "eval" / role / condition / "summary.json")

    def teacher_busy(self) -> bool:
        """True while the main run's teacher may still need its GPU block (download, pilot, gate, sampling)."""
        m = self.runs["M"]
        if (m / "samples" / "teacher" / "DONE.json").exists() or gate_failed(m) or self.download_failed("teacher"):
            return False
        if any(self.dead(n) for n in ("S:sample-teacher", "S:build-teacher", "M:eval-B235-dev", "M:gate",
                                      "M:sample-teacher")):
            return False
        if not (m / "samples" / "teacher" / "claims").exists() and self.stop_epoch - time.time() < 2.5 * 3600:
            return False  # too late for TR to finish: give the block back
        return True

    def keep_off_block(self) -> set[int]:
        return set(self.block) if self.teacher_busy() else set()

    def gates(self) -> dict[str, Callable[[], bool]]:
        """Main-run code paths, each released once the smoke run has completed the same path."""
        first, run = self.settings["S"]["seeds"][0], self.runs["S"]
        return {"sample-hint": exists(run / "data" / "hint.json"),
                "sample-teacher": exists(run / "data" / "teacher.json"),
                "eval-teacher": every(exists(run / "data" / "teacher.json"), self.evaluated("S", "dev", "B")),
                "train-student": lambda: any(self.trained("S", f"{f}_seed{first}")() for f in ("HR", "TR")),
                "prefer": lambda: any(self.trained("S", f"{f}_seed{first}")() for f in ("PHR", "PTR")),
                "eval-imported": self.evaluated("S", "dev", f"R_seed{first}"),
                "eval-trained": lambda: any(self.evaluated("S", "dev", f"{f}_seed{first}")() for f in ("HR", "TR"))}

    def __call__(self) -> tuple[list[Job], list[Group]]:
        jobs, groups = [], []
        for tag in ("S", "M"):
            j, g = self.for_run(tag)
            jobs += j
            groups += g
        return jobs, groups

    def for_run(self, tag: str) -> tuple[list[Job], list[Group]]:
        from .rollouts import sampling_state

        run, s = self.runs[tag], self.settings[tag]
        smoke = tag == "S"
        base = 0 if smoke else 100
        seeds = s["seeds"]
        r = str(run)
        jobs: list[Job] = []
        groups: list[Group] = []
        gate = (lambda name: (lambda: True)) if smoke else (lambda name, g=self.gates(): g[name])
        est = (lambda seconds: 600.0) if smoke else (lambda seconds: float(seconds))
        kind = (lambda k: "") if smoke else (lambda k: k)
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        roles = ("dev",) if smoke else ROLES
        width = s["teacher_gpus"]
        off_block = self.keep_off_block

        def evaluation(condition: str, role: str, deps: Callable[[], bool], priority: int, seconds: float,
                       path: str) -> None:
            teacher = condition == "B235"
            jobs.append(Job(f"{tag}:eval-{condition}-{role}",
                            self.py("vllm", "evaluate", "--run", r, "--condition", condition, "--role", role,
                                    "--stop", stop_iso),
                            2 if teacher else 1, every(deps, self.model("teacher" if teacher else "student"), gate(path)),
                            self.evaluated(tag, role, condition), base + priority, est(seconds),
                            stage=f"{tag}:eval-{stage_of(condition)}", retries=1 if teacher else 2,
                            width=width if teacher else 1,
                            kind=kind(f"eval-{'teacher' if teacher else 'student'}-{role}")))

        def sampler(name: str, model_kind: str, deps: Callable[[], bool], priority: int, minutes: float, post: float,
                    group_width: int, failures: int) -> None:
            marker = run / "samples" / name / "DONE.json"

            def close(name: str = name, marker: Path = marker) -> None:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps(sampling_state(run, name, s["sample_shards"])), encoding="utf-8")

            def claims_left(name: str = name) -> bool:
                claims = run / "samples" / name / "claims"
                return not claims.exists() or len(list(claims.iterdir())) < s["sample_shards"]

            groups.append(Group(
                f"{tag}:sample-{name}",
                lambda n, stop, name=name: self.py("vllm", "sample", "--run", r, "--sampler", name, "--stop", stop,
                                                   "--seed", str(20261013 + 1009 * n)),
                every(self.model(model_kind), deps), claims_left, exists(marker), close, base + priority, minutes,
                600 if smoke else post, stage=f"{tag}:sample-{name}", width=group_width, max_failures=failures))
            jobs.append(Job(f"{tag}:build-{name}", self.py("main", "build", "--run", r, "--sampler", name), 0,
                            exists(marker), exists(run / "data" / f"{name}.json"), base + priority + 1, est(600),
                            stage=f"{tag}:build-{name}", retries=1))

        # ---- the teacher: pilot on dev, gate, then sampling (main run); the smoke run only samples
        if not smoke:
            evaluation("B", "dev", lambda: True, 1, 900, "eval-imported")
            evaluation("B235", "dev", lambda: True, 5, 1800, "eval-teacher")
            jobs.append(Job("M:gate", self.py("main", "gate", "--run", r), 0,
                            every(self.evaluated("M", "dev", "B"), self.evaluated("M", "dev", "B235")),
                            exists(run / "gate.json"), base + 6, 120, stage="M:gate", retries=1))
            teacher_ok = every(gate_passed(run), gate("sample-teacher"))
        else:
            jobs.append(Job("S:gate", self.py("main", "gate", "--run", r), 0, lambda: True, exists(run / "gate.json"),
                            6, 120, stage="S:gate", retries=1))
            teacher_ok = lambda: True
        sampler("teacher", "teacher", teacher_ok, 7, s["teacher_minutes"], 2.5 * 3600, width, 2)
        sampler("hint", "student", gate("sample-hint"), 10, s["hint_minutes"], 3 * 3600, 1, 0)

        # ---- training: SFT on TR / HR, then the v0.5 preference stage on each (sized from the frozen data)
        for family, source, offset in (("TR", "teacher", 20), ("HR", "hint", 25)):
            data = run / "data" / f"{source}.json"
            examples = ((read(data) or {}).get("datasets") or {}).get(family, {}).get("examples", 0)
            for i, seed in enumerate(seeds):
                jobs.append(Job(f"{tag}:train-{family}_seed{seed}",
                                self.py("main", "train", "--run", r, "--family", family, "--seed", str(seed)),
                                2, every(exists(data), self.model("student"), gate("train-student")),
                                self.trained(tag, f"{family}_seed{seed}"), base + offset + i,
                                # v0.7 measured 0.98 s per example on one H20 (load and save included); HR and TR
                                # differ in size, so this stays size-based instead of copying the other family's time
                                est(300 + 1.1 * examples), stage=f"{tag}:train-{family}", avoid=off_block))
                jobs.append(Job(f"{tag}:prefer-P{family}_seed{seed}",
                                self.py("main", "prefer", "--run", r, "--family", f"P{family}", "--seed", str(seed)),
                                2, every(self.trained(tag, f"{family}_seed{seed}"), self.model("student"), gate("prefer")),
                                self.trained(tag, f"P{family}_seed{seed}"), base + 40, est(1200),
                                stage=f"{tag}:prefer-{family}", kind=kind("prefer"), avoid=off_block))

        # ---- evaluation
        for role in roles:
            p = 50 if role == "dev" else 60
            if smoke or role != "dev":
                evaluation("B", role, lambda: True, p + 5, 900, "eval-imported")
            for seed in seeds:
                for family in ("R", "P"):
                    evaluation(f"{family}_seed{seed}", role, lambda: True, p + 6, 900, "eval-imported")
                for family in ("TR", "HR", "PTR", "PHR"):
                    evaluation(f"{family}_seed{seed}", role, self.trained(tag, f"{family}_seed{seed}"), p, 900,
                               "eval-trained")
            if not smoke and role != "dev":  # the teacher's own score, for the report only; after its sampling
                evaluation("B235", role, every(gate_passed(run), exists(run / "samples" / "teacher" / "DONE.json")),
                           90, 1800, "eval-teacher")
        if smoke:  # one model on every role exercises the validation/test gold caches
            for role in ("validation", "test"):
                evaluation("B", role, lambda: True, 58, 900, "eval-imported")
            jobs.append(Job("S:report", self.py("main", "report", "--run", r, "--procs", "8"), 0,
                            self.evaluated("S", "dev", f"PHR_seed{seeds[0]}"), exists(run / "report" / "report.json"),
                            90, 900, stage="S:report", retries=1))
        return jobs, groups


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

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        if tag == "S" and stage != "S:report":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        subprocess.run([args.main_py, "-m", "exec_feedback.v8", "progress", "--run", str(plan.runs[tag]), "--stage",
                        stage, "--commit", args.commit, "--file", str(args.progress)], cwd=args.repo,
                       capture_output=True, timeout=600)
        if args.export_hook:
            subprocess.run(["bash", args.export_hook, stage.replace(":", "-")], capture_output=True, timeout=1800)

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=60s", str(int(seconds)), *cmd]

    os.chdir(args.repo)
    scheduler = Scheduler(gpus=gpus, stop=stop, status=args.status, log_dir=args.log_dir, plan=plan,
                          external=plan.downloads_pending, wrap=wrap, milestone=milestone)
    plan.scheduler = scheduler
    scheduler.note(f"v0.8 scheduler: GPUs {gpus}, teacher block {sorted(plan.block)}, stop {args.stop}")
    scheduler.loop()
    subprocess.run([args.main_py, "-m", "exec_feedback.v8", "report", "--run", str(plan.runs["M"]), "--procs", "96"],
                   cwd=args.repo, timeout=KILL_BEFORE_STOP - 120)
    milestone("M:report")
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
