"""v0.7 scheduler for one 8-GPU host.

A smoke run (tiny subset, every job type, numbers never results) and the main run share one scheduler. Main work on a
code path waits until the smoke run has exercised that path; everything else follows artifacts on disk, so a restart
resumes. Jobs start by priority when their dependencies hold, a GPU slot is free and their estimated duration fits
before the absolute stop. Plan: docs/RESEARCH_V07.md; operation: docs/H20_V07_RUNBOOK.md.
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


@dataclasses.dataclass
class Job:
    name: str
    cmd: list[str]
    slots: int                      # 0 = CPU, 1 = half GPU, 2 = full GPU
    deps: Callable[[], bool]
    done: Callable[[], bool]
    priority: int
    estimate: float                 # seconds; not started unless it fits before stop - reserve
    stage: str = ""
    retries: int = 2


@dataclasses.dataclass
class Group:
    """Claim-based samplers: one full-GPU worker per free GPU until every shard is claimed or the window closes."""
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


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


class Scheduler:
    def __init__(self, *, gpus: list[int], stop: float, status: Path, log_dir: Path,
                 plan: Callable[[], tuple[list[Job], list[Group]]], external: Callable[[], bool] = lambda: False,
                 reserve: float = 30 * 60, wrap: Callable[[list[str], float], list[str]] | None = None,
                 milestone: Callable[[str], None] | None = None, poll: float = 10.0, stagger: float = 90.0,
                 retry_delay: float = 60.0) -> None:
        self.gpus, self.stop, self.status, self.log_dir, self.plan = gpus, stop, status, log_dir, plan
        self.external, self.reserve, self.poll, self.stagger, self.retry_delay = external, reserve, poll, stagger, retry_delay
        self.wrap = wrap or (lambda cmd, seconds: cmd)
        self.milestone = milestone or (lambda stage: None)
        self.used = {g: 0 for g in gpus}
        self.last_start = {g: -1e9 for g in gpus}
        self.running: dict[int, dict[str, Any]] = {}
        self.failures: dict[str, int] = {}
        self.retry_after: dict[str, float] = {}
        self.window: dict[str, float] = {}
        self.group_workers: dict[str, int] = {}
        self.group_failures: dict[str, int] = {}
        self.reached: set[str] = set()
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

    def free_gpu(self, slots: int) -> int | None:
        now = time.monotonic()
        for g in self.gpus:
            if self.used[g] + slots <= 2 and (self.used[g] == 0 or now - self.last_start[g] >= self.stagger):
                return g
        return None

    def launch(self, name: str, cmd: list[str], slots: int, timeout: float, group: str = "") -> bool:
        gpu = self.free_gpu(slots) if slots else None
        if slots and gpu is None:
            return False
        env = {**os.environ}
        if slots:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        tries = self.failures.get(name, 0) + 1
        log = self.log_dir / f"{name.replace(':', '_')}.try{tries}.log"
        fh = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(self.wrap(cmd, max(60.0, timeout)), stdout=fh, stderr=subprocess.STDOUT, env=env)
        self.running[proc.pid] = {"name": name, "proc": proc, "gpu": gpu, "slots": slots, "fh": fh, "group": group,
                                  "started": time.monotonic()}
        if slots:
            self.used[gpu] += slots
            self.last_start[gpu] = time.monotonic()
        self.note(f"start {name} on " + (f"GPU {gpu} ({'full' if slots == 2 else 'half'})" if slots else "CPU")
                  + f", try {tries}")
        return True

    def reap(self) -> None:
        for pid, job in list(self.running.items()):
            rc = job["proc"].poll()
            if rc is None:
                continue
            job["fh"].close()
            if job["slots"]:
                self.used[job["gpu"]] -= job["slots"]
            del self.running[pid]
            self.note(f"done {job['name']} exit {rc} ({(time.monotonic() - job['started']) / 60:.1f} min)")
            if rc == 0:
                continue
            if job["group"]:
                self.group_failures[job["group"]] = self.group_failures.get(job["group"], 0) + 1
            else:
                self.failures[job["name"]] = self.failures.get(job["name"], 0) + 1
                self.retry_after[job["name"]] = time.monotonic() + self.retry_delay

    def step(self) -> bool:
        """Start what can start; True while anything runs or may still become runnable."""
        self.reap()
        jobs, groups = self.plan()
        alive = bool(self.running) or self.external()
        for job in sorted([*jobs, *groups], key=lambda j: j.priority):  # stable: equal priorities keep plan order
            if isinstance(job, Group):
                alive |= self.run_group(job)
                continue
            if job.done() or self.is_running(job.name) or self.failures.get(job.name, 0) > job.retries:
                continue
            if time.monotonic() < self.retry_after.get(job.name, 0):
                alive = True
                continue
            if not job.deps():
                continue  # becomes ready only through something running (counted above) or external downloads
            if time.time() + job.estimate > self.stop - self.reserve:
                continue
            if self.launch(job.name, job.cmd, job.slots, self.left() - KILL_BEFORE_STOP):
                alive = True
            else:
                alive = True  # waits for a GPU slot
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
        broken = self.group_failures.get(group.name, 0) > 2 * len(self.gpus)
        if group.claims_left() and now < window - 120 and not broken:
            while (gpu := self.free_gpu(2)) is not None:
                n = self.group_workers.get(group.name, 0) + 1
                self.group_workers[group.name] = n
                self.launch(f"{group.name}-w{n}", group.cmd(n, iso(window)), 2,
                            min(window - now + 1800, self.left() - KILL_BEFORE_STOP), group=group.name)
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
            if all(j.done() or self.failures.get(j.name, 0) > j.retries for j in members):
                self.reached.add(stage)
                self.milestone(stage)

    def loop(self) -> None:
        while self.step():
            time.sleep(self.poll)
        self.note("scheduler: nothing running, nothing left that can start")


# ------------------------------------------------------------------------------------------------ v0.7 plan
def exists(path: Path) -> Callable[[], bool]:
    return lambda: path.exists()


def every(*checks: Callable[[], bool]) -> Callable[[], bool]:
    return lambda: all(c() for c in checks)


class Plan:
    """Jobs for the smoke run S and the main run M, recomputed from disk on every scheduler pass."""

    def __init__(self, *, run: Path, smoke_run: Path, models_root: Path, main_py: str, vllm_py: str,
                 stop_epoch: float) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.settings = {tag: json.loads((r / "manifest.json").read_text(encoding="utf-8"))["settings"]
                         for tag, r in self.runs.items()}
        self.models_root, self.main_py, self.vllm_py, self.stop_epoch = models_root.resolve(), main_py, vllm_py, stop_epoch

    def model(self, kind: str) -> Callable[[], bool]:
        return exists(self.models_root / kind / ".efa-verified.json")

    def downloads_pending(self) -> bool:
        return any(not (self.models_root / k / ".efa-verified.json").exists()
                   and not (self.models_root / f"{k}.download.rc").exists() for k in ("student", "scale", "teacher"))

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.v7", *a]

    def trained(self, tag: str, name: str) -> Callable[[], bool]:
        result = self.runs[tag] / "train" / name / "result.json"

        def check() -> bool:
            if not result.exists():
                return False
            r = json.loads(result.read_text(encoding="utf-8"))
            return bool(r.get("training_completed") and r.get("adapter_reload_verified"))
        return check

    def evaluated(self, tag: str, role: str, condition: str) -> Callable[[], bool]:
        from .rollouts import eval_shards
        out = self.runs[tag] / "eval" / role / condition
        shards = eval_shards(self.settings[tag], condition)
        return lambda: all((out / f"shard{i:02d}.summary.json").exists() for i in range(shards))

    def gates(self) -> dict[str, Callable[[], bool]]:
        """Main-run code paths, each released once the smoke run has completed the same path."""
        s = self.settings["S"]
        first, scale_first, run = s["seeds"][0], s["scale_seeds"][0], self.runs["S"]
        return {"sample-scale": exists(run / "data" / "scale.json"),
                "sample-teacher": exists(run / "data" / "teacher.json"),
                "train-student": self.trained("S", f"T_seed{first}"),
                "train-scale": self.trained("S", f"R14_seed{scale_first}"),
                "prefer": lambda: any(self.trained("S", f"{p}_seed{first}")() for p in ("PT", "PTR")),
                "eval-imported": self.evaluated("S", "dev", f"R_seed{first}"),
                "eval-trained": self.evaluated("S", "dev", f"T_seed{first}"),
                "eval-scale": self.evaluated("S", "dev", f"R14_seed{scale_first}"),
                "eval-teacher": self.evaluated("S", "dev", "B32"),
                "eval-vote": lambda: any((run / "eval" / "validation").glob("*+vote/shard00.summary.json"))}

    def __call__(self) -> tuple[list[Job], list[Group]]:
        jobs, groups = [], []
        for tag in ("S", "M"):
            j, g = self.for_run(tag)
            jobs += j
            groups += g
        return jobs, groups

    def for_run(self, tag: str) -> tuple[list[Job], list[Group]]:
        from .rollouts import eval_shards, parse, sampling_state

        run, s = self.runs[tag], self.settings[tag]
        smoke = tag == "S"
        base = 0 if smoke else 100
        seeds, raw_seeds, scale_seeds = s["seeds"], s["raw_seeds"], s["scale_seeds"]
        r = str(run)
        jobs: list[Job] = []
        groups: list[Group] = []
        gate = (lambda name: (lambda: True)) if smoke else (lambda name, g=self.gates(): g[name])
        est = (lambda seconds: 600.0) if smoke else (lambda seconds: float(seconds))
        stop_iso = iso(self.stop_epoch - JOB_STOP_BEFORE_STOP)
        roles = ("dev",) if smoke else ROLES

        def evaluation(condition: str, role: str, deps: Callable[[], bool], priority: int, seconds: float,
                       path: str) -> None:
            kind, _, _ = parse(condition)
            for shard in range(eval_shards(s, condition)):
                out = run / "eval" / role / condition / f"shard{shard:02d}.summary.json"
                jobs.append(Job(f"{tag}:eval-{condition}-{role}-{shard}",
                                self.py("vllm", "evaluate", "--run", r, "--condition", condition, "--role", role,
                                        "--shard", str(shard), "--stop", stop_iso),
                                1 if kind == "student" else 2, every(deps, self.model(kind), gate(path)), exists(out),
                                base + priority, est(seconds), stage=f"{tag}:eval-{role}"))

        # ---- sampling: the 14B first (its training is the long pole), then the teacher
        for kind, priority, minutes, post in (("scale", 10, s["scale_minutes"], 4 * 3600),
                                              ("teacher", 20, s["teacher_minutes"], 2.5 * 3600)):
            marker = run / "samples" / kind / "DONE.json"

            def close(kind: str = kind, marker: Path = marker) -> None:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps(sampling_state(run, kind, s["sample_shards"])), encoding="utf-8")

            def claims_left(kind: str = kind) -> bool:
                claims = run / "samples" / kind / "claims"
                return not claims.exists() or len(list(claims.iterdir())) < s["sample_shards"]

            groups.append(Group(
                f"{tag}:sample-{kind}",
                lambda n, stop, kind=kind: self.py("vllm", "sample", "--run", r, "--kind", kind, "--stop", stop,
                                                   "--seed", str(20261012 + 1009 * n)),
                every(self.model(kind), gate(f"sample-{kind}")), claims_left, exists(marker), close, base + priority,
                minutes, 600 if smoke else post, stage=f"{tag}:sample-{kind}"))
            jobs.append(Job(f"{tag}:build-{kind}", self.py("main", "build", "--run", r, "--kind", kind), 0,
                            exists(marker), exists(run / "data" / f"{kind}.json"), base + priority + 1, est(900),
                            stage=f"{tag}:build-{kind}", retries=1))
        teacher_data, scale_data = exists(run / "data" / "teacher.json"), exists(run / "data" / "scale.json")

        # ---- training
        scale_ok = every(scale_data, self.model("scale"), gate("train-scale"))
        jobs.append(Job(f"{tag}:train-R14_seed{scale_seeds[0]}",
                        self.py("main", "train", "--run", r, "--family", "R14", "--seed", str(scale_seeds[0])),
                        2, scale_ok, self.trained(tag, f"R14_seed{scale_seeds[0]}"), base + 30, est(3.2 * 3600),
                        stage=f"{tag}:train-R14"))
        teacher_closed = exists(run / "samples" / "teacher" / "DONE.json")
        for seed in scale_seeds[1:]:  # extra 14B seeds: after the teacher sampling, and only if they fit (registered)
            jobs.append(Job(f"{tag}:train-R14_seed{seed}",
                            self.py("main", "train", "--run", r, "--family", "R14", "--seed", str(seed)),
                            2, every(scale_ok, teacher_closed), self.trained(tag, f"R14_seed{seed}"), base + 70,
                            est(3.2 * 3600), stage=f"{tag}:train-R14-extra"))
        student_ok = every(teacher_data, self.model("student"), gate("train-student"))
        for family, seed_list, seconds in (("T", seeds, 3600), ("TR", seeds, 4800), ("Traw", raw_seeds, 3600)):
            for i, seed in enumerate(seed_list):
                jobs.append(Job(f"{tag}:train-{family}_seed{seed}",
                                self.py("main", "train", "--run", r, "--family", family, "--seed", str(seed)),
                                2, student_ok, self.trained(tag, f"{family}_seed{seed}"), base + 40 + i, est(seconds),
                                stage=f"{tag}:train-4b"))

        # ---- evaluation
        for role in roles:
            p = 50 if role == "dev" else 60
            evaluation("B", role, lambda: True, p + 5, 900, "eval-imported")
            for seed in seeds:
                for family in ("R", "P"):
                    evaluation(f"{family}_seed{seed}", role, lambda: True, p + 6, 900, "eval-imported")
                for family in ("T", "TR"):
                    evaluation(f"{family}_seed{seed}", role, self.trained(tag, f"{family}_seed{seed}"), p, 900,
                               "eval-trained")
            for seed in raw_seeds:
                evaluation(f"Traw_seed{seed}", role, self.trained(tag, f"Traw_seed{seed}"), p + 7, 900, "eval-trained")
            evaluation("B14", role, lambda: True, p + 8, 2400, "eval-scale")
            evaluation("B32", role, lambda: True, p + 8, 3000, "eval-teacher")
            for seed in scale_seeds:
                evaluation(f"R14_seed{seed}", role, self.trained(tag, f"R14_seed{seed}"), p + 9, 2400, "eval-scale")
        if smoke:  # one model on every role exercises the validation/test gold caches
            for role in ("validation", "test"):
                evaluation("B", role, lambda: True, 58, 900, "eval-imported")

        # ---- second stage on the dev-selected distilled family, then the final family's vote
        dev_tx = every(*[self.evaluated(tag, "dev", f"{f}_seed{x}") for f in ("T", "TR") for x in seeds])
        choice, final = run / "select_x.json", run / "select_final.json"
        jobs.append(Job(f"{tag}:select-x", self.py("main", "select-x", "--run", r), 0, dev_tx, exists(choice),
                        base + 52, est(120), stage=f"{tag}:select-x", retries=1))
        if choice.exists():
            px = {"T": "PT", "TR": "PTR"}[json.loads(choice.read_text(encoding="utf-8"))["x"]]
            for seed in seeds:
                jobs.append(Job(f"{tag}:prefer-{px}_seed{seed}",
                                self.py("main", "prefer", "--run", r, "--family", px, "--seed", str(seed)),
                                2, every(self.model("student"), gate("prefer")), self.trained(tag, f"{px}_seed{seed}"),
                                base + 53, est(2400), stage=f"{tag}:prefer"))
                for role in roles:
                    evaluation(f"{px}_seed{seed}", role, self.trained(tag, f"{px}_seed{seed}"),
                               54 if role == "dev" else 61, 900, "eval-trained")
            candidates = [self.evaluated(tag, "dev", f"{f}_seed{x}") for f in ("R", "P", "T", "TR", px) for x in seeds]
            jobs.append(Job(f"{tag}:select-final", self.py("main", "select-final", "--run", r), 0, every(*candidates),
                            exists(final), base + 56, est(120), stage=f"{tag}:select-final", retries=1))
        if final.exists():
            head = json.loads(final.read_text(encoding="utf-8"))["head"]
            for role in ("validation",) if smoke else ("validation", "test"):
                evaluation(f"{head}+vote", role, lambda: True, 80, 2400, "eval-vote")
            if smoke:
                vote_done = self.evaluated("S", "validation", f"{head}+vote")
                jobs.append(Job("S:report", self.py("main", "report", "--run", r, "--procs", "8"), 0, vote_done,
                                exists(run / "report" / "report.json"), 90, 900, stage="S:report", retries=1))
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
    plan = Plan(run=args.run, smoke_run=args.smoke_run, models_root=args.models_root, main_py=args.main_py,
                vllm_py=args.vllm_py, stop_epoch=stop)

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        if tag == "S" and stage != "S:report":
            return  # smoke details stay in STATUS; one progress entry when the smoke run is reported
        subprocess.run([args.main_py, "-m", "exec_feedback.v7", "progress", "--run", str(plan.runs[tag]), "--stage",
                        stage, "--commit", args.commit, "--file", str(args.progress)], cwd=args.repo,
                       capture_output=True, timeout=600)
        if args.export_hook:
            subprocess.run(["bash", args.export_hook, stage.replace(":", "-")], capture_output=True, timeout=1800)

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=60s", str(int(seconds)), *cmd]

    os.chdir(args.repo)
    scheduler = Scheduler(gpus=[int(g) for g in args.gpus.split(",")], stop=stop, status=args.status,
                          log_dir=args.log_dir, plan=plan, external=plan.downloads_pending, wrap=wrap,
                          milestone=milestone)
    scheduler.note(f"v0.7 scheduler: GPUs {scheduler.gpus}, stop {args.stop}")
    scheduler.loop()
    subprocess.run([args.main_py, "-m", "exec_feedback.v7", "report", "--run", str(plan.runs["M"]), "--procs", "96"],
                   cwd=args.repo, timeout=KILL_BEFORE_STOP - 120)
    milestone("M:report")
    scheduler.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
