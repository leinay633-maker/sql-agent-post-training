"""Deployment-study driver for one H20: every step in order on the whole GPU, the smoke run S first (16 tasks per role,
8 load-test tasks at two concurrency levels), then the main run M. Steps: merge the adapter (CPU, shared by both runs),
the greedy evaluations (three deployments on each role), the load tests (one deployment at a time), the report. A main
step runs only if the smoke run went through the same kind of step, and only if its estimate fits before the stop. Each
try of a step is limited to twice its estimate (at least 30 minutes) and told to start no new work 5 minutes before
that; after every try, servers a killed load test left running are stopped. Plan: docs/RESEARCH_DEPLOY.md; operation:
docs/H20_DEPLOY_RUNBOOK.md.
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
from typing import Callable

from ..v8.orchestrate import CST, exists, iso
from . import CONDITIONS

KILL_BEFORE_STOP = 10 * 60
JOB_STOP_BEFORE_STOP = 15 * 60       # evaluations and load tests start no new work after this
RESERVE = 15 * 60                    # a step must fit before stop - this
ESTIMATES = {"merge": 1200.0, "evaluate": 900.0, "bench": 3600.0, "report": 900.0}
SMOKE_ESTIMATE = 900.0
BENCH_ORDER = ("merged", "fp8", "lora")
MIN_LIMIT = 30 * 60                  # a try's limit: twice its estimate, at least this
WIND_DOWN = 5 * 60                   # a try starts no new work this long before its limit


@dataclasses.dataclass
class Step:
    name: str
    key: str                         # kind and deployment ("evaluate:fp8"): a smoke failure blocks the same key in M
    cmd: Callable[[str], list[str]]  # given the ISO time after which the try starts no new work
    done: Callable[[], bool]
    estimate: float
    needs: tuple[str, ...] = ()      # keys whose failure (in either run) makes this step pointless
    retries: int = 1


class Driver:
    def __init__(self, *, run: Path, smoke_run: Path, main_py: str, vllm_py: str, stop: float, status: Path,
                 log_dir: Path, wrap: Callable[[list[str], float], list[str]] | None = None,
                 milestone: Callable[[str], None] | None = None) -> None:
        self.runs = {"S": smoke_run.resolve(), "M": run.resolve()}
        self.settings = {t: json.loads((r / "manifest.json").read_text(encoding="utf-8"))["settings"]
                         for t, r in self.runs.items()}
        self.main_py, self.vllm_py, self.stop, self.status, self.log_dir = main_py, vllm_py, stop, status, log_dir
        self.wrap = wrap or (lambda cmd, seconds: cmd)
        self.milestone = milestone or (lambda stage: None)
        self.failed: set[tuple[str, str]] = set()
        log_dir.mkdir(parents=True, exist_ok=True)

    def note(self, text: str) -> None:
        line = f"{dt.datetime.now(CST).strftime('%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        with self.status.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def py(self, venv: str, *a: str) -> list[str]:
        return [self.vllm_py if venv == "vllm" else self.main_py, "-m", "exec_feedback.deploy", *a]

    def steps(self, tag: str) -> list[Step]:
        run, s = self.runs[tag], self.settings[tag]
        r = str(run)
        est = (lambda kind: SMOKE_ESTIMATE) if tag == "S" else (lambda kind: ESTIMATES[kind])
        needs = lambda condition: () if condition == "lora" else ("merge",)
        out = [Step(f"{tag}:merge", "merge", lambda until: self.py("main", "merge", "--run", r), self.merged(tag),
                    est("merge"))]
        for role in s["roles"]:
            for condition in CONDITIONS:
                out.append(Step(f"{tag}:eval-{condition}-{role}", f"evaluate:{condition}",
                                lambda until, c=condition, o=role: self.py("vllm", "evaluate", "--run", r, "--condition",
                                                                           c, "--role", o, "--stop", until),
                                exists(run / "eval" / role / condition / "summary.json"), est("evaluate"),
                                needs(condition)))
        for condition in BENCH_ORDER:
            out.append(Step(f"{tag}:bench-{condition}", f"bench:{condition}",
                            lambda until, c=condition: self.py("main", "bench", "--run", r, "--condition", c, "--until",
                                                               until, "--vllm-py", self.vllm_py),
                            exists(run / "bench" / condition / "result.json"), est("bench"), needs(condition)))
        out.append(Step(f"{tag}:report", "report", lambda until: self.py("main", "report", "--run", r),
                        exists(run / "report" / "report.json"), est("report")))
        return out

    def merged(self, tag: str) -> Callable[[], bool]:
        def check() -> bool:
            from .freeze import load, merged_model
            try:
                merged_model(self.runs[tag], load(self.runs[tag], check_code=False)[0])
                return True
            except (OSError, ValueError, KeyError):
                return False
        return check

    def execute(self, step: Step) -> bool:
        for attempt in range(1, step.retries + 2):
            left = self.stop - KILL_BEFORE_STOP - time.time()
            if left < 60:
                break
            limit = min(left, max(2 * step.estimate, MIN_LIMIT))
            until = iso(min(self.stop - JOB_STOP_BEFORE_STOP, time.time() + limit - WIND_DOWN))
            log = self.log_dir / f"{step.name.replace(':', '_')}.log"  # appended: a restarted driver keeps old tries
            self.note(f"start {step.name}, try {attempt}, limit {limit / 60:.0f} min")
            started = time.monotonic()
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"===== {dt.datetime.now(CST):%m-%d %H:%M:%S} try {attempt}, limit {limit / 60:.0f} min\n")
                fh.flush()
                rc = subprocess.run(self.wrap(step.cmd(until), limit), stdout=fh, stderr=subprocess.STDOUT).returncode
            self.note(f"done {step.name} exit {rc} ({(time.monotonic() - started) / 60:.1f} min)")
            self.reap()
            if rc == 0 and step.done():
                return True
        return False

    def reap(self) -> None:
        """Servers a killed load test or demo left behind take the GPU and the ports from the next step: stop them."""
        from .bench import reap_servers

        for run in self.runs.values():
            try:
                for line in reap_servers(run):
                    self.note(line)
            except (OSError, ValueError, KeyError) as exc:
                self.note(f"WARNING: could not check {run.name}'s servers ({type(exc).__name__}: {exc})")

    def blocked(self, tag: str, step: Step) -> str:
        if tag == "M" and ("S", step.key) in self.failed:
            return f"the smoke run failed {step.key}"
        lost = [n for n in step.needs if (tag, n) in self.failed or ("S", n) in self.failed]
        return f"{', '.join(lost)} failed" if lost else ""

    def loop(self) -> None:
        self.reap()  # a driver started again after its host process died
        for tag in ("S", "M"):
            evaluated = False
            for step in self.steps(tag):
                if tag == "M" and not evaluated and not step.key.startswith(("merge", "evaluate")):
                    evaluated = True
                    self.milestone("M:eval")  # a package with every evaluation before the long load tests
                if step.done():
                    continue
                reason = self.blocked(tag, step)
                if reason:
                    self.failed.add((tag, step.key))
                    self.note(f"not starting {step.name}: {reason}")
                    continue
                if time.time() + step.estimate > self.stop - RESERVE:
                    self.note(f"not starting {step.name}: needs ~{step.estimate / 60:.0f} min, "
                              f"{max(0.0, self.stop - RESERVE - time.time()) / 60:.0f} min left before the stop reserve")
                    continue
                if not self.execute(step):
                    self.failed.add((tag, step.key))
                if step.name == "S:report" or (tag == "M" and step.key.split(":")[0] in ("bench", "report")):
                    self.milestone(step.name)
        self.note("driver: every step done or given up")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--smoke-run", type=Path, required=True)
    ap.add_argument("--stop", required=True, help="absolute compute stop, ISO with offset")
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

    def milestone(stage: str) -> None:
        tag = stage.split(":", 1)[0]
        # bounded by the stop (results are pulled in the 25 minutes after it), at least a minute each
        budget = lambda most: max(60.0, min(most, stop + 10 * 60 - time.time()))
        try:
            if stage in ("S:report", "M:report"):
                subprocess.run([args.main_py, "-m", "exec_feedback.deploy", "progress", "--run", str(driver.runs[tag]),
                                "--stage", stage, "--commit", args.commit, "--file", str(args.progress)],
                               cwd=args.repo, capture_output=True, timeout=budget(600))
            if args.export_hook:
                done = subprocess.run(["bash", args.export_hook, stage.replace(":", "-")], capture_output=True,
                                      text=True, timeout=budget(1800))
                if done.returncode != 0:
                    driver.note(f"export {stage} failed (exit {done.returncode}): {done.stderr.strip()[-200:]}")
        except subprocess.TimeoutExpired as exc:
            driver.note(f"{stage}: {exc.cmd[:4]} did not finish in {exc.timeout:.0f} s")

    def wrap(cmd: list[str], seconds: float) -> list[str]:
        return ["timeout", "--signal=INT", "--kill-after=120s", str(int(seconds)), *cmd]

    os.chdir(args.repo)
    driver = Driver(run=args.run, smoke_run=args.smoke_run, main_py=args.main_py, vllm_py=args.vllm_py, stop=stop,
                    status=args.status, log_dir=args.log_dir, wrap=wrap, milestone=milestone)
    driver.note(f"deployment driver: stop {args.stop}, steps stop starting new work at "
                f"{dt.datetime.fromtimestamp(stop - JOB_STOP_BEFORE_STOP, CST):%m-%d %H:%M}")
    driver.loop()
    if not (driver.runs["M"] / "report" / "report.json").exists():  # a report the driver never got to: what time is left
        try:
            subprocess.run([args.main_py, "-m", "exec_feedback.deploy", "report", "--run", str(driver.runs["M"])],
                           cwd=args.repo, capture_output=True, timeout=max(120.0, stop - time.time()))
        except subprocess.TimeoutExpired:
            driver.note("the report did not finish before the stop; the package has the evaluations and load tests")
        milestone("M:report")
    driver.note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
