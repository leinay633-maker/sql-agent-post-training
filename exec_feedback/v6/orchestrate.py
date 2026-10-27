"""v0.6 GPU scheduler for one 8-GPU host: dependency-ordered jobs, half/full-GPU slots, retries, absolute deadline.

Plan (docs/RESEARCH_V06.md):
  round 1   vLLM samplers (one per GPU) until every pool shard is claimed and finished
  build 1   production selection (CPU)
  train 1   R_r1 seeds + Rclean_r1 (full GPU each); B evaluations fill the other GPUs (half GPU each)
  round 2   only if the measured round-1 time fits before the stop (preregistered rule); policy R_r1_seed<p>
  train 2   R_r2 seeds; R_r1 evaluations alongside
  evaluate  validation and test for every frozen model, greedy; +vote for B and the seed-<p> models
  report    project score + official EX, preregistered selection; progress entries after every stage
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

CST = dt.timezone(dt.timedelta(hours=8))


class Scheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.run = args, args.run.resolve()
        self.stop = dt.datetime.fromisoformat(args.stop)
        self.gpus = [int(g) for g in args.gpus.split(",")]
        self.used = {g: 0 for g in self.gpus}          # half-slots in use (2 = full GPU)
        self.last_start = {g: 0.0 for g in self.gpus}
        self.running: dict[int, dict[str, Any]] = {}
        self.done: set[str] = set()
        self.failed: dict[str, int] = {}
        self.log_dir = self.run.parent / "logs" / self.run.name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.frozen = json.loads((self.run / "manifest.json").read_text())
        self.settings = self.frozen["settings"]
        self.times: dict[str, float] = {}

    # ------------------------------------------------------------------ utilities
    def note(self, text: str) -> None:
        line = f"{dt.datetime.now(CST).strftime('%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        with open(self.args.status, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def left(self) -> float:
        return (self.stop - dt.datetime.now(dt.timezone.utc)).total_seconds()

    def iso(self, seconds_before_stop: float) -> str:
        return (self.stop - dt.timedelta(seconds=seconds_before_stop)).isoformat()

    def progress(self, stage: str) -> None:
        subprocess.run([self.args.main_py, "-m", "exec_feedback.v6", "progress", "--run", str(self.run), "--stage", stage,
                        "--commit", self.args.commit, "--file", str(self.args.progress)],
                       cwd=self.args.repo, capture_output=True, timeout=600)
        if self.args.export_hook:
            subprocess.run(["bash", self.args.export_hook, stage], capture_output=True, timeout=1800)

    def cpu(self, name: str, cmd: list[str], timeout: float) -> bool:
        self.note(f"start {name}")
        started = time.monotonic()
        with open(self.log_dir / f"{name}.log", "w") as fh:
            rc = subprocess.run(cmd, cwd=self.args.repo, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        self.times[name] = time.monotonic() - started
        self.note(f"done {name} exit {rc} ({self.times[name] / 60:.1f} min)")
        return rc == 0

    def free_gpu(self, half: bool) -> int | None:
        need = 1 if half else 2
        now = time.monotonic()
        for g in self.gpus:
            if self.used[g] + need <= 2 and now - self.last_start[g] >= (90 if self.used[g] else 0):
                return g
        return None

    def launch(self, name: str, cmd: list[str], *, half: bool, gpu: int, timeout_at: float) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        tries = self.failed.get(name, 0) + 1
        fh = open(self.log_dir / f"{name}.try{tries}.log", "w")
        timeout = max(60, int(self.left() - timeout_at))
        proc = subprocess.Popen(["timeout", "--signal=INT", "--kill-after=60s", str(timeout), *cmd], cwd=self.args.repo,
                                env=env, stdout=fh, stderr=subprocess.STDOUT)
        self.running[proc.pid] = {"name": name, "proc": proc, "gpu": gpu, "half": half, "fh": fh,
                                  "started": time.monotonic(), "cmd": cmd, "timeout_at": timeout_at}
        self.used[gpu] += 1 if half else 2
        self.last_start[gpu] = time.monotonic()
        self.note(f"start {name} on GPU {gpu} ({'half' if half else 'full'}, try {tries})")

    def reap(self) -> list[tuple[str, int]]:
        finished = []
        for pid, job in list(self.running.items()):
            rc = job["proc"].poll()
            if rc is None:
                continue
            job["fh"].close()
            self.used[job["gpu"]] -= 1 if job["half"] else 2
            del self.running[pid]
            self.times[job["name"]] = time.monotonic() - job["started"]
            self.note(f"done {job['name']} exit {rc} on GPU {job['gpu']} ({self.times[job['name']] / 60:.1f} min)")
            finished.append((job["name"], rc))
        return finished

    def is_running(self, name: str) -> bool:
        return any(j["name"] == name for j in self.running.values())

    # ------------------------------------------------------------------ stage helpers
    def vpy(self, *a: str) -> list[str]:
        return [self.args.vllm_py, "-m", "exec_feedback.v6", *a, "--run", str(self.run)]

    def mpy(self, *a: str) -> list[str]:
        return [self.args.main_py, "-m", "exec_feedback.v6", *a, "--run", str(self.run)]

    def claims_left(self, round_: int) -> bool:
        claims = self.run / f"round{round_}" / "claims"
        return not claims.exists() or len(list(claims.iterdir())) < self.settings["sample_shards"]

    def sample_round(self, round_: int, policy: str, minutes: float, reserve: float) -> None:
        stop_at = min(time.time() + minutes * 60, self.stop.timestamp() - reserve)
        stop_iso = dt.datetime.fromtimestamp(stop_at, dt.timezone.utc).isoformat()
        self.note(f"round {round_}: sampling with {policy} until all {self.settings['sample_shards']} shards are done "
                  f"or {dt.datetime.fromtimestamp(stop_at, CST).strftime('%H:%M')}")
        started, n, crashes = time.monotonic(), 0, 0
        while True:
            for name, rc in self.reap():
                if rc != 0:
                    self.failed[name] = self.failed.get(name, 0) + 1
                    crashes += 1
            if crashes > 2 * len(self.gpus):
                self.note(f"round {round_}: {crashes} sampler failures; no new samplers (see logs)")
                self.wait_all(f"sample{round_}-")
                break
            if time.time() < stop_at - 120 and self.claims_left(round_):
                gpu = self.free_gpu(half=False)
                if gpu is not None:
                    n += 1
                    self.launch(f"sample{round_}-{n}", self.vpy("sample", "--round", str(round_), "--policy", policy,
                                "--stop", stop_iso, "--mem", str(self.args.sample_mem), "--seed", str(20261012 + 1009 * n)),
                                half=False, gpu=gpu, timeout_at=reserve - 300)
                    continue
            if not any(j["name"].startswith(f"sample{round_}-") for j in self.running.values()):
                if not self.claims_left(round_) or time.time() >= stop_at - 120:
                    break
            time.sleep(10)
        self.times[f"sample{round_}"] = time.monotonic() - started
        self.note(f"round {round_}: sampling finished in {self.times[f'sample{round_}'] / 60:.1f} min")

    def eval_jobs(self, condition: str, role: str) -> list[tuple[str, list[str]]]:
        vote = condition.endswith("+vote")
        shards = range(self.settings["vote_shards"]) if vote else [0]
        jobs = []
        for shard in shards:
            name = f"eval-{condition}-{role}" + (f"-{shard}" if vote else "")
            jobs.append((name, self.vpy("evaluate", "--condition", condition, "--role", role, "--shard", str(shard),
                                        "--stop", self.iso(self.args.eval_reserve), "--mem", str(self.args.eval_mem))))
        return jobs

    def eval_done(self, name: str, condition: str, role: str, shard: int) -> bool:
        return (self.run / "eval" / role / condition / f"shard{shard:02d}.summary.json").exists()

    def work(self, trains: list[tuple[str, list[str], Callable[[], bool]]], evals: list[tuple[str, str]],
             until: Callable[[], bool], timeout_reserve: float) -> None:
        """Run trainings (full GPU, first) and ready evaluations (half GPU) until `until()`."""
        queue = []
        for cond, role in evals:
            for name, cmd in self.eval_jobs(cond, role):
                shard = int(name.rsplit("-", 1)[1]) if cond.endswith("+vote") else 0
                queue.append((name, cmd, cond, role, shard))
        trains = list(trains)
        while self.left() > timeout_reserve:
            for name, rc in self.reap():
                if rc != 0:
                    self.failed[name] = self.failed.get(name, 0) + 1
                    item = next((t for t in self.retry_pool if t[0] == name), None)
                    if item and self.failed[name] < 3:
                        (trains if item[3] == "train" else queue).insert(0, item[1])
                        self.note(f"re-queued {name} (failure {self.failed[name]})")
            started = False
            if trains:
                gpu = self.free_gpu(half=False)
                if gpu is not None:
                    name, cmd, ready = trains[0]
                    if ready():
                        trains.pop(0)
                        self.retry_pool.append((name, (name, cmd, ready), None, "train"))
                        self.launch(name, cmd, half=False, gpu=gpu, timeout_at=timeout_reserve)
                        started = True
            if not started:
                training = trains or any(j["name"].startswith("train-") for j in self.running.values())
                for i, (name, cmd, cond, role, shard) in enumerate(queue):
                    if self.eval_done(name, cond, role, shard) or self.is_running(name):
                        queue.pop(i)
                        started = True
                        break
                    if not self.adapter_ready(cond):
                        if not training:  # its training failed or never ran: nothing will make it ready
                            self.note(f"dropped {name}: adapter not available")
                            queue.pop(i)
                            started = True
                            break
                        continue
                    gpu = self.free_gpu(half=True)
                    if gpu is None:
                        break
                    queue.pop(i)
                    self.retry_pool.append((name, (name, cmd, cond, role, shard), None, "eval"))
                    self.launch(name, cmd, half=True, gpu=gpu, timeout_at=self.args.eval_reserve - 120)
                    started = True
                    break
            if until() and not trains:
                if not queue or (not self.args.drain and not started):
                    break
            if not started:
                time.sleep(10)
        self.pending_evals = [(c, r) for _, _, c, r, _ in queue]

    def adapter_ready(self, condition: str) -> bool:
        base = condition.split("+")[0]
        if base == "B":
            return True
        variant, rest = base.split("_r", 1)
        round_, seed = rest.split("_seed")
        result = self.run / f"round{round_}" / f"train_{variant}_seed{seed}" / "result.json"
        if not result.exists():
            return False
        r = json.loads(result.read_text())
        return bool(r.get("training_completed") and r.get("adapter_reload_verified"))

    def wait_all(self, prefix: str | None = None) -> None:
        while any(prefix is None or j["name"].startswith(prefix) for j in self.running.values()):
            self.reap()
            time.sleep(10)

    def train_cmd(self, round_: int, variant: str, seed: int) -> tuple[str, list[str], Callable[[], bool]]:
        name = f"train-{variant}_r{round_}_seed{seed}"
        return name, self.mpy("train", "--round", str(round_), "--variant", variant, "--seed", str(seed)), lambda: True

    # ------------------------------------------------------------------ plan
    def main(self) -> int:
        s, p = self.settings, self.settings["round2_policy_seed"]
        self.retry_pool: list[Any] = []
        self.note(f"v0.6 scheduler: run {self.frozen['identity'][:12]}, GPUs {self.gpus}, stop {self.stop.isoformat()}")
        # round 1
        if not (self.run / "round1" / "selection.json").exists():
            self.sample_round(1, "B", s["sample_minutes"], reserve=self.args.r1_reserve)
            self.progress("sample1")
            if not self.cpu("build1", self.mpy("build", "--round", "1"), 3600):
                self.note("build1 failed; stopping")
                return 1
            self.progress("build1")
        trains1 = [self.train_cmd(1, "R", seed) for seed in s["seeds"]] + \
                  [self.train_cmd(1, "Rclean", seed) for seed in s["clean_seeds"]]
        evals_b = [("B", "validation"), ("B", "test"), ("B+vote", "validation"), ("B+vote", "test")]
        r1 = [f"R_r1_seed{x}" for x in s["seeds"]]
        t1_started = time.monotonic()
        self.work(trains1, evals_b, until=lambda: all(self.adapter_ready(c) for c in r1) and
                  not any(j["name"].startswith("train-") for j in self.running.values()), timeout_reserve=self.args.eval_reserve)
        self.times["train1"] = time.monotonic() - t1_started
        self.progress("train1")
        # round 2 decision (preregistered): measured round-1 sampling + training + eval reserve must fit
        need = self.times.get("sample1", 0) + self.times["train1"] + self.args.eval_reserve + 600
        do_r2 = s["samples_round2"] > 0 and self.adapter_ready(f"R_r1_seed{p}") and self.left() >= need
        self.note(f"round 2 {'runs' if do_r2 else 'skipped'}: {self.left() / 60:.0f} min left, needs {need / 60:.0f} min")
        evals_r1 = [(c, "validation") for c in r1] + [(f"Rclean_r1_seed{x}", "validation") for x in s["clean_seeds"]]
        if do_r2 and not (self.run / "round2" / "selection.json").exists():
            # Evaluations of R_r1 on validation share the GPUs with the round-2 samplers.
            self.args.drain = False
            self.work([], evals_r1, until=lambda: True, timeout_reserve=self.args.eval_reserve)
            self.sample_round(2, f"R_r1_seed{p}", s["sample_minutes"], reserve=self.times["train1"] + self.args.eval_reserve)
            self.progress("sample2")
            if self.cpu("build2", self.mpy("build", "--round", "2"), 3600):
                self.progress("build2")
                trains2 = [self.train_cmd(2, "R", seed) for seed in s["seeds"]]
                r2 = [f"R_r2_seed{x}" for x in s["seeds"]]
                self.args.drain = True
                self.work(trains2, [*getattr(self, "pending_evals", []), *evals_r1,
                                    *[(c, "test") for c in r1], (f"R_r1_seed{p}+vote", "validation")],
                          until=lambda: all(self.adapter_ready(c) for c in r2) and
                          not any(j["name"].startswith("train-") for j in self.running.values()),
                          timeout_reserve=self.args.eval_reserve)
                self.progress("train2")
        r2 = [f"R_r2_seed{x}" for x in s["seeds"]] if (self.run / "round2" / "selection.json").exists() else []
        final = [*getattr(self, "pending_evals", []), *evals_r1, *[(c, "validation") for c in r2],
                 (f"R_r1_seed{p}+vote", "validation"), *[(c, "test") for c in r1],
                 *[(c, "test") for c in r2], (f"Rclean_r1_seed{s['clean_seeds'][0]}", "test"),
                 (f"R_r1_seed{p}+vote", "test")]
        if r2:
            final += [(f"R_r2_seed{p}+vote", "validation"), (f"R_r2_seed{p}+vote", "test")]
        self.args.drain = True
        self.work([], final, until=lambda: not self.running, timeout_reserve=self.args.final_reserve)
        for job in list(self.running.values()):
            job["proc"].send_signal(2)
        self.wait_all()
        self.note(f"GPU phase over; unstarted evaluations: {getattr(self, 'pending_evals', [])}")
        self.cpu("report", self.mpy("report", "--procs", "96"), 2400)
        self.progress("report")
        self.note("ALL DONE")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stop", required=True, help="absolute compute stop, ISO with offset")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--main-py", required=True)
    ap.add_argument("--vllm-py", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--export-hook", default="")
    ap.add_argument("--r1-reserve", type=float, default=3 * 3600, help="seconds kept free after round-1 sampling")
    ap.add_argument("--eval-reserve", type=float, default=40 * 60, help="no new dialogues this close to the stop")
    ap.add_argument("--final-reserve", type=float, default=25 * 60)
    ap.add_argument("--sample-mem", type=float, default=0.85)
    ap.add_argument("--eval-mem", type=float, default=0.40)
    args = ap.parse_args()
    args.drain = True
    return Scheduler(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
