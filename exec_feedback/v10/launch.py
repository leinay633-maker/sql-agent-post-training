"""One online run on the GPUs the scheduler gave this job: the first trainer_gpus run the FSDP trainer under torchrun,
each of the next rollout_gpus runs one rollout worker. The launcher wipes the run's channel, starts everything, and stops
everything when any part ends: a worker that exits early stops the run (STOP file, the trainer then fails its wait), the
trainer's exit stops the workers. Workers that leave after the trainer's stop request are the normal end: the trainer
still writes its result. Its exit code is the trainer's, except that a run the trainer left unfinished with 0 (the
smoke's planned pause, a deadline) exits RESUMABLE."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..storage import append_record, read_json
from .freeze import load
from .merge import merged_start
from .protocol import Channel, channel_root
from .trainer import run_name

POLL = 2.0
WORKER_GRACE = 120.0        # after the trainer ends, workers may finish the request they are on
TRAINER_GRACE = 600.0       # after its stop request the trainer writes its result (and at a deadline a ~25 s checkpoint)
# a stopped, resumable run: the scheduler counts a non-zero exit as a try and resumes after its retry delay, at most
# `retries` times. On 0 it relaunched at once in the same second without counting, overwriting the stopped try's logs
RESUMABLE = 75


def split_gpus(visible: list[str], settings: dict[str, Any]) -> tuple[list[str], list[str]]:
    trainer, rollout = settings["trainer_gpus"], settings["rollout_gpus"]
    if len(visible) != trainer + rollout:
        raise ValueError(f"The run needs {trainer} trainer + {rollout} rollout GPUs, the job has {visible}")
    return visible[:trainer], visible[trainer:]


def _stop(procs: list[subprocess.Popen], grace: float) -> None:
    """SIGTERM every process group, SIGKILL what is left after `grace` seconds."""
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM) if hasattr(os, "killpg") else proc.terminate()
            except (ProcessLookupError, PermissionError):
                pass
    end = time.monotonic() + grace
    while time.monotonic() < end and any(p.poll() is None for p in procs):
        time.sleep(0.5)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL) if hasattr(os, "killpg") else proc.kill()
            except (ProcessLookupError, PermissionError):
                pass


def commands(trainer_python: str, worker_python: str, common: list[str], root: Path, trainer_gpus: list[str],
             rollout_gpus: list[str], until: str | None) -> tuple[list[list[str]], list[str]]:
    """The rollout workers' commands and the trainer's (torchrun over the trainer GPUs)."""
    workers = [[worker_python, "-m", "exec_feedback.v10", "rollout", *common, "--worker", str(w), "--root", str(root)]
               for w in range(len(rollout_gpus))]
    trainer = [trainer_python, "-m", "torch.distributed.run", "--standalone", "--nnodes", "1",
               "--nproc-per-node", str(len(trainer_gpus)), "-m", "exec_feedback.v10.trainer", *common,
               "--root", str(root)] + (["--until", until] if until else [])
    return workers, trainer


def launch(run: Path, *, family: str, seed: int, until: str | None, trainer_python: str, worker_python: str,
           log_dir: Path) -> int:
    """trainer_python: the main venv (torch the CPU tests ran on); worker_python: the vLLM venv."""
    frozen, _ = load(run)
    s = frozen["settings"]
    merged_start(run, frozen, seed)  # fail here, not in eight processes
    visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    trainer_gpus, rollout_gpus = split_gpus(visible, s)
    name = run_name(family, seed)
    out = run / "train" / name
    out.mkdir(parents=True, exist_ok=True)
    root = channel_root(frozen["identity"], name, out / "channel")
    channel = Channel.reset(root)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M%S")
    base_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    common = ["--run", str(run), "--family", family, "--seed", str(seed)]
    started = time.time()
    worker_cmds, trainer_cmd = commands(trainer_python, worker_python, common, root, trainer_gpus, rollout_gpus, until)
    workers: list[subprocess.Popen] = []
    logs = []
    for w, (gpu, cmd) in enumerate(zip(rollout_gpus, worker_cmds)):
        fh = open(log_dir / f"{name}.{stamp}.worker{w}.log", "w", encoding="utf-8")
        logs.append(fh)
        workers.append(subprocess.Popen(cmd, env={**base_env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=fh,
                                        stderr=subprocess.STDOUT, start_new_session=True))
    fh = open(log_dir / f"{name}.{stamp}.trainer.log", "w", encoding="utf-8")
    logs.append(fh)
    trainer = subprocess.Popen(trainer_cmd, env={**base_env, "CUDA_VISIBLE_DEVICES": ",".join(trainer_gpus)},
                               stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"{name}: trainer pid {trainer.pid} on GPUs {trainer_gpus}, workers "
          f"{[p.pid for p in workers]} on GPUs {rollout_gpus}, channel {root}", flush=True)
    reason, code = "", None
    try:
        while True:
            rc = trainer.poll()
            if rc is not None:
                code, reason = rc, f"trainer exited {rc}"
                break
            dead = [(w, p.returncode) for w, p in enumerate(workers) if p.poll() is not None]
            if dead and all(rc == 0 for _, rc in dead) and channel.stop_requested():
                # the workers left as the trainer asked, seconds before it exits: taken for a dead worker, this
                # recorded every finished v0.10 run as exit 1
                end = time.monotonic() + TRAINER_GRACE
                while trainer.poll() is None and time.monotonic() < end:
                    time.sleep(POLL)
                rc = trainer.poll()
                if rc is None:
                    _stop([trainer], 30.0)
                    code, reason = 1, f"trainer still running {TRAINER_GRACE:.0f} s after its stop request"
                else:
                    code, reason = rc, f"trainer exited {rc} after the workers left on its stop request"
                break
            if dead:
                reason = f"rollout worker {dead[0][0]} exited {dead[0][1]}"
                channel.stop(reason)  # rank 0 raises out of its wait; torchrun then stops every rank
                end = time.monotonic() + WORKER_GRACE
                while trainer.poll() is None and time.monotonic() < end:
                    time.sleep(POLL)
                _stop([trainer], 30.0)
                code = 1
                break
            time.sleep(POLL)
    except KeyboardInterrupt:  # the scheduler's timeout (SIGINT)
        reason, code = "interrupted", 130
        channel.stop(reason)
        _stop([trainer], 60.0)
    channel.stop(reason)
    _stop(workers, WORKER_GRACE if code == 0 else 30.0)
    result = out / "result.json"
    if code == 0 and not (result.exists() and read_json(result).get("finished")):
        code, reason = RESUMABLE, f"{reason}; stopped before its planned steps"
    for fh in logs:
        fh.close()
    append_record(out / "launches.jsonl", {
        "started": started, "seconds": time.time() - started, "trainer_gpus": trainer_gpus,
        "rollout_gpus": rollout_gpus, "channel": str(root), "reason": reason, "code": code,
        "workers": [p.returncode for p in workers]})
    print(f"{name}: {reason}; workers {[p.returncode for p in workers]}", flush=True)
    return int(code or 0) if code is not None else 1


def main_args(args: Any) -> int:
    return launch(args.run, family=args.family, seed=args.seed, until=args.until,
                  trainer_python=args.trainer_python or sys.executable, worker_python=args.worker_python,
                  log_dir=args.log_dir or args.run / "train" / run_name(args.family, args.seed) / "logs")
