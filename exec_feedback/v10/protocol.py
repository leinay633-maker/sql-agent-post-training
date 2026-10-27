"""Files through which one run's trainer (rank 0) and its rollout workers talk.

Requests are numbered; every worker handles every request in order and answers with one result file. A request names
the weight version it must be served with; weights/v<version>/model.safetensors becomes visible through
weights/v<version>.ready.json, written last. Workers write a heartbeat every few seconds. Everything lives on /dev/shm
when the host has it (8 GB per version), and is wiped by the launcher before every attempt.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..storage import atomic_json, read_json

HEARTBEAT_SECONDS = 5.0


def channel_root(run_identity: str, name: str, fallback: Path) -> Path:
    shm = Path("/dev/shm")
    return (shm / "efa-v10" / f"{run_identity[:12]}-{name}") if shm.is_dir() else fallback


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


class Channel:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for sub in ("requests", "results", "weights", "heartbeat"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def reset(root: Path) -> "Channel":
        if Path(root).exists():
            shutil.rmtree(root)
        return Channel(root)

    # requests and results
    def post(self, seq: int, request: dict[str, Any]) -> None:
        atomic_json(self.root / "requests" / f"{seq:06d}.json", {"seq": seq, "posted": time.time(), **request})

    def request(self, seq: int) -> dict[str, Any] | None:
        return _read(self.root / "requests" / f"{seq:06d}.json")

    def put_result(self, seq: int, worker: int, payload: dict[str, Any]) -> None:
        folder = self.root / "results" / f"{seq:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / f"worker-{worker}.json", payload)

    def results(self, seq: int, workers: int) -> list[dict[str, Any]] | None:
        folder = self.root / "results" / f"{seq:06d}"
        parts = [_read(folder / f"worker-{w}.json") for w in range(workers)]
        return None if any(p is None for p in parts) else parts

    def drop_results(self, seq: int) -> None:
        shutil.rmtree(self.root / "results" / f"{seq:06d}", ignore_errors=True)

    # weights
    def weights(self, version: int) -> Path:
        return self.root / "weights" / f"v{version:05d}"

    def ready(self, version: int) -> dict[str, Any] | None:
        return _read(self.weights(version).with_name(f"v{version:05d}.ready.json"))

    def prune(self, keep_from: int) -> None:
        """Weight versions older than keep_from are no longer requested by anyone."""
        for path in (self.root / "weights").glob("v*.ready.json"):
            version = int(path.name[1:6])
            if version < keep_from:
                path.unlink(missing_ok=True)
                shutil.rmtree(self.weights(version), ignore_errors=True)

    def stop_requested(self) -> bool:
        """The newest request is the trainer's stop: it asked the workers to leave (its run ended as planned or at its
        deadline). Nothing is posted after it."""
        newest = max((self.root / "requests").glob("*.json"), default=None)
        return newest is not None and (_read(newest) or {}).get("kind") == "stop"

    # liveness
    def stop(self, reason: str = "") -> None:
        atomic_json(self.root / "STOP.json", {"time": time.time(), "reason": reason})

    def stopped(self) -> dict[str, Any] | None:
        return _read(self.root / "STOP.json")

    def beat(self, worker: int, state: str) -> None:
        atomic_json(self.root / "heartbeat" / f"worker-{worker}.json", {"time": time.time(), "pid": os.getpid(),
                                                                         "state": state})

    def silent(self, workers: int, limit: float, since: float) -> list[int]:
        """Workers whose last heartbeat is older than `limit` seconds (or that never beat, once `since` is that old)."""
        now, out = time.time(), []
        for w in range(workers):
            beat = _read(self.root / "heartbeat" / f"worker-{w}.json")
            last = beat["time"] if beat else since
            if now - last > limit:
                out.append(w)
        return out


class Heartbeat:
    """Background heartbeat of one worker; `state` is updated by the worker loop."""

    def __init__(self, channel: Channel, worker: int) -> None:
        self.channel, self.worker, self.state = channel, worker, "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.channel.beat(self.worker, self.state)
            except OSError:
                pass
            self._stop.wait(HEARTBEAT_SECONDS)

    def close(self) -> None:
        self._stop.set()


def wait_for(get: Callable[[], Any], *, timeout: float, what: str, channel: Channel,
             check: Callable[[], None] | None = None, poll: float = 0.2) -> Any:
    """Poll until get() returns something; raise on STOP, on a failed check() or after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        value = get()
        if value is not None:
            return value
        stop = channel.stopped()
        if stop is not None:
            raise RuntimeError(f"Stopped while waiting for {what}: {stop.get('reason')}")
        if check is not None:
            check()
        if time.monotonic() > deadline:
            raise TimeoutError(f"Waited {timeout / 60:.0f} min for {what}")
        time.sleep(poll)
