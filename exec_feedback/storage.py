"""Versioned, crash-safe local artifacts. No secrets or silent overwrite."""
from __future__ import annotations

import contextlib
import errno
import math
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any, Iterator


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_record(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def recover_records(path: Path) -> list[dict[str, Any]]:
    """Recover only an incomplete final write; retain its bytes for forensics."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        suffix = path.with_name(path.name + f".torn-{time.time_ns()}")
        suffix.write_bytes(raw[cut:])
        with path.open("r+b") as stream:
            stream.truncate(cut)
            stream.flush()
            os.fsync(stream.fileno())
        raw = raw[:cut]
    # A corrupt completed line is not repaired or ignored.
    values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("Journal records must be objects")
    return values


@contextlib.contextmanager
def exclusive_lock(path: Path, *, blocking: bool = False,
                   timeout: float | None = None) -> Iterator[None]:
    """OS-released lock; writers fail fast unless a bounded wait is requested."""
    if type(blocking) is not bool or (timeout is not None and (not math.isfinite(timeout) or timeout < 0)):
        raise ValueError("Lock timeout must be finite and nonnegative")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        # Retain the remote blocking API; explicit timeouts take precedence.
        wait = timeout if timeout is not None else None if blocking else 0.0
        deadline = None if wait is None else time.monotonic() + wait
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                # Only contention is retryable; disk/descriptor errors are not.
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                remaining = 0.01 if deadline is None else deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Another process owns {path}; lock wait exhausted") from exc
                time.sleep(min(0.01, remaining))
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_info() -> dict[str, Any]:
    return {"python": platform.python_version(), "os": platform.platform(),
            "hostname": socket.gethostname(), "pid": os.getpid()}


def code_fingerprint() -> str:
    root = Path(__file__).parent
    return fingerprint({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.rglob("*.py"))})


def checked_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or "\\" in relative:
        raise ValueError("Expected a safe relative POSIX path")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Path escapes root")
    return target


class CallBudget:
    """Durable call reservations, including failed/in-flight requests.

    Count/time limits are enforceable locally. Monetary limits are not inferred
    from unknown remote pricing; providers must be authorized separately.
    """
    def __init__(self, path: Path, max_calls: int, max_seconds: float) -> None:
        if type(max_calls) is not int or not 1 <= max_calls <= 1_000_000:
            raise ValueError("max_calls must be 1..1000000")
        if not 0 < max_seconds <= 604800:
            raise ValueError("max_seconds must be in (0, 604800]")
        self.path = path
        self.limit = max_calls
        self.seconds = max_seconds
        # Concurrent stages share this file. Creation and the initial read must
        # use the SAME lock as reserve(), or a late initializer can erase calls.
        with exclusive_lock(path.with_suffix(".lock"), timeout=30.0):
            if not path.exists():
                atomic_json(path, {"max_calls": max_calls, "max_seconds": max_seconds,
                                   "calls": 0, "active_seconds": 0.0})
            saved = read_json(path)
            if (saved["max_calls"], saved["max_seconds"]) != (max_calls, max_seconds):
                raise ValueError("Budget differs from the original run; start a new study")
        self.started = time.monotonic()
        self.initial_active = saved["active_seconds"]

    def reserve(self, count: int = 1) -> None:
        if type(count) is not int or count < 1:
            raise ValueError("Reservation count must be positive")
        with exclusive_lock(self.path.with_suffix(".lock"), timeout=30.0):
            saved = read_json(self.path)
            # A slower/stale process must not move the durable clock backwards.
            # The study parent/lease driver remains the overall wall-time cap.
            elapsed = max(saved["active_seconds"],
                          self.initial_active + time.monotonic() - self.started)
            if saved["calls"] + count > self.limit or elapsed >= self.seconds:
                raise BudgetExceeded("Configured model-call/time budget exhausted")
            saved.update(calls=saved["calls"] + count, active_seconds=elapsed)
            atomic_json(self.path, saved)


class BudgetExceeded(RuntimeError):
    pass
