"""Leave a vLLM process without the interpreter's shutdown.

2026-10-19 8xH20 (vLLM 0.11.2, peft 0.18.0): the vLLM gate printed its sample and log-probs, then hung at interpreter
exit until its 900 s timeout. A training or evaluation job hanging there would keep its GPU until the scheduler kills
it near the stop, so the v0.9 commands end through `hard_exit` once their results are on disk.
"""
from __future__ import annotations

import os
import sys
from typing import NoReturn


def hard_exit(code: int, *, grace: float = 20.0) -> NoReturn:
    """Flush stdio, stop every descendant (SIGTERM, SIGKILL after `grace` s) so no vLLM engine keeps GPU memory,
    then os._exit: no atexit handlers, no joins on threads or child processes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - a closed stream must not keep the process alive
            pass
    try:
        import psutil

        children = psutil.Process().children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(children, timeout=grace)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=10)
    except Exception:  # noqa: BLE001 - leave regardless
        pass
    os._exit(code)
