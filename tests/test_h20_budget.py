"""CPU multiprocessing checks, not evidence of GPU speed or model quality."""
import errno
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from exec_feedback.storage import (BudgetExceeded, CallBudget, atomic_json,
                                   exclusive_lock, read_json)


def _reserve_worker(path, barrier, queue, calls, cap):
    try:
        barrier.wait(timeout=20)
        budget = CallBudget(Path(path), cap, 120)
        completed = 0
        for _ in range(calls):
            try:
                budget.reserve()
                completed += 1
            except BudgetExceeded:
                break
        queue.put((completed, None))
    except BaseException as exc:
        queue.put((0, repr(exc)))


@pytest.mark.parametrize("cap,expected", [(1000, 120), (53, 53)])
def test_shared_initialization_reservations_and_cap(tmp_path, cap, expected):
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(6), ctx.Queue()
    path = tmp_path / "budget.json"
    workers = [ctx.Process(target=_reserve_worker,
                          args=(str(path), barrier, queue, 20, cap)) for _ in range(6)]
    try:
        for worker in workers:
            worker.start()
        results = [queue.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
        assert all(worker.exitcode == 0 for worker in workers)
        assert all(error is None for _, error in results), results
        assert sum(count for count, _ in results) == expected
        assert read_json(path)["calls"] == expected
        # A later process/resume must not reset the shared count either.
        CallBudget(path, cap, 120)
        assert read_json(path)["calls"] == expected
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        queue.close()


def test_stale_budget_cannot_rewind_time_or_evade_time_cap(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    budget = CallBudget(path, 10, 120)
    saved = read_json(path)
    saved["active_seconds"] = 119.0
    atomic_json(path, saved)
    monkeypatch.setattr("exec_feedback.storage.time.monotonic", lambda: budget.started + 1)
    budget.reserve()
    assert read_json(path)["active_seconds"] == 119.0
    saved = read_json(path)
    saved["active_seconds"] = 120.0
    atomic_json(path, saved)
    with pytest.raises(BudgetExceeded):
        budget.reserve()
    assert read_json(path)["calls"] == 1


def test_writer_lock_stays_fail_fast_and_wait_is_bounded(tmp_path):
    path = tmp_path / "writer.lock"
    with exclusive_lock(path):
        with pytest.raises(RuntimeError):
            with exclusive_lock(path):
                pytest.fail("two writers acquired one lock")
        with pytest.raises(RuntimeError):
            with exclusive_lock(path, timeout=0.02):
                pytest.fail("bounded wait acquired a held lock")
    with exclusive_lock(path, timeout=0.1):
        pass


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_lock_wait(tmp_path, timeout):
    with pytest.raises(ValueError):
        with exclusive_lock(tmp_path / "lock", timeout=timeout):
            pass


def test_budget_mismatch_is_not_reinitialized(tmp_path):
    path = tmp_path / "budget.json"
    budget = CallBudget(path, 10, 120)
    budget.reserve(2)
    with pytest.raises(ValueError):
        CallBudget(path, 11, 120)
    assert read_json(path)["calls"] == 2


@pytest.mark.skipif(os.name == "nt", reason="fcntl error injection is POSIX-only")
def test_unrelated_lock_error_is_not_retried(tmp_path, monkeypatch):
    def broken(*args):
        raise OSError(errno.EIO, "simulated disk error")
    monkeypatch.setattr("fcntl.flock", broken)
    with pytest.raises(OSError, match="simulated disk error"):
        with exclusive_lock(tmp_path / "lock", timeout=30):
            pass


def test_remote_blocking_keyword_is_preserved_and_explicit_timeout_wins(tmp_path):
    path = tmp_path / "compat.lock"
    with exclusive_lock(path, blocking=True):
        with pytest.raises(RuntimeError):
            with exclusive_lock(path, blocking=True, timeout=0.02):
                pytest.fail("explicit timeout must remain bounded")
    with exclusive_lock(path, blocking=True, timeout=0.1):
        pass
