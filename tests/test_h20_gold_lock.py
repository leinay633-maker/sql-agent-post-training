"""Multiple CPU processes share a resumable SQL reference preflight."""
import multiprocessing as mp
import sqlite3
from pathlib import Path

from exec_feedback.core import Limits, Task, digest
from exec_feedback.datasets import validate_gold
from exec_feedback.storage import read_json, recover_records


def _gold_worker(task, raw, out, barrier, queue):
    try:
        barrier.wait(timeout=20)
        result = validate_gold([task], Path(raw), Path(out), Limits(seconds=2), workers=1, lock_timeout=20)
        queue.put((result['valid'], None))
    except BaseException as exc:
        queue.put((0, repr(exc)))


def test_parallel_gold_waits_without_duplicate_records(tmp_path):
    database = tmp_path / 'fixture.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE t(x)')
        db.execute('INSERT INTO t VALUES (7)')
    task = Task('fixture:1', 'fixture', 'validation', 'Fixture value?', 'SELECT x FROM t',
                'fixture.sqlite', digest(database), 'test-fixture', 'test', fixture=True)
    ctx = mp.get_context('spawn')
    barrier, queue = ctx.Barrier(4), ctx.Queue()
    out = tmp_path / 'gold'
    workers = [ctx.Process(target=_gold_worker, args=(task, str(tmp_path), str(out), barrier, queue)) for _ in range(4)]
    try:
        for p in workers: p.start()
        results = [queue.get(timeout=30) for _ in workers]
        for p in workers: p.join(timeout=10)
        assert all(p.exitcode == 0 for p in workers)
        assert results == [(1, None)] * 4
        assert len(recover_records(out / 'gold.jsonl')) == 1
        assert read_json(out / 'summary.json')['valid'] == 1
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate(); p.join(timeout=10)
        queue.close()
