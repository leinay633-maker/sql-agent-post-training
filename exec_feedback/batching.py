"""Deterministic lockstep batching of independent gold-free dialogues.

One generate call per active microbatch/turn, not parallel calls into the same
GPU model. SQLite feedback remains per-task and never crosses conversations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .agent import dialogue_stream
from .core import Limits, Task
from .pipeline import grade_episode


def batched_episodes(tasks: list[Task], root: Path, policy: Any, *, batch_size: int,
                     max_steps: int, feedback: bool, limits: Limits,
                     review_success: bool = False) -> Iterator[dict[str, Any]]:
    if not 1 <= batch_size <= 64 or not hasattr(policy, "generate_batch"):
        raise ValueError("Batch size/provider does not support batched inference")
    for start in range(0, len(tasks), batch_size):
        active = []
        try:
            for task in tasks[start:start+batch_size]:
                database = task.database(root)
                engine = dialogue_stream(database, task.question, task.evidence, max_steps=max_steps,
                            feedback=feedback, limits=limits, review_success=review_success)
                try:
                    messages = next(engine)
                    active.append((task, database, engine, messages))
                except StopIteration as done:
                    yield grade_episode(task, database, policy.kind, done.value, limits, max_steps, review_success)
            while active:
                responses = policy.generate_batch([state[3] for state in active])
                if len(responses) != len(active):
                    raise ValueError("Batch response cardinality mismatch; no rows were reassigned")
                remaining = []
                for (task, database, engine, _), response in zip(active, responses):
                    try:
                        messages = engine.send(response)
                        remaining.append((task, database, engine, messages))
                    except StopIteration as done:
                        yield grade_episode(task, database, policy.kind, done.value, limits, max_steps, review_success)
                active = remaining
        finally:
            for _, _, engine, _ in active:
                engine.close()
