"""Gold-free multi-turn rollouts and a separate, train-only reward oracle."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterator

from ..agent import dialogue_stream
from ..core import Limits, Task, audit, compare, digest, execute


def rollouts(tasks: list[Task], root: Path, policy: Any, *,
             slots: int, limits: Limits, max_steps: int = 2) -> Iterator[dict[str, Any]]:
    """Refill completed dialogue slots; never share a conversation across tasks.

    Repeated task ids are intentional for RL groups; request_index, not task id,
    restores task-major sample order after scheduling/bucketing.
    """
    if not 1 <= slots <= 256 or max_steps != 2:
        raise ValueError("v0.4-v1 uses 1..256 slots and exactly two assistant steps")
    pending = deque(enumerate(tasks))
    active = []
    try:
        while pending or active:
            while pending and len(active) < slots:
                index, task = pending.popleft()
                engine = dialogue_stream(
                    task.database(root), task.question, task.evidence,
                    max_steps=max_steps, feedback=True, review_success=True,
                    limits=limits)
                state = {"request_index": index, "task_id": task.id,
                         "db_id": task.db_id, "sampled_turns": [],
                         "origin": "natural_model", "generation_failed": False,
                         "generation_errors": [],
                         "engine": engine}
                try:
                    state["messages"] = next(engine)
                    active.append(state)
                except StopIteration as done:
                    yield _finished(state, done.value)
            if not active:
                continue
            responses = policy.generate_batch([s["messages"] for s in active])
            if len(responses) != len(active):
                raise ValueError("Generation cardinality mismatch; refuse reassignment")
            remaining = []
            for state, response in zip(active, responses):
                if not isinstance(response, Exception):
                    state["sampled_turns"].append({
                        "prompt_ids": response["prompt_ids"],
                        "completion_ids": response["completion_ids"],
                        "finish_reason": response["finish_reason"],
                    })
                    state["generation_failed"] |= response["finish_reason"] != "stop"
                else:
                    state["generation_failed"] = True
                    state["generation_errors"].append({
                        "type": type(response).__name__, "reason": str(response)[:300]})
                try:
                    state["messages"] = state["engine"].send(response)
                    remaining.append(state)
                except StopIteration as done:
                    yield _finished(state, done.value)
            active = remaining
    finally:
        for state in active:
            state["engine"].close()


def _finished(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    public = {k: v for k, v in state.items() if k not in {"engine", "messages"}}
    if state["generation_failed"]:
        record.update(status="generation_failed", correct=False)
    return {**public, "record": record}


class TrainingOracle:
    """Gold stays here, never on the model-facing policy or dialogue object."""

    def __init__(self, tasks: list[Task], root: Path, limits: Limits,
                 *, allow_fixture: bool = False) -> None:
        if not tasks or any(t.split != "train" for t in tasks):
            raise ValueError("Execution reward accepts registered train tasks only")
        if any(t.source != "bird-train" for t in tasks) and not allow_fixture:
            raise ValueError("Only audited BIRD train is allowed in research RL")
        if any(t.fixture for t in tasks) and not allow_fixture:
            raise ValueError("Fixture reward is an engineering test, not research")
        audit(tasks, root)
        self.tasks = {t.id: t for t in tasks}
        self.root, self.limits = root, limits
        self._gold = {}
        for task in tasks:
            answer = execute(task.database(root), task.gold_sql, limits)
            if answer["status"] != "ok":
                raise ValueError(f"Invalid train reference: {task.id}; no task removed")
            self._gold[task.id] = answer

    def score(self, episode: dict[str, Any]) -> dict[str, Any]:
        task = self.tasks.get(episode["task_id"])
        if task is None or episode["db_id"] != task.db_id:
            raise ValueError("Unregistered reward request")
        record = episode["record"]
        verdict = {"correct": False, "status": record["status"]}
        sql = record.get("final_sql")
        if sql and record["status"] == "submitted" and not episode["generation_failed"]:
            # Never trust an externally supplied correct/result field.
            prediction = execute(task.database(self.root), sql, self.limits)
            verdict = compare(prediction, self._gold[task.id], task.ordered)
        if digest(task.database(self.root)) != task.db_sha256:
            raise ValueError("Training database changed")
        return {**episode, "reward": float(verdict["correct"]),
                "reward_status": verdict["status"], "fixture": task.fixture}
