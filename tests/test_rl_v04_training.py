"""Actual CPU optimizer/restart tests with an explicit synthetic bandit fixture."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exec_feedback.core import Limits
from exec_feedback.rl import training
from exec_feedback.storage import read_json, recover_records


def model_and_tasks():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(3))
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return SimpleNamespace(
                logits=self.scores.expand(1, input_ids.shape[1], -1))

    tasks = [SimpleNamespace(id=f"t{i}", db_id="fixture", split="train")
             for i in range(4)]
    return Tiny(), tasks


def synthetic_rollouts(tasks, root, policy, **kwargs):
    """A small bandit environment; these are NOT SQL/model benchmark records."""
    import torch

    for index, task in enumerate(tasks):
        sampled = int(torch.multinomial(policy.scores.softmax(0), 1))
        yield {"task_id": task.id, "db_id": task.db_id,
               "request_index": index, "fixture": True,
               "reward": float(sampled == 0), "sampled_turns": [
                   {"prompt_ids": [2], "completion_ids": [sampled]}]}


def settings():
    return {"learning_rate": 0.02, "updates": 6, "prompt_batch": 2,
            "group_size": 4, "save_steps": 2, "slots": 8,
            "train_seconds": 100}


def run_loop(model, tasks, out, **kwargs):
    oracle = SimpleNamespace(limits=Limits(), score=lambda row: row)
    return training.train_loop(model, model, oracle, tasks, out,
                               settings(), out, seed=42, identity="fixture", **kwargs)


def test_online_resume_matches_uninterrupted_optimizer_and_rng(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(training, "rollouts", synthetic_rollouts)
    full, tasks = model_and_tasks()
    expected = run_loop(full, tasks, tmp_path / "full")
    expected_rng = torch.rand(4)
    partial, _ = model_and_tasks()
    paused = run_loop(partial, tasks, tmp_path / "resume", stop_after=2)
    assert paused["completed"] is False
    resumed, _ = model_and_tasks()
    actual = run_loop(resumed, tasks, tmp_path / "resume")
    assert actual["completed"] is True
    assert actual["cursor"] == expected["cursor"] == 12
    assert actual["optimizer_updates"] == expected["optimizer_updates"] > 0
    assert torch.equal(full.scores, resumed.scores)
    assert torch.equal(torch.rand(4), expected_rng)
    assert resumed.dropout.p == 0
    for left, right in zip(actual["history"], expected["history"]):
        for key in ("prompt_ids", "reward_sum", "sampled_assistant_tokens"):
            assert left[key] == right[key]


def test_interrupted_attempts_retained_but_not_double_counted(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(training, "rollouts", synthetic_rollouts)
    full, tasks = model_and_tasks()
    expected = run_loop(full, tasks, tmp_path / "full")
    original = training.backward_batch
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate interrupted backward")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "backward_batch", fail_second)
    partial, _ = model_and_tasks()
    with pytest.raises(RuntimeError, match="deliberate"):
        run_loop(partial, tasks, tmp_path / "resume")
    assert read_json(tmp_path / "resume/status.json")["durable_update"] == 0
    journal = tmp_path / "resume/rollouts.jsonl"
    prefix = journal.read_bytes()
    monkeypatch.setattr(training, "backward_batch", original)
    resumed, _ = model_and_tasks()
    actual = run_loop(resumed, tasks, tmp_path / "resume")
    assert journal.read_bytes().startswith(prefix)
    assert len(actual["history"]) == len(expected["history"]) == 6
    assert len(recover_records(tmp_path / "resume/metrics.jsonl")) == 7
    assert torch.equal(full.scores, resumed.scores)


def test_all_zero_groups_consume_schedule_without_adam_drift(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    def failed(*args, **kwargs):
        for row in synthetic_rollouts(*args, **kwargs):
            yield {**row, "reward": 0.0}

    monkeypatch.setattr(training, "rollouts", failed)
    model, tasks = model_and_tasks()
    initial = model.scores.detach().clone()
    result = run_loop(model, tasks, tmp_path / "zero")
    assert result["completed"] is True
    assert result["optimizer_updates"] == 0
    assert result["cursor"] == 12
    assert torch.equal(initial, model.scores)
    assert sum(row["natural_trajectories"] for row in result["history"]) == 48
    assert all(row["all_failed_groups"] == 2 for row in result["history"])


def test_train_loop_rejects_validation_before_sampling(tmp_path, monkeypatch):
    model, tasks = model_and_tasks()
    tasks[0].split = "validation"
    monkeypatch.setattr(training, "rollouts", lambda *a, **k: pytest.fail("sampled"))
    with pytest.raises(ValueError, match="train"):
        run_loop(model, tasks, tmp_path)


def test_training_budget_stop_is_not_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "rollouts", synthetic_rollouts)
    model, tasks = model_and_tasks()
    oracle = SimpleNamespace(limits=Limits(), score=lambda row: row)
    config = {**settings(), "train_seconds": 0}
    result = training.train_loop(model, model, oracle, tasks, tmp_path,
                                 config, tmp_path, seed=42, identity="fixture")
    assert result["completed"] is False
    assert result["update"] == result["cursor"] == 0
