"""v0.4 engineering tests. SQLite fixtures/tiny tensors are NOT model results."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from exec_feedback.core import Limits, Task, digest
from exec_feedback.rl import PROTOCOL
from exec_feedback.rl.backend import LocalPolicy
from exec_feedback.rl.config import DEFAULT, plan, validate_settings
from exec_feedback.rl.objective import advantages, group_metrics
from exec_feedback.rl.reporting import aligned, paired_report
from exec_feedback.rl.rollout import TrainingOracle, rollouts
from exec_feedback.rl.training import draw_tasks


@pytest.fixture
def task(tmp_path):
    database = tmp_path / "example.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items(v INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (1,), (2,), (None,)])
    return Task("t0", "example", "train", "What is the sum of v?",
                "SELECT sum(v) FROM items /* GOLD_ONLY_SECRET */",
                database.name, digest(database), "engineering", "fixture-v1", fixture=True)


def candidate(task, sql, *, failed=False, status="submitted"):
    return {"task_id": task.id, "db_id": task.db_id, "generation_failed": failed,
            "sampled_turns": [], "record": {"final_sql": sql, "correct": True,
                                             "status": status}}


@pytest.mark.parametrize("sql,expected", [
    ("SELECT sum(v) FROM items", 1.0),
    ("SELECT sum(v)+1 FROM items", 0.0),
    ("SELECT count(v) FROM items", 0.0),
    ("SELECT missing FROM items", 0.0),
    ("DELETE FROM items", 0.0),
    (None, 0.0),
])
def test_real_execution_reward(task, tmp_path, sql, expected):
    oracle = TrainingOracle([task], tmp_path, Limits(), allow_fixture=True)
    assert oracle.score(candidate(task, sql))["reward"] == expected
    assert digest(tmp_path / task.db_file) == task.db_sha256


@pytest.mark.parametrize("split", ["validation", "test"])
def test_reward_split_rejected_before_sql(task, tmp_path, split):
    with pytest.raises(ValueError, match="train"):
        TrainingOracle([dataclasses.replace(task, split=split)], tmp_path, Limits(), allow_fixture=True)


def test_reward_requires_fixture_opt_in(task, tmp_path):
    with pytest.raises(ValueError):
        TrainingOracle([task], tmp_path, Limits())


def test_invalid_gold_is_fatal_not_filtered(task, tmp_path):
    invalid = dataclasses.replace(task, gold_sql="SELECT missing FROM items")
    with pytest.raises(ValueError, match="no task removed"):
        TrainingOracle([invalid], tmp_path, Limits(), allow_fixture=True)


def test_unknown_reward_task_rejected(task, tmp_path):
    oracle = TrainingOracle([task], tmp_path, Limits(), allow_fixture=True)
    row = candidate(task, "SELECT 4")
    row["task_id"] = "validation:1"
    with pytest.raises(ValueError, match="Unregistered"):
        oracle.score(row)


def test_truncated_generation_cannot_earn_reward(task, tmp_path):
    oracle = TrainingOracle([task], tmp_path, Limits(), allow_fixture=True)
    assert oracle.score(candidate(task, "SELECT 4", failed=True))["reward"] == 0


def test_truncated_sql_result_cannot_earn_reward(task, tmp_path):
    oracle = TrainingOracle([task], tmp_path, Limits(max_rows=1), allow_fixture=True)
    assert oracle.score(candidate(task, "SELECT v FROM items"))["reward"] == 0


@pytest.mark.parametrize("gold,sql,expected", [
    ("SELECT v FROM items", "SELECT DISTINCT v FROM items", 0.0),
    ("SELECT v FROM items", "SELECT v FROM items ORDER BY v DESC", 1.0),
    ("SELECT v FROM items ORDER BY v", "SELECT v FROM items ORDER BY v DESC", 0.0),
    ("SELECT v FROM items WHERE v IS NULL", "SELECT 0", 0.0),
])
def test_reward_duplicates_order_and_null(task, tmp_path, gold, sql, expected):
    changed = dataclasses.replace(task, gold_sql=gold, ordered="ORDER BY" in gold)
    oracle = TrainingOracle([changed], tmp_path, Limits(), allow_fixture=True)
    assert oracle.score(candidate(changed, sql))["reward"] == expected


class ScriptPolicy:
    """Clearly labeled test double; only the SQLite environment is real."""
    kind = "fixture"

    def __init__(self, *, length=False, mismatch=False):
        self.messages = []
        self.batches = []
        self.length, self.mismatch = length, mismatch

    def generate_batch(self, conversations):
        self.batches.append(len(conversations))
        if self.mismatch:
            return []
        results = []
        for messages in conversations:
            self.messages.append(copy.deepcopy(messages))
            last = messages[-1]["content"]
            sql = "SELECT missing FROM items" if "tool_result" not in last else "SELECT sum(v) FROM items"
            action = "query" if "tool_result" not in last else "finish"
            results.append({"text": json.dumps({"action": action, "sql": sql}),
                            "prompt_ids": [1, 2], "completion_ids": [3],
                            "finish_reason": "length" if self.length else "stop",
                            "usage": {"prompt_tokens": 2, "completion_tokens": 1}})
        return results


def test_multiturn_feedback_gold_absence_and_group_indices(task, tmp_path):
    policy = ScriptPolicy()
    rows = list(rollouts([task] * 4, tmp_path, policy, slots=2, limits=Limits()))
    assert sorted(r["request_index"] for r in rows) == [0, 1, 2, 3]
    assert all(len(r["sampled_turns"]) == 2 for r in rows)
    assert all(r["record"]["steps"][0]["observation"]["status"] == "sql_error" for r in rows)
    assert "GOLD_ONLY_SECRET" not in json.dumps(policy.messages)
    assert all("gold_sql" not in json.dumps(m) for m in policy.messages)
    oracle = TrainingOracle([task], tmp_path, Limits(), allow_fixture=True)
    assert [oracle.score(r)["reward"] for r in rows] == [1.0] * 4


def test_rollout_length_failure_retained(task, tmp_path):
    rows = list(rollouts([task] * 4, tmp_path, ScriptPolicy(length=True), slots=4, limits=Limits()))
    oracle = TrainingOracle([task], tmp_path, Limits(), allow_fixture=True)
    assert len(rows) == 4
    assert [oracle.score(r)["reward"] for r in rows] == [0.0] * 4


def test_rollout_cardinality_mismatch_is_fatal(task, tmp_path):
    with pytest.raises(ValueError, match="cardinality"):
        list(rollouts([task], tmp_path, ScriptPolicy(mismatch=True), slots=1, limits=Limits()))


def test_rollout_refills_finished_slots(task, tmp_path):
    class EarlyFailure(ScriptPolicy):
        def generate_batch(self, conversations):
            result = super().generate_batch(conversations)
            if len(self.batches) == 1:
                result[0] = ValueError("Deliberate fixture context rejection")
            return result
    policy = EarlyFailure()
    rows = list(rollouts([task] * 3, tmp_path, policy, slots=2, limits=Limits()))
    assert policy.batches[:2] == [2, 2]
    assert len(rows) == 3
    assert sum(r["generation_failed"] for r in rows) == 1


@pytest.mark.parametrize("rewards,expected", [
    ([1, 0, 0, 0], [1, -1/3, -1/3, -1/3]),
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([1, 1, 1, 1], [0, 0, 0, 0]),
    ([0, 1, 0, 1], [-2/3, 2/3, -2/3, 2/3]),
])
def test_leave_one_out_advantages(rewards, expected):
    assert advantages(rewards, 4) == pytest.approx(expected)


@pytest.mark.parametrize("rewards,group", [([], 4), ([0, 1, 0], 4), ([0], 1), ([float("nan"), 1], 2), ([2, 0], 2)])
def test_invalid_reward_groups_rejected(rewards, group):
    with pytest.raises(ValueError):
        advantages(rewards, group)


def test_zero_signal_groups_are_counted():
    info = group_metrics([0, 0, 0, 0, 1, 1, 1, 1], 4)
    assert info["trajectories"] == 8
    assert info["informative_groups"] == 0
    assert info["all_failed_groups"] == info["all_correct_groups"] == 1


def tiny_model():
    torch = pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(5))
        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
            length = logits_to_keep or input_ids.shape[1]
            return SimpleNamespace(logits=self.scores.expand(1, length, -1))
    return Tiny()


def test_action_loss_masks_prompt_but_trains_every_sampled_token():
    torch = pytest.importorskip("torch")
    from exec_feedback.rl.objective import action_logprob
    model = tiny_model()
    value = action_logprob(model, [4] * 100, [0, 1, 1])
    expected = 3 * torch.log_softmax(model.scores, -1)[0]
    assert float(value.detach()) == pytest.approx(float(expected.detach()))
    (-value).backward()
    assert model.scores.grad.tolist() == pytest.approx([-0.4, -1.4, 0.6, 0.6, 0.6])


def test_rloo_gradient_includes_first_and_final_actions():
    torch = pytest.importorskip("torch")
    from exec_feedback.rl.objective import backward_batch
    model = tiny_model()
    good = {"reward": 1.0, "sampled_turns": [
        {"prompt_ids": [4], "completion_ids": [0]},
        {"prompt_ids": [4, 0, 4], "completion_ids": [1, 1]}]}
    bad = {"reward": 0.0, "sampled_turns": [
        {"prompt_ids": [4], "completion_ids": [2]}]}
    metrics = backward_batch(model, [good, bad], 2)
    # Sum over all three good tokens, not mean over a response or final-only.
    assert model.scores.grad.tolist() == pytest.approx([-0.3, -0.8, 0.7, 0.2, 0.2])
    assert metrics["sampled_assistant_tokens"] == 4


def test_zero_advantage_does_not_build_gradients():
    from exec_feedback.rl.objective import backward_batch
    model = tiny_model()
    rows = [{"reward": 0.0, "sampled_turns": [{"prompt_ids": [0], "completion_ids": [1]}]}] * 4
    assert backward_batch(model, rows, 4)["has_gradient"] is False
    assert model.scores.grad is None


def test_backend_bucketing_restores_request_order_and_sampling_config(monkeypatch):
    torch = pytest.importorskip("torch")
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=Config))
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = []
        def generate(self, **kwargs):
            assert set(kwargs) == {"input_ids", "attention_mask", "generation_config"}
            config = kwargs["generation_config"]
            assert (config.top_k, config.top_p, config.temperature) == (0, 1.0, 1.0)
            ids, masks = kwargs["input_ids"], kwargs["attention_mask"]
            self.calls.append(masks.sum(1).tolist())
            tail = torch.stack([ids[:, -1], torch.full_like(ids[:, -1], 9)], dim=1)
            return torch.cat([ids, tail], dim=1)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9,
                                apply_chat_template=lambda m, **kw: m,
                                decode=lambda ids, **kw: str(ids))
    model = FakeModel()
    policy = LocalPolicy(model, tokenizer, batch_size=2, max_context=20, max_tokens=2, sample=True)
    rows = policy.generate_batch([[1, 1, 1], [2], [3, 3]])
    assert [row["completion_ids"] for row in rows] == [[1, 9], [2, 9], [3, 9]]
    assert model.calls == [[1, 2], [3]]
    assert policy.metadata["padded_prompt_tokens"] == 7
    assert isinstance(policy.generate_batch([[1] * 19])[0], ValueError)


def test_checkpoint_restores_optimizer_rng_cursor_and_lora(tmp_path):
    torch = pytest.importorskip("torch")
    from exec_feedback.rl.checkpoint import restore_checkpoint, save_checkpoint
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.scores.sum().backward()
    optimizer.step()
    state = {"identity": "frozen", "update": 1, "cursor": 8, "history": []}
    save_checkpoint(tmp_path, model, optimizer, state)
    expected_weights = model.scores.detach().clone()
    torch_next, python_next = torch.rand(3), random.random()
    with torch.no_grad():
        model.scores.add_(10)
    restored = restore_checkpoint(tmp_path, model, optimizer, "frozen")
    assert restored["cursor"] == 8
    assert torch.equal(model.scores, expected_weights)
    assert torch.equal(torch.rand(3), torch_next)
    assert random.random() == python_next
    assert len(optimizer.state) == 1
    with pytest.raises(ValueError, match="identity"):
        restore_checkpoint(tmp_path, model, optimizer, "changed")
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path, model, optimizer, state)
    weights = tmp_path / "checkpoint-00000001" / "trainable.pt"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        restore_checkpoint(tmp_path, model, optimizer, "frozen")


def test_registered_prompt_epoch_covers_all_candidates(task):
    tasks = [dataclasses.replace(task, id=f"t{i}") for i in range(1024)]
    selected = [t.id for update in range(128) for t in draw_tasks(tasks, update * 8, 8, 42)]
    assert len(selected) == len(set(selected)) == 1024
    assert selected[:8] == [t.id for t in draw_tasks(tasks, 0, 8, 42)]


def test_v04_plan_is_not_a_gpu_result():
    result = plan(copy.deepcopy(DEFAULT))
    assert result["maximum_model_calls"] == 28672
    assert result["maximum_generated_tokens"] == 11010048
    assert result["gpu_seconds"] is result["quality"] is None
    assert result["protocol"] != "efa-v3"


@pytest.mark.parametrize("key,value", [
    ("protocol", "efa-v3"), ("group_size", 1), ("seeds", [42, 42]),
    ("learning_rate", float("nan")), ("train_calls", 1), ("generation_batch", True),
    ("phase", "pilot"), ("eval_calls", 1), ("max_context", 999999),
])
def test_config_fail_closed(key, value):
    settings = copy.deepcopy(DEFAULT)
    settings[key] = value
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_unknown_config_field_rejected():
    with pytest.raises(ValueError):
        validate_settings({**copy.deepcopy(DEFAULT), "top_p": 0.9})


def statistical_tasks():
    return [SimpleNamespace(id=str(i), db_id=f"db{i // 2}", fixture=False) for i in range(4)]


def score_rows(tasks, correct):
    return [{"task_id": t.id, "db_id": t.db_id, "correct": correct,
             "status": "correct" if correct else "incorrect"} for t in tasks]


def test_paired_database_seed_report_and_failures():
    tasks = statistical_tasks()
    baseline = score_rows(tasks, False)
    runs = {str(seed): score_rows(tasks, True) for seed in (42, 43, 44)}
    report = paired_report(tasks, baseline, runs, draws=100)
    assert report["mean_delta"] == 1
    assert report["paired_database_seed_bootstrap_95"] == [1, 1]
    assert report["tasks_per_condition"] == 4  # not 3 x 4 independent tasks
    incomplete = paired_report(tasks, baseline, {"42": runs["42"][:1]}, draws=100)
    assert incomplete["per_seed"]["42"]["success_rate"] == 0.25
    assert incomplete["per_seed"]["42"]["statuses"]["not_attempted"] == 3
    assert incomplete["paired_database_seed_bootstrap_95"] is None
    assert incomplete["complete"] is False


def test_report_rejects_duplicate_and_foreign_rows():
    tasks = statistical_tasks()
    rows = score_rows(tasks, False)
    with pytest.raises(ValueError):
        aligned(tasks, rows + [rows[0]])
    with pytest.raises(ValueError):
        aligned(tasks, [{**rows[0], "task_id": "unknown"}])


def test_fixture_report_does_not_claim_model_quality():
    tasks = statistical_tasks()
    tasks[0].fixture = True
    result = paired_report(tasks, score_rows(tasks, False), {"42": score_rows(tasks, True)})
    assert result["mean_delta"] is None
    assert result["baseline_success_rate"] is None
    assert result["paired_database_seed_bootstrap_95"] is None


def test_existing_v03_dialogue_and_grader_bytes_unchanged():
    root = Path(__file__).resolve().parents[1] / "exec_feedback"
    # v0.6 deliberately extended both files with default-off parameters (extra prompt fields, fenced JSON,
    # frozen gold). Default behaviour is pinned by tests/test_v06.py::test_prompt_without_extra_is_byte_identical_to_v05
    # and the unchanged v0.3/v0.5 suites; previous blobs: agent 74d7e5d6..., pipeline ed6aad02....
    # v0.9 passes an optional per-turn token trace from the response into the step record (only when a generator
    # returns one; no earlier generator does); previous agent blob a7874ac2....
    expected = {"agent.py": "e9d70ad1b896c051685f42bd72c8d953570250ea",
                "pipeline.py": "438a395b7ba703788399a8fba8de003e8bd7bf53"}
    for name, sha in expected.items():
        data = (root / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert actual == sha


@pytest.mark.parametrize("status", ["budget_exhausted", "execution_failed", "provider_error"])
def test_unfinished_or_failed_record_cannot_earn_reward(task, tmp_path, status):
    oracle = TrainingOracle([task], tmp_path, Limits(), allow_fixture=True)
    row = candidate(task, "SELECT 4", status=status)
    assert oracle.score(row)["reward"] == 0


@pytest.mark.parametrize("change", [
    {"db_id": "different"}, {"correct": 1}, {"status": "execution_failed"},
])
def test_report_rejects_inconsistent_boolean_database_status(change):
    tasks = statistical_tasks()
    row = {**score_rows(tasks, True)[0], **change}
    with pytest.raises(ValueError):
        aligned(tasks, [row])
