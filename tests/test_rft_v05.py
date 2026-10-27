"""v0.5 engineering tests. SQLite fixtures and scripted generators are NOT model results."""
from __future__ import annotations

import copy
import json
import sqlite3
import time

import pytest

from exec_feedback.core import Limits, Task, digest
from exec_feedback.rft.config import DEFAULT, plan, validate_settings
from exec_feedback.rft.reporting import section
from exec_feedback.rft.runner import run_dialogues
from exec_feedback.rft.sampling import Oracle, sample_items
from exec_feedback.rft.selection import supervised_turns, task_order, tokenize_trajectory


@pytest.fixture
def tasks(tmp_path):
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items(v INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,), (4,)])
    return [Task(f"bird-train:{i}", "shop", "train", f"Question {i}?", "SELECT sum(v) FROM items",
                 database.name, digest(database), "bird-train", "sha256:test") for i in range(5)]


def test_registered_settings_and_pilot_guards():
    validate_settings(DEFAULT)
    assert plan(DEFAULT)["optimizer_steps_per_model"] == 256
    for change in ({"quantization": "nf4"}, {"pool_limit": 10}, {"clean_seeds": [7]}, {"extra": 1}):
        with pytest.raises(ValueError):
            validate_settings({**DEFAULT, **change})
    pilot = {**DEFAULT, "phase": "pilot", "quantization": "nf4", "pool_limit": 50, "dev_limit": 20}
    validate_settings(pilot)
    with pytest.raises(ValueError, match="Pilot"):
        validate_settings({**pilot, "validation_limit": 5})


def test_sample_order_is_k_major(tasks):
    keys = [key for key, _ in sample_items(tasks[:2], 3)]
    assert keys == ["bird-train:0#0", "bird-train:1#0", "bird-train:0#1", "bird-train:1#1",
                    "bird-train:0#2", "bird-train:1#2"]


def row(steps, *, correct=True, status="submitted", checks=()):
    return {"correct": correct, "status": status, "steps": steps, "finish_checks": list(checks)}


def test_masks_error_feedback_and_corrected_wrong_finish():
    query_error = {"action": {"action": "query", "sql": "SELECT x"}, "observation": {"status": "sql_error"}}
    query_ok = {"action": {"action": "query", "sql": "SELECT 1"}, "observation": {"status": "ok", "rows": []}}
    reviewed = {"action": {"action": "finish", "sql": "SELECT 2"}, "observation": {"status": "ok", "review": "check"}}
    protocol = {"observation": {"status": "protocol_error"}}
    final = {"action": {"action": "finish", "sql": "SELECT 7"}}
    assert supervised_turns(row([query_error, query_ok, final])) == [False, True, True]
    assert supervised_turns(row([reviewed, final], checks=[{"step": 0, "correct": False}])) == [False, True]
    assert supervised_turns(row([reviewed, final], checks=[{"step": 0, "correct": True}])) == [True, True]
    assert supervised_turns(row([protocol, final])) == [False, True]
    assert supervised_turns(row([final], correct=False)) is None
    assert supervised_turns(row([final], status="execution_failed")) is None


class ChatML:
    """Character-level stand-in with Qwen's ChatML layout; ids are shifted code points."""
    eos_token_id = pad_token_id = 2
    chat_template = "chatml-test"

    def __init__(self, drop_history: bool = False) -> None:
        self.drop_history = drop_history

    def text(self, value: str) -> list[int]:
        return [ord(c) + 10 for c in value]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_start|>": 1, "<|im_end|>": 2}[token]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for i, message in enumerate(messages):
            content = message["content"]
            if self.drop_history and message["role"] == "assistant" and i < len(messages) - 1:
                content = "[old]"
            ids += [1] + self.text(message["role"] + "\n" + content) + [2] + self.text("\n")
        if add_generation_prompt:
            ids += [1] + self.text("assistant\n")
        return ids


MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "bad"}, {"role": "user", "content": "err"},
            {"role": "assistant", "content": "good"}]


def test_labels_cover_kept_turn_content_and_end_token_only():
    tokenizer = ChatML()
    tokens = tokenize_trajectory(MESSAGES, [False, True], tokenizer, 4096)
    supervised = [t for t in tokens["labels"] if t != -100]
    assert supervised == tokenizer.text("good") + [2]
    assert len(tokens["labels"]) == len(tokens["input_ids"])
    both = tokenize_trajectory(MESSAGES, [True, True], tokenizer, 4096)
    assert [t for t in both["labels"] if t != -100] == tokenizer.text("bad") + [2] + tokenizer.text("good") + [2]
    assert tokenize_trajectory(MESSAGES, [False, True], tokenizer, 10) is None


def test_template_that_rewrites_history_is_refused():
    with pytest.raises(ValueError, match="prefix-consistent"):
        tokenize_trajectory(MESSAGES, [False, True], ChatML(drop_history=True), 4096)


def test_oracle_grades_final_and_reviewed_finish(tasks, tmp_path):
    oracle = Oracle(tasks, tmp_path, Limits())
    record = {"status": "submitted", "final_sql": "SELECT sum(v) FROM items",
              "result": {"status": "ok", "columns": ["s"], "rows": [[7]]},
              "steps": [{"action": {"action": "finish", "sql": "SELECT max(v) FROM items"},
                         "observation": {"status": "ok", "review": "check"}},
                        {"action": {"action": "finish", "sql": "SELECT sum(v) FROM items"}}]}
    scored = oracle.score(tasks[0], copy.deepcopy(record))
    assert scored["verdict"]["correct"] is True
    assert scored["finish_checks"] == [{"step": 0, "correct": False}]
    wrong = {**record, "result": {"status": "ok", "columns": ["s"], "rows": [[4]]}}
    assert oracle.score(tasks[0], wrong)["verdict"]["correct"] is False
    with pytest.raises(ValueError):
        Oracle([Task(**{**tasks[0].__dict__, "split": "validation"})], tmp_path, Limits())


def test_oracle_marks_failed_reference_unusable(tasks, tmp_path):
    broken = Task(**{**tasks[0].__dict__, "id": "bird-train:x", "gold_sql": "SELECT missing FROM items"})
    oracle = Oracle([broken], tmp_path, Limits())
    record = {"status": "submitted", "final_sql": "SELECT 1", "result": {"status": "ok", "columns": ["a"], "rows": [[1]]},
              "steps": [{"action": {"action": "finish", "sql": "SELECT 1"}}]}
    assert oracle.score(broken, record)["verdict"] == {"status": "oracle_error", "correct": False}


class Scripted:
    """Queries once, then finishes; responses depend only on each conversation."""
    kind = "model"

    def __init__(self) -> None:
        self.batches = []

    def generate(self, conversations):
        self.batches.append(len(conversations))
        replies = []
        for messages in conversations:
            assistant_turns = sum(m["role"] == "assistant" for m in messages)
            action = ({"action": "query", "sql": "SELECT count(*) FROM items"} if assistant_turns == 0
                      else {"action": "finish", "sql": "SELECT sum(v) FROM items"})
            replies.append({"text": json.dumps(action), "usage": {}, "finish_reason": "stop"})
        return replies


def test_refilled_dialogues_finish_independently(tasks, tmp_path):
    generator = Scripted()
    items = [(f"{t.id}#0", t) for t in tasks]
    rows = list(run_dialogues(items, tmp_path, generator, slots=2, max_steps=4, review_success=True,
                              limits=Limits(), workers=4,
                              finish=lambda key, task, record: {"key": key, **record}))
    assert sorted(r["key"] for r in rows) == sorted(k for k, _ in items)
    assert all(r["status"] == "submitted" and r["final_sql"] == "SELECT sum(v) FROM items" for r in rows)
    assert max(generator.batches) == 2


def test_deadline_stops_new_dialogues_but_finishes_started_ones(tasks, tmp_path):
    rows = list(run_dialogues([(t.id, t) for t in tasks], tmp_path, Scripted(), slots=2, max_steps=4,
                              review_success=True, limits=Limits(), workers=2,
                              finish=lambda key, task, record: record, deadline=time.monotonic() - 1))
    assert rows == []


def test_frontier_tasks_ordered_before_easy_and_unsolved_dropped():
    by_task = {"a": [{"correct": True}, {"correct": True}], "b": [{"correct": True}, {"correct": False}],
               "c": [{"correct": False}, {"correct": False}], "d": [{"correct": False}, {"correct": True}]}
    order, frontier, easy, histogram = task_order(by_task, "s", 1)
    assert set(order[:2]) == {"b", "d"} and order[2:] == ["a"]
    assert sorted(frontier) == ["b", "d"] and easy == ["a"] and histogram["0/2"] == 1


def test_report_pools_seeds_and_filters_subsets(tasks):
    def rows(correct):
        return [{"task_id": t.id, "db_id": t.db_id, "correct": bool(c), "status": "correct" if c else "incorrect"}
                for t, c in zip(tasks, correct)]
    runs = {"B": rows([1, 0, 0, 1, 0]), "R_r1_seed42": rows([1, 1, 0, 1, 1]),
            "R_r1_seed43": rows([1, 1, 0, 1, 0]), "Rclean_r1_seed42": rows([1, 0, 0, 1, 1])}
    result = section(tasks, runs)
    assert result["conditions"]["B"]["correct"] == 2
    assert result["R_r1_minus_B"]["mean_delta"] == pytest.approx(0.3)
    assert result["R_r1_seed42_minus_Rclean_r1_seed42"]["delta_right_minus_left"] == pytest.approx(0.2)
    subset = section(tasks[:2], runs)
    assert subset["tasks"] == 2 and subset["conditions"]["R_r1_seed42"]["correct"] == 2


def test_behavior_counts_query_repair_and_repeats():
    from exec_feedback.rft.reporting import behavior
    query = {"action": {"action": "query", "sql": "SELECT 1"}, "observation": {"status": "ok"}}
    error = {"action": {"action": "query", "sql": "SELECT x"}, "observation": {"status": "sql_error"}}
    finish = {"action": {"action": "finish", "sql": "SELECT 2"}}
    rows = [{"correct": True, "steps": [error, query, finish]},
            {"correct": False, "steps": [finish, finish], "mismatch": "columns"},
            {"correct": False, "status": "not_attempted"}]
    result = behavior(rows)
    assert result == {"episodes": 2, "tasks_using_query": 1, "tasks_with_error_feedback": 1,
                      "error_tasks_finally_correct": 1, "repeated_identical_finish": 1,
                      "incorrect_wrong_columns": 1, "incorrect_wrong_rows": 0, "mean_assistant_steps": 2.5}


def test_markdown_summary_lists_pooled_and_seed_deltas(tasks):
    from exec_feedback.rft.reporting import markdown
    def rows(correct):
        return [{"task_id": t.id, "db_id": t.db_id, "correct": bool(c), "status": "correct" if c else "incorrect",
                 "steps": [{"action": {"action": "finish", "sql": "SELECT 1"}}]} for t, c in zip(tasks, correct)]
    runs = {"B": rows([1, 0, 0, 1, 0]), "R_r1_seed42": rows([1, 1, 0, 1, 1]), "R_r1_seed43": rows([1, 1, 0, 1, 0])}
    text = markdown({"roles": {"dev": section(tasks, runs)}})
    assert "| B | 2 | 40.0% |" in text and "错因：列数" in text
    assert "R_r1_minus_B：seed 平均 +30.0" in text and "R_r1_seed43：+20.0" in text


def test_rejected_trajectory_is_trimmed_to_last_assistant_turn():
    from exec_feedback.rft.preference import rejected_turns
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "q1"}, {"role": "user", "content": "r1"},
                {"role": "assistant", "content": "q2"}, {"role": "user", "content": "r2"}]
    row = {"correct": False, "verdict": "budget_exhausted", "steps": [{}, {}], "messages": messages}
    trimmed, flags = rejected_turns(row)
    assert trimmed == messages[:5] and flags == [True, True]
    assert rejected_turns({**row, "verdict": "provider_error"}) is None
    assert rejected_turns({**row, "correct": True, "verdict": "correct"}) is None


def test_pair_loss_is_dpo_margin_plus_chosen_nll():
    import math
    torch = pytest.importorskip("torch")
    from exec_feedback.rft.preference import pair_loss
    same = pair_loss((torch.tensor(-10.0), torch.tensor(-12.0)), (-10.0, -12.0), 5, beta=0.1, sft_weight=0.0)
    assert float(same) == pytest.approx(math.log(2))
    better = pair_loss((torch.tensor(-9.0), torch.tensor(-13.0)), (-10.0, -12.0), 5, beta=0.1, sft_weight=0.0)
    assert float(better) < math.log(2)
    with_nll = pair_loss((torch.tensor(-10.0), torch.tensor(-12.0)), (-10.0, -12.0), 5, beta=0.1, sft_weight=1.0)
    assert float(with_nll) == pytest.approx(math.log(2) + 2.0)


def test_sequence_logprob_matches_manual_token_sum():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from exec_feedback.rft.preference import sequence_logprob
    torch.manual_seed(3)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=64, pad_token_id=1, eos_token_id=2)).eval()
    ids = [3, 5, 7, 9, 11, 2, 4, 6]
    labels = [-100, -100, -100, 9, 11, 2, -100, 6]
    with torch.no_grad():
        value, count = sequence_logprob(model, {"input_ids": ids, "labels": labels})
        logits = model(input_ids=torch.tensor([ids])).logits[0].log_softmax(-1)
    manual = sum(float(logits[i - 1, ids[i]]) for i, label in enumerate(labels) if label != -100)
    assert count == 4 and float(value) == pytest.approx(manual, abs=1e-5)
