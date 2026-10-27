"""CPU tests for v0.6 (no GPU, no vLLM, no model weights)."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from exec_feedback.pipeline import messages_for
from exec_feedback.v6 import PROTOCOL
from exec_feedback.v6.config import DEFAULT, validate_settings
from exec_feedback.v6.freeze import render
from exec_feedback.v6.rollouts import decide, result_key

SCHEMA = {"status": "ok", "rows": [["t", "CREATE TABLE t (a INTEGER)"]]}
TASK = SimpleNamespace(question="How many?", evidence="a is a count")


def test_default_settings_valid_and_strict():
    validate_settings(copy.deepcopy(DEFAULT))
    bad = copy.deepcopy(DEFAULT)
    bad["unknown"] = 1
    with pytest.raises(ValueError):
        validate_settings(bad)
    bad = copy.deepcopy(DEFAULT)
    bad["protocol"] = "efa-v5-rft-natural-v1"
    with pytest.raises(ValueError):
        validate_settings(bad)
    bad = copy.deepcopy(DEFAULT)
    bad["note_budget"] = bad["max_context"]
    with pytest.raises(ValueError):
        validate_settings(bad)
    assert DEFAULT["protocol"] == PROTOCOL


def test_prompt_without_extra_is_byte_identical_to_v05():
    plain = messages_for(TASK, SCHEMA, 4, True, review_success=True)
    assert messages_for(TASK, SCHEMA, 4, True, review_success=True, extra=None) == plain
    assert json.loads(plain[1]["content"]) == {"question": "How many?", "evidence": "a is a count",
                                               "schema": SCHEMA["rows"]}


def test_extra_adds_public_notes_but_cannot_replace_fields():
    msgs = messages_for(TASK, SCHEMA, 4, True, review_success=True, extra={"column_notes": {"t": ["a: count"]}})
    assert json.loads(msgs[1]["content"])["column_notes"] == {"t": ["a: count"]}
    with pytest.raises(ValueError):
        messages_for(TASK, SCHEMA, 4, True, extra={"question": "other"})


def test_column_note_tiers_shrink_and_drop_uninformative_rows():
    raw = {"t": [["a", "a", "a", ""], ["b", "Bee", "long description " * 20, "0: no; 1: yes"]]}
    full, short = render(raw, 200, 300), render(raw, 50, 0)
    assert "a" not in "".join(x.split(":")[0] for x in full["t"] if x.startswith("a"))
    assert full["t"][0].startswith("b (Bee): ") and "| values: 0: no; 1: yes" in full["t"][0]
    assert len(short["t"][0]) < len(full["t"][0]) and "values" not in short["t"][0]


def test_result_key_is_gold_free_order_insensitive_and_column_aware():
    a = {"status": "ok", "columns": ["x"], "rows": [[1], [2]]}
    b = {"status": "ok", "columns": ["y"], "rows": [[2], [1]]}
    c = {"status": "ok", "columns": ["x", "z"], "rows": [[1, None], [2, None]]}
    assert result_key(a) == result_key(b) != result_key(c)
    assert result_key({"status": "timeout"}) is None and result_key(None) is None


def test_vote_picks_largest_group_then_earliest_and_reports_oracle_stats():
    rows = [{"_key": "A", "_k": 0, "correct": False, "status": "incorrect"},
            {"_key": "B", "_k": 1, "correct": True, "status": "correct"},
            {"_key": "B", "_k": 2, "correct": True, "status": "correct"},
            {"_key": None, "_k": 3, "correct": False, "status": "execution_failed"}]
    out = decide(rows)
    assert out["status"] == "correct" and out["vote"]["chosen_k"] == 1 and out["vote"]["votes"] == 2
    assert out["vote"]["any_correct"] and out["vote"]["sample_correct"] == 2 and "_key" not in out
    tie = decide([{"_key": "A", "_k": 1, "correct": False, "status": "incorrect"},
                  {"_key": "B", "_k": 0, "correct": True, "status": "correct"}])
    assert tie["vote"]["chosen_k"] == 0
    none = decide([{"_key": None, "_k": 2, "correct": False, "status": "x"}, {"_key": None, "_k": 0, "correct": False, "status": "y"}])
    assert none["vote"]["votes"] == 0 and none["vote"]["chosen_k"] == 0


def test_preregistered_selection_rule():
    from exec_feedback.v6.analysis import selection

    def cond(rate, n=10):
        return {"rate": rate, "correct": int(rate * n), "attempted": n}
    roles = {"validation": {"project": {"tasks": 10, "conditions": {
        "R_r1_seed42": cond(0.5), "R_r1_seed43": cond(0.5), "R_r1_seed44": cond(0.5),
        "R_r2_seed42": cond(0.6), "R_r2_seed43": cond(0.6), "R_r2_seed44": cond(0.6), "R_r2_seed42+vote": cond(0.7)}}}}
    out = selection(roles, [42, 43, 44])
    assert out["family"] == "R_r2" and out["vote"] and out["headline_condition"] == "R_r2_seed42+vote"
    del roles["validation"]["project"]["conditions"]["R_r2_seed44"]  # incomplete round 2 never wins
    assert selection(roles, [42, 43, 44])["family"] == "R_r1"


def test_fenced_json_only_in_v06_mode_and_only_a_single_object():
    from exec_feedback.pipeline import parse_action
    fenced = '```json\n{"action": "query", "sql": "SELECT 1"}\n```'
    with pytest.raises(ValueError):
        parse_action(fenced)  # v0.5 behaviour unchanged
    assert parse_action(fenced, fenced_json=True) == {"action": "query", "sql": "SELECT 1"}
    assert parse_action('```\n{"action": "schema"}\n```', fenced_json=True) == {"action": "schema"}
    for bad in ('Sure:\n```json\n{"action": "query", "sql": "SELECT 1"}\n```',
                '```json\n{"action": "query", "sql": "SELECT 1"}\n```\nand more',
                '```json\n{"action": "answer", "sql": "SELECT 1"}\n```'):
        with pytest.raises(ValueError):
            parse_action(bad, fenced_json=True)
