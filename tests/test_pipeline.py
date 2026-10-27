import argparse
import copy
import dataclasses
import json

import pytest

from exec_feedback.cli import import_bird, main, make_fixture
from exec_feedback.core import digest, read_jsonl
from exec_feedback.pipeline import FixturePolicy, HTTPPolicy, build_sft, episode, evaluate, parse_action


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "db"
    task = make_fixture(root)
    return task, root


def repaired(task, root):
    return episode(task, root, FixturePolicy([
        {"action": "query", "sql": "SELECT gross_sales FROM orders"},
        {"action": "finish", "sql": task.gold_sql},
    ]))


def test_feedback_and_gold_separation(sample):
    task, root = sample
    trace = repaired(task, root)
    assert trace["correct"]
    assert trace["steps"][0]["observation"]["status"] == "sql_error"
    assert task.gold_sql not in trace["messages"][0]["content"]
    assert "gold_sql" not in json.dumps(trace["messages"][:2])


def test_budget_is_hard_limit(sample):
    task, root = sample
    trace = episode(task, root, FixturePolicy([{"action": "schema"}] * 4))
    assert trace["status"] == "budget_exhausted"
    assert len(trace["steps"]) == 4
    assert not trace["correct"]


def test_direct_disallows_tools(sample):
    task, root = sample
    trace = episode(task, root, FixturePolicy([{"action": "query", "sql": task.gold_sql}]), max_steps=1, feedback=False)
    assert trace["status"] == "protocol_error"


def test_no_mock_accuracy_and_no_overwrite(sample, tmp_path):
    task, root = sample
    out = tmp_path / "eval"
    summary = evaluate([task], root, FixturePolicy([{"action": "finish", "sql": task.gold_sql}]), out)
    assert summary["run_kind"] == "engineering_fixture"
    assert summary["strict_success_rate"] is None
    assert len(read_jsonl(out / "episodes.jsonl")) == 1
    with pytest.raises(FileExistsError):
        evaluate([task], root, FixturePolicy([]), out)


def test_failures_remain_in_denominator(sample, tmp_path):
    task, root = sample
    class Broken:
        kind = "model"
        metadata = {"provider": "unit-test-double"}
        def __call__(self, messages):
            raise ValueError("unavailable")
    # Metadata mutation here tests accounting only; this is not a reported model run.
    task = dataclasses.replace(task, fixture=False)
    summary = evaluate([task], root, Broken(), tmp_path / "broken")
    assert summary["tasks"] == 1
    assert summary["strict_success_rate"] == 0
    assert summary["statuses"] == {"provider_error": 1}


@pytest.mark.parametrize("text", ['[]', '{}', '{"action":"shell"}', '{"action":"query","sql":1}',
                                    '{"action":"schema","sql":"x"}', '```json\n{}\n```'])
def test_bad_protocol(text):
    with pytest.raises((ValueError, TypeError)):
        parse_action(text)


@pytest.mark.parametrize("url", ["https://api.example.com/v1", "http://api.example.com/v1",
                                   "https://localhost.evil.test/v1", "http://u:p@localhost/v1",
                                   "file:///tmp/api", "http://localhost/v1?api_key=x"])
def test_remote_or_credential_endpoints_fail_closed(url):
    with pytest.raises(ValueError):
        HTTPPolicy(url, "test-model")


def test_local_endpoint_and_model_mismatch():
    policy = HTTPPolicy("http://127.0.0.1:8000/v1", "expected")
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return b'{"model":"wrong","choices":[{"message":{"content":"x"}}]}'
    class Opener:
        def open(self, *args, **kwargs): return Response()
    policy.opener = Opener()
    with pytest.raises(ValueError, match="model id"):
        policy([])


def test_sft_only_train_and_fixture_opt_in(sample):
    task, root = sample
    with pytest.raises(ValueError, match="Fixtures"):
        build_sft([task], root, "success")
    with pytest.raises(ValueError, match="training tasks"):
        build_sft([dataclasses.replace(task, split="test")], root, "success", allow_fixture=True)
    row = build_sft([task], root, "success", allow_fixture=True)[0]
    assert row["fixture"] and row["split"] == "train"


def test_repair_reexecutes_and_rejects_tampering(sample):
    task, root = sample
    trace = repaired(task, root)
    rows = build_sft([task], root, "repair", [trace], allow_fixture=True)
    assert "gross_sales" in json.dumps(rows[0]["prompt"])
    fake = copy.deepcopy(trace)
    fake["final_sql"] = "SELECT 999"
    with pytest.raises(ValueError, match="independent grading"):
        build_sft([task], root, "repair", [fake], allow_fixture=True)
    fake = copy.deepcopy(trace)
    fake["messages"][3]["content"] = "fabricated feedback"
    with pytest.raises(ValueError, match="replay"):
        build_sft([task], root, "repair", [fake], allow_fixture=True)


def test_bird_import_hash_and_holdout(sample, tmp_path):
    task, root = sample
    import shutil
    folder = root / "shop"
    folder.mkdir()
    shutil.copy(root / "shop.sqlite", folder / "shop.sqlite")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([{"question_id": 1, "db_id": "shop", "question": task.question, "SQL": task.gold_sql}]))
    args = argparse.Namespace(input=raw, db_root=root, output=tmp_path / "bird.jsonl",
                              expected_sha256=digest(raw), source_revision="fixture-test", kind="bird-mini-dev",
                              split="test", ack_data_terms=True)
    imported = import_bird(args)
    assert imported["tasks"] == 1
    assert read_jsonl(args.output)[0]["split"] == "test"
    args.split = "train"
    with pytest.raises(ValueError, match="test-only"):
        import_bird(args)
    args.expected_sha256 = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        import_bird(args)


def test_doctor_does_not_require_ml_packages(capsys):
    assert main(["doctor"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["network_requests"] == 0
    assert info["spending_authorized"] == 0
