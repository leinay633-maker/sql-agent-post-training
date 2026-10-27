import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from exec_feedback.cli import make_fixture
from exec_feedback.core import Limits, Task, audit, cell_equal, compare, digest, execute


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "db"
    task = make_fixture(root)
    return task, root, root / task.db_file


def result(rows, columns=None):
    return {"status": "ok", "columns": columns or ["x"], "rows": rows}


def test_real_query_and_unchanged_db(sample):
    task, _, db = sample
    answer = execute(db, task.gold_sql)
    assert answer["status"] == "ok"
    assert answer["rows"] == [[40.0]]
    assert digest(db) == task.db_sha256


@pytest.mark.parametrize("sql", [
    "INSERT INTO orders VALUES(6,4,1,'paid')", "DELETE FROM orders", "DROP TABLE orders",
    "UPDATE orders SET amount=0", "ATTACH ':memory:' AS extra", "PRAGMA query_only=OFF",
    "WITH x AS (SELECT 1) DELETE FROM orders", "SELECT load_extension('x')",
    "SELECT readfile('/etc/passwd')", "SELECT writefile('x','y')",
    "SELECT randomblob(1000000)", "SELECT zeroblob(1000000)",
    "SELECT 1; DELETE FROM orders", "SELECT * FROM pragma_database_list",
])
def test_dangerous_sql_denied(sample, sql):
    task, _, db = sample
    assert execute(db, sql)["status"] != "ok"
    assert digest(db) == task.db_sha256


def test_cte_and_comments_allowed(sample):
    _, _, db = sample
    assert execute(db, "-- safe\n/* comment */ WITH x AS (SELECT 7 AS a) SELECT a FROM x")["rows"] == [[7]]


def test_missing_column_is_feedback(sample):
    _, _, db = sample
    answer = execute(db, "SELECT imaginary FROM orders")
    assert answer["status"] == "sql_error"
    assert "imaginary" in answer["error"]


def test_instruction_limit(sample):
    _, _, db = sample
    sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n"
    assert execute(db, sql, Limits(vm_steps=1000))["status"] == "timeout"


def test_no_partial_result_can_pass(sample):
    _, _, db = sample
    answer = execute(db, "SELECT id FROM orders", Limits(max_rows=1))
    assert answer["status"] == "result_limit"
    assert "rows" not in answer
    assert not compare(answer, result([[1]]))["correct"]


@pytest.mark.parametrize("a,b,ordered,expected", [
    ([[1], [2]], [[2], [1]], False, True),
    ([[1], [2]], [[2], [1]], True, False),
    ([[1], [1]], [[1], [2]], False, False),
    ([[None]], [[None]], False, True),
    ([[None]], [[0]], False, False),
    ([["1"]], [[1]], False, False),
    ([[0.1 + 0.2]], [[0.3]], False, True),
    ([[10000001]], [[10000002.0]], False, False),
    ([], [], False, True),
    ([[{"$blob": "01"}]], [[{"$blob": "01"}]], False, True),
])
def test_grader_semantics(a, b, ordered, expected):
    assert compare(result(a), result(b), ordered)["correct"] is expected


def test_empty_results_still_check_arity():
    assert not compare(result([], ["a", "b"]), result([], ["x"]))["correct"]


def test_invalid_gold_is_not_a_prediction_failure():
    answer = compare(result([[1]]), {"status": "sql_error"})
    assert answer["status"] == "invalid_gold"
    assert not answer["correct"]


def test_bag_matching_not_greedy():
    # a[0] matches both gold rows, a[1] matches only gold[0].
    answer = compare(result([[0.0000009], [0.0000018]]), result([[0.0000009], [0.0000001]]))
    assert answer["correct"]


def test_audit_duplicate_and_leakage(sample):
    task, root, _ = sample
    assert audit([task], root)["tasks"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        audit([task, task], root)
    with pytest.raises(ValueError, match="leakage"):
        audit([task, dataclasses.replace(task, id="test", split="test")], root)
    with pytest.raises(ValueError, match="leakage"):
        audit([task, dataclasses.replace(task, id="test", db_id="renamed", split="test")], root)


def test_snapshot_tampering(sample):
    task, root, db = sample
    db.write_bytes(db.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        task.database(root)


def test_path_escape_and_absolute(sample, tmp_path):
    task, root, db = sample
    outside = tmp_path / "outside.sqlite"
    shutil.copy(db, outside)
    with pytest.raises(ValueError, match="escapes"):
        dataclasses.replace(task, db_file="../outside.sqlite").database(root)
    with pytest.raises(ValueError, match="relative"):
        dataclasses.replace(task, db_file=str(outside)).database(root)


def test_open_wal_rejected(sample):
    task, root, db = sample
    Path(str(db) + "-wal").write_bytes(b"active")
    with pytest.raises(ValueError, match="snapshot"):
        task.database(root)


def test_mini_dev_cannot_be_train(sample):
    task, _, _ = sample
    with pytest.raises(ValueError, match="held-out"):
        dataclasses.replace(task, source="bird-mini-dev")


def test_nonfinite_and_typed_booleans(sample):
    assert not cell_equal(float("inf"), float("inf"))
    task, _, _ = sample
    with pytest.raises(ValueError, match="booleans"):
        dataclasses.replace(task, fixture="false")
