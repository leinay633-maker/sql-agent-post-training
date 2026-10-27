"""v0.3.1: pre-selection reference screening, larger VM-step cap, BIRD AppleDouble files."""
import copy
import io
import json
import sqlite3
import zipfile

import pytest

from exec_feedback.configuration import DEFAULT, validate_config
from exec_feedback.core import Limits
from exec_feedback.storage import read_json
from exec_feedback.study import execute_stage


def test_vm_step_cap_allows_large_scans_but_stays_bounded():
    assert Limits(vm_steps=1_000_000_000).vm_steps == 1_000_000_000
    with pytest.raises(ValueError, match="instruction"):
        Limits(vm_steps=2_000_000_001)


def screened_config(tmp_path, screen, train_limit=0, validation_limit=0):
    root = tmp_path / "raw"
    root.mkdir()
    rows = []
    for i in range(6):
        folder = root / f"shop{i}"
        folder.mkdir()
        with sqlite3.connect(folder / f"shop{i}.sqlite") as con:
            con.executescript(f"CREATE TABLE orders(id INTEGER, amount REAL); INSERT INTO orders VALUES(1,{i+10});")
        for q in range(2):
            sql = "SELECT SUM(amount) FROM orders"
            if (i, q) == (0, 1):
                sql = "SELECT missing_column FROM orders"  # broken upstream reference
            rows.append({"question_id": i * 2 + q, "db_id": f"shop{i}",
                         "question": f"Revenue {i}/{q}?", "SQL": sql})
    (root / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    config = copy.deepcopy(DEFAULT)
    config["data"].update(raw_root=str(root), train_json=str(root / "train.json"),
                          terms_accepted=True, validation_fraction=0.5)
    config["version"] = 4 if screen else 3
    config["experiment"].update(output_root=str(tmp_path / "studies"), train_limit=train_limit,
                                validation_limit=validation_limit, gold_screen=screen)
    validate_config(config)
    return config, tmp_path / "studies" / config["name"]


def test_gold_screen_is_its_own_config_version(tmp_path):
    config, _ = screened_config(tmp_path, screen=True)
    config["version"] = 3
    with pytest.raises(ValueError, match="version 4"):
        validate_config(config)
    config["version"], config["experiment"]["gold_screen"] = 4, False
    with pytest.raises(ValueError, match="version 4"):
        validate_config(config)


def test_screen_excludes_broken_reference_before_selection(tmp_path):
    # shop0 (holding the broken reference) lands in train under seed 42: 5 valid train tasks.
    config, study = screened_config(tmp_path, screen=True, train_limit=5, validation_limit=6)
    execute_stage(config, study, "prepare")
    selection = read_json(study / "selection.json")
    assert "shop0" in selection["train_databases"]
    assert selection["gold_screen_excluded"] == ["bird-train:1"]
    assert "bird-train:1" not in selection["train"] + selection["evaluation"]
    assert (len(selection["train"]), len(selection["evaluation"])) == (5, 6)
    screen = read_json(study / "prepared" / "gold_screen" / "excluded.json")
    assert screen["candidates"] == 12 and screen["statuses"] == {"sql_error": 1}
    assert "missing_column" in screen["reasons"]["bird-train:1"]["reason"]
    assert execute_stage(config, study, "gold")["invalid_ids"] == []


def test_screen_refuses_to_shrink_the_registered_sample(tmp_path):
    config, study = screened_config(tmp_path, screen=True, train_limit=6, validation_limit=6)
    with pytest.raises(ValueError, match="silently shrink"):
        execute_stage(config, study, "prepare")


def test_machine_failures_are_retried_not_excluded(tmp_path, monkeypatch):
    import exec_feedback.datasets as datasets

    real, calls = datasets.execute, {}
    def flaky(database, sql, limits):
        key = (str(database), sql)
        calls[key] = calls.get(key, 0) + 1
        if "shop1" in str(database) and calls[key] == 1:
            return {"status": "worker_error", "error": "worker crashed"}
        if "shop2" in str(database) and calls[key] == 1:
            return {"status": "timeout", "error": "interrupted"}
        return real(database, sql, limits)
    monkeypatch.setattr(datasets, "execute", flaky)
    config, study = screened_config(tmp_path, screen=True)
    execute_stage(config, study, "prepare")
    screen = read_json(study / "prepared" / "gold_screen" / "excluded.json")
    assert screen["excluded"] == ["bird-train:1"]
    assert set(screen["retried_serially"].values()) == {"ok"} and len(screen["retried_serially"]) == 2

    monkeypatch.setattr(datasets, "execute", lambda database, sql, limits: {"status": "sql_error", "error": "database is locked"})
    tasks = datasets.load_prepared(study / "prepared")
    with pytest.raises(RuntimeError, match="machine failures"):
        datasets.screen_references(tasks, tmp_path / "raw", tmp_path / "locked")


def test_report_states_what_the_screen_removed(tmp_path):
    from tests.test_reporting import fixture_run
    from exec_feedback.reporting import compare_runs

    fixture_run(tmp_path / "a", [True, False]); fixture_run(tmp_path / "b", [True, True])
    screen = {"candidates": 12, "excluded": ["bird-train:1"], "statuses": {"sql_error": 1}}
    report = compare_runs({"a": tmp_path / "a", "b": tmp_path / "b"}, tmp_path / "r", screen=screen)
    assert report["gold_screen"] == screen
    for name in ("report.md", "report.html"):
        assert "1 of 12 candidate tasks excluded" in (tmp_path / "r" / name).read_text(encoding="utf-8")


def test_without_screen_the_gold_gate_still_stops(tmp_path):
    config, study = screened_config(tmp_path, screen=False)
    execute_stage(config, study, "prepare")
    assert "gold_screen_excluded" not in read_json(study / "selection.json")
    assert not (study / "prepared" / "gold_screen").exists()
    with pytest.raises(ValueError, match="preflight failed"):
        execute_stage(config, study, "gold")


def test_fetch_bird_ignores_macos_appledouble_zip(tmp_path, monkeypatch):
    import exec_feedback.downloads as downloads

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("train_databases/shop/shop.sqlite", b"SQLite format 3\x00")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("train/train.json", "[]")
        z.writestr("train/train_databases.zip", inner.getvalue())
        z.writestr("__MACOSX/train/._train_databases.zip", b"\x00\x05\x16\x07" + b"\x00" * 60)
    monkeypatch.setattr(downloads.urllib.request, "urlopen",
                        lambda request, timeout: io.BytesIO(outer.getvalue()))
    manifest = downloads.fetch_bird("train", tmp_path / "raw", accept_terms=True)
    assert manifest["sqlite_databases"] == 1
    assert (tmp_path / "raw" / "train.source.json").exists()
    assert (tmp_path / "raw" / "train" / "__MACOSX" / "train" / "._train_databases.zip").exists()


def test_dot_underscore_zip_outside_macosx_is_still_inspected(tmp_path, monkeypatch):
    import exec_feedback.downloads as downloads

    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("train/train.json", "[]")
        z.writestr("train/._hidden.zip", b"not a zip")
    monkeypatch.setattr(downloads.urllib.request, "urlopen",
                        lambda request, timeout: io.BytesIO(outer.getvalue()))
    with pytest.raises(zipfile.BadZipFile):
        downloads.fetch_bird("train", tmp_path / "raw", accept_terms=True)


def test_v04_prepare_reuses_source_screen(tmp_path, monkeypatch):
    import exec_feedback.models as models
    from exec_feedback.core import digest, write_jsonl
    from exec_feedback.datasets import load_prepared
    from exec_feedback.rl import MODEL_ID, MODEL_REVISION
    from exec_feedback.rl.config import DEFAULT as V04
    from exec_feedback.rl.study import prepare
    from exec_feedback.storage import atomic_json

    config, study = screened_config(tmp_path, screen=True)
    execute_stage(config, study, "prepare")
    artifacts = {name: digest(study / name) for name in ("prepared/gold_screen/excluded.json", "selection.json")}
    atomic_json(study / "study.json", {"status": "complete", "config": config,
                                       "stages": {"prepare": {"status": "complete", "artifacts": artifacts}}})
    tasks = load_prepared(study / "prepared")
    train = [t for t in tasks if t.split == "train"]
    warm = study / "train_repair_42"
    pairs = study / "pairs" / "repair.jsonl"
    write_jsonl(pairs, [{"task_id": t.id, "split": "train", "fixture": False,
                         "db_sha256": t.db_sha256} for t in train if t.id != "bird-train:1"])
    atomic_json(warm / "result.json", {"identity": "w", "adapter_hash": "h",
                                       "training_completed": True, "adapter_reload_verified": True})
    atomic_json(warm / "manifest.json", {"identity": "w", "fixture": False, "data_sha256": digest(pairs),
                                         "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                                         "origins": {"natural": 1}})
    monkeypatch.setattr(models, "model_identity", lambda spec, adapter=None: {"adapter_hash": "h"})
    valid_train = [t for t in train if t.id != "bird-train:1"]
    assert len(valid_train) < len(train)
    settings = {**copy.deepcopy(V04), "phase": "pilot", "validation_tasks": 0,
                "train_tasks": len(valid_train), "limits": config["limits"]}
    frozen = prepare(study, tmp_path / "raw", tmp_path / "v04", settings)
    assert frozen["source_gold_screen_excluded"] == ["bird-train:1"]
    assert frozen["source_protocol"] == "efa-v3.1-gold-screen"
    chosen = {json.loads(line)["id"] for line in (tmp_path / "v04" / "train.jsonl").read_text().splitlines()}
    assert chosen == {t.id for t in valid_train}
    record = study / "prepared" / "gold_screen" / "excluded.json"
    atomic_json(record, {**read_json(record), "excluded": []})
    with pytest.raises(ValueError, match="changed after the study"):
        prepare(study, tmp_path / "raw", tmp_path / "v04b", settings)
