import copy
import dataclasses
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from exec_feedback.cli import make_fixture
from exec_feedback.configuration import DEFAULT, initialize, load_config
from exec_feedback.core import Limits, digest, read_jsonl
from exec_feedback.corpus import create_pairs, injected_trace
from exec_feedback.datasets import load_prepared, ordered_sql, prepare_data, validate_gold
from exec_feedback.downloads import safe_extract
from exec_feedback.evaluation import evaluate_run
from exec_feedback.models import BudgetedPolicy
from exec_feedback.pipeline import FixturePolicy
from exec_feedback.reporting import compare_runs, paired_difference, wilson
from exec_feedback.storage import BudgetExceeded, CallBudget, atomic_json, read_json, recover_records


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "db"
    task = make_fixture(root)
    return task, root


@pytest.mark.parametrize("sql,ordered", [
    ("SELECT 1 ORDER BY 1", True), ("SELECT 'ORDER BY'", False),
    ("SELECT row_number() OVER (ORDER BY x) FROM t", False),
    ("SELECT * FROM (SELECT x FROM t ORDER BY x)", False),
    ("SELECT * FROM t /* ORDER BY x */", False),
    ("WITH t AS (SELECT 1 ORDER BY 1) SELECT * FROM t ORDER BY 1", True),
    ("SELECT `ORDER BY` FROM t", False),
])
def test_order_detection(sql, ordered):
    assert ordered_sql(sql) is ordered


def bird_files(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    rows = []
    for i in range(4):
        folder = root / f"shop{i}"
        folder.mkdir()
        with sqlite3.connect(folder / f"shop{i}.sqlite") as con:
            con.executescript(f"CREATE TABLE orders(id INTEGER, amount REAL); INSERT INTO orders VALUES(1,{i+10}), (2,{i+20});")
        rows.append({"question_id":i,"db_id":f"shop{i}","question":f"Revenue of shop {i}?", "SQL":"SELECT SUM(amount) FROM orders"})
    source = root / "train.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    settings = {"raw_root":str(root),"train_json":str(source),"holdout_json":"", "terms_accepted":True,"seed":42,"validation_fraction":.25}
    return settings


def test_prepare_database_split_is_locked(tmp_path):
    settings = bird_files(tmp_path)
    out = tmp_path / "prepared"
    first = prepare_data(settings, out)
    tasks = load_prepared(out)
    assert len(tasks)==4
    assert {t.db_id for t in tasks if t.split=="train"}.isdisjoint({t.db_id for t in tasks if t.split=="validation"})
    assert prepare_data(settings,out)==first
    with pytest.raises(ValueError,match="changed"):
        prepare_data({**settings,"seed":99},out)
    with (out/"tasks.jsonl").open("a") as f: f.write("{}\n")
    with pytest.raises(ValueError,match="changed"): load_prepared(out)


def test_gold_preflight_keeps_bad_queries_and_resumes(sample,tmp_path):
    task,root=sample
    bad=dataclasses.replace(task,id="bad",gold_sql="SELECT imaginary FROM orders")
    out=tmp_path/"gold"
    report=validate_gold([task,bad],root,out,workers=2)
    assert report["valid"]==1 and report["invalid_ids"]==["bad"]
    assert validate_gold([task,bad],root,out,workers=2)==report
    assert len(read_jsonl(out/"gold.jsonl"))==2


def test_actual_injected_training_feedback_is_labelled(sample,tmp_path):
    task,root=sample
    trace,kind=injected_trace(task,root,Limits(),4)
    assert trace["correct"] and trace["origin"]=="rule_perturbation"
    assert trace["steps"][0]["observation"]["status"]=="sql_error"
    report=create_pairs([task],root,tmp_path/"pairs",mode="injected",allow_fixture=True)
    assert report["origins"]=={"rule_perturbation":1}
    assert not report["research_ready"]
    a=read_jsonl(tmp_path/"pairs/success.jsonl")[0]
    b=read_jsonl(tmp_path/"pairs/repair.jsonl")[0]
    assert a["completion"]==b["completion"]
    assert len(a["prompt"])<len(b["prompt"])
    with pytest.raises(ValueError,match="training-only"):
        injected_trace(dataclasses.replace(task,split="test"),root,Limits(),4)


def test_budget_and_evaluation_restart_no_duplicate(sample,tmp_path):
    task,root=sample
    second=dataclasses.replace(task,id="second")
    budget=CallBudget(tmp_path/"budget.json",1,100)
    policy=BudgetedPolicy(FixturePolicy([{"action":"finish","sql":task.gold_sql}]),budget)
    out=tmp_path/"eval"
    with pytest.raises(BudgetExceeded): evaluate_run([task,second],root,policy,out)
    assert read_json(out/"summary.json")["completed"]==1
    policy=FixturePolicy([{"action":"finish","sql":task.gold_sql}])
    result=evaluate_run([task,second],root,policy,out)
    assert result["completed"]==2 and result["strict_success_rate"] is None
    assert len(read_jsonl(out/"episodes.jsonl"))==2
    with pytest.raises(ValueError,match="changed"):
        evaluate_run([task,dataclasses.replace(second,question="changed")],root,FixturePolicy([]),out)


def test_torn_journal_tail_is_preserved(tmp_path):
    path=tmp_path/"rows.jsonl"
    path.write_bytes(b'{"ok":1}\n{"broken":')
    assert recover_records(path)==[{"ok":1}]
    assert list(tmp_path.glob("*.torn-*"))[0].read_bytes()==b'{"broken":'
    path.write_bytes(b'NOT JSON\n')
    with pytest.raises(json.JSONDecodeError): recover_records(path)


def test_budget_reservations_persist_on_failure(tmp_path):
    path=tmp_path/"b.json"
    budget=CallBudget(path,2,100)
    budget.reserve();budget.reserve()
    with pytest.raises(BudgetExceeded): CallBudget(path,2,100).reserve()
    with pytest.raises(ValueError,match="differs"): CallBudget(path,3,100)


@pytest.mark.parametrize("name",["../escape","/root/x","C:/x","a\\b"])
def test_unsafe_archives_rejected(tmp_path,name):
    archive=tmp_path/"bad.zip"
    wire_name = name.replace("\\", "_")
    with zipfile.ZipFile(archive,"w") as z: z.writestr(wire_name,"x")
    if "\\" in name:
        archive.write_bytes(archive.read_bytes().replace(wire_name.encode(), name.encode()))
    with pytest.raises(ValueError,match="Unsafe"): safe_extract(archive,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_extract_budget_and_valid_archive(tmp_path):
    archive=tmp_path/"ok.zip"
    with zipfile.ZipFile(archive,"w") as z: z.writestr("nested/data.sql","SELECT 1;")
    with pytest.raises(ValueError,match="limits"): safe_extract(archive,tmp_path/"small",max_bytes=1)
    assert safe_extract(archive,tmp_path/"out")["files"]==1
    assert (tmp_path/"out/nested/data.sql").read_text(encoding="utf-8")=="SELECT 1;"


def test_configuration_no_unknown_switch_or_overwrite(tmp_path):
    path=tmp_path/"local.json"
    initialize(path)
    assert Path(load_config(path)["data"]["raw_root"]).is_absolute()
    with pytest.raises(ValueError,match="overwritten"): initialize(path)
    atomic_json(path,{"model":{"allow_dowload":True}})
    with pytest.raises(ValueError,match="Unknown"): load_config(path)


def test_paired_cluster_statistics_and_fixture_rejection(sample,tmp_path):
    a=[{"task_id":"1","db_id":"a","correct":False},{"task_id":"2","db_id":"b","correct":True}]
    b=[{**a[0],"correct":True},a[1]]
    report=paired_difference(a,b,resamples=100)
    assert report["delta_right_minus_left"]==.5
    assert report["database_cluster_bootstrap_95ci"] is not None
    assert wilson(0,0) is None and wilson(0,10)[0]==0
    with pytest.raises(ValueError): paired_difference(a,b[:1])
    task,root=sample
    for name in ("a","b"):
        evaluate_run([task],root,FixturePolicy([{"action":"finish","sql":task.gold_sql}]),tmp_path/name)
    with pytest.raises(ValueError,match="fixture"):
        compare_runs({"a":tmp_path/"a","b":tmp_path/"b"},tmp_path/"report")


def test_finish_execution_error_can_be_repaired(tmp_path):
    from exec_feedback.cli import make_fixture
    from exec_feedback.pipeline import FixturePolicy, episode
    task=make_fixture(tmp_path/'db')
    policy=FixturePolicy([{'action':'finish','sql':'SELECT absent FROM orders'},
                          {'action':'finish','sql':task.gold_sql}])
    result=episode(task,tmp_path/'db',policy)
    assert result['correct'] and len(result['steps'])==2
    assert result['steps'][0]['observation']['status']=='sql_error'


def test_completed_evaluation_cannot_be_rewritten(tmp_path):
    from exec_feedback.cli import make_fixture
    from exec_feedback.pipeline import FixturePolicy
    from exec_feedback.evaluation import evaluate_run
    task=make_fixture(tmp_path/'db')
    policy=FixturePolicy([{'action':'finish','sql':task.gold_sql}])
    out=tmp_path/'run'
    evaluate_run([task],tmp_path/'db',policy,out)
    (out/'episodes.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='modified'):
        evaluate_run([task],tmp_path/'db',FixturePolicy([]),out)
