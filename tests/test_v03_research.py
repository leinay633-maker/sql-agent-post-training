import copy
import dataclasses
import json
from pathlib import Path

import pytest

from exec_feedback.cli import make_fixture
from exec_feedback.configuration import DEFAULT, load_config
from exec_feedback.core import Limits, read_jsonl
from exec_feedback.corpus import create_pairs, injected_trace
from exec_feedback.datasets import choose_tasks
from exec_feedback.pilot import estimate
from exec_feedback.pipeline import build_sft
from exec_feedback.storage import atomic_json
from exec_feedback.study import plan


def test_executable_semantic_mutation_changes_result_not_gold_feedback(tmp_path):
    task=make_fixture(tmp_path/'db')
    trace,kind=injected_trace(task,tmp_path/'db',Limits(),4,semantic=True,review_success=True)
    assert kind.startswith('semantic_') and trace['correct']
    assert trace['steps'][0]['observation']['status']=='ok'
    assert task.gold_sql not in json.dumps(trace['messages'][:-1])
    rows=build_sft([task],tmp_path/'db','repair',[trace],allow_fixture=True)
    assert rows[0]['completion']==json.dumps({'action':'finish','sql':task.gold_sql})
    forged=copy.deepcopy(trace);forged['steps'][0]['observation']['rows']=[[999]]
    with pytest.raises(ValueError,match='replay'):
        build_sft([task],tmp_path/'db','repair',[forged],allow_fixture=True)


def test_mixed_semantic_corpus_identical_targets_and_no_copy_leak(tmp_path):
    task=make_fixture(tmp_path/'db')
    summary=create_pairs([task],tmp_path/'db',tmp_path/'pairs',mode='injected',allow_fixture=True,
                         semantic=True,review_success=True)
    assert summary['paired_tasks']==1 and summary['semantic_augmentation']
    a=read_jsonl(tmp_path/'pairs/success.jsonl')[0];b=read_jsonl(tmp_path/'pairs/repair.jsonl')[0]
    assert a['completion']==b['completion'] and a['prompt'][:2]==b['prompt'][:2]
    assert task.gold_sql not in json.dumps(b['prompt'])
    assert b['origin']=='rule_perturbation' and b['error_type'].startswith('semantic_')


def test_stratified_sampling_covers_databases_and_is_order_invariant(tmp_path):
    task=make_fixture(tmp_path/'db')
    tasks=[dataclasses.replace(task,id=f'{db}-{i}',db_id=f'db{db}',difficulty=['easy','hard'][i%2])
           for db in range(5) for i in range(10)]
    a=choose_tasks(tasks,15,42,stratified=True)
    b=choose_tasks(list(reversed(tasks)),15,42,stratified=True)
    assert a==b and len({x.db_id for x in a})==5
    assert all(sum(x.db_id==f'db{db}' for x in a)==3 for db in range(5))
    assert set(x.id for x in a)<=set(x.id for x in tasks)


def test_h20_profile_budget_no_quantization_and_no_hidden_download(tmp_path):
    profile=Path(__file__).resolve().parents[1]/'configs/h20-24h.json'
    c=load_config(profile)
    assert c['version']==4 and c['experiment']['gold_screen'] and c['model']['quantization']=='none'
    assert c['model']['batch_size']==4 and not c['model']['allow_download']
    assert c['experiment']['validation_limit']==256
    assert plan(c)['maximum_model_calls_estimate']==9472 and c['experiment']['seeds']==[42,43,44]
    assert plan(c)['maximum_model_calls_estimate']<c['resources']['max_model_calls']
    e=estimate(c,8,80,3)
    assert e['projected_total_seconds']>e['projected_inference_seconds']
    with pytest.raises(ValueError): estimate(c,0,80,3)


def test_sampling_must_be_greedy(tmp_path):
    c=copy.deepcopy(DEFAULT);c['model'].update(batch_size=4,temperature=1)
    file=tmp_path/'bad.json';atomic_json(file,c)
    with pytest.raises(ValueError,match='greedy|temperature'):
        load_config(file)
