import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest

from exec_feedback.cli import make_fixture
from exec_feedback.core import Limits, read_jsonl
from exec_feedback.batching import batched_episodes
from exec_feedback.evaluation import evaluate_run
from exec_feedback.models import TransformersPolicy, BudgetedPolicy
from exec_feedback.pipeline import FixturePolicy, episode
from exec_feedback.storage import BudgetExceeded, CallBudget, read_json


def test_successful_wrong_query_receives_real_feedback_without_gold(tmp_path):
    task = make_fixture(tmp_path/'db')
    wrong = 'SELECT SUM(amount) FROM orders'
    result = episode(task, tmp_path/'db', FixturePolicy([
        {'action':'finish', 'sql':wrong}, {'action':'finish', 'sql':task.gold_sql}
    ]), review_success=True)
    assert result['correct'] and len(result['steps']) == 2
    observed = result['steps'][0]['observation']
    assert observed['status'] == 'ok' and observed['rows'] == [[60.0]]
    assert 'review' in observed and 'correct' not in observed
    assert task.gold_sql not in json.dumps(result['messages'][:-1])


def test_query_feedback_does_not_repeat_correct_target_before_training(tmp_path):
    task = make_fixture(tmp_path/'db')
    result = episode(task, tmp_path/'db', FixturePolicy([
        {'action':'query', 'sql':'SELECT absent FROM orders'},
        {'action':'finish', 'sql':task.gold_sql}
    ]), review_success=True)
    assert result['correct'] and len(result['steps']) == 2
    assert task.gold_sql not in json.dumps(result['messages'][:-1])


class BatchFixture:
    kind = 'fixture'
    def __init__(self, sql):
        self.sql = sql
        self.metadata = {'provider':'batch_fixture', 'calls':0, 'usage':{}}
        self.widths = []
    def generate_batch(self, messages):
        self.widths.append(len(messages))
        return [{'text':json.dumps({'action':'finish','sql':self.sql}),
                 'usage':{'prompt_tokens':3, 'completion_tokens':2}} for _ in messages]


def test_batch_matches_scalar_transcripts_and_accounts_per_task(tmp_path):
    task = make_fixture(tmp_path/'db')
    tasks = [dataclasses.replace(task, id=f't{i}', question=f'Question {i}') for i in range(3)]
    policy = BatchFixture(task.gold_sql)
    rows = list(batched_episodes(tasks, tmp_path/'db', policy, batch_size=2, max_steps=4,
                                feedback=True, limits=Limits(), review_success=True))
    assert policy.widths == [2,2,1,1]
    for task, batched in zip(tasks, rows):
        scalar = episode(task,tmp_path/'db',FixturePolicy([{'action':'finish','sql':task.gold_sql}]*2),review_success=True)
        assert batched['messages'] == scalar['messages']
        assert batched['steps'] == scalar['steps'] and batched['correct']
        assert batched['usage'] == {'prompt_tokens':6,'completion_tokens':4}
    summary = evaluate_run(tasks, tmp_path/'db', BatchFixture(task.gold_sql), tmp_path/'eval', batch_size=2,review_success=True)
    assert summary['strict_success_rate'] is None
    assert summary['completed']==3 and summary['usage']['completion_tokens']==12
    assert summary['active_wall_seconds'] > 0


def test_batch_failure_is_not_resampled_or_silently_changed(tmp_path):
    task = make_fixture(tmp_path/'db')
    tasks = [task,dataclasses.replace(task,id='other')]
    class Broken(BatchFixture):
        def generate_batch(self, messages): raise RuntimeError('OOM')
    with pytest.raises(RuntimeError,match='OOM'):
        evaluate_run(tasks,tmp_path/'db',Broken(task.gold_sql),tmp_path/'bad',batch_size=2)
    assert read_json(tmp_path/'bad/summary.json')['statuses']['not_attempted']==2
    assert not (tmp_path/'bad/completed.json').exists()


def test_batch_reservation_is_atomic(tmp_path):
    budget=CallBudget(tmp_path/'budget.json',3,100)
    budget.reserve(2)
    with pytest.raises(BudgetExceeded): budget.reserve(2)
    assert read_json(tmp_path/'budget.json')['calls']==2
    budget.reserve()
    with pytest.raises(BudgetExceeded): budget.reserve()


def test_transformers_input_whitelist_left_padding_and_eos_accounting():
    torch=pytest.importorskip('torch')
    class Tokenizer:
        pad_token_id=0; eos_token_id=9
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['return_token_type_ids'] is False
            n=int(messages[0]['content'])
            return {'input_ids':torch.tensor([[1]*n]), 'attention_mask':torch.tensor([[1]*n]),
                    'token_type_ids':torch.tensor([[0]*n])}  # Deliberately ignore request flag.
        def decode(self, tokens, **kwargs): return str(tokens)
    class Model:
        device=torch.device('cpu')
        def generate(self, input_ids, attention_mask, **kwargs):
            assert set(kwargs)=={'max_new_tokens','do_sample','pad_token_id','eos_token_id','use_cache'}
            assert input_ids.tolist()==[[0,0,1],[1,1,1]]
            assert attention_mask.tolist()==[[0,0,1],[1,1,1]]
            return torch.cat([input_ids,torch.tensor([[7,9,0],[7,8,9]])],dim=1)
    p=object.__new__(TransformersPolicy)
    p.model,p.tokenizer=Model(),Tokenizer()
    p.spec={'temperature':0,'max_tokens':3,'max_context_tokens':10}
    p.metadata={'calls':0,'usage':{}};p.task_seed=42;p.call_index=0
    rows=p.generate_batch([[{'role':'user','content':'1'}],[{'role':'user','content':'3'}]])
    assert [r['usage']['completion_tokens'] for r in rows]==[2,3]
    assert p.metadata['usage']['prompt_tokens']==4
    assert p.metadata['calls']==2 and p.metadata['generate_batches']==1
    assert all(r['finish_reason']=='stop' for r in rows)
