"""Real SQLite fixture/provenance tests. No pretrained model or GPU is used."""
import copy
import json
import sqlite3
from pathlib import Path

import pytest

from exec_feedback.registered_baseline import registered_tasks
from exec_feedback.configuration import DEFAULT, load_config
from exec_feedback.core import Limits, digest, read_jsonl
from exec_feedback.datasets import load_prepared, prepare_data, validate_gold
from exec_feedback.storage import atomic_json, code_fingerprint, fingerprint, read_json


def seal(source, config):
    records = {}
    for stage, folder in (('prepare', 'prepared'), ('gold', 'gold')):
        files = {p.relative_to(source).as_posix(): digest(p)
                 for p in (source / folder).rglob('*')
                 if p.is_file() and not p.name.startswith('.')}
        if stage == 'prepare':
            files['selection.json'] = digest(source / 'selection.json')
        records[stage] = {'status': 'complete', 'artifacts': files}
    atomic_json(source / 'study.json', {
        'config': config, 'signature': fingerprint({'config': config, 'code': code_fingerprint()}),
        'stages': records, 'status': 'paused'})


@pytest.fixture
def source(tmp_path):
    raw, study = tmp_path / 'raw', tmp_path / 'source'
    raw.mkdir()
    rows = []
    for i in range(4):
        with sqlite3.connect(raw / f'db{i}.sqlite') as db:
            db.execute('CREATE TABLE numbers (n INTEGER)')
            db.execute('INSERT INTO numbers VALUES (?)', (i,))
        for j in range(3):
            rows.append({'db_id': f'db{i}', 'question_id': f'{i}-{j}',
                         'question': f'Fixture database {i} value plus {j}?', 'SQL': f'SELECT n + {j} FROM numbers'})
    atomic_json(raw / 'train.json', rows)
    c = copy.deepcopy(DEFAULT)
    c.update(version=4)
    c['data'].update(raw_root=str(raw), terms_accepted=True, validation_fraction=0.5)
    c['experiment'].update(gold_screen=True, train_limit=6, validation_limit=6, output_root=str(tmp_path / 'studies'))
    c['model']['quantization'] = 'none'
    prepare_data(c['data'], study / 'prepared')
    tasks = load_prepared(study / 'prepared')
    selection = {'train': [t.id for t in tasks if t.split == 'train'],
                 'evaluation': list(reversed([t.id for t in tasks if t.split == 'validation'])),
                 'split': 'validation', 'gold_screen_excluded': []}
    atomic_json(study / 'prepared/gold_screen/excluded.json', {'excluded': []})
    atomic_json(study / 'selection.json', selection)
    validate_gold(tasks, raw, study / 'gold', Limits(**c['limits']))
    seal(study, c)
    return study, c


def test_registered_order_all_tasks_and_train_only_smoke(source):
    study, c = source
    large = copy.deepcopy(c)
    large['model'].update(id='Qwen/Qwen3-30B-A3B-Instruct-2507', batch_size=1)
    tasks, evidence = registered_tasks(large, study)
    assert [t.id for t in tasks] == read_json(study / 'selection.json')['evaluation']
    assert len(tasks) == 6 and all(t.split == 'validation' for t in tasks)
    assert evidence['training_performed'] is False
    smoke, record = registered_tasks(large, study, train_smoke_tasks=4)
    assert len(smoke) == 4 and all(t.split == 'train' for t in smoke)
    assert record['kind'] == 'train_only_inference_smoke'


@pytest.mark.parametrize('section,key,value', [
    ('limits', 'seconds', 9), ('experiment', 'max_steps', 5),
    ('experiment', 'review_success', True), ('model', 'max_tokens', 256),
    ('model', 'max_context_tokens', 4096), ('model', 'quantization', 'nf4'),
    ('model', 'seed', 99),
])
def test_baseline_protocol_mismatch_rejected(source, section, key, value):
    study, c = source
    c = copy.deepcopy(c)
    c[section][key] = value
    with pytest.raises(ValueError, match='Baseline must'):
        registered_tasks(c, study)


def test_source_hash_and_code_mismatch_rejected(source):
    study, c = source
    old = (study / 'selection.json').read_bytes()
    atomic_json(study / 'selection.json', {'evaluation': ['changed']})
    with pytest.raises(ValueError, match='Source artifact changed'):
        registered_tasks(c, study)
    (study / 'selection.json').write_bytes(old)
    saved = read_json(study / 'study.json')
    saved['signature'] = 'wrong-code'
    atomic_json(study / 'study.json', saved)
    with pytest.raises(ValueError, match='code/config differs'):
        registered_tasks(c, study)


def test_incomplete_gold_and_incomplete_denominator_rejected(source):
    study, c = source
    saved = read_json(study / 'study.json')
    saved['stages']['gold']['status'] = 'running'
    atomic_json(study / 'study.json', saved)
    with pytest.raises(ValueError, match='gold must be complete'):
        registered_tasks(c, study)
    rows = read_jsonl(study / 'gold/gold.jsonl')
    (study / 'gold/gold.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[:-1]) + '\n', encoding='utf-8')
    seal(study, c)
    with pytest.raises(ValueError, match='does not cover'):
        registered_tasks(c, study)


def test_duplicate_and_wrong_split_selection_rejected(source):
    study, c = source
    selected = read_json(study / 'selection.json')
    selected['evaluation'].append(selected['evaluation'][0])
    atomic_json(study / 'selection.json', selected)
    seal(study, c)
    with pytest.raises(ValueError, match='duplicate'):
        registered_tasks(c, study)
    selected['evaluation'].pop()
    selected['split'] = 'test'
    atomic_json(study / 'selection.json', selected)
    seal(study, c)
    with pytest.raises(ValueError, match='validation-only'):
        registered_tasks(c, study)


def test_database_bytes_and_terms_checked(source):
    study, c = source
    blocked = copy.deepcopy(c)
    blocked['data']['terms_accepted'] = False
    with pytest.raises(ValueError, match='terms'):
        registered_tasks(blocked, study)
    with sqlite3.connect(Path(c['data']['raw_root']) / 'db0.sqlite') as db:
        db.execute('INSERT INTO numbers VALUES (100)')
    with pytest.raises(ValueError, match='Database hash mismatch'):
        registered_tasks(c, study)


@pytest.mark.parametrize('count', [0, -1, 65, 7])
def test_smoke_cannot_resample_or_silently_shrink(source, count):
    study, c = source
    with pytest.raises(ValueError):
        registered_tasks(c, study, train_smoke_tasks=count)


def test_profiles_keep_explicit_budgets_and_single_card_fallback():
    root = Path(__file__).resolve().parents[1]
    main = load_config(root / 'configs/h20-24h.json')
    small = load_config(root / 'configs/h20-single-small.json')
    large = load_config(root / 'configs/h20-30b-a3b-baseline.json')
    for c, calls in ((main, 9472), (small, 1344)):
        e = c['experiment']
        upper = e['validation_limit'] * (1 + e['max_steps'] + 2 * len(e['seeds']) * e['max_steps']) + e['train_limit'] * e['max_steps']
        assert upper == calls <= c['resources']['max_model_calls']
        assert c['resources']['max_active_seconds'] == 43200
    assert main['resources']['gpus'] == 6
    assert main['experiment']['seeds'] == [42, 43, 44]
    assert small['resources']['gpus'] == small['training']['batch_size'] == 1
    assert small['training']['batch_size'] * small['training']['gradient_accumulation_steps'] == 8
    assert large['model']['batch_size'] == 1
    assert large['resources']['max_model_calls'] == 1500


def test_inference_smoke_resume_uses_own_budget_and_never_trains(source, tmp_path, monkeypatch):
    from exec_feedback.registered_baseline import run_registered
    from exec_feedback.models import BudgetedPolicy
    from exec_feedback.pipeline import FixturePolicy

    study, c = source
    closed = []

    class TrackedFixture(FixturePolicy):
        def close(self):
            closed.append(True)

    def fixture_policy(spec, adapter=None, budget=None):
        assert adapter is None  # Baseline must never load or train a LoRA.
        return BudgetedPolicy(TrackedFixture([{'action': 'finish', 'sql': 'SELECT 1'}] * 4), budget)

    monkeypatch.setattr('exec_feedback.models.create_policy', fixture_policy)
    out = tmp_path / 'smoke'
    first = run_registered(c, study, out, direct=True, train_smoke_tasks=2)
    assert first['run_kind'] == 'engineering_fixture' and first['correct'] is None
    assert first['complete'] and first['tasks'] == 2
    assert read_json(out / 'budget.json')['calls'] == 2
    assert not (study / 'budget.json').exists()
    second = run_registered(c, study, out, direct=True, train_smoke_tasks=2)
    assert first == second and read_json(out / 'budget.json')['calls'] == 2
    assert len(closed) == 2
    changed = copy.deepcopy(c)
    changed['model']['batch_size'] = 2
    with pytest.raises(ValueError, match='config/source changed'):
        run_registered(changed, study, out, direct=True, train_smoke_tasks=2)
    with pytest.raises(ValueError, match='separate'):
        run_registered(c, study, study / 'baseline', direct=True)
    assert read_json(out / 'source-study.json')['training_performed'] is False


def test_baseline_cli_rejects_ignored_overrides_before_model_loading(source, tmp_path, capsys):
    from exec_feedback.manage import main

    study, c = source
    config = tmp_path / 'config.json'
    atomic_json(config, c)
    common = ['baseline', '--config', str(config), '--out', str(tmp_path / 'out')]
    assert main(common + ['--study', str(study), '--limit', '1']) == 2
    assert '--study' in capsys.readouterr().err
    assert main(common + ['--train-smoke-tasks', '2']) == 2
    assert '--study' in capsys.readouterr().err
    assert not (tmp_path / 'out').exists()


def test_source_screen_tampering_is_rejected(source):
    study, c = source
    selection = read_json(study / 'selection.json')
    atomic_json(study / 'prepared/gold_screen/excluded.json', {'excluded': [selection['train'][0]]})
    seal(study, c)
    with pytest.raises(ValueError, match='screen and selection disagree'):
        registered_tasks(c, study)
