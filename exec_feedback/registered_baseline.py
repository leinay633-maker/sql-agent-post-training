"""A different base model on a source study's immutable, registered questions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import Limits, Task, digest, read_jsonl
from .datasets import load_prepared
from .storage import (CallBudget, atomic_json, checked_child, code_fingerprint,
                      exclusive_lock, fingerprint, read_json)


def registered_tasks(config: dict[str, Any], source: Path, *,
                     train_smoke_tasks: int | None = None) -> tuple[list[Task], dict[str, Any]]:
    """Validate before loading weights; never draw a fresh validation subset."""
    if not config['data']['terms_accepted']:
        raise ValueError('Accept the data terms before baseline execution')
    source = source.resolve(strict=True)
    saved = read_json(source / 'study.json')
    old = saved['config']
    if saved['signature'] != fingerprint({'config': old, 'code': code_fingerprint()}):
        raise ValueError('Source study code/config differs; use the same frozen code')
    artifacts = {}
    for stage, required in (
        ('prepare', {'selection.json', 'prepared/manifest.json', 'prepared/tasks.jsonl'}),
        ('gold', {'gold/summary.json', 'gold/gold.jsonl'}),
    ):
        if stage == 'prepare' and old['experiment']['gold_screen']:
            required.add('prepared/gold_screen/excluded.json')
        record = saved['stages'].get(stage, {})
        files = record.get('artifacts', {})
        if record.get('status') != 'complete' or not required <= files.keys():
            raise ValueError(f'Source {stage} must be complete with recorded artifacts')
        for name, value in files.items():
            path = checked_child(source, name)
            if not path.is_file() or digest(path) != value:
                raise ValueError(f'Source artifact changed: {name}')
        artifacts[stage] = files
    selection = read_json(source / 'selection.json')
    if selection['split'] != 'validation':
        raise ValueError('Registered baseline is validation-only; holdout needs a separate protocol')
    for key in ('train', 'evaluation'):
        ids = selection[key]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Empty or duplicate source {key} ids')
    if set(selection['train']) & set(selection['evaluation']):
        raise ValueError('Source train/evaluation overlap')
    if old['experiment']['gold_screen']:
        excluded = read_json(source / 'prepared/gold_screen/excluded.json')['excluded']
        if selection.get('gold_screen_excluded') != sorted(excluded):
            raise ValueError('Source screen and selection disagree')
        if set(excluded) & (set(selection['train']) | set(selection['evaluation'])):
            raise ValueError('Source selected a screened-out task')
        for key, limit in (('train', 'train_limit'), ('evaluation', 'validation_limit')):
            n = old['experiment'][limit]
            if n and len(selection[key]) != n:
                raise ValueError('Source selected denominator differs from its registered limit')
    gold = read_json(source / 'gold/summary.json')
    rows = read_jsonl(source / 'gold/gold.jsonl')
    wanted_gold = set(selection['train']) | set(selection['evaluation'])
    if (gold['invalid_ids'] or gold['valid'] != len(wanted_gold)
            or len(rows) != len(wanted_gold)
            or {r['task_id'] for r in rows} != wanted_gold
            or any(r['status'] != 'ok' for r in rows)):
        raise ValueError('Source gold preflight does not cover the registered tasks')
    if old.get('version') not in {3, 4} or config['version'] != old['version'] or config['limits'] != old['limits']:
        raise ValueError('Baseline must use source protocol version and SQL limits')
    for section, keys in (
        ('experiment', ('max_steps', 'review_success', 'gold_screen')),
        ('model', ('max_tokens', 'max_context_tokens', 'temperature', 'quantization', 'seed')),
    ):
        for key in keys:
            if config[section][key] != old[section][key]:
                raise ValueError(f'Baseline must match source {section}.{key}')
    # Batch shape may differ for capacity; it is frozen and reported, not called
    # numerically equivalent. Database bytes are checked even after relocation.
    tasks = load_prepared(source / 'prepared', Path(config['data']['raw_root']))
    by_id = {task.id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('Duplicate prepared task ids')
    key, split = 'evaluation', 'validation'
    if train_smoke_tasks is not None:
        if type(train_smoke_tasks) is not int or not 1 <= train_smoke_tasks <= 64:
            raise ValueError('train-smoke-tasks must be 1..64')
        if train_smoke_tasks > len(selection['train']):
            raise ValueError('Not enough registered train tasks for smoke')
        key, split = 'train', 'train'
    ids = selection[key] if train_smoke_tasks is None else selection[key][:train_smoke_tasks]
    if any(i not in by_id for i in ids):
        raise ValueError('Registered tasks are missing from prepared data')
    selected = [by_id[i] for i in ids]
    if any(t.split != split or t.fixture for t in selected):
        raise ValueError('Wrong split or fixture in registered baseline')
    evidence = {
        'source_study': str(source), 'source_signature': saved['signature'],
        'source_artifacts': artifacts, 'task_ids': ids, 'split': split,
        'kind': 'train_only_inference_smoke' if train_smoke_tasks is not None else 'registered_external_baseline',
        'source_batch_size': old['model']['batch_size'],
        'baseline_batch_size': config['model']['batch_size'],
        'training_performed': False, 'pilot_quality_claimed': False,
    }
    return selected, evidence


def run_registered(config: dict[str, Any], source: Path, out: Path, *,
                   direct: bool = False, train_smoke_tasks: int | None = None) -> dict[str, Any]:
    from .evaluation import evaluate_run
    from .models import create_policy

    source, out = source.resolve(strict=True), out.resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Keep baseline output separate from the source study')
    tasks, evidence = registered_tasks(config, source, train_smoke_tasks=train_smoke_tasks)
    steps = 1 if direct else config['experiment']['max_steps']
    if len(tasks) * steps > config['resources']['max_model_calls']:
        raise ValueError('Registered baseline maximum exceeds its independent call cap')
    evidence.update(config=config, code=code_fingerprint(), direct=direct)
    evidence['identity'] = fingerprint(evidence)
    with exclusive_lock(out / '.baseline.lock'):
        record = out / 'source-study.json'
        if record.exists():
            if read_json(record) != evidence:
                raise ValueError('Baseline config/source changed; use a new output directory')
        elif any(p.name != '.baseline.lock' for p in out.iterdir()):
            raise ValueError('Unregistered existing baseline output; use a new directory')
        else:
            atomic_json(record, evidence)
        budget = CallBudget(out / 'budget.json', config['resources']['max_model_calls'],
                            config['resources']['max_active_seconds'])
        policy = create_policy(config['model'], budget=budget)
        try:
            return evaluate_run(tasks, Path(config['data']['raw_root']), policy, out,
                                max_steps=steps, feedback=not direct,
                                limits=Limits(**config['limits']),
                                condition=evidence['kind'] + ('_direct' if direct else '_feedback'),
                                batch_size=config['model']['batch_size'],
                                review_success=config['experiment']['review_success'])
        finally:
            policy.close()
