"""Explicit real-model integration/throughput pilot on TRAIN data only."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

from .configuration import load_config
from .core import Limits, digest, read_jsonl
from .corpus import create_pairs
from .datasets import choose_tasks, load_prepared, prepare_data, screen_references, validate_gold
from .evaluation import evaluate_run
from .models import create_policy, load_tokenizer
from .storage import atomic_json, exclusive_lock, fingerprint, read_json


def estimate(config: dict[str, Any], tasks: int, measured_seconds: float,
             train_step_seconds: float) -> dict[str, Any]:
    if tasks <= 0 or measured_seconds <= 0 or train_step_seconds <= 0:
        raise ValueError('Positive measured throughput is required')
    e = config['experiment']
    if not e['train_limit'] or not e['validation_limit']:
        raise ValueError('Use finite task limits for a 24-hour run')
    # Treat every evaluation condition as a full feedback dialogue (conservative
    # wrt direct mode, not guaranteed conservative wrt schema/token-length shift).
    planned_tasks = e['validation_limit'] * (2 + 2 * len(e['seeds']))
    if e['collection'] != 'injected': planned_tasks += e['train_limit']
    inference = planned_tasks * measured_seconds / tasks
    training = 2 * len(e['seeds']) * config['training']['max_steps'] * train_step_seconds
    total = 1.5 * (inference + training) + 1800
    return {'measured_dialogues': tasks, 'measured_dialogue_seconds': measured_seconds,
            'measured_train_step_seconds': train_step_seconds, 'projected_inference_seconds': inference,
            'projected_training_seconds': training, 'planning_multiplier': 1.5,
            'startup_io_allowance_seconds': 1800, 'projected_total_seconds': total,
            'fits_configured_compute_cap': total <= config['resources']['max_active_seconds'],
            'warning': 'Planning estimate, not a runtime guarantee. Pilot length mix, KV cache and generated lengths can differ from the full run.'}


def cache_model(config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    spec = config['model']
    if spec.get('local_path'): raise ValueError('Local snapshot configured; do not download another model')
    path = Path(snapshot_download(spec['id'], revision=spec['revision'],
                    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.md', 'LICENSE']))
    files = {f.relative_to(path).as_posix(): digest(f) for f in sorted(path.rglob('*')) if f.is_file()}
    result = {'model': spec['id'], 'revision': spec['revision'], 'path': str(path), 'files': files}
    evidence = Path('outputs/environment')
    key = fingerprint({'model': spec['id'], 'revision': spec['revision']})
    record = evidence / 'model-snapshots' / f'{key}.json'
    with exclusive_lock(evidence / '.model-snapshots.lock', timeout=30):
        if record.exists() and read_json(record) != result:
            raise ValueError('Cached snapshot identity changed; retain the original evidence')
        if not record.exists():
            atomic_json(record, result)
        # Compatibility pointer for the first cached model, never overwritten
        # when a second model (e.g. 30B) is cached for the same experiment.
        legacy = evidence / 'model-snapshot.json'
        if not legacy.exists():
            atomic_json(legacy, result)
    return result


def run_pilot(config: dict[str, Any], out: Path, count: int = 8) -> dict[str, Any]:
    if not 2 <= count <= 64: raise ValueError('Pilot size must be 2..64')
    if config['model']['backend'] != 'transformers': raise ValueError('Pilot needs the actual local trainable model')
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'pilot-config.json', config)
    started = time.monotonic()
    prepare_data(config['data'], out / 'prepared')
    train = [t for t in load_prepared(out / 'prepared') if t.split == 'train']
    root, limits = Path(config['data']['raw_root']), Limits(**config['limits'])
    screen = None
    if config['experiment']['gold_screen']:
        # Same rule as the study: screen a 3x candidate draw, sample from what passes.
        candidates = choose_tasks(train, 3 * count, config['data']['seed'], stratified=True)
        screen = screen_references(candidates, root, out / 'gold_screen', limits)
        train = [t for t in candidates if t.id not in set(screen['excluded'])]
    tasks = choose_tasks(train, count, config['data']['seed'], stratified=True)
    if len(tasks) < count: raise ValueError('Too few pilot tasks with valid reference SQL')
    if not tasks or any(t.fixture for t in tasks): raise ValueError('Pilot requires real TRAIN data, not test or fixtures')
    gold = validate_gold(tasks, root, out / 'gold', limits)
    if gold['invalid_ids']: raise ValueError('Pilot gold preflight failed; fix data before using GPU')
    e, spec = config['experiment'], config['model']
    policy = create_policy(spec)
    try:
        common = dict(max_steps=e['max_steps'], feedback=True, limits=limits, review_success=e['review_success'])
        scalar = evaluate_run(tasks, root, policy, out / 'scalar', batch_size=1, condition='pilot_scalar', **common)
        batched = evaluate_run(tasks, root, policy, out / 'batched', batch_size=spec['batch_size'], condition='pilot_batch', **common)
    finally:
        policy.close()
    a = {x['task_id']: x for x in read_jsonl(out / 'scalar/episodes.jsonl')}
    b = {x['task_id']: x for x in read_jsonl(out / 'batched/episodes.jsonl')}
    differences = [key for key in sorted(a) if a[key]['final_sql'] != b[key]['final_sql']]
    tokenizer = load_tokenizer(spec)
    pairs = create_pairs(tasks, root, out / 'pairs', mode='injected', limits=limits,
                          max_steps=e['max_steps'], tokenizer=tokenizer,
                          max_length=config['training']['max_length'], min_pairs=1,
                          semantic=e['semantic_augmentation'], review_success=e['review_success'])
    if not pairs['minimum_pairs_met']: raise ValueError('No length-checked pilot examples')
    from .train import fit
    settings = {**config['training'], 'max_steps': 8, 'save_steps': 4, 'max_seconds': 3600}
    trained = fit(out / 'pairs/repair.jsonl', spec, settings, out / 'fit', seed=e['seeds'][0])
    metrics = read_jsonl(out / 'fit/checkpoints/metrics.jsonl')
    seconds_per_step = sum(x['step_seconds'] for x in metrics[2:]) / len(metrics[2:])
    policy = create_policy(spec, out / 'fit/adapter')
    try:
        adapted = evaluate_run(tasks, root, policy, out / 'adapted', batch_size=spec['batch_size'],
                               condition='pilot_adapter', **common)
    finally:
        policy.close()
    result = {'kind': 'real_model_integration_pilot_train_only', 'task_ids': list(a),
              'gold_screen_excluded': screen['excluded'] if screen else None,
              'adapter_reload_verified': trained['adapter_reload_verified'],
              'scalar_vs_batch_different_sql_ids': differences,
              'provider_errors': sum(v.get('statuses', {}).get('provider_error', 0) for v in (scalar, batched, adapted)),
              'estimates': estimate(config, len(tasks), batched['active_wall_seconds'], seconds_per_step),
              'peak_allocated_bytes': trained['metrics']['peak_allocated_bytes'],
              'elapsed_seconds': time.monotonic() - started,
              'not_a_quality_claim': 'Pilot uses training tasks and 8-step integration training. Do not put pilot accuracy on a resume.',
              'review_required': bool(differences)}
    atomic_json(out / 'pilot.json', result)
    if result['provider_errors']: raise RuntimeError('Pilot provider errors; inspect logs instead of starting full study')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['cache-model', 'run'])
    parser.add_argument('--config', type=Path, default=Path('local.json'))
    parser.add_argument('--out', type=Path, default=Path('outputs/pilot/first'))
    parser.add_argument('--tasks', type=int, default=8)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required for downloads or GPU work')
    config = load_config(args.config)
    result = cache_model(config) if args.action == 'cache-model' else run_pilot(config, args.out, args.tasks)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
