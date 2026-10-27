"""Run a study inside a real lease deadline; preserve time for local retrieval."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .configuration import load_config
from .handoff import export_results
from .storage import atomic_json, read_json
from .study import study_dir


def deadline(value: str, reserve_seconds: float, now: float | None = None) -> float:
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Expiry must include timezone, e.g. +08:00 or Z')
    if reserve_seconds < 60:
        raise ValueError('Reserve at least 60 seconds for export/retrieval')
    stop = parsed.timestamp() - reserve_seconds
    if stop <= (time.time() if now is None else now):
        raise ValueError('No compute time left before the retrieval reserve')
    return stop


def terminate_group(process: subprocess.Popen, grace: float = 15) -> None:
    if process.poll() is not None: return
    if os.name != 'posix':
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if os.name == 'posix': os.killpg(process.pid, signal.SIGKILL)
        else: process.kill()
        process.wait()


def run_window(config_path: Path, expires_at: str, reserve_seconds: float = 3600) -> dict[str, Any]:
    if os.name != 'posix':
        raise ValueError('The lease runner targets the Linux GPU host')
    config = load_config(config_path)
    stop = deadline(expires_at, reserve_seconds)
    root = study_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    record = {'expires_at': expires_at, 'stop_at_unix': stop, 'reserve_seconds': reserve_seconds,
              'started_unix': time.time(), 'cloud_instance_stopped': False}
    record_path = root / 'lease.json'
    if record_path.exists():
        saved = read_json(record_path)
        if saved['expires_at'] != expires_at or saved['reserve_seconds'] != reserve_seconds:
            raise ValueError('Lease deadline changed; never silently reset a rental clock')
    atomic_json(record_path, record)
    env = {**os.environ, 'EFA_STOP_AT_UNIX': str(stop), 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'}
    process = None
    error = None
    try:
        with (root / 'lease-driver.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-m', 'exec_feedback', 'study', '--config', str(config_path.resolve()), '--execute'],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=max(0.01, stop-time.time()))
            except subprocess.TimeoutExpired:
                terminate_group(process); code = 124
    except BaseException as exc:
        error = exc
        if process is not None: terminate_group(process)
        code = 130
    finally:
        record.update(exit_code=code, finished_unix=time.time())
        atomic_json(record_path, record)
        if (root / 'study.json').exists():
            record['export'] = export_results(root, Path(config['resources']['export_root']), label='window-exit')
            atomic_json(record_path, record)
    if error: raise error
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('local.json'))
    parser.add_argument('--expires-at', required=True, help='Actual provider expiry, including timezone; not now+24h')
    parser.add_argument('--reserve-seconds', type=float, default=3600)
    args = parser.parse_args()
    result = run_window(args.config, args.expires_at, args.reserve_seconds)
    print(__import__('json').dumps(result, indent=2))
    raise SystemExit(result['exit_code'])


if __name__ == '__main__': main()
