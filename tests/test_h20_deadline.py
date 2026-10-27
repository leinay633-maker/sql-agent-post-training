"""Real Linux deadline wrapper tests; no CUDA, network or billing API."""
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'docs/h20_deadline.sh'
pytestmark = pytest.mark.skipif(os.name != 'posix' or not shutil.which('timeout'),
                                reason='Linux/GNU timeout wrapper')


def run(at, *args):
    return subprocess.run(['bash', str(SCRIPT), '--stop-at', at, *args],
                          capture_output=True, text=True, timeout=10)


def test_expired_deadline_does_not_start_command(tmp_path):
    marker = tmp_path / 'started'
    result = run('2000-01-01T00:00:00Z', sys.executable, '-c',
                 f'from pathlib import Path; Path({str(marker)!r}).touch()')
    assert result.returncode == 124 and not marker.exists()


def test_deadline_requires_timezone():
    result = run('2030-01-01T00:00:00', sys.executable, '-c', 'pass')
    assert result.returncode == 2 and 'timezone' in result.stderr


def test_deadline_sets_absolute_epoch_and_preserves_command_exit():
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)).isoformat()
    result = run(future, sys.executable, '-c',
                 'import os,sys; print(os.environ["EFA_STOP_AT_UNIX"]); sys.exit(7)')
    assert result.returncode == 7 and int(result.stdout) == int(dt.datetime.fromisoformat(future).timestamp())


def test_deadline_interrupts_actual_child(tmp_path):
    marker = tmp_path / 'interrupted'
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=3)).isoformat()
    code = ('import signal, time, sys\nfrom pathlib import Path\n'
            f'def stop(*args):\n Path({str(marker)!r}).touch(); sys.exit(0)\n'
            'signal.signal(signal.SIGINT, stop)\ntime.sleep(30)\n')
    result = run(future, sys.executable, '-c', code)
    assert result.returncode == 124 and marker.exists()
