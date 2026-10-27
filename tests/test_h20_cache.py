"""Local dummy weight bytes test cache bookkeeping; no network or model load."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from exec_feedback.configuration import DEFAULT
from exec_feedback.pilot import cache_model
from exec_feedback.storage import read_json


def test_multiple_model_records_are_immutable_and_do_not_clobber_4b(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    def download(repo, *, revision, allow_patterns):
        calls.append((repo, revision))
        folder = tmp_path / 'fake-cache' / repo.split('/')[-1]
        folder.mkdir(parents=True, exist_ok=True)
        weights = folder / 'model.safetensors'
        if not weights.exists(): weights.write_bytes(b'fixture bytes, NOT model weights')
        return str(folder)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    c = copy.deepcopy(DEFAULT)
    a = cache_model(c)
    large = copy.deepcopy(c)
    large['model'].update(id='Qwen/Qwen3-30B-A3B-Instruct-2507', revision='0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe')
    b = cache_model(large)
    records = list(Path('outputs/environment/model-snapshots').glob('*.json'))
    assert len(records) == 2
    assert {read_json(p)['model'] for p in records} == {a['model'], b['model']}
    assert read_json(Path('outputs/environment/model-snapshot.json')) == a
    assert calls[1] == (large['model']['id'], large['model']['revision'])
    assert cache_model(c) == a
    before = {p: p.read_bytes() for p in records}
    (Path(b['path']) / 'model.safetensors').write_bytes(b'changed fixture bytes')
    with pytest.raises(ValueError, match='snapshot identity changed'):
        cache_model(large)
    assert {p: p.read_bytes() for p in records} == before
    assert read_json(Path('outputs/environment/model-snapshot.json')) == a


def test_local_model_path_does_not_trigger_snapshot_download(tmp_path, monkeypatch):
    c = copy.deepcopy(DEFAULT)
    c['model']['local_path'] = str(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('no download expected')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=forbidden))
    with pytest.raises(ValueError, match='Local snapshot'):
        cache_model(c)
