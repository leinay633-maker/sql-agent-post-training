import copy
import datetime as dt
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from exec_feedback.configuration import DEFAULT
from exec_feedback.handoff import _write_bundle, package_code, export_results, verify_bundle
from exec_feedback.storage import atomic_json, read_json
from exec_feedback.window import deadline, run_window


def test_source_bundle_only_committed_project_no_git_credentials(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    for args in (['init','-q'],['config','user.name','Test'],['config','user.email','test@example.invalid']):
        subprocess.run(['git',*args],cwd=repo,check=True)
    root=repo/'projects'/'agent';root.mkdir(parents=True)
    (root/'code.py').write_text('print("ok")\n',encoding='utf-8')
    (repo/'private.txt').write_text('not part of project',encoding='utf-8')
    subprocess.run(['git','add','.'],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','fixture'],cwd=repo,check=True)
    out=tmp_path/'source.zip'; result=package_code(root,out)
    assert verify_bundle(out,result['sha256'])['verified']
    with zipfile.ZipFile(out) as z:
        assert set(z.namelist())=={'BUNDLE.json','exec-feedback-agent/code.py'}
    (root/'code.py').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Commit'):package_code(root,tmp_path/'dirty.zip')


def test_result_bundle_preserves_complete_checkpoint_and_detects_tamper(tmp_path):
    study=tmp_path/'study';atomic_json(study/'study.json',{'status':'interrupted','signature':'abc'})
    atomic_json(study/'fit/checkpoint-1/complete.json',{'ok':True})
    (study/'fit/checkpoint-1/model.pt').write_bytes(b'complete checkpoint fixture')
    atomic_json(study/'fit/checkpoint-2/state.json',{'partial':True})
    (study/'exclude.sqlite').write_bytes(b'database')
    exports=tmp_path/'exports';result=export_results(study,exports)
    archive=exports/result['file']
    assert verify_bundle(archive,result['sha256'])['verified']
    assert read_json(exports/'latest.json')==result
    with zipfile.ZipFile(archive) as z:
        assert 'study/fit/checkpoint-1/model.pt' in z.namelist()
        assert not any('checkpoint-2' in name or name.endswith('.sqlite') for name in z.namelist())
    with pytest.raises(ValueError,match='SHA-256'):verify_bundle(archive,'0'*64)
    with pytest.raises(ValueError,match='outside'):export_results(study,study/'exports')


def test_bundle_path_and_duplicate_protection(tmp_path):
    f=tmp_path/'x';f.write_text('x',encoding='utf-8')
    for name in ('../escape','/absolute','a\\b','C:x','BUNDLE.json'):
        with pytest.raises(ValueError):_write_bundle([(f,name)],tmp_path/'bad.zip',{'kind':'test'})
    with pytest.raises(ValueError,match='Duplicate'):
        _write_bundle([(f,'x'),(f,'x')],tmp_path/'bad.zip',{'kind':'test'})


def test_deadline_absolute_timezone_and_reserve():
    expiry='2026-10-10T12:00:00+08:00'
    end=dt.datetime.fromisoformat(expiry).timestamp()
    assert deadline(expiry,3600,now=end-7200)==end-3600
    with pytest.raises(ValueError,match='timezone'):deadline('2026-10-10T12:00:00',3600,now=0)
    with pytest.raises(ValueError,match='No compute'):deadline(expiry,3600,now=end-100)


@pytest.mark.skipif(__import__('os').name!='posix',reason='Linux lease runner')
def test_window_preserves_exit_failure_and_exports_without_resetting_lease(tmp_path,monkeypatch):
    import exec_feedback.window as w
    config=copy.deepcopy(DEFAULT);config['experiment']['output_root']=str(tmp_path/'studies')
    config['resources']['export_root']=str(tmp_path/'exports')
    path=tmp_path/'config.json';atomic_json(path,config)
    root=tmp_path/'studies'/config['name'];atomic_json(root/'study.json',{'status':'interrupted','signature':'test'})
    expiry='2099-01-01T00:00:00Z'
    class Process:
        def __init__(self,args,**kwargs):
            assert kwargs['start_new_session'] and 'EFA_STOP_AT_UNIX' in kwargs['env']
        def wait(self,timeout):return 2
    monkeypatch.setattr(w.subprocess,'Popen',Process)
    result=run_window(path,expiry)
    assert result['exit_code']==2 and not result['cloud_instance_stopped']
    assert verify_bundle(tmp_path/'exports'/result['export']['file'])['verified']
    with pytest.raises(ValueError,match='deadline changed'):
        run_window(path,'2099-01-02T00:00:00Z')


def test_stage_pause_is_resumable_and_does_not_mark_complete(tmp_path,monkeypatch):
    import exec_feedback.study as m
    config=copy.deepcopy(DEFAULT);config['data'].update(raw_root=str(tmp_path),terms_accepted=True)
    config['experiment']['output_root']=str(tmp_path/'studies')
    path=tmp_path/'config.json';atomic_json(path,config)
    monkeypatch.setattr(m,'stage_names',lambda _:['prepare','gold']);calls=[]
    def stage(command,**kwargs):
        name=command[-1];calls.append(name);root=tmp_path/'studies'/config['name']
        atomic_json(root/('prepared' if name=='prepare' else name)/'summary.json',{'stage':name})
        if name=='prepare':atomic_json(root/'selection.json',{'train':['a']})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',stage)
    # Provenance uses subprocess too; isolate that read, not the stage assertions.
    monkeypatch.setattr('exec_feedback.study.code_revision',lambda:{'git_commit':'test'})
    assert m.run_study(path,execute=True,until='prepare')['status']=='paused'
    assert calls==['prepare']
    assert m.run_study(path,execute=True)['status']=='complete'
    assert calls==['prepare','gold']
