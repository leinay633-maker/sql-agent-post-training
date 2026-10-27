"""Stage transactions are tested separately from the actual optimizer tests."""
import copy
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from exec_feedback.configuration import DEFAULT
from exec_feedback.storage import atomic_json,read_json
from exec_feedback.study import run_study, stage_names


def test_stage_resume_avoids_redo_and_checks_completed_files(tmp_path,monkeypatch):
    import exec_feedback.study as module
    config=copy.deepcopy(DEFAULT)
    config['data'].update(raw_root=str(tmp_path),terms_accepted=True)
    config['experiment']['output_root']=str(tmp_path/'studies')
    path=tmp_path/'config.json';atomic_json(path,config)
    monkeypatch.setattr(module,'stage_names',lambda c:['prepare','gold'])
    attempts=[]
    failed=[False]
    def stage_process(command,**kwargs):
        name=command[-1];attempts.append(name)
        target=tmp_path/'studies'/config['name']
        if name=='gold' and not failed[0]:
            failed[0]=True;return SimpleNamespace(returncode=2)
        output=target/('prepared' if name=='prepare' else name)
        atomic_json(output/'summary.json',{'stage':name})
        if name=='prepare':atomic_json(target/'selection.json',{'train':['one']})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',stage_process)
    monkeypatch.setattr('exec_feedback.study.code_revision',lambda:{'git_commit':'test'})
    with pytest.raises(RuntimeError,match='Stage gold failed'): run_study(path,execute=True)
    assert read_json(tmp_path/'studies'/config['name']/'study.json')['status']=='interrupted'
    assert run_study(path,execute=True)['status']=='complete'
    assert attempts==['prepare','gold','gold']
    artifact=tmp_path/'studies'/config['name']/'prepared/summary.json'
    artifact.write_text('{}')
    with pytest.raises(ValueError,match='artifacts changed'):run_study(path,execute=True)


def test_all_required_lifecycle_stages_exist():
    names=stage_names(DEFAULT)
    for stage in ['prepare','gold','base_direct','base_feedback','collect','pairs',
                  'train_success_42','train_repair_42','eval_success_42','eval_repair_42','report_42']:
        assert stage in names


def test_test_split_cannot_be_a_training_study(tmp_path):
    from exec_feedback.configuration import load_config
    path=tmp_path/'config.json';atomic_json(path,{'experiment':{'final_test':True}})
    with pytest.raises(ValueError,match='validation only'):load_config(path)
