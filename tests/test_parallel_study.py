"""Parallel stage scheduling: dependencies, one GPU per stage, failure and resume."""
import copy

import pytest

from exec_feedback.configuration import DEFAULT
from exec_feedback.storage import atomic_json, read_json


class FakeStages:
    """Stand-in for stage processes: each stays running for one poll, then writes its output."""

    def __init__(self, root, fail=()):
        self.root, self.fail = root, set(fail)
        self.launched, self.running, self.peak = [], set(), 0

    def __call__(self, command, *, env, **kwargs):
        name, owner = command[-1], self
        self.launched.append((name, env["CUDA_VISIBLE_DEVICES"]))
        self.running.add(name)
        self.peak = max(self.peak, len(self.running))

        class Process:
            polls = 0

            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                owner.running.discard(name)
                if name in owner.fail:
                    owner.fail.discard(name)
                    return 3
                atomic_json(owner.root / ("prepared" if name == "prepare" else name) / "summary.json", {"stage": name})
                if name == "prepare":
                    atomic_json(owner.root / "selection.json", {"train": ["a"]})
                return 0

            def kill(self):
                pass

            def wait(self):
                return 0
        return Process()


def setup(tmp_path, monkeypatch, gpus=4, fail=()):
    import exec_feedback.study as module

    config = copy.deepcopy(DEFAULT)
    config["data"].update(raw_root=str(tmp_path), terms_accepted=True)
    config["experiment"].update(output_root=str(tmp_path / "studies"), seeds=[42, 43])
    config["resources"]["gpus"] = gpus
    path = tmp_path / "config.json"
    atomic_json(path, config)
    fake = FakeStages(tmp_path / "studies" / config["name"], fail)
    monkeypatch.setattr(module.subprocess, "Popen", fake)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr("exec_feedback.study.code_revision", lambda: {"git_commit": "test"})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return module, config, path, fake


def test_parallel_study_respects_dependencies_and_gpu_slots(tmp_path, monkeypatch):
    module, config, path, fake = setup(tmp_path, monkeypatch)
    assert module.run_study(path, execute=True)["status"] == "complete"
    order = [name for name, _ in fake.launched]
    assert sorted(order) == sorted(module.stage_names(config))
    for stage, deps in module.stage_dependencies(config).items():
        assert all(order.index(d) < order.index(stage) for d in deps)
    assert 1 < fake.peak <= 4
    assert {device for _, device in fake.launched} <= {"0", "1", "2", "3"}
    manifest = read_json(tmp_path / "studies" / config["name"] / "study.json")
    assert all(s["status"] == "complete" and s["artifacts"] for s in manifest["stages"].values())


def test_parallel_failure_finishes_running_stages_and_resumes(tmp_path, monkeypatch):
    module, config, path, fake = setup(tmp_path, monkeypatch, fail=["train_repair_43"])
    with pytest.raises(RuntimeError, match="Stage train_repair_43 failed"):
        module.run_study(path, execute=True)
    root = tmp_path / "studies" / config["name"]
    manifest = read_json(root / "study.json")
    assert manifest["status"] == "interrupted"
    assert manifest["stages"]["train_repair_43"]["status"] == "interrupted"
    assert "eval_repair_43" not in manifest["stages"] and "report_43" not in manifest["stages"]
    first = len(fake.launched)
    assert module.run_study(path, execute=True)["status"] == "complete"
    again = [name for name, _ in fake.launched[first:]]
    assert "train_repair_43" in again and "prepare" not in again and "pairs" not in again


def test_parallel_until_stops_at_the_named_stage(tmp_path, monkeypatch):
    module, config, path, fake = setup(tmp_path, monkeypatch)
    assert module.run_study(path, execute=True, until="gold")["status"] == "paused"
    assert [name for name, _ in fake.launched] == ["prepare", "gold"]


def test_parallel_uses_the_visible_device_list(tmp_path, monkeypatch):
    module, config, path, fake = setup(tmp_path, monkeypatch, gpus=2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    module.run_study(path, execute=True)
    assert {device for _, device in fake.launched} == {"5", "7"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    config["name"] = "other"
    atomic_json(path, config)
    with pytest.raises(ValueError, match="exceeds the devices"):
        module.run_study(path, execute=True)


def test_export_skips_stages_still_running(tmp_path):
    from exec_feedback.handoff import export_results
    import zipfile

    study = tmp_path / "study"
    atomic_json(study / "study.json", {"status": "running"})
    atomic_json(study / "done" / "summary.json", {})
    atomic_json(study / "busy" / "summary.json", {})
    (study / "busy.log").write_text("partial", encoding="utf-8")
    result = export_results(study, tmp_path / "exports", label="t", lock=False, skip={"busy", "busy.log"})
    names = zipfile.ZipFile(tmp_path / "exports" / result["file"]).namelist()
    assert "study/done/summary.json" in names
    assert not any(n.startswith("study/busy") for n in names)


@pytest.mark.parametrize('visible', ['', '-1', '0,0', '0,,1', '0,1,'])
def test_invalid_visible_devices_never_launch(tmp_path, monkeypatch, visible):
    module, config, path, fake = setup(tmp_path, monkeypatch, gpus=2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='distinct enabled'):
        module.run_study(path, execute=True)
    assert not fake.launched


def test_launch_failure_closes_log_and_records_real_error(tmp_path, monkeypatch):
    module, config, path, fake = setup(tmp_path, monkeypatch)
    handles = []
    def cannot_start(*args, **kwargs):
        handles.append(kwargs['stdout'])
        raise OSError('simulated launch failure')
    monkeypatch.setattr(module.subprocess, 'Popen', cannot_start)
    with pytest.raises(OSError, match='simulated launch failure'):
        module.run_study(path, execute=True)
    assert handles and all(h.closed for h in handles)
    state = read_json(fake.root / 'study.json')
    assert state['status'] == 'interrupted'
    assert state['stages']['prepare']['error'] == 'OSError'


def test_parallel_export_omits_live_budget_and_final_export_includes_it(tmp_path, monkeypatch):
    module, config, path, fake = setup(tmp_path, monkeypatch)
    config['resources'].update(export_stages=True, export_root=str(tmp_path / 'exports'))
    atomic_json(path, config)
    observed = []
    def capture(study, destination, **kwargs):
        observed.append((set(fake.running), kwargs.get('skip', set())))
        return {}
    monkeypatch.setattr('exec_feedback.handoff.export_results', capture)
    module.run_study(path, execute=True)
    assert any(active and 'budget.json' in skip for active, skip in observed)
    assert observed[-1] == (set(), set())


def test_export_does_not_include_atomic_temporary_files(tmp_path):
    from exec_feedback.handoff import export_results, verify_bundle
    import zipfile
    study = tmp_path / 'study'
    atomic_json(study / 'study.json', {'status': 'paused'})
    atomic_json(study / 'budget.json', {'calls': 8})
    (study / 'budget.json.123.tmp').write_text('partial', encoding='utf-8')
    result = export_results(study, tmp_path / 'exports')
    archive = tmp_path / 'exports' / result['file']
    assert verify_bundle(archive)['verified']
    with zipfile.ZipFile(archive) as z:
        assert 'study/budget.json' in z.namelist()
        assert not any(n.endswith('.tmp') for n in z.namelist())
