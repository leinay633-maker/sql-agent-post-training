"""v0.11 engineering tests: SQLite fixtures, a character tokenizer, a scripted engine and a randomly initialised tiny
Qwen3 with a LoRA (all from test_v09); no GPU, vLLM or real weights. Nothing here is a model result."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v09 import GOLD, ChatTokenizer, ScriptedEngine, fake_adapter, factory, make_db, tiny_policy  # noqa: E402

from exec_feedback.core import Task, digest, write_jsonl  # noqa: E402
from exec_feedback.storage import fingerprint, read_json, recover_records  # noqa: E402
from exec_feedback.v8.freeze import expected_identity, mark_verified, model_spec  # noqa: E402
from exec_feedback.v11 import MODELS, PROTOCOL, V08_PROTOCOL, V09_PROTOCOL  # noqa: E402
from exec_feedback.v11 import online  # noqa: E402
from exec_feedback.v11 import prepass as pre  # noqa: E402
from exec_feedback.v11.config import DEFAULT, SAME_AS_V09_RL, validate_settings  # noqa: E402
from exec_feedback.v11.evaluate import adapter_for, common_steps, evaluate, parse  # noqa: E402
from exec_feedback.v11.freeze import freeze, load, same_training  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def smoke_settings() -> dict:
    return json.loads((REPO / "configs" / "v11-smoke.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(dict(DEFAULT))
    assert DEFAULT["protocol"] == PROTOCOL
    assert json.loads((REPO / "configs" / "h20-v11.json").read_text(encoding="utf-8")) == DEFAULT
    smoke = smoke_settings()
    validate_settings(smoke)
    assert smoke["phase"] == "smoke" and (smoke["planned_steps"], smoke["explore_steps"]) == (2, 3)
    models = json.loads((REPO / "configs" / "v11-models.json").read_text(encoding="utf-8"))
    assert set(models) == {MODELS["student"][0]} and models[MODELS["student"][0]]["revision"] == MODELS["student"][1]
    for bad in ({"prepass_samples": 4}, {"explore_steps": 200}, {"explore_steps": 405}, {"boundary_gate": [0.6, 0.05]},
                {"boundary_gate": [0.05]}, {"curve_every": 15}, {"rl_slots": 64}, {"max_steps": 6},
                {"seeds": [42, 42]}):
        with pytest.raises(ValueError):
            validate_settings({**DEFAULT, **bad})
    with pytest.raises(ValueError):
        validate_settings({k: v for k, v in DEFAULT.items() if k != "boundary_gate"})


def test_training_settings_are_v09s_rl(tmp_path):
    """The one change from v0.9's RL is the task pool: every training setting is checked against v0.9's at the freeze."""
    v09 = json.loads((REPO / "configs" / "h20-v09.json").read_text(encoding="utf-8"))
    assert {k: DEFAULT[k] for k in SAME_AS_V09_RL} == {k: v09[k] for k in SAME_AS_V09_RL}
    (tmp_path / "v09").mkdir()
    (tmp_path / "v09" / "manifest.json").write_text(json.dumps({"settings": v09}), encoding="utf-8")
    same_training(tmp_path, dict(DEFAULT))
    with pytest.raises(ValueError, match="learning_rate"):
        same_training(tmp_path, {**DEFAULT, "learning_rate": 5e-05})
    same_training(tmp_path, {**smoke_settings(), "learning_rate": 5e-05})  # smoke settings are never results


def test_condition_names():
    assert parse("B") == (None, None, None)
    assert parse("PTR_seed43") == ("PTR", 43, None) and parse("RL9_seed42") == ("RL9", 42, None)
    assert parse("RLb_seed42") == ("RLb", 42, None) and parse("RLb_seed44@400") == ("RLb", 44, 400)
    for bad in ("PTR_seed42@10", "RL9_seed42@200", "RL_seed42", "RAFT_seed42", "RLb", "RLb_seed42@", "RLb400_seed42"):
        with pytest.raises(ValueError):
            parse(bad)


# ------------------------------------------------------------------------------------------------ boundary rule
def row(task: str, k: int, correct: bool, status: str | None = None) -> dict:
    status = status or ("correct" if correct else "wrong_result")
    return {"key": f"{task}#{k}", "task_id": task, "k": k, "verdict": status, "correct": correct,
            "valid": status != "oracle_error", "turns": 2}


def test_boundary_tasks_are_solved_sometimes_but_not_always():
    pool = [Task(f"t{i}", "db", "train", "q", GOLD, "db.sqlite", "0" * 64, "bird-train", "sha256:test") for i in range(5)]
    rows = [row("t0", k, False) for k in range(4)] + [row("t1", k, k == 0) for k in range(4)]
    rows += [row("t2", k, k < 3) for k in range(4)] + [row("t3", k, True) for k in range(4)]
    rows += [row("t4", k, False, "oracle_error" if k == 2 else None) for k in range(4)]  # unscorable gold: left out
    summary = pre.summarise(rows, pool, 4)
    assert summary["boundary"] == ["t1", "t2"] and summary["invalid"] == ["t4"] and summary["scorable"] == 4
    assert summary["histogram"] == {"0": 1, "1": 1, "3": 1, "4": 1} and summary["boundary_share"] == 0.5
    assert summary["boundary_sha256"] == pre.hashlib.sha256(json.dumps(["t1", "t2"]).encode()).hexdigest()
    with pytest.raises(ValueError, match="expected 4"):
        pre.summarise(rows[:-1], pool, 4)  # a task with a dialogue missing is never counted


def test_three_degenerate_behaviours():
    q = lambda sql: {"action": {"action": "query", "sql": sql}}
    f = {"action": {"action": "finish", "sql": GOLD}}
    bad = {"output": "not json", "action": None}
    assert online.shape([q("a"), f]) == {"unparsed": False, "no_query": False, "repeat": False}
    assert online.shape([bad, q("a"), f]) == {"unparsed": True, "no_query": False, "repeat": False}
    assert online.shape([f]) == {"unparsed": False, "no_query": True, "repeat": False}
    assert online.shape([q("a"), q("a"), f])["repeat"] and not online.shape([q("a"), q("b"), q("a")])["repeat"]
    assert online.shape([]) == {"unparsed": False, "no_query": False, "repeat": False}
    counts = online.behaviour([{"steps": [f]}, {"steps": [bad, f]}, {"steps": []}, {}])
    assert counts == {"dialogues": 2, "unparsed": 1, "no_query": 2, "repeat": 0}  # rejected dialogues have no steps


# ------------------------------------------------------------------------------------------------ fixture run
def v09_settings() -> dict:
    return json.loads((REPO / "configs" / "h20-v09.json").read_text(encoding="utf-8"))


@pytest.fixture
def fixture_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dbs = {"train": ("shop", make_db(raw / "shop.sqlite", [1, 2, 4])),
           "validation": ("shop_v", make_db(raw / "shop_v.sqlite", [1, 2, 5])),
           "test": ("shop_t", make_db(raw / "shop_t.sqlite", [1, 3, 4]))}

    def task(i: int, split: str, source: str, difficulty: str = "unknown") -> Task:
        db_id, sha = dbs[split]
        return Task(f"{source}:{i}", db_id, split, f"Question {i}?", GOLD, f"{db_id}.sqlite", sha, source, "sha256:test",
                    difficulty=difficulty)
    roles = {"pool": [task(i, "train", "bird-train") for i in range(8)],
             "dev": [task(i, "train", "bird-train") for i in range(20, 24)],
             "validation": [task(i, "validation", "bird-train") for i in range(30, 34)],
             "test": [task(i, "test", "bird-dev", "simple" if i % 2 else "moderate") for i in range(40, 44)]}
    reg = tmp_path / "registration"
    reg.mkdir()
    for role, tasks in roles.items():
        write_jsonl(reg / f"{role}.jsonl", [dataclasses.asdict(t) for t in tasks])
    source = {"protocol": "efa-v5-rft-natural-v1", "counts": {r: len(t) for r, t in roles.items()},
              "sha256": {r: digest(reg / f"{r}.jsonl") for r in roles}, "test_screen": {"candidates": 5}}
    v05 = fingerprint(source)
    (reg / "manifest.json").write_text(json.dumps({**source, "identity": v05}), encoding="utf-8")
    model_id, revision = MODELS["student"]
    folder = tmp_path / "models" / "student"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"kind": "student"}), encoding="utf-8")
    files = {"config.json": {"size": (folder / "config.json").stat().st_size, "sha256": digest(folder / "config.json")}}
    mark_verified(folder, files)
    (tmp_path / "models.json").write_text(json.dumps({model_id: {"revision": revision, "files": files}}), encoding="utf-8")
    snapshot = expected_identity(model_spec("student", folder, DEFAULT), files)["local_snapshot"]
    imports = tmp_path / "imports"
    v08 = {"protocol": V08_PROTOCOL, "registration": {"source_identity": v05},
           "models": {"student": {"identity": {"local_snapshot": snapshot}}}}
    v08["identity"] = fingerprint(v08)
    (imports / "v08").mkdir(parents=True)
    (imports / "v08" / "manifest.json").write_text(json.dumps(v08), encoding="utf-8")
    v09 = {"protocol": V09_PROTOCOL, "registration": {"source_identity": v05},
           "models": {"student": {"identity": {"local_snapshot": snapshot}}}, "imports": {"v08_run": v08["identity"]},
           "settings": v09_settings()}
    v09["identity"] = fingerprint(v09)
    (imports / "v09").mkdir(parents=True)
    (imports / "v09" / "manifest.json").write_text(json.dumps(v09), encoding="utf-8")
    chosen = {}
    for seed in (42, 43, 44):
        base = imports / "adapters" / f"PTR_seed{seed}"
        h = fake_adapter(base / "adapter", f"PTR{seed}")
        (base / "result.json").write_text(json.dumps({"training_completed": True, "adapter_reload_verified": True,
                                                      "identity": f"i{seed}", "adapter_hash": h}), encoding="utf-8")
        (base / "manifest.json").write_text(json.dumps({"identity": f"i{seed}", "run": v08["identity"], "family": "PTR",
                                                        "seed": seed}), encoding="utf-8")
        lora = imports / "v09" / "adapters" / f"RL_seed{seed}"
        h = fake_adapter(lora / "adapter", f"RL{seed}")
        (lora / "result.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
        (lora / "manifest.json").write_text(json.dumps({"run": v09["identity"], "family": "RL", "seed": seed}),
                                            encoding="utf-8")
        (lora / "milestones.json").write_text(json.dumps({"3": h}), encoding="utf-8")
        chosen[f"RL_seed{seed}"] = {"step": 3, "adapter_hash": h, "short": False}
    (imports / "v09" / "steps.json").write_text(json.dumps({"run": v09["identity"], "k": 3, "evaluate": chosen}),
                                                encoding="utf-8")
    train = imports / "v09" / "train" / "RL_seed42"  # v0.9's records of the arm, set beside the boundary runs
    train.mkdir(parents=True)
    (train / "metrics.jsonl").write_text("".join(json.dumps(metric(step)) + "\n" for step in (1, 2, 3)),
                                         encoding="utf-8")
    (train / "result.json").write_text(json.dumps({"finished": True, "stopped_by": "planned", "last_milestone": 3}),
                                       encoding="utf-8")
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, smoke_settings(), models_root=tmp_path / "models",
                    manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100, v08_run=v08["identity"],
                    v09_run=v09["identity"])
    return run, frozen


def metric(step: int, behaviour: dict | None = None) -> dict:
    """One per-step record as the online loop writes it (the fields the report reads)."""
    return {"step": step, "optimizer_updates": step, "reward_mean": 0.5, "informative_groups": 2,
            "all_correct_groups": 1, "all_failed_groups": 1, "trained_trajectories": 8, "entropy": 0.4,
            "mean_turns": 2.0, "mean_completion_tokens": 30.0, "mean_abs_logp_gap": 0.001, "max_abs_logp_gap": 0.01,
            "tis_clipped_fraction": 0.0, "grad_norm": 0.1, "step_seconds": 40.0, "generate_seconds": 20.0,
            "train_seconds": 15.0, "active_seconds": 40.0 * step, "tasks": [],
            **({"behaviour": behaviour} if behaviour is not None else {})}


def test_freeze_verifies_both_imports(fixture_run, tmp_path):
    run, frozen = fixture_run
    assert frozen["protocol"] == PROTOCOL and frozen["counts"] == {"pool": 8, "dev": 4, "validation": 4, "test": 4}
    assert set(frozen["imports"]["adapters"]) == {"PTR_seed42"} and set(frozen["imports"]["v09"]["rl"]) == {"RL_seed42"}
    assert frozen["imports"]["v09"]["k"] == 3 and load(run)[0]["identity"] == frozen["identity"]
    reg, raw, imports = tmp_path / "registration", tmp_path / "raw", tmp_path / "imports"
    kw = dict(models_root=tmp_path / "models", manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100,
              v08_run=frozen["imports"]["v08_run"])
    with pytest.raises(ValueError, match="registered v0.9 run"):
        freeze(reg, raw, tmp_path / "other", smoke_settings(), **kw)  # the real v0.9 identity is not this fixture's
    (imports / "v09" / "adapters" / "RL_seed42" / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree|changed"):
        freeze(reg, raw, tmp_path / "other2", smoke_settings(), **kw, v09_run=frozen["imports"]["v09"]["v09_run"])


def scripted(holder: dict | None = None):
    engines: list[ScriptedEngine] = []

    def make(spec, settings, *, seed):
        engines.append(ScriptedEngine(holder if holder is not None else {}, seed))
        return engines[-1]
    return make, engines


# ------------------------------------------------------------------------------------------------ pre-pass
def test_prepass_counts_every_task_resumes_by_chunk_and_stops_at_a_deadline(fixture_run, monkeypatch):
    run, frozen = fixture_run
    make, engines = scripted()
    out = pre.prepass_dir(run, 42)
    real, calls = pre.sample, []

    def crash_on_second(*a, **kw):
        calls.append(len(a[2]))
        if len(calls) == 2:
            raise RuntimeError("simulated crash in the second chunk")
        return real(*a, **kw)

    monkeypatch.setattr(pre, "sample", crash_on_second)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pre.prepass(run, seed=42, engine_factory=make, tokenizer=ChatTokenizer())
    first = recover_records(out / "rows.jsonl")
    assert len(first) == 4 * 4 and not (out / "result.json").exists()  # one chunk of 4 tasks x 4 dialogues
    # a crash while journalling: rows of a task that is not complete are set aside, the task is sampled again
    done = {r["task_id"] for r in first}
    torn = {**first[0], "task_id": next(t.id for t in load(run, "pool")[1] if t.id not in done), "key": "x#0", "k": 0}
    with (out / "rows.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(torn) + "\n")
    calls.clear()
    monkeypatch.setattr(pre, "sample", lambda *a, **kw: (calls.append(len(a[2])), real(*a, **kw))[1])
    result = pre.prepass(run, seed=42, engine_factory=make, tokenizer=ChatTokenizer())
    assert calls == [4] and list(out.glob("rows.partial-*.jsonl"))  # only the second chunk ran again
    rows = recover_records(out / "rows.jsonl")
    per_task = {t.id: [r for r in rows if r["task_id"] == t.id] for t in load(run, "pool")[1]}
    assert all(sorted(r["k"] for r in v) == [0, 1, 2, 3] for v in per_task.values())
    counts = {t: sum(r["correct"] for r in v) for t, v in per_task.items()}
    assert result["finished"] and result["scorable"] == 8 and result["gate"]["passed"]
    assert result["boundary"] == sorted(t for t, c in counts.items() if 0 < c < 4)
    assert sum(result["histogram"].values()) == 8 and engines[-1].versions == [0]  # sampled with the start LoRA
    assert pre.prepass(run, seed=42, engine_factory=make, tokenizer=ChatTokenizer()) == result  # never twice
    tasks, sha = pre.boundary_tasks(run, frozen, load(run, "pool")[1], 42)
    assert sha == result["boundary_sha256"] and ([t.id for t in tasks] == result["boundary"] or not result["boundary"])
    # a failed gate is final: nothing trains on that seed
    saved = read_json(out / "result.json")
    (out / "result.json").write_text(json.dumps({**saved, "gate": {**saved["gate"], "passed": False}}), encoding="utf-8")
    assert pre.gate_failed(run, 42)
    with pytest.raises(ValueError, match="failed its gate"):
        pre.boundary_tasks(run, frozen, load(run, "pool")[1], 42)
    # a deadline before the first chunk: nothing kept, the attempt recorded, the next attempt resumes
    other = run.parent / "other"
    shutil.copytree(run, other, ignore=shutil.ignore_patterns("prepass"))
    past = dt.datetime.now(dt.timezone.utc).isoformat()
    started = len(engines)
    stopped = pre.prepass(other, seed=42, until=past, engine_factory=make, tokenizer=ChatTokenizer())
    assert stopped == {"finished": False, "stopped_by": "deadline", "remaining": 8}
    assert len(engines) == started  # no engine loaded for a deadline already past
    assert not (pre.prepass_dir(other, 42) / "result.json").exists()
    assert recover_records(pre.prepass_dir(other, 42) / "attempts.jsonl")[0]["stopped_by"] == "deadline"


# ------------------------------------------------------------------------------------------------ online run
def test_boundary_run_pauses_at_k_then_continues_to_the_exploratory_end(fixture_run):
    """The tiny model on the boundary tasks only: two steps to K, a pause, the continuation to step 3 from the K
    checkpoint; then a run whose deadline comes before its first step."""
    pytest.importorskip("torch")
    run, frozen = fixture_run
    holder: dict = {}
    make, engines = scripted(holder)
    pre.prepass(run, seed=42, engine_factory=make, tokenizer=ChatTokenizer())
    boundary, _ = pre.boundary_tasks(run, frozen, load(run, "pool")[1], 42)
    loader = tiny_policy(holder, checkpointing=True)
    out = run / "train" / "RLb_seed42"
    other = run.parent / "late"
    shutil.copytree(run, other)
    first = online.train_online(run, seed=42, until=None, stop_step=2, engine_factory=make, policy_loader=loader)
    assert first["finished"] is False and first["stopped_by"] == "paused" and first["steps_completed"] == 2
    assert not (out / "result.json").exists() and online.trained_to(out) == 2 and online.ended(out) is None
    assert first["lora_check"]["passed"]
    metrics = recover_records(out / "metrics.jsonl")
    ids = {t.id for t in boundary}
    assert [r["step"] for r in metrics] == [1, 2] and all(set(r["tasks"]) <= ids for r in metrics)
    assert all(r["behaviour"]["dialogues"] == 16 and r["behaviour"]["no_query"] == 0 for r in metrics)
    assert [r["boundary_pass"] for r in metrics] == [0, 4 // len(ids)]
    manifest = read_json(out / "manifest.json")
    assert manifest["plan"]["pool"] == "boundary" and manifest["plan"]["boundary_tasks"] == len(ids)
    held = online.train_online(run, seed=42, until=None, stop_step=2, engine_factory=make, policy_loader=loader)
    assert held["stopped_by"] == "paused" and len(recover_records(out / "metrics.jsonl")) == 2  # not trained twice
    final = online.train_online(run, seed=42, until=None, stop_step=3, engine_factory=make, policy_loader=loader)
    assert final["finished"] and final["steps_completed"] == 3 and final["stopped_by"] == "planned"
    assert set(final["milestones"]) == {"1", "2", "3"} and final["lora_check"] == first["lora_check"]
    assert [r["step"] for r in recover_records(out / "metrics.jsonl")] == [1, 2, 3]
    assert engines[-1].versions[0] == 2  # the continuation starts from the checkpoint at K
    curve = [r for r in recover_records(out / "curve.jsonl") if "correct" in r]
    assert sorted(r["step"] for r in curve) == [0, 1, 2, 3] and all("behaviour" in r for r in curve)
    assert online.train_online(run, seed=42, until=None, stop_step=2, engine_factory=make,
                               policy_loader=loader) == final  # an ended run is never trained again
    with pytest.raises(ValueError, match="registered milestone"):
        online.train_online(run, seed=42, until=None, stop_step=1, engine_factory=make, policy_loader=loader)
    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)).isoformat()
    late = online.train_online(other, seed=42, until=soon, stop_step=2, engine_factory=make, policy_loader=loader)
    assert late["finished"] and late["stopped_by"] == "deadline" and late["steps_completed"] == 0
    assert online.ended(other / "train" / "RLb_seed42") == late and online.trained_to(other / "train" / "RLb_seed42") == 0


# ------------------------------------------------------------------------------------------------ evaluation, report
def fake_train(run: Path, seed: int, leg: str, behaviour: bool = True) -> None:
    """What a run leaves behind after its main leg (milestones to K, paused) or its continuation (on to explore_steps,
    ended), without training."""
    frozen, _ = load(run, check_code=False)
    s = frozen["settings"]
    out = run / "train" / f"RLb_seed{seed}"
    end = s["planned_steps"] if leg == "main" else s["explore_steps"]
    marks = read_json(out / "milestones.json") if (out / "milestones.json").exists() else {}
    for step in range(len(marks) + 1, end + 1):
        marks[str(step)] = fake_adapter(out / "milestones" / f"step-{step:04d}" / "adapter", f"RLb{seed}-{step}")
    (out / "milestones.json").write_text(json.dumps(marks), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"run": frozen["identity"], "identity": "x",
                                                   "plan": {"boundary_tasks": 5}}), encoding="utf-8")
    shares = {"dialogues": 16, "unparsed": 1, "no_query": 2, "repeat": 0} if behaviour else None
    (out / "metrics.jsonl").write_text("".join(json.dumps(metric(i, shares)) + "\n" for i in range(1, end + 1)),
                                       encoding="utf-8")
    record = {"steps_completed": end, "last_milestone": end, "optimizer_updates": end, "lora_check": {"passed": True}}
    if leg == "main":
        (out / "paused.json").write_text(json.dumps({**record, "step": end, "stopped_by": "paused"}), encoding="utf-8")
    else:
        (out / "result.json").write_text(json.dumps({**record, "finished": True, "stopped_by": "planned"}),
                                         encoding="utf-8")


def fake_prepass(run: Path, seed: int, passed: bool = True) -> None:
    out = pre.prepass_dir(run, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({
        "finished": True, "seed": seed, "tasks": 8, "scorable": 8, "invalid": [], "boundary": ["a", "b"],
        "boundary_share": 0.25, "histogram": {"0": 3, "1": 1, "3": 1, "4": 3}, "statuses": {"correct": 12},
        "mean_turns": 2.0, "gate": {"range": [0.05, 0.6], "passed": passed}, "last_attempt": {"wall_seconds": 60.0}}),
        encoding="utf-8")


def test_common_milestone_rule_ignores_the_continuation(fixture_run, monkeypatch):
    from exec_feedback.v11 import evaluate as ev

    run, frozen = fixture_run
    fake_train(run, 42, "main")
    fake_train(run, 42, "explore")
    result = common_steps(run)
    assert result["k"] == 2 and result["short"] == [] and result["evaluate"]["RLb_seed42"]["step"] == 2  # not 3
    folder, h, step = adapter_for(run, frozen, "RLb_seed42")
    assert step == 2 and folder.parent.name == "step-0002"
    assert adapter_for(run, frozen, "RLb_seed42@3")[2] == 3 and adapter_for(run, frozen, "PTR_seed42")[2] == 0
    arm = adapter_for(run, frozen, "RL9_seed42")
    assert arm[2] == 3 and arm[0] == run / "imports" / "v09" / "adapters" / "RL_seed42" / "adapter"
    with pytest.raises(ValueError, match="no milestone"):
        adapter_for(run, frozen, "RLb_seed42@7")
    with pytest.raises(ValueError, match="Unregistered"):
        adapter_for(run, frozen, "RL9_seed43")
    # v0.9's short rule over three seeds: a run below half of the best does not pull K down
    s = dict(frozen["settings"], seeds=[42, 43, 44], planned_steps=30)
    monkeypatch.setattr(ev, "load", lambda r, role=None, **kw: ({**frozen, "settings": s}, []))
    for seed, steps in ((42, 40), (43, 20), (44, 10)):
        out = run / "train" / f"RLb_seed{seed}"
        shutil.rmtree(out, ignore_errors=True)
        marks = {str(i): fake_adapter(out / "milestones" / f"step-{i:04d}" / "adapter", f"{seed}-{i}")
                 for i in range(10, steps + 1, 10)}
        (out / "milestones.json").write_text(json.dumps(marks), encoding="utf-8")
    result = ev.common_steps(run)
    assert result["last_milestones"] == {"RLb_seed42": 30, "RLb_seed43": 20, "RLb_seed44": 10}  # 40 is past K's cap
    assert result["k"] == 20 and result["short"] == ["RLb_seed44"] and result["evaluate"]["RLb_seed44"]["step"] == 10


def test_evaluate_and_both_reports_end_to_end(fixture_run):
    from exec_feedback.v11.analysis import progress_entry, report

    run, frozen = fixture_run
    fake_prepass(run, 42)
    fake_train(run, 42, "main")
    common_steps(run)
    for condition in ("B", "PTR_seed42", "RL9_seed42", "RLb_seed42"):
        for role in ("dev", "validation", "test"):
            result = evaluate(run, condition=condition, role=role, stop=None, factory=factory)
            assert result["complete"] and result["correct"] == 2  # even question numbers
    assert read_json(run / "eval" / "test" / "RL9_seed42" / "summary.json")["step"] == 3
    main = report(run, name="main", procs=1)
    test = main["roles"]["test"]["project"]
    assert {"RLb-PTR", "RLb-RL9", "RL9-PTR", "PTR-B", "RLb-B", "RL9-B"} <= set(test["deltas"])
    assert not any(k.startswith("RLb3") for k in test["deltas"])
    decision = main["decisions"]["RLb-PTR"]
    assert decision["decided"] and decision["holds"] is False and decision["seeds"] == 1  # ties
    assert test["conditions"]["B"]["dialogues"] == 4 and test["conditions"]["B"]["no_query"] == 0
    assert main["prepass"]["PTR_seed42"]["boundary"] == 2 and main["dynamics"]["RLb_seed42"]["stopped_by"] == "paused"
    assert main["dynamics"]["RLb_seed42"]["passes"] == pytest.approx(2 * 4 / 5)
    assert main["dynamics"]["RLb_seed42"]["blocks"][0]["no_query"] == pytest.approx(2 / 16)
    assert main["dynamics"]["RL9_seed42"]["steps"] == 3 and "blocks" not in main["dynamics"]["RL9_seed42"]
    assert main["errors"]["test"]["conditions"]["B"]["outcomes"] == {"correct": 2, "wrong_values": 2}
    text = (run / "report" / "main.md").read_text(encoding="utf-8")
    assert text.startswith("# v0.11 预注册报告") and "预采样" in text and "三种行为" in text and "错误分析：test" in text
    assert (run / "report" / "report.md").read_text(encoding="utf-8") == text
    # the continuation, evaluated at explore_steps, joins the final report as RLb3 (exploratory)
    fake_train(run, 42, "explore")
    for role in ("dev", "validation", "test"):
        assert evaluate(run, condition="RLb_seed42@3", role=role, stop=None, factory=factory)["complete"]
    (run / "eval" / "dev" / "RLb_seed42@1").mkdir()  # an ad hoc milestone stays out of the report
    final = report(run, name="final", procs=1)
    test = final["roles"]["test"]["project"]
    assert {"RLb3-PTR", "RLb3-RLb"} <= set(test["deltas"]) and "RLb3_seed42" in test["conditions"]
    assert final["decisions"] == {k: v for k, v in main["decisions"].items()}  # the decisions never see the continuation
    assert "RLb3_seed42" in final["errors"]["test"]["conditions"] and final["dynamics"]["RLb_seed42"]["finished"]
    assert read_json(run / "report" / "main.json")["report"] == "main" and read_json(run / "report" / "report.json")["report"] == "final"
    assert "（探索）" in (run / "report" / "final.md").read_text(encoding="utf-8")
    entry = progress_entry(run, "M:report-final", "bundle-x")
    assert "预采样 PTR_seed42：边界题 2/8" in entry and "RLb_seed42：3 步（planned）" in entry and "比较点 K = 2" in entry
    assert "RLb-PTR：不成立（1 个 seed）" in entry


# ------------------------------------------------------------------------------------------------ scheduler
FAKE_CLI = r'''
import sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_v11 as t
from exec_feedback.v11 import analysis
from exec_feedback.v11.evaluate import common_steps, evaluate
from exec_feedback.v11.prepass import prepass

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
run = Path(opt("--run"))
if (run / f"FAIL_{{cmd}}_{{opt('--leg') or opt('--name') or ''}}").exists():  # a path made to fail in this run
    raise SystemExit(1)
if cmd == "prepass":
    make, _ = t.scripted()
    prepass(run, seed=int(opt("--seed")), until=opt("--until"), engine_factory=make, tokenizer=t.ChatTokenizer())
elif cmd == "train":
    t.fake_train(run, int(opt("--seed")), opt("--leg"))
elif cmd == "steps":
    common_steps(run)
elif cmd == "evaluate":
    evaluate(run, condition=opt("--condition"), role=opt("--role"), stop=opt("--stop"), factory=t.factory)
elif cmd == "report":
    analysis.report(run, name=opt("--name"), procs=1)
else:
    raise SystemExit("unexpected command " + cmd)
'''


def main_copy(smoke: Path, tmp_path: Path) -> Path:
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["smoke_limit"] = 0  # the copy plays the main run
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return main


def schedule(smoke: Path, main: Path, frozen: dict, tmp_path: Path, stop: float, **extra):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v11.orchestrate import Plan

    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py=sys.executable,
                vllm_py=sys.executable, stop_epoch=stop, gpus=[0], **extra)
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=[0], stop=stop, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs", plan=plan,
                          poll=0.1, stagger=0, retry_delay=0, milestone=reached.append)
    plan.scheduler = scheduler
    return plan, scheduler, reached


def test_scheduler_runs_the_smoke_then_the_main_plan_on_one_gpu(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    out = tmp_path / "regroup.json"
    start = Path(frozen["models_root"]) / "merged" / "PTR_seed42"
    start.mkdir(parents=True)
    start.with_name("PTR_seed42.json").write_text("{}", encoding="utf-8")
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600,
                                        regroup={"run": tmp_path / "v10", "start": start, "out": out})
    plan.regroup_command = lambda: [sys.executable, "-c", f"open({str(out)!r}, 'w').write('{{}}')"]
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status and " exit 3" not in status, status
    for run in (smoke, main):
        assert all((run / "report" / f"{n}.json").exists() for n in ("main", "final"))
    for role in ("dev", "validation", "test"):
        for condition in ("B", "PTR_seed42", "RL9_seed42", "RLb_seed42", "RLb_seed42@3"):
            assert (main / "eval" / role / condition / "summary.json").exists(), (role, condition)
    first = lambda text: status.index(text)
    # the whole smoke run first; trainings take the whole GPU, evaluations and pre-passes half of it
    assert first("start M:") > first("start S:report-final")
    assert re.search(r"start M:train-RLb_seed42 on GPU 0 \(full\)", status)
    assert re.search(r"start M:prepass-PTR_seed42 on GPU 0 \(half\)", status)
    assert first("start S:explore-RLb_seed42") > first("start S:report-main")
    assert first("start M:prepass-PTR_seed42") < first("start M:train-RLb_seed42") < first("start M:steps")
    assert first("start M:steps") < first("start M:eval-RLb_seed42-test") < first("start M:report-main")
    # the preregistered report, then the 4% replay, then the continuation, its evaluations and the final report
    assert first("start M:report-main") < first("start R:regroup") < first("start M:explore-RLb_seed42")
    assert first("start M:explore-RLb_seed42") < first("start M:eval-RLb_seed42@3-test") < first("start M:report-final")
    assert out.exists() and read_json(main / "report" / "final.json")["decisions"]["RLb-PTR"]["decided"]
    assert {"S:report-final", "M:eval-start", "M:prepass", "M:train", "M:steps", "M:eval-k", "M:report-main",
            "R:regroup", "M:explore", "M:eval-explore", "M:report-final"} <= set(reached)


def three_seed_copy(smoke: Path, tmp_path: Path) -> Path:
    main = main_copy(smoke, tmp_path)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["settings"]["seeds"] = [42, 43, 44]
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return main


def test_continuations_queue_on_one_gpu_and_the_one_that_no_longer_fits_is_noted(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v11.orchestrate import EXPLORE_RESERVE, RUN_OVERHEAD, SCHEDULER_RESERVE, Plan

    smoke, frozen = fixture_run
    main = three_seed_copy(smoke, tmp_path)
    for seed in (42, 43, 44):
        fake_train(main, seed, "main")  # paused at K = 2; 40 s per measured step
    need = 1 * 1.1 * 40.0 + RUN_OVERHEAD  # one step to explore_steps = 3
    stop = time.time() + EXPLORE_RESERVE + 2.5 * need  # two continuations fit before the cutoff, three do not
    status = tmp_path / "STATUS.txt"

    def plan_on(gpus: list[int]) -> Plan:
        plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                    stop_epoch=stop, gpus=gpus)
        plan.scheduler = Scheduler(gpus=gpus, stop=stop, status=status, log_dir=tmp_path / "logs", plan=plan,
                                   reserve=SCHEDULER_RESERVE)
        return plan

    one = plan_on([0])
    assert one.queued_need(44) == pytest.approx(3 * need)
    assert one.fits(42)() and one.fits(43)() and not one.fits(44)() and not one.fits(44)()
    assert status.read_text(encoding="utf-8").count("not starting M:explore-RLb_seed44:") == 1
    fake_train(main, 42, "explore")  # one continuation done: the last one now fits
    assert one.fits(44)()
    assert plan_on([0, 1]).queued_need(44) == pytest.approx(need + need / 2)  # two GPUs share the queue


def test_a_report_does_not_wait_for_an_evaluation_that_can_no_longer_start(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v11.orchestrate import SCHEDULER_RESERVE, Plan

    run, frozen = fixture_run
    plan = Plan(run=run, smoke_run=run, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 9 * 3600, gpus=[0])
    plan.scheduler = Scheduler(gpus=[0], stop=time.time() + 9 * 3600, status=tmp_path / "STATUS.txt",
                               log_dir=tmp_path / "logs", plan=plan, reserve=SCHEDULER_RESERVE)
    fake_prepass(run, 42)
    fake_train(run, 42, "main")
    common_steps(run)
    report = {j.name: j for j in plan()[0]}["S:report-main"]
    assert not report.deps()  # the evaluations are still to run
    plan.scheduler.stop = time.time() + 20 * 60  # none of them would end before stop - 15 min any more
    assert report.deps()


def test_export_lists_the_adapters_and_skips_a_run_without_milestones(fixture_run, tmp_path):
    run, frozen = fixture_run
    fake_prepass(run, 42)
    fake_train(run, 42, "main")
    common_steps(run)
    fake_train(run, 42, "explore")
    early = run / "train" / "RLb_seed43"  # a run stopped before its first milestone
    early.mkdir()
    (early / "manifest.json").write_text("{}", encoding="utf-8")
    layout = tmp_path / "repo" / "outputs" / "v11"
    layout.mkdir(parents=True)
    shutil.copytree(run, layout / "run")
    script = (REPO / "scripts" / "v11_export.sh").read_text(encoding="utf-8")
    code = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\n", script, re.S).group(1)
    done = subprocess.run([sys.executable, "-", str(layout / "run")], input=code, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    listed = {Path(line).as_posix() for line in done.stdout.splitlines()}
    prefix = "outputs/v11/run/train/"
    steps = {re.search(r"step-(\d+)", p).group(1) for p in listed if "/milestones/" in p}
    assert steps == {"0002", "0003"}  # K and explore_steps (also the last milestone); not step 1
    assert f"{prefix}RLb_seed42/result.json" in listed and f"{prefix}RLb_seed43/manifest.json" in listed
    assert not any(p.startswith(f"{prefix}RLb_seed43/milestones") for p in listed)


def test_a_path_the_smoke_failed_holds_back_only_that_path(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    (smoke / "FAIL_train_main").write_text("", encoding="utf-8")
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert status.count("done S:train-RLb_seed42 exit 1") == 3 and "start M:train-" not in status
    # what did work in the smoke runs on the full sets, and both reports come inside the schedule
    assert (main / "prepass" / "PTR_seed42" / "result.json").exists()
    assert (main / "eval" / "test" / "RL9_seed42" / "summary.json").exists()
    assert (main / "report" / "main.json").exists() and (main / "report" / "final.json").exists()
    assert read_json(main / "steps.json")["k"] == 0 and "M:report-final" in reached


def test_a_preregistered_report_that_gave_up_does_not_hold_back_the_continuation(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    (main / "FAIL_report_main").write_text("", encoding="utf-8")
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert status.count("done M:report-main exit 1") == 2 and "start M:explore-RLb_seed42" in status
    assert (main / "eval" / "test" / "RLb_seed42@3" / "summary.json").exists()
    assert not (main / "report" / "main.json").exists() and read_json(main / "report" / "report.json")["report"] == "final"


def test_a_short_booking_keeps_the_preregistered_part(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600)
    plan.step_seconds = lambda: 40000.0  # one more step would end past the exploratory cutoff
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert status.count("not starting M:explore-RLb_seed42:") == 1 and "start M:explore-" not in status
    assert (main / "report" / "main.json").exists() and (main / "report" / "final.json").exists()
    assert not (main / "eval" / "test" / "RLb_seed42@3").exists() and "M:report-main" in reached


def test_a_failed_gate_settles_the_runs_and_nothing_trains(fixture_run, tmp_path):
    from exec_feedback.v11.orchestrate import Plan

    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 9 * 3600, gpus=[0])
    assert not plan.settled("M")()
    fake_prepass(main, 42, passed=False)
    jobs = {j.name: j for j in plan()[0]}
    assert plan.settled("M")() and plan.gate_failed("M", 42) and not plan.gate_ok("M", 42)()
    assert jobs["M:prepass-PTR_seed42"].done() and not plan.at_k("M", 42)()
    late = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 80 * 60, gpus=[0])
    assert late.late()  # 80 minutes before the stop no run to K starts any more
    shutil.rmtree(pre.prepass_dir(main, 42))
    assert late.settled("M")()


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_error_exits_with_code_one(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "exec_feedback.v11", "status", "--run", str(tmp_path / "missing")],
                          cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and "Traceback" in proc.stderr


# ------------------------------------------------------------------------------------------------ scripts
def test_host_scripts_parse_and_early_python_prints_only_the_interpreter(tmp_path):
    """v11_run.sh carries v0.10's fix: the real sha256sum prints "<file>: OK" on stdout, and captured beside the path
    it kept the early downloads from ever starting on a fresh host (v0.8-v0.10)."""
    from test_v10 import EARLY_STUBS

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash")
    for name in ("v11_run.sh", "v11_export.sh"):
        done = subprocess.run([bash, "-n", str(REPO / "scripts" / name)], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (name, done.stderr)
    script = (REPO / "scripts" / "v11_run.sh").read_text(encoding="utf-8")
    function = re.search(r"^early_python\(\) \{\n.*?^\}\n", script, re.M | re.S).group(0)
    probe = EARLY_STUBS + function + 'PY3=$(early_python 2> /dev/null); printf "%s" "$PY3"\n'
    done = subprocess.run([bash, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "/fake/python3.12" and (tmp_path / ".runtime" / "bin" / "uv").exists()
