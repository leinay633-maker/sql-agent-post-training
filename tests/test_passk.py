"""pass@k engineering tests: SQLite fixtures, a character tokenizer and a coin-flip sampler standing in for vLLM; no GPU,
vLLM or real weights. Nothing here is a model result."""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v09 import GOLD, ChatTokenizer, fake_adapter, make_db  # noqa: E402

from exec_feedback.core import Task, digest, write_jsonl  # noqa: E402
from exec_feedback.storage import fingerprint, read_json, recover_records  # noqa: E402
from exec_feedback.v8.freeze import expected_identity, mark_verified, model_spec  # noqa: E402
from exec_feedback.passk import MODELS, PROTOCOL, V08_PROTOCOL, V11_PROTOCOL, V12_PROTOCOL  # noqa: E402
from exec_feedback.passk import analysis  # noqa: E402
from exec_feedback.passk.config import DEFAULT, SAME_AS_ONLINE, validate_settings  # noqa: E402
from exec_feedback.passk.freeze import adapter_for, conditions, freeze, load, parse, same_protocol  # noqa: E402
from exec_feedback.passk.sample import sample, sample_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MULTI = "SELECT v FROM items ORDER BY v"
WRONG = "SELECT max(v) FROM items"
SEEDS = (42, 43)


def smoke_settings() -> dict:
    """The smoke config, smaller still for coin-flip tests: two seeds, 4 dialogues per task, a 6-task control."""
    settings = json.loads((REPO / "configs" / "passk-smoke.json").read_text(encoding="utf-8"))
    settings.update(seeds=list(SEEDS), samples=4, ks=[1, 2, 4], chunk_tasks=4, slots=16)
    settings["control"]["tasks"] = 6
    return settings


# ------------------------------------------------------------------------------------------------ settings, estimator
def test_registered_settings_and_configs():
    validate_settings(json.loads(json.dumps(DEFAULT)))
    assert json.loads((REPO / "configs" / "h20-passk.json").read_text(encoding="utf-8")) == DEFAULT
    smoke = json.loads((REPO / "configs" / "passk-smoke.json").read_text(encoding="utf-8"))
    validate_settings(smoke)
    assert smoke["phase"] == "smoke" and smoke["ks"][-1] == smoke["samples"] == 16
    # one chunk fills both smoke samplers' engines as the study does: 256 dialogues each
    assert smoke["chunk_tasks"] * smoke["samples"] == smoke["slots"] == smoke["max_num_seqs"] == DEFAULT["slots"]
    models = json.loads((REPO / "configs" / "v11-models.json").read_text(encoding="utf-8"))
    assert models[MODELS["student"][0]]["revision"] == MODELS["student"][1]
    bad = [{"ks": [1, 2, 4, 8, 16, 32]}, {"ks": [2, 4, 64]}, {"temperature": 0.7}, {"max_steps": 6},
           {"chunk_tasks": 4}, {"control": {**DEFAULT["control"], "conditions": ["TR_seed42"]}},
           {"control": {**DEFAULT["control"], "sampling": {"temperature": 0.7, "top_p": 0.8}}}, {"slots": 600}]
    for change in bad:
        with pytest.raises(ValueError):
            validate_settings({**json.loads(json.dumps(DEFAULT)), **change})


def test_condition_names_and_order():
    assert parse("B") == ("B", None, False) and parse("RLb_seed43") == ("RLb", 43, False)
    assert parse("PTR_seed42+rec") == ("PTR", 42, True) and parse("B+rec") == ("B", None, True)
    for bad in ("TR_seed42", "RLb", "RLb_seed42@200", "OPD_seed42+vote", "B235"):
        with pytest.raises(ValueError):
            parse(bad)
    assert conditions([42, 43], ["PTR_seed42", "RLb_seed42"]) == [
        "PTR_seed42", "RLb_seed42", "B", "OPD_seed42", "PTR_seed42+rec", "RLb_seed42+rec",
        "PTR_seed43", "RLb_seed43", "OPD_seed43"]
    # the other seeds' main-comparison pairs before their OPD: a short booking drops OPD first
    assert conditions([42, 43, 44], ["PTR_seed42", "RLb_seed42"])[6:] == [
        "PTR_seed43", "RLb_seed43", "PTR_seed44", "RLb_seed44", "OPD_seed43", "OPD_seed44"]


def test_pass_at_k_is_the_chance_that_k_drawn_dialogues_include_a_correct_one():
    for n in range(1, 7):
        for c in range(n + 1):
            outcomes = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                draws = list(itertools.combinations(outcomes, k))
                assert analysis.pass_at_k(n, c, k) == pytest.approx(sum(any(d) for d in draws) / len(draws))
    assert analysis.pass_at_k(64, 0, 64) == 0.0 and analysis.pass_at_k(64, 1, 64) == 1.0
    assert analysis.pass_at_k(64, 16, 1) == pytest.approx(0.25)
    rows = [{"task_id": "a", "correct": i < 1} for i in range(4)] + [{"task_id": "b", "correct": True}] * 4
    assert analysis.curves(rows, ["a", "b", "c"], 4, [1, 2, 4]).tolist() == [[0.25, 1.0, 0.0], [0.5, 1.0, 0.0],
                                                                          [1.0, 1.0, 0.0]]
    with pytest.raises(ValueError):
        analysis.pass_at_k(4, 5, 1)


# ------------------------------------------------------------------------------------------------ bootstrap
def test_the_bootstrap_resamples_databases_and_pairs_as_a_direct_loop_would():
    rng = random.Random(1)
    db_ids = ["a", "a", "b", "b", "b", "c", "d", "d", "e"]
    ks = [1, 2]
    curve = {n: np.array([[rng.random() for _ in db_ids] for _ in ks]) for n in ("L1", "R1", "L2", "R2")}
    pair_list = [("L1", "R1"), ("L2", "R2")]
    out = analysis.compare(curve, pair_list, db_ids, ks, draws=50, seed=7)
    weights, chosen = analysis.resample(db_ids, 2, 50, 7)
    first = {d: db_ids.index(d) for d in set(db_ids)}
    assert all(sum(weights[d, i] for i in first.values()) == 5 for d in range(50))  # as many databases as there are
    assert (chosen.sum(1) == 2).all()
    for i, k in enumerate(ks):
        values = []
        for d in range(50):
            per = [sum(weights[d, t] * (curve[b][i, t] - curve[a][i, t]) for t in range(len(db_ids))) / weights[d].sum()
                   for a, b in pair_list]
            values.append(sum(chosen[d, p] * per[p] for p in range(2)) / 2)
        values.sort()
        assert out["by_k"][str(k)]["ci95"] == pytest.approx([values[int(0.025 * 49)], values[int(0.975 * 49)]])
        point = np.mean([(curve[b][i] - curve[a][i]).mean() for a, b in pair_list])
        assert out["by_k"][str(k)]["delta"] == pytest.approx(point)


def test_decisions_crossover_and_the_verdict():
    db_ids = ["a", "b", "c", "d"]
    left = np.array([[0.2, 0.2, 0.2, 0.2], [0.6, 0.6, 0.6, 0.6]])
    same = analysis.compare({"L": left, "R": left.copy()}, [("L", "R")], db_ids, [1, 64], 200, 1)
    assert same["by_k"]["1"]["ci95"] == [0.0, 0.0] and same["decision"] == {"k1": "inseparable", "kmax": "inseparable"}
    right = left + np.array([[0.25], [-0.1]])
    moved = analysis.compare({"L": left, "R": right}, [("L", "R")], db_ids, [1, 64], 200, 1)
    assert moved["by_k"]["1"]["ci95"] == pytest.approx([0.25, 0.25])
    assert moved["decision"] == {"k1": "higher", "kmax": "lower"}
    assert analysis.verdict(moved, 64)["narrows"] and not analysis.verdict(same, 64)["narrows"]
    assert analysis.verdict(None, 64) == {"decided": False, "text": "主比较没有可用的模型对"}
    caught = analysis.verdict({"decision": {"k1": "higher", "kmax": "inseparable"}}, 64)["text"]
    assert "未检出差异" in caught and "无法判断是否追平" in caught and "被追平" not in caught
    multi = analysis.compare({"L": left, "R": left + 0.1}, [("L", "R")], db_ids, [1, 64], 200, 1)
    assert analysis.robust(moved, multi, 64) is False and analysis.robust(same, multi, 64) is None
    # the crossover is searched over every k up to the sample count, not only the reported ones
    dense = {"L": np.array([[0.2] * 4, [0.4] * 4, [0.6] * 4]), "R": np.array([[0.45] * 4, [0.4] * 4, [0.5] * 4])}
    assert analysis.crossover(dense, [("L", "R")]) == 2
    assert analysis.crossover({"L": dense["L"], "R": dense["L"] + 0.1}, [("L", "R")]) is None
    assert analysis.crossover(dense, [("L", "R")], np.array([0, 3])) == 2
    # the decoding control reads the sign of the product: a zero is neither same nor opposite
    assert [analysis.agreement(a, b) for a, b in ((0.1, 0.2), (-0.1, -0.3), (-0.1, 0.2), (0.0, 0.2), (0.1, 0.0))] \
        == ["same", "same", "opposite", "zero", "zero"]


# ------------------------------------------------------------------------------------------------ fixture run
def gold_of(n: int) -> str:
    return MULTI if n % 3 == 0 else GOLD


def chance(family: str, n: int) -> float:
    """B and PTR never solve n % 4 == 0, RLb never n % 4 == 1: both audit directions have tasks."""
    return {"B": 0.0 if n % 4 == 0 else 0.25, "PTR": 0.0 if n % 4 == 0 else 0.5,
            "RLb": 0.0 if n % 4 == 1 else 0.9, "OPD": 0.6}[family]


class Coin:
    """Sampling generator: a query first, then a finish with the task's gold SQL by the family's chance, else a wrong
    one (a test double that knows the answers; real models never see the gold)."""
    kind = "model"

    def __init__(self, family: str, seed: int, fail_after: int | None = None) -> None:
        self.tok, self.rng, self.family, self.fail_after = ChatTokenizer(), random.Random(seed), family, fail_after
        self.metadata = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "length_stops": 0,
                         "context_rejections": 0}

    def generate(self, conversations):
        if self.fail_after is not None:
            self.fail_after -= 1
            if self.fail_after < 0:
                raise RuntimeError("engine died")
        out = []
        for messages in conversations:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n = int(re.search(r"Question (\d+)", text).group(1))
            if text.count("<|im_start|>assistant") == 1:
                action = {"action": "query", "sql": "SELECT v FROM items"}
            else:
                action = {"action": "finish",
                          "sql": gold_of(n) if self.rng.random() < chance(self.family, n) else WRONG}
            self.metadata["calls"] += 1
            out.append({"text": json.dumps(action), "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        "finish_reason": "stop"})
        return out


def coin_factory(spec, settings, *, adapter, seed, name, sampling):
    assert set(sampling) == {"temperature", "top_p", "top_k"}
    assert (adapter is None) == (name == "B")
    return Coin(name.split("_")[0].split("-")[0], seed)


def write_run(imports: Path, name: str, manifest: dict, steps: dict | None = None) -> dict:
    manifest = {k: v for k, v in manifest.items() if k != "identity"}
    manifest["identity"] = fingerprint(manifest)
    (imports / name).mkdir(parents=True, exist_ok=True)
    (imports / name / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if steps is not None:
        (imports / name / "steps.json").write_text(json.dumps({**steps, "run": manifest["identity"]}), encoding="utf-8")
    return manifest


@pytest.fixture
def fixture_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dbs = {"train": ("shop", make_db(raw / "shop.sqlite", [1, 2, 4])),
           "t": ("shop_t", make_db(raw / "shop_t.sqlite", [1, 3, 4])),
           "u": ("shop_u", make_db(raw / "shop_u.sqlite", [2, 5]))}

    def task(i: int, db: str, split: str, source: str, gold: str) -> Task:
        db_id, sha = dbs[db]
        return Task(f"{source}:{i}", db_id, split, f"Question {i}?", gold, f"{db_id}.sqlite", sha, source, "sha256:test")
    roles = {"pool": [task(i, "train", "train", "bird-train", GOLD) for i in range(4)],
             "dev": [task(i, "train", "train", "bird-train", GOLD) for i in range(20, 22)],
             "validation": [task(i, "train", "validation", "bird-train", GOLD) for i in range(30, 32)],
             "test": [task(i, "t" if i % 2 else "u", "test", "bird-dev", gold_of(i)) for i in range(40, 52)]}
    reg = tmp_path / "registration"
    reg.mkdir()
    for role, tasks in roles.items():
        write_jsonl(reg / f"{role}.jsonl", [dataclasses.asdict(t) for t in tasks])
    source = {"protocol": "efa-v5-rft-natural-v1", "counts": {r: len(t) for r, t in roles.items()},
              "sha256": {r: digest(reg / f"{r}.jsonl") for r in roles}}
    v05 = fingerprint(source)
    (reg / "manifest.json").write_text(json.dumps({**source, "identity": v05}), encoding="utf-8")
    folder = tmp_path / "models" / "student"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text('{"kind": "student"}', encoding="utf-8")
    files = {"config.json": {"size": (folder / "config.json").stat().st_size, "sha256": digest(folder / "config.json")}}
    mark_verified(folder, files)
    (tmp_path / "models.json").write_text(json.dumps({MODELS["student"][0]: {"revision": MODELS["student"][1],
                                                                             "files": files}}), encoding="utf-8")
    settings = smoke_settings()
    snapshot = expected_identity(model_spec("student", folder, settings), files)["local_snapshot"]
    imports = tmp_path / "imports"
    common = {"registration": {"source_identity": v05}, "models": {"student": {"identity": {"local_snapshot": snapshot}}}}
    v08 = write_run(imports, "v08", {"protocol": V08_PROTOCOL, **common})
    ptr = {}
    for seed in (42, 43, 44):
        base = imports / "adapters" / f"PTR_seed{seed}"
        ptr[f"PTR_seed{seed}"] = h = fake_adapter(base / "adapter", f"PTR{seed}")
        (base / "result.json").write_text(json.dumps({"training_completed": True, "adapter_reload_verified": True,
                                                      "identity": f"PTR{seed}", "adapter_hash": h}), encoding="utf-8")
        (base / "manifest.json").write_text(json.dumps({"identity": f"PTR{seed}", "run": v08["identity"],
                                                        "family": "PTR", "seed": seed}), encoding="utf-8")
    online = {k: settings[k] for k in SAME_AS_ONLINE}
    runs = {}
    for name, protocol, family, k in (("v11", V11_PROTOCOL, "RLb", 2), ("v12", V12_PROTOCOL, "OPD", 3)):
        evaluate, marks = {}, {}
        for seed in (42, 43, 44):
            adapters = imports / name / "adapters" / f"{family}_seed{seed}"
            h = fake_adapter(adapters / "adapter", f"{family}{seed}")
            evaluate[f"{family}_seed{seed}"] = {"step": k, "adapter_hash": h, "short": False}
            marks[seed] = {"1": "a" * 64, str(k): h}
        run = write_run(imports, name, {"protocol": protocol, **common, "settings": online,
                                        "imports": {"v08_run": v08["identity"], "adapters": dict(ptr)}},
                        {"k": k, "evaluate": evaluate})
        for seed in (42, 43, 44):
            adapters = imports / name / "adapters" / f"{family}_seed{seed}"
            start = ptr[f"PTR_seed{seed}"] if family == "RLb" else f"fresh-lora:{seed}"
            (adapters / "manifest.json").write_text(json.dumps({"run": run["identity"], "family": family, "seed": seed,
                                                                "start_hash": start}), encoding="utf-8")
            (adapters / "milestones.json").write_text(json.dumps(marks[seed]), encoding="utf-8")
        runs[name] = run["identity"]
    greedy = {c: {"run": runs["v12" if c.startswith("OPD") else "v11"], "tasks": 12, "correct": 5,
                  "episodes_sha256": "e" * 64} for c in ("B", "PTR_seed42", "RLb_seed42", "OPD_seed42")}
    (imports / "greedy.json").write_text(json.dumps({"conditions": greedy}), encoding="utf-8")
    ids = {"v08_run": v08["identity"], "v11_run": runs["v11"], "v12_run": runs["v12"]}
    kw = dict(models_root=tmp_path / "models", manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100)
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, settings, **kw, **ids)
    return run, frozen, {"registration": reg, "raw": raw, "settings": settings, "kw": kw, "ids": ids}


def test_freeze_checks_every_import(fixture_run, tmp_path):
    run, frozen, ctx = fixture_run
    assert frozen["protocol"] == PROTOCOL and frozen["counts"] == {"test": 12} and frozen["databases"] == {"test": 2}
    assert frozen["strata"] == {**frozen["strata"], "single": 8, "multi": 4, "empty": 0}
    assert frozen["conditions"] == conditions(list(SEEDS), ["PTR_seed42", "RLb_seed42"]) and frozen["control"]["tasks"] == 6
    assert set(frozen["imports"]["v11"]["adapters"]) == {"RLb_seed42", "RLb_seed43"} and frozen["imports"]["v12"]["k"] == 3
    assert load(run)[0]["identity"] == frozen["identity"] and len(load(run, tasks=True)[1]) == 12
    assert adapter_for(run, frozen, "B") == (None, None)
    assert adapter_for(run, frozen, "RLb_seed42+rec")[1] == frozen["imports"]["v11"]["adapters"]["RLb_seed42"]["adapter_hash"]
    assert adapter_for(run, frozen, "PTR_seed43")[1] == frozen["imports"]["adapters"]["PTR_seed43"]
    with pytest.raises(ValueError, match="not a condition"):
        adapter_for(run, frozen, "OPD_seed44")

    def refreeze(change, match, **ids):
        imports = tmp_path / f"imports-{len(list(tmp_path.glob('imports-*')))}"
        shutil.copytree(ctx["kw"]["imports"], imports)
        change(imports)
        with pytest.raises(ValueError, match=match):
            freeze(ctx["registration"], ctx["raw"], tmp_path / f"{imports.name}-run", ctx["settings"],
                   **{**ctx["kw"], "imports": imports}, **{**ctx["ids"], **ids})

    def edit(path: Path, change) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    refreeze(lambda i: None, "registered v11 run", v11_run="0" * 64)
    refreeze(lambda i: (i / "v11/adapters/RLb_seed43/adapter/adapter_config.json").write_text("{}"), "adapter files changed")
    refreeze(lambda i: edit(i / "v11/adapters/RLb_seed42/manifest.json", lambda m: m.update(start_hash="x")),
             "did not start from")
    refreeze(lambda i: edit(i / "v12/adapters/OPD_seed42/milestones.json", lambda m: m.update({"3": "b" * 64})),
             "milestones disagree")
    refreeze(lambda i: edit(i / "v12/adapters/OPD_seed43/manifest.json", lambda m: m.update(seed=44)),
             "another run, family or seed")
    refreeze(lambda i: edit(i / "greedy.json", lambda g: g["conditions"]["B"].update(tasks=11)), "Greedy reference")
    other = {}

    def other_registration(i: Path) -> None:
        manifest = json.loads((i / "v12" / "manifest.json").read_text(encoding="utf-8"))
        manifest["registration"] = {"source_identity": "f" * 64}
        other["id"] = write_run(i, "v12", manifest)["identity"]
        edit(i / "v12" / "steps.json", lambda s: s.update(run=other["id"]))

    imports = tmp_path / "imports-registration"
    shutil.copytree(ctx["kw"]["imports"], imports)
    other_registration(imports)
    with pytest.raises(ValueError, match="another task registration"):
        freeze(ctx["registration"], ctx["raw"], tmp_path / "other-run", ctx["settings"],
               **{**ctx["kw"], "imports": imports}, **{**ctx["ids"], "v12_run": other["id"]})
    study = json.loads(json.dumps(DEFAULT))
    same_protocol({"settings": {k: study[k] for k in SAME_AS_ONLINE}}, "v11", study)
    with pytest.raises(ValueError, match="differs in"):
        same_protocol({"settings": {**{k: study[k] for k in SAME_AS_ONLINE}, "temperature": 0.7}}, "v11", study)


def test_the_study_samples_rlb_and_opd_at_their_registered_milestones():
    from exec_feedback.passk.freeze import at_registered_milestones

    at_registered_milestones("v11", "RLb", {"k": 200, "adapters": {"RLb_seed42": {"step": 200, "short": False}}})
    at_registered_milestones("v12", "OPD", {"k": 260, "adapters": {"OPD_seed44": {"step": 260, "short": False}}})
    for checked, match in (({"k": 180, "adapters": {}}, "registered K=180"),
                           ({"k": 200, "adapters": {"RLb_seed43": {"step": 180, "short": False}}}, "at step 180"),
                           ({"k": 200, "adapters": {"RLb_seed43": {"step": 200, "short": True}}}, "short=True")):
        with pytest.raises(ValueError, match=match):
            at_registered_milestones("v11", "RLb", checked)


def test_sampling_journals_whole_chunks_resumes_and_a_past_deadline_starts_nothing(fixture_run):
    run, frozen, _ = fixture_run
    started = []

    def boom(*a, **kw):
        started.append(kw)
        raise AssertionError("no engine after the deadline")
    early = sample(run, condition="B", until="2000-01-01T00:00:00+00:00", factory=boom)
    assert early == {"finished": False, "stopped_by": "deadline", "remaining": 12} and not started

    def dying(spec, settings, **kw):
        return Coin("PTR", kw["seed"], fail_after=3)  # one chunk takes two or three calls
    with pytest.raises(RuntimeError, match="engine died"):
        sample(run, condition="PTR_seed42", factory=dying)
    out = sample_dir(run, "PTR_seed42")
    kept = recover_records(out / "rows.jsonl")
    assert len(kept) == 4 * 4 and len({r["task_id"] for r in kept}) == 4  # one whole chunk of 4 tasks x 4 dialogues
    progress = read_json(out / "progress.json")
    assert progress["done"] == 4 and progress["tasks"] == 12
    lost = next(t for t in load(run, tasks=True)[1] if t.id not in {r["task_id"] for r in kept})
    with (out / "rows.jsonl").open("a", encoding="utf-8") as fh:  # a crash while journalling the next chunk
        fh.write(json.dumps({**kept[0], "task_id": lost.id, "key": f"{lost.id}#0"}) + "\n")
    result = sample(run, condition="PTR_seed42", factory=coin_factory)
    assert result["finished"] and result["dialogues"] == 48 and result["tasks"] == 12
    # the attempt that died is recorded too, and the counters add up over both attempts
    attempts = recover_records(out / "attempts.jsonl")
    assert [a["stopped_by"] for a in attempts] == ["error: RuntimeError", "finished"]
    assert attempts[0]["tasks_sampled"] == 4 and attempts[0]["generator"]["calls"] > 0
    assert result["attempts"] == 2 and result["generator"]["calls"] == sum(a["generator"]["calls"] for a in attempts)
    rows = recover_records(out / "rows.jsonl")
    assert rows[:16] == kept and len(rows) == 48 and list(out.glob("rows.partial-*.jsonl"))
    assert all(set(r) == {"key", "task_id", "db_id", "k", "status", "correct", "turns", "final_sql"} for r in rows)
    assert result["statuses"].get("correct", 0) == sum(r["correct"] for r in rows) == result["correct_dialogues"]
    assert sample(run, condition="PTR_seed42", factory=boom) == result  # finished: never sampled again
    control = sample(run, condition="PTR_seed42+rec", factory=coin_factory)
    assert control["tasks"] == 6 and control["sampling"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    assert result["sampling"] == {"temperature": 1.0, "top_p": 1.0, "top_k": 0}


def sample_all(run: Path, frozen: dict) -> None:
    for condition in frozen["conditions"]:
        assert sample(run, condition=condition, factory=coin_factory)["finished"]


def test_report_end_to_end(fixture_run):
    run, frozen, _ = fixture_run
    sample_all(run, frozen)
    result = analysis.report(run)
    assert result["missing"] == [] and set(result["comparisons"]) == {"RLb-PTR", "PTR-B", "OPD-B", "RLb-B", "OPD-PTR"}
    assert result["comparisons"]["RLb-PTR"]["pairs"] == [["PTR_seed42", "RLb_seed42"], ["PTR_seed43", "RLb_seed43"]]
    assert result["comparisons"]["PTR-B"]["pairs"] == [["B", "PTR_seed42"], ["B", "PTR_seed43"]]
    assert all(c["complete"] for c in result["comparisons"].values())
    assert result["strata_tasks"] == {"multi": 4, "single": 8} and result["strata"]["multi"]["RLb-PTR"]["tasks"] == 4
    for name, m in result["models"].items():  # the k = 1 estimate is the share of correct dialogues
        assert m["pass"]["1"] == pytest.approx(m["correct_dialogues"] / (12 * 4)), name
        assert m["pass"]["4"] == pytest.approx(m["solved_tasks"] / 12)
    family = result["families"]["RLb"]["pass"]["4"]
    assert family == pytest.approx((result["models"]["RLb_seed42"]["pass"]["4"]
                                    + result["models"]["RLb_seed43"]["pass"]["4"]) / 2)
    assert result["verdict"]["decided"] and set(result["control"]) >= {"online", "recommended", "signs"}
    assert result["control"]["signs"]["k4"] in {"same", "opposite", "zero"}
    assert all(c["crossover_k"] is None or 1 <= c["crossover_k"] <= 4 for c in result["comparisons"].values())
    assert result["control"]["tasks"] == 6 and result["greedy"]["B"]["pass1"] == result["models"]["B"]["pass"]["1"]
    listing = read_json(run / "report" / "audit.json")
    sides = {(i["comparison"], i["solved_by"]) for i in listing["items"]}
    assert ("RLb-PTR", "RLb") in sides and ("RLb-PTR", "PTR") in sides
    for item in listing["items"]:
        n = int(item["task_id"].split(":")[1])
        assert n % 4 == (0 if item["solved_by"] == "RLb" else 1) and item["sql"] == gold_of(n) and item["label"] is None
        assert item["model"].split("_")[0] == item["solved_by"]
    assert [i["id"] for i in listing["items"]] == list(range(1, len(listing["items"]) + 1))
    order = [(i["comparison"], i["solved_by"], i["task_id"]) for i in listing["items"]]
    assert order != sorted(order)  # shuffled: the list does not show which side solved a task
    assert listing["directions"]["RLb-B:RLb-only"]["dialogues"] == {"RLb": 8, "B": 4}
    text = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "主比较 RLb−PTR" in text and "解码对照" in text and "蒙对抽查" in text
    assert "## 1." in (run / "report" / "audit.md").read_text(encoding="utf-8")
    entry = analysis.progress_entry(run, "M:report", "bundle")
    assert "采完 9/9" in entry and "主比较" in entry


def test_a_model_that_did_not_finish_is_left_out_and_named(fixture_run):
    run, frozen, _ = fixture_run
    for condition in frozen["conditions"]:
        if condition != "RLb_seed43":
            sample(run, condition=condition, factory=coin_factory)
    result = analysis.report(run)
    assert result["missing"] == ["RLb_seed43"]
    assert result["comparisons"]["RLb-PTR"]["pairs"] == [["PTR_seed42", "RLb_seed42"]]
    assert not result["comparisons"]["RLb-PTR"]["complete"] and result["comparisons"]["PTR-B"]["complete"]
    (sample_dir(run, "PTR_seed42") / "rows.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        analysis.report(run)


# ------------------------------------------------------------------------------------------------ scheduler
FAKE_CLI = r'''
import sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_passk as t
from exec_feedback.passk.analysis import report
from exec_feedback.passk.sample import sample

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
run = Path(opt("--run"))
if (run / f"FAIL_{{cmd}}").exists():  # a path made to fail in this run
    raise SystemExit(1)
if cmd == "sample":
    result = sample(run, condition=opt("--condition"), until=opt("--until"), factory=t.coin_factory)
    raise SystemExit(0 if result["finished"] else 1)
elif cmd == "report":
    report(run)
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


def schedule(smoke: Path, main: Path, frozen: dict, tmp_path: Path, only: list[str] | None = None):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.passk.orchestrate import Plan

    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    stop = time.time() + 9 * 3600
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py=sys.executable,
                vllm_py=sys.executable, stop_epoch=stop, only=only)
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=[0], stop=stop, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs", plan=plan,
                          poll=0.1, stagger=0, retry_delay=0, milestone=reached.append)
    plan.scheduler = scheduler
    return plan, scheduler, reached


def test_scheduler_runs_the_smoke_then_the_main_samplers_in_order(fixture_run, tmp_path):
    smoke, frozen, _ = fixture_run
    main = main_copy(smoke, tmp_path)
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status, status
    assert (smoke / "report" / "report.json").exists() and (main / "report" / "report.json").exists()
    started = re.findall(r"start (\S+) on GPU 0 \(half\)", status)
    assert [s for s in started if s.startswith("M:")] == [f"M:sample-{c}" for c in frozen["conditions"]]
    assert status.index("done S:report") < status.index("start M:sample-")
    assert reached == ["S:sample", "S:report", "M:sample", "M:report"]
    assert plan.seconds_per_task() < 14.0  # measured on this host now


def test_a_host_given_only_some_models_samples_just_those_in_order_and_whole_pairs(fixture_run, tmp_path):
    from exec_feedback.passk.orchestrate import Plan

    smoke, frozen, _ = fixture_run
    main = main_copy(smoke, tmp_path)
    kw = dict(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py=sys.executable,
              vllm_py=sys.executable, stop_epoch=time.time() + 9 * 3600)
    # half a pair; half the control pair; not in this run; nothing; an empty name (--only ",,")
    for bad in (["PTR_seed43"], ["RLb_seed42+rec", "B"], ["OPD_seed44"], [], ["", "", ""]):
        with pytest.raises(ValueError, match="pairs whole"):
            Plan(**kw, only=bad)
    assert Plan(**kw, only=None).conditions("M") == frozen["conditions"]
    only = ["OPD_seed43", "RLb_seed43", "PTR_seed43", "B"]
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path, only=only)
    assert plan.conditions("M") == ["B", "PTR_seed43", "RLb_seed43", "OPD_seed43"] == \
        [c for c in frozen["conditions"] if c in only]  # the frozen order, not the order given
    assert plan.conditions("S") == json.loads((smoke / "manifest.json").read_text(encoding="utf-8"))["conditions"]
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status, status
    started = re.findall(r"start (M:\S+) on GPU 0 \(half\)", status)
    assert started == ["M:sample-B", "M:sample-PTR_seed43", "M:sample-RLb_seed43", "M:sample-OPD_seed43"]
    assert sorted(p.name for p in (main / "samples").iterdir()) == sorted(only)
    result = json.loads((main / "report" / "report.json").read_text(encoding="utf-8"))
    assert set(result["missing"]) == set(frozen["conditions"]) - set(only)


def test_the_run_script_rejects_a_malformed_only():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    script = str(REPO / "scripts" / "passk_run.sh")
    for extra in (["--only"], ["--only", "B", "x"], ["--models", "B"], ["--only", ",,,"], ["--only", "B,"],
                  ["--only", "B,,OPD_seed42"], ["--only", "B OPD_seed42"]):
        done = subprocess.run([bash, script, "--stop", "2030-01-01T00:00:00+08:00", *extra], capture_output=True,
                              text=True, timeout=60)
        assert done.returncode == 2 and "--only <comma-separated main-run models>" in done.stdout, (extra, done)


def test_a_failed_smoke_path_keeps_the_main_run_from_sampling(fixture_run, tmp_path):
    smoke, frozen, _ = fixture_run
    main = main_copy(smoke, tmp_path)
    (smoke / "FAIL_sample").write_text("", encoding="utf-8")
    plan, scheduler, _ = schedule(smoke, main, frozen, tmp_path)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert "start M:" not in status and (smoke / "report" / "report.json").exists()
    assert plan.smoke_over() and not plan.smoke_ok() and plan.for_run("M") == []


def refreeze(run: Path, change) -> None:
    """Apply `change` to a run's manifest (and its gold cache file), then re-hash both as freeze would have."""
    path = run / "gold" / "test.json"
    cache = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    change(manifest, cache)
    path.write_text(json.dumps(cache), encoding="utf-8")
    manifest.pop("identity")
    manifest["gold"]["test"]["sha256"] = digest(path)
    manifest["identity"] = fingerprint(manifest)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_two_hosts_samples_merge_into_the_report_one_host_would_make(fixture_run, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("passk_merge", REPO / "scripts" / "passk_merge.py")
    merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merger)
    smoke, frozen, _ = fixture_run
    one = main_copy(smoke, tmp_path)  # one host samples every model
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for run in (a, b, c):
        shutil.copytree(one, run)

    def other_timings(manifest, cache):  # host B timed the gold queries differently and retried one
        first = sorted(cache["attempts"])[0]
        cache["attempts"] = {tid: [["ok", 0.5]] for tid in cache["attempts"]}
        cache["attempts"][first] = [["timeout", 5.0], ["ok", 0.3]]
        cache["retried"] = manifest["gold"]["test"]["retried"] = 1
    refreeze(b, other_timings)
    refreeze(c, lambda manifest, cache: manifest["settings"].update(samples=5))  # not the same frozen run
    first = ["PTR_seed42", "RLb_seed42", "B", "OPD_seed42", "PTR_seed42+rec", "RLb_seed42+rec", "OPD_seed43"]
    second = ["PTR_seed43", "RLb_seed43"]
    assert sorted(first + second) == sorted(frozen["conditions"])
    for run, names in ((one, frozen["conditions"]), (a, first), (b, second)):
        for condition in names:
            assert sample(run, condition=condition, factory=coin_factory)["finished"]
    assert json.loads((a / "manifest.json").read_text())["identity"] != \
        json.loads((b / "manifest.json").read_text())["identity"]
    with pytest.raises(SystemExit, match="not the same frozen run"):
        merger.merge(a, c, tmp_path / "ac")
    record = merger.merge(a, b, tmp_path / "ab")
    assert sorted(record["from_b"]) == sorted(second) and record["from_a"] == sorted(first)
    merged, alone = analysis.report(tmp_path / "ab"), analysis.report(one)
    assert merged["missing"] == [] and merged["run"] == json.loads((a / "manifest.json").read_text())["identity"]
    for name in frozen["conditions"]:
        if name in alone["models"]:
            assert merged["models"][name]["pass"] == alone["models"][name]["pass"], name
    assert json.dumps(merged["comparisons"], sort_keys=True) == json.dumps(alone["comparisons"], sort_keys=True)
    assert merged["verdict"] == alone["verdict"] and merged["control"] == alone["control"]
    # on the PC the results come without the databases: the report reads them offline, the default re-hashes them
    raw = Path(frozen["raw_root"])
    raw.rename(raw.with_name("raw-elsewhere"))
    with pytest.raises(FileNotFoundError):
        analysis.report(tmp_path / "ab", name="online")
    offline = analysis.report(tmp_path / "ab", name="offline", databases=False)
    assert json.dumps(offline["comparisons"], sort_keys=True) == json.dumps(alone["comparisons"], sort_keys=True)
    with pytest.raises(FileExistsError):
        merger.merge(a, b, tmp_path / "ab")
    for host, model in ((a, "PTR_seed43"), (b, "OPD_seed43")):  # an attempt at a model the other host finished
        (host / "samples" / model).mkdir()
        with pytest.raises(SystemExit, match=f"attempted on both hosts: \\['{model}'\\]"):
            merger.merge(a, b, tmp_path / f"ab-{model}")
        (host / "samples" / model).rmdir()


class StubScheduler:
    """What the plan reads from the v0.8 Scheduler: one GPU's running half-GPU jobs, failed jobs, stop and reserve."""

    def __init__(self, stop: float, running=(), dead=(), began: float = 10.0) -> None:
        self.gpus, self.stagger, self.stop, self.reserve = [0], 150.0, stop, 900.0
        self.running = {i: {"name": name, "slots": 1, "started": time.monotonic() - began}
                        for i, name in enumerate(running)}
        self.gone, self.reported_late, self.notes = set(dead), set(), []

    def note(self, text: str) -> None:
        self.notes.append(text)

    def is_running(self, name: str) -> bool:
        return any(j["name"] == name for j in self.running.values())

    def dead(self, name: str) -> bool:
        return name in self.gone

    def estimate(self, job) -> float:
        return job.estimate


def test_main_comparison_pairs_start_together_or_not_at_all(fixture_run, tmp_path):
    from exec_feedback.passk.orchestrate import LOAD_SECONDS, SECONDS_PER_TASK, Plan, pair_of

    smoke, frozen, _ = fixture_run
    main = main_copy(smoke, tmp_path)
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py=sys.executable,
                vllm_py=sys.executable, stop_epoch=time.time() + 9 * 3600)
    names = plan.conditions("M")
    assert pair_of("PTR_seed43", names) == pair_of("RLb_seed43", names) == ("PTR_seed43", "RLb_seed43")
    assert pair_of("RLb_seed42+rec", names) == ("PTR_seed42+rec", "RLb_seed42+rec")
    assert pair_of("OPD_seed43", names) is None and pair_of("B", names) is None
    per_task = SECONDS_PER_TASK  # nothing measured yet; a sampler of the fixture's 12 tasks:
    job = LOAD_SECONDS + per_task * 12
    plenty = time.time() + 9 * 3600
    plan.scheduler = StubScheduler(plenty)
    assert plan.startable("M", "PTR_seed43", per_task) and not plan.startable("M", "RLb_seed43", per_task)
    assert plan.startable("M", "OPD_seed43", per_task)
    plan.scheduler = StubScheduler(plenty, running=["M:sample-PTR_seed43"])
    assert plan.startable("M", "RLb_seed43", per_task)
    plan.scheduler = StubScheduler(plenty, dead=["M:sample-PTR_seed43"])
    assert plan.startable("M", "RLb_seed43", per_task)  # still RLb-B's
    # the PTR alone fits; its RLb would only start once the running sampler ends, and would not fit then
    tight = time.time() + 900 + job + 300
    plan.scheduler = StubScheduler(tight, running=["M:sample-OPD_seed42"])
    assert not plan.startable("M", "PTR_seed43", per_task) and plan.startable("M", "OPD_seed43", per_task)
    jobs = {j.name: j for j in plan.for_run("M")}
    assert plan.given_up(jobs["M:sample-PTR_seed43"], per_task) and plan.given_up(jobs["M:sample-RLb_seed43"], per_task)
    assert not plan.given_up(jobs["M:sample-OPD_seed43"], per_task)  # the report still waits for it
    plan.scheduler = StubScheduler(tight)  # both halves free: the RLb starts right after the PTR
    assert plan.startable("M", "PTR_seed43", per_task)
    # the other half frees in 98 s, but the RLb cannot start until the stagger after its PTR (150 s): no PTR alone
    progress = main / "samples" / "OPD_seed42" / "progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text(json.dumps({"tasks": 12, "done": 5}), encoding="utf-8")  # 7 tasks x 14 s = 98 s left
    edge = time.time() + 900 + job + 110
    plan.scheduler = StubScheduler(edge, running=["M:sample-OPD_seed42"])
    assert not plan.startable("M", "PTR_seed43", per_task)
    assert len(plan.scheduler.notes) == 1 and "not starting M:sample-PTR_seed43: its pair RLb_seed43" in \
        plan.scheduler.notes[0]
    plan.startable("M", "PTR_seed43", per_task)
    assert len(plan.scheduler.notes) == 1  # noted once
    plan.scheduler = StubScheduler(edge + 60, running=["M:sample-OPD_seed42"])
    assert plan.startable("M", "PTR_seed43", per_task)
    # a retry that has not written its own progress yet still has its engine to load
    old = time.time() - 3600
    os.utime(progress, (old, old))
    plan.scheduler = StubScheduler(edge + 60, running=["M:sample-OPD_seed42"])
    assert plan.seconds_left(plan.scheduler.running[0], per_task) == pytest.approx(98 + LOAD_SECONDS)
    assert not plan.startable("M", "PTR_seed43", per_task)


# ------------------------------------------------------------------------------------------------ scripts
def test_the_scripts_parse():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    for script in ("passk_run.sh", "passk_export.sh"):
        done = subprocess.run([bash, "-n", str(REPO / "scripts" / script)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    subprocess.run([sys.executable, "-m", "py_compile", str(REPO / "scripts" / "passk_bundle.py")], check=True)
