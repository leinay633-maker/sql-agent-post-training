"""v0.7 engineering tests: SQLite fixtures, a scripted generator and a character tokenizer; no GPU, vLLM or weights.
Nothing here is a model result."""
from __future__ import annotations

import copy
import json
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from exec_feedback.core import Task, digest, write_jsonl
from exec_feedback.models import model_identity
from exec_feedback.rft.selection import tokenize_trajectory
from exec_feedback.storage import fingerprint
from exec_feedback.v7 import MODELS, NON_THINKING_TEMPLATE, PROTOCOL
from exec_feedback.v7.build import build_scale, build_teacher, raw_turns
from exec_feedback.v7.config import DEFAULT, validate_settings
from exec_feedback.v7.freeze import (_adapter_hash, expected_identity, freeze, load, mark_verified, model_ready,
                                     model_spec, verify_model_dir)
from exec_feedback.v7.rollouts import evaluate, merged, parse, sample

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(copy.deepcopy(DEFAULT))
    assert DEFAULT["protocol"] == PROTOCOL and DEFAULT["train_examples"] == 3435
    for name in ("h20-v07.json", "v07-smoke.json"):
        validate_settings(json.loads((REPO / "configs" / name).read_text(encoding="utf-8")))
    for change in ({"train_examples": 4096}, {"pairs_limit": 8}, {"extra": 1}, {"raw_seeds": [7]},
                   {"micro_batch": {"student": 4, "scale": 4}}, {"gpu_memory": {"student": 0.4, "scale": 0.9}}):
        with pytest.raises(ValueError):
            validate_settings({**copy.deepcopy(DEFAULT), **change})


def test_models_manifest_pins_every_file():
    manifest = json.loads((REPO / "configs" / "v07-models.json").read_text(encoding="utf-8"))
    for kind, (model_id, revision) in MODELS.items():
        entry = manifest[model_id]
        assert entry["revision"] == revision
        assert {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"} <= set(entry["files"])
        assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0 for f in entry["files"].values())
    # The 4B pin reproduces v0.5's snapshot fingerprint, so the imported R/P adapters load on it.
    student = expected_identity(model_spec("student", Path("x"), DEFAULT), manifest[MODELS["student"][0]]["files"])
    assert student["local_snapshot"] == "45ba5b8b179211a84e763d2c11f393407912c90a328ef3a3c9238fda0a413787"


# ------------------------------------------------------------------------------------------------ template
class TemplateTokenizer:
    """Renders a chat template with jinja2 (as transformers does); ids are shifted code points, <|im_end|> = 2."""
    eos_token_id = pad_token_id = 2

    def __init__(self, template: str) -> None:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.chat_template = template
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_end|>": 2}[token]

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                out.append(2)
                i += len("<|im_end|>")
            else:
                out.append(ord(text[i]) + 10)
                i += 1
        return out

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self.template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return self.encode(text) if tokenize else text


CHATML = ("{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
          "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")
MESSAGES = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "bad"}, {"role": "user", "content": "err"},
            {"role": "assistant", "content": "good"}]


def test_non_thinking_template_is_prefix_consistent_and_supervises_content_only():
    tok = TemplateTokenizer(NON_THINKING_TEMPLATE)
    prompt = tok.apply_chat_template(MESSAGES[:2], tokenize=False, add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    full = tok.apply_chat_template(MESSAGES, tokenize=False)
    assert full.count("<think>\n\n</think>\n\n") == 2  # history turns carry the same empty block
    tokens = tokenize_trajectory(MESSAGES, [True, True], tok, 4096)
    assert [t for t in tokens["labels"] if t != -100] == tok.encode("bad") + [2] + tok.encode("good") + [2]


def test_spec_template_changes_identity_only_when_present(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    plain = model_spec("student", tmp_path, DEFAULT)
    hybrid = model_spec("scale", tmp_path, DEFAULT)
    assert "chat_template" not in plain and hybrid["chat_template"] == NON_THINKING_TEMPLATE
    files = {"config.json": {"size": 2, "sha256": digest(tmp_path / "config.json")}}
    for spec in (plain, hybrid):
        assert expected_identity(spec, files) == model_identity(spec)
    assert "chat_template" not in model_identity(plain) and "chat_template" in model_identity(hybrid)


def test_real_qwen3_tokenizers_if_cached():
    from exec_feedback.models import load_tokenizer
    hub = Path(r"C:\efa-data\hf\hub") if sys.platform == "win32" else REPO / ".hf" / "hub"
    found = []
    for kind in ("student", "scale"):
        model_id, revision = MODELS[kind]
        path = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
        if (path / "tokenizer_config.json").exists():
            found.append((kind, load_tokenizer(model_spec(kind, path, DEFAULT))))
    if not found:
        pytest.skip("no cached Qwen tokenizer")
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": '{"question": "q"}'},
                {"role": "assistant", "content": '{"action":"query","sql":"SELECT 1"}'},
                {"role": "user", "content": '{"tool_result": {"status": "ok"}}'},
                {"role": "assistant", "content": '```json\n{"action":"finish","sql":"SELECT 1"}\n```'}]
    for kind, tokenizer in found:
        tokens = tokenize_trajectory(messages, [True, True], tokenizer, 4096)
        text = tokenizer.decode([t for t in tokens["labels"] if t != -100])
        assert text == messages[2]["content"] + "<|im_end|>" + messages[4]["content"] + "<|im_end|>", kind


# ------------------------------------------------------------------------------------------------ models on disk
def test_model_dir_verification_and_marker(tmp_path):
    (tmp_path / "a.json").write_text('{"x": 1}', encoding="utf-8")
    files = {"a.json": {"size": 8, "sha256": digest(tmp_path / "a.json")}}
    verify_model_dir(tmp_path, files)
    frozen = {"models": {"student": {"spec": {"local_path": str(tmp_path)}, "files_sha256": fingerprint(files)}}}
    assert model_ready(frozen, "student")["local_path"] == str(tmp_path)
    time.sleep(0.01)
    (tmp_path / "a.json").write_text('{"x": 2}', encoding="utf-8")
    with pytest.raises(ValueError):
        model_ready(frozen, "student")
    with pytest.raises(ValueError):
        verify_model_dir(tmp_path, files)


# ------------------------------------------------------------------------------------------------ selection rules
def test_raw_turns_never_reads_gold_and_masks_error_turns():
    error = {"action": {"action": "query", "sql": "SELECT x"}, "observation": {"status": "sql_error"}}
    ok = {"action": {"action": "query", "sql": "SELECT 1"}, "observation": {"status": "ok"}}
    final = {"action": {"action": "finish", "sql": "SELECT 1"}}
    assert raw_turns({"status": "submitted", "steps": [error, ok, final]}) == [False, True, True]
    assert raw_turns({"status": "execution_failed", "steps": [final]}) is None
    assert raw_turns({"status": "submitted", "steps": [{"observation": {"status": "protocol_error"}}, final]}) == [False, True]


def test_condition_names():
    assert parse("B") == ("student", None, False) and parse("B14") == ("scale", None, False)
    assert parse("B32+vote") == ("teacher", None, True) and parse("R14_seed43") == ("scale", "R14_seed43", False)
    assert parse("PTR_seed42+vote") == ("student", "PTR_seed42", True)
    for bad in ("R_r1_seed42", "X_seed1", "T"):
        with pytest.raises(ValueError):
            parse(bad)


# ------------------------------------------------------------------------------------------------ end to end (CPU)
GOLD = "SELECT sum(v) FROM items"
WRONG = "SELECT max(v) FROM items"


class Scripted:
    """Finishes with the gold SQL on even question numbers, a wrong one on odd numbers; confirms after review."""
    kind = "model"

    def __init__(self) -> None:
        self.metadata = {"calls": 0}

    def generate(self, conversations):
        out = []
        for messages in conversations:
            n = int(re.search(r"Question (\d+)", messages[1]["content"]).group(1))
            sql = GOLD if n % 2 == 0 else WRONG
            self.metadata["calls"] += 1
            out.append({"text": json.dumps({"action": "finish", "sql": sql}), "usage": {}, "finish_reason": "stop"})
        return out


def factory(spec, settings, kind, **_):
    return Scripted()


def make_db(path: Path, values: list[int]) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items(v INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(v,) for v in values])
    return digest(path)


def tokenized(messages: list[dict[str, str]], flags: list[bool]) -> dict[str, list[int]]:
    return tokenize_trajectory(messages, flags, TemplateTokenizer(CHATML), 8192)


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    dbs = {"train": ("shop", make_db(raw / "shop.sqlite", [1, 2, 4])),
           "validation": ("shop_v", make_db(raw / "shop_v.sqlite", [1, 2, 5])),
           "test": ("shop_t", make_db(raw / "shop_t.sqlite", [1, 3, 4]))}

    def task(i: int, split: str, source: str) -> Task:
        db_id, sha = dbs[split]
        return Task(f"{source}:{i}", db_id, split, f"Question {i}?", GOLD, f"{db_id}.sqlite", sha, source, "sha256:test")
    roles = {"pool": [task(i, "train", "bird-train") for i in range(8)],
             "dev": [task(i, "train", "bird-train") for i in range(20, 24)],
             "validation": [task(i, "validation", "bird-train") for i in range(30, 34)],
             "test": [task(i, "test", "bird-dev") for i in range(40, 44)]}
    reg = tmp_path / "registration"
    reg.mkdir()
    import dataclasses
    for role, tasks in roles.items():
        write_jsonl(reg / f"{role}.jsonl", [dataclasses.asdict(t) for t in tasks])
    source = {"protocol": "efa-v5-rft-natural-v1", "counts": {r: len(t) for r, t in roles.items()},
              "sha256": {r: digest(reg / f"{r}.jsonl") for r in roles}, "test_screen": {"candidates": 5}}
    v05 = fingerprint(source)
    (reg / "manifest.json").write_text(json.dumps({**source, "identity": v05}), encoding="utf-8")
    # model manifest + downloaded (tiny) snapshots
    models_root, manifest = tmp_path / "models", {}
    for kind, (model_id, revision) in MODELS.items():
        folder = models_root / kind
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
        files = {"config.json": {"size": (folder / "config.json").stat().st_size, "sha256": digest(folder / "config.json")}}
        manifest[model_id] = {"revision": revision, "files": files}
        mark_verified(folder, files)
    (tmp_path / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
    student_snapshot = expected_identity(model_spec("student", models_root / "student", DEFAULT),
                                         manifest[MODELS["student"][0]]["files"])["local_snapshot"]
    # v0.5 imports: the 4B solved pool tasks 0 and 2 in round 1 (R rows), never solved 1, 3, 5, 7
    imports = tmp_path / "imports"
    imports.mkdir()
    msgs = lambda i, sql: [{"role": "system", "content": "S"}, {"role": "user", "content": f"Question {i}?"},
                           {"role": "assistant", "content": json.dumps({"action": "finish", "sql": sql})}]
    r_rows = []
    for i in (0, 2):
        t = roles["pool"][i]
        r_rows.append({"task_id": t.id, "db_id": t.db_id, "split": "train", "source": "bird-train", "fixture": False,
                       "key": f"{t.id}#0", "frontier": False, "clean": True, "masked_turns": 0, "turns": 1,
                       "supervised_tokens": 5, **tokenized(msgs(i, GOLD), [True])})
    write_jsonl(imports / "sft_R.jsonl", r_rows)
    outcomes = {t.id: [2, 2 if i in (0, 2) else 0] for i, t in enumerate(roles["pool"])}
    (imports / "student_round1.json").write_text(json.dumps({"run": v05, "tasks": outcomes}), encoding="utf-8")
    (imports / "selection.json").write_text(json.dumps({
        "run": v05, "trajectories": 16, "sampled_tasks": 8, "frontier_tasks": 0, "easy_tasks": 2, "unsolved_tasks": 6,
        "datasets": {"R": {"sha256": digest(imports / "sft_R.jsonl"), "examples": 2}}}), encoding="utf-8")
    pair = {"task_id": roles["pool"][4].id, "db_id": "shop", "split": "train", "source": "bird-train", "fixture": False,
            "chosen": tokenized(msgs(4, GOLD), [True]), "rejected": tokenized(msgs(4, WRONG), [True])}
    write_jsonl(imports / "pairs.jsonl", [pair])
    (imports / "pairs.json").write_text(json.dumps({"run": v05, "pairs": 1, "sha256": digest(imports / "pairs.jsonl")}),
                                        encoding="utf-8")
    for name in ("R_seed42", "P_seed42"):
        folder = imports / "adapters" / name / "adapter"
        folder.mkdir(parents=True)
        (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
        (folder / "efa-base.json").write_text(json.dumps({"local_snapshot": student_snapshot}), encoding="utf-8")
        (folder.parent / "result.json").write_text(json.dumps({
            "training_completed": True, "adapter_reload_verified": True, "identity": "i", "adapter_hash": _adapter_hash(folder)}),
            encoding="utf-8")
        (folder.parent / "manifest.json").write_text(json.dumps({"identity": "i", "run": v05}), encoding="utf-8")
    settings = json.loads((REPO / "configs" / "v07-smoke.json").read_text(encoding="utf-8"))
    settings.update(sample_shards=2)
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, settings, models_root=models_root, manifest_path=tmp_path / "models.json",
                    imports=imports, smoke_limit=100)
    return run, frozen


def fake_trained(run: Path, frozen: dict, name: str) -> None:
    folder = run / "train" / name / "adapter"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    (folder.parent / "result.json").write_text(json.dumps({"training_completed": True, "adapter_reload_verified": True,
                                                           "adapter_hash": _adapter_hash(folder)}), encoding="utf-8")
    (folder.parent / "manifest.json").write_text(json.dumps({"run": frozen["identity"]}), encoding="utf-8")


def test_freeze_rejects_tampered_imports(fixture_run, tmp_path):
    run, frozen = fixture_run
    assert frozen["n_examples"] == 2 and frozen["imports"]["adapters"].keys() == {"R_seed42", "P_seed42"}
    assert set(frozen["gold"]) == {"dev", "validation", "test"}
    with open(run / "imports" / "sft_R.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    from exec_feedback.v7.freeze import imported
    with pytest.raises(ValueError):
        imported(run, frozen, "sft_R.jsonl")


def test_end_to_end_sample_build_evaluate_select_report(fixture_run):
    from exec_feedback.v7.analysis import report, select_final, select_x
    run, frozen = fixture_run
    for kind in ("teacher", "scale"):
        sample(run, kind=kind, stop=None, factory=factory)
        rows = [json.loads(l) for p in sorted((run / "samples" / kind / "samples").glob("shard*.jsonl"))
                for l in p.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 8 * 2 and all(r["correct"] == (int(r["task_id"].split(":")[1]) % 2 == 0) for r in rows)
        assert all(r["status"] == "submitted" and len(r["steps"]) == 2 for r in rows)  # finish, review, finish
    summary = build_teacher(run, tokenizer=TemplateTokenizer(CHATML))
    d = summary["datasets"]
    # T: N = |R in pool| = 2 teacher-verified examples (even tasks). TR: R rows + teacher picks on never-solved tasks
    # that the teacher solved (4, 6). T_raw: N rows without gold, so odd (wrong) tasks may enter.
    assert d["T"]["examples"] == 2 and d["T"]["origins"] == {"teacher": 2}
    assert d["TR"]["student_examples"] == 2 and d["TR"]["teacher_gap_examples"] == 2
    tr = [json.loads(l) for l in (run / "data" / "sft_TR.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["task_id"] for r in tr if r["origin"] == "teacher"} == {"bird-train:4", "bird-train:6"}
    assert d["Traw"]["examples"] == 2
    scale = build_scale(run, tokenizer=TemplateTokenizer(NON_THINKING_TEMPLATE))
    assert scale["datasets"]["R14"]["examples"] == 2
    for name in ("T_seed42", "TR_seed42", "Traw_seed42"):
        fake_trained(run, frozen, name)
    for condition in ("B", "R_seed42", "P_seed42", "T_seed42", "TR_seed42"):
        result = evaluate(run, condition=condition, role="dev", stop=None, factory=factory)
        assert result["complete"] and result["correct"] == 2  # even questions 20, 22
    rows, _ = merged(run, "dev", "T_seed42")
    assert len(rows) == 4 and all(r.get("gold_source") == "frozen_cache" for r in rows if r["final_sql"])
    assert select_x(run)["x"] in {"T", "TR"}
    px = {"T": "PT", "TR": "PTR"}[select_x(run)["x"]]
    fake_trained(run, frozen, f"{px}_seed42")
    evaluate(run, condition=f"{px}_seed42", role="dev", stop=None, factory=factory)
    assert select_final(run)["head"].endswith("_seed42")
    vote = evaluate(run, condition="B+vote", role="validation", stop=None, factory=factory)
    assert vote["complete"]
    result = report(run, procs=1)
    assert result["roles"]["dev"]["project"]["conditions"]["B"]["correct"] == 2
    assert "T-R" in result["roles"]["dev"]["project"]["deltas"]
    assert (run / "report" / "report.md").read_text(encoding="utf-8").startswith("# v0.7")
    # a model changed after verification is refused
    (Path(frozen["models"]["student"]["spec"]["local_path"]) / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        model_ready(load(run)[0], "student")


def test_untrained_or_foreign_adapter_is_refused(fixture_run):
    run, frozen = fixture_run
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate(run, condition="T_seed42", role="dev", stop=None, factory=factory)
    fake_trained(run, frozen, "T_seed42")
    (run / "train" / "T_seed42" / "adapter" / "adapter_config.json").write_text('{"changed": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        evaluate(run, condition="T_seed42", role="dev", stop=None, factory=factory)


# ------------------------------------------------------------------------------------------------ scheduler
def test_scheduler_dependencies_retries_and_deadline(tmp_path):
    from exec_feedback.v7.orchestrate import Job, Scheduler, exists

    py = sys.executable
    touch = lambda name: [py, "-c", f"open(r'{tmp_path / name}', 'w').close()"]
    attempts = tmp_path / "attempts"
    flaky = [py, "-c", f"import pathlib,sys; p=pathlib.Path(r'{attempts}'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
                       f"sys.exit(0 if len(p.read_text()) >= 2 else 1)"]
    jobs = [Job("a", touch("a"), 2, lambda: True, exists(tmp_path / "a"), 1, 1),
            Job("b", touch("b"), 1, exists(tmp_path / "a"), exists(tmp_path / "b"), 2, 1),
            Job("flaky", flaky + [], 0, exists(tmp_path / "b"), lambda: attempts.exists() and len(attempts.read_text()) >= 2,
                3, 1, retries=2),
            Job("never", touch("never"), 0, lambda: False, exists(tmp_path / "never"), 4, 1),
            Job("late", touch("late"), 2, lambda: True, exists(tmp_path / "late"), 5, 10 * 3600)]
    reached = []
    scheduler = Scheduler(gpus=[0], stop=time.time() + 3600, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                          plan=lambda: (jobs, []), poll=0.05, stagger=0, retry_delay=0, reserve=60,
                          milestone=reached.append)
    scheduler.loop()
    assert (tmp_path / "a").exists() and (tmp_path / "b").exists()
    assert len(attempts.read_text()) == 2  # failed once, retried, succeeded
    assert not (tmp_path / "never").exists() and not (tmp_path / "late").exists()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert status.index("start a ") < status.index("start b ") < status.index("start flaky ")


def test_scheduler_sampling_group_closes_when_claims_are_done(tmp_path):
    from exec_feedback.v7.orchestrate import Group, Job, Scheduler, exists

    py = sys.executable
    claims, marker = tmp_path / "claims", tmp_path / "DONE"
    claims.mkdir()
    worker = lambda n, stop: [py, "-c", f"import pathlib; [pathlib.Path(r'{claims}', str(i)).mkdir(exist_ok=True) for i in range(4)]"]
    group = Group("g", worker, lambda: True, lambda: len(list(claims.iterdir())) < 4, exists(marker),
                  lambda: marker.write_text("closed"), 1, 5, 0)
    after = Job("build", [py, "-c", f"open(r'{tmp_path / 'built'}', 'w').close()"], 0, exists(marker),
                exists(tmp_path / "built"), 2, 1)
    scheduler = Scheduler(gpus=[0, 1], stop=time.time() + 3600, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                          plan=lambda: ([after], [group]), poll=0.05, stagger=0, reserve=60)
    scheduler.loop()
    assert marker.read_text() == "closed" and (tmp_path / "built").exists()


def test_plan_holds_main_jobs_until_the_smoke_path_ran(fixture_run, tmp_path):
    from exec_feedback.v7.orchestrate import Plan

    run, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(run, main)
    plan = Plan(run=main, smoke_run=run, models_root=Path(frozen["models"]["student"]["spec"]["local_path"]).parent,
                main_py="py", vllm_py="vpy", stop_epoch=time.time() + 8 * 3600)
    jobs, groups = plan()
    by_name = {j.name: j for j in jobs}
    assert {"S:sample-scale", "S:sample-teacher", "M:sample-scale", "M:sample-teacher"} == {g.name for g in groups}
    assert by_name["S:eval-B-dev-0"].deps() and not by_name["M:eval-B-dev-0"].deps()  # waits for S:eval-R_seed42-dev
    assert not next(g for g in groups if g.name == "M:sample-teacher").deps()  # waits for S teacher data
    assert by_name["M:eval-B-dev-0"].priority > by_name["S:eval-B-dev-0"].priority


FAKE_CLI = r'''
import sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_v07 as t
from exec_feedback.v7 import analysis
from exec_feedback.v7.build import build_scale, build_teacher
from exec_feedback.v7.freeze import load
from exec_feedback.v7.rollouts import evaluate, sample

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
run = Path(opt("--run"))
if cmd == "sample":
    sample(run, kind=opt("--kind"), stop=opt("--stop"), factory=t.factory)
elif cmd == "evaluate":
    evaluate(run, condition=opt("--condition"), role=opt("--role"), shard=int(opt("--shard", "0")), stop=opt("--stop"),
             factory=t.factory)
elif cmd == "build":
    teacher = opt("--kind") == "teacher"
    (build_teacher if teacher else build_scale)(
        run, tokenizer=t.TemplateTokenizer(t.CHATML if teacher else t.NON_THINKING_TEMPLATE))
elif cmd in ("train", "prefer"):
    t.fake_trained(run, load(run)[0], opt("--family") + "_seed" + opt("--seed"))
elif cmd == "select-x":
    analysis.select_x(run)
elif cmd == "select-final":
    analysis.select_final(run)
elif cmd == "report":
    analysis.report(run, procs=1)
else:
    raise SystemExit("unexpected command " + cmd)
'''


def test_scheduler_runs_smoke_then_main_plan_end_to_end(fixture_run, tmp_path):
    """Both runs through the real Plan and Scheduler; GPU commands replaced by the scripted generator and fake
    trainings, CPU commands are the production functions."""
    from exec_feedback.v7.orchestrate import Plan, Scheduler

    smoke, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models"]["student"]["spec"]["local_path"]).parent,
                main_py=sys.executable, vllm_py=sys.executable, stop_epoch=time.time() + 9 * 3600)
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=[0, 1], stop=time.time() + 9 * 3600, status=tmp_path / "STATUS.txt",
                          log_dir=tmp_path / "logs", plan=plan, poll=0.1, stagger=0, retry_delay=0,
                          milestone=reached.append)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status, status
    for run in (smoke, main):
        head = json.loads((run / "select_final.json").read_text(encoding="utf-8"))["head"]
        assert (run / "eval" / "validation" / f"{head}+vote" / "shard00.summary.json").exists()
        assert (run / "samples" / "teacher" / "DONE.json").exists() and (run / "data" / "scale.json").exists()
    assert (smoke / "report" / "report.json").exists()
    for role in ("dev", "validation", "test"):
        for condition in ("B", "R_seed42", "P_seed42", "T_seed42", "TR_seed42", "Traw_seed42", "B14", "B32", "R14_seed42"):
            assert (main / "eval" / role / condition / "shard00.summary.json").exists(), (role, condition)
    # every main job started only after the smoke run had produced the same path
    assert status.index("start S:eval-R_seed42-dev-0") < status.index("start M:eval-B-dev-0")
    assert status.index("start S:build-teacher") < status.index("start M:sample-teacher-w1")
    assert "M:sample-teacher" in reached and "M:prefer" in reached
