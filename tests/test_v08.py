"""v0.8 engineering tests: SQLite fixtures, a scripted generator and a character tokenizer; no GPU, vLLM or weights.
Nothing here is a model result."""
from __future__ import annotations

import copy
import dataclasses
import json
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from exec_feedback.core import Task, digest, write_jsonl
from exec_feedback.pipeline import messages_for
from exec_feedback.rft.selection import tokenize_trajectory
from exec_feedback.storage import fingerprint
from exec_feedback.v8 import HINT_KEYS, MODELS, PROTOCOL
from exec_feedback.v8.build import build, strip_hint
from exec_feedback.v8.config import DEFAULT, validate_settings
from exec_feedback.v8.freeze import _adapter_hash, expected_identity, freeze, load, mark_verified, model_spec
from exec_feedback.v8.rollouts import evaluate, merged, parse, sample

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(copy.deepcopy(DEFAULT))
    assert DEFAULT["protocol"] == PROTOCOL and DEFAULT["train_examples"] == 3435
    for name in ("h20-v08.json", "v08-smoke.json"):
        validate_settings(json.loads((REPO / "configs" / name).read_text(encoding="utf-8")))
    assert json.loads((REPO / "configs" / "h20-v08.json").read_text(encoding="utf-8")) == DEFAULT
    for change in ({"train_examples": 4096}, {"pairs_limit": 8}, {"extra": 1}, {"teacher_gpus": 3},
                   {"gpu_memory": {"student": 0.4}}, {"gate_delta": 0.9}, {"seeds": [42, 42]}):
        with pytest.raises(ValueError):
            validate_settings({**copy.deepcopy(DEFAULT), **change})


def test_models_manifest_pins_every_file():
    manifest = json.loads((REPO / "configs" / "v08-models.json").read_text(encoding="utf-8"))
    for model_id, revision in MODELS.values():
        entry = manifest[model_id]
        assert entry["revision"] == revision
        assert {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"} <= set(entry["files"])
        assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0 for f in entry["files"].values())
    teacher = manifest[MODELS["teacher"][0]]["files"]
    assert sum(f["size"] for f in teacher.values()) > 230e9 and len([n for n in teacher if n.endswith(".safetensors")]) == 24
    # The 4B pin reproduces v0.5's snapshot fingerprint, so the imported R/P adapters load on it.
    student = expected_identity(model_spec("student", Path("x"), DEFAULT), manifest[MODELS["student"][0]]["files"])
    assert student["local_snapshot"] == "45ba5b8b179211a84e763d2c11f393407912c90a328ef3a3c9238fda0a413787"
    # Same fast-tokenizer files and chat template, so teacher dialogues tokenize as the student's own (merges.txt
    # differs only by a "#version: 0.2" header line; the fast tokenizer reads tokenizer.json).
    s = manifest[MODELS["student"][0]]["files"]
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        assert teacher[name]["sha256"] == s[name]["sha256"], name


def test_moe_glibc_patch_on_a_synthetic_library(tmp_path):
    """The patch rewrites only a GLIBC_2.29 requirement bound to log2; anything else bound to it is refused."""
    sys.path.insert(0, str(REPO / "scripts"))
    import patch_moe_glibc as p

    lib = synthetic_elf(tmp_path / "lib.so", ["log2"])
    assert p.patch(lib).startswith("patched ['log2']")
    assert p.patch(lib).startswith("nothing to do") and (tmp_path / "lib.so.orig").exists()
    secs = p.sections(lib.read_bytes())
    data = lib.read_bytes()
    aux = secs[".gnu.version_r"]["off"] + 16
    assert data[secs[".dynstr"]["off"] + int.from_bytes(data[aux + 8:aux + 12], "little"):].startswith(b"GLIBC_2.2.5\0")
    assert int.from_bytes(data[aux:aux + 4], "little") == p.elf_hash(b"GLIBC_2.2.5")
    with pytest.raises(ValueError, match="refusing"):
        p.patch(synthetic_elf(tmp_path / "other.so", ["log2", "exp"]))


def synthetic_elf(path: Path, symbols: list[str]) -> Path:
    """Minimal ELF64 with .dynstr/.dynsym/.gnu.version/.gnu.version_r: libm needs GLIBC_2.29 (index 2) for
    `symbols`, and GLIBC_2.2.5 is present in .dynstr (as in the real _moe_C)."""
    import struct
    dynstr = b"\0libm.so.6\0GLIBC_2.29\0GLIBC_2.2.5\0" + b"".join(s.encode() + b"\0" for s in symbols)
    off = lambda s: dynstr.index(b"\0" + s + b"\0") + 1
    dynsym = b"\0" * 24 + b"".join(struct.pack("<IBBHQQ", off(s.encode()), 0x12, 0, 0, 0, 0) for s in symbols)
    versym = struct.pack("<H", 0) + b"".join(struct.pack("<H", 2) for _ in symbols)
    verneed = struct.pack("<HHIII", 1, 1, off(b"libm.so.6"), 16, 0) + \
        struct.pack("<IHHII", 0x6967a99, 0, 2, off(b"GLIBC_2.29"), 0)
    shstr = b"\0.dynstr\0.dynsym\0.gnu.version\0.gnu.version_r\0.shstrtab\0"
    blobs = [dynstr, dynsym, versym, verneed, shstr]
    offsets, pos = [], 64
    for b in blobs:
        offsets.append(pos)
        pos += len(b)
    shoff = pos
    names = [shstr.index(n) for n in (b".dynstr", b".dynsym", b".gnu.version", b".gnu.version_r", b".shstrtab")]
    types = [3, 11, 0x6fffffff, 0x6ffffffe, 3]
    infos = [0, 1, 0, 1, 0]
    header = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9 + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64,
                                                                    0, 0, 64, 6, 5)
    sections = b"\0" * 64 + b"".join(struct.pack("<IIQQQQIIQQ", names[i], types[i], 0, 0, offsets[i], len(blobs[i]),
                                                 0, infos[i], 1, 0) for i in range(5))
    path.write_bytes(header + b"".join(blobs) + sections)
    return path


@pytest.mark.parametrize("ranges_supported", [True, False])
def test_ranged_download_assembles_and_verifies(tmp_path, monkeypatch, ranges_supported):
    """Files larger than two parts arrive as byte ranges and are assembled and hash-checked; a server that ignores
    ranges gets a whole-file download instead."""
    import hashlib
    import http.server
    import threading
    from exec_feedback.v7 import download as v7download
    from exec_feedback.v8 import download

    served = tmp_path / "served"
    served.mkdir()
    (served / "big.bin").write_bytes(bytes(range(256)) * 20000)  # 5.12 MB = 6 parts of 1 MiB
    (served / "config.json").write_text('{"a": 1}', encoding="utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data = (served / self.path.lstrip("/")).read_bytes()
            spec = self.headers.get("Range")
            if spec and ranges_supported:
                start, end = (int(x) for x in spec.split("=")[1].split("-"))
                body, code = data[start:end + 1], 206
            else:
                body, code = data, 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{{name}}"
        monkeypatch.setattr(download, "PART", 1 << 20)
        monkeypatch.setattr(download, "SOURCES", {"modelscope": url})
        monkeypatch.setattr(v7download, "SOURCES", {"modelscope": url})
        files = {n: {"size": (served / n).stat().st_size, "sha256": hashlib.sha256((served / n).read_bytes()).hexdigest()}
                 for n in ("big.bin", "config.json")}
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({"x/y": {"revision": "r", "files": files}}), encoding="utf-8")
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", ["download", "--manifest", str(manifest), "--model", "x/y", "--out", str(out),
                                          "--sources", "modelscope", "--workers", "4"])
        assert download.main() == 0
    finally:
        server.shutdown()
    assert (out / "big.bin").read_bytes() == (served / "big.bin").read_bytes()
    assert (out / ".efa-verified.json").exists() and not list(out.glob("*.range*")) and not list(out.glob("*.part"))


# ------------------------------------------------------------------------------------------------ hint
def test_strip_hint_restores_the_exact_unhinted_prompt():
    task = SimpleNamespace(question="How many?", evidence="e = \"x\"", schema=None)
    schema = {"rows": [["table", "t", "CREATE TABLE t(v)"]]}
    plain = messages_for(task, schema, 4, True, review_success=True)
    hinted = messages_for(task, schema, 4, True, review_success=True,
                          extra={"reference_sql": "SELECT 1", "reference_note": "note"})
    assert hinted[1]["content"] != plain[1]["content"]
    dialogue = hinted + [{"role": "assistant", "content": '{"action":"finish","sql":"SELECT 1"}'}]
    assert strip_hint(dialogue) == plain + dialogue[2:]
    assert all(key not in strip_hint(dialogue)[1]["content"] for key in HINT_KEYS)
    with pytest.raises(ValueError):
        strip_hint(plain + dialogue[2:])


def test_condition_names():
    assert parse("B") == ("student", None) and parse("B235") == ("teacher", None)
    assert parse("PHR_seed43") == ("student", "PHR_seed43") and parse("TR_seed42") == ("student", "TR_seed42")
    for bad in ("T_seed42", "B32", "HR", "B+vote"):
        with pytest.raises(ValueError):
            parse(bad)


# ------------------------------------------------------------------------------------------------ end to end (CPU)
class TemplateTokenizer:
    """ChatML rendered with jinja2 (as transformers does); ids are shifted code points, <|im_end|> = 2."""
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
GOLD = "SELECT sum(v) FROM items"
WRONG = "SELECT max(v) FROM items"


class Scripted:
    """Uses a reference SQL when the prompt carries one; otherwise the gold SQL on even question numbers and a wrong
    one on odd numbers. Confirms after the review turn."""
    kind = "model"

    def __init__(self) -> None:
        self.metadata = {"calls": 0}

    def generate(self, conversations):
        out = []
        for messages in conversations:
            public = json.loads(messages[1]["content"])
            n = int(re.search(r"Question (\d+)", public["question"]).group(1))
            sql = public.get("reference_sql") or (GOLD if n % 2 == 0 else WRONG)
            self.metadata["calls"] += 1
            out.append({"text": json.dumps({"action": "finish", "sql": sql}), "usage": {}, "finish_reason": "stop"})
        return out


def factory(spec, settings, kind, **_):
    return Scripted()


class Oracle(Scripted):
    """A teacher that always submits the gold SQL (to drive the main run through a passed gate)."""

    def generate(self, conversations):
        self.metadata["calls"] += len(conversations)
        return [{"text": json.dumps({"action": "finish", "sql": GOLD}), "usage": {}, "finish_reason": "stop"}
                for _ in conversations]


def strong_factory(spec, settings, kind, **_):
    return Oracle() if kind == "teacher" else Scripted()


def make_db(path: Path, values: list[int]) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items(v INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(v,) for v in values])
    return digest(path)


def tokenized(messages: list[dict[str, str]], flags: list[bool]) -> dict[str, list[int]]:
    return tokenize_trajectory(messages, flags, TemplateTokenizer(CHATML), 8192)


@pytest.fixture
def fixture_run(tmp_path):
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
    for role, tasks in roles.items():
        write_jsonl(reg / f"{role}.jsonl", [dataclasses.asdict(t) for t in tasks])
    source = {"protocol": "efa-v5-rft-natural-v1", "counts": {r: len(t) for r, t in roles.items()},
              "sha256": {r: digest(reg / f"{r}.jsonl") for r in roles}, "test_screen": {"candidates": 5}}
    v05 = fingerprint(source)
    (reg / "manifest.json").write_text(json.dumps({**source, "identity": v05}), encoding="utf-8")
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
    # v0.5 imports: the 4B solved pool tasks 0 and 2 in round 1 (R rows), never solved 1, 3, 4, 5, 6, 7
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
    settings = json.loads((REPO / "configs" / "v08-smoke.json").read_text(encoding="utf-8"))
    settings.update(sample_shards=2)
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, settings, models_root=models_root, manifest_path=tmp_path / "models.json",
                    imports=imports, smoke_limit=100)
    return run, frozen


def fake_trained(run: Path, frozen: dict, name: str) -> None:
    from exec_feedback.storage import atomic_json
    folder = run / "train" / name / "adapter"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    atomic_json(folder.parent / "manifest.json", {"run": frozen["identity"]})
    atomic_json(folder.parent / "result.json", {"training_completed": True, "adapter_reload_verified": True,
                                                "adapter_hash": _adapter_hash(folder)})


def test_freeze_records_the_unsolved_tasks(fixture_run):
    run, frozen = fixture_run
    assert frozen["n_examples"] == 2 and frozen["imports"]["adapters"].keys() == {"R_seed42", "P_seed42"}
    assert frozen["unsolved_tasks"] == [f"bird-train:{i}" for i in (1, 3, 4, 5, 6, 7)]
    with open(run / "imports" / "sft_R.jsonl", "a", encoding="utf-8") as fh:
        fh.write("\n")
    from exec_feedback.v8.freeze import imported
    with pytest.raises(ValueError):
        imported(run, frozen, "sft_R.jsonl")


def test_end_to_end_sample_build_evaluate_gate_report(fixture_run):
    from exec_feedback.v8.analysis import gate, report

    run, frozen = fixture_run
    for sampler, k in (("teacher", 4), ("hint", 2)):
        sample(run, sampler=sampler, stop=None, factory=factory)
        rows = [json.loads(l) for p in sorted((run / "samples" / sampler / "samples").glob("shard*.jsonl"))
                for l in p.read_text(encoding="utf-8").splitlines()]
        assert {r["task_id"] for r in rows} == set(frozen["unsolved_tasks"]) and len(rows) == 6 * k
        assert all(r["hinted"] == (sampler == "hint") for r in rows)
        hinted = ["reference_sql" in r["messages"][1]["content"] for r in rows]
        assert all(hinted) if sampler == "hint" else not any(hinted)
    tok = TemplateTokenizer(CHATML)
    tr = build(run, sampler="teacher", tokenizer=tok)["datasets"]["TR"]
    hr = build(run, sampler="hint", tokenizer=tok)["datasets"]["HR"]
    # TR: R rows + the teacher's verified picks on never-solved tasks it solved (even numbers 4, 6);
    # HR: R rows + the hinted 4B on every never-solved task.
    assert (tr["student_examples"], tr["gap_examples"]) == (2, 2) and (hr["student_examples"], hr["gap_examples"]) == (2, 6)
    rows = [json.loads(l) for l in (run / "data" / "sft_HR.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["task_id"] for r in rows if r["origin"] == "hint"} == set(frozen["unsolved_tasks"])
    for r in rows:  # no hint text reaches the training tokens
        text = "".join(chr(i - 10) for i in r["input_ids"] if i != 2)
        assert "reference_sql" not in text and "reference_note" not in text
    for name in ("TR_seed42", "HR_seed42"):
        fake_trained(run, frozen, name)
    for condition in ("B", "B235", "R_seed42", "P_seed42", "TR_seed42", "HR_seed42"):
        result = evaluate(run, condition=condition, role="dev", stop=None, factory=factory)
        assert result["complete"] and result["correct"] == 2  # even questions 20, 22
    rows, _ = merged(run, "dev", "HR_seed42")
    assert len(rows) == 4 and all(r.get("gold_source") == "frozen_cache" for r in rows if r["final_sql"])
    assert gate(run) == {"run": frozen["identity"], "passed": True, "smoke": True,
                         "rule": "smoke run: engineering path only, the gate is not evaluated"}
    for name in ("PTR_seed42", "PHR_seed42"):
        fake_trained(run, frozen, name)
        evaluate(run, condition=name, role="dev", stop=None, factory=factory)
    result = report(run, procs=1)
    dev = result["roles"]["dev"]["project"]
    assert dev["conditions"]["B"]["correct"] == 2 and {"TR-R", "HR-R", "TR-HR", "B235-B"} <= set(dev["deltas"])
    assert result["select_final"]["family"] in {"R", "P", "TR", "HR", "PTR", "PHR"}
    assert (run / "report" / "report.md").read_text(encoding="utf-8").startswith("# v0.8")


def test_main_gate_rule(fixture_run, monkeypatch):
    """Outside smoke the gate compares the complete dev evaluations of B and B235 and applies the registered rule."""
    from exec_feedback.v8 import analysis

    run, frozen = fixture_run
    monkeypatch.setattr(analysis, "load", lambda r, role=None, **kw: ({**frozen, "smoke_limit": 0}, load(run, role)[1]))
    with pytest.raises(ValueError, match="complete dev evaluations"):
        analysis.gate(run)
    for condition in ("B", "B235"):
        evaluate(run, condition=condition, role="dev", stop=None, factory=factory)
    result = analysis.gate(run)
    assert result["passed"] is False and result["delta"] == 0 and result["threshold"] == 0.05
    assert result["dev_rates"] == {"B": 0.5, "B235": 0.5}


def test_untrained_or_foreign_adapter_is_refused(fixture_run):
    run, frozen = fixture_run
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate(run, condition="TR_seed42", role="dev", stop=None, factory=factory)
    fake_trained(run, frozen, "TR_seed42")
    (run / "train" / "TR_seed42" / "adapter" / "adapter_config.json").write_text('{"changed": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        evaluate(run, condition="TR_seed42", role="dev", stop=None, factory=factory)


# ------------------------------------------------------------------------------------------------ scheduler
def test_scheduler_wide_job_holds_its_block_and_long_jobs_avoid_it(tmp_path):
    from exec_feedback.v8.orchestrate import Job, Scheduler, exists

    py = sys.executable
    touch = lambda name, wait=0.0: [py, "-c", f"import time; time.sleep({wait}); open(r'{tmp_path / name}', 'w').close()"]
    jobs = [Job("short", touch("short", 1.0), 2, lambda: True, exists(tmp_path / "short"), 1, 1),
            Job("wide", touch("wide"), 2, lambda: True, exists(tmp_path / "wide"), 2, 1, width=2),
            Job("later", touch("later"), 2, lambda: True, exists(tmp_path / "later"), 3, 1),
            Job("long", touch("long"), 2, lambda: True, exists(tmp_path / "long"), 4, 1, avoid=lambda: {0, 1})]
    scheduler = Scheduler(gpus=[0, 1, 2], stop=time.time() + 3600, status=tmp_path / "STATUS.txt",
                          log_dir=tmp_path / "logs", plan=lambda: (jobs, []), poll=0.05, stagger=0, reserve=60)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    # 'short' takes GPU 2 (high end first); 'wide' gets block 0,1; 'later' waits for a GPU and 'long' never uses 0/1
    assert "start short on GPU 2" in status and "start wide on GPUs 0,1" in status
    assert re.search(r"start long on GPU 2 ", status)
    assert all((tmp_path / n).exists() for n in ("short", "wide", "later", "long"))


def test_scheduler_held_block_blocks_smaller_jobs(tmp_path):
    from exec_feedback.v8.orchestrate import Job, Scheduler, exists

    py = sys.executable
    touch = lambda name, wait=0.0: [py, "-c", f"import time; time.sleep({wait}); open(r'{tmp_path / name}', 'w').close()"]
    jobs = [Job("busy", touch("busy", 1.5), 1, lambda: True, exists(tmp_path / "busy"), 1, 1),
            Job("wide", touch("wide"), 2, exists(tmp_path / "go"), exists(tmp_path / "wide"), 2, 1, width=2),
            Job("small", touch("small"), 1, exists(tmp_path / "go"), exists(tmp_path / "small"), 3, 1)]
    scheduler = Scheduler(gpus=[0, 1], stop=time.time() + 3600, status=tmp_path / "STATUS.txt",
                          log_dir=tmp_path / "logs", plan=lambda: (jobs, []), poll=0.05, stagger=0, reserve=60)
    scheduler.step()                    # 'busy' starts on GPU 1
    (tmp_path / "go").write_text("")
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    # 'small' could have shared GPU 0 or 1 at once, but the waiting wide job holds the block until 'busy' ends
    assert status.index("start wide") < status.index("start small")


def test_scheduler_measured_estimates_and_late_notice(tmp_path):
    from exec_feedback.v8.orchestrate import Job, Scheduler, exists

    late = Job("M:late", [sys.executable, "-c", "pass"], 0, lambda: True, exists(tmp_path / "x"), 1, 10 * 3600,
               kind="train")
    scheduler = Scheduler(gpus=[0], stop=time.time() + 3600, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                          plan=lambda: ([late], []), poll=0.05, reserve=60)
    assert scheduler.estimate(late) == 10 * 3600
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert status.count("not starting M:late") == 1
    scheduler.durations["train"] = [600.0, 900.0]
    assert scheduler.estimate(late) == pytest.approx(1.15 * 900 + 60)


def test_scheduler_survives_a_plan_error(tmp_path):
    from exec_feedback.v8.orchestrate import Job, Scheduler, exists

    job = Job("a", [sys.executable, "-c", f"open(r'{tmp_path / 'a'}', 'w').close()"], 0, lambda: True,
              exists(tmp_path / "a"), 1, 1)
    calls = []

    def plan():
        calls.append(1)
        if len(calls) <= 2:
            raise ValueError("half-written artifact")
        return [job], []
    scheduler = Scheduler(gpus=[0], stop=time.time() + 3600, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                          plan=plan, poll=0.05, reserve=60)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert (tmp_path / "a").exists() and status.count("plan error (ValueError: half-written artifact)") == 1


def test_scheduler_dependencies_retries_and_wide_group(tmp_path):
    from exec_feedback.v8.orchestrate import Group, Job, Scheduler, exists

    py = sys.executable
    claims, marker = tmp_path / "claims", tmp_path / "DONE"
    claims.mkdir()
    worker = lambda n, stop: [py, "-c", f"import os, pathlib; assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2; "
                                         f"[pathlib.Path(r'{claims}', str(i)).mkdir(exist_ok=True) for i in range(4)]"]
    group = Group("g", worker, lambda: True, lambda: len(list(claims.iterdir())) < 4, exists(marker),
                  lambda: marker.write_text("closed"), 1, 5, 0, width=2, max_failures=2)
    attempts = tmp_path / "attempts"
    flaky = [py, "-c", f"import pathlib,sys; p=pathlib.Path(r'{attempts}'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
                       f"sys.exit(0 if len(p.read_text()) >= 2 else 1)"]
    after = Job("flaky", flaky, 0, exists(marker), lambda: attempts.exists() and len(attempts.read_text()) >= 2, 2, 1)
    scheduler = Scheduler(gpus=[0, 1], stop=time.time() + 3600, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                          plan=lambda: ([after], [group]), poll=0.05, stagger=0, retry_delay=0, reserve=60)
    scheduler.loop()
    assert marker.read_text() == "closed" and len(attempts.read_text()) == 2


FAKE_CLI = r'''
import sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_v08 as t
from exec_feedback.v8 import analysis
from exec_feedback.v8.build import build
from exec_feedback.v8.freeze import load
from exec_feedback.v8.rollouts import evaluate, sample

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
run = Path(opt("--run"))
if cmd == "sample":
    sample(run, sampler=opt("--sampler"), stop=opt("--stop"), factory=t.{factory})
elif cmd == "evaluate":
    evaluate(run, condition=opt("--condition"), role=opt("--role"), stop=opt("--stop"), factory=t.{factory})
elif cmd == "build":
    build(run, sampler=opt("--sampler"), tokenizer=t.TemplateTokenizer(t.CHATML))
elif cmd in ("train", "prefer"):
    t.fake_trained(run, load(run)[0], opt("--family") + "_seed" + opt("--seed"))
elif cmd == "gate":
    analysis.gate(run)
elif cmd == "report":
    analysis.report(run, procs=1)
else:
    raise SystemExit("unexpected command " + cmd)
'''


@pytest.mark.parametrize("teacher", ["weak", "strong"])
def test_scheduler_runs_smoke_then_main_plan_end_to_end(fixture_run, tmp_path, teacher):
    """Both runs through the real Plan and Scheduler; GPU commands replaced by scripted generators and fake
    trainings, CPU commands are the production functions. The main run's gate decides on real (scripted) dev
    evaluations: a weak teacher ties B, so its path stops at the gate and HR carries the run; a strong one passes and
    the TR / PTR path runs to the end."""
    from exec_feedback.v8.orchestrate import Plan, Scheduler

    smoke, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["smoke_limit"] = 0  # the copy plays the main run (smoke_limit only switches the gate rule here)
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests"),
                                   factory="factory" if teacher == "weak" else "strong_factory"), encoding="utf-8")
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models"]["student"]["spec"]["local_path"]).parent,
                main_py=sys.executable, vllm_py=sys.executable, stop_epoch=time.time() + 9 * 3600, gpus=[0, 1, 2, 3])
    plan.block = {0, 1}
    plan.settings["M"]["teacher_gpus"] = plan.settings["S"]["teacher_gpus"] = 2
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=[0, 1, 2, 3], stop=time.time() + 9 * 3600, status=tmp_path / "STATUS.txt",
                          log_dir=tmp_path / "logs", plan=plan, poll=0.1, stagger=0, retry_delay=0,
                          milestone=reached.append)
    plan.scheduler = scheduler
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status, status
    assert (smoke / "report" / "report.json").exists() and (smoke / "data" / "teacher.json").exists()
    gate = json.loads((main / "gate.json").read_text(encoding="utf-8"))
    families = ["B", "R_seed42", "P_seed42", "HR_seed42", "PHR_seed42"]
    if teacher == "weak":
        assert gate["passed"] is False and gate["delta"] == 0
        assert not (main / "samples" / "teacher").exists() and not (main / "train" / "TR_seed42").exists()
    else:
        assert gate["passed"] is True and gate["delta"] == 0.5
        assert json.loads((main / "data" / "teacher.json").read_text(encoding="utf-8"))["datasets"]["TR"]["gap_examples"] == 6
        families += ["TR_seed42", "PTR_seed42"]
        for role in ("validation", "test"):
            assert (main / "eval" / role / "B235" / "summary.json").exists()
        assert status.index("M:sample-teacher: sampling closed") < status.index("start M:train-TR_seed42")
    for role in ("dev", "validation", "test"):
        for condition in families:
            assert (main / "eval" / role / condition / "summary.json").exists(), (role, condition)
    # every main path started only after the smoke run had produced it; the teacher pilot ran on its block
    assert status.index("start S:eval-R_seed42-dev") < status.index("start M:eval-P_seed42-dev")
    assert status.index("start S:build-hint") < status.index("start M:sample-hint-w1")
    assert re.search(r"start M:eval-B235-dev on GPUs (0,1|2,3)", status)
    assert {"M:gate", "M:prefer-HR", "M:eval-HR", "M:eval-PHR", "M:eval-base"} <= set(reached)
    assert ("M:prefer-TR" in reached) == (teacher == "strong")
