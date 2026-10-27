"""v0.12 engineering tests: SQLite fixtures, a character tokenizer, a scripted engine and a randomly initialised tiny
Qwen3 with a LoRA (from test_v09), fake teachers and a local scoring service; no GPU, vLLM or real weights. Nothing here
is a model result."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v09 import GOLD, ChatTokenizer, Scripted, ScriptedEngine, fake_adapter, factory, make_db, tiny_policy  # noqa: E402

from exec_feedback.core import Task, digest, write_jsonl  # noqa: E402
from exec_feedback.storage import fingerprint, read_json, recover_records  # noqa: E402
from exec_feedback.v8.freeze import expected_identity, mark_verified, model_spec  # noqa: E402
from exec_feedback.v12 import EXPLORE, FAMILY, MODELS, PROTOCOL, TEACHER, V08_PROTOCOL  # noqa: E402
from exec_feedback.v12 import online, teacher as tm  # noqa: E402
from exec_feedback.v12.config import DEFAULT, REGISTERED, validate_settings  # noqa: E402
from exec_feedback.v12.evaluate import adapter_for, common_steps, evaluate, evaluate_teacher, parse  # noqa: E402
from exec_feedback.v12.freeze import freeze, load, models_manifest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def smoke_settings() -> dict:
    return json.loads((REPO / "configs" / "v12-smoke.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(dict(DEFAULT))
    assert DEFAULT["protocol"] == PROTOCOL and {k: DEFAULT[k] for k in REGISTERED} == REGISTERED
    assert json.loads((REPO / "configs" / "h20-v12.json").read_text(encoding="utf-8")) == DEFAULT
    smoke = smoke_settings()
    validate_settings(smoke)
    assert smoke["phase"] == "smoke" and smoke["planned_steps"] == 3 and smoke["lora_check_step"] < 3
    models = models_manifest(REPO / "configs" / "v12-models.json")
    assert {models[m]["revision"] for m, _ in MODELS.values()} == {r for _, r in MODELS.values()}
    for bad in ({"learning_rate": 2e-05}, {"tasks_per_step": 16}, {"group_size": 8}, {"planned_steps": 200},
                {"explore_seeds": [45]}, {"teacher_score_context": 8192}, {"lora_check_step": 400},
                {"rl_slots": 64}, {"max_steps": 6}, {"seeds": [42, 42]}, {"teacher_gpus": 2}):
        with pytest.raises(ValueError):
            validate_settings({**DEFAULT, **bad})
    validate_settings({**smoke, "learning_rate": 2e-05})  # smoke settings are never results


def test_condition_names():
    assert parse("B") == ("B", None, None) and parse(TEACHER) == (TEACHER, None, None)
    assert parse("TR_seed43") == ("TR", 43, None) and parse("PTR_seed42") == ("PTR", 42, None)
    assert parse("OPD_seed44") == ("OPD", 44, None) and parse("OPD_seed42@100") == ("OPD", 42, 100)
    assert parse("TROPD_seed42") == ("TROPD", 42, None)
    for bad in ("TR_seed42@10", "R_seed42", "RL_seed42", "OPD", "OPD_seed42@", "B235_seed42", "TROPD42"):
        with pytest.raises(ValueError):
            parse(bad)


def test_the_teacher_must_share_the_tokenizer(tmp_path):
    manifest = json.loads((REPO / "configs" / "v12-models.json").read_text(encoding="utf-8"))
    manifest[MODELS["teacher"][0]]["files"]["tokenizer.json"]["sha256"] = "0" * 64
    (tmp_path / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer files differ"):
        models_manifest(tmp_path / "models.json")


# ------------------------------------------------------------------------------------------------ teacher alignment
class LP:
    def __init__(self, v):
        self.logprob = v


def test_prompt_logprobs_line_up_with_the_tokens_sent():
    sent = [5, 6, 7]
    rows = [None, {6: LP(-0.5), 9: LP(-0.1)}, {7: LP(-2.0)}]
    assert tm.aligned(sent, [5, 6, 7], rows) == [None, -0.5, -2.0]
    assert tm.aligned(sent, None, [None, {9: LP(-0.1)}, {7: LP(-2.0)}]) == [None, None, -2.0]  # missing, not invented
    with pytest.raises(tm.TeacherMismatch, match="other token ids"):
        tm.aligned(sent, [5, 6, 8], rows)
    with pytest.raises(tm.TeacherMismatch, match="one log-prob per position"):
        tm.aligned(sent, [5, 6, 7], rows[:2])


class PositionTeacher:
    """log p_T of position p is -p/1000 (so a misread position shows); `holes` positions come back missing."""

    def __init__(self, holes: set[int] | None = None) -> None:
        self.holes, self.calls, self.last = holes or set(), [], {"score_seconds": 0.5, "wait_seconds": 0.25}

    def score(self, sequences):
        self.calls.append([list(s) for s in sequences])
        return [[None] + [None if p in self.holes else -p / 1000 for p in range(1, len(s))] for s in sequences]


def test_teacher_values_are_taken_at_the_trained_positions():
    a = {"usable": True, "input_ids": list(range(20, 30)), "positions": [4, 5, 9], "old_logprobs": [-0.1, None, -0.3]}
    b = {"usable": False, "reason": "retokenized"}
    c = {"usable": True, "input_ids": list(range(40, 48)), "positions": [6, 7], "old_logprobs": [-0.2, -0.2]}
    teacher = PositionTeacher()
    out, missing, timing = online.with_teacher([a, None, b, c], teacher)
    assert len(teacher.calls) == 1 and teacher.calls[0] == [a["input_ids"], c["input_ids"]]  # one call, usable only
    assert out[0]["teacher_logprobs"] == [-0.004, -0.005, -0.009] and out[3]["teacher_logprobs"] == [-0.006, -0.007]
    assert out[1] is None and out[2] is None and missing == [None] * 4
    assert timing["teacher_sequences"] == 2 and timing["teacher_tokens"] == 18 and timing["teacher_wait_seconds"] == 0.25
    assert online.dialogue_kl(out[0]) == pytest.approx(((-0.1 + 0.004) + (-0.3 + 0.009)) / 2)  # sampler-known only
    out, missing, _ = online.with_teacher([a, c], PositionTeacher(holes={7}))
    assert out[0] is not None and out[1] is None and missing == [None, "teacher_missing"]


# ------------------------------------------------------------------------------------------------ the OPD objective
def plain(model):
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model


def student_logprobs(model, item):
    import torch
    with torch.no_grad():
        return online.sequence_logprobs(model, [item], pad_id=2)[0].tolist()


def teacher_model():
    """A second tiny Qwen3 (other weights) as the teacher."""
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=8192,
                         tie_word_embeddings=False)
    return Qwen3ForCausalLM(config).eval()


def test_the_signal_is_zero_when_the_teacher_is_the_sampler():
    torch = pytest.importorskip("torch")
    model = plain(tiny_policy()()[0])
    item = {"input_ids": list(range(20, 32)), "positions": [6, 7, 8, 9, 10, 11]}
    own = student_logprobs(model, item)
    stats = online.opd_backward(model, [{**item, "old_logprobs": own, "teacher_logprobs": own}], pad_id=2,
                                normaliser=1.0, cap=2.0, budget=100)
    assert stats["reverse_kl_sum"] == pytest.approx(0.0, abs=1e-6) and stats["advantage_abs_sum"] < 1e-5
    assert all(p.grad is None or p.grad.abs().max() < 1e-6 for p in model.parameters() if p.requires_grad)


def opd_step(item, teacher_lp, flip: bool = False) -> tuple[float, float]:
    """Mean |log p_student - log p_teacher| on the item's tokens before and after one SGD step on the OPD loss. flip
    reverses the signal's sign (the 'wrong sign' bug the test must catch)."""
    torch = pytest.importorskip("torch")
    model = plain(tiny_policy()()[0])
    before = student_logprobs(model, item)
    target = [2 * s - t for s, t in zip(before, teacher_lp)] if flip else teacher_lp
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    online.opd_backward(model, [{**item, "old_logprobs": before, "teacher_logprobs": target}], pad_id=2,
                        normaliser=1.0, cap=2.0, budget=100)
    optimizer.step()
    after = student_logprobs(model, item)
    gap = lambda values: sum(abs(v - t) for v, t in zip(values, teacher_lp)) / len(values)
    return gap(before), gap(after)


def test_an_opd_step_moves_the_student_toward_the_teacher_and_a_flipped_sign_would_not():
    torch = pytest.importorskip("torch")
    teacher = teacher_model()
    item = {"input_ids": [20, 25, 31, 22, 40, 41, 52, 33, 60, 61, 62, 70], "positions": list(range(4, 12))}
    with torch.no_grad():
        logits = teacher(input_ids=torch.tensor([item["input_ids"]])).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        teacher_lp = [float(lp[p - 1, item["input_ids"][p]]) for p in item["positions"]]
    before, after = opd_step(item, teacher_lp)
    assert after < before
    before, after = opd_step(item, teacher_lp, flip=True)
    assert after > before  # the same check fails if the sign is wrong


def test_every_lora_weight_gets_an_opd_gradient_under_checkpointing():
    import warnings
    torch = pytest.importorskip("torch")
    model = tiny_policy(checkpointing=True)()[0]
    model.train()
    item = {"input_ids": list(range(20, 60)), "positions": list(range(30, 40)), "old_logprobs": [None] * 10,
            "teacher_logprobs": [-3.0] * 10}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "None of the inputs have requires_grad=True" would fail here
        online.opd_backward(model, [item], pad_id=2, normaliser=1.0, cap=2.0, budget=1000)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_tokens_without_a_sampler_logprob_use_the_trainer_and_the_ratio_is_capped():
    torch = pytest.importorskip("torch")
    model = plain(tiny_policy()()[0])
    item = {"input_ids": list(range(20, 30)), "positions": [5, 6, 7]}
    own = student_logprobs(model, item)
    far = {**item, "old_logprobs": [own[0] - 5.0, None, own[2]], "teacher_logprobs": [own[0], own[1] - 1.0, own[2]]}
    stats = online.opd_backward(model, [far], pad_id=2, normaliser=1.0, cap=2.0, budget=100)
    assert stats["known_targets"] == 2 and stats["tis_clipped"] == 1
    # token 0: A = own - (own - 5) = +5; token 1 (no sampler value): A = -1 against the trainer's own; token 2: 0
    assert stats["advantage_abs_sum"] == pytest.approx(6.0, rel=1e-4) and stats["advantage_low"] == 0
    assert stats["reverse_kl_sum"] == pytest.approx(-5.0 + 1.0, rel=1e-4)


# ------------------------------------------------------------------------------------------------ the service
class ThreadEngine:
    """Records which thread made it and which thread each call runs on; the first call holds the engine a while."""

    def __init__(self) -> None:
        self.built_on, self.calls = threading.get_ident(), []

    def score(self, sequences):
        self.calls.append((threading.get_ident(), len(sequences), sequences[0][0]))
        if len(self.calls) == 1:
            time.sleep(0.4)
        return [[None] + [-1.0] * (len(s) - 1) for s in sequences]


def test_the_engine_thread_runs_every_call_and_a_queued_main_request_goes_first(tmp_path):
    results, threads = {}, []
    service, keep = None, threading.Event()
    keep.set()

    def engine_thread():  # builds the engine, then serves on the same thread, as run_service does
        nonlocal service
        service = tm.Service(ThreadEngine(), "T1", tmp_path, max_len=64)
        results["stopped_by"] = service.run(keep.is_set, poll=0.02)

    owner = threading.Thread(target=engine_thread)
    owner.start()
    while service is None:
        time.sleep(0.01)

    def post(name, priority, sequences, delay):
        time.sleep(delay)
        results[name] = service.score({"identity": "T1", "sequences": sequences, "priority": priority})

    for name, priority, first, n, delay in (("first", 0, 10, 2, 0.0), ("low", 1, 20, 40, 0.1), ("main", 0, 30, 3, 0.2)):
        threads.append(threading.Thread(target=post, args=(name, priority, [[first, 5, 6]] * n, delay)))
        threads[-1].start()
    for t in threads:
        t.join()
    keep.clear()
    owner.join(timeout=5)
    engine = service.engine
    assert {ident for ident, _, _ in engine.calls} == {engine.built_on}  # never on an HTTP-side thread
    # the main request queued after the low one goes first; the low one runs in pieces of LOW_PRIORITY_CHUNK
    assert [(n, first) for _, n, first in engine.calls] == [(2, 10), (3, 30), (32, 20), (8, 20)]
    assert len(results["low"]["logprobs"]) == 40 and results["low"]["wait_seconds"] > 0.2
    assert results["stopped_by"] == "stop"
    with pytest.raises(tm.ServiceClosed):  # nothing is queued once the service has ended
        service.score({"identity": "T1", "sequences": [[1, 2]], "priority": 0})


class FailingEngine:
    """Refuses (ValueError) a sequence starting with 99; fails (RuntimeError) on one starting with 66; returns NaN at
    position 1 of one starting with 77."""

    def score(self, sequences):
        if sequences[0][0] == 99:
            raise ValueError("prompt refused")
        if sequences[0][0] == 66:
            raise RuntimeError("CUDA error: an illegal memory access")
        return [[None] + [float("nan") if s[0] == 77 and p == 1 else -1.0 for p in range(1, len(s))]
                for s in sequences]


def test_a_refused_request_is_a_400_and_an_engine_failure_ends_the_service_with_503s(tmp_path):
    home = tmp_path / "teacher"
    home.mkdir()
    result, keep = {}, threading.Event()
    keep.set()
    thread = threading.Thread(target=lambda: result.update(tm.serve(FailingEngine(), identity="T1", home=home,
                                                                    max_len=64, keep_serving=keep.is_set, poll=0.05)))
    thread.start()
    try:
        deadline = time.time() + 10
        while not (home / "serving.json").exists() and time.time() < deadline:
            time.sleep(0.05)
        client = tm.TeacherClient(home, "T1", wait=0.5, poll=0.05)
        with pytest.raises(ValueError, match="refused"):
            client.score([[99, 1, 2]])
        assert client.score([[5, 1, 2]]) == [[None, -1.0, -1.0]]  # the engine goes on after a refused request
        assert client.score([[77, 1, 2], [5, 1, 2]]) == [[None, None, -1.0], [None, -1.0, -1.0]]  # NaN -> missing
        with pytest.raises(tm.TeacherUnavailable, match="in time"):  # 503, then nothing serves within the wait
            client.score([[66, 1, 2]])
    finally:
        keep.clear()
        thread.join(timeout=10)
    assert result["stopped_by"] == "engine_error" and "illegal memory access" in result["error"]
    assert result["errors"] == 2 and not (home / "serving.json").exists()
    assert [r["incomplete"] for r in recover_records(home / "requests.jsonl")] == [0, 1]


def test_the_scoring_engine_drops_only_the_dialogue_whose_ids_come_back_changed(monkeypatch):
    import types
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: kw
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)

    class LLM:
        def generate(self, prompts, params, use_tqdm):
            assert params["prompt_logprobs"] == 0 and params["max_tokens"] == 1 and not params["detokenize"]
            out = []
            for prompt in prompts:
                ids = list(prompt["prompt_token_ids"])
                echoed = ids if ids[0] != 13 else ids[:-1] + [ids[-1] + 1]  # this one comes back changed
                out.append(types.SimpleNamespace(prompt_token_ids=echoed, prompt_logprobs=[None] + [
                    {t: LP(-0.1 * p)} for p, t in enumerate(echoed) if p]))
            return out

    engine = object.__new__(tm.TeacherEngine)
    engine.llm = LLM()
    rows = engine.score([[5, 6, 7], [13, 6, 7], [8, 9]])
    assert rows == [[None, -0.1, -0.2], [None, None, None], [None, -0.1]]
    out, missing, _ = online.with_teacher(
        [{"usable": True, "input_ids": s, "positions": [2] if len(s) == 3 else [1], "old_logprobs": [-0.1]}
         for s in ([5, 6, 7], [13, 6, 7], [8, 9])], types.SimpleNamespace(score=lambda seqs: rows, last={}))
    assert out[0] is not None and out[1] is None and out[2] is not None and missing[1] == "teacher_missing"


class FakeTeacherEngine:
    """Deterministic scorer: log p of position p depends on the token and p. shift reads the neighbour position (the
    off-by-one the gate must catch)."""

    def __init__(self, spec=None, settings=None, shift: int = 0) -> None:
        self.shift, self.scored = shift, 0

    @staticmethod
    def value(seq, p):
        return -0.05 - 0.4 * ((seq[p] + p) % 13)  # neighbours differ by far more than the gate's 0.15 nat

    def score(self, sequences):
        self.scored += len(sequences)
        return [[None] + [self.value(s, min(p + self.shift, len(s) - 1)) for p in range(1, len(s))] for s in sequences]

    def sample(self, prompts, *, max_tokens, stop_ids):
        out = []
        for p in prompts:
            tokens = [20 + len(p) % 7, 31, 32, stop_ids[0]]
            seq = list(p) + tokens
            out.append({"tokens": tokens, "logprobs": [self.value(seq, len(p) + i) for i in range(len(tokens))]})
        return out


def test_the_gate_passes_a_consistent_teacher_and_fails_an_off_by_one():
    prompts = [list(range(30, 30 + n)) for n in (5, 9, 14)]
    ok = tm.gate(FakeTeacherEngine(), prompts, max_tokens=8, stop_ids=[2], max_gap=0.15)
    assert ok["passed"] and ok["mean_abs_gap"] == 0 and ok["sampled_tokens"] == 12 and ok["missing"] == 0
    bad = tm.gate(FakeTeacherEngine(shift=1), prompts, max_tokens=8, stop_ids=[2], max_gap=0.15)
    assert not bad["passed"] and bad["mean_abs_gap"] > 0.15
    silent = FakeTeacherEngine()
    silent.score = lambda sequences: [[None] * len(s) for s in sequences]  # every row came back unusable
    none = tm.gate(silent, prompts, max_tokens=8, stop_ids=[2], max_gap=0.15)
    assert not none["passed"] and none["mean_abs_gap"] is None and none["missing"] == 12
    json.dumps(none, allow_nan=False)  # gate.json can still be written, so the failure is recorded (exit 3)


def test_service_and_client_round_trip(tmp_path):
    home = tmp_path / "teacher"
    home.mkdir()
    engine, keep = FakeTeacherEngine(), threading.Event()
    keep.set()
    result = {}
    thread = threading.Thread(target=lambda: result.update(tm.serve(engine, identity="T1", home=home, max_len=64,
                                                                    keep_serving=keep.is_set, poll=0.05)))
    thread.start()
    try:
        deadline = time.time() + 10
        while not (home / "serving.json").exists() and time.time() < deadline:
            time.sleep(0.05)
        client = tm.TeacherClient(home, "T1", wait=5, caller="S:OPD_seed42", poll=0.05)
        seqs = [[3, 4, 5, 6], [7, 8, 9]]
        rows = client.score(seqs)
        assert rows == FakeTeacherEngine().score(seqs) and set(client.last) == {"score_seconds", "wait_seconds"}
        low = tm.TeacherClient(home, "T1", wait=5, priority=1, poll=0.05)
        assert low.score([list(range(3, 3 + n)) for n in range(2, 80, 2)][:31]) is not None  # chunked, same answer
        with pytest.raises(ValueError, match="refused"):
            client.score([[1] * 70])  # longer than the teacher's context
        with pytest.raises(tm.TeacherUnavailable):
            tm.TeacherClient(home, "other teacher", wait=0.2, poll=0.05).score(seqs)
        log = recover_records(home / "requests.jsonl")
        assert [r["priority"] for r in log] == [0, 1] and log[0]["caller"] == "S:OPD_seed42"
    finally:
        keep.clear()
        thread.join(timeout=10)
    assert result["stopped_by"] == "stop" and result["requests"] == 2 and not (home / "serving.json").exists()
    (home / "served.json").write_text("{}", encoding="utf-8")
    with pytest.raises(tm.TeacherUnavailable, match="stopped serving"):
        tm.TeacherClient(home, "T1", wait=5, poll=0.05).score([[1, 2]])


# ------------------------------------------------------------------------------------------------ fixture run
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
    manifest = {}
    for kind in ("student", "teacher"):
        folder = tmp_path / "models" / kind
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
        (folder / "tokenizer.json").write_text('{"shared": true}', encoding="utf-8")
        files = {n: {"size": (folder / n).stat().st_size, "sha256": digest(folder / n)}
                 for n in ("config.json", "tokenizer.json")}
        mark_verified(folder, files)
        manifest[MODELS[kind][0]] = {"revision": MODELS[kind][1], "files": files}
    (tmp_path / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
    files = manifest[MODELS["student"][0]]["files"]
    snapshot = expected_identity(model_spec("student", tmp_path / "models" / "student", DEFAULT), files)["local_snapshot"]
    imports = tmp_path / "imports"
    v08 = {"protocol": V08_PROTOCOL, "registration": {"source_identity": v05},
           "models": {"student": {"identity": {"local_snapshot": snapshot}}}}
    v08["identity"] = fingerprint(v08)
    (imports / "v08").mkdir(parents=True)
    (imports / "v08" / "manifest.json").write_text(json.dumps(v08), encoding="utf-8")
    for family in ("TR", "PTR"):
        for seed in (42, 43, 44):
            base = imports / "adapters" / f"{family}_seed{seed}"
            h = fake_adapter(base / "adapter", f"{family}{seed}")
            (base / "result.json").write_text(json.dumps({"training_completed": True, "adapter_reload_verified": True,
                                                          "identity": f"{family}{seed}", "adapter_hash": h}),
                                              encoding="utf-8")
            (base / "manifest.json").write_text(json.dumps({"identity": f"{family}{seed}", "run": v08["identity"],
                                                            "family": family, "seed": seed}), encoding="utf-8")
    for condition, right in (("R_seed42", {30, 31}), (TEACHER, {30, 31, 32})):  # v0.8's own validation outcomes
        rows = [{"task_id": t.id, "db_id": t.db_id, "correct": int(t.id.split(":")[1]) in right,
                 "status": "correct" if int(t.id.split(":")[1]) in right else "incorrect"} for t in roles["validation"]]
        path = imports / "v08" / "eval-rows" / "validation" / f"{condition}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"run": v08["identity"], "role": "validation", "condition": condition,
                                    "episodes_sha256": "e" * 64, "rows": rows}), encoding="utf-8")
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, smoke_settings(), models_root=tmp_path / "models",
                    manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100, v08_run=v08["identity"])
    return run, frozen


def test_freeze_verifies_the_arms_the_teacher_and_the_references(fixture_run, tmp_path):
    run, frozen = fixture_run
    assert frozen["protocol"] == PROTOCOL and frozen["counts"] == {"pool": 8, "dev": 4, "validation": 4, "test": 4}
    assert set(frozen["imports"]["adapters"]) == {"TR_seed42", "PTR_seed42"}  # the smoke settings' one seed
    assert set(frozen["imports"]["references"]) == {"validation/R_seed42", f"validation/{TEACHER}"}
    assert set(frozen["models"]) == {"student", "teacher"} and load(run)[0]["identity"] == frozen["identity"]
    reg, raw, imports = tmp_path / "registration", tmp_path / "raw", tmp_path / "imports"
    kw = dict(models_root=tmp_path / "models", manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100)
    with pytest.raises(ValueError, match="registered v0.8 run"):
        freeze(reg, raw, tmp_path / "other", smoke_settings(), **kw)  # the real v0.8 identity is not this fixture's
    (imports / "adapters" / "TR_seed42" / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter files changed"):
        freeze(reg, raw, tmp_path / "other2", smoke_settings(), **kw, v08_run=frozen["imports"]["v08_run"])


# ------------------------------------------------------------------------------------------------ OPD runs
def loader(holder: dict, starts: list):
    load = tiny_policy(holder, checkpointing=True)

    def make(spec, start, *, settings, seed):
        starts.append(start)
        return load(spec, start)
    return make


def engines(holder: dict):
    made: list[ScriptedEngine] = []

    def make(spec, settings, *, seed):
        made.append(ScriptedEngine(holder, seed))
        return made[-1]
    return make, made


def test_an_opd_run_pauses_resumes_and_records_the_teacher(fixture_run):
    """Fresh LoRA, a fake teacher: two steps, a pause, the third step from the checkpoint in a new call."""
    pytest.importorskip("torch")
    run, frozen = fixture_run
    holder, starts = {}, []
    make, made = engines(holder)
    teacher = PositionTeacher()
    out = run / "train" / "OPD_seed42"
    first = online.train_online(run, family=FAMILY, seed=42, until=None, teacher=teacher, stop_step=2,
                                engine_factory=make, policy_loader=loader(holder, starts))
    assert first["finished"] is False and first["stopped_by"] == "paused" and first["steps_completed"] == 2
    assert first["lora_check"]["step"] == 1 and first["lora_check"]["passed"]  # smoke: lora_check_step = 1
    assert starts == [None] and online.ended(out) is None
    final = online.train_online(run, family=FAMILY, seed=42, until=None, teacher=teacher, engine_factory=make,
                                policy_loader=loader(holder, starts))
    assert final["finished"] and final["steps_completed"] == 3 and final["stopped_by"] == "planned"
    assert set(final["milestones"]) == {"1", "2", "3"} and made[-1].versions[0] == 2  # resumed from the checkpoint
    metrics = recover_records(out / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [1, 2, 3]
    for m in metrics:
        assert m["teacher_sequences"] == 16 and m["trained_trajectories"] == 16 and m["optimizer_updated"]
        assert m["teacher_wait_seconds"] == 0.25 and m["reverse_kl"] != 0 and 0 <= m["advantage_low_fraction"] <= 1
        assert m["behaviour"]["dialogues"] == 16 and set(m["tasks"]) <= {t.id for t in load(run, "pool")[1]}
    assert len(teacher.calls) == 3 and all(len(c) == 16 for c in teacher.calls)
    rows = recover_records(out / "rollouts" / "step-0000.jsonl")
    assert all(r["trained"] and r["teacher_kl"] is not None and "reward" in r for r in rows)
    manifest = read_json(out / "manifest.json")
    assert manifest["plan"]["method"] == "opd" and manifest["plan"]["teacher_priority"] == 0
    assert manifest["teacher"] == frozen["models"]["teacher"]["identity"]
    assert online.train_online(run, family=FAMILY, seed=42, until=None, teacher=teacher, engine_factory=make,
                               policy_loader=loader(holder, starts)) == final  # an ended run never trains again
    with pytest.raises(ValueError, match="Unregistered"):
        online.train_online(run, family=FAMILY, seed=43, until=None, teacher=teacher, engine_factory=make,
                            policy_loader=loader(holder, starts))


def test_tropd_starts_from_tr_at_low_priority_and_a_past_deadline_ends_a_run(fixture_run):
    pytest.importorskip("torch")
    run, frozen = fixture_run
    holder, starts = {}, []
    make, _ = engines(holder)
    other = run.parent / "late"
    shutil.copytree(run, other)
    result = online.train_online(run, family=EXPLORE, seed=42, until=None, teacher=PositionTeacher(),
                                 engine_factory=make, policy_loader=loader(holder, starts))
    assert result["finished"] and starts == [run / "imports" / "adapters" / "TR_seed42" / "adapter"]
    assert read_json(run / "train" / "TROPD_seed42" / "manifest.json")["plan"]["teacher_priority"] == 1
    client = online.teacher_client(run.parent / "teacher", frozen, EXPLORE, 42, "M")
    assert client.priority == 1 and client.caller == "M:TROPD_seed42"
    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)).isoformat()
    late = online.train_online(other, family=FAMILY, seed=42, until=soon, teacher=PositionTeacher(),
                               engine_factory=make, policy_loader=loader(holder, starts))
    assert late["finished"] and late["stopped_by"] == "deadline" and late["steps_completed"] == 0
    with pytest.raises(ValueError, match="Only smoke"):
        study = run.parent / "study"
        shutil.copytree(run, study)
        manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("identity")
        manifest["settings"]["phase"] = "study"
        manifest["identity"] = fingerprint(manifest)
        (study / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        online.train_online(study, family=FAMILY, seed=42, until=None, teacher=PositionTeacher(), stop_step=2,
                            engine_factory=make, policy_loader=loader(holder, starts))


class GoneTeacher:
    """Raises as the client does when no answer comes in time; `gone` says whether a service will ever answer again."""

    def __init__(self, gone: bool) -> None:
        self._gone, self.last = gone, {}

    def gone(self) -> bool:
        return self._gone

    def score(self, sequences):
        raise tm.TeacherUnavailable("No teacher answer in time")


def test_a_run_ends_cleanly_once_no_teacher_will_answer_and_fails_the_attempt_while_one_may(fixture_run):
    pytest.importorskip("torch")
    run, _ = fixture_run
    holder, starts = {}, []
    make, _ = engines(holder)
    out = run / "train" / "OPD_seed42"
    with pytest.raises(tm.TeacherUnavailable):  # only away: this attempt fails, the scheduler retries it
        online.train_online(run, family=FAMILY, seed=42, until=None, teacher=GoneTeacher(False), engine_factory=make,
                            policy_loader=loader(holder, starts))
    assert online.ended(out) is None
    result = online.train_online(run, family=FAMILY, seed=42, until=None, teacher=GoneTeacher(True),
                                 engine_factory=make, policy_loader=loader(holder, starts))
    assert result["finished"] and result["stopped_by"] == "teacher_gone" and result["steps_completed"] == 0
    assert online.ended(out)["stopped_by"] == "teacher_gone"


# ------------------------------------------------------------------------------------------------ evaluation, report
def metric(step: int) -> dict:
    return {"step": step, "optimizer_updates": step, "reward_mean": 0.5, "trained_trajectories": 16, "entropy": 0.4,
            "mean_turns": 2.0, "mean_completion_tokens": 30.0, "mean_abs_logp_gap": 0.001, "max_abs_logp_gap": 0.01,
            "tis_clipped_fraction": 0.0, "grad_norm": 0.1, "step_seconds": 90.0, "generate_seconds": 30.0,
            "teacher_seconds": 40.0, "teacher_wait_seconds": 5.0, "teacher_score_seconds": 30.0, "train_seconds": 20.0,
            "active_seconds": 90.0 * step, "tasks": [], "reverse_kl": 0.5 / step, "teacher_logp": -1.0,
            "sampler_logp": -0.8, "advantage_low_fraction": 0.01, "pass": 0,
            "behaviour": {"dialogues": 16, "unparsed": 0, "no_query": 1, "repeat": 0}}


def fake_train(run: Path, family: str, seed: int, steps: int, *, pause: bool = False) -> None:
    """What a run leaves behind (milestone adapters, records, paused/result), without training."""
    frozen, _ = load(run, check_code=False)
    out = run / "train" / f"{family}_seed{seed}"
    marks = read_json(out / "milestones.json") if (out / "milestones.json").exists() else {}
    for step in range(len(marks) + 1, steps + 1):
        marks[str(step)] = fake_adapter(out / "milestones" / f"step-{step:04d}" / "adapter", f"{family}{seed}-{step}")
    (out / "milestones.json").write_text(json.dumps(marks), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"run": frozen["identity"], "identity": "x"}), encoding="utf-8")
    (out / "metrics.jsonl").write_text("".join(json.dumps(metric(i)) + "\n" for i in range(1, steps + 1)),
                                       encoding="utf-8")
    record = {"steps_completed": steps, "last_milestone": steps, "optimizer_updates": steps,
              "lora_check": {"passed": True, "verdict": "closer_to_lora", "step": 1}}
    if pause:
        (out / "paused.json").write_text(json.dumps({**record, "step": steps, "stopped_by": "paused"}), encoding="utf-8")
    else:
        (out / "result.json").write_text(json.dumps({**record, "finished": True, "stopped_by": "planned"}),
                                         encoding="utf-8")


def teacher_factory(spec, settings, **_):
    return Scripted()


def test_k_rule_tropd_at_min_k_and_the_conditions_adapters(fixture_run):
    run, frozen = fixture_run
    fake_train(run, FAMILY, 42, 3)
    fake_train(run, EXPLORE, 42, 2)
    chosen = common_steps(run)
    assert chosen["k"] == 3 and chosen["evaluate"]["OPD_seed42"]["step"] == 3
    assert chosen["evaluate"]["TROPD_seed42"]["step"] == 2 and not chosen["short"]
    assert adapter_for(run, frozen, "OPD_seed42")[2] == 3 and adapter_for(run, frozen, "OPD_seed42@1")[2] == 1
    folder, h, step = adapter_for(run, frozen, "TR_seed42")
    assert folder.parent.name == "TR_seed42" and step is None and h == frozen["imports"]["adapters"]["TR_seed42"]
    with pytest.raises(ValueError, match="Unregistered seed"):
        adapter_for(run, frozen, "TR_seed43")
    assert adapter_for(run, frozen, "B") == (None, None, None) == adapter_for(run, frozen, TEACHER)
    with pytest.raises(ValueError, match="evaluate_teacher"):
        evaluate(run, condition=TEACHER, role="dev", stop=None, factory=factory)


def test_evaluate_the_teacher_and_report_end_to_end(fixture_run):
    from exec_feedback.v12.analysis import report

    run, frozen = fixture_run
    summaries = evaluate_teacher([(run, "dev"), (run, "test"), (run, "validation")], stop=None, factory=teacher_factory)
    assert [s["role"] for s in summaries] == ["dev", "test", "validation"] and all(s["complete"] for s in summaries)
    assert all(s["engine_shared"] for s in summaries) and summaries[1]["generator"]["calls"] > 0
    assert evaluate_teacher([(run, "dev")], stop=None, factory=lambda *a, **k: 1 / 0)[0]["complete"]  # never twice
    fake_train(run, FAMILY, 42, 3)
    fake_train(run, EXPLORE, 42, 2)
    common_steps(run)
    for role in ("dev", "validation", "test"):
        for condition in ("B", "TR_seed42", "PTR_seed42", "OPD_seed42", "TROPD_seed42"):
            assert evaluate(run, condition=condition, role=role, stop=None, factory=factory)["complete"]
    home = run.parent / "teacher"
    home.mkdir()
    (home / "gate.json").write_text(json.dumps({"passed": True, "sequences": 32, "sampled_tokens": 900, "missing": 0,
                                                "mean_abs_gap": 0.004, "max_abs_gap": 0.05,
                                                "gpu_used_mib": [76000, 76100, 76100, 76000]}), encoding="utf-8")
    (home / "requests.jsonl").write_text("".join(json.dumps({"priority": p, "sequences": 128, "tokens": 400000,
                                                             "score_seconds": 40.0, "wait_seconds": w,
                                                             "gpu_used_mib": m}) + "\n"
                                                 for p, w, m in ((0, 1.0, [80000, 79000]), (0, 2.0, None),
                                                                 (1, 30.0, [81920, 80000]))), encoding="utf-8")
    result = report(run, procs=1)
    for label in (f"{FAMILY}-TR", f"{FAMILY}-B"):
        assert result["decisions"][label]["decided"] and result["decisions"][label]["seeds"] == 1
    deltas = result["roles"]["test"]["project"]["deltas"]
    assert {"OPD-TR", "OPD-B", "OPD-PTR", "OPD-B235", "TROPD-TR", "TROPD-OPD"} <= set(deltas)
    assert deltas["OPD-TR"]["mean_delta"] == 0  # the scripted generator answers the same for every adapter
    refs = result["references"]["validation"]
    assert {"R_seed42", "B235v08"} <= set(refs["conditions"]) and {"OPD-R", "OPD-B235v08", "TR-R"} <= set(refs["deltas"])
    assert result["dynamics"]["OPD_seed42"]["reverse_kl_first10"] == pytest.approx((0.5 + 0.25 + 0.5 / 3) / 3)
    assert result["teacher_service"]["by_priority"]["1"]["wait_seconds_median"] == 30.0
    assert result["teacher_service"]["gpu_used_mib_max"] == 81920
    assert result["teacher_service"]["gpu_used_mib_after_gate"] == 76100
    md = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert "主比较 OPD-TR" in md and "隔晚参照" in md and "老师打分服务" in md and "TROPD_seed42" in md
    assert "打分期间最高 80.0 GiB" in md


# ------------------------------------------------------------------------------------------------ scheduler
FAKE_CLI = r'''
import sys, time
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_v12 as t
from exec_feedback.v12 import analysis
from exec_feedback.v12.evaluate import common_steps, evaluate, evaluate_teacher
from exec_feedback.v12.freeze import load
from exec_feedback.v12.online import teacher_client
from exec_feedback.v12.teacher import run_service

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
if cmd == "teacher-eval":
    jobs = [(Path(args[i + 1]), args[i + 2]) for i, a in enumerate(args) if a == "--job"]
    evaluate_teacher(jobs, stop=opt("--stop"), factory=t.teacher_factory)
    raise SystemExit(0)
run = Path(opt("--run"))
if (run / f"FAIL_{{cmd}}").exists():  # a path made to fail in this run
    raise SystemExit(1)
if cmd == "serve-teacher":
    shift = 1 if (run / "BAD_TEACHER").exists() else 0
    result = run_service(run, home=Path(opt("--home")), until=opt("--until"),
                         engine_factory=lambda spec, s: t.FakeTeacherEngine(shift=shift), tokenizer=t.ChatTokenizer(),
                         poll=0.1)
    raise SystemExit(0 if result["served"] else 3)
elif cmd == "train":
    family, seed = opt("--family"), int(opt("--seed"))
    frozen = load(run, check_code=False)[0]
    client = teacher_client(Path(opt("--teacher-home")), frozen, family, seed, opt("--tag"))
    assert client.score([[1, 2, 3]])[0][0] is None  # the service answers this run
    stop = opt("--stop-step")
    t.fake_train(run, family, seed, int(stop) if stop else frozen["settings"]["planned_steps"], pause=bool(stop))
elif cmd == "steps":
    common_steps(run)
elif cmd == "evaluate":
    evaluate(run, condition=opt("--condition"), role=opt("--role"), stop=opt("--stop"), factory=t.factory)
elif cmd == "report":
    analysis.report(run, procs=1, teacher_home=Path(opt("--teacher-home")))
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


def schedule(smoke: Path, main: Path, frozen: dict, tmp_path: Path, stop: float):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v12.orchestrate import Plan

    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    gpus = list(range(8))
    plan = Plan(run=main, smoke_run=smoke, teacher_home=tmp_path / "teacher", models_root=Path(frozen["models_root"]),
                main_py=sys.executable, vllm_py=sys.executable, stop_epoch=stop, gpus=gpus)
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=gpus, stop=stop, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs", plan=plan,
                          poll=0.1, stagger=0, retry_delay=0, milestone=reached.append)
    plan.scheduler = scheduler
    return plan, scheduler, reached


def starts(status: str) -> list[tuple[str, str]]:
    return re.findall(r"start (\S+) on (GPUs? \d+(?:,\d+)*)", status)


def test_scheduler_runs_the_teacher_the_smoke_and_the_main_plan_on_eight_gpus(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    plan, scheduler, reached = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status and " exit 3" not in status, status
    for run in (smoke, main):
        assert (run / "report" / "report.json").exists()
    for role in ("dev", "validation", "test"):
        for condition in ("B", "TR_seed42", "PTR_seed42", "OPD_seed42", "TROPD_seed42"):
            assert (main / "eval" / role / condition / "summary.json").exists(), (role, condition)
    for run, role in ((smoke, "dev"), (main, "dev"), (main, "test")):
        assert (run / "eval" / role / TEACHER / "summary.json").exists()
    first = lambda text: status.index(text)
    where = dict((name, gpu) for name, gpu in reversed(starts(status)))  # the first start of each job
    # the teacher's block: its evaluation, then the service; the runs on their own GPUs
    assert where["T:teacher-eval"] == "GPUs 0,1,2,3" == where["T:serve"]
    assert first("start T:teacher-eval") < first("start T:serve") < first("start S:train-OPD_seed42-pause")
    assert where["S:train-OPD_seed42-pause"] == "GPU 4" == where["M:train-OPD_seed42"]
    assert where["S:train-TROPD_seed42-pause"] == "GPU 5" == where["M:train-TROPD_seed42"]
    # the smoke run paused and resumed before the main runs; TROPD after the arms' evaluations
    assert first("start S:train-OPD_seed42 ") < first("start M:train-OPD_seed42")
    arms = [m.start() for m in re.finditer(r"start M:eval-(B|TR_seed42|PTR_seed42)-", status)]
    assert max(arms) < first("start M:train-TROPD_seed42")
    # no evaluation on the smoke OPD's GPU before the service, none on the teacher's block while it serves
    serving = first("start T:serve")
    for m in re.finditer(r"start ([SM]:eval-\S+) on GPU (\d)", status):
        assert not (m.start() < serving and m.group(2) == "4"), m.group(0)
    stopped = status.index("the service is told to stop")
    for m in re.finditer(r"start ([SM]:eval-\S+) on GPU (\d)", status):
        if m.group(2) in "0123":
            assert m.start() > stopped, m.group(0)
    assert first("start M:steps") > stopped and (tmp_path / "teacher" / "served.json").exists()
    assert read_json(tmp_path / "teacher" / "gate.json")["passed"] and status.count("teacher scoring gate passed") == 1
    assert read_json(main / "report" / "report.json")["decisions"][f"{FAMILY}-TR"]["decided"]
    assert {"T:teacher-eval", "T:serve", "S:train", "S:report", "M:train", "M:steps", "M:eval-k", "M:report"} <= \
        set(reached)


def test_a_failed_teacher_gate_settles_every_run_and_the_evaluations_still_report(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    (main / "BAD_TEACHER").write_text("", encoding="utf-8")  # the service loads the main run: an off-by-one teacher
    plan, scheduler, _ = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600)
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert "start S:train" not in status and "start M:train" not in status
    assert not read_json(tmp_path / "teacher" / "gate.json")["passed"] and status.count("start T:serve") == 1
    assert "teacher scoring gate FAILED" in status
    assert (main / "report" / "report.json").exists() and (main / "eval" / "test" / "TR_seed42" / "summary.json").exists()
    assert not read_json(main / "report" / "report.json")["decisions"][f"{FAMILY}-TR"]["decided"]


def test_a_service_that_died_after_the_runs_settled_is_not_started_again(fixture_run, tmp_path):
    smoke, frozen = fixture_run
    main = main_copy(smoke, tmp_path)
    plan, scheduler, _ = schedule(smoke, main, frozen, tmp_path, time.time() + 9 * 3600)
    serve = lambda: next(j for j in plan()[0] if j.name == "T:serve")
    scheduler.failures["T:serve"] = 1  # it died once (an engine error): no served.json, no stop file, retries left
    assert not serve().done()  # the runs still need it: it is restarted
    for tag, run in (("S", smoke), ("M", main)):
        for family, seed in plan.members(tag):
            fake_train(run, family, seed, 3)
    assert serve().done() and not (tmp_path / "teacher" / "stop").exists()
    assert not (tmp_path / "teacher" / "over").exists()
    scheduler.failures["T:serve"] = 4  # its last retry failed too: clients stop waiting
    plan()
    assert (tmp_path / "teacher" / "over").exists() and "failed for the last time" in \
        (tmp_path / "STATUS.txt").read_text(encoding="utf-8")


def test_the_registered_layout_on_eight_gpus(fixture_run, tmp_path):
    from exec_feedback.v12.orchestrate import Plan

    smoke, _ = fixture_run
    main = tmp_path / "layout"
    main.mkdir()
    (main / "manifest.json").write_text(json.dumps({"settings": DEFAULT}), encoding="utf-8")
    make = lambda n: Plan(run=main, smoke_run=smoke, teacher_home=tmp_path / "teacher", models_root=tmp_path,
                          main_py="python", vllm_py="python", stop_epoch=time.time() + 3600, gpus=list(range(n)))
    eight = make(8)
    assert eight.block == [0, 1, 2, 3] and eight.rest == [4, 5, 6, 7]
    assert [eight.gpu_for(FAMILY, s) for s in (42, 43, 44)] == [4, 5, 6] and eight.gpu_for(EXPLORE, 42) == 7
    assert make(7).gpu_for(EXPLORE, 42) is None  # no spare GPU: TROPD is not planned
    with pytest.raises(ValueError, match="more than 4 GPUs"):
        make(4)


class FlakyTeacher(FakeTeacherEngine):
    """Passes the gate, then fails on the first request it serves."""

    def score(self, sequences):
        if self.scored:
            raise RuntimeError("engine died")
        return super().score(sequences)


def test_a_restart_after_an_engine_error_keeps_the_passed_gate(fixture_run, tmp_path):
    run, frozen = fixture_run
    home = tmp_path / "teacher"
    result = {}
    thread = threading.Thread(target=lambda: result.update(tm.run_service(
        run, home=home, until=None, engine_factory=lambda spec, s: FlakyTeacher(), tokenizer=ChatTokenizer(),
        poll=0.05)))
    thread.start()
    deadline = time.time() + 30
    while not (home / "serving.json").exists() and time.time() < deadline:
        time.sleep(0.05)
    with pytest.raises(tm.TeacherUnavailable):
        tm.TeacherClient(home, tm.identity_of(frozen), wait=0.5, poll=0.05).score([[5, 6, 7]])
    thread.join(timeout=10)
    assert result["stopped_by"] == "engine_error" and not (home / "served.json").exists()
    first = read_json(home / "gate.json")
    assert first["passed"]
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
    # an off-by-one engine would fail a fresh gate: the passed one is kept, and the past deadline ends the service
    again = tm.run_service(run, home=home, until=past, engine_factory=lambda spec, s: FakeTeacherEngine(shift=1),
                           tokenizer=ChatTokenizer(), poll=0.05)
    assert again["served"] and again["stopped_by"] == "deadline" and read_json(home / "gate.json") == first
    attempts = recover_records(home / "gate-attempts.jsonl")
    assert len(attempts) == 2 and attempts[1]["kept"] == "passed before"


def test_tropd_waits_for_its_turn_up_to_its_deadline_and_no_run_waits_once_the_service_is_over(fixture_run, tmp_path):
    _, frozen = fixture_run
    home = tmp_path / "teacher"
    home.mkdir()
    until = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2)).isoformat()
    tropd = online.teacher_client(home, frozen, EXPLORE, 42, "M", until=until)
    assert tropd.priority == 1 and tropd.wait is None and tropd.until > time.monotonic() + 7000
    main = online.teacher_client(home, frozen, FAMILY, 42, "M", until=until)
    assert main.priority == 0 and main.wait == frozen["settings"]["teacher_wait_seconds"] and main.until is not None
    assert online.teacher_client(home, frozen, EXPLORE, 42, "S").wait == frozen["settings"]["teacher_wait_seconds"]
    started = time.monotonic()
    with pytest.raises(tm.TeacherUnavailable, match="in time"):
        tm.TeacherClient(home, "T1", wait=None, priority=1, poll=0.05, until=started + 1.0).score([[1, 2]])
    assert time.monotonic() - started >= 1.0  # the low-priority run waits up to its deadline
    started = time.monotonic()
    with pytest.raises(tm.TeacherUnavailable, match="in time"):
        tm.TeacherClient(home, "T1", wait=30, poll=0.05, until=started + 0.5).score([[1, 2]])
    assert time.monotonic() - started < 5  # and no run waits past its own deadline
    assert not tm.TeacherClient(home, "T1", wait=30).gone()
    (home / "over").write_text("", encoding="utf-8")
    assert tm.TeacherClient(home, "T1", wait=30).gone()
    started = time.monotonic()
    with pytest.raises(tm.TeacherUnavailable, match="last time"):
        tm.TeacherClient(home, "T1", wait=30, poll=0.05).score([[1, 2]])
    assert time.monotonic() - started < 1.0


def test_cli_error_exits_with_code_one(tmp_path):
    done = subprocess.run([sys.executable, "-m", "exec_feedback.v12", "status", "--run", str(tmp_path / "missing")],
                          cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == 1 and "Traceback" in done.stderr
