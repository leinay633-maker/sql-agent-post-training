"""v0.9 engineering tests: SQLite fixtures, a character tokenizer, a scripted engine and a randomly initialised tiny
Qwen3 with a LoRA; no GPU, vLLM or real weights. Nothing here is a model result."""
from __future__ import annotations

import dataclasses
import json
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from exec_feedback.core import Task, digest, write_jsonl
from exec_feedback.storage import fingerprint, read_json, recover_records
from exec_feedback.v8.freeze import expected_identity, mark_verified, model_spec
from exec_feedback.v7.freeze import _adapter_hash
from exec_feedback.v9 import MODELS, V08_PROTOCOL
from exec_feedback.v9 import online
from exec_feedback.v9.config import DEFAULT, validate_settings
from exec_feedback.v9.evaluate import adapter_for, common_steps, evaluate, merged, parse
from exec_feedback.v9.freeze import freeze, load

REPO = Path(__file__).resolve().parents[1]
GOLD = "SELECT sum(v) FROM items"
WRONG = "SELECT max(v) FROM items"
CHATML = ("{%- for message in messages %}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
          "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(dict(DEFAULT))
    assert json.loads((REPO / "configs" / "h20-v09.json").read_text(encoding="utf-8")) == DEFAULT
    smoke = json.loads((REPO / "configs" / "v09-smoke.json").read_text(encoding="utf-8"))
    validate_settings(smoke)
    assert smoke["phase"] == "smoke" and smoke["planned_steps"] == 2
    models = json.loads((REPO / "configs" / "v09-models.json").read_text(encoding="utf-8"))
    assert set(models) == {MODELS["student"][0]} and models[MODELS["student"][0]]["revision"] == MODELS["student"][1]
    for bad in ({"rl_slots": 64}, {"group_size": 1}, {"curve_every": 15}, {"hi_seeds": [45]}, {"max_steps": 6}):
        with pytest.raises(ValueError):
            validate_settings({**DEFAULT, **bad})


def test_condition_names():
    assert parse("B") == (None, None, None)
    assert parse("PTR_seed43") == ("PTR", 43, None)
    assert parse("RL_seed42") == ("RL", 42, None) and parse("RAFT_seed44@60") == ("RAFT", 44, 60)
    assert parse("RLhi_seed42") == ("RLhi", 42, None)
    for bad in ("PTR_seed42@10", "TR_seed42", "RL", "B235", "RL_seed42@"):
        with pytest.raises(ValueError):
            parse(bad)


# ------------------------------------------------------------------------------------------------ objective
def test_rloo_and_raft_advantages():
    assert online.advantages([1, 0, 0, 0], 4, "rloo") == pytest.approx([1, -1 / 3, -1 / 3, -1 / 3])
    assert online.advantages([1, 1, 1, 1, 0, 0, 0, 0], 4, "rloo") == [0.0] * 8
    assert online.advantages([1, 0, 1, 0], 2, "raft") == [1.0, 0.0, 1.0, 0.0]
    assert sum(online.advantages([1, 1, 0, 1, 0, 0], 3, "rloo")) == pytest.approx(0)
    for rewards, group in (([1, 0, 1], 2), ([0.5, 0], 2), ([1, 0], 1)):
        with pytest.raises(ValueError):
            online.advantages(rewards, group, "rloo")
    counts = online.group_counts([1, 1, 0, 0, 1, 0, 1, 1], [True] * 6 + [False, True], 2)
    assert counts == {"all_correct_groups": 1, "all_failed_groups": 1, "informative_groups": 1, "invalid_groups": 1}


def turn(prompt, tokens, stop=2, logprobs=None):
    return {"trace": {"prompt_ids": prompt, "tokens": tokens, "stop": stop,
                      "logprobs": logprobs if logprobs is not None else [-0.1 * (i + 1) for i in range(len(tokens) + (stop is not None))]}}


def test_trajectory_is_one_sequence_with_every_sampled_token():
    p1 = [5, 6, 7]
    t1 = [11, 12]
    p2 = p1 + t1 + [2, 9, 9, 8]  # the template re-rendered turn 1 (with its end token) and added the tool result
    t2 = [13, 14, 15]
    ex = online.trajectory([turn(p1, t1), turn(p2, t2)])
    assert ex["usable"] and ex["input_ids"] == p2 + t2 + [2]
    assert ex["positions"] == [3, 4, 5, 9, 10, 11, 12]  # turn-1 content + its end token, turn-2 content + end token
    assert [ex["input_ids"][p] for p in ex["positions"]] == [11, 12, 2, 13, 14, 15, 2]
    assert ex["old_logprobs"] == pytest.approx([-0.1, -0.2, -0.3, -0.1, -0.2, -0.3, -0.4])
    # a turn whose text re-tokenized differently is kept out of the gradient
    assert online.trajectory([turn(p1, t1), turn(p1 + [11, 99, 2, 9], t2)])["reason"] == "retokenized"
    # a length-stopped turn: no end token sampled, the template's end token is context only
    ex = online.trajectory([turn(p1, t1, stop=None), turn(p1 + t1 + [2, 9], t2)])
    assert [ex["input_ids"][p] for p in ex["positions"]] == [11, 12, 13, 14, 15, 2]
    # an end token other than the template's (e.g. <|endoftext|>) is trained only where it really ended the dialogue
    ex = online.trajectory([turn(p1, t1, stop=3), turn(p1 + t1 + [2, 9], t2, stop=3)])
    assert [ex["input_ids"][p] for p in ex["positions"]] == [11, 12, 13, 14, 15, 3]
    assert online.trajectory([]) is None
    assert online.trajectory([{"output": "x"}])["reason"] == "missing_trace"


def test_micro_batches_respect_the_padded_budget():
    items = [{"input_ids": [0] * n} for n in (5, 50, 7, 30, 6)]
    batches = online.micro_batches(items, 60)
    assert [[len(x["input_ids"]) for x in b] for b in batches] == [[5, 6, 7], [30], [50]]
    assert online.micro_batches([{"input_ids": [0] * 100}], 60) == [[{"input_ids": [0] * 100}]]


# ------------------------------------------------------------------------------------------------ tokenizer, engine
class ChatTokenizer:
    """ChatML through jinja2 (as transformers renders it); ids are shifted code points, <|im_end|> = 2."""
    eos_token_id = pad_token_id = 2

    def __init__(self, template: str = CHATML) -> None:
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

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join("" if i == 2 and skip_special_tokens else "<|im_end|>" if i == 2 else chr(i - 10) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self.template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        return self.encode(text) if tokenize else text


class ScriptedEngine:
    """Queries first, then finishes: greedy answers even question numbers right; sampling is a coin flip. Log-probs
    come from the live trainer model (holder['model']) when one is attached, else a constant."""

    def __init__(self, holder: dict | None = None, seed: int = 0) -> None:
        self.holder = holder if holder is not None else {}  # shared: the trainer attaches its model later
        self.rng, self.versions, self.tok = random.Random(seed), [], ChatTokenizer()

    def set_policy(self, path, version):
        assert (Path(path) / "adapter_model.safetensors").exists() or self.holder.get("allow_missing")
        self.versions.append(version)

    def action(self, ids: list[int], temperature: float) -> dict:
        text = self.tok.decode(ids)
        if text.count("<|im_start|>assistant") == 1:
            return {"action": "query", "sql": "SELECT v FROM items"}
        n = int(re.search(r"Question (\d+)", text).group(1))
        right = self.rng.random() < 0.5 if temperature > 0 else n % 2 == 0
        return {"action": "finish", "sql": GOLD if right else WRONG}

    def generate(self, prompts, *, temperature, max_tokens, stop_ids, logprobs):
        out = []
        for ids in prompts:
            tokens = self.tok.encode(json.dumps(self.action(ids, temperature))) + [2]
            out.append({"tokens": tokens, "finish_reason": "stop", "stop_token": 2, "cached_tokens": 0,
                        "logprobs": self.logprobs(ids, tokens) if logprobs else None})
        return out

    def logprobs(self, prompt: list[int], tokens: list[int]) -> list[float]:
        model = self.holder.get("model")
        if model is None:
            return [-0.5] * len(tokens)
        import torch
        with torch.no_grad():
            ids = torch.tensor([prompt + tokens])
            logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1:-1].float()
            return torch.log_softmax(logits, -1).gather(1, torch.tensor(tokens)[:, None])[:, 0].tolist()


def test_tracing_generator_traces_and_rejects_overlong_prompts():
    tok = ChatTokenizer()
    engine = ScriptedEngine()
    gen = online.TracingGenerator(engine, tok, max_context=600, max_tokens=64, temperature=1.0, stop_ids=[2], trace=True)
    short = [{"role": "system", "content": "S"}, {"role": "user", "content": "Question 4?"}]
    long = [{"role": "system", "content": "S" * 600}, {"role": "user", "content": "Question 4?"}]
    first, rejected = gen.generate([short, long])
    assert isinstance(rejected, ValueError) and gen.metadata["context_rejections"] == 1
    assert json.loads(first["text"]) == {"action": "query", "sql": "SELECT v FROM items"}
    trace = first["trace"]
    assert trace["prompt_ids"] == tok.apply_chat_template(short, add_generation_prompt=True)
    assert trace["stop"] == 2 and tok.decode(trace["tokens"]) == first["text"] and len(trace["logprobs"]) == len(trace["tokens"]) + 1
    greedy = online.TracingGenerator(engine, tok, max_context=600, max_tokens=64, temperature=0.0, stop_ids=[2],
                                     trace=False)
    assert "trace" not in greedy.generate([short])[0]


def test_sampled_logprobs_reads_the_sampled_token():
    class LP:
        def __init__(self, v):
            self.logprob = v
    rows = [{5: LP(-0.5), 6: LP(-1.0)}, {7: LP(-2.0)}]
    assert online.sampled_logprobs(rows, [5, 7]) == [-0.5, -2.0]
    assert online.sampled_logprobs(rows, [6, 8]) == [-1.0, None]


# ------------------------------------------------------------------------------------------------ tiny model
def tiny_policy(holder: dict | None = None, checkpointing: bool = False):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    def load(spec=None, start=None):
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=8192,
                             tie_word_embeddings=False)
        base = Qwen3ForCausalLM(config)
        if checkpointing:  # as models.load_model(training=True) prepares the 4B
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
            base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.05, target_modules=["q_proj", "v_proj", "o_proj"],
            init_lora_weights=False))
        if holder is not None:
            holder["model"] = model
        return model, ChatTokenizer()
    return load


def test_batched_logprobs_equal_single_sequences_and_a_step_follows_the_advantage():
    torch = pytest.importorskip("torch")
    model, _ = tiny_policy()()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    a = {"input_ids": [20, 21, 22, 23, 24, 25], "positions": [3, 4, 5], "old_logprobs": [None] * 3, "weight": 1.0}
    b = {"input_ids": [30, 31, 32, 33], "positions": [2, 3], "old_logprobs": [None] * 2, "weight": -1.0}
    both, _ = online.sequence_logprobs(model, [a, b], pad_id=2)
    one_a, _ = online.sequence_logprobs(model, [a], pad_id=2)
    one_b, _ = online.sequence_logprobs(model, [b], pad_id=2)
    assert torch.allclose(both, torch.cat([one_a, one_b]), atol=1e-5)
    stats = online.backward_step(model, [a, b], pad_id=2, normaliser=1.0, cap=2.0, budget=100)
    assert stats["targets"] == 5 and stats["micro_batches"] == 1 and stats["known_targets"] == 0
    for item, sign in ((a, 1), (b, -1)):  # a small step moves each sequence's log-prob with its advantage
        model, _ = tiny_policy()()
        before = online.sequence_logprobs(model, [item], pad_id=2)[0].sum().item()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        online.backward_step(model, [item], pad_id=2, normaliser=1.0, cap=2.0, budget=100)
        optimizer.step()
        after = online.sequence_logprobs(model, [item], pad_id=2)[0].sum().item()
        assert sign * (after - before) > 0


def test_every_lora_weight_gets_a_gradient_under_checkpointing():
    """The 4B trains with gradient checkpointing (reentrant); inputs must require grad or LoRA weights stay frozen."""
    import warnings
    torch = pytest.importorskip("torch")
    model, _ = tiny_policy(checkpointing=True)()
    model.train()
    item = {"input_ids": list(range(20, 60)), "positions": list(range(30, 40)), "old_logprobs": [None] * 10,
            "weight": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # "None of the inputs have requires_grad=True" would fail here
        online.backward_step(model, [item], pad_id=2, normaliser=1.0, cap=2.0, budget=1000)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_truncated_importance_weights_and_lora_check():
    torch = pytest.importorskip("torch")
    model, _ = tiny_policy()()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    item = {"input_ids": [20, 21, 22, 23, 24], "positions": [2, 3, 4], "weight": 1.0}
    with torch.no_grad():
        lora, _ = online.sequence_logprobs(model, [item | {"old_logprobs": [None] * 3}], pad_id=2)
        with model.disable_adapter():
            base, _ = online.sequence_logprobs(model, [item | {"old_logprobs": [None] * 3}], pad_id=2)
    far = item | {"old_logprobs": [v - 5.0 for v in lora.tolist()]}  # sampler thought these were much less likely
    stats = online.backward_step(model, [far], pad_id=2, normaliser=1.0, cap=2.0, budget=100)
    assert stats["tis_clipped"] == 3 and stats["abs_gap_sum"] == pytest.approx(15.0, rel=1e-4)
    noise = [0.01, -0.01, 0.01]  # kernel noise between engine and trainer, symmetric around zero
    with_lora = online.lora_check(model, [item | {"old_logprobs": [v + e for v, e in zip(lora.tolist(), noise)]}],
                                  pad_id=2, budget=100)
    assert with_lora["passed"] and with_lora["verdict"] == "closer_to_lora"
    assert with_lora["mean_abs_lora_effect"] == pytest.approx(float((lora - base).abs().mean()), rel=1e-4)
    without = online.lora_check(model, [item | {"old_logprobs": [v + e for v, e in zip(base.tolist(), noise)]}],
                                pad_id=2, budget=100)
    assert not without["passed"] and without["verdict"] == "failed"  # the engine sampled from the bare base
    misaligned = online.lora_check(model, [item | {"old_logprobs": [v - 1.0 for v in lora.tolist()]}], pad_id=2,
                                   budget=100)
    assert misaligned["verdict"] == "failed"
    with torch.no_grad():  # a LoRA without any effect cannot be checked: recorded, not fatal
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
    assert online.lora_check(model, [item | {"old_logprobs": lora.tolist()}], pad_id=2,
                             budget=100)["verdict"] in {"inconclusive", "failed"}


# ------------------------------------------------------------------------------------------------ fixture run
def make_db(path: Path, values: list[int]) -> str:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items(v INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(v,) for v in values])
    return digest(path)


def fake_adapter(folder: Path, name: str) -> str:
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(name.encode())
    return _adapter_hash(folder)


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
    for seed in (42, 43, 44):
        base = imports / "adapters" / f"PTR_seed{seed}"
        h = fake_adapter(base / "adapter", f"PTR{seed}")
        (base / "result.json").write_text(json.dumps({"training_completed": True, "adapter_reload_verified": True,
                                                      "identity": f"i{seed}", "adapter_hash": h}), encoding="utf-8")
        (base / "manifest.json").write_text(json.dumps({"identity": f"i{seed}", "run": v08["identity"], "family": "PTR",
                                                        "seed": seed}), encoding="utf-8")
    settings = json.loads((REPO / "configs" / "v09-smoke.json").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, settings, models_root=tmp_path / "models", manifest_path=tmp_path / "models.json",
                    imports=imports, smoke_limit=100, v08_run=v08["identity"])
    return run, frozen


def test_freeze_verifies_the_v08_start_points(fixture_run, tmp_path):
    run, frozen = fixture_run
    assert frozen["counts"] == {"pool": 8, "dev": 4, "validation": 4, "test": 4}
    assert set(frozen["imports"]["adapters"]) == {"PTR_seed42"}  # the smoke settings register seed 42 only
    assert load(run)[0]["identity"] == frozen["identity"]
    reg, raw, imports = tmp_path / "registration", tmp_path / "raw", tmp_path / "imports"
    settings = json.loads((REPO / "configs" / "v09-smoke.json").read_text(encoding="utf-8"))
    kw = dict(models_root=tmp_path / "models", manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100)
    with pytest.raises(ValueError, match="registered v0.8 run"):
        freeze(reg, raw, tmp_path / "other", settings, **kw)  # the real v0.8 identity is not this fixture's
    v08 = read_json(imports / "v08" / "manifest.json")
    (imports / "adapters" / "PTR_seed42" / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        freeze(reg, raw, tmp_path / "other2", settings, **kw, v08_run=v08["identity"])


def fake_online(run: Path, family: str, seed: int, steps=(1, 2)) -> None:
    """What a finished online run leaves behind (milestone adapters, records), without training."""
    frozen, _ = load(run)
    out = run / "train" / f"{family}_seed{seed}"
    marks = {}
    for step in steps:
        marks[str(step)] = fake_adapter(out / "milestones" / f"step-{step:04d}" / "adapter", f"{family}{seed}-{step}")
    (out / "milestones.json").write_text(json.dumps(marks), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"run": frozen["identity"], "identity": "x"}), encoding="utf-8")
    (out / "result.json").write_text(json.dumps({"finished": True, "steps_completed": max(steps),
                                                 "last_milestone": max(steps), "stopped_by": "planned"}),
                                     encoding="utf-8")


class Scripted:
    """Evaluation generator: queries, then finishes with the gold SQL on even question numbers."""
    kind = "model"

    def __init__(self) -> None:
        self.metadata = {"calls": 0}
        self.engine = ScriptedEngine()

    def generate(self, conversations):
        tok = self.engine.tok
        out = []
        for messages in conversations:
            ids = tok.apply_chat_template(messages, add_generation_prompt=True)
            self.metadata["calls"] += 1
            out.append({"text": json.dumps(self.engine.action(ids, 0.0)), "usage": {}, "finish_reason": "stop"})
        return out


def factory(spec, settings, **_):
    return Scripted()


def test_common_milestone_rule(fixture_run):
    run, frozen = fixture_run
    fake_online(run, "RL", 42, steps=(1, 2))
    fake_online(run, "RAFT", 42, steps=(1,))
    fake_online(run, "RLhi", 42, steps=(1, 2))
    result = common_steps(run)
    assert result["k"] == 1 and result["short"] == []
    assert {n: v["step"] for n, v in result["evaluate"].items()} == {"RL_seed42": 1, "RAFT_seed42": 1, "RLhi_seed42": 1}
    folder, h, step = adapter_for(run, frozen, "RL_seed42")
    assert step == 1 and folder.parent.name == "step-0001" and h == read_json(run / "train" / "RL_seed42" / "milestones.json")["1"]
    assert adapter_for(run, frozen, "RL_seed42@2")[2] == 2
    with pytest.raises(ValueError, match="no milestone"):
        adapter_for(run, frozen, "RAFT_seed42@2")
    (folder / "adapter_config.json").write_text('{"changed": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        adapter_for(run, frozen, "RL_seed42")


def test_short_main_runs_do_not_pull_the_milestone_down(fixture_run, monkeypatch):
    from exec_feedback.v9 import evaluate as ev

    run, frozen = fixture_run
    s = dict(frozen["settings"], seeds=[42, 43, 44])
    monkeypatch.setattr(ev, "load", lambda r, role=None, **kw: ({**frozen, "settings": s}, []))
    for name, steps in (("RL_seed42", (10, 20, 30)), ("RAFT_seed42", (10, 20, 30)), ("RL_seed43", (10, 20)),
                        ("RAFT_seed43", (10, 20, 30)), ("RL_seed44", (10,)), ("RAFT_seed44", (10, 20, 30)),
                        ("RLhi_seed42", (10, 20, 30))):
        family, seed = name.split("_seed")
        fake_online(run, family, int(seed), steps)
    result = ev.common_steps(run)
    assert result["k"] == 20 and result["short"] == ["RL_seed44"]
    assert result["evaluate"]["RL_seed44"] == {"step": 10, "adapter_hash": result["evaluate"]["RL_seed44"]["adapter_hash"],
                                               "short": True}
    assert result["evaluate"]["RAFT_seed44"]["step"] == 20 and result["evaluate"]["RLhi_seed42"]["step"] == 20


def test_evaluate_and_report_end_to_end(fixture_run):
    from exec_feedback.v9.analysis import report

    run, frozen = fixture_run
    for family in ("RL", "RAFT", "RLhi"):
        fake_online(run, family, 42)
    common_steps(run)
    for condition in ("B", "PTR_seed42", "RL_seed42", "RAFT_seed42", "RLhi_seed42", "RL_seed42@1"):
        for role in ("dev", "validation", "test"):
            result = evaluate(run, condition=condition, role=role, stop=None, factory=factory)
            assert result["complete"] and result["correct"] == 2  # even question numbers
    rows, info = merged(run, "test", "RL_seed42")
    assert info["step"] == 2 and all(r.get("gold_source") == "frozen_cache" for r in rows)
    with pytest.raises(ValueError, match="Unknown condition"):
        evaluate(run, condition="TR_seed42", role="dev", stop=None, factory=factory)
    result = report(run, procs=1)
    test = result["roles"]["test"]["project"]
    assert "RL_seed42@1" not in test["conditions"] and {"RL-PTR", "RL-RAFT", "RLhi-RL", "PTR-B"} <= set(test["deltas"])
    assert result["decisions"]["RL-PTR"]["decided"] and result["decisions"]["RL-PTR"]["holds"] is False  # ties
    assert result["select_final"]["family"] in {"PTR", "RL", "RAFT"}
    assert result["errors"]["test"]["conditions"]["B"]["outcomes"] == {"correct": 2, "wrong_values": 2}
    text = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# v0.9") and "错误分析：test" in text


# ------------------------------------------------------------------------------------------------ online run
def test_online_run_end_to_end_then_resume(fixture_run, monkeypatch):
    """Two online steps with the tiny model: real dialogues, oracle rewards, LoRA check, updates, milestones, curves;
    then a crash before the second checkpoint and a resume that replays exactly that step."""
    pytest.importorskip("torch")
    run, frozen = fixture_run
    holder: dict = {}
    loader = tiny_policy(holder, checkpointing=True)
    engines: list[ScriptedEngine] = []

    def engine_factory(spec, settings, *, seed):
        engines.append(ScriptedEngine(holder, seed))
        return engines[-1]

    real_checkpoint = online.save_checkpoint
    calls = {"n": 0}

    def crashing_checkpoint(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash before the second checkpoint")
        return real_checkpoint(*a, **kw)

    monkeypatch.setattr(online, "save_checkpoint", crashing_checkpoint)
    with pytest.raises(RuntimeError, match="simulated crash"):
        online.train_online(run, family="RL", seed=42, until=None, engine_factory=engine_factory, policy_loader=loader)
    out = run / "train" / "RL_seed42"
    assert [r["step"] for r in recover_records(out / "metrics.jsonl")] == [1, 2]
    monkeypatch.setattr(online, "save_checkpoint", real_checkpoint)
    result = online.train_online(run, family="RL", seed=42, until=None, engine_factory=engine_factory,
                                 policy_loader=loader)
    assert result["finished"] and result["steps_completed"] == 2 and result["stopped_by"] == "planned"
    assert result["lora_check"]["passed"] and result["lora_check"]["mean_abs_gap_lora"] < 1e-4
    assert set(result["milestones"]) == {"1", "2"} and result["last_milestone"] == 2
    metrics = recover_records(out / "metrics.jsonl")
    assert [r["step"] for r in metrics] == [1, 2] and len({r["attempt"] for r in metrics}) == 2
    assert list(out.glob("metrics.rolled-back-*.jsonl")) and list((out / "rollouts").glob("step-0001.jsonl.rolled-back-*"))
    assert all(r["mean_abs_logp_gap"] < 1e-4 and r["tis_clipped_fraction"] == 0 for r in metrics)
    rows = recover_records(out / "rollouts" / "step-0000.jsonl")
    assert len(rows) == 16 and all("trace" not in s for r in rows for s in r["steps"])
    assert {r["weight"] for r in rows if r["correct"]} and all(r["valid"] for r in rows)
    curve = [r for r in recover_records(out / "curve.jsonl") if "correct" in r]
    assert sorted({r["step"] for r in curve}) == [0, 1, 2] and all(r["correct"] == 2 for r in curve)
    assert engines[-1].versions[0] == 1 and engines[-1].versions[-1] == 2  # resumed at step 1, LoRA re-read each step
    marks = online.milestones(out)
    assert all(_adapter_hash(out / "milestones" / f"step-{k:04d}" / "adapter") == h for k, h in marks.items())
    assert (out / "milestones" / "step-0002" / "adapter" / "efa-base.json").exists()
    again = online.train_online(run, family="RL", seed=42, until=None, engine_factory=engine_factory,
                                policy_loader=loader)
    assert again == result  # a finished run is not trained twice
    with pytest.raises(ValueError, match="Unregistered"):
        online.train_online(run, family="RL", seed=43, until=None, engine_factory=engine_factory, policy_loader=loader)


def test_raft_trains_on_correct_dialogues_only_and_stops_at_the_deadline(fixture_run):
    pytest.importorskip("torch")
    import datetime as dt
    run, frozen = fixture_run
    holder: dict = {}
    result = online.train_online(run, family="RAFT", seed=42, until=None,
                                 engine_factory=lambda spec, settings, seed: ScriptedEngine(holder, seed),
                                 policy_loader=tiny_policy(holder))
    rows = [r for p in sorted((run / "train" / "RAFT_seed42" / "rollouts").glob("step-*.jsonl")) for r in recover_records(p)]
    assert result["finished"] and all(r["trained"] == r["correct"] for r in rows)
    assert all(r["weight"] == (1.0 if r["correct"] else 0.0) for r in rows)
    late = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)).isoformat()
    result = online.train_online(run, family="RLhi", seed=42, until=late,
                                 engine_factory=lambda spec, settings, seed: ScriptedEngine(holder, seed),
                                 policy_loader=tiny_policy(holder))
    assert result["stopped_by"] == "deadline" and result["steps_completed"] == 0 and result["last_milestone"] is None


# ------------------------------------------------------------------------------------------------ scheduler
FAKE_CLI = r'''
import sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_v09 as t
from exec_feedback.v9 import analysis
from exec_feedback.v9.evaluate import common_steps, evaluate

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
run = Path(opt("--run"))
if cmd == "train":
    t.fake_online(run, opt("--family"), int(opt("--seed")))
elif cmd == "steps":
    common_steps(run)
elif cmd == "evaluate":
    evaluate(run, condition=opt("--condition"), role=opt("--role"), stop=opt("--stop"), factory=t.factory)
elif cmd == "report":
    analysis.report(run, procs=1)
else:
    raise SystemExit("unexpected command " + cmd)
'''


def test_scheduler_runs_smoke_then_main_plan_end_to_end(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v9.orchestrate import Plan

    smoke, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["smoke_limit"] = 0  # the copy plays the main run
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    stop = time.time() + 9 * 3600
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models"]["student"]["spec"]["local_path"]).parent,
                main_py=sys.executable, vllm_py=sys.executable, stop_epoch=stop, gpus=[0, 1, 2, 3])
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=[0, 1, 2, 3], stop=stop, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs",
                          plan=plan, poll=0.1, stagger=0, retry_delay=0, milestone=reached.append)
    plan.scheduler = scheduler
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status, status
    assert (smoke / "report" / "report.json").exists() and (main / "report" / "report.json").exists()
    for role in ("dev", "validation", "test"):
        for condition in ("B", "PTR_seed42", "RL_seed42", "RAFT_seed42", "RLhi_seed42"):
            assert (main / "eval" / role / condition / "summary.json").exists(), (role, condition)
    # smoke trainings on the first two GPUs, main ones only after both finished; evaluations kept off GPUs 0-1 then
    assert re.search(r"start S:train-RL_seed42 on GPU [01] ", status) and re.search(r"start S:train-RAFT_seed42 on GPU [01] ", status)
    first_main = status.index("start M:train-")  # the gate reads both smoke results from disk
    assert first_main > status.index("done S:train-") and first_main > status.index("start S:train-RAFT_seed42")
    assert status.index("start S:eval-PTR_seed42-dev") < status.index("start M:eval-PTR_seed42-test")
    assert status.index("start M:steps") < status.index("start M:eval-RL_seed42-test")
    assert {"M:eval-base", "M:train", "M:steps", "M:eval-online", "M:report", "S:report"} <= set(reached)


# ------------------------------------------------------------------------------------------------ BIRD fetch
@pytest.mark.parametrize("ranges_supported", [True, False])
def test_parallel_bird_fetch_or_clean_fallback(tmp_path, monkeypatch, ranges_supported):
    import hashlib
    import http.server
    import threading
    from exec_feedback import downloads
    from exec_feedback.v9 import fetch

    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train/train.json", json.dumps([{"q": i} for i in range(3000)]))
        z.writestr("train/db/shop.sqlite", bytes(range(256)) * 400)
    data = archive.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()

        def do_GET(self):
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
    out = tmp_path / "raw"
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/train.zip"
        monkeypatch.setattr(fetch, "PART", 4096)
        monkeypatch.setattr(fetch, "SOURCES", {"train": url})
        monkeypatch.setattr(downloads, "SOURCES", {"train": url})
        monkeypatch.setattr(sys, "argv", ["fetch", "--part", "train", "--sha256", hashlib.sha256(data).hexdigest(),
                                          "--out", str(out), "--workers", "4"])
        code = fetch.main()
    finally:
        server.shutdown()
    if ranges_supported:
        assert code == 0 and (out / ".downloads" / "train.zip").read_bytes() == data
        assert read_json(out / "train.source.json")["sha256"] == hashlib.sha256(data).hexdigest()
        assert (out / "train" / "train" / "db" / "shop.sqlite").exists()
    else:
        assert code == 1 and not list((out / ".downloads").iterdir()) and not (out / "train").exists()


# ------------------------------------------------------------------------------------------------ process exit
def test_hard_exit_keeps_the_code_and_leaves_no_child(tmp_path):
    """2026-10-19 H20: a vLLM process hung at interpreter exit. A non-daemon thread blocks a normal exit the same
    way; hard_exit must still return the code at once and stop the child (a vLLM engine core keeps GPU memory)."""
    psutil = pytest.importorskip("psutil")
    pid_file = tmp_path / "child.pid"
    script = ("import subprocess, sys, threading, time\n"
              "from exec_feedback.v9.shutdown import hard_exit\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
              f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
              "threading.Thread(target=time.sleep, args=(120,)).start()\n"
              "print('work done', flush=True)\n"
              "hard_exit(3, grace=5)\n")
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 3, proc.stderr
    assert "work done" in proc.stdout and time.monotonic() - started < 60
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_cli_error_exits_with_code_one(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "exec_feedback.v9", "status", "--run", str(tmp_path / "missing")],
                          cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and "Traceback" in proc.stderr
