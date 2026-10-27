"""v0.10 engineering tests: SQLite fixtures, a character tokenizer, a randomly initialised tiny Qwen3 (tied embeddings,
as the 4B) with real LoRA adapters merged into full weights, FSDP2 on a one-rank threaded process group on the CPU, and
rollout workers in threads whose engines hold a CPU copy of the published weights. No GPU, vLLM or real weights; nothing
here is a model result."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v09 import ChatTokenizer, Scripted, make_db  # noqa: E402

from exec_feedback.core import Task, digest, write_jsonl  # noqa: E402
from exec_feedback.storage import fingerprint, read_json, recover_records  # noqa: E402
from exec_feedback.v7.freeze import _adapter_hash  # noqa: E402
from exec_feedback.v8.freeze import expected_identity, mark_verified, model_spec  # noqa: E402
from exec_feedback.v10 import MODELS, V08_PROTOCOL, V09_PROTOCOL  # noqa: E402
from exec_feedback.v10.config import DEFAULT, per_worker, validate_settings  # noqa: E402
from exec_feedback.v10.freeze import freeze, load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GOLD = "SELECT sum(v) FROM items"


def smoke_settings(**extra) -> dict:
    settings = json.loads((REPO / "configs" / "v10-smoke.json").read_text(encoding="utf-8"))
    return {**settings, "trainer_gpus": 1, **extra}  # one CPU rank in these tests


# ------------------------------------------------------------------------------------------------ settings
def test_registered_settings_and_configs():
    validate_settings(dict(DEFAULT))
    assert json.loads((REPO / "configs" / "h20-v10.json").read_text(encoding="utf-8")) == DEFAULT
    smoke = json.loads((REPO / "configs" / "v10-smoke.json").read_text(encoding="utf-8"))
    validate_settings(smoke)
    assert smoke["phase"] == "smoke" and smoke["trainer_gpus"] + smoke["rollout_gpus"] == 4
    assert per_worker(DEFAULT) == 4 and DEFAULT["planned_steps"] % DEFAULT["save_steps"] == 0
    models = json.loads((REPO / "configs" / "v10-models.json").read_text(encoding="utf-8"))
    assert set(models) == {MODELS["student"][0]} and models[MODELS["student"][0]]["revision"] == MODELS["student"][1]
    for bad in ({"rollout_gpus": 3}, {"trainer_gpus": 5}, {"rollout_slots": 16}, {"curve_every": 15},
                {"learning_rate": 0.0}, {"reshard_after_forward": 1}, {"group_size": 4}, {"max_steps": 6}):
        with pytest.raises(ValueError):
            validate_settings({**DEFAULT, **bad})


def test_condition_names():
    from exec_feedback.v10.evaluate import parse

    assert parse("B") == (None, None, None)
    assert parse("PTR_seed43") == ("PTR", 43, None) and parse("RL_seed42") == ("RL", 42, None)
    assert parse("RLF_seed42") == ("RLF", 42, None) and parse("RLFa_seed44@60") == ("RLFa", 44, 60)
    for bad in ("PTR_seed42@10", "RL_seed42@200", "RAFT_seed42", "RLhi_seed42", "RLF", "RLF_seed42@"):
        with pytest.raises(ValueError):
            parse(bad)


# ------------------------------------------------------------------------------------------------ weights
def test_checksum_ignores_stacking_and_order_but_sees_one_ulp():
    import math
    torch = pytest.importorskip("torch")
    from exec_feedback.v10.weights import checksum, same_checksum

    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(256, 64, generator=g).mul(0.02).to(torch.bfloat16) for _ in range(3))
    big = torch.randn(1 << 20, generator=g).mul(0.02).to(torch.bfloat16)
    parts = checksum([q, k, v, big])
    assert same_checksum(parts, checksum([big, torch.cat([q, k, v])]))  # vLLM's stacked qkv, another order
    i = int((big.float().abs() > 0.01).nonzero()[0])  # a typical weight; one bf16 ulp up
    x = float(big[i])
    bumped = big.clone()
    bumped[i] = x + 2.0 ** (math.floor(math.log2(abs(x))) - 7)
    assert float(bumped[i]) != x and not same_checksum(parts, checksum([q, k, v, bumped]))


def test_export_and_write_weights_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from exec_feedback.v10.weights import WEIGHTS, checksum, export_full, same_checksum, write_weights

    model = tiny_base()
    state, sums = export_full(model, 0)
    assert "lm_head.weight" not in state and all(t.dtype == torch.bfloat16 for t in state.values())
    ready = write_weights(tmp_path / "v00007", state, {"version": 7}, sums)
    assert read_json(tmp_path / "v00007.ready.json") == ready and ready["version"] == 7
    back = load_file(str(tmp_path / "v00007" / WEIGHTS))
    assert set(back) == set(state) and all(torch.equal(back[k], state[k]) for k in state)
    assert same_checksum(checksum(back.values()), ready["checksum"])
    assert export_full(model, 1) == (None, None)  # other ranks only join the gathers


def test_rank_shards_balance_tokens_and_pad_with_dummies():
    from exec_feedback.v10.trainer import rank_shards

    items = [{"key": f"t#{i}", "input_ids": [5] * n, "positions": [n - 1], "old_logprobs": [None], "weight": 1.0}
             for i, n in enumerate((600, 500, 500, 100))]
    shards = rank_shards(items, 2, 1000, pad_id=2)
    assert len({len(s) for s in shards}) == 1  # every rank runs the same number of forward/backward passes
    real = [x for s in shards for b in s for x in b if not x.get("dummy")]
    assert sorted(x["key"] for x in real) == sorted(x["key"] for x in items)
    loads = [sum(len(x["input_ids"]) for b in s for x in b if not x.get("dummy")) for s in shards]
    assert sorted(loads) == [700, 1000]  # longest first onto the lighter rank
    dummies = [x for s in shards for b in s for x in b if x.get("dummy")]
    assert len(dummies) == 1 and dummies[0]["weight"] == 0.0 and dummies[0]["input_ids"] == [2, 2]
    assert rank_shards([], 2, 1000, pad_id=2) == [[], []]


def test_request_assignment_keeps_whole_groups_in_task_order():
    from exec_feedback.v10.rollout import assignment, curve_assignment

    tasks = [Task(f"t{i}", "db", "train", "q", GOLD, "db.sqlite", "0" * 64, "s", "h") for i in range(8)]
    assert assignment(tasks, 4) == {"0": ["t0", "t1"], "1": ["t2", "t3"], "2": ["t4", "t5"], "3": ["t6", "t7"]}
    assert curve_assignment(tasks[:5], 2) == {"0": ["t0", "t2", "t4"], "1": ["t1", "t3"]}
    with pytest.raises(ValueError):
        assignment(tasks[:6], 4)


def test_channel_requests_results_weights_and_liveness(tmp_path):
    from exec_feedback.v10.protocol import Channel, wait_for

    channel = Channel.reset(tmp_path / "c")
    channel.post(0, {"kind": "sample", "step": 0, "version": 0, "assign": {}})
    assert channel.request(0)["kind"] == "sample" and channel.request(1) is None
    channel.put_result(0, 0, {"x": 1})
    assert channel.results(0, 2) is None
    channel.put_result(0, 1, {"x": 2})
    assert [p["x"] for p in channel.results(0, 2)] == [1, 2]
    for version in range(4):
        (channel.weights(version)).mkdir(parents=True)
        (channel.root / "weights" / f"v{version:05d}.ready.json").write_text(json.dumps({"path": "p"}), encoding="utf-8")
    channel.prune(keep_from=2)
    assert [channel.ready(v) is not None for v in range(4)] == [False, False, True, True]
    channel.beat(0, "ok")
    assert channel.silent(2, 60.0, since=time.time()) == [] and channel.silent(2, 60.0, since=time.time() - 120) == [1]
    channel.stop("test")
    with pytest.raises(RuntimeError, match="Stopped"):
        wait_for(lambda: None, timeout=5, what="nothing", channel=channel)


# ------------------------------------------------------------------------------------------------ tiny models
def tiny_config():
    from transformers import Qwen3Config

    return Qwen3Config(vocab_size=9000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=8192,
                       tie_word_embeddings=True)


def tiny_base():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    return Qwen3ForCausalLM(tiny_config())


def save_adapter(base_dir: Path, folder: Path, seed: int) -> str:
    """A real LoRA on the tiny base (non-zero B, so merging changes the model)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(str(base_dir))
    peft = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
                                            target_modules=["q_proj", "v_proj", "o_proj"], init_lora_weights=False))
    folder.mkdir(parents=True)
    peft.save_pretrained(str(folder))
    return _adapter_hash(folder)


# from_pretrained swaps process-wide state while it loads (meta-device init, default dtype); the worker threads and the
# trainer share one process here (separate processes on the host), so their loads take turns
LOADING = threading.Lock()


def locked_policy(start: Path, device):
    from exec_feedback.v10.trainer import load_policy

    with LOADING:
        return load_policy(start, device)


class FakeFullEngine:
    """The rollout engine's interface over a bf16 CPU copy of the weights: scripted actions (a query, then a final
    answer: a coin flip when sampling, right on even question numbers when greedy); log-probs from the loaded weights,
    so they follow every weight version the trainer publishes."""

    def __init__(self, model_dir: Path, settings: dict | None = None, *, seed: int = 0) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        with LOADING:
            self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
        self.model.eval()
        self.rng, self.tok, self.loads = random.Random(seed), ChatTokenizer(), []

    def checksum(self) -> list[float]:
        from exec_feedback.v10.weights import checksum
        return checksum(self.model.parameters())

    def load(self, path: Path) -> list[float]:
        from safetensors.torch import load_file
        state = load_file(str(path))
        _, unexpected = self.model.load_state_dict(state, strict=False)
        assert not unexpected
        self.loads.append(Path(path).parent.name)
        return self.checksum()

    def action(self, ids: list[int], temperature: float) -> dict:
        text = self.tok.decode(ids)
        if text.count("<|im_start|>assistant") == 1:
            return {"action": "query", "sql": "SELECT v FROM items"}
        n = int(re.search(r"Question (\d+)", text).group(1))
        right = self.rng.random() < 0.5 if temperature > 0 else n % 2 == 0
        return {"action": "finish", "sql": GOLD if right else "SELECT max(v) FROM items"}

    def generate(self, prompts, *, temperature, max_tokens, stop_ids, logprobs):
        out = []
        for ids in prompts:
            tokens = self.tok.encode(json.dumps(self.action(ids, temperature))) + [2]
            out.append({"tokens": tokens, "finish_reason": "stop", "stop_token": 2, "cached_tokens": 0,
                        "logprobs": self.logprobs(ids, tokens) if logprobs else None})
        return out

    def logprobs(self, prompt: list[int], tokens: list[int]) -> list[float]:
        import torch
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([prompt + tokens]), use_cache=False).logits[0, len(prompt) - 1:-1]
            return torch.log_softmax(logits.float(), -1).gather(1, torch.tensor(tokens)[:, None])[:, 0].tolist()


# ------------------------------------------------------------------------------------------------ fixture run
@pytest.fixture
def fixture_run(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
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
    # the tiny base, saved as the pinned 4B is (one safetensors file, tied embeddings)
    model_id, revision = MODELS["student"]
    student = tmp_path / "models" / "student"
    tiny_base().save_pretrained(str(student), safe_serialization=True)
    files = {p.name: {"size": p.stat().st_size, "sha256": digest(p)} for p in sorted(student.iterdir()) if p.is_file()}
    mark_verified(student, files)
    (tmp_path / "models.json").write_text(json.dumps({model_id: {"revision": revision, "files": files}}), encoding="utf-8")
    snapshot = expected_identity(model_spec("student", student, DEFAULT), files)["local_snapshot"]
    imports = tmp_path / "imports"
    v08 = {"protocol": V08_PROTOCOL, "registration": {"source_identity": v05},
           "models": {"student": {"identity": {"local_snapshot": snapshot}}}}
    v08["identity"] = fingerprint(v08)
    (imports / "v08").mkdir(parents=True)
    (imports / "v08" / "manifest.json").write_text(json.dumps(v08), encoding="utf-8")
    v09 = {"protocol": V09_PROTOCOL, "registration": {"source_identity": v05},
           "models": {"student": {"identity": {"local_snapshot": snapshot}}}, "imports": {"v08_run": v08["identity"]},
           "settings": {}}
    v09["identity"] = fingerprint(v09)
    (imports / "v09").mkdir(parents=True)
    (imports / "v09" / "manifest.json").write_text(json.dumps(v09), encoding="utf-8")
    chosen = {}
    for seed in (42, 43, 44):
        base = imports / "adapters" / f"PTR_seed{seed}"
        h = save_adapter(student, base / "adapter", seed)
        (base / "result.json").write_text(json.dumps({"training_completed": True, "adapter_reload_verified": True,
                                                      "identity": f"i{seed}", "adapter_hash": h}), encoding="utf-8")
        (base / "manifest.json").write_text(json.dumps({"identity": f"i{seed}", "run": v08["identity"], "family": "PTR",
                                                        "seed": seed}), encoding="utf-8")
        lora = imports / "v09" / "adapters" / f"RL_seed{seed}"
        h = save_adapter(student, lora / "adapter", 100 + seed)
        (lora / "result.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
        (lora / "manifest.json").write_text(json.dumps({"run": v09["identity"], "family": "RL", "seed": seed}),
                                            encoding="utf-8")
        (lora / "milestones.json").write_text(json.dumps({"3": h}), encoding="utf-8")
        chosen[f"RL_seed{seed}"] = {"step": 3, "adapter_hash": h, "short": False}
    (imports / "v09" / "steps.json").write_text(json.dumps({"run": v09["identity"], "k": 3, "evaluate": chosen}),
                                                encoding="utf-8")
    run = tmp_path / "run"
    frozen = freeze(reg, raw, run, smoke_settings(), models_root=tmp_path / "models",
                    manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100, v08_run=v08["identity"],
                    v09_run=v09["identity"])
    return run, frozen


def test_freeze_verifies_both_imports(fixture_run, tmp_path):
    run, frozen = fixture_run
    assert frozen["counts"] == {"pool": 8, "dev": 4, "validation": 4, "test": 4}
    assert set(frozen["imports"]["adapters"]) == {"PTR_seed42"} and set(frozen["imports"]["v09"]["rl"]) == {"RL_seed42"}
    assert frozen["imports"]["v09"]["k"] == 3 and load(run)[0]["identity"] == frozen["identity"]
    reg, raw, imports = tmp_path / "registration", tmp_path / "raw", tmp_path / "imports"
    kw = dict(models_root=tmp_path / "models", manifest_path=tmp_path / "models.json", imports=imports, smoke_limit=100,
              v08_run=frozen["imports"]["v08_run"])
    with pytest.raises(ValueError, match="registered v0.9 run"):
        freeze(reg, raw, tmp_path / "other", smoke_settings(), **kw)  # the real v0.9 identity is not this fixture's
    v09 = frozen["imports"]["v09"]["v09_run"]
    (imports / "v09" / "adapters" / "RL_seed42" / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree|changed"):
        freeze(reg, raw, tmp_path / "other2", smoke_settings(), **kw, v09_run=v09)


def test_merge_reproduces_the_lora_model_and_is_verified(fixture_run):
    torch = pytest.importorskip("torch")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from exec_feedback.v10.merge import merge, merged_dir, merged_model, record_of

    run, frozen = fixture_run
    record = merge(run, name="PTR_seed42")
    assert merge(run, name="PTR_seed42") == record  # a verified merge is not redone
    folder, again = merged_model(run, frozen, "PTR_seed42")
    assert again == record and record["fp32_merge_max_logit_delta"] < 1e-3
    probe = torch.tensor([list(range(100, 140))])
    base = AutoModelForCausalLM.from_pretrained(frozen["models"]["student"]["spec"]["local_path"])
    lora = PeftModel.from_pretrained(base, str(run / "imports" / "adapters" / "PTR_seed42" / "adapter")).eval()
    full = AutoModelForCausalLM.from_pretrained(str(folder)).eval()
    with torch.no_grad():
        assert torch.allclose(lora(input_ids=probe).logits, full(input_ids=probe).logits, atol=0.05)
    assert merge(run, name="RL_seed42")["inputs"]["adapter"] == frozen["imports"]["v09"]["rl"]["RL_seed42"]["adapter_hash"]
    with pytest.raises(ValueError, match="Unregistered"):
        merge(run, name="PTR_seed43")  # the smoke settings register seed 42 only
    (folder / "config.json").write_text((folder / "config.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed on disk"):
        merged_model(run, frozen, "PTR_seed42")
    assert record_of(merged_dir(Path(frozen["models_root"]), "PTR_seed42")).exists()


# ------------------------------------------------------------------------------------------------ one-rank FSDP2
@pytest.fixture
def one_rank(tmp_path):
    dist = pytest.importorskip("torch.distributed")
    from torch.testing._internal.distributed.multi_threaded_pg import _install_threaded_pg, _uninstall_threaded_pg

    _install_threaded_pg()
    dist.init_process_group("threaded", rank=0, world_size=1, store=dist.FileStore(str(tmp_path / "pg-store"), 1))
    try:
        yield
    finally:
        dist.destroy_process_group()
        _uninstall_threaded_pg()


def test_fsdp_gradient_matches_the_v09_objective(one_rank):
    """One FSDP2 rank in bf16 compute against v0.9's backward on a plain bf16 copy: same loss, same gradient."""
    torch = pytest.importorskip("torch")
    from torch.distributed.device_mesh import init_device_mesh
    from exec_feedback.v9.online import backward_step
    from exec_feedback.v10.trainer import backward_rank, clip_and_norm, rank_shards, shard

    items = [{"key": "a", "input_ids": list(range(20, 60)), "positions": list(range(30, 40)),
              "old_logprobs": [-3.0] * 10, "weight": 0.75},
             {"key": "b", "input_ids": list(range(70, 95)), "positions": list(range(15, 25)),
              "old_logprobs": [None] * 10, "weight": -0.25}]
    plain = tiny_base().to(torch.bfloat16)
    ref = backward_step(plain, items, pad_id=2, normaliser=64.0, cap=2.0, budget=4096)
    model = shard(tiny_base(), init_device_mesh("cpu", (1,)), True)
    model.train()
    stats = backward_rank(model, rank_shards(items, 1, 4096, pad_id=2)[0], pad_id=2, normaliser=64.0, cap=2.0, scale=1.0)
    assert stats["targets"] == ref["targets"] == 20 and stats["known_targets"] == 10
    assert stats["loss"] == pytest.approx(ref["loss"], rel=0.05)
    grads = {n: p.grad.full_tensor().float() for n, p in model.named_parameters()}
    for name, p in plain.named_parameters():
        a, b = grads[name].flatten(), p.grad.float().flatten()
        assert torch.nn.functional.cosine_similarity(a, b, dim=0) > 0.98, name
    norm = clip_and_norm(model, 1e-3)
    assert norm > 1e-3 and sum(float(p.grad.full_tensor().float().pow(2).sum()) for p in model.parameters()) ** 0.5 \
        == pytest.approx(1e-3, rel=0.01)


def fsdp_gradients(items: list[dict], worlds: tuple[int, ...], scratch: Path, compute: str = "float32",
                   budget: int = 200, replay: bool = False) -> dict:
    """{world: (global norm, full gradients)} of one backward over `items` on `world` CPU ranks (threads), through the
    trainer's path: activation checkpointing as in load_policy, shard, rank_shards, backward_rank scaled by the world
    size, clip_and_norm. compute "bfloat16" is the runs' policy; "float32" swaps only the dtype shard() computes in.
    With replay one rank runs each rank count's split instead, as bench --regroup does on one GPU."""
    import copy

    import torch
    import torch.distributed as dist
    import torch.distributed.fsdp as fsdp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.testing._internal.distributed.multi_threaded_pg import (ProcessLocalGroup, _install_threaded_pg,
                                                                        _uninstall_threaded_pg)
    from exec_feedback.v10.bench import regrouped
    from exec_feedback.v10.trainer import backward_rank, clip_and_norm, rank_shards, shard

    base = tiny_base()
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scratch.mkdir(parents=True, exist_ok=True)
    out, errors = {}, []

    def rank(world: int, ranks: int, r: int, model, store) -> None:
        try:
            dist.init_process_group("threaded", rank=r, world_size=ranks, store=store)
            model = shard(model, init_device_mesh("cpu", (ranks,)), True)
            model.train()
            mine = regrouped(items, world, budget, pad_id=2) if replay else rank_shards(items, world, budget, pad_id=2)[r]
            backward_rank(model, mine, pad_id=2, normaliser=512.0, cap=2.0, scale=float(ranks))
            norm = clip_and_norm(model, 1e9)  # the norm only, nothing clipped
            grads = {n: p.grad.full_tensor().detach().clone() for n, p in model.named_parameters()}
            if r == 0:
                out[world] = (norm, grads)
        except BaseException as exc:  # noqa: BLE001 - reported by the caller
            errors.append(exc)
            ProcessLocalGroup.exception_handle(exc)  # releases the ranks waiting in a collective
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    policy = fsdp.MixedPrecisionPolicy
    if compute != "bfloat16":  # shard() imports the policy class when called
        fsdp.MixedPrecisionPolicy = lambda **_: policy(param_dtype=getattr(torch, compute), reduce_dtype=torch.float32)
    _install_threaded_pg()
    torch._C._distributed_c10d._set_thread_isolation_mode(True)  # a group registry per thread, as torch's own tests
    try:
        for world in worlds:
            ranks = 1 if replay else world
            store = dist.FileStore(str(scratch / f"pg-store-{compute}-{world}"), ranks)
            threads = [threading.Thread(target=rank, args=(world, ranks, r, copy.deepcopy(base), store), daemon=True)
                       for r in range(ranks)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=300)
            assert not errors and not any(t.is_alive() for t in threads), errors or f"{world} ranks hung"
    finally:
        ProcessLocalGroup.reset()
        torch._C._distributed_c10d._set_thread_isolation_mode(False)
        _uninstall_threaded_pg()
        fsdp.MixedPrecisionPolicy = policy
    return out


def gradient_items() -> list[dict]:
    """Eleven trajectories of uneven length, signed weights, sampler log-probs partly unknown and partly far enough below
    the policy's for the TIS cap: on 2 to 8 ranks the ranks hold different numbers of micro-batches, so some run
    zero-weight dummies."""
    rng = random.Random(7)
    items = []
    for i in range(11):
        n = rng.randint(30, 160)
        positions = list(range(rng.randint(10, n - 5), n))
        items.append({"key": f"t{i:02d}", "input_ids": [rng.randint(3, 8999) for _ in range(n)], "positions": positions,
                      "old_logprobs": [None if rng.random() < 0.3 else -rng.uniform(5.0, 13.0) for _ in positions],
                      "weight": rng.choice([0.875, -0.125, -0.5, 0.25])})
    return items


def flat_gradient(grads: dict):
    import torch

    return torch.cat([grads[n].double().flatten() for n in sorted(grads)])


def relative_gap(a: dict, b: dict) -> float:
    x, y = flat_gradient(a), flat_gradient(b)
    return float((x - y).norm() / y.norm())


def test_fsdp_gradient_does_not_depend_on_the_rank_count(tmp_path):
    """The H20 layout benchmark measured first-step gradient norms 4% apart on 4 and 8 ranks over the same trajectories
    (0.0956 against 0.0919; the two resharding layouts of one rank count agree to 5e-4). Computed in fp32, the gradient
    is the same on 1, 2, 4 and 8 ranks: how the trajectories are split over the ranks, the dummies, the world-size loss
    scaling and FSDP's average leave it unchanged."""
    pytest.importorskip("torch")
    from exec_feedback.v10.trainer import rank_shards

    items = gradient_items()
    for world in (4, 8):  # dummies on both rank counts the benchmark compared
        assert any(x.get("dummy") for part in rank_shards(items, world, 200, pad_id=2) for b in part for x in b), world
    measured = fsdp_gradients(items, (1, 2, 4, 8), tmp_path)
    norm, grads = measured[1]
    assert norm > 0
    for world in (2, 4, 8):
        # the norm's own fp32 sums differ by ~1e-5 with the shard sizes; the gradients agree to ~1e-7
        assert measured[world][0] == pytest.approx(norm, rel=1e-4), world
        assert relative_gap(measured[world][1], grads) < 1e-5, world


def test_one_rank_replays_a_rank_counts_bf16_gradient(tmp_path):
    """bench --regroup: in the runs' bf16 compute one rank running all micro-batches of a rank count's split gets the
    gradient those ranks get together (FSDP sums their bf16 gradients in fp32), though the splits' gradients differ."""
    pytest.importorskip("torch")
    items = gradient_items()
    ranks = fsdp_gradients(items, (2, 4, 8), tmp_path / "ranks", "bfloat16")
    one = fsdp_gradients(items, (2, 4, 8), tmp_path / "one", "bfloat16", replay=True)
    for world in (2, 4, 8):
        assert one[world][0] == pytest.approx(ranks[world][0], rel=1e-4), world
        assert relative_gap(one[world][1], ranks[world][1]) < 1e-6, world
    assert relative_gap(ranks[4][1], ranks[8][1]) > 1e-4  # ~1e-3 here: what the replay has to tell apart


def test_regroup_replays_the_first_kept_step_on_one_rank(fixture_run, one_rank, tmp_path):
    """bench --regroup on a finished run: its first kept step under each split on one rank, from a start compared with
    the one the run trained from (a copy merged on another host may differ by a few roundings: recorded, not refused)."""
    torch = pytest.importorskip("torch")
    from exec_feedback.v10.bench import regroup_rank, regrouped
    from exec_feedback.v10.merge import merge, merged_model

    run, frozen = fixture_run
    merge(run, name="PTR_seed42")
    start, record = merged_model(run, frozen, "PTR_seed42")
    elsewhere = tmp_path / "start-elsewhere"  # the start merged again on another host
    shutil.copytree(start, elsewhere)
    kept = run / "train" / "RLF_seed42" / "bench-items"
    kept.mkdir(parents=True)
    items = gradient_items()
    (kept / "step-0000.json").write_text(json.dumps({"step": 0, "items": items}), encoding="utf-8")
    (kept / "step-0001.json").write_text(json.dumps({"step": 1, "items": items[:3]}), encoding="utf-8")
    manifest = kept.parent / "manifest.json"
    (run / "bench").mkdir(exist_ok=True)
    (run / "bench" / "z3-4.json").write_text(json.dumps({"first_grad_norms": [0.25, 0.5]}), encoding="utf-8")
    out = tmp_path / "regroup.json"
    kw = dict(family="RLF", seed=42, out=out, start=elsewhere, device=torch.device("cpu"), pad_id=2)
    manifest.write_text(json.dumps({"start_checksum": [record["checksum"][0] + 1.0, record["checksum"][1]]}),
                        encoding="utf-8")
    regroup_rank(run, worlds=[1], **kw)
    other = read_json(out)["start"]
    assert other["same"] is False and other["sum_delta"] == pytest.approx(-1.0) and other["squares_relative_delta"] == 0
    manifest.write_text(json.dumps({"start_checksum": record["checksum"]}), encoding="utf-8")
    regroup_rank(run, worlds=[1, 4], **kw)
    result = read_json(out)
    assert result["start"]["same"] and result["start"]["sum_delta"] == 0.0
    assert result["step"] == 0 and result["trajectories"] == len(items)
    rows = {r["world"]: r for r in result["rows"]}
    budget = frozen["settings"]["micro_tokens"]
    assert [rows[w]["micro_batches"] for w in (1, 4)] == [len(regrouped(items, w, budget, 2)) for w in (1, 4)]
    assert rows[4]["layout_first_grad_norm"] == 0.25 and rows[1]["layout_first_grad_norm"] is None
    norms = rows[1]["grad_norms"] + rows[4]["grad_norms"]
    assert len(norms) == 4 and min(norms) > 0 and max(norms) == pytest.approx(min(norms), rel=1e-3)


@contextlib.contextmanager
def workers(run: Path, family: str, root: Path, engines: list):
    """Two rollout workers in threads on a fresh channel, stopped on the way out even when the test body fails."""
    from exec_feedback.v10.protocol import Channel
    from exec_feedback.v10.rollout import serve

    def factory(model_dir, settings, *, seed):
        engines.append(FakeFullEngine(model_dir, settings, seed=seed))
        return engines[-1]

    Channel.reset(root)
    threads = [threading.Thread(target=serve, args=(run,), daemon=True,
                                kwargs=dict(family=family, seed=42, worker=w, root=root, engine_factory=factory,
                                            tokenizer=ChatTokenizer(), stop_ids=[2])) for w in range(2)]
    for t in threads:
        t.start()
    try:
        yield
    finally:
        Channel(root).stop("test over")
        for t in threads:
            t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)


def train(run: Path, family: str, root: Path, seed: int = 42) -> dict:
    from exec_feedback.v10 import trainer

    return trainer.train_full(run, family=family, seed=seed, until=None, root=root, pad_id=2,
                              policy_loader=locked_policy)


def test_online_sync_run_end_to_end_then_crash_and_resume(fixture_run, one_rank, tmp_path, monkeypatch):
    """RLF with two worker threads: real dialogues, oracle rewards, the weights check, FSDP updates, a weight file per
    step that every worker loads and checksums, milestones and curves; a crash before the second checkpoint, then a
    resume that replays exactly that step and finishes (a restored run does not take the smoke's planned stop)."""
    pytest.importorskip("torch")
    from exec_feedback.v10 import trainer
    from exec_feedback.v10.merge import merge

    run, frozen = fixture_run
    merge(run, name="PTR_seed42")
    root = tmp_path / "channel"
    engines: list = []
    real_checkpoint = trainer.save_checkpoint
    calls = {"n": 0}

    def crashing(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash before the second checkpoint")
        return real_checkpoint(*a, **kw)

    monkeypatch.setattr(trainer, "save_checkpoint", crashing)
    with workers(run, "RLF", root, engines), pytest.raises(RuntimeError, match="simulated crash"):
        train(run, "RLF", root)
    out = run / "train" / "RLF_seed42"
    assert [r["step"] for r in recover_records(out / "metrics.jsonl")] == [1, 2]
    monkeypatch.setattr(trainer, "save_checkpoint", real_checkpoint)
    resumed = len(engines)
    with workers(run, "RLF", root, engines):
        result = train(run, "RLF", root)
    assert result["finished"] and result["steps_completed"] == 3 and result["stopped_by"] == "planned"
    assert (out / "result.json").exists() and not (out / "stopped.json").exists()
    assert list(out.glob("metrics.rolled-back-*.jsonl")) and list((out / "rollouts").glob("step-0001.jsonl.rolled-back-*"))
    assert result["weights_check"]["passed"] and result["weights_check"]["mean_abs_gap"] < 0.05
    assert set(result["milestones"]) == {"1", "2", "3"} and result["last_milestone"] == 3
    metrics = recover_records(out / "metrics.jsonl")
    assert [r["step"] for r in metrics] == [1, 2, 3] and len({r["attempt"] for r in metrics}) == 2
    assert [r["versions_used"] for r in metrics] == [[0], [1], [2]] and all(r["staleness"] == 0 for r in metrics)
    assert all(r["weights_verified"] and r["mean_abs_logp_gap"] < 0.05 for r in metrics)
    assert any(r["optimizer_updated"] for r in metrics) and len(recover_records(out / "attempts.jsonl")) == 2
    rows = recover_records(out / "rollouts" / "step-0000.jsonl")
    assert len(rows) == 16 and all("trace" not in s for r in rows for s in r["steps"]) and all(r["valid"] for r in rows)
    curve = [r for r in recover_records(out / "curve.jsonl") if "correct" in r]
    assert {r["step"] for r in curve} == {0, 1, 2, 3} and all(r["correct"] == 2 and r["complete"] for r in curve)
    # after the resume every worker loaded each published version once, in order (the curve shares its version)
    assert all(e.loads == ["v00001", "v00002", "v00003"] for e in engines[resumed:])
    assert not (out / "ckpt").exists()  # a finished run keeps its milestones, not 48 GB of optimizer state
    from exec_feedback.v10.weights import dir_identity
    assert all(dir_identity(out / "milestones" / f"step-{k:04d}") == v for k, v in trainer.milestones(out).items())
    assert train(run, "RLF", root) == read_json(out / "result.json")
    with pytest.raises(ValueError, match="Unregistered"):
        train(run, "RLF", root, seed=43)


def test_one_step_stale_run_samples_with_the_previous_weights(fixture_run, one_rank, tmp_path):
    pytest.importorskip("torch")
    from exec_feedback.v10.merge import merge

    run, frozen = fixture_run
    merge(run, name="PTR_seed42")
    root = tmp_path / "channel-a"
    for _ in range(2):  # the planned smoke stop, then the resume
        with workers(run, "RLFa", root, []):
            result = train(run, "RLFa", root)
    assert result["finished"] and result["staleness"] == 1
    metrics = recover_records(run / "train" / "RLFa_seed42" / "metrics.jsonl")
    # step 0 has only the start weights; step 1 was posted before step 0 trained; after the resume at step 2 the first
    # sample is on-policy again (the restored weights are the newest)
    assert [r["versions_used"] for r in metrics] == [[0], [0], [2]] and [r["staleness"] for r in metrics] == [0, 1, 0]


def test_weights_gate_sees_a_load_in_the_outputs(fixture_run, monkeypatch):
    pytest.importorskip("torch")
    from exec_feedback.v10 import gate
    from exec_feedback.v10.gate import gate_weights
    from exec_feedback.v10.merge import merge

    # the tiny random model's layers add ~1% to its residual stream, so doubling v_proj or gate_proj moves its
    # log-probs by ~1e-3; on the 4B the greedy tokens change. A load the computation never sees moves them by 0.
    monkeypatch.setattr(gate, "MOVED", 1e-4)
    run, _ = fixture_run
    merge(run, name="PTR_seed42")
    report = gate_weights(run, engine_factory=lambda d, s, seed: FakeFullEngine(d, s, seed=seed),
                          tokenizer=ChatTokenizer(), stop_ids=[2])
    assert report["passed"], report
    assert report["perturbed"]["mean_abs_logp_delta"] > 0.01 and report["restored"]["max_abs_logp_delta"] == 0.0
    assert {"perturbed_checksum", "qkv_checksum", "gate_up_checksum", "qkv_outputs_moved",
            "gate_up_outputs_moved"} <= set(report["checks"]) and report["qkv"]["tensors"] == 2

    class Partial(FakeFullEngine):  # loads of plain parameters work, slices of the stacked ones never arrive
        def load(self, path):
            keep = {k: v.clone() for k, v in self.model.state_dict().items() if "v_proj" in k or "gate_proj" in k}
            sums = super().load(path)
            self.model.load_state_dict(keep, strict=False)
            return sums

    partial = gate_weights(run, engine_factory=lambda d, s, seed: Partial(d, s, seed=seed), tokenizer=ChatTokenizer(),
                           stop_ids=[2])
    assert not partial["passed"] and partial["checks"]["perturbed_outputs_moved"] and not partial["checks"][
        "qkv_outputs_moved"] and not partial["checks"]["gate_up_outputs_moved"]

    class Frozen(FakeFullEngine):  # an engine whose loads never reach the computation
        def load(self, path):
            sums = super().load(path)
            self.model.load_state_dict(self.first, strict=False)
            return sums

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.first = {k: v.clone() for k, v in self.model.state_dict().items()}

    assert not gate_weights(run, engine_factory=lambda d, s, seed: Frozen(d, s, seed=seed), tokenizer=ChatTokenizer(),
                            stop_ids=[2])["passed"]


# ------------------------------------------------------------------------------------------------ evaluation, report
def fake_online(run: Path, family: str, seed: int, steps=(1, 2, 3)) -> None:
    """What a finished online run leaves behind (milestone directories, records), without training."""
    from exec_feedback.v10.weights import dir_identity

    frozen, _ = load(run)
    out = run / "train" / f"{family}_seed{seed}"
    marks = {}
    for step in steps:
        folder = out / "milestones" / f"step-{step:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(f"{family}{seed}-{step}".encode())
        marks[str(step)] = dir_identity(folder)
    (out / "milestones.json").write_text(json.dumps(marks), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"run": frozen["identity"], "identity": "x"}), encoding="utf-8")
    (out / "result.json").write_text(json.dumps({"finished": True, "identity": "x", "steps_completed": max(steps),
                                                 "last_milestone": max(steps), "stopped_by": "planned"}),
                                     encoding="utf-8")
    rows = [{"step": s, "attempt": "a", "reward_mean": 0.5, "optimizer_updates": s, "step_seconds": 30.0 + 10 * (
        family == "RLF"), "wait_seconds": 10.0, "rollout_seconds": [9.0, 11.0], "load_seconds": [3.0, 4.0],
        "train_seconds": 6.0, "staleness": int(family == "RLFa")} for s in steps]
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def factory(spec, settings, **_):
    return Scripted()


def test_evaluate_and_report_end_to_end(fixture_run, tmp_path):
    from exec_feedback.v10.analysis import progress_entry, report
    from exec_feedback.v10.evaluate import evaluate, merged, model_for
    from exec_feedback.v10.merge import merge

    run, frozen = fixture_run
    for name in ("PTR_seed42", "RL_seed42"):
        merge(run, name=name)
    for family in ("RLF", "RLFa"):
        fake_online(run, family, 42)
    assert model_for(run, frozen, "RLF_seed42")[2] == 3 and model_for(run, frozen, "RLFa_seed42@1")[2] == 1
    assert model_for(run, frozen, "RL_seed42")[2] == 3 and model_for(run, frozen, "PTR_seed42")[2] == 0
    for condition in ("B", "PTR_seed42", "RL_seed42", "RLF_seed42", "RLFa_seed42"):
        for role in ("dev", "validation", "test"):
            result = evaluate(run, condition=condition, role=role, stop=None, factory=factory)
            assert result["complete"] and result["correct"] == 2  # even question numbers
    rows, info = merged(run, "test", "RLF_seed42")
    assert info["step"] == 3 and all(r.get("gold_source") == "frozen_cache" for r in rows)
    (run / "train" / "RLFa_seed42" / "result.json").unlink()
    with pytest.raises(ValueError, match="not finished"):
        model_for(run, frozen, "RLFa_seed42")
    result = report(run, procs=1)
    test = result["roles"]["test"]["project"]
    assert {"RLF-RL", "RLF-PTR", "RLFa-RLF", "RL-PTR", "PTR-B"} <= set(test["deltas"])
    decision = result["decisions"]["RLF-RL"]
    assert decision["decided"] and decision["holds"] is False and decision["reversed"] is False  # ties
    assert result["systems"]["async_speedup"] == {"42": pytest.approx(40.0 / 30.0)}
    assert result["dynamics"]["RLF_seed42"]["rollout_seconds"] == 11.0
    text = (run / "report" / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# v0.10") and "主比较 RLF-RL" in text and "每步时间去向" in text and "初步" in text
    entry = progress_entry(run, "M:report", "abc123")
    assert entry.startswith("\n### ") and "v0.10 自动记录" in entry and "RLF-RL" in entry

    # a second booking that trained seed 43: a copy of this run under its own identity, its seed-42 outcomes renamed;
    # the combined report reads only the saved per-task outcomes (no database at home)
    from exec_feedback.v10.analysis import combined
    other = tmp_path / "run-43"
    shutil.copytree(run, other)
    for folder in (run, other):  # both play main runs
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("identity")
        manifest.update(smoke_limit=0, booking=folder.name)
        manifest["identity"] = fingerprint(manifest)
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for role in ("test", "validation", "dev"):
        path = other / "report" / f"rows-{role}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved = {part: {n.replace("_seed42", "_seed43"): v for n, v in saved[part].items()} for part in saved}
        path.write_text(json.dumps(saved), encoding="utf-8")
    # the first booking's benchmark lost a layout, the second measured all; the gate is in the second's report only
    layout = {"exit": 0, "train_seconds": 2.0, "optimizer_seconds": 0.1, "gather_seconds": 1.0, "write_seconds": 1.0,
              "tokens_per_second": 900.0, "mfu": 0.3, "peak_gpu_bytes": 5e10, "first_grad_norms": [0.5, 0.6, 0.7]}
    for folder, seed, measured in ((run, 42, {"exit": 1}), (other, 43, layout)):
        (folder / "bench").mkdir(exist_ok=True)
        (folder / "bench" / "summary.json").write_text(json.dumps(
            {"family": "RLF", "seed": seed, "layouts": {"z3-4": measured}, "complete": measured["exit"] == 0}),
            encoding="utf-8")
    saved = json.loads((other / "report" / "report.json").read_text(encoding="utf-8"))
    saved["systems"]["weights_gate"] = {"passed": True, "checks": {"restored_checksum": True},
                                        "seconds": {"full_load": 4.2, "write_full": 3.1}}
    (other / "report" / "report.json").write_text(json.dumps(saved), encoding="utf-8")
    both = combined([run, other], tmp_path / "combined")
    first, second = (r["identity"][:8] for r in both["runs"])
    test = both["roles"]["test"]
    assert test["sources"]["B"] == first and test["sources"]["RLF_seed43"] == second and test["b_agreement"] == {second: 1.0}
    assert both["decisions"]["RLF-RL"]["seeds"] == 2 and set(test["project"]["conditions"]) == {
        "B", "PTR_seed42", "PTR_seed43", "RL_seed42", "RL_seed43", "RLF_seed42", "RLF_seed43", "RLFa_seed42", "RLFa_seed43"}
    assert both["systems"]["bench"]["seed"] == 43 and both["systems"]["weights_gate"]["passed"]
    text = (tmp_path / "combined" / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# v0.10 合并报告（2 次预约") and "B 在其他机器上重评" in text and "（2 个 seed）" in text
    assert "RLF seed43" in text and "| 0.5000 |" in text and "权重门禁：通过" in text
    with pytest.raises(ValueError, match="distinct"):
        combined([run, run], tmp_path / "again")


def test_pack_cuts_the_final_model_into_checked_parts(fixture_run, tmp_path):
    from exec_feedback.v10.pack import pack, pack_stem, unpack
    from exec_feedback.v10.weights import dir_identity

    run, frozen = fixture_run
    fake_online(run, "RLF", 42)
    exports, home = tmp_path / "exports", tmp_path / "home"
    manifest = pack(run, name="RLF_seed42", exports=exports, part_bytes=4096)
    sizes = [p["bytes"] for p in manifest["parts"]]
    assert manifest["step"] == 3 and len(sizes) >= 3 and set(sizes[:-1]) == {4096} and sum(sizes) == manifest["bytes"]
    assert pack(run, name="RLF_seed42", exports=exports, part_bytes=4096) == manifest  # packed once
    path = exports / f"{pack_stem(frozen['identity'], 'RLF_seed42', 3)}.json"
    folder = unpack(path, home)
    assert folder.name == "RLF_seed42-step0003" and (folder / "model.safetensors").read_bytes() == b"RLF42-3"
    assert dir_identity(folder) == manifest["milestone_identity"] == read_json(
        run / "train" / "RLF_seed42" / "milestones.json")["3"]
    last = exports / manifest["parts"][-1]["file"]  # the end-of-archive padding: extraction alone would not notice
    last.write_bytes(last.read_bytes()[:-1] + b"\x01")
    shutil.rmtree(folder)
    with pytest.raises(ValueError, match="differs from its manifest"):
        unpack(path, home)
    (run / "train" / "RLF_seed42" / "result.json").unlink()
    with pytest.raises(ValueError, match="not finished"):
        pack(run, name="RLF_seed42", exports=tmp_path / "other")


# ------------------------------------------------------------------------------------------------ scheduler
FAKE_CLI = r'''
import json, sys
from pathlib import Path
sys.path[:0] = [{repo!r}, {tests!r}]
import test_v10 as t
from exec_feedback.v10 import analysis
from exec_feedback.v10.evaluate import evaluate
from exec_feedback.v10.gate import gate_weights
from exec_feedback.v10.merge import merge

cmd, args = sys.argv[1], sys.argv[2:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
run = Path(opt("--run"))
if cmd == "merge":
    merge(run, name=opt("--name"))
elif cmd == "gate":
    import exec_feedback.v10.gate as g
    g.MOVED = 1e-4  # the tiny model's layers barely move its outputs (see test_weights_gate_sees_a_load_in_the_outputs)
    report = gate_weights(run, engine_factory=lambda d, s, seed: t.FakeFullEngine(d, s, seed=seed),
                          tokenizer=t.ChatTokenizer(), stop_ids=[2])
    raise SystemExit(0 if report["passed"] else 1)
elif cmd == "run":
    t.fake_online(run, opt("--family"), int(opt("--seed")))
elif cmd == "evaluate":
    evaluate(run, condition=opt("--condition"), role=opt("--role"), stop=opt("--stop"), factory=t.factory)
elif cmd == "pack":
    from exec_feedback.v10.pack import pack
    pack(run, name=opt("--name"), exports=Path(opt("--exports")), part_bytes=4096)
elif cmd == "bench":
    (run / "bench").mkdir(exist_ok=True)
    (run / "bench" / "summary.json").write_text(json.dumps({{"layouts": {{}}, "complete": True}}), encoding="utf-8")
elif cmd == "report":
    analysis.report(run, procs=1, smoke=Path(opt("--smoke-run")) if opt("--smoke-run") else None)
else:
    raise SystemExit("unexpected command " + cmd)
'''


def test_async_steps_are_estimated_from_the_sync_runs_with_sampling_overlapped(fixture_run):
    from exec_feedback.v10.orchestrate import RUN_OVERHEAD, STEP_SECONDS, Plan

    run, frozen = fixture_run
    plan = Plan(run=run, smoke_run=run, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 3600, gpus=list(range(8)), seeds=[42])

    def measured(family: str, rows: list[dict], finished: bool = True) -> None:
        out = run / "train" / f"{family}_seed42"
        out.mkdir(parents=True, exist_ok=True)
        if finished:
            (out / "result.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
        (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    assert plan.step_seconds("RLF") == STEP_SECONDS["RLF"] and plan.step_seconds("RLFa") == STEP_SECONDS["RLFa"]
    measured("RLFa", [{"step": 1, "step_seconds": 99.0}], finished=False)  # a run still going is not a measurement
    measured("RLF", [{"step": 1, "step_seconds": 40.0, "wait_seconds": 25.0},
                     {"step": 2, "step_seconds": 44.0, "wait_seconds": 14.0}])
    assert plan.step_seconds("RLF") == pytest.approx(1.1 * 42.0)
    assert plan.step_seconds("RLFa") == pytest.approx(1.1 * 27.5)  # mean of max(25, 15) and max(14, 30)
    measured("RLFa", [{"step": 1, "step_seconds": 20.0}])
    assert plan.step_seconds("RLFa") == pytest.approx(22.0)
    assert plan.run_estimate("RLF_seed42") == pytest.approx(3 * 1.1 * 42.0 + RUN_OVERHEAD)
    ckpt = run / "train" / "RLF_seed42" / "ckpt.old"  # a retry after a crash resumes from the newest checkpoint
    ckpt.mkdir()
    (ckpt / "state.json").write_text(json.dumps({"step": 2}), encoding="utf-8")
    assert plan.resume_step("RLF_seed42") == 2 and plan.run_estimate("RLF_seed42") == pytest.approx(
        1 * 1.1 * 42.0 + RUN_OVERHEAD)


def test_one_booking_runs_every_sync_seed_before_the_async_ones_and_reserves_more_per_seed(fixture_run, tmp_path):
    from exec_feedback.v10.orchestrate import FINAL_RESERVE, SEED_RESERVE, Plan

    smoke, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["settings"]["seeds"] = [42, 43, 44]
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    stop = time.time() + 13 * 3600
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=stop, gpus=list(range(8)), seeds=[42, 43, 44])
    # a late async run must never take the place of a later seed's sync run (the main comparison)
    order = ["RLF_seed42", "RLF_seed43", "RLF_seed44", "RLFa_seed42", "RLFa_seed43", "RLFa_seed44"]
    assert plan.online("M") == order and plan.online("S") == ["RLF_seed42", "RLFa_seed42"]
    runs = sorted((j for j in plan()[0] if j.name.startswith("M:run-")), key=lambda j: j.priority)
    assert [j.name for j in runs] == [f"M:run-{n}" for n in order]
    # three seeds evaluate four more conditions per extra seed after the last run
    assert plan.until == pytest.approx(stop - FINAL_RESERVE - 2 * SEED_RESERVE)
    single = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                  stop_epoch=stop, gpus=list(range(8)), seeds=[43])
    assert single.until == pytest.approx(stop - FINAL_RESERVE)
    # a run lasts the sum of its steps: the mean is measured, not the median a long sampling tail hides
    out = main / "train" / "RLF_seed42"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
    rows = [{"step": i + 1, "step_seconds": t, "wait_seconds": w} for i, (t, w) in enumerate([(30, 20), (30, 20), (90, 80)])]
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert plan.step_seconds("RLF") == pytest.approx(1.1 * 50.0)
    assert plan.step_seconds("RLFa") == pytest.approx(1.1 * 40.0)  # mean of max(20, 10), max(20, 10), max(80, 10)


def test_a_main_run_that_no_longer_fits_is_noted_once_in_status(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v10.orchestrate import SCHEDULER_RESERVE, Plan

    run, frozen = fixture_run
    stop = time.time() + 50 * 60  # `until` is 10 minutes away, a main run needs more
    plan = Plan(run=run, smoke_run=run, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=stop, gpus=list(range(8)), seeds=[42])
    status = tmp_path / "STATUS.txt"
    plan.scheduler = Scheduler(gpus=list(range(8)), stop=stop, status=status, log_dir=tmp_path / "logs", plan=plan,
                               reserve=SCHEDULER_RESERVE)
    fits = plan.fits("RLF_seed42")
    # while another main run trains its steps are not in the estimate yet: no note (the monitor alerts on it)
    plan.scheduler.running[-1] = {"name": "M:run-RLFa_seed42"}
    assert not fits() and "not starting" not in (status.read_text(encoding="utf-8") if status.exists() else "")
    del plan.scheduler.running[-1]
    assert not fits() and not fits()
    # its dependencies are never all met, so the scheduler itself would never say so
    assert status.read_text(encoding="utf-8").count("not starting M:run-RLF_seed42:") == 1


def test_a_failed_gate_settles_the_main_runs_at_once(fixture_run):
    from exec_feedback.v10.orchestrate import Plan

    run, frozen = fixture_run
    plan = Plan(run=run, smoke_run=run, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 9 * 3600, gpus=list(range(8)), seeds=[42])
    assert not plan.blocked() and not plan.settled()  # the main runs fit and may still start
    (run / "gate").mkdir(exist_ok=True)
    (run / "gate" / "weights.json").write_text(json.dumps({"passed": False}), encoding="utf-8")
    assert plan.blocked() and plan.settled()  # B, PTR and RL are evaluated now, not after `until`


def test_the_report_waits_for_the_benchmark_only_while_it_can_still_start(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v10.orchestrate import SCHEDULER_RESERVE, Plan

    run, frozen = fixture_run
    # `until` is 10 minutes away: a main run that has not finished can no longer start
    plan = Plan(run=run, smoke_run=run, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 50 * 60, gpus=list(range(8)), seeds=[42])
    for family in ("RLF", "RLFa"):
        fake_online(run, family, 42)
    for role in ("dev", "validation", "test"):
        for condition in ("B", "PTR_seed42", "RL_seed42", "RLF_seed42", "RLFa_seed42"):
            (run / "eval" / role / condition).mkdir(parents=True, exist_ok=True)
            (run / "eval" / role / condition / "summary.json").write_text("{}", encoding="utf-8")
    plan.scheduler = Scheduler(gpus=list(range(8)), stop=time.time() + 9 * 3600, status=tmp_path / "STATUS.txt",
                               log_dir=tmp_path / "logs", plan=plan, reserve=SCHEDULER_RESERVE)
    jobs = {j.name: j for j in plan()[0]}
    bench, report = jobs["M:bench"], jobs["M:report"]
    assert bench.deps() and not report.deps()  # evaluations done, the benchmark still fits: the report waits for it
    plan.scheduler.stop = time.time() + 30 * 60  # 25 minutes of benchmark no longer end before stop - 15 min
    assert plan.late(bench) and report.deps()
    plan.scheduler.stop = time.time() + 9 * 3600
    (run / "train" / "RLF_seed42" / "result.json").unlink()  # the sync run never finished: nothing to replay
    assert plan.settled() and not bench.deps() and report.deps()


def test_the_report_waits_neither_for_a_run_that_never_trained_nor_for_a_benchmark_it_could_not_follow(fixture_run,
                                                                                                        tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v10.orchestrate import SCHEDULER_RESERVE, Plan

    run, frozen = fixture_run
    # `until` is 10 minutes away: the async run, never started, no longer fits (the expected end of a 13-hour booking)
    plan = Plan(run=run, smoke_run=run, models_root=Path(frozen["models_root"]), main_py="py", vllm_py="vpy",
                stop_epoch=time.time() + 50 * 60, gpus=list(range(8)), seeds=[42])
    fake_online(run, "RLF", 42)
    for role in ("dev", "validation", "test"):
        for condition in ("B", "PTR_seed42", "RL_seed42", "RLF_seed42"):
            (run / "eval" / role / condition).mkdir(parents=True, exist_ok=True)
            (run / "eval" / role / condition / "summary.json").write_text("{}", encoding="utf-8")
    plan.scheduler = Scheduler(gpus=list(range(8)), stop=time.time() + 9 * 3600, status=tmp_path / "STATUS.txt",
                               log_dir=tmp_path / "logs", plan=plan, reserve=SCHEDULER_RESERVE)
    jobs = {j.name: j for j in plan()[0]}
    assert plan.settled() and not any(name.startswith("M:eval-RLFa_seed42") for name in jobs)
    bench, report = jobs["M:bench"], jobs["M:report"]
    assert bench.deps() and not report.deps()  # evaluations done, benchmark and report still fit: the report waits
    # 45 minutes before the stop the benchmark alone (25 min) would end before stop - 15 min, the report after it not
    plan.scheduler.stop = time.time() + 45 * 60
    assert plan.late(bench) and report.deps()


def test_scheduler_runs_merges_gate_smoke_then_main_plan_end_to_end(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v10.orchestrate import Plan

    smoke, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["smoke_limit"] = 0  # the copy plays the main run, on all eight GPUs
    manifest["settings"].update(trainer_gpus=4, rollout_gpus=4)
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    stop = time.time() + 9 * 3600
    gpus = list(range(8))
    exports = tmp_path / "exports"
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py=sys.executable,
                vllm_py=sys.executable, stop_epoch=stop, gpus=gpus, seeds=[42], exports=exports)
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=gpus, stop=stop, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs", plan=plan,
                          poll=0.1, stagger=0, retry_delay=0, milestone=reached.append)
    plan.scheduler = scheduler
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert " exit 1" not in status, status
    assert (smoke / "report" / "report.json").exists() and (main / "report" / "report.json").exists()
    for role in ("dev", "validation", "test"):
        for condition in ("B", "PTR_seed42", "RL_seed42", "RLF_seed42", "RLFa_seed42"):
            assert (main / "eval" / role / condition / "summary.json").exists(), (role, condition)
    # A job may start as soon as the artifact it waits for exists, before the scheduler reaps the job that wrote it,
    # so orders are checked between starts, and against "done" only where GPUs force it.
    first = lambda text: status.index(text)
    assert first("start M:merge-PTR_seed42") < first("start S:gate-weights") < first("start S:run-RLF_seed42")
    record = Path(frozen["models_root"]) / "merged" / "PTR_seed42.json"
    assert record.stat().st_mtime <= (smoke / "gate" / "weights.json").stat().st_mtime
    # the smoke pair side by side on two blocks (3 GPUs wide here), the main runs on all eight only after both
    assert re.search(r"start S:run-RLF_seed42 on GPUs 0,1,2,", status) and \
        re.search(r"start S:run-RLFa_seed42 on GPUs 3,4,5,", status)
    assert first("start S:run-RLFa_seed42") < first("done S:run-RLF_seed42")
    assert re.search(r"start M:run-RLF_seed42 on GPUs 0,1,2,3,4,5,6,7,", status)
    assert first("start M:run-RLF_seed42") > first("start S:run-RLFa_seed42")
    assert first("start M:run-RLFa_seed42") > first("done M:run-RLF_seed42")
    assert first("start M:eval-RLF_seed42-test") > first("done M:run-RLFa_seed42")
    assert first("start M:bench") > first("done M:eval-RLF_seed42-test") and first("start M:report") > first("start M:bench")
    assert {"M:merge", "S:gate", "S:run", "S:eval", "S:report", "M:run", "M:eval", "M:bench", "M:report"} <= set(reached)
    # the smoke's evaluations wait for the main runs: they are checks nobody acts on during the booking
    assert first("start S:eval-") > first("done M:run-RLFa_seed42") and "S:bench" not in status
    # each main run's model at K is packed as soon as that run ends (the sync run's before the async run starts)
    from exec_feedback.v10.pack import pack_stem
    assert all((exports / f"{pack_stem(manifest['identity'], n, 3)}.json").exists() for n in ("RLF_seed42", "RLFa_seed42"))
    assert first("start M:pack-RLF_seed42") < first("start M:run-RLFa_seed42") and "S:pack" not in status


def test_main_runs_that_never_fit_still_end_in_the_report_inside_the_schedule(fixture_run, tmp_path):
    from exec_feedback.v8.orchestrate import Scheduler
    from exec_feedback.v10.orchestrate import SCHEDULER_RESERVE, Plan

    smoke, frozen = fixture_run
    main = tmp_path / "main"
    shutil.copytree(smoke, main)
    manifest = json.loads((main / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("identity")
    manifest["smoke_limit"] = 0
    manifest["settings"].update(trainer_gpus=4, rollout_gpus=4)
    manifest["identity"] = fingerprint(manifest)
    (main / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI.format(repo=str(REPO), tests=str(REPO / "tests")), encoding="utf-8")
    stop = time.time() + 50 * 60  # `until` is 10 minutes away: no main run fits, the evaluations and the report do
    gpus = list(range(8))
    plan = Plan(run=main, smoke_run=smoke, models_root=Path(frozen["models_root"]), main_py=sys.executable,
                vllm_py=sys.executable, stop_epoch=stop, gpus=gpus, seeds=[42], exports=tmp_path / "exports")
    plan.py = lambda venv, *a: [sys.executable, str(cli), *a]
    reached: list[str] = []
    scheduler = Scheduler(gpus=gpus, stop=stop, status=tmp_path / "STATUS.txt", log_dir=tmp_path / "logs", plan=plan,
                          reserve=SCHEDULER_RESERVE, poll=0.1, stagger=0, retry_delay=0, milestone=reached.append)
    plan.scheduler = scheduler
    scheduler.loop()
    status = (tmp_path / "STATUS.txt").read_text(encoding="utf-8")
    assert "not starting M:run-RLF_seed42:" in status and "start M:run-" not in status
    # the report runs inside the schedule (no fallback after it) and the evaluation stage is reached and exported
    assert "start M:report" in status and (main / "report" / "report.json").exists()
    assert "M:eval-RLF_seed42" not in status and "M:eval-RLFa_seed42" not in status
    assert {"M:eval", "M:report"} <= set(reached)


# ------------------------------------------------------------------------------------------------ scripts
EARLY_STUBS = r'''
curl() { : > "${@: -1}"; }
sha256sum() { cat > /dev/null; echo "uv-x86_64-unknown-linux-musl.tar.gz: OK"; }
tar() { mkdir -p "$4/uv-x86_64-unknown-linux-musl"
        printf '#!/bin/sh\n[ "$2" = find ] && echo /fake/python3.12\nexit 0\n' > "$4/uv-x86_64-unknown-linux-musl/uv"; }
'''


def test_early_python_prints_only_the_interpreter_on_a_fresh_host(tmp_path):
    """The real sha256sum prints "<file>: OK" on stdout; captured beside the path, it kept the early downloads from
    ever starting on a new host (v0.8-v0.10)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash")
    script = (REPO / "scripts" / "v10_run.sh").read_text(encoding="utf-8")
    function = re.search(r"^early_python\(\) \{\n.*?^\}\n", script, re.M | re.S).group(0)
    probe = EARLY_STUBS + function + 'PY3=$(early_python 2> /dev/null); printf "%s" "$PY3"\n'
    done = subprocess.run([bash, "-c", probe], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "/fake/python3.12"
    assert (tmp_path / ".runtime" / "bin" / "uv").exists()


# ------------------------------------------------------------------------------------------------ launcher
FAKE_TRAINER = r'''
import json, sys, time
from pathlib import Path
root, out, code, mode, finished, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], \
    sys.argv[5] == "finished", int(sys.argv[6])
if mode == "stop":
    (root / "requests").mkdir(parents=True, exist_ok=True)
    (root / "requests" / "000000.json").write_text(json.dumps({"seq": 0, "kind": "stop"}), encoding="utf-8")
    end = time.time() + 30  # still writing its result once every worker has gone
    while time.time() < end and len(list(root.glob("left-*"))) < workers:
        time.sleep(0.05)
    time.sleep(0.5)
else:
    time.sleep(30)  # waiting on workers that will never answer
if finished:
    (out / "result.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
raise SystemExit(code)
'''
FAKE_WORKER = r'''
import sys, time
from pathlib import Path
root, worker, early = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "early"
end = time.time() + 60
while not early and time.time() < end and not list((root / "requests").glob("*.json")):
    time.sleep(0.05)
(root / f"left-{worker}").write_text("", encoding="utf-8")
'''


@pytest.mark.parametrize("trainer, first_worker, code, reason", [
    # the normal end: every worker leaves on the stop request while the trainer is still writing its result
    (("0", "stop", "finished"), "stop", 0, "trainer exited 0 after the workers left on its stop request"),
    # failing after the stop request still fails
    (("3", "stop", "unfinished"), "stop", 3, "trainer exited 3 after the workers left on its stop request"),
    # left unfinished with 0 (the smoke's planned pause, a deadline): a try the scheduler counts and resumes later
    (("0", "stop", "unfinished"), "stop", "resumable",
     "trainer exited 0 after the workers left on its stop request; stopped before its planned steps"),
    # a worker gone without a stop request
    (("0", "none", "unfinished"), "early", 1, "rollout worker 0 exited 0"),
])
def test_launcher_takes_workers_leaving_after_the_stop_request_as_the_normal_end(fixture_run, tmp_path, monkeypatch,
                                                                                 trainer, first_worker, code, reason):
    from exec_feedback.v10 import launch as L
    from exec_feedback.v10.merge import merge

    run, _ = fixture_run
    merge(run, name="PTR_seed42")
    out = run / "train" / "RLF_seed42"
    fake_trainer, fake_worker = tmp_path / "trainer.py", tmp_path / "worker.py"
    fake_trainer.write_text(FAKE_TRAINER, encoding="utf-8")
    fake_worker.write_text(FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(L, "POLL", 0.05)
    monkeypatch.setattr(L, "WORKER_GRACE", 5.0)
    monkeypatch.setattr(L, "commands", lambda tp, wp, common, root, tg, rg, until: (
        [[sys.executable, str(fake_worker), str(root), str(w), first_worker if w == 0 else "stop"]
         for w in range(len(rg))],
        [sys.executable, str(fake_trainer), str(root), str(out), *trainer, str(len(rg))]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")  # one trainer GPU, two rollout workers
    rc = L.launch(run, family="RLF", seed=42, until=None, trainer_python=sys.executable, worker_python=sys.executable,
                  log_dir=tmp_path / "logs")
    record = list(recover_records(out / "launches.jsonl"))[-1]
    assert L.RESUMABLE == 75  # the booking monitor takes exit 75 for a resumable stop, not a failure
    code = L.RESUMABLE if code == "resumable" else code
    assert (rc, record["code"], record["reason"]) == (code, code, reason)
