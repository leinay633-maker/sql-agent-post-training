"""CPU provenance/phase gates; dependency doubles never stand for model scores."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from exec_feedback.rl import PROTOCOL
from exec_feedback.rl.config import DEFAULT
from exec_feedback.rl.evaluation import frozen_adapters
from exec_feedback.rl.study import load_run
from exec_feedback.storage import atomic_json, code_fingerprint, fingerprint


def registry(monkeypatch):
    # Gate tests need content identities, not a pretrained model download.
    identity = lambda spec, adapter=None: {"adapter_hash": "fixed-adapter"}
    monkeypatch.setitem(sys.modules, "exec_feedback.models",
                        SimpleNamespace(model_identity=identity))


def frozen():
    return {"identity": "run", "protocol": PROTOCOL,
            "settings": copy.deepcopy(DEFAULT), "model": {},
            "warm_adapter": "warm", "warm_adapter_hash": "fixed-adapter"}


def write_seed(run, seed, config, **changes):
    result = {
        "identity": fingerprint({"run": config["identity"], "seed": seed,
                                 "method": "online-rloo"}),
        "training_completed": True, "adapter_reload_verified": True,
        "adapter_hash": "fixed-adapter",
    }
    atomic_json(run / f"E4_seed_{seed}" / "result.json", {**result, **changes})


def test_all_registered_seeds_required_before_validation(tmp_path, monkeypatch):
    registry(monkeypatch)
    config = frozen()
    write_seed(tmp_path, 42, config)
    write_seed(tmp_path, 43, config)
    with pytest.raises(FileNotFoundError):
        frozen_adapters(tmp_path, config)
    write_seed(tmp_path, 44, config)
    assert set(frozen_adapters(tmp_path, config)) == {
        "S4", "E4_seed_42", "E4_seed_43", "E4_seed_44"}


@pytest.mark.parametrize("change", [
    {"training_completed": False}, {"adapter_reload_verified": False},
    {"identity": "different-run"}, {"adapter_hash": "changed"},
])
def test_unverified_or_foreign_rl_checkpoint_blocks_eval(tmp_path, monkeypatch, change):
    registry(monkeypatch)
    config = frozen()
    for seed in config["settings"]["seeds"]:
        write_seed(tmp_path, seed, config, **(change if seed == 43 else {}))
    with pytest.raises(ValueError):
        frozen_adapters(tmp_path, config)


def test_pilot_has_no_validation_entry(tmp_path, monkeypatch):
    registry(monkeypatch)
    config = frozen()
    config["settings"]["phase"] = "pilot"
    with pytest.raises(ValueError, match="Pilot"):
        frozen_adapters(tmp_path, config)


def test_frozen_manifest_mutation_rejected(tmp_path, monkeypatch):
    registry(monkeypatch)
    config = frozen()
    atomic_json(tmp_path / "manifest.json", config)
    with pytest.raises(ValueError, match="manifest"):
        load_run(tmp_path, training=True)


def test_real_hf_peft_cpu_gradient_and_saved_adapter(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from exec_feedback.rl.backend import LocalPolicy
    from exec_feedback.rl.objective import backward_batch

    config = transformers.Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, eos_token_id=2,
        pad_token_id=0, attention_dropout=0.0)
    torch.manual_seed(42)
    base = transformers.Qwen3ForCausalLM(config)
    model = peft.get_peft_model(base, peft.LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"]))
    tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=2,
        apply_chat_template=lambda messages, **kw: [1, 3, 4],
        decode=lambda ids, **kw: str(ids))
    policy = LocalPolicy(model, tokenizer, batch_size=2, max_context=64,
                         max_tokens=8, sample=True)
    sampled = policy.generate_batch([[{"role": "user", "content": "fixture"}]] * 2)
    assert all(row["completion_ids"] for row in sampled)
    # Artificial 1/0 rewards test mechanics only, NOT Qwen4B/BIRD accuracy.
    rows = [{"reward": float(i == 0), "sampled_turns": [row]}
            for i, row in enumerate(sampled)]
    model.train()
    info = backward_batch(model, rows, 2)
    assert info["has_gradient"]
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    model.eval()
    probe = torch.tensor([[1, 3, 4]])
    with torch.inference_mode():
        expected = model(probe).logits
    model.save_pretrained(tmp_path / "adapter", safe_serialization=True)
    torch.manual_seed(42)
    reloaded = peft.PeftModel.from_pretrained(
        transformers.Qwen3ForCausalLM(config), tmp_path / "adapter")
    reloaded.eval()
    with torch.inference_mode():
        actual = reloaded(probe).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_h20_completed_pilot_evidence():
    path = os.environ.get("EFA_V04_PILOT_RUN")
    if not path:
        pytest.skip("Run H20 V4-2/V4-3 first; set EFA_V04_PILOT_RUN to audit its evidence")
    run = Path(path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == PROTOCOL
    assert manifest["settings"]["phase"] == "pilot"
    assert manifest["settings"]["validation_tasks"] == 0
    result = json.loads((run / "E4_seed_42/result.json").read_text(encoding="utf-8"))
    assert result["training_completed"] and result["adapter_reload_verified"]
    assert result["peak_allocated_bytes"] > 0
    assert result["durable_training"]["completed"]
    timing = json.loads((run / "throughput/comparison.json").read_text(encoding="utf-8"))
    assert timing["gold_used"] is False
    assert timing["run_identity"] == manifest["identity"]
    for row in timing["runs"].values():
        assert row["generate_seconds"] > 0
        assert row["questions"] == manifest["settings"]["train_tasks"]


def report_fixture(tmp_path, monkeypatch, *, sealed=True, change=None):
    from exec_feedback.rl import reporting
    from exec_feedback.core import digest, write_jsonl

    tasks = [SimpleNamespace(id=str(i), db_id=f"db{i // 2}", fixture=False)
             for i in range(4)]
    config = frozen()
    config["warm_corpus_origins"] = {"rule_perturbation": 2}
    monkeypatch.setattr(reporting, "load_run", lambda *a, **kw: (config, tasks))
    names = ["S4", "E4_seed_42", "E4_seed_43", "E4_seed_44"]
    for name in names:
        root = tmp_path / f"eval_{name}"
        meta = {"protocol": PROTOCOL, "run_identity": "run", "condition": name,
                "task_ids": [t.id for t in tasks], "adapter_hash": "fixed-adapter",
                "backend": "hf-v04-bucketed", "batch": 16, "slots": 64,
                "max_steps": 2, "max_tokens": 384, "greedy": True}
        if name == "E4_seed_43":
            meta.update(change or {})
        identity = fingerprint(meta)
        atomic_json(root / "manifest.json", {**meta, "identity": identity})
        correct = name != "S4"
        rows = [{"task_id": t.id, "db_id": t.db_id, "correct": correct,
                 "status": "correct" if correct else "incorrect"} for t in tasks]
        write_jsonl(root / "episodes.jsonl", rows)
        atomic_json(root / "summary.json", {})
        if sealed:
            atomic_json(root / "done.json", {"identity": identity, "files": {
                file: digest(root / file) for file in ("episodes.jsonl", "summary.json")}})
        if name != "S4":
            atomic_json(tmp_path / name / "result.json", {
                "adapter_hash": "fixed-adapter", "run_identity": "run"})
    return reporting


def test_report_complete_requires_sealed_conditions(tmp_path, monkeypatch):
    module = report_fixture(tmp_path, monkeypatch, sealed=False)
    result = module.report(tmp_path, tmp_path / "report")
    assert result["mean_delta"] == 1  # fixture for aggregation, not research data
    assert result["complete"] is False
    assert result["paired_database_seed_bootstrap_95"] is None


def test_sealed_paired_report_keeps_all_registered_seeds(tmp_path, monkeypatch):
    module = report_fixture(tmp_path, monkeypatch)
    result = module.report(tmp_path, tmp_path / "report")
    assert result["complete"]
    assert result["seeds"] == 3
    assert result["tasks_per_condition"] == 4
    assert len(result["sealed_conditions"]) == 4


@pytest.mark.parametrize("change", [
    {"protocol": "efa-v3"}, {"max_tokens": 512}, {"adapter_hash": "other"},
])
def test_report_blocks_protocol_budget_and_adapter_mixing(tmp_path, monkeypatch, change):
    module = report_fixture(tmp_path, monkeypatch, change=change)
    with pytest.raises(ValueError):
        module.report(tmp_path, tmp_path / "report")


def test_report_rejects_modified_completion_identity(tmp_path, monkeypatch):
    module = report_fixture(tmp_path, monkeypatch)
    path = tmp_path / "eval_S4/done.json"
    data = json.loads(path.read_text())
    atomic_json(path, {**data, "identity": "other"})
    with pytest.raises(ValueError, match="identity"):
        module.report(tmp_path, tmp_path / "report")
