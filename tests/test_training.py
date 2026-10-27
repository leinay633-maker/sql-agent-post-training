import argparse
import copy
import json
from pathlib import Path

import pytest

from exec_feedback.cli import make_fixture
from exec_feedback.core import write_jsonl
from exec_feedback.pipeline import build_sft
from exec_feedback.train import collate, tokenize_row, train, validate_rows


class TinyTokenizer:
    """Test double for boundary/mask logic, NOT the Qwen tokenizer."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text]


@pytest.fixture
def row(tmp_path):
    root = tmp_path / "db"
    task = make_fixture(root)
    return build_sft([task], root, "success", allow_fixture=True)[0]


def test_loss_mask_and_completion(row):
    tok = TinyTokenizer()
    encoded = tokenize_row(row, tok, 10000)
    prefix = tok.apply_chat_template(row["prompt"], True, True)
    assert encoded["labels"][:len(prefix)] == [-100] * len(prefix)
    assert encoded["labels"][len(prefix):] == encoded["input_ids"][len(prefix):]
    assert any(t != -100 for t in encoded["labels"])


def test_no_silent_truncation(row):
    with pytest.raises(ValueError, match="Overlength"):
        tokenize_row(row, TinyTokenizer(), 10)


def test_template_boundary_checked(row):
    class Broken(TinyTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            values = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return values if add_generation_prompt else [0] + values
    with pytest.raises(ValueError, match="boundary"):
        tokenize_row(row, Broken(), 10000)


def test_padding_not_supervised():
    pytest.importorskip("torch")
    batch = collate([{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                     {"input_ids": [3], "attention_mask": [1], "labels": [3]}], 0)
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_fixture_and_holdout_train_gates(row):
    with pytest.raises(ValueError, match="Fixture"):
        validate_rows([row])
    heldout = {**row, "split": "test"}
    with pytest.raises(ValueError, match="held-out"):
        validate_rows([heldout], True)
    renamed = {**row, "source": "bird-mini-dev"}
    with pytest.raises(ValueError, match="held-out"):
        validate_rows([renamed], True)


def test_plan_is_offline_and_unmeasured(row, tmp_path):
    data = tmp_path / "sft.jsonl"
    write_jsonl(data, [row])
    config = Path(__file__).resolve().parents[1] / "configs" / "qlora-smoke.json"
    args = argparse.Namespace(data=data, config=config, out=tmp_path / "out", execute=False,
                              allow_download=False, allow_fixture=True)
    plan = train(args)
    assert plan["mode"] == "plan_only"
    assert not plan["weights_downloaded"] and not plan["training_started"]
    assert not args.out.exists()
