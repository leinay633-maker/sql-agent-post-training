"""Real local Transformers/PEFT and explicit HTTP policies; no fake fallback."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

from .core import digest
from .pipeline import HTTPPolicy
from .storage import CallBudget, fingerprint, read_json


def model_identity(spec: dict[str, Any], adapter: Path | None = None) -> dict[str, Any]:
    identity = {k: spec.get(k) for k in ("backend", "id", "revision", "device", "quantization",
                                        "max_context_tokens", "max_tokens", "temperature", "seed", "batch_size")}
    if spec.get("chat_template"):  # v0.7 hybrid models; absent keeps every earlier identity unchanged
        identity["chat_template"] = hashlib.sha256(spec["chat_template"].encode()).hexdigest()
    if spec.get("local_path"):
        root = Path(spec["local_path"])
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".txt"})
        if not files:
            raise ValueError("No model/tokenizer files in local_path")
        identity["local_snapshot"] = fingerprint({p.relative_to(root).as_posix(): digest(p) for p in files})
    if spec["backend"] == "http":
        identity["endpoint"] = spec["endpoint"]
        identity["weights_verified"] = False
    if adapter:
        files = sorted(p for p in adapter.rglob("*") if p.is_file())
        if not (adapter / "adapter_config.json").exists() or not files:
            raise ValueError(f"Not a saved PEFT adapter: {adapter}")
        identity["adapter_hash"] = fingerprint({p.relative_to(adapter).as_posix(): digest(p) for p in files})
    return identity


def load_tokenizer(spec: dict[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the ml extra: python -m pip install -e '.[ml]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(spec.get("local_path") or spec["id"],
                    revision=spec.get("revision"), local_files_only=not spec.get("allow_download", False),
                    trust_remote_code=False)
    if spec.get("chat_template"):
        tokenizer.chat_template = spec["chat_template"]
    if not tokenizer.chat_template:
        raise ValueError("The selected tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(spec: dict[str, Any], adapter: Path | None = None, training: bool = False) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the ml extra on the selected compute host") from exc
    cuda = spec["device"] == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback or cloud rental")
    if cuda and torch.cuda.device_count() != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    tokenizer = load_tokenizer(spec)
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16 if cuda else torch.float32
    kwargs: dict[str, Any] = {"revision": spec.get("revision"), "trust_remote_code": False,
                             "local_files_only": not spec.get("allow_download", False),
                             "torch_dtype": dtype, "attn_implementation": "sdpa"}
    if cuda:
        # Stream shards onto the assigned GPU instead of first materializing a
        # full BF16 30B model in host RAM. This is not auto sharding/offloading.
        kwargs.update(device_map={"": 0}, low_cpu_mem_usage=True)
    if spec.get("quantization") == "nf4":
        if not cuda:
            raise ValueError("NF4 requires a CUDA host in this project")
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(spec.get("local_path") or spec["id"], **kwargs)
    if not cuda:
        model.to(spec["device"])
    if spec.get("quantization") == "nf4":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training)
    if training:
        if spec.get("quantization") != "nf4":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.config.use_cache = False
    if adapter:
        from peft import PeftConfig, PeftModel
        saved = PeftConfig.from_pretrained(str(adapter), local_files_only=True)
        expected = spec.get("local_path") or spec["id"]
        evidence = adapter / "efa-base.json"
        if evidence.exists():
            recorded, current = read_json(evidence), model_identity(spec)
            if recorded.get("local_snapshot") or current.get("local_snapshot"):
                if recorded.get("local_snapshot") != current.get("local_snapshot"):
                    raise ValueError("Adapter local base snapshot differs from selected model")
            elif any(recorded[k] != current[k] for k in ("id", "revision")):
                raise ValueError("Adapter base model/revision differs from selected model")
        elif saved.base_model_name_or_path != expected:
            raise ValueError("Adapter base model differs from selected model")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=training)
    return model, tokenizer


class TransformersPolicy:
    kind = "model"
    def __init__(self, spec: dict[str, Any], adapter: Path | None = None) -> None:
        self.spec = spec
        identity = model_identity(spec, adapter)
        self.model, self.tokenizer = load_model(spec, adapter)
        self.model.eval()
        self.task_seed = spec.get("seed", 42)
        self.call_index = 0
        self.metadata = {"provider": "transformers", "model": spec["id"], "identity": identity,
                         "weights_verified": True, "calls": 0, "usage": {},
                         "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft")}}

    def set_task(self, task_id: str) -> None:
        self.task_seed = (self.spec.get("seed", 42) + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)) % (2**31)
        self.call_index = 0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        row = self.generate_batch([messages])[0]
        if isinstance(row, Exception):
            raise row
        return row["text"]

    def generate_batch(self, conversations: list[list[dict[str, str]]]) -> list[Any]:
        import torch
        import time
        if len(conversations) > 1 and self.spec["temperature"] != 0:
            raise ValueError("Batched research inference requires greedy decoding for comparable runs")
        encoded_rows, valid, results = [], [], [None] * len(conversations)
        for i, messages in enumerate(conversations):
            # Do not trust a tokenizer's default model_input_names (fast tokenizers
            # can emit token_type_ids although Qwen3.generate does not accept it).
            encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                            return_tensors="pt", return_dict=True, return_token_type_ids=False)
            ids = encoded["input_ids"][0].tolist()
            if len(ids) + self.spec["max_tokens"] > self.spec["max_context_tokens"]:
                results[i] = ValueError("Context budget exceeded; no silent truncation")
                continue
            valid.append(i)
            encoded_rows.append(ids)
        if not valid:
            return results
        width = max(map(len, encoded_rows))
        pad = self.tokenizer.pad_token_id
        inputs = {"input_ids": torch.tensor([[pad] * (width-len(x)) + x for x in encoded_rows], device=self.model.device),
                  "attention_mask": torch.tensor([[0] * (width-len(x)) + [1] * len(x) for x in encoded_rows], device=self.model.device)}
        torch.manual_seed(self.task_seed + self.call_index)
        self.call_index += 1
        options = {"max_new_tokens": self.spec["max_tokens"], "do_sample": self.spec["temperature"] > 0,
                   "pad_token_id": pad, "eos_token_id": self.tokenizer.eos_token_id, "use_cache": True}
        if options["do_sample"]:
            options["temperature"] = self.spec["temperature"]
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        # OOM is intentionally fatal, not a silent change of batch/precision or
        # thousands of misleading provider-error rows. Tune batch size in pilot.
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **options)[:, width:]
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.metadata["calls"] += len(valid)
        self.metadata["generate_batches"] = self.metadata.get("generate_batches", 0) + 1
        self.metadata["generate_seconds"] = self.metadata.get("generate_seconds", 0.0) + elapsed
        for index, ids, tokens in zip(valid, encoded_rows, generated.tolist()):
            eos = self.tokenizer.eos_token_id
            stopped = eos in tokens
            if stopped:
                tokens = tokens[:tokens.index(eos)+1]
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(tokens)}
            for key, value in usage.items():
                self.metadata["usage"][key] = self.metadata["usage"].get(key, 0) + value
            reason = "stop" if stopped else "length"
            self.metadata["last_finish_reason"] = reason
            results[index] = {"text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                              "usage": usage, "finish_reason": reason}
        return results

    def close(self) -> None:
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BudgetedPolicy:
    def __init__(self, policy: Any, budget: CallBudget) -> None:
        self.inner, self.budget = policy, budget
        self.kind, self.metadata = policy.kind, policy.metadata
    @property
    def spec(self) -> dict[str, Any]:
        return self.inner.spec
    def set_task(self, task_id: str) -> None:
        if hasattr(self.inner, "set_task"):
            self.inner.set_task(task_id)
    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.budget.reserve()
        return self.inner(messages)
    @property
    def complete(self) -> Any:
        """The inner policy's per-call usage path, behind the same budget; absent when the inner policy has none."""
        if not hasattr(self.inner, "complete"):
            raise AttributeError("complete")
        def call(messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
            self.budget.reserve()
            return self.inner.complete(messages)
        return call
    def generate_batch(self, messages: list[list[dict[str, str]]]) -> list[Any]:
        if not hasattr(self.inner, "generate_batch"):
            raise ValueError("This provider does not support batched inference")
        self.budget.reserve(len(messages))
        return self.inner.generate_batch(messages)
    def close(self) -> None:
        if hasattr(self.inner, "close"):
            self.inner.close()


def create_policy(spec: dict[str, Any], adapter: Path | None = None, budget: CallBudget | None = None) -> Any:
    if spec["backend"] == "http":
        if adapter:
            raise ValueError("HTTP backend cannot automatically load local adapters; use transformers for the complete study")
        policy = HTTPPolicy(spec["endpoint"], spec["id"], allow_remote=spec.get("allow_remote", False),
                            max_tokens=spec["max_tokens"], timeout=spec.get("timeout", 60),
                            extra_body=spec.get("extra_body"))
        policy.metadata["identity"] = model_identity(spec)
        policy.temperature, policy.seed = spec["temperature"], spec.get("seed", 42)
        policy.metadata["temperature"] = policy.temperature
    else:
        policy = TransformersPolicy(spec, adapter)
    return BudgetedPolicy(policy, budget) if budget else policy
