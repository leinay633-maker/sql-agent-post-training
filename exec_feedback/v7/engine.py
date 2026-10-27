"""vLLM engines for the three v0.7 models (same chat template, context rule and Generator interface as v0.6)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..v6.engine import VLLMGenerator


def stop_ids(spec: dict[str, Any], tokenizer: Any) -> list[int]:
    """Every eos the model card's generation_config lists (Qwen3: <|im_end|> and <|endoftext|>)."""
    ids = {tokenizer.eos_token_id}
    config = Path(spec["local_path"]) / "generation_config.json"
    if config.exists():
        eos = json.loads(config.read_text(encoding="utf-8")).get("eos_token_id")
        ids.update(eos if isinstance(eos, list) else [eos] if isinstance(eos, int) else [])
    return sorted(i for i in ids if isinstance(i, int) and i >= 0)


def make_engine(spec: dict[str, Any], settings: dict[str, Any], *, mem: float, seed: int, lora: bool,
                max_num_seqs: int) -> tuple[Any, Any]:
    from vllm import LLM
    from ..models import load_tokenizer

    kw = dict(model=spec["local_path"], tokenizer=spec["local_path"], dtype="bfloat16",
              max_model_len=settings["max_context"], gpu_memory_utilization=mem, enable_prefix_caching=True,
              seed=seed, generation_config="vllm", max_num_seqs=max_num_seqs, trust_remote_code=False)
    if lora:
        kw.update(enable_lora=True, max_lora_rank=settings["lora_r"], max_loras=1)
    return LLM(**kw), load_tokenizer(spec)


def generator(llm: Any, tokenizer: Any, spec: dict[str, Any], settings: dict[str, Any], *,
              sampling: dict[str, Any] | None, lora: Any) -> VLLMGenerator:
    gen = VLLMGenerator(llm, tokenizer, max_context=settings["max_context"], max_tokens=settings["max_tokens"],
                        sampling=sampling, lora=lora)
    gen.stop_ids = stop_ids(spec, tokenizer)
    gen.metadata["stop_ids"] = gen.stop_ids
    return gen
