"""vLLM engines for the v0.8 models: the 4B on one GPU (or half of one), the 235B FP8 teacher tensor-parallel over
`teacher_gpus` whole GPUs. Same Generator interface, context rule and stop tokens as v0.6/v0.7."""
from __future__ import annotations

import os
from typing import Any

from ..v7.engine import generator, stop_ids  # noqa: F401  (re-exported for v0.8 callers)


def make_engine(spec: dict[str, Any], settings: dict[str, Any], kind: str, *, mem: float, seed: int, lora: bool,
                max_num_seqs: int) -> tuple[Any, Any]:
    from vllm import LLM
    from ..models import load_tokenizer

    kw = dict(model=spec["local_path"], tokenizer=spec["local_path"], dtype="bfloat16",
              max_model_len=settings["max_context"], gpu_memory_utilization=mem, enable_prefix_caching=True,
              seed=seed, generation_config="vllm", max_num_seqs=max_num_seqs, trust_remote_code=False)
    if kind == "teacher":
        visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        if len(visible) != settings["teacher_gpus"]:
            raise RuntimeError(f"The teacher needs exactly {settings['teacher_gpus']} visible GPUs, got {visible}")
        kw["tensor_parallel_size"] = settings["teacher_gpus"]
        # 2026-10-12 H20: the FULL decode-graph capture ran out of memory at 13/35 sizes after the piecewise capture
        # had fit, so the teacher captures piecewise graphs only
        kw["compilation_config"] = {"cudagraph_mode": "PIECEWISE"}
    if lora:
        kw.update(enable_lora=True, max_lora_rank=settings["lora_r"], max_loras=1)
    return LLM(**kw), load_tokenizer(spec)
