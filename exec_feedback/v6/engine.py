"""vLLM generation behind the exec_feedback Generator interface (same chat template, context rule and EOS)."""
from __future__ import annotations

import time
from typing import Any


class VLLMGenerator:
    kind = "model"

    def __init__(self, llm: Any, tokenizer: Any, *, max_context: int, max_tokens: int,
                 sampling: dict[str, Any] | None = None, lora: Any = None) -> None:
        self.llm, self.tokenizer, self.lora = llm, tokenizer, lora
        self.max_context, self.max_tokens, self.sampling = max_context, max_tokens, sampling
        self.eos = tokenizer.eos_token_id
        self.stop_ids = sorted({self.eos, *[i for i in (tokenizer.convert_tokens_to_ids("<|im_end|>"),)
                                            if isinstance(i, int) and i >= 0]})
        self.metadata = {"engine": "vllm", "calls": 0, "generate_batches": 0, "generate_seconds": 0.0,
                         "prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0,
                         "length_stops": 0, "context_rejections": 0, "max_batch": 0}

    def params(self) -> Any:
        from vllm import SamplingParams

        common = {"n": 1, "max_tokens": self.max_tokens, "stop_token_ids": self.stop_ids, "skip_special_tokens": True}
        if not self.sampling:
            return SamplingParams(temperature=0.0, **common)
        extra = {"top_k": self.sampling["top_k"]} if self.sampling.get("top_k") else {}
        return SamplingParams(temperature=self.sampling["temperature"], top_p=self.sampling["top_p"], **extra, **common)

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[Any]:
        from vllm.inputs import TokensPrompt

        results: list[Any] = [None] * len(conversations)
        prompts, index = [], []
        for i, messages in enumerate(conversations):
            ids = list(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            if not ids or len(ids) + self.max_tokens > self.max_context:
                self.metadata["context_rejections"] += 1
                results[i] = ValueError("Context budget exceeded; no silent truncation")
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            index.append((i, len(ids)))
        if not prompts:
            return results
        started = time.perf_counter()
        outputs = self.llm.generate(prompts, self.params(), use_tqdm=False, lora_request=self.lora)
        self.metadata["generate_seconds"] += time.perf_counter() - started
        self.metadata["generate_batches"] += 1
        self.metadata["max_batch"] = max(self.metadata["max_batch"], len(prompts))
        for (i, plen), out in zip(index, outputs):
            choice = out.outputs[0]
            tokens = list(choice.token_ids)
            stopped = choice.finish_reason == "stop"
            completion = len(tokens) + (1 if stopped and (not tokens or tokens[-1] not in self.stop_ids) else 0)
            self.metadata["calls"] += 1
            self.metadata["prompt_tokens"] += plen
            self.metadata["cached_prompt_tokens"] += getattr(out, "num_cached_tokens", 0) or 0
            self.metadata["completion_tokens"] += completion
            self.metadata["length_stops"] += not stopped
            results[i] = {"text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                          "usage": {"prompt_tokens": plen, "completion_tokens": completion},
                          "finish_reason": "stop" if stopped else "length"}
        return results


def make_engine(frozen: dict[str, Any], *, mem: float, seed: int, lora: bool, max_num_seqs: int = 256) -> tuple[Any, Any]:
    from vllm import LLM
    from ..models import load_tokenizer

    spec = frozen["model"]
    kw = dict(model=spec["local_path"], tokenizer=spec["local_path"], dtype="bfloat16",
              max_model_len=frozen["settings"]["max_context"], gpu_memory_utilization=mem,
              enable_prefix_caching=True, seed=seed, generation_config="vllm", max_num_seqs=max_num_seqs,
              trust_remote_code=False)
    if lora:
        kw.update(enable_lora=True, max_lora_rank=frozen["settings"]["lora_r"], max_loras=1)
    return LLM(**kw), load_tokenizer(spec)


def lora_request(name: str, adapter: Any) -> Any:
    from vllm.lora.request import LoRARequest
    return LoRARequest(name, 1, str(adapter))
