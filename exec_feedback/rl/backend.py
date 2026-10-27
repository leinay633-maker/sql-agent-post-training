"""A live HF policy: length-bucketed batches and exact sampled token capture.

Uses the same policy weights for sampling and the immediately following update.
This is NOT vLLM or token-level continuous batching. No server/weight sync is
introduced. Performance improvements must be measured against batch 4.
"""
from __future__ import annotations

import time
from typing import Any


class LocalPolicy:
    kind = "model"

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int,
                 max_context: int, max_tokens: int, sample: bool,
                 bucket: bool = True, budget: Any = None) -> None:
        if not 1 <= batch_size <= 64 or max_tokens < 1:
            raise ValueError("Invalid generation batch/output budget")
        self.model, self.tokenizer = model, tokenizer
        self.batch_size, self.max_context = batch_size, max_context
        self.max_tokens, self.sample = max_tokens, sample
        self.bucket, self.budget = bucket, budget
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.metadata = {"provider": "hf-v04-bucketed", "calls": 0,
                         "generate_seconds": 0.0, "generate_batches": 0,
                         "prompt_tokens": 0, "completion_tokens": 0,
                         "padded_prompt_tokens": 0}

    def generate_batch(self, conversations: list[list[dict[str, str]]]) -> list[Any]:
        results: list[Any] = [None] * len(conversations)
        encoded = []
        for index, messages in enumerate(conversations):
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            if not ids or len(ids) + self.max_tokens > self.max_context:
                results[index] = ValueError("Context limit; no silent truncation")
                continue
            encoded.append((index, ids))
        if self.bucket:
            encoded.sort(key=lambda item: (len(item[1]), item[0]))
        for start in range(0, len(encoded), self.batch_size):
            rows = encoded[start:start + self.batch_size]
            if self.budget is not None:
                self.budget.reserve(len(rows))
            self._generate(rows, results)
        return results

    def _generate(self, rows: list[tuple[int, list[int]]], results: list[Any]) -> None:
        import torch
        from transformers import GenerationConfig

        device = next(self.model.parameters()).device
        pad = self.tokenizer.pad_token_id
        eos = self.tokenizer.eos_token_id
        if pad is None or eos is None:
            raise ValueError("Explicit pad and EOS ids are required")
        width = max(len(ids) for _, ids in rows)
        ids = [[pad] * (width - len(row)) + row for _, row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for _, row in rows]
        # Fresh config avoids Qwen's saved top-k/top-p defaults changing the
        # behavior distribution while the loss assumes ordinary softmax.
        config = GenerationConfig(
            max_new_tokens=self.max_tokens, do_sample=self.sample,
            temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
            eos_token_id=eos, pad_token_id=pad, use_cache=True)
        self.model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=torch.tensor(ids, device=device),
                attention_mask=torch.tensor(masks, device=device),
                generation_config=config)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.metadata["generate_seconds"] += elapsed
        self.metadata["generate_batches"] += 1
        self.metadata["padded_prompt_tokens"] += width * len(rows)
        stops = {eos} if isinstance(eos, int) else set(eos)
        for (index, prompt), tail in zip(rows, output[:, width:].tolist()):
            stop = next((i for i, token in enumerate(tail) if token in stops), None)
            completion = tail if stop is None else tail[:stop + 1]
            usage = {"prompt_tokens": len(prompt), "completion_tokens": len(completion)}
            self.metadata["calls"] += 1
            for key, value in usage.items():
                self.metadata[key] += value
            results[index] = {
                "text": self.tokenizer.decode(completion, skip_special_tokens=True),
                "prompt_ids": prompt, "completion_ids": completion,
                "finish_reason": "length" if stop is None else "stop",
                "usage": usage,
            }
