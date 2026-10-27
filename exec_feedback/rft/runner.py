"""Many independent gold-free dialogues on one GPU: refilled slots, bucketed generate, parallel SQLite.

Conversations never share messages. Only the order of work changes with
scheduling; each dialogue still runs the shared agent state machine.
"""
from __future__ import annotations

import concurrent.futures
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

from ..agent import dialogue_stream
from ..core import Limits, Task


class Generator:
    """Length-bucketed Transformers generation; greedy unless sampling is given."""
    kind = "model"

    def __init__(self, model: Any, tokenizer: Any, *, batch_size: int, max_context: int,
                 max_tokens: int, sampling: dict[str, Any] | None = None, seed: int = 0) -> None:
        if not 1 <= batch_size <= 512 or max_tokens < 1:
            raise ValueError("Invalid generation batch/output budget")
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Explicit pad and EOS ids are required")
        self.model, self.tokenizer = model, tokenizer
        self.batch_size, self.max_context, self.max_tokens = batch_size, max_context, max_tokens
        self.sampling, self.seed = sampling, seed
        self.metadata = {"calls": 0, "generate_batches": 0, "generate_seconds": 0.0,
                         "prompt_tokens": 0, "completion_tokens": 0, "padded_prompt_tokens": 0,
                         "length_stops": 0, "context_rejections": 0}

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[Any]:
        results: list[Any] = [None] * len(conversations)
        encoded = []
        for index, messages in enumerate(conversations):
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if not ids or len(ids) + self.max_tokens > self.max_context:
                self.metadata["context_rejections"] += 1
                results[index] = ValueError("Context budget exceeded; no silent truncation")
                continue
            encoded.append((index, ids))
        encoded.sort(key=lambda item: (len(item[1]), item[0]))
        for start in range(0, len(encoded), self.batch_size):
            self._generate(encoded[start:start + self.batch_size], results)
        return results

    def _generate(self, rows: list[tuple[int, list[int]]], results: list[Any]) -> None:
        import torch

        device = next(self.model.parameters()).device
        pad, eos = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        width = max(len(ids) for _, ids in rows)
        ids = [[pad] * (width - len(row)) + row for _, row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for _, row in rows]
        # Keyword arguments override the model's saved generation_config
        # (Qwen ships do_sample=True); a fresh GenerationConfig object would not.
        options: dict[str, Any] = {"max_new_tokens": self.max_tokens, "eos_token_id": eos,
                                   "pad_token_id": pad, "use_cache": True, "num_beams": 1}
        if self.sampling:
            options.update(do_sample=True, temperature=self.sampling["temperature"],
                           top_p=self.sampling["top_p"], top_k=self.sampling["top_k"])
            torch.manual_seed(self.seed + self.metadata["generate_batches"])
        else:
            options.update(do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids=torch.tensor(ids, device=device),
                                         attention_mask=torch.tensor(masks, device=device), **options)
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.metadata["generate_seconds"] += time.perf_counter() - started
        self.metadata["generate_batches"] += 1
        self.metadata["padded_prompt_tokens"] += width * len(rows)
        for (index, prompt), tail in zip(rows, output[:, width:].tolist()):
            stop = next((i for i, token in enumerate(tail) if token == eos), None)
            completion = tail if stop is None else tail[:stop + 1]
            usage = {"prompt_tokens": len(prompt), "completion_tokens": len(completion)}
            self.metadata["calls"] += 1
            self.metadata["prompt_tokens"] += usage["prompt_tokens"]
            self.metadata["completion_tokens"] += usage["completion_tokens"]
            self.metadata["length_stops"] += stop is None
            results[index] = {"text": self.tokenizer.decode(completion, skip_special_tokens=True),
                              "usage": usage, "finish_reason": "length" if stop is None else "stop"}


def run_dialogues(items: list[tuple[str, Task]], root: Path, generator: Any, *, slots: int,
                  max_steps: int, review_success: bool, limits: Limits, workers: int,
                  finish: Callable[[str, Task, dict[str, Any]], dict[str, Any]],
                  deadline: float | None = None, reserve: Callable[[int], None] | None = None,
                  extra_for: Callable[[Task], dict[str, Any] | None] | None = None, fenced_json: bool = False,
                  ) -> Iterator[dict[str, Any]]:
    """Yield finish(key, task, record) per dialogue in completion order.

    New dialogues stop starting once `deadline` (time.monotonic) passes; the
    already active ones still finish so no started trajectory is cut short.
    """
    if not 1 <= slots <= 512 or not 1 <= workers <= 64:
        raise ValueError("Invalid slot/worker count")
    pending = deque(items)
    active: list[dict[str, Any]] = []

    def start(item: tuple[str, Task]) -> dict[str, Any]:
        key, task = item
        engine = dialogue_stream(task.database(root), task.question, task.evidence, max_steps=max_steps,
                                 feedback=True, limits=limits, review_success=review_success,
                                 extra=extra_for(task) if extra_for else None, fenced_json=fenced_json)
        state = {"key": key, "task": task, "engine": engine, "truncated_turns": 0}
        try:
            state["messages"] = next(engine)
        except StopIteration as done:
            state["row"] = finish(key, task, done.value)
        return state

    def advance(pair: tuple[dict[str, Any], Any]) -> dict[str, Any]:
        state, response = pair
        if isinstance(response, dict) and response.get("finish_reason") == "length":
            state["truncated_turns"] += 1
        try:
            state["messages"] = state["engine"].send(response)
        except StopIteration as done:
            done.value["truncated_turns"] = state["truncated_turns"]
            state["row"] = finish(state["key"], state["task"], done.value)
        return state

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while pending or active:
                fresh = []
                while pending and len(active) + len(fresh) < slots and (deadline is None or time.monotonic() < deadline):
                    fresh.append(pending.popleft())
                for state in pool.map(start, fresh):
                    if "row" in state:
                        yield state["row"]
                    else:
                        active.append(state)
                if not active:
                    if pending and deadline is not None and time.monotonic() >= deadline:
                        break
                    continue
                if reserve is not None:
                    reserve(len(active))
                responses = generator.generate([state["messages"] for state in active])
                if len(responses) != len(active):
                    raise ValueError("Generation cardinality mismatch; refuse reassignment")
                remaining = []
                for state in pool.map(advance, zip(active, responses)):
                    if "row" in state:
                        yield state["row"]
                    else:
                        remaining.append(state)
                active = remaining
        finally:
            for state in active:
                state["engine"].close()
