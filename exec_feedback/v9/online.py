"""v0.9 online runs on one GPU: vLLM samples dialogues with the current LoRA, the train-only oracle scores the final
SQL, PEFT takes one on-policy step on every sampled assistant token, the new LoRA goes back to vLLM.

Token-in/token-out: the generator keeps each turn's prompt ids, sampled ids and vLLM log-probabilities. The last
turn's prompt already contains every earlier turn as the chat template re-rendered it, so a dialogue trains as one
sequence; a turn whose decoded text re-tokenizes to different ids would put tokens the model never sampled under the
loss, so such a trajectory is kept out of the gradient (counted, still in its group's baseline).
"""
from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..core import Limits, Task, digest
from ..storage import append_record, atomic_json, exclusive_lock, fingerprint, read_json, recover_records
from ..v6.rollouts import deadline_from, versions
from . import METHODS
from .freeze import load, model_ready, start_adapter

LORA_CHECK_TRAJECTORIES = 32
LORA_CHECK_MAX_GAP = 0.15         # mean |log p_vLLM - log p_trainer| per sampled token
LORA_CHECK_MIN_EFFECT = 0.005     # mean |log p_LoRA - log p_base| below which the check cannot tell them apart


# ------------------------------------------------------------------------------------------------ objective
def advantages(rewards: list[float], group_size: int, method: str) -> list[float]:
    """Task-major rewards in complete groups. rloo: r_i minus the mean of the other G-1 rewards (all-equal groups
    get zero); raft: r_i, i.e. only correct trajectories are trained on, without a baseline."""
    if group_size < 2 or not rewards or len(rewards) % group_size:
        raise ValueError("Expected complete groups of at least two trajectories")
    if any(r not in (0.0, 1.0) for r in rewards):
        raise ValueError("Execution rewards are zero or one")
    if method == "raft":
        return [float(r) for r in rewards]
    if method != "rloo":
        raise ValueError(f"Unknown method: {method}")
    out = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        total = sum(group)
        out.extend(r - (total - r) / (group_size - 1) for r in group)
    return out


def group_counts(rewards: list[float], valid: list[bool], group_size: int) -> dict[str, int]:
    counts = Counter()
    for start in range(0, len(rewards), group_size):
        if not all(valid[start:start + group_size]):
            counts["invalid_groups"] += 1
            continue
        total = sum(rewards[start:start + group_size])
        counts["all_failed_groups" if total == 0 else "all_correct_groups" if total == group_size
               else "informative_groups"] += 1
    return dict(counts)


def trajectory(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One training sequence from the per-turn traces, or None when no assistant token was sampled."""
    traces = [s.get("trace") for s in steps]
    if not traces:
        return None
    if any(t is None for t in traces):
        return {"usable": False, "reason": "missing_trace", "turns": len(traces)}
    last = traces[-1]
    sequence = list(last["prompt_ids"]) + list(last["tokens"]) + ([last["stop"]] if last["stop"] is not None else [])
    positions: list[int] = []
    old: list[float | None] = []
    for trace in traces:
        p, content = len(trace["prompt_ids"]), list(trace["tokens"])
        if sequence[:p] != list(trace["prompt_ids"]) or sequence[p:p + len(content)] != content:
            return {"usable": False, "reason": "retokenized", "turns": len(traces)}
        logprobs = trace.get("logprobs") or []
        positions.extend(range(p, p + len(content)))
        old.extend(logprobs[i] if i < len(logprobs) else None for i in range(len(content)))
        end = p + len(content)
        # The stop token is trained where it really follows the turn: always for the last turn; for earlier turns
        # only if the template re-rendered the same end-of-turn token the model sampled.
        if trace["stop"] is not None and end < len(sequence) and sequence[end] == trace["stop"]:
            positions.append(end)
            old.append(logprobs[len(content)] if len(content) < len(logprobs) else None)
    if not positions:
        return None
    return {"usable": True, "input_ids": sequence, "positions": positions, "old_logprobs": old,
            "turns": len(traces)}


def micro_batches(items: list[dict[str, Any]], budget: int) -> list[list[dict[str, Any]]]:
    """Length-sorted micro-batches whose padded size stays within `budget` tokens (one long item may exceed it)."""
    ordered = sorted(items, key=lambda x: len(x["input_ids"]))
    batches, current = [], []
    for item in ordered:
        width = max([len(x["input_ids"]) for x in current] + [len(item["input_ids"])])
        if current and width * (len(current) + 1) > budget:
            batches.append(current)
            current = []
        current.append(item)
    if current:
        batches.append(current)
    return batches


def sequence_logprobs(model: Any, batch: list[dict[str, Any]], pad_id: int) -> tuple[Any, Any]:
    """Log-probabilities of every target position of a right-padded micro-batch, and the logits rows they came
    from (for diagnostics). Only logits from the first target on are computed."""
    import torch
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    width = max(len(x["input_ids"]) for x in batch)
    ids = torch.tensor([x["input_ids"] + [pad_id] * (width - len(x["input_ids"])) for x in batch], device=device)
    mask = torch.tensor([[1] * len(x["input_ids"]) + [0] * (width - len(x["input_ids"])) for x in batch], device=device)
    first = min(min(x["positions"]) for x in batch)
    keep = width - first + 1
    cuda = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
    offset = width - logits.shape[1]
    rows = [r for r, x in enumerate(batch) for _ in x["positions"]]
    cols = [p - 1 - offset for x in batch for p in x["positions"]]
    targets = torch.tensor([x["input_ids"][p] for x in batch for p in x["positions"]], device=device)
    picked = logits[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
    logp = -functional.cross_entropy(picked.float(), targets, reduction="none")
    return logp, picked


def entropy_sum(picked: Any, chunk: int = 1024) -> float:
    import torch

    total = 0.0
    with torch.no_grad():
        for part in picked.detach().split(chunk):
            lp = torch.log_softmax(part.float(), dim=-1)
            total += float(-(lp.exp() * lp).sum())
    return total


def backward_step(model: Any, items: list[dict[str, Any]], *, pad_id: int, normaliser: float, cap: float,
                  budget: int) -> dict[str, Any]:
    """Accumulate -sum(w * min(ratio, cap) * logp) / normaliser over all items; the caller steps the optimizer."""
    import torch

    stats = Counter()
    for batch in micro_batches(items, budget):
        logp, picked = sequence_logprobs(model, batch, pad_id)
        device = logp.device
        weights = torch.tensor([x["weight"] for x in batch for _ in x["positions"]], device=device, dtype=torch.float32)
        old_values = [v for x in batch for v in x["old_logprobs"]]
        known = torch.tensor([v is not None for v in old_values], device=device)
        old = torch.tensor([v if v is not None else 0.0 for v in old_values], device=device, dtype=torch.float32)
        gap = (logp.detach() - old).masked_fill(~known, 0.0)
        ratio = torch.where(known, gap.exp(), torch.ones_like(gap))
        tis = ratio.clamp(max=cap)
        loss = -(weights * tis * logp).sum() / normaliser
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite policy loss")
        loss.backward()
        stats["loss"] += float(loss.detach())
        stats["targets"] += int(logp.numel())
        stats["known_targets"] += int(known.sum())
        stats["abs_gap_sum"] += float(gap.abs().sum())
        stats["max_abs_gap"] = max(stats["max_abs_gap"], float(gap.abs().max()) if gap.numel() else 0.0)
        stats["tis_clipped"] += int((ratio > cap).sum())
        stats["entropy_sum"] += entropy_sum(picked)
        stats["logp_sum"] += float(logp.detach().sum())
        stats["tokens"] += sum(len(x["input_ids"]) for x in batch)
        stats["micro_batches"] += 1
        del logp, picked, loss
    return dict(stats)


def lora_check(model: Any, items: list[dict[str, Any]], *, pad_id: int, budget: int) -> dict[str, Any]:
    """Before the first update: vLLM's sampled-token log-probs must be closer to the trainer with the start LoRA than
    to the bare base model, i.e. the engine really sampled from the policy being trained."""
    import torch

    gaps = {"lora": 0.0, "base": 0.0, "effect": 0.0}
    known = 0
    with torch.no_grad():
        for batch in micro_batches(items, budget):
            old_values = [v for x in batch for v in x["old_logprobs"]]
            keep = torch.tensor([v is not None for v in old_values])
            old = torch.tensor([v if v is not None else 0.0 for v in old_values], dtype=torch.float32)
            lora, _ = sequence_logprobs(model, batch, pad_id)
            with model.disable_adapter():
                base, _ = sequence_logprobs(model, batch, pad_id)
            lora, base = lora.cpu(), base.cpu()
            gaps["lora"] += float(((lora - old).abs() * keep).sum())
            gaps["base"] += float(((base - old).abs() * keep).sum())
            gaps["effect"] += float(((lora - base).abs() * keep).sum())
            known += int(keep.sum())
    lora_gap, base_gap, effect = (gaps[k] / max(1, known) for k in ("lora", "base", "effect"))
    # Kernel noise between vLLM and the trainer is roughly symmetric and independent of what the LoRA changes, so if
    # the engine sampled with the start LoRA its log-probs are on average closer to the trainer with the LoRA than to
    # the bare base, however small the LoRA's effect (measured 0.03-0.15 nat per token on PTR's own greedy outputs,
    # median 0). A large gap to the trainer means misaligned tokens or wrong weights. A LoRA with no measurable
    # effect cannot be checked this way and is recorded as inconclusive.
    if not known or lora_gap >= LORA_CHECK_MAX_GAP:
        verdict = "failed"
    elif effect < LORA_CHECK_MIN_EFFECT:
        verdict = "inconclusive"
    else:
        verdict = "closer_to_lora" if lora_gap < base_gap else "failed"
    return {"targets": known, "mean_abs_gap_lora": lora_gap, "mean_abs_gap_base": base_gap,
            "mean_abs_lora_effect": effect, "verdict": verdict, "passed": verdict != "failed"}


# ------------------------------------------------------------------------------------------------ generation
class TracingGenerator:
    """run_dialogues generator over a rollout engine (same chat template, context rule and stop ids as v0.6-v0.8);
    each response carries the turn's prompt ids, sampled ids and their log-probabilities."""
    kind = "model"

    def __init__(self, engine: Any, tokenizer: Any, *, max_context: int, max_tokens: int, temperature: float,
                 stop_ids: list[int], trace: bool) -> None:
        self.engine, self.tokenizer, self.max_context, self.max_tokens = engine, tokenizer, max_context, max_tokens
        self.temperature, self.stop_ids, self.trace = temperature, list(stop_ids), trace
        self.metadata = {"calls": 0, "generate_batches": 0, "generate_seconds": 0.0, "prompt_tokens": 0,
                         "cached_prompt_tokens": 0, "completion_tokens": 0, "length_stops": 0,
                         "context_rejections": 0, "max_batch": 0}

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[Any]:
        results: list[Any] = [None] * len(conversations)
        prompts, index = [], []
        for i, messages in enumerate(conversations):
            ids = list(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            if not ids or len(ids) + self.max_tokens > self.max_context:
                self.metadata["context_rejections"] += 1
                results[i] = ValueError("Context budget exceeded; no silent truncation")
                continue
            prompts.append(ids)
            index.append(i)
        if not prompts:
            return results
        started = time.perf_counter()
        samples = self.engine.generate(prompts, temperature=self.temperature, max_tokens=self.max_tokens,
                                       stop_ids=self.stop_ids, logprobs=self.trace)
        self.metadata["generate_seconds"] += time.perf_counter() - started
        self.metadata["generate_batches"] += 1
        self.metadata["max_batch"] = max(self.metadata["max_batch"], len(prompts))
        if len(samples) != len(prompts):
            raise ValueError("Engine returned a different number of samples")
        for i, ids, sample in zip(index, prompts, samples):
            tokens, stopped = list(sample["tokens"]), sample["finish_reason"] == "stop"
            stop = None
            if stopped and tokens and tokens[-1] in self.stop_ids:
                stop, tokens = tokens[-1], tokens[:-1]
            elif stopped and sample.get("stop_token") in self.stop_ids:
                stop = sample["stop_token"]  # the engine reported the stop id without returning it
            self.metadata["calls"] += 1
            self.metadata["prompt_tokens"] += len(ids)
            self.metadata["cached_prompt_tokens"] += sample.get("cached_tokens") or 0
            self.metadata["completion_tokens"] += len(tokens) + (stop is not None)
            self.metadata["length_stops"] += not stopped
            result = {"text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                      "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens) + (stop is not None)},
                      "finish_reason": "stop" if stopped else "length"}
            if self.trace:
                result["trace"] = {"prompt_ids": ids, "tokens": tokens, "stop": stop,
                                   "logprobs": sample.get("logprobs")}
            results[i] = result
        return results


def sampled_logprobs(logprobs: Any, tokens: list[int]) -> list[float | None]:
    """The sampled token's log-probability at every position (vLLM returns a dict per position)."""
    out: list[float | None] = []
    for i, token in enumerate(tokens):
        entry = logprobs[i] if i < len(logprobs) else None
        item = entry.get(token) if entry is not None and hasattr(entry, "get") else None
        out.append(float(item.logprob) if item is not None else None)
    return out


class VLLMEngine:
    """The run's vLLM engine; the policy LoRA is re-read from disk under a fresh id after every update."""

    def __init__(self, spec: dict[str, Any], settings: dict[str, Any], *, seed: int) -> None:
        from vllm import LLM

        self.llm = LLM(model=spec["local_path"], tokenizer=spec["local_path"], dtype="bfloat16",
                       max_model_len=settings["max_context"], gpu_memory_utilization=settings["rl_gpu_memory"],
                       enable_prefix_caching=True, seed=seed, generation_config="vllm",
                       max_num_seqs=settings["rl_max_num_seqs"], trust_remote_code=False, enable_lora=True,
                       max_lora_rank=settings["lora_r"], max_loras=1)
        self.lora = None

    def set_policy(self, path: Path, version: int) -> None:
        from vllm.lora.request import LoRARequest
        self.lora = LoRARequest(f"policy-{version}", version + 1, str(path))

    def generate(self, prompts: list[list[int]], *, temperature: float, max_tokens: int, stop_ids: list[int],
                 logprobs: bool) -> list[dict[str, Any]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(n=1, temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                                stop_token_ids=stop_ids, logprobs=0 if logprobs else None, skip_special_tokens=True)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False,
                                    lora_request=self.lora)
        samples = []
        for out in outputs:
            choice = out.outputs[0]
            tokens = list(choice.token_ids)
            samples.append({"tokens": tokens, "finish_reason": choice.finish_reason,
                            "stop_token": choice.stop_reason if isinstance(choice.stop_reason, int) else None,
                            "logprobs": sampled_logprobs(choice.logprobs, tokens)
                            if logprobs and choice.logprobs is not None else None,
                            "cached_tokens": getattr(out, "num_cached_tokens", 0) or 0})
        return samples


def vllm_engine(spec: dict[str, Any], settings: dict[str, Any], *, seed: int) -> VLLMEngine:
    return VLLMEngine(spec, settings, seed=seed)


def peft_policy(spec: dict[str, Any], adapter: Path) -> tuple[Any, Any]:
    """Base 4B (gradient checkpointing) plus the start LoRA, trainable. Not through models.load_model's adapter path:
    that compares the adapter's recorded base fingerprint, which for v0.8 adapters includes the v0.8 machine's
    download marker; the freeze proved the base through v0.8's manifest instead."""
    from peft import PeftModel
    from ..models import load_model

    model, tokenizer = load_model(spec, training=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    return model, tokenizer


# ------------------------------------------------------------------------------------------------ run helpers
def gold_for(run: Path, frozen: dict[str, Any], role: str) -> dict[str, Any]:
    path = run / "gold" / f"{role}.json"
    if role not in frozen["gold"] or digest(path) != frozen["gold"][role]["sha256"]:
        raise ValueError(f"Gold cache for {role} missing or changed")
    return read_json(path)["results"]


def task_order(pool: list[Task], salt: str, seed: int) -> list[Task]:
    """One fixed order per seed, shared by RL, RAFT and RLhi of that seed (paired comparisons see the same tasks)."""
    return sorted(pool, key=lambda t: hashlib.sha256(f"{salt}:online:{seed}:{t.id}".encode()).digest())


def step_tasks(order: list[Task], step: int, count: int) -> list[Task]:
    return [order[(step * count + i) % len(order)] for i in range(count)]


def trainable_state(model: Any) -> dict[str, Any]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_adapter(model: Any, folder: Path, base: dict[str, Any] | None = None) -> str:
    """PEFT adapter files (written to a temporary folder, then renamed), read back and compared tensor by tensor."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from ..v7.freeze import _adapter_hash

    tmp = folder.with_name(folder.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp), safe_serialization=True)
    if base is not None:
        atomic_json(tmp / "efa-base.json", base)
    saved = load_file(str(tmp / "adapter_model.safetensors"))
    current = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
    if set(saved) != set(current) or any(not torch.equal(saved[k], current[k].to(saved[k].dtype)) for k in saved):
        raise RuntimeError(f"Saved adapter differs from the trained weights: {folder}")
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return _adapter_hash(folder)


def save_checkpoint(out: Path, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
    import torch

    tmp, final, old = out / "ckpt.tmp", out / "ckpt", out / "ckpt.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    torch.save(trainable_state(model), tmp / "trainable.pt")
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    atomic_json(tmp / "state.json", {**state, "files": {p.name: digest(p) for p in tmp.glob("*.pt")}})
    if final.exists():
        if old.exists():
            shutil.rmtree(old)
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)


def restore_checkpoint(out: Path, model: Any, optimizer: Any, identity: str) -> dict[str, Any] | None:
    import torch

    folder = out / "ckpt" if (out / "ckpt" / "state.json").exists() else out / "ckpt.old"
    if not (folder / "state.json").exists():
        return None
    state = read_json(folder / "state.json")
    if state["identity"] != identity:
        raise ValueError("Checkpoint belongs to another online run")
    if any(digest(folder / name) != value for name, value in state.pop("files").items()):
        raise ValueError("Checkpoint content hash mismatch")
    weights = torch.load(folder / "trainable.pt", map_location="cpu", weights_only=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(weights) != set(params):
        raise ValueError("Checkpoint trainable names differ")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(weights[name].to(parameter.device))
    optimizer.load_state_dict(torch.load(folder / "optimizer.pt", map_location=next(iter(params.values())).device,
                                         weights_only=True))
    return state


def compact(record: dict[str, Any]) -> dict[str, Any]:
    """Journal form of a dialogue: no system/user prompt (rebuilt from the task), no token traces."""
    row = {k: v for k, v in record.items() if k not in ("messages", "result")}
    row["steps"] = [{k: v for k, v in s.items() if k != "trace"} for s in record.get("steps", [])]
    return row


def rollouts(engine: Any, tokenizer: Any, tasks: list[Task], root: Path, oracle: Any, settings: dict[str, Any],
             stop_ids: list[int], group: int) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None], dict]:
    from ..rft.runner import run_dialogues

    generator = TracingGenerator(engine, tokenizer, max_context=settings["max_context"],
                                 max_tokens=settings["max_tokens"], temperature=settings["temperature"],
                                 stop_ids=stop_ids, trace=True)
    items = [(f"{t.id}#{k}", t) for t in tasks for k in range(group)]  # a task's G dialogues back to back
    examples: dict[str, Any] = {}

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        scored = oracle.score(task, record)
        examples[key] = trajectory(record.get("steps", []))
        verdict = scored["verdict"]
        return {**compact(record), "key": key, "task_id": task.id, "db_id": task.db_id,
                "k": int(key.rsplit("#", 1)[1]), "verdict": verdict["status"], "correct": bool(verdict["correct"]),
                "valid": verdict["status"] != "oracle_error"}

    rows = {row["key"]: row for row in run_dialogues(
        items, root, generator, slots=settings["rl_slots"], max_steps=settings["max_steps"], review_success=True,
        limits=Limits(**settings["limits"]), workers=settings["sql_workers"], finish=finish, fenced_json=True)}
    if set(rows) != {k for k, _ in items}:
        raise RuntimeError("A sampled dialogue is missing; never train on an incomplete group")
    return [rows[k] for k, _ in items], [examples.get(k) for k, _ in items], dict(generator.metadata)


def curve_eval(engine: Any, tokenizer: Any, tasks: list[Task], root: Path, gold: dict[str, Any],
               settings: dict[str, Any], stop_ids: list[int], deadline: float | None) -> tuple[list, dict]:
    """Greedy dev evaluation with the engine's current LoRA (the learning curve; never used for any decision)."""
    from ..pipeline import grade_episode
    from ..rft.runner import run_dialogues

    limits = Limits(**settings["limits"])
    generator = TracingGenerator(engine, tokenizer, max_context=settings["max_context"],
                                 max_tokens=settings["max_tokens"], temperature=0.0, stop_ids=stop_ids, trace=False)
    started, rows = time.monotonic(), []

    def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        return compact(grade_episode(task, task.database(root), "model", record, limits, settings["max_steps"], True,
                                     gold=gold.get(task.id)))

    for row in run_dialogues([(t.id, t) for t in tasks], root, generator, slots=settings["rl_slots"],
                             max_steps=settings["max_steps"], review_success=True, limits=limits,
                             workers=settings["sql_workers"], finish=finish, deadline=deadline, fenced_json=True):
        rows.append(row)
    statuses = Counter(r["status"] for r in rows)
    return rows, {"tasks": len(tasks), "episodes": len(rows), "complete": len(rows) == len(tasks),
                  "correct": statuses.get("correct", 0), "statuses": dict(statuses),
                  "wall_seconds": time.monotonic() - started, "generator": dict(generator.metadata)}


# ------------------------------------------------------------------------------------------------ the run
def run_name(family: str, seed: int) -> str:
    return f"{family}_seed{seed}"


def registered(settings: dict[str, Any], family: str) -> list[int]:
    if family not in METHODS:
        raise ValueError(f"Unknown online family: {family}")
    return settings["hi_seeds"] if family == "RLhi" else settings["seeds"]


def plan_of(settings: dict[str, Any], family: str) -> dict[str, Any]:
    keys = ("tasks_per_step", "group_size", "planned_steps", "temperature", "warmup_steps", "max_grad_norm",
            "loss_tokens", "tis_cap", "micro_tokens", "save_steps", "curve_every")
    return {"method": METHODS[family], "learning_rate": settings["hi_learning_rate" if family == "RLhi" else
                                                                  "learning_rate"],
            **{k: settings[k] for k in keys}}


def milestones(out: Path) -> dict[int, str]:
    """Saved, verified milestone adapters of a run: step -> adapter hash."""
    path = out / "milestones.json"
    return {int(k): v for k, v in read_json(path).items()} if path.exists() else {}


def train_online(run: Path, *, family: str, seed: int, until: str | None,
                 engine_factory: Callable[..., Any] = vllm_engine,
                 policy_loader: Callable[..., tuple[Any, Any]] = peft_policy) -> dict[str, Any]:
    import torch
    from ..models import model_identity
    from ..rft.sampling import Oracle
    from ..v7.engine import stop_ids as model_stop_ids

    frozen, pool = load(run, "pool")
    _, dev = load(run, "dev")
    s = frozen["settings"]
    if seed not in registered(s, family):
        raise ValueError("Unregistered family/seed")
    spec = model_ready(frozen) if engine_factory is vllm_engine else frozen["models"]["student"]["spec"]
    start, start_hash = start_adapter(run, frozen, seed)
    plan = plan_of(s, family)
    identity = fingerprint({"run": frozen["identity"], "family": family, "seed": seed, "start": start_hash,
                            "plan": plan, "model": frozen["models"]["student"]["identity"]})
    out = run / "train" / run_name(family, seed)
    out.mkdir(parents=True, exist_ok=True)
    root, limits = Path(frozen["raw_root"]), Limits(**s["limits"])
    with exclusive_lock(out / ".train.lock"):
        if (out / "manifest.json").exists() and read_json(out / "manifest.json")["identity"] != identity:
            raise ValueError("Online run inputs changed; use a new run")
        if (out / "result.json").exists() and read_json(out / "result.json").get("finished"):
            return read_json(out / "result.json")
        deadline = deadline_from(until)
        attempt = uuid.uuid4().hex[:12]
        started = time.monotonic()
        # vLLM first: its start-up memory profiling must not see the trainer's allocations appear underneath it.
        engine = engine_factory(spec, s, seed=20261014 + 1009 * seed)
        engine_seconds = time.monotonic() - started
        torch.manual_seed(seed)
        model, tokenizer = policy_loader(spec, start)
        for module in model.modules():  # sampling has no dropout, so teacher forcing must not have it either
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        base = model_identity(spec)
        stops = model_stop_ids(spec, tokenizer)
        pad_id = tokenizer.pad_token_id
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable LoRA parameters")
        optimizer = torch.optim.AdamW(params, lr=plan["learning_rate"], weight_decay=0.0)
        state = restore_checkpoint(out, model, optimizer, identity)
        fresh = state is None
        if fresh:
            state = {"identity": identity, "step": 0, "optimizer_updates": 0, "active_seconds": 0.0,
                     "curve_steps": [], "lora_check": None}
        atomic_json(out / "manifest.json", {
            "identity": identity, "run": frozen["identity"], "family": family, "seed": seed, "start_adapter": str(start),
            "start_hash": start_hash, "plan": plan, "packages": {**versions(), "peft": _version("peft")}})
        append_record(out / "attempts.jsonl", {"attempt": attempt, "resumed_step": state["step"],
                                               "engine_load_seconds": engine_seconds,
                                               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        _set_aside(out, state["step"])
        prior, clock = state["active_seconds"], time.monotonic()
        order = task_order(pool, s["salt"], seed)
        oracle = Oracle(pool, root, limits)
        gold_dev = gold_for(run, frozen, "dev")
        policy_root = out / "policy"
        current = policy_root / f"step-{state['step']:04d}"
        save_adapter(model, current)
        engine.set_policy(current, state["step"])
        recent: list[float] = []
        stopped_by = "planned"
        normaliser = float(s["tasks_per_step"] * s["group_size"] * s["loss_tokens"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def left() -> float:
            return math.inf if deadline is None else deadline - time.monotonic()

        def curve(step: int) -> None:
            if step % s["curve_every"] or step in state["curve_steps"]:
                return
            estimate = (read_json(out / "curve-estimate.json")["seconds"] * 1.3
                        if (out / "curve-estimate.json").exists() else 600.0)
            if left() < estimate + 120:
                append_record(out / "curve.jsonl", {"step": step, "skipped": "deadline", "attempt": attempt})
                state["curve_steps"].append(step)
                return
            rows, summary = curve_eval(engine, tokenizer, dev, root, gold_dev, s, stops, deadline)
            path = out / "curve" / f"step-{step:04d}.jsonl"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                path.rename(path.with_name(path.name + f".interrupted-{time.time_ns()}"))
            for row in rows:
                append_record(path, row)
            append_record(out / "curve.jsonl", {"step": step, "attempt": attempt, **summary,
                                                "episodes_sha256": digest(path)})
            atomic_json(out / "curve-estimate.json", {"seconds": summary["wall_seconds"]})
            state["curve_steps"].append(step)

        while state["step"] < s["planned_steps"]:
            curve(state["step"])
            estimate = 1.3 * max(recent[-3:]) if recent else 600.0
            if left() < estimate + 120:
                stopped_by = "deadline"
                break
            t0 = time.monotonic()
            index = state["step"]
            tasks = step_tasks(order, index, s["tasks_per_step"])
            rows, examples, gen = rollouts(engine, tokenizer, tasks, root, oracle, s, stops, s["group_size"])
            t_gen = time.monotonic() - t0
            rewards = [1.0 if r["correct"] else 0.0 for r in rows]
            valid = [r["valid"] for r in rows]
            weights = advantages(rewards, s["group_size"], plan["method"])
            for g in range(0, len(rows), s["group_size"]):  # a group with an unscorable reference teaches nothing
                if not all(valid[g:g + s["group_size"]]):
                    weights[g:g + s["group_size"]] = [0.0] * s["group_size"]
            items = []
            for row, example, reward, weight in zip(rows, examples, rewards, weights):
                row["reward"], row["weight"] = reward, weight
                row["trained"] = bool(weight and example and example["usable"])
                row["unusable"] = example["reason"] if example and not example["usable"] else None
                if row["trained"]:
                    items.append({**example, "weight": weight})
            check = None
            if fresh and index == 0 and state["lora_check"] is None:
                usable = [e for e in examples if e and e["usable"]][:LORA_CHECK_TRAJECTORIES]
                model.eval()
                check = lora_check(model, usable, pad_id=pad_id, budget=s["micro_tokens"])
                state["lora_check"] = check
                if not check["passed"]:
                    raise RuntimeError(f"vLLM did not sample from the start LoRA: {check}")
            t1 = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = backward_step(model, items, pad_id=pad_id, normaliser=normaliser, cap=plan["tis_cap"],
                                  budget=s["micro_tokens"]) if items else {}
            lr = plan["learning_rate"] * min(1.0, (state["optimizer_updates"] + 1) / plan["warmup_steps"])
            norm = 0.0
            if items:
                norm = float(torch.nn.utils.clip_grad_norm_(params, plan["max_grad_norm"], error_if_nonfinite=True))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                state["optimizer_updates"] += 1
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():  # micro-batch shapes change every step; hand cached blocks back
                torch.cuda.empty_cache()
            t_train = time.monotonic() - t1
            state["step"] += 1
            nxt = policy_root / f"step-{state['step']:04d}"
            save_adapter(model, nxt)
            engine.set_policy(nxt, state["step"])
            if state["step"] % s["save_steps"] == 0:
                saved = save_adapter(model, out / "milestones" / f"step-{state['step']:04d}" / "adapter", base)
                marks = milestones(out)
                marks[state["step"]] = saved
                atomic_json(out / "milestones.json", {str(k): v for k, v in sorted(marks.items())})
            state["active_seconds"] = prior + time.monotonic() - clock
            journal = out / "rollouts" / f"step-{index:04d}.jsonl"
            journal.parent.mkdir(exist_ok=True)
            for row in rows:
                append_record(journal, row)
            step_seconds = time.monotonic() - t0
            recent.append(step_seconds)
            targets = max(1, stats.get("targets", 0))
            append_record(out / "metrics.jsonl", {
                "step": state["step"], "attempt": attempt, "tasks": [t.id for t in tasks],
                "reward_mean": sum(rewards) / len(rewards), **group_counts(rewards, valid, s["group_size"]),
                "statuses": dict(Counter(r["status"] for r in rows)), "trained_trajectories": len(items),
                "unusable": dict(Counter(r["unusable"] for r in rows if r["unusable"])),
                "mean_turns": sum(len(r["steps"]) for r in rows) / len(rows),
                "mean_completion_tokens": gen["completion_tokens"] / len(rows),
                "length_stops": gen["length_stops"], "context_rejections": gen["context_rejections"],
                "optimizer_updated": bool(items), "optimizer_updates": state["optimizer_updates"],
                "lr": lr if items else 0.0, "grad_norm": norm, "loss": stats.get("loss", 0.0),
                "targets": stats.get("targets", 0), "train_tokens": stats.get("tokens", 0),
                "micro_batches": stats.get("micro_batches", 0),
                "mean_abs_logp_gap": stats.get("abs_gap_sum", 0.0) / max(1, stats.get("known_targets", 0)),
                "max_abs_logp_gap": stats.get("max_abs_gap", 0.0),
                "tis_clipped_fraction": stats.get("tis_clipped", 0) / targets,
                "entropy": stats.get("entropy_sum", 0.0) / targets, "mean_logp": stats.get("logp_sum", 0.0) / targets,
                "lora_check": check, "generate_seconds": t_gen, "train_seconds": t_train, "step_seconds": step_seconds,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                "active_seconds": state["active_seconds"]})
            # durable last: a crash before this line replays the step, and _set_aside keeps the replayed records
            save_checkpoint(out, model, optimizer, state)
            for old in sorted(policy_root.glob("step-*"))[:-2]:
                shutil.rmtree(old, ignore_errors=True)
            gc.collect()
        curve(state["step"])
        state["active_seconds"] = prior + time.monotonic() - clock
        save_checkpoint(out, model, optimizer, state)
        marks = milestones(out)
        if s["phase"] == "smoke" and stopped_by == "planned" and not state["optimizer_updates"]:
            raise RuntimeError("Smoke run never updated the policy; the gradient path is unexercised")
        result = {"finished": True, "identity": identity, "family": family, "seed": seed, "method": plan["method"],
                  "stopped_by": stopped_by, "steps_completed": state["step"], "planned_steps": s["planned_steps"],
                  "optimizer_updates": state["optimizer_updates"], "milestones": {str(k): v for k, v in marks.items()},
                  "last_milestone": max(marks) if marks else None, "lora_check": state["lora_check"],
                  "curve_steps": state["curve_steps"], "active_seconds": state["active_seconds"],
                  "attempt": attempt}
        atomic_json(out / "result.json", result)
        return result


def _version(package: str) -> str:
    import importlib.metadata
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _set_aside(out: Path, step: int) -> None:
    """Records of steps after the last checkpoint were not durable: keep them beside the journal for forensics, never
    count them twice (the step is sampled again)."""
    stamp = time.time_ns()
    path = out / "metrics.jsonl"
    rows = recover_records(path)
    durable = [r for r in rows if r["step"] <= step]
    if len(durable) != len(rows):
        path.rename(out / f"metrics.rolled-back-{stamp}.jsonl")
        for row in durable:
            append_record(path, row)
    for journal in sorted((out / "rollouts").glob("step-*.jsonl")) if (out / "rollouts").exists() else []:
        if int(journal.stem.split("-")[1]) >= step:
            journal.rename(journal.with_name(journal.name + f".rolled-back-{stamp}"))
