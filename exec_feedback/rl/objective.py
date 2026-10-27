"""On-policy trajectory REINFORCE with a leave-one-out baseline.

There is one update per fresh rollout batch: no PPO epochs, stale replay,
critic, reward model, standard-deviation normalization, or length averaging.
"""
from __future__ import annotations

import inspect
import math
from typing import Any


def advantages(rewards: list[float], group_size: int) -> list[float]:
    """Input order is task-major; all failures remain in each complete group."""
    if group_size < 2 or not rewards or len(rewards) % group_size:
        raise ValueError("Expected complete groups of at least two trajectories")
    if any(not math.isfinite(r) or r not in (0.0, 1.0) for r in rewards):
        raise ValueError("Execution rewards must be finite zero/one values")
    result = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        total = sum(group)
        result.extend(r - (total - r) / (group_size - 1) for r in group)
    return result


def group_metrics(rewards: list[float], group_size: int) -> dict[str, Any]:
    advantages(rewards, group_size)
    totals = [sum(rewards[i:i + group_size])
              for i in range(0, len(rewards), group_size)]
    return {
        "trajectories": len(rewards),
        "reward_sum": sum(rewards),
        "reward_mean": sum(rewards) / len(rewards),
        "groups": len(totals),
        "all_failed_groups": sum(t == 0 for t in totals),
        "all_correct_groups": sum(t == group_size for t in totals),
        "informative_groups": sum(0 < t < group_size for t in totals),
    }


def action_logprob(model: Any, prompt: list[int], completion: list[int]) -> Any:
    """Sum log p of exactly the sampled assistant tokens, including sampled EOS.

    Prompt/tool/history tokens condition this action, but are not loss targets.
    Every earlier assistant action gets its own call, so it is not masked out
    of the trajectory objective. No decode/re-tokenize boundary approximation.
    """
    import torch
    import torch.nn.functional as functional

    if not prompt or not completion:
        raise ValueError("A sampled action needs a prompt and completion")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + completion], device=device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    options = {}
    if "logits_to_keep" in inspect.signature(base.forward).parameters:
        options["logits_to_keep"] = len(completion) + 1
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, **options)
    # Last logit predicts an unsampled next token; exclude it.
    logits = output.logits[0, -(len(completion) + 1):-1].float()
    targets = ids[0, -len(completion):]
    value = -functional.cross_entropy(logits, targets, reduction="sum")
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite sampled-action log probability")
    return value


def backward_batch(model: Any, episodes: list[dict[str, Any]],
                   group_size: int) -> dict[str, Any]:
    """Accumulate a single unbiased trajectory-score estimate; caller steps once."""
    rewards = [row["reward"] for row in episodes]
    weights = advantages(rewards, group_size)
    loss_value, target_tokens, backward_calls = 0.0, 0, 0
    for episode, weight in zip(episodes, weights):
        # Zero-advantage groups stay in denominator/logs; no pointless Adam step
        # is taken when the entire batch has zero signal (see training.py).
        for turn in episode["sampled_turns"]:
            target_tokens += len(turn["completion_ids"])
            if weight == 0:
                continue
            score = action_logprob(model, turn["prompt_ids"],
                                   turn["completion_ids"])
            loss = -weight * score / len(episodes)
            loss.backward()
            backward_calls += 1
            loss_value += float(loss.detach())
    return {**group_metrics(rewards, group_size), "loss": loss_value,
            "sampled_assistant_tokens": target_tokens,
            "has_gradient": backward_calls > 0}
