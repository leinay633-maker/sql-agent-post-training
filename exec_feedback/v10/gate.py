"""The v0.10 weights gate, on one GPU before any online run: the engine serves the merged start, loads weights in place
and really computes with what it loaded. CUDA graphs captured at start-up keep reading the same parameter storage, so a
load that only replaced Python references, or a load the graphs never see, shows up here as unchanged log-probs. Three
partial loads, each on top of the last: the final norm (a plain parameter), every layer's v_proj (a slice of vLLM's
stacked qkv_proj) and every layer's gate_proj (a slice of gate_up_proj); then the full start weights back."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..storage import atomic_json
from .freeze import load, model_ready
from .merge import merged_start
from .rollout import full_engine
from .weights import WEIGHTS, checksum, same_checksum, write_weights

PROMPT = [{"role": "user", "content": "Name three SQL aggregate functions and say what each returns."}]
NORM = "model.norm.weight"
# vLLM keeps these as slices of one stacked tensor per layer (qkv_proj, gate_up_proj), loaded through another path than
# a plain parameter like the norm. Not k_proj or q_proj: Qwen3 normalises q and k per head, so scaling them is a no-op.
STACKED = {"qkv": "self_attn.v_proj.weight", "gate_up": "mlp.gate_proj.weight"}
MOVED = 0.01          # mean |delta log p| per greedy token after a perturbation (final norm x1.5, a stacked slice x2)
RESTORED = 1e-3       # max |delta log p| after loading the start weights back


def scratch(name: str, fallback: Path) -> Path:
    shm = Path("/dev/shm")
    return shm / "efa-v10" / name if shm.is_dir() else fallback


def gate_weights(run: Path, *, engine_factory: Callable[..., Any] = full_engine, tokenizer: Any = None,
                 stop_ids: list[int] | None = None) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    frozen, _ = load(run)
    s = frozen["settings"]
    seed = s["seeds"][0]
    start, record = merged_start(run, frozen, seed)
    if tokenizer is None:
        from ..models import load_tokenizer
        from ..v7.engine import stop_ids as model_stop_ids
        spec = model_ready(frozen)
        tokenizer = load_tokenizer(spec)
        stop_ids = model_stop_ids(spec, tokenizer)
    out = run / "gate"
    out.mkdir(exist_ok=True)
    work = scratch(f"{frozen['identity'][:12]}-gate", out / "scratch")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report: dict[str, Any] = {"start": str(start), "checks": {}, "seconds": {}}
    checks, seconds = report["checks"], report["seconds"]
    try:
        t0 = time.monotonic()
        engine = engine_factory(start, {**s, "rollout_gpu_memory": min(0.5, s["rollout_gpu_memory"])}, seed=20261019)
        seconds["engine_start"] = time.monotonic() - t0
        prompt = list(tokenizer.apply_chat_template(PROMPT, tokenize=True, add_generation_prompt=True))

        def greedy() -> tuple[list[int], list[float]]:
            sample = engine.generate([prompt], temperature=0.0, max_tokens=24, stop_ids=stop_ids, logprobs=True)[0]
            return list(sample["tokens"]), [float(v) for v in sample["logprobs"]]

        sums0 = engine.checksum()
        checks["start_checksum"] = same_checksum(sums0, record["checksum"])
        tokens0, lp0 = greedy()

        def perturb(label: str, names: list[str], factor: float, before: tuple[list[int], list[float], list[float]]
                    ) -> tuple[list[int], list[float], list[float]]:
            """Load `names` scaled by `factor` on top of the current weights: the checksum must move by exactly their
            change, and so must the greedy log-probs (or tokens) against the state before."""
            tokens_b, lp_b, sums_b = before
            with safe_open(str(start / WEIGHTS), framework="pt") as handle:
                original = {n: handle.get_tensor(n) for n in names}
            scaled = {n: (t.float() * factor).to(t.dtype) for n, t in original.items()}
            save_file(scaled, str(work / f"{label}.safetensors"))
            t0 = time.monotonic()
            sums = engine.load(work / f"{label}.safetensors")
            seconds[f"{label}_load"] = time.monotonic() - t0
            delta = [a - b for a, b in zip(checksum(scaled.values()), checksum(original.values()))]
            checks[f"{label}_checksum"] = same_checksum(sums, [sums_b[0] + delta[0], sums_b[1] + delta[1]])
            tokens, lp = greedy()
            common = [abs(a - b) for a, b, x, y in zip(lp_b, lp, tokens_b, tokens) if x == y]
            mean = sum(common) / len(common) if common else None
            report[label] = {"tensors": len(names), "same_tokens": tokens == tokens_b, "mean_abs_logp_delta": mean}
            checks[f"{label}_outputs_moved"] = tokens != tokens_b or (mean is not None and mean > MOVED)
            return tokens, lp, sums

        with safe_open(str(start / WEIGHTS), framework="pt") as handle:
            layers = sorted({int(k.split(".")[2]) for k in handle.keys() if k.startswith("model.layers.")})
        current = perturb("perturbed", [NORM], 1.5, (tokens0, lp0, sums0))
        for label, suffix in STACKED.items():  # cumulative: each is compared with the weights just before it
            current = perturb(label, [f"model.layers.{i}.{suffix}" for i in layers], 2.0, current)
        # the full start weights through the trainer's own writer, from the same place the online runs load from
        t0 = time.monotonic()
        state = load_file(str(start / WEIGHTS))
        seconds["read_start"] = time.monotonic() - t0
        t0 = time.monotonic()
        ready = write_weights(work / "v00000", state, {"version": 0, "attempt": "gate"})
        seconds["write_full"] = time.monotonic() - t0
        del state
        t0 = time.monotonic()
        sums2 = engine.load(Path(ready["path"]))
        seconds["full_load"] = time.monotonic() - t0
        checks["restored_checksum"] = same_checksum(sums2, record["checksum"]) and \
            same_checksum(ready["checksum"], record["checksum"])
        tokens2, lp2 = greedy()
        gap = max((abs(a - b) for a, b in zip(lp0, lp2)), default=0.0)
        report["restored"] = {"same_tokens": tokens2 == tokens0, "max_abs_logp_delta": gap}
        checks["outputs_restored"] = tokens2 == tokens0 and gap < RESTORED
        report["bytes"] = ready["bytes"]
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            report["gpu_memory"] = {"free": free, "total": total}
    except Exception as exc:  # noqa: BLE001 - recorded, then the gate fails
        report["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report["passed"] = "error" not in report and bool(checks) and all(checks.values())
    report["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    atomic_json(out / "weights.json", report)
    return report
