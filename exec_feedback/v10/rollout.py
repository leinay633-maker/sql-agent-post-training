"""A v0.10 rollout worker: one process and one vLLM engine per rollout GPU.

It serves the run's requests in order: a sample request names pool tasks (whole groups) and a weight version, a curve
request names dev tasks for the greedy learning curve. Before serving it loads the requested version in place and checks
that the engine's weight checksum is the one the trainer recorded for that version. Sampling, the agent protocol, the
oracle and the token traces are v0.9's (exec_feedback.v9.online.rollouts / curve_eval).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ..core import Limits
from ..v6.rollouts import versions
from ..v9.online import curve_eval, gold_for, rollouts, sampled_logprobs
from .config import per_worker
from .freeze import load, model_ready
from .merge import merged_start
from .protocol import Channel, Heartbeat, wait_for
from .weights import same_checksum


class FullEngine:
    """vLLM over a full-weight directory; `load` swaps the weights in place (same interface as v0.9's engine)."""

    def __init__(self, model_dir: Path, settings: dict[str, Any], *, seed: int) -> None:
        from vllm import LLM

        self.llm = LLM(model=str(model_dir), tokenizer=str(model_dir), dtype="bfloat16",
                       max_model_len=settings["max_context"], gpu_memory_utilization=settings["rollout_gpu_memory"],
                       enable_prefix_caching=True, seed=seed, generation_config="vllm",
                       max_num_seqs=settings["rollout_max_num_seqs"], trust_remote_code=False,
                       worker_extension_cls="exec_feedback.v10.vllm_ext.WeightLoader")

    def checksum(self) -> list[float]:
        return self.llm.collective_rpc("efa_checksum")[0]

    def load(self, path: Path) -> list[float]:
        sums = self.llm.collective_rpc("efa_load", args=(str(path),))[0]
        self.llm.reset_prefix_cache()  # cached prompt blocks hold activations of the old weights
        return sums

    def generate(self, prompts: list[list[int]], *, temperature: float, max_tokens: int, stop_ids: list[int],
                 logprobs: bool) -> list[dict[str, Any]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(n=1, temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                                stop_token_ids=stop_ids, logprobs=0 if logprobs else None, skip_special_tokens=True)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
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


def full_engine(model_dir: Path, settings: dict[str, Any], *, seed: int) -> FullEngine:
    return FullEngine(model_dir, settings, seed=seed)


def assignment(tasks: list[Any], workers: int) -> dict[str, list[str]]:
    """Contiguous blocks of whole tasks per worker: worker order is task order, so merged results stay task-major."""
    size = len(tasks) // workers
    if size * workers != len(tasks):
        raise ValueError("Tasks do not split evenly over the rollout workers")
    return {str(w): [t.id for t in tasks[w * size:(w + 1) * size]] for w in range(workers)}


def curve_assignment(tasks: list[Any], workers: int) -> dict[str, list[str]]:
    return {str(w): [t.id for t in tasks[w::workers]] for w in range(workers)}


def serve(run: Path, *, family: str, seed: int, worker: int, root: Path,
          engine_factory: Callable[..., Any] = full_engine, tokenizer: Any = None,
          stop_ids: list[int] | None = None) -> dict[str, Any]:
    from ..models import load_tokenizer
    from ..rft.sampling import Oracle
    from ..v7.engine import stop_ids as model_stop_ids

    frozen, pool = load(run, "pool")
    _, dev = load(run, "dev")
    s = frozen["settings"]
    if not 0 <= worker < s["rollout_gpus"]:
        raise ValueError("Unregistered rollout worker")
    spec = model_ready(frozen) if engine_factory is full_engine else frozen["models"]["student"]["spec"]
    start, record = merged_start(run, frozen, seed)
    channel = Channel(root)
    beat = Heartbeat(channel, worker)
    served = 0
    try:
        tokenizer = tokenizer or load_tokenizer(spec)
        stops = stop_ids if stop_ids is not None else model_stop_ids(spec, tokenizer)
        engine = engine_factory(start, s, seed=20261019 + 1009 * seed + 7 * worker)
        sums = engine.checksum()
        if not same_checksum(sums, record["checksum"]):
            raise RuntimeError(f"Engine did not load the merged start weights: {sums} vs {record['checksum']}")
        version, load_seconds = 0, 0.0
        root_db, limits = Path(frozen["raw_root"]), Limits(**s["limits"])
        oracle = Oracle(pool, root_db, limits)
        gold_dev = gold_for(run, frozen, "dev")
        pool_by_id, dev_by_id = {t.id: t for t in pool}, {t.id: t for t in dev}
        local = {**s, "rl_slots": s["rollout_slots"]}  # v0.9's rollout helpers read rl_slots
        wait = 60.0 * s["wait_minutes"]
        seq = 0
        while True:
            beat.state = f"waiting {seq}"
            try:
                request = wait_for(lambda: channel.request(seq), timeout=24 * 3600.0, what=f"request {seq}",
                                   channel=channel)
            except RuntimeError:  # STOP: the run is over
                break
            if request["kind"] == "stop":
                break
            load_seconds = 0.0
            if request["version"] != version:
                beat.state = f"loading v{request['version']}"
                ready = wait_for(lambda: channel.ready(request["version"]), timeout=wait,
                                 what=f"weights v{request['version']}", channel=channel)
                t0 = time.monotonic()
                sums = engine.load(Path(ready["path"]))
                load_seconds = time.monotonic() - t0
                if not same_checksum(sums, ready["checksum"]):
                    raise RuntimeError(f"Engine weights differ from version {request['version']}: "
                                       f"{sums} vs {ready['checksum']}")
                version = request["version"]
            beat.state = f"{request['kind']} {seq} step {request['step']}"
            started, t0 = time.time(), time.monotonic()
            mine = request["assign"][str(worker)]
            if request["kind"] == "sample":
                if len(mine) != per_worker(s):
                    raise ValueError("Sample request does not give this worker its share of whole groups")
                rows, examples, generator = rollouts(engine, tokenizer, [pool_by_id[i] for i in mine], root_db,
                                                     oracle, local, stops, s["group_size"])
                payload = {"rows": rows, "examples": examples, "generator": generator}
            elif request["kind"] == "curve":
                rows, summary = curve_eval(engine, tokenizer, [dev_by_id[i] for i in mine], root_db, gold_dev, local,
                                           stops, None)
                payload = {"rows": rows, "summary": summary}
            else:
                raise ValueError(f"Unknown request kind: {request['kind']}")
            channel.put_result(seq, worker, {
                "seq": seq, "kind": request["kind"], "step": request["step"], "version": version, "worker": worker,
                "checksum": sums, "load_seconds": load_seconds, "seconds": time.monotonic() - t0, "started": started,
                "finished": time.time(), **payload})
            served += 1
            seq += 1
        return {"worker": worker, "served": served, "packages": versions()}
    finally:
        beat.close()
