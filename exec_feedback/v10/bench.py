"""Trainer-only benchmark on identical work: the trained trajectories a finished run kept from its first steps, replayed
through the run's own FSDP2 step (forward/backward over the rank shards, clipping, AdamW, then the full bf16 gather the
engines would load) under several layouts: full sharding (ZeRO-3 like) against gradient/optimizer sharding (ZeRO-2
like), on the run's trainer GPU count and on twice as many. The first repeat warms up and is reported, not averaged.

With --regroup one GPU replays how the layouts split the first kept step (regroup_rank): the layouts' first gradient
norms differed by 4% between 4 and 8 ranks, where fp32 on CPU gives the same gradient for any rank count.

    python -m torch.distributed.run --standalone --nproc-per-node 1 -m exec_feedback.v10.bench --run <run> \\
        --family RLF --seed 42 --regroup 4,8,2,16 --start <merged PTR_seed42> --out <run>/bench/regroup.json"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from ..storage import atomic_json, read_json
from .freeze import load, model_ready
from .merge import merged_start
from .trainer import (H20_BF16_DENSE, Dist, backward_rank, clip_and_norm, load_policy, make_optimizer, rank_shards,
                      reduce_stats, run_name, shard)
from .weights import WEIGHTS, checksum, export_full, same_checksum, write_weights

REPEATS = 2


def layouts(settings: dict[str, Any], gpus: int) -> list[tuple[str, int, bool]]:
    """(name, ranks, reshard_after_forward) for the run's trainer count and twice that, as far as the job has GPUs."""
    worlds = [w for w in (settings["trainer_gpus"], 2 * settings["trainer_gpus"]) if w <= gpus]
    return [(f"{'z3' if reshard else 'z2'}-{w}", w, reshard) for w in worlds for reshard in (True, False)]


def bench_items(run: Path, family: str, seed: int) -> list[Path]:
    folder = run / "train" / run_name(family, seed) / "bench-items"
    files = sorted(folder.glob("step-*.json")) if folder.exists() else []
    if not files:
        raise ValueError(f"{run_name(family, seed)} kept no benchmark items")
    return files


def bench_rank(run: Path, *, family: str, seed: int, layout: str, reshard: bool, out: Path, scratch: Path) -> None:
    """Inside torchrun: every rank replays the same steps; rank 0 writes the measurements."""
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from ..models import load_tokenizer

    frozen, _ = load(run)
    s = frozen["settings"]
    device = torch.device("cuda", torch.cuda.current_device())
    d = Dist(device)
    mesh = init_device_mesh("cuda", (d.world,))
    start, _ = merged_start(run, frozen, seed)
    pad_id = load_tokenizer(model_ready(frozen)).pad_token_id
    t0 = time.monotonic()
    model = shard(load_policy(start, device), mesh, reshard)
    optimizer = make_optimizer(model, s["learning_rate"])
    load_seconds = time.monotonic() - t0
    n_params = int(sum(p.numel() for p in model.parameters()))
    normaliser = float(s["tasks_per_step"] * s["group_size"] * s["loss_tokens"])
    rows = []
    model.train()
    for repeat in range(REPEATS):
        for path in bench_items(run, family, seed):
            items = read_json(path)["items"]
            mine = rank_shards(items, d.world, s["micro_tokens"], pad_id)[d.rank]  # deterministic on every rank
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            t1 = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            totals = reduce_stats(d, backward_rank(model, mine, pad_id=pad_id, normaliser=normaliser,
                                                   cap=s["tis_cap"], scale=d.world))
            torch.cuda.synchronize()
            t2 = time.monotonic()
            # same items on 4 and 8 ranks: equal norms show the loss scaling undoes FSDP's gradient average
            norm = clip_and_norm(model, s["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t3 = time.monotonic()
            weights, sums = export_full(model, d.rank)
            t4 = time.monotonic()
            write = 0.0
            if d.rank == 0:
                write_weights(scratch / "v00000", weights, {"version": 0, "attempt": "bench"}, sums)
                write = time.monotonic() - t4
                shutil.rmtree(scratch / "v00000", ignore_errors=True)
            del weights
            peak = d.reduce([float(torch.cuda.max_memory_allocated())], "max")[0]
            train = t2 - t1
            rows.append({"repeat": repeat, "step": read_json(path)["step"], "trajectories": len(items),
                         "tokens": int(totals["tokens"]), "micro_batches_per_rank": len(mine), "grad_norm": norm,
                         "train_seconds": train, "optimizer_seconds": t3 - t2, "gather_seconds": t4 - t3,
                         "write_seconds": write, "tokens_per_second": totals["tokens"] / train,
                         "mfu": 6.0 * n_params * totals["tokens"] / train / (d.world * H20_BF16_DENSE),
                         "peak_gpu_bytes": peak})
    if d.rank == 0:
        timed = [r for r in rows if r["repeat"] > 0] or rows
        median = lambda key: statistics.median(r[key] for r in timed)
        atomic_json(out, {"layout": layout, "ranks": d.world, "reshard_after_forward": reshard, "family": family,
                          "seed": seed, "parameters": n_params, "load_seconds": load_seconds, "rows": rows,
                          "median": {k: median(k) for k in ("train_seconds", "optimizer_seconds", "gather_seconds",
                                                            "write_seconds", "tokens_per_second", "mfu",
                                                            "peak_gpu_bytes")},
                          # the first repeat's, before any update: the later ones follow each layout's own steps
                          "first_grad_norms": [r["grad_norm"] for r in rows if r["repeat"] == 0],
                          "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "finished": dt.datetime.now(dt.timezone.utc).isoformat()})


def regrouped(items: list[dict[str, Any]], world: int, budget: int, pad_id: int) -> list[list[dict[str, Any]]]:
    """The micro-batches `world` ranks would run, rank after rank, without the dummies (they add nothing)."""
    return [b for part in rank_shards(items, world, budget, pad_id) for b in part if not all(x.get("dummy") for x in b)]


def regroup_rank(run: Path, *, family: str, seed: int, worlds: list[int], out: Path, start: Path | None = None,
                 device: Any = None, pad_id: int | None = None) -> None:
    """Inside a one-process torchrun: the first kept step's gradient norm as each rank count in `worlds` splits the
    trajectories, all its micro-batches run on this GPU and summed in fp32, as FSDP's reduce-scatter sums the ranks' bf16
    gradients. Norms that come back as the layouts' first_grad_norms leave only the bf16 arithmetic on differently
    composed micro-batches between the layouts, not the ranks or their communication. `run` may be a finished run's
    package on another host: `start` is then any merged copy of its start, compared with the weights it trained from.
    A copy merged again on another host's CPU may round a few values differently, which moves the norms far less than
    the 4% between the layouts; the replay runs all the same and records how far the two starts' checksums are apart."""
    import torch
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh
    from ..models import load_tokenizer

    frozen, _ = load(run, check_code=False)  # a finished run, replayed with this code
    s = frozen["settings"]
    device = device or torch.device("cuda", torch.cuda.current_device())
    start = Path(start) if start is not None else merged_start(run, frozen, seed)[0]
    trained = read_json(run / "train" / run_name(family, seed) / "manifest.json")["start_checksum"]
    mine = checksum(load_file(str(start / WEIGHTS)).values())
    start_check = {"same": same_checksum(mine, trained), "checksum": mine, "trained_checksum": trained,
                   "sum_delta": mine[0] - trained[0],
                   "squares_relative_delta": (mine[1] - trained[1]) / max(abs(trained[1]), 1.0)}
    if pad_id is None:
        pad_id = load_tokenizer({"local_path": str(start)}).pad_token_id
    model = shard(load_policy(start, device), init_device_mesh(device.type, (1,)), True)
    model.train()
    normaliser = float(s["tasks_per_step"] * s["group_size"] * s["loss_tokens"])
    kept = read_json(bench_items(run, family, seed)[0])
    rows = []
    for world in worlds:
        batches = regrouped(kept["items"], world, s["micro_tokens"], pad_id)
        norms = []
        for _ in range(REPEATS):  # the same micro-batches twice: the GPU's own run-to-run spread
            model.zero_grad(set_to_none=True)
            backward_rank(model, batches, pad_id=pad_id, normaliser=normaliser, cap=s["tis_cap"], scale=1.0)
            norms.append(clip_and_norm(model, s["max_grad_norm"]))
        layout = run / "bench" / f"z3-{world}.json"
        rows.append({"world": world, "micro_batches": len(batches), "grad_norms": norms,
                     "layout_first_grad_norm": read_json(layout)["first_grad_norms"][0] if layout.exists() else None})
    model.zero_grad(set_to_none=True)
    atomic_json(out, {"family": family, "seed": seed, "step": kept["step"], "trajectories": len(kept["items"]),
                      "start": start_check, "rows": rows, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "finished": dt.datetime.now(dt.timezone.utc).isoformat()})


def bench(run: Path, *, family: str, seed: int, python: str, log_dir: Path) -> dict[str, Any]:
    """Every layout that fits the job's GPUs, one torchrun each; the summary lists what ran and what failed."""
    frozen, _ = load(run)
    s = frozen["settings"]
    bench_items(run, family, seed)
    visible = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    out = run / "bench"
    out.mkdir(exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shm = Path("/dev/shm")
    scratch = (shm / "efa-v10" / f"{frozen['identity'][:12]}-bench") if shm.is_dir() else out / "scratch"
    summary: dict[str, Any] = {"family": family, "seed": seed, "layouts": {}}
    for name, world, reshard in layouts(s, len(visible)):
        target = out / f"{name}.json"
        if not target.exists():
            with open(log_dir / f"bench-{name}.log", "w", encoding="utf-8") as fh:
                rc = subprocess.run(
                    [python, "-m", "torch.distributed.run", "--standalone", "--nnodes", "1",
                     "--nproc-per-node", str(world), "-m", "exec_feedback.v10.bench", "--run", str(run),
                     "--family", family, "--seed", str(seed), "--layout", name,
                     "--reshard", "1" if reshard else "0", "--out", str(target), "--scratch", str(scratch)],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(visible[:world])}, stdout=fh,
                    stderr=subprocess.STDOUT).returncode
            summary["layouts"][name] = {"exit": rc}
        if target.exists():
            measured = read_json(target)
            summary["layouts"][name] = {"exit": 0, **measured["median"], "first_grad_norms": measured["first_grad_norms"]}
    shutil.rmtree(scratch, ignore_errors=True)
    summary["complete"] = all(v.get("exit") == 0 for v in summary["layouts"].values()) and bool(summary["layouts"])
    atomic_json(out / "summary.json", summary)
    return summary


def main() -> int:
    import torch
    import torch.distributed as dist

    ap = argparse.ArgumentParser(prog="exec_feedback.v10.bench")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layout")
    ap.add_argument("--reshard", choices=["0", "1"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path)
    ap.add_argument("--regroup", default=None, help="rank counts whose split one GPU replays, e.g. 4,8")
    ap.add_argument("--start", type=Path, default=None, help="with --regroup: a merged copy of the run's start")
    args = ap.parse_args()
    if args.regroup is None and None in (args.layout, args.reshard, args.scratch):
        ap.error("--layout, --reshard and --scratch are required without --regroup")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=dt.timedelta(minutes=30), device_id=torch.device("cuda", local))
    try:
        if args.regroup is not None:
            regroup_rank(args.run, family=args.family, seed=args.seed,
                         worlds=[int(w) for w in args.regroup.split(",")], out=args.out, start=args.start)
        else:
            bench_rank(args.run, family=args.family, seed=args.seed, layout=args.layout, reshard=args.reshard == "1",
                       out=args.out, scratch=args.scratch)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    from ..v9.shutdown import hard_exit

    hard_exit(main())
