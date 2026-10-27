"""python -m exec_feedback.v6 <freeze|sample|build|train|evaluate|report|progress|status> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v6")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--registration", type=Path, required=True)
    f.add_argument("--raw-root", type=Path, required=True)
    f.add_argument("--model-path", type=Path, required=True)
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--policy", default="B")
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True)
    e.add_argument("--shard", type=int, default=0)
    for p in (s, e):
        p.add_argument("--stop", default=None)
        p.add_argument("--mem", type=float, default=0.85)
        p.add_argument("--seed", type=int, default=20261012)
        p.add_argument("--max-num-seqs", type=int, default=256)
    b = sub.add_parser("build")
    b.add_argument("--round", type=int, required=True)
    t = sub.add_parser("train")
    t.add_argument("--round", type=int, required=True)
    t.add_argument("--variant", default="R")
    t.add_argument("--seed", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--procs", type=int, default=64)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    sub.add_parser("status")
    for p in (s, e, b, t, r, g, sub.choices["status"]):
        p.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "freeze":
        from .config import load_settings
        from .freeze import freeze
        result = freeze(args.registration, args.raw_root, args.out, load_settings(args.config), model_path=args.model_path,
                        smoke_limit=args.smoke_limit)
        result = {k: result[k] for k in ("identity", "counts", "databases", "column_notes", "gold")}
    elif args.cmd == "sample":
        from .rollouts import sample
        result = sample(args.run, round_=args.round, policy=args.policy, stop=args.stop, mem=args.mem, seed=args.seed,
                        max_num_seqs=args.max_num_seqs)
    elif args.cmd == "evaluate":
        from .rollouts import evaluate
        result = evaluate(args.run, condition=args.condition, role=args.role, stop=args.stop, mem=args.mem, shard=args.shard,
                          max_num_seqs=args.max_num_seqs)
        result = {k: result.get(k) for k in ("condition", "role", "shard", "episodes", "shard_tasks", "correct", "complete",
                                             "wall_seconds")}
    elif args.cmd == "build":
        from ..rft.selection import build
        result = build(args.run, args.round)
        result = {k: result.get(k) for k in ("trajectories", "sampled_tasks", "frontier_tasks", "easy_tasks",
                                             "unsolved_tasks", "datasets")}
    elif args.cmd == "train":
        from ..rft.training import train_model
        result = train_model(args.run, round_=args.round, variant=args.variant, seed=args.seed)
        result = {k: result.get(k) for k in ("training_completed", "adapter_reload_verified", "reload_max_abs_delta")}
    elif args.cmd == "report":
        from .analysis import report
        result = report(args.run, procs=args.procs)["selection"]
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"]}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
