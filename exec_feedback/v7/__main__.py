"""python -m exec_feedback.v7 <freeze|sample|evaluate|build|train|prefer|select-x|select-final|report|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v7")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--kind", required=True, choices=["teacher", "scale"])
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    e.add_argument("--shard", type=int, default=0)
    for p in (s, e):
        p.add_argument("--stop", default=None)
        p.add_argument("--mem", type=float, default=None)
        p.add_argument("--seed", type=int, default=20261012)
    b = sub.add_parser("build")
    b.add_argument("--kind", required=True, choices=["teacher", "scale"])
    t = sub.add_parser("train")
    t.add_argument("--family", required=True, choices=["T", "TR", "Traw", "R14"])
    t.add_argument("--seed", type=int, required=True)
    pr = sub.add_parser("prefer")
    pr.add_argument("--family", required=True, choices=["PT", "PTR"])
    pr.add_argument("--seed", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--procs", type=int, default=64)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    for name in ("select-x", "select-final", "status"):
        sub.add_parser(name)
    for p in (s, e, b, t, pr, r, g, sub.choices["select-x"], sub.choices["select-final"], sub.choices["status"]):
        p.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "freeze":
        from .config import load_settings
        from .freeze import freeze
        result = freeze(args.registration, args.raw_root, args.out, load_settings(args.config),
                        models_root=args.models_root, manifest_path=args.models_manifest, imports=args.imports,
                        smoke_limit=args.smoke_limit)
        result = {k: result[k] for k in ("identity", "counts", "databases", "n_examples", "gold")}
    elif args.cmd == "sample":
        from .rollouts import sample
        result = sample(args.run, kind=args.kind, stop=args.stop, mem=args.mem, seed=args.seed)
    elif args.cmd == "evaluate":
        from .rollouts import evaluate
        result = evaluate(args.run, condition=args.condition, role=args.role, stop=args.stop, mem=args.mem,
                          shard=args.shard)
        result = {k: result.get(k) for k in ("condition", "role", "shard", "episodes", "shard_tasks", "correct",
                                             "complete", "wall_seconds")}
    elif args.cmd == "build":
        from .build import build_scale, build_teacher
        result = (build_teacher if args.kind == "teacher" else build_scale)(args.run)
        result = {k: result.get(k) for k in ("kind", "n_examples", "datasets")}
        result["datasets"] = {k: v.get("examples") for k, v in result["datasets"].items()}
    elif args.cmd == "train":
        from .training import train_sft
        result = train_sft(args.run, family=args.family, seed=args.seed)
        result = {k: result.get(k) for k in ("training_completed", "adapter_reload_verified", "reload_max_abs_delta")}
    elif args.cmd == "prefer":
        from .training import train_preference
        result = train_preference(args.run, family=args.family, seed=args.seed)
        result = {k: result.get(k) for k in ("training_completed", "adapter_reload_verified", "reload_max_abs_delta")}
    elif args.cmd == "select-x":
        from .analysis import select_x
        result = select_x(args.run)
    elif args.cmd == "select-final":
        from .analysis import select_final
        result = select_final(args.run)
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run, procs=args.procs).items() if k in ("select_x", "select_final")}
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "n_examples": frozen["n_examples"]}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
