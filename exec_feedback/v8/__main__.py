"""python -m exec_feedback.v8 <freeze|sample|evaluate|gate|build|train|prefer|report|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v8")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--sampler", required=True, choices=["teacher", "hint"])
    s.add_argument("--seed", type=int, default=20261013)
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    for p in (s, e):
        p.add_argument("--stop", default=None)
        p.add_argument("--mem", type=float, default=None)
    b = sub.add_parser("build")
    b.add_argument("--sampler", required=True, choices=["teacher", "hint"])
    t = sub.add_parser("train")
    t.add_argument("--family", required=True, choices=["TR", "HR"])
    t.add_argument("--seed", type=int, required=True)
    pr = sub.add_parser("prefer")
    pr.add_argument("--family", required=True, choices=["PTR", "PHR"])
    pr.add_argument("--seed", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--procs", type=int, default=64)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    for name in ("gate", "status"):
        sub.add_parser(name)
    for p in (s, e, b, t, pr, r, g, sub.choices["gate"], sub.choices["status"]):
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
        result = sample(args.run, sampler=args.sampler, stop=args.stop, mem=args.mem, seed=args.seed)
    elif args.cmd == "evaluate":
        from .rollouts import evaluate
        result = evaluate(args.run, condition=args.condition, role=args.role, stop=args.stop, mem=args.mem)
        result = {k: result.get(k) for k in ("condition", "role", "episodes", "registered_tasks", "correct",
                                             "complete", "wall_seconds", "engine_load_seconds")}
        if not result["complete"]:
            json.dump(result, sys.stdout, default=str)
            print("\nincomplete evaluation (deadline)")
            return 1
    elif args.cmd == "gate":
        from .analysis import gate
        result = gate(args.run)
    elif args.cmd == "build":
        from .build import build
        result = build(args.run, sampler=args.sampler)
        result = {k: result.get(k) for k in ("family", "unsolved_tasks", "sampled_tasks", "datasets")}
        result["datasets"] = {k: (v.get("examples"), v.get("gap_examples")) for k, v in result["datasets"].items()}
    elif args.cmd == "train":
        from .training import train_sft
        result = train_sft(args.run, family=args.family, seed=args.seed)
        result = {k: result.get(k) for k in ("training_completed", "adapter_reload_verified", "reload_max_abs_delta")}
    elif args.cmd == "prefer":
        from .training import train_preference
        result = train_preference(args.run, family=args.family, seed=args.seed)
        result = {k: result.get(k) for k in ("training_completed", "adapter_reload_verified", "reload_max_abs_delta")}
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run, procs=args.procs).items() if k in ("gate", "select_final")}
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "n_examples": frozen["n_examples"],
                  "unsolved_tasks": len(frozen["unsolved_tasks"])}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
