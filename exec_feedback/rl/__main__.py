"""Explicit research stages; no model loading or downloads without --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings, plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent v0.4 execution-reward RL")
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--config", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--source-study", type=Path, required=True)
    prepare.add_argument("--raw-root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--execute", action="store_true")
    for name in ("train", "evaluate", "benchmark", "report"):
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--execute", action="store_true")
        if name == "train":
            command.add_argument("--seed", type=int, required=True)
        if name == "evaluate":
            command.add_argument("--condition", required=True)
        if name == "report":
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = plan(load_settings(args.config))
        elif not args.execute:
            parser.error("Explicit --execute is required; no work was started")
        elif args.command == "prepare":
            from .study import prepare as freeze
            result = freeze(args.source_study, args.raw_root, args.out, load_settings(args.config))
        elif args.command == "train":
            from .training import train_seed
            result = train_seed(args.run, args.seed)
        elif args.command == "evaluate":
            from .evaluation import evaluate_condition
            result = evaluate_condition(args.run, args.condition)
        elif args.command == "benchmark":
            from .evaluation import benchmark
            result = benchmark(args.run)
        else:
            from .reporting import report
            result = report(args.run, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
