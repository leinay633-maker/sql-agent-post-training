"""Explicit v0.5 stages; nothing loads a model or touches data without --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings, plan


def status(run: Path) -> dict:
    from ..storage import read_json

    result = {}
    for folder in sorted(run.glob("round*/samples")):
        shards = [read_json(p) for p in sorted(folder.glob("shard*.status.json"))]
        result[folder.parent.name] = {"shards": len(shards), "completed": sum(s["completed"] for s in shards),
                                      "correct": sum(s["correct"] for s in shards),
                                      "final": all(s["final"] for s in shards)}
    for path in sorted(run.glob("round*/train_*/result.json")):
        saved = read_json(path)
        result[path.parent.name] = {"completed": saved.get("training_completed"),
                                    "verified": saved.get("adapter_reload_verified")}
    for path in sorted(run.glob("eval/*/*/summary.json")):
        saved = read_json(path)
        result[f"eval/{path.parent.parent.name}/{path.parent.name}"] = {
            "complete": saved["complete"], "correct": saved["correct"], "tasks": saved["tasks"]}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="v0.5 execution-verified multi-turn RFT")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").add_argument("--config", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    for name in ("--config", "--source-study", "--raw-root", "--model-path", "--out"):
        prepare.add_argument(name, type=Path, required=True)
    prepare.add_argument("--dev-json", type=Path)
    sample = sub.add_parser("sample")
    sample.add_argument("--shard", type=int, required=True)
    sample.add_argument("--shards", type=int, required=True)
    sample.add_argument("--round", type=int, default=1)
    sample.add_argument("--policy", default="B", help="B for round 1; a frozen condition such as R_r1_seed42 later")
    sub.add_parser("pairs")
    prefer = sub.add_parser("prefer")
    prefer.add_argument("--seed", type=int, required=True)
    build = sub.add_parser("build")
    build.add_argument("--round", type=int, default=1)
    train = sub.add_parser("train")
    train.add_argument("--round", type=int, default=1)
    train.add_argument("--variant", choices=["R", "Rclean"], required=True)
    train.add_argument("--seed", type=int, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--condition", required=True)
    evaluate.add_argument("--role", choices=["dev", "validation", "test"], required=True)
    report = sub.add_parser("report")
    report.add_argument("--out", type=Path, required=True)
    sub.add_parser("status")
    for command in (sample, build, train, evaluate, report, prefer, sub.choices["pairs"], sub.choices["status"]):
        command.add_argument("--run", type=Path, required=True)
    for command in sub.choices.values():
        command.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = plan(load_settings(args.config))
        elif args.command == "status":
            result = status(args.run)
        elif not args.execute:
            parser.error("Explicit --execute is required; no work was started")
        elif args.command == "prepare":
            from .study import prepare as freeze
            result = freeze(args.source_study, args.raw_root, args.out, load_settings(args.config),
                            model_path=args.model_path, dev_json=args.dev_json)
        elif args.command == "sample":
            from .evaluation import frozen_adapter
            from .sampling import sample_shard
            from .study import load_run
            adapter = None if args.policy == "B" else frozen_adapter(args.run, load_run(args.run)[0], args.policy)
            result = sample_shard(args.run, shard=args.shard, shards=args.shards, round_=args.round, adapter=adapter)
        elif args.command == "build":
            from .selection import build as select
            result = select(args.run, args.round)
        elif args.command == "pairs":
            from .preference import build_pairs
            result = build_pairs(args.run)
        elif args.command == "prefer":
            from .preference import train_preference
            result = train_preference(args.run, seed=args.seed)
        elif args.command == "train":
            from .training import train_model
            result = train_model(args.run, round_=args.round, variant=args.variant, seed=args.seed)
        elif args.command == "evaluate":
            from .evaluation import evaluate as score
            result = score(args.run, args.condition, args.role)
        else:
            from .reporting import report as summarize
            result = summarize(args.run, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str)[:20000])
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
