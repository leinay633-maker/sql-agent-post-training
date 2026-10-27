"""python -m exec_feedback.v9 <freeze|train|steps|evaluate|report|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v9")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--family", required=True, choices=["RL", "RAFT", "RLhi"])
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--until", default=None, help="absolute end of the online run, ISO with offset")
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    e.add_argument("--stop", default=None)
    e.add_argument("--mem", type=float, default=None)
    r = sub.add_parser("report")
    r.add_argument("--procs", type=int, default=64)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    for name in ("steps", "status"):
        sub.add_parser(name)
    for p in (t, e, r, g, sub.choices["steps"], sub.choices["status"]):
        p.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "freeze":
        from .config import load_settings
        from .freeze import freeze
        result = freeze(args.registration, args.raw_root, args.out, load_settings(args.config),
                        models_root=args.models_root, manifest_path=args.models_manifest, imports=args.imports,
                        smoke_limit=args.smoke_limit)
        result = {k: result[k] for k in ("identity", "counts", "databases", "gold")}
    elif args.cmd == "train":
        from .online import train_online
        result = train_online(args.run, family=args.family, seed=args.seed, until=args.until)
        result = {k: result.get(k) for k in ("finished", "stopped_by", "steps_completed", "optimizer_updates",
                                             "last_milestone", "lora_check")}
    elif args.cmd == "steps":
        from .evaluate import common_steps
        result = common_steps(args.run)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.run, condition=args.condition, role=args.role, stop=args.stop, mem=args.mem)
        result = {k: result.get(k) for k in ("condition", "role", "step", "episodes", "registered_tasks", "correct",
                                             "complete", "wall_seconds", "engine_load_seconds")}
        if not result["complete"]:
            json.dump(result, sys.stdout, default=str)
            print("\nincomplete evaluation (deadline)")
            return 1
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run, procs=args.procs).items() if k in ("decisions", "select_final")}
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "v08_run": frozen["imports"]["v08_run"],
                  "smoke_limit": frozen["smoke_limit"]}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    from .shutdown import hard_exit  # vLLM processes can hang at interpreter exit (see shutdown.py)

    try:
        code = main()
    except KeyboardInterrupt:  # the scheduler's timeout sends SIGINT
        traceback.print_exc()
        code = 130
    except SystemExit as exc:  # argparse errors keep their code
        code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
        if exc.code is not None and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
    except BaseException:  # noqa: BLE001 - reported as the interpreter would, then the same exit path
        traceback.print_exc()
        code = 1
    hard_exit(code)
