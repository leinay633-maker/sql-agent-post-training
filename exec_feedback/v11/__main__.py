"""python -m exec_feedback.v11 <freeze|prepass|train|steps|evaluate|report|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

GATE_FAILED = 3  # a finished pre-pass whose boundary share is outside the registered range: nothing trains


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v11")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    p = sub.add_parser("prepass")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--until", default=None, help="absolute end, ISO with offset (the pre-pass resumes from its chunks)")
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--leg", required=True, choices=["main", "explore"],
                   help="main: to the registered milestone, then pause; explore: on to explore_steps")
    t.add_argument("--until", default=None, help="absolute end of the online run, ISO with offset")
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    e.add_argument("--stop", default=None)
    e.add_argument("--mem", type=float, default=None)
    r = sub.add_parser("report")
    r.add_argument("--name", default="main", choices=["main", "final"])
    r.add_argument("--procs", type=int, default=64)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    for name in ("steps", "status"):
        sub.add_parser(name)
    for parser in (p, t, e, r, g, sub.choices["steps"], sub.choices["status"]):
        parser.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    code = 0
    if args.cmd == "freeze":
        from .config import load_settings
        from .freeze import freeze
        result = freeze(args.registration, args.raw_root, args.out, load_settings(args.config),
                        models_root=args.models_root, manifest_path=args.models_manifest, imports=args.imports,
                        smoke_limit=args.smoke_limit)
        result = {k: result[k] for k in ("identity", "counts", "databases", "gold")}
    elif args.cmd == "prepass":
        from .prepass import prepass
        result = prepass(args.run, seed=args.seed, until=args.until)
        if not result["finished"]:
            code = 1
        elif not result["gate"]["passed"]:
            code = GATE_FAILED
        result = {k: result.get(k) for k in ("finished", "stopped_by", "remaining", "seed", "tasks", "scorable",
                                             "boundary_share", "histogram", "gate", "last_attempt")}
    elif args.cmd == "train":
        from .freeze import load
        from .online import train_online
        settings = load(args.run, check_code=False)[0]["settings"]
        result = train_online(args.run, seed=args.seed, until=args.until,
                              stop_step=settings["planned_steps" if args.leg == "main" else "explore_steps"])
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
        code = 0 if result["complete"] else 1
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run, name=args.name, procs=args.procs).items()
                  if k in ("decisions", "prepass")}
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "v08_run": frozen["imports"]["v08_run"],
                  "v09_run": frozen["imports"]["v09"]["v09_run"], "smoke_limit": frozen["smoke_limit"]}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return code


if __name__ == "__main__":
    from ..v9.shutdown import hard_exit  # vLLM processes can hang at interpreter exit (see v9/shutdown.py)

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
