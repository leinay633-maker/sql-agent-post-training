"""python -m exec_feedback.v12 <freeze|serve-teacher|teacher-eval|train|steps|evaluate|report|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

GATE_FAILED = 3  # the teacher's scoring gate failed: nothing can train against it


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v12")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    v = sub.add_parser("serve-teacher", help="load the teacher, pass the scoring gate, serve until <home>/stop")
    v.add_argument("--home", type=Path, required=True)
    v.add_argument("--until", default=None)
    e2 = sub.add_parser("teacher-eval", help="the teacher on (run, role) pairs in order, one engine load")
    e2.add_argument("--job", nargs=2, action="append", metavar=("RUN", "ROLE"), required=True)
    e2.add_argument("--stop", default=None)
    t = sub.add_parser("train")
    t.add_argument("--family", required=True, choices=["OPD", "TROPD"])
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--until", default=None, help="absolute end of the online run, ISO with offset")
    t.add_argument("--teacher-home", type=Path, required=True)
    t.add_argument("--stop-step", type=int, default=None, help="smoke only: pause after this step")
    t.add_argument("--tag", default="")
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    e.add_argument("--stop", default=None)
    e.add_argument("--mem", type=float, default=None)
    r = sub.add_parser("report")
    r.add_argument("--procs", type=int, default=64)
    r.add_argument("--teacher-home", type=Path, default=None)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    g.add_argument("--teacher-home", type=Path, default=None)
    for name in ("steps", "status"):
        sub.add_parser(name)
    for parser in (v, t, e, r, g, sub.choices["steps"], sub.choices["status"]):
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
    elif args.cmd == "serve-teacher":
        from .teacher import run_service
        result = run_service(args.run, home=args.home, until=args.until)
        if not result["served"]:
            code = GATE_FAILED
        elif result["stopped_by"] == "engine_error":
            code = 1
        result = {k: result.get(k) for k in ("served", "stopped_by", "error", "requests", "sequences", "tokens",
                                             "busy_seconds", "serving_seconds", "gate")}
    elif args.cmd == "teacher-eval":
        from .evaluate import evaluate_teacher
        summaries = evaluate_teacher([(Path(run), role) for run, role in args.job], stop=args.stop)
        result = [{k: s.get(k) for k in ("run", "role", "episodes", "registered_tasks", "correct", "complete",
                                         "wall_seconds", "engine_load_seconds")} for s in summaries]
        code = 0 if len(summaries) == len(args.job) and all(s["complete"] for s in summaries) else 1
    elif args.cmd == "train":
        from .freeze import load
        from .online import teacher_client, train_online
        frozen = load(args.run, check_code=False)[0]
        client = teacher_client(args.teacher_home, frozen, args.family, args.seed, args.tag, until=args.until)
        result = train_online(args.run, family=args.family, seed=args.seed, until=args.until, teacher=client,
                              stop_step=args.stop_step)
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
        result = report(args.run, procs=args.procs, teacher_home=args.teacher_home)["decisions"]
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit, teacher_home=args.teacher_home)
        with args.file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "v08_run": frozen["imports"]["v08_run"],
                  "adapters": sorted(frozen["imports"]["adapters"]), "references": sorted(frozen["imports"]["references"]),
                  "smoke_limit": frozen["smoke_limit"]}
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
