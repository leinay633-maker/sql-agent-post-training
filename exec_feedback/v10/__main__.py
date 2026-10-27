"""python -m exec_feedback.v10 <freeze|merge|gate|run|rollout|evaluate|pack|bench|report|combine|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.v10")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--name", required=True, help="PTR_seed<N> or RL_seed<N>")
    g = sub.add_parser("gate")
    t = sub.add_parser("run")
    t.add_argument("--family", required=True, choices=["RLF", "RLFa"])
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--until", default=None, help="absolute end of the online run, ISO with offset")
    t.add_argument("--worker-python", required=True, help="python of the vLLM venv")
    t.add_argument("--trainer-python", default=None, help="python of the trainer (default: this one)")
    t.add_argument("--log-dir", type=Path, default=None)
    w = sub.add_parser("rollout")
    w.add_argument("--family", required=True, choices=["RLF", "RLFa"])
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--worker", type=int, required=True)
    w.add_argument("--root", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True)
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    e.add_argument("--stop", default=None)
    e.add_argument("--mem", type=float, default=None)
    k = sub.add_parser("pack")
    k.add_argument("--name", required=True, help="a finished online run, e.g. RLF_seed42")
    k.add_argument("--exports", type=Path, required=True)
    b = sub.add_parser("bench")
    b.add_argument("--family", required=True, choices=["RLF", "RLFa"])
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--log-dir", type=Path, default=None)
    r = sub.add_parser("report")
    r.add_argument("--procs", type=int, default=64)
    r.add_argument("--smoke-run", type=Path, default=None)
    c = sub.add_parser("combine", help="one report over the bookings' main runs (each trained its own seeds)")
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("progress")
    p.add_argument("--stage", required=True)
    p.add_argument("--commit", default="unknown")
    p.add_argument("--file", type=Path, required=True)
    s = sub.add_parser("status")
    for parser in (m, g, t, w, e, k, b, r, p, s):
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
    elif args.cmd == "merge":
        from .merge import merge
        result = {k: v for k, v in merge(args.run, name=args.name).items() if k != "checksum"}
    elif args.cmd == "gate":
        from .gate import gate_weights
        result = gate_weights(args.run)
        code = 0 if result["passed"] else 1
    elif args.cmd == "run":
        from .launch import main_args
        code = main_args(args)
        out = args.run / "train" / f"{args.family}_seed{args.seed}"
        state = next((out / n for n in ("result.json", "stopped.json") if (out / n).exists()), None)
        result = {"exit": code, "state": json.loads(state.read_text(encoding="utf-8")) if state else None}
        if result["state"]:
            result["state"] = {k: result["state"].get(k) for k in ("finished", "stopped_by", "steps_completed",
                                                                   "optimizer_updates", "last_milestone")}
    elif args.cmd == "rollout":
        from .rollout import serve
        result = serve(args.run, family=args.family, seed=args.seed, worker=args.worker, root=args.root)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.run, condition=args.condition, role=args.role, stop=args.stop, mem=args.mem)
        result = {k: result.get(k) for k in ("condition", "role", "step", "episodes", "registered_tasks", "correct",
                                             "complete", "wall_seconds", "engine_load_seconds")}
        code = 0 if result["complete"] else 1
    elif args.cmd == "pack":
        from .pack import pack
        manifest = pack(args.run, name=args.name, exports=args.exports)
        result = {"name": manifest["name"], "step": manifest["step"], "bytes": manifest["bytes"],
                  "parts": len(manifest["parts"])}
    elif args.cmd == "bench":
        from .bench import bench
        result = bench(args.run, family=args.family, seed=args.seed, python=sys.executable,
                       log_dir=args.log_dir or args.run / "bench" / "logs")
        code = 0 if result["complete"] else 1
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run, procs=args.procs, smoke=args.smoke_run).items()
                  if k in ("decisions", "systems")}
    elif args.cmd == "combine":
        from .analysis import combined
        result = {k: v for k, v in combined(args.runs, args.out).items() if k in ("run", "decisions")}
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "v09_run": frozen["imports"]["v09"]["v09_run"],
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
