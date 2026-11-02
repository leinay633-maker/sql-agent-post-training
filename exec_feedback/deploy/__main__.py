"""python -m exec_feedback.deploy <freeze|merge|evaluate|bench|report|api|demo|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.deploy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    m = sub.add_parser("merge")
    e = sub.add_parser("evaluate")
    e.add_argument("--condition", required=True, choices=["lora", "merged", "fp8"])
    e.add_argument("--role", required=True, choices=["dev", "validation", "test"])
    e.add_argument("--stop", default=None)
    b = sub.add_parser("bench")
    b.add_argument("--condition", required=True, choices=["lora", "merged", "fp8"])
    b.add_argument("--until", default=None, help="absolute end, ISO with offset: no new concurrency level after it")
    b.add_argument("--vllm-py", default=sys.executable)
    r = sub.add_parser("report")
    a = sub.add_parser("api", help="the analysis service from a JSON config (started by bench and demo)")
    a.add_argument("--config", type=Path, required=True)
    d = sub.add_parser("demo")
    d.add_argument("--condition", default="merged", choices=["lora", "merged", "fp8"])
    d.add_argument("--minutes", type=float, default=60.0)
    d.add_argument("--port", type=int, default=8082)
    d.add_argument("--vllm-py", default=sys.executable)
    d.add_argument("--key-file", type=Path, required=True)
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    st = sub.add_parser("status")
    for parser in (m, e, b, r, d, g, st):
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
        from .freeze import merge
        result = {k: v for k, v in merge(args.run).items() if k != "checksum"}
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.run, condition=args.condition, role=args.role, stop=args.stop)
        result = {k: result.get(k) for k in ("condition", "role", "episodes", "registered_tasks", "correct",
                                             "complete", "wall_seconds", "engine_load_seconds")}
        code = 0 if result["complete"] else 1
    elif args.cmd == "bench":
        from .bench import bench
        result = bench(args.run, condition=args.condition, vllm_python=args.vllm_py, main_python=sys.executable,
                       until=args.until)
        code = 0 if result["complete"] else 1
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run).items() if k in ("accuracy", "deltas")}
    elif args.cmd == "api":
        from ..server import serve
        serve(json.loads(args.config.read_text(encoding="utf-8")))
        return 0
    elif args.cmd == "demo":
        from .bench import demo
        result = demo(args.run, condition=args.condition, minutes=args.minutes, port=args.port,
                      vllm_python=args.vllm_py, main_python=sys.executable, key_file=args.key_file)
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "smoke_limit": frozen["smoke_limit"],
                  "adapter": frozen["imports"]["v11"]["adapters"]}
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return code


if __name__ == "__main__":
    from ..v9.shutdown import hard_exit  # vLLM processes can hang at interpreter exit (see v9/shutdown.py)

    try:
        code = main()
    except KeyboardInterrupt:  # the driver's timeout and the servers' shutdown send SIGINT
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
