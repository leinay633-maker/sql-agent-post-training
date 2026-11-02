"""python -m exec_feedback.passk <freeze|sample|report|progress|status>"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="exec_feedback.passk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    for name in ("registration", "raw-root", "models-root", "models-manifest", "imports", "config", "out"):
        f.add_argument(f"--{name}", type=Path, required=True)
    f.add_argument("--smoke-limit", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--condition", required=True)
    s.add_argument("--until", default=None, help="absolute end, ISO with offset (sampling resumes from its chunks)")
    r = sub.add_parser("report")
    r.add_argument("--offline", action="store_true",
                   help="on the PC, where the results come without the databases: do not re-hash them")
    g = sub.add_parser("progress")
    g.add_argument("--stage", required=True)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--file", type=Path, required=True)
    st = sub.add_parser("status")
    for parser in (s, r, g, st):
        parser.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    code = 0
    if args.cmd == "freeze":
        from .config import load_settings
        from .freeze import freeze
        result = freeze(args.registration, args.raw_root, args.out, load_settings(args.config),
                        models_root=args.models_root, manifest_path=args.models_manifest, imports=args.imports,
                        smoke_limit=args.smoke_limit)
        result = {k: result[k] for k in ("identity", "counts", "databases", "gold", "strata", "control", "conditions")}
    elif args.cmd == "sample":
        from .sample import sample
        result = sample(args.run, condition=args.condition, until=args.until)
        code = 0 if result["finished"] else 1
        result = {k: result.get(k) for k in ("finished", "stopped_by", "remaining", "condition", "tasks", "dialogues",
                                             "solved_tasks", "correct_dialogues", "sampling_seconds")}
    elif args.cmd == "report":
        from .analysis import report
        result = {k: v for k, v in report(args.run, databases=not args.offline).items() if k in ("verdict", "missing")}
    elif args.cmd == "progress":
        from .analysis import progress_entry
        entry = progress_entry(args.run, args.stage, args.commit)
        with args.file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(entry)
        result = {"appended": str(args.file)}
    else:
        from .freeze import load
        frozen, _ = load(args.run)
        result = {"identity": frozen["identity"], "counts": frozen["counts"], "strata": frozen["strata"],
                  "v08_run": frozen["imports"]["v08_run"], "v11_run": frozen["imports"]["v11"]["run"],
                  "v12_run": frozen["imports"]["v12"]["run"], "conditions": frozen["conditions"],
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
