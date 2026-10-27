"""Turnkey project commands; legacy stage-0 commands remain available."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .configuration import initialize, load_config
from .core import Limits
from .datasets import load_prepared, prepare_data, validate_gold
from .downloads import fetch_bird
from .models import create_policy, model_identity
from .storage import atomic_json, read_json
from .study import execute_stage, plan, run_study, study_dir

COMMANDS = {"init", "plan", "prepare", "validate-gold", "study", "status", "serve", "fetch-bird", "check", "baseline", "report", "final-test", "_stage"}


def connection_check(config: dict[str, Any] | None = None, probe: bool = False) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "fastapi", "uvicorn", "pytest"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    result: dict[str, Any] = {"python": sys.version, "packages": packages, "gpu_probed": False,
                             "api_key_present": bool(os.environ.get("EFA_API_KEY")),
                             "server_key_present": bool(os.environ.get("EFA_SERVER_KEY")),
                             "remote_calls": 0, "automatic_gpu_rental": False}
    if probe:
        try:
            import torch
            result.update(gpu_probed=True, cuda_available=torch.cuda.is_available(),
                          gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        except ImportError:
            result["gpu_error"] = "torch not installed"
    if config:
        result["plan"] = plan(config)
    return result


def server_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.study:
        state = read_json(args.study / "study.json")
        if state["status"] != "complete":
            raise ValueError("Study has not completed; use --config to serve the base model instead")
        config = copy.deepcopy(state["config"])
        seed = args.seed if args.seed is not None else config["experiment"]["seeds"][0]
        trained = args.study / f"train_{args.variant}_{seed}"
        check = read_json(trained / "result.json")
        if not check.get("adapter_reload_verified"):
            raise ValueError("Adapter has not passed reload verification")
        if model_identity(config["model"], trained / "adapter")["adapter_hash"] != check["adapter_hash"]:
            raise ValueError("Completed adapter files changed")
        config["service"]["adapter"] = str((trained / "adapter").resolve())
    else:
        config = load_config(args.config)
    if args.host: config["service"]["host"] = args.host
    if args.port: config["service"]["port"] = args.port
    return config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="efa", description="Execution-feedback agent research and service")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--config", type=Path, default=Path("local.json"))
    fetch = sub.add_parser("fetch-bird")
    fetch.add_argument("--part", choices=["train", "dev"], required=True)
    fetch.add_argument("--out", type=Path, default=Path("data/raw"))
    fetch.add_argument("--accept-terms", action="store_true")
    fetch.add_argument("--max-download-gb", type=int, default=40)
    for command in ("plan", "prepare", "validate-gold", "study", "status", "check", "baseline", "_stage"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", type=Path, default=Path("local.json"))
        if command == "study":
            cmd.add_argument("--execute", action="store_true")
            cmd.add_argument("--until", help="Stop after this completed stage, e.g. gold; no data/model config mutation")
        if command == "check": cmd.add_argument("--probe-gpu", action="store_true")
        if command == "_stage": cmd.add_argument("--name", required=True)
        if command == "baseline":
            cmd.add_argument("--out", type=Path, required=True)
            cmd.add_argument("--direct", action="store_true")
            cmd.add_argument("--split", choices=["validation", "test"])
            cmd.add_argument("--limit", type=int)
            cmd.add_argument("--study", type=Path, help="Evaluate exactly this study's registered evaluation tasks (e.g. another model on the same questions)")
            cmd.add_argument("--train-smoke-tasks", type=int, help="With --study: inference only on N registered TRAIN tasks, never LoRA training")
    server = sub.add_parser("serve")
    group = server.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path, default=Path("local.json"))
    group.add_argument("--study", type=Path)
    server.add_argument("--variant", choices=["success", "repair"], default="repair")
    server.add_argument("--seed", type=int)
    server.add_argument("--host")
    server.add_argument("--port", type=int)
    report = sub.add_parser("report")
    report.add_argument("--run", action="append", required=True, help="label=run_directory; repeat for each condition")
    report.add_argument("--out", type=Path, required=True)
    final = sub.add_parser("final-test")
    final.add_argument("--study", type=Path, required=True)
    final.add_argument("--execute", action="store_true")
    return p


def final_test(study: Path, execute: bool) -> dict[str, Any]:
    from .datasets import choose_tasks
    from .evaluation import evaluate_run
    from .reporting import compare_runs
    from .storage import CallBudget, exclusive_lock, fingerprint
    saved = read_json(study / "study.json")
    if saved["status"] != "complete":
        raise ValueError("Finish the validation study before final testing")
    config = saved["config"]
    tasks = load_prepared(study / "prepared")
    selected = choose_tasks([t for t in tasks if t.split == "test"], config["experiment"]["test_limit"], config["data"]["seed"])
    if not selected:
        raise ValueError("No registered holdout: connect data.holdout_json before preparing the study")
    stages = ["base_direct", "base_feedback"] + [f"{v}_{s}" for s in config["experiment"]["seeds"] for v in ("success", "repair")]
    count = len(selected) * (1 + (len(stages)-1)*config["experiment"]["max_steps"])
    if not execute:
        return {"tasks": len(selected), "stages": stages, "upper_model_calls": count, "retraining": False,
                "note": "Requires explicit --execute. Completed final test is never resampled."}
    if count > config["resources"]["max_model_calls"]:
        raise ValueError("Final evaluation would exceed configured call budget")
    output = study / "final-test"
    output.mkdir(exist_ok=True)
    with exclusive_lock(output / ".lock"):
        if (output / "complete.json").exists(): return read_json(output / "complete.json")
        checked = validate_gold(selected, Path(config["data"]["raw_root"]), output / "gold", Limits(**config["limits"]))
        if checked["invalid_ids"]: raise ValueError("Held-out gold preflight failed")
        budget = CallBudget(output / "budget.json", config["resources"]["max_model_calls"], config["resources"]["max_active_seconds"])
        for stage in stages:
            adapter = None if stage.startswith("base_") else study / f"train_{stage}" / "adapter"
            policy = create_policy(config["model"], adapter, budget)
            try:
                direct = stage == "base_direct"
                evaluate_run(selected, Path(config["data"]["raw_root"]), policy, output / stage,
                             max_steps=1 if direct else config["experiment"]["max_steps"], feedback=not direct,
                             limits=Limits(**config["limits"]), condition=stage, batch_size=config["model"].get("batch_size", 1),
                             review_success=config["experiment"].get("review_success", False))
            finally:
                policy.close()
        reports = []
        for seed in config["experiment"]["seeds"]:
            folder = output / f"report_{seed}"
            compare_runs({"A_base_direct":output/"base_direct", "B_base_feedback":output/"base_feedback",
                          "C_success_sft":output/f"success_{seed}", "D_repair_sft":output/f"repair_{seed}"}, folder)
            reports.append(str(folder / "report.html"))
        result = {"status":"complete", "reports":reports, "tasks":len(selected), "retraining":False}
        atomic_json(output / "complete.json", result)
        return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"--help", "-h"}:
        parser().print_help()
        print("Legacy commands: doctor demo hash import-bird audit eval sft")
        return 0
    if argv[0] not in COMMANDS:
        from .cli import main as legacy
        return legacy(argv)
    args = parser().parse_args(argv)
    try:
        if args.command == "init": result = initialize(args.config)
        elif args.command == "fetch-bird": result = fetch_bird(args.part, args.out, accept_terms=args.accept_terms, max_download_gb=args.max_download_gb)
        elif args.command == "serve":
            from .server import serve
            serve(server_config(args))
            return 0
        elif args.command == "report":
            from .reporting import compare_runs
            pairs = [part.split("=", 1) for part in args.run]
            if any(len(part) != 2 for part in pairs) or len({p[0] for p in pairs}) != len(pairs):
                raise ValueError("Use unique --run label=directory entries")
            result = compare_runs({k:Path(v) for k,v in pairs}, args.out)
        elif args.command == "final-test": result = final_test(args.study, args.execute)
        elif args.command == "check": result = connection_check(load_config(args.config) if args.config.exists() else None, args.probe_gpu)
        elif args.command == "study": result = run_study(args.config, execute=args.execute, until=args.until)
        else:
            config = load_config(args.config)
            root = study_dir(config)
            if args.command == "plan": result = plan(config)
            elif args.command == "prepare": result = prepare_data(config["data"], root / "prepared")
            elif args.command == "validate-gold":
                tasks = load_prepared(root / "prepared")
                result = validate_gold(tasks, Path(config["data"]["raw_root"]), root / "gold-all", Limits(**config["limits"]), config["resources"]["workers"])
            elif args.command == "status": result = read_json(root / "study.json") if (root / "study.json").exists() else {"status":"not_started", "study":str(root)}
            elif args.command == "baseline":
                from .datasets import choose_tasks
                from .evaluation import evaluate_run
                from .storage import CallBudget
                if args.study:
                    from .registered_baseline import run_registered
                    if args.split is not None or args.limit is not None:
                        raise ValueError("--study fixes the task set; do not combine it with --split/--limit")
                    result = run_registered(config, args.study, args.out, direct=args.direct,
                                            train_smoke_tasks=args.train_smoke_tasks)
                else:
                    if args.train_smoke_tasks is not None:
                        raise ValueError("--train-smoke-tasks requires --study")
                    tasks = choose_tasks([t for t in load_prepared(root / "prepared") if t.split == (args.split or "validation")],
                                         64 if args.limit is None else args.limit, config["data"]["seed"])
                    budget = CallBudget(args.out.with_suffix(".budget.json"), config["resources"]["max_model_calls"], config["resources"]["max_active_seconds"])
                    policy = create_policy(config["model"], budget=budget)
                    try:
                        result = evaluate_run(tasks, Path(config["data"]["raw_root"]), policy, args.out, max_steps=1 if args.direct else config["experiment"]["max_steps"], feedback=not args.direct, limits=Limits(**config["limits"]), batch_size=config["model"]["batch_size"], review_success=config["experiment"]["review_success"])
                    finally: policy.close()
            else: result = execute_stage(config, root, args.name)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
