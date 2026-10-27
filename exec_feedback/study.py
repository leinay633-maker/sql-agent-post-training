"""A restartable local study: prepare -> baselines -> paired SFT -> eval -> report."""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .core import Limits, Task, digest, read_jsonl, write_jsonl
from .configuration import load_config
from .corpus import create_pairs
from .datasets import choose_tasks, load_prepared, prepare_data, screen_references, validate_gold
from .evaluation import evaluate_run
from .models import create_policy, load_tokenizer
from .pipeline import code_revision
from .reporting import compare_runs
from .storage import CallBudget, append_record, atomic_json, code_fingerprint, exclusive_lock, fingerprint, read_json


def study_dir(config: dict[str, Any]) -> Path:
    return Path(config["experiment"]["output_root"]) / config["name"]


def stage_names(config: dict[str, Any]) -> list[str]:
    stages = ["prepare", "gold", "base_direct", "base_feedback"]
    if config["experiment"]["collection"] != "injected":
        stages += ["collect"]
    stages += ["pairs"]
    for seed in config["experiment"]["seeds"]:
        stages += [f"train_success_{seed}", f"eval_success_{seed}", f"train_repair_{seed}", f"eval_repair_{seed}", f"report_{seed}"]
    return stages


def stage_dependencies(config: dict[str, Any]) -> dict[str, list[str]]:
    """Which completed stages each stage reads; stage_names() is a topological order of this."""
    deps = {"prepare": [], "gold": ["prepare"], "base_direct": ["gold"], "base_feedback": ["gold"]}
    if config["experiment"]["collection"] != "injected":
        deps["collect"] = ["gold"]
    deps["pairs"] = ["collect"] if "collect" in deps else ["gold"]
    for seed in config["experiment"]["seeds"]:
        for variant in ("success", "repair"):
            deps[f"train_{variant}_{seed}"] = ["pairs"]
            deps[f"eval_{variant}_{seed}"] = [f"train_{variant}_{seed}"]
        deps[f"report_{seed}"] = ["base_direct", "base_feedback", f"eval_success_{seed}", f"eval_repair_{seed}"]
    return deps


def plan(config: dict[str, Any]) -> dict[str, Any]:
    e = config["experiment"]
    missing = []
    if not config["data"]["terms_accepted"]: missing.append("data.terms_accepted")
    if not Path(config["data"]["raw_root"]).exists(): missing.append("data.raw_root")
    if config["model"]["backend"] != "transformers": missing.append("complete study requires model.backend=transformers; HTTP is for standalone evaluation/service")
    if e["final_test"] and not config["data"]["holdout_json"]: missing.append("data.holdout_json for final test")
    upper = None
    if e["train_limit"] and (e["test_limit"] if e["final_test"] else e["validation_limit"]):
        n = e["test_limit"] if e["final_test"] else e["validation_limit"]
        upper = n * (1 + e["max_steps"] + 2*len(e["seeds"])*e["max_steps"])
        if e["collection"] != "injected": upper += e["train_limit"] * e["max_steps"]
    return {"study": str(study_dir(config)), "stages": stage_names(config), "missing_connections": missing,
            "maximum_model_calls_estimate": upper, "configured_call_cap": config["resources"]["max_model_calls"],
            "training_runs": 2 * len(e["seeds"]), "max_train_steps_each": config["training"]["max_steps"],
            "parallel_gpus": config["resources"]["gpus"],
            "network_downloads_allowed": config["model"]["allow_download"],
            "gpu_rental_automatic": False, "cost_estimate": None,
            "note": "Runtime and quality are measured on execution; no estimated GPU speed is reported as a result."}


def screened_out(config: dict[str, Any], root: Path) -> set[str]:
    """Ids whose reference SQL failed the pre-selection screen; empty when screening is off."""
    if not config["experiment"]["gold_screen"]:
        return set()
    return set(read_json(root / "prepared" / "gold_screen" / "excluded.json")["excluded"])


def _selections(config: dict[str, Any], root: Path) -> tuple[list[Task], list[Task], list[Task]]:
    tasks = load_prepared(root / "prepared")
    e = config["experiment"]
    excluded = screened_out(config, root)
    pool = [t for t in tasks if t.id not in excluded]
    train = choose_tasks([t for t in pool if t.split == "train"], e["train_limit"], config["data"]["seed"], stratified=e["stratified_sampling"])
    split = "test" if e["final_test"] else "validation"
    eval_limit = e["test_limit"] if e["final_test"] else e["validation_limit"]
    evaluated = choose_tasks([t for t in pool if t.split == split], eval_limit, config["data"]["seed"], stratified=e["stratified_sampling"])
    if not train or not evaluated:
        raise ValueError("Training and selected evaluation split must both be nonempty")
    if e["gold_screen"] and (len(train) < e["train_limit"] or len(evaluated) < eval_limit):
        raise ValueError("Too few tasks left after the reference screen; do not silently shrink the registered study")
    if any(t.fixture for t in tasks) and not config["resources"]["allow_fixture"]:
        raise ValueError("Fixture tasks cannot enter a real study")
    return tasks, train, evaluated


def execute_stage(config: dict[str, Any], root: Path, stage: str) -> dict[str, Any]:
    if stage not in stage_names(config):
        raise ValueError("Unknown study stage")
    e, r = config["experiment"], config["resources"]
    database_root = Path(config["data"]["raw_root"])
    limits = Limits(**config["limits"])
    if stage == "prepare":
        result = prepare_data(config["data"], root / "prepared")
        if e["gold_screen"]:
            # Screen the whole candidate pool before sampling, so broken BIRD
            # references are excluded with a recorded reason instead of
            # aborting the gold stage; the registered denominator stays frozen.
            pool = [t for t in load_prepared(root / "prepared") if t.split in ("train", "validation")]
            screen = screen_references(pool, database_root, root / "prepared" / "gold_screen", limits, r["workers"])
            result = {**result, "gold_screen": {k: screen[k] for k in ("candidates", "statuses")}}
        _, training, evaluated = _selections(config, root)
        selected_path = root / "selection.json"
        selection = {"train": [t.id for t in training], "evaluation": [t.id for t in evaluated],
                     "split": "test" if e["final_test"] else "validation",
                     "sampling": "database/difficulty-balanced" if e["stratified_sampling"] else "hash",
                     "evaluation_databases": sorted({t.db_id for t in evaluated}),
                     "train_databases": sorted({t.db_id for t in training})}
        if e["gold_screen"]:
            selection["gold_screen_excluded"] = sorted(screened_out(config, root))
        if selected_path.exists() and read_json(selected_path) != selection:
            raise ValueError("Registered selection changed")
        atomic_json(selected_path, selection)
        actual_upper = len(evaluated)*(1+e["max_steps"]+2*len(e["seeds"])*e["max_steps"])
        if e["collection"] != "injected": actual_upper += len(training)*e["max_steps"]
        if actual_upper > r["max_model_calls"]:
            raise ValueError(f"Planned maximum {actual_upper} model calls exceeds cap {r['max_model_calls']}; edit config before starting a new study")
        return result
    tasks, training, evaluated = _selections(config, root)
    if stage == "gold":
        result = validate_gold(training + evaluated, database_root, root / "gold", limits, r["workers"])
        if result["invalid_ids"]:
            raise ValueError("Reference-query preflight failed; see gold/gold.jsonl. No rows were silently removed.")
        return result
    if stage == "pairs":
        traces = read_jsonl(root / "collect" / "episodes.jsonl") if e["collection"] != "injected" else []
        if (root / "pairs").exists():
            if (root / "pairs" / "summary.json").exists():
                result = read_json(root / "pairs" / "summary.json")
                if result["minimum_pairs_met"]: return result
                raise ValueError("Corpus selection did not meet the configured minimum; inspect pairs/rejected.jsonl")
            # Derived corpus interrupted before finalization: retain it, then
            # deterministically rebuild from the already journaled model traces.
            (root / "pairs").rename(root / f"pairs.interrupted-{time.time_ns()}")
        tokenizer = load_tokenizer(config["model"])
        result = create_pairs(training, database_root, root / "pairs", traces=traces,
                mode=e["collection"], limits=limits, max_steps=e["max_steps"], tokenizer=tokenizer,
                max_length=config["training"]["max_length"], min_pairs=e["min_pairs"], allow_fixture=r["allow_fixture"],
                semantic=e["semantic_augmentation"], review_success=e["review_success"])
        if not result["minimum_pairs_met"]:
            raise ValueError("Not enough length-checked pairs; all rejected cases retained in pairs/rejected.jsonl")
        return result
    if stage.startswith("train_"):
        from .train import fit
        _, variant, seed = stage.split("_")
        return fit(root / "pairs" / f"{variant}.jsonl", config["model"], config["training"],
                   root / stage, seed=int(seed), allow_fixture=r["allow_fixture"])
    if stage.startswith("report_"):
        seed = stage.split("_")[-1]
        runs = {"A_base_direct": root / "base_direct", "B_base_feedback": root / "base_feedback",
                "C_success_sft": root / f"eval_success_{seed}", "D_repair_sft": root / f"eval_repair_{seed}"}
        screen = None
        if e["gold_screen"]:
            screen = read_json(root / "prepared" / "gold_screen" / "excluded.json")
            screen = {k: screen[k] for k in ("candidates", "excluded", "statuses")}
        result = compare_runs(runs, root / stage, screen=screen)
        result["corpus"] = read_json(root / "pairs" / "summary.json")
        atomic_json(root / stage / "comparison.json", result)
        return {"report": str(root / stage / "report.html"), "tasks": result["tasks"]}
    adapter = None
    if stage.startswith("eval_"):
        _, variant, seed = stage.split("_")
        adapter = root / f"train_{variant}_{seed}" / "adapter"
    budget = CallBudget(root / "budget.json", r["max_model_calls"], r["max_active_seconds"])
    policy = create_policy(config["model"], adapter, budget)
    try:
        direct = stage == "base_direct"
        return evaluate_run(training if stage == "collect" else evaluated, database_root, policy, root / stage,
                            max_steps=1 if direct else e["max_steps"], feedback=not direct,
                            limits=limits, condition=stage, resume=True, batch_size=config["model"]["batch_size"],
                            review_success=e["review_success"])
    finally:
        policy.close()


def _artifacts(root: Path, stage: str) -> dict[str, str]:
    stage_root = root / ("prepared" if stage == "prepare" else stage)
    artifacts = {p.relative_to(root).as_posix(): digest(p) for p in stage_root.rglob("*")
                 if p.is_file() and not p.name.startswith(".")}
    if stage == "prepare": artifacts["selection.json"] = digest(root / "selection.json")
    if not artifacts: raise RuntimeError(f"Stage did not produce artifacts: {stage}")
    return artifacts


def _remaining(config: dict[str, Any], active: float) -> float:
    remaining = config["resources"]["max_active_seconds"] - active
    if os.environ.get("EFA_STOP_AT_UNIX"):
        remaining = min(remaining, float(os.environ["EFA_STOP_AT_UNIX"]) - time.time())
    return remaining


def _run_parallel(config: dict[str, Any], root: Path, manifest: dict[str, Any], targets: list[str]) -> None:
    """Run independent stages concurrently, one GPU each; the manifest is written only here.

    Stage processes are the same as in the sequential path, so per-stage results
    do not depend on scheduling. Wall time, not summed stage time, is charged.
    """
    gpus = config["resources"]["gpus"]
    raw_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [str(i) for i in range(gpus)] if raw_devices is None else [d.strip() for d in raw_devices.split(",")]
    if any(not d or d.startswith("-") for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct enabled device identifiers")
    if len(devices) < gpus:
        raise ValueError("resources.gpus exceeds the devices in CUDA_VISIBLE_DEVICES; choose hardware/config before freezing")
    free = devices[:gpus]
    deps = stage_dependencies(config)
    manifest_path = root / "study.json"
    done = {s for s in targets if manifest["stages"].get(s, {}).get("status") == "complete"}
    running: dict[str, tuple[subprocess.Popen, str, Any]] = {}
    base_active, started = manifest["active_seconds"], time.monotonic()
    failure: BaseException | None = None

    def save() -> None:
        manifest["active_seconds"] = base_active + time.monotonic() - started
        atomic_json(manifest_path, manifest)

    def export(label: str) -> None:
        if not config["resources"]["export_stages"]:
            return
        from .handoff import export_results
        skip = {"prepared" if s == "prepare" else s for s in running} | {f"{s}.log" for s in running}
        if running:
            skip.add("budget.json")  # mutable shared counter; final export includes it
        try:
            export_results(root, Path(config["resources"]["export_root"]), label=label, lock=False, skip=skip)
        except Exception as exc:  # A racing file must not stop GPU work; the final export retries.
            with (root / "export-errors.log").open("a", encoding="utf-8") as log:
                log.write(f"{time.time()} {label} {type(exc).__name__}: {exc}\n")

    try:
        while True:
            for stage in targets:
                if failure is not None or not free:
                    break
                if stage in done or stage in running or not all(d in done for d in deps.get(stage, [])):
                    continue
                if _remaining(config, base_active + time.monotonic() - started) <= 0:
                    failure = RuntimeError("Study wall-time budget exhausted; no further stages were started")
                    break
                device = free.pop(0)
                log = (root / f"{stage}.log").open("a", encoding="utf-8")
                command = [sys.executable, "-m", "exec_feedback.manage", "_stage", "--config", str(root / "resolved-config.json"), "--name", stage]
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env={**os.environ, "CUDA_VISIBLE_DEVICES": device})
                except BaseException as exc:
                    log.close()
                    free.append(device)
                    manifest["stages"][stage] = {"status": "interrupted", "error": type(exc).__name__}
                    raise
                running[stage] = (process, device, log)
                manifest["status"] = "running"
                manifest["stages"][stage] = {"status": "running", "started_unix": time.time(), "device": device}
                save()
            if not running:
                if failure is None and set(targets) - done:
                    failure = RuntimeError("Unfinished stages have no runnable dependency path")
                break
            time.sleep(1)
            out_of_time = _remaining(config, base_active + time.monotonic() - started) <= 0
            for stage, (process, device, log) in list(running.items()):
                code = process.poll()
                if code is None and out_of_time:
                    process.kill()
                    code = process.wait()
                    failure = failure or subprocess.TimeoutExpired(stage, 0)
                if code is None:
                    continue
                log.close()
                del running[stage]
                free.append(device)
                record = manifest["stages"][stage]
                try:
                    if code:
                        raise RuntimeError(f"Stage {stage} failed (exit {code}); details: {root / f'{stage}.log'}")
                    record.update(status="complete", artifacts=_artifacts(root, stage))
                    done.add(stage)
                except Exception as exc:
                    record.update(status="interrupted", error=type(exc).__name__)
                    failure = failure or exc
                save()
                if record["status"] == "complete":
                    export(stage)
    except BaseException as exc:
        for stage, (process, _, log) in running.items():
            process.kill()
            process.wait()
            log.close()
            manifest["stages"][stage].update(status="interrupted", error=type(exc).__name__)
        manifest["status"] = "interrupted"
        save()
        raise
    if failure is not None:
        manifest["status"] = "interrupted"
        save()
        raise failure
    save()


def run_study(config_path: Path, *, execute: bool = False, until: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    details = plan(config)
    if until is not None and until not in stage_names(config):
        raise ValueError("Unknown --until stage")
    if not execute:
        return details
    if details["missing_connections"]:
        raise ValueError("Missing connections: " + ", ".join(details["missing_connections"]))
    root = study_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"config": config, "code": code_fingerprint()})
    with exclusive_lock(root / ".study.lock"):
        manifest_path = root / "study.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["signature"] != signature:
                raise ValueError("Study config/code changed; use a new study name")
        else:
            manifest = {"signature": signature, "config": config, "stages": {}, "active_seconds": 0.0,
                        "status": "created", "created_unix": time.time(),
                        "provenance": code_revision()}
            atomic_json(manifest_path, manifest)
            atomic_json(root / "resolved-config.json", config)
        if config["resources"]["gpus"] > 1:
            names = stage_names(config)
            targets = names[: names.index(until) + 1] if until else names
            for stage in targets:
                previous = manifest["stages"].get(stage, {})
                if previous.get("status") == "complete" and any(
                        not (root / name).is_file() or digest(root / name) != value
                        for name, value in previous["artifacts"].items()):
                    raise ValueError(f"Completed stage artifacts changed: {stage}")
            _run_parallel(config, root, manifest, targets)
        for stage in stage_names(config) if config["resources"]["gpus"] == 1 else []:
            previous = manifest["stages"].get(stage, {})
            if previous.get("status") == "complete":
                if any(not (root / name).is_file() or digest(root / name) != value
                       for name, value in previous["artifacts"].items()):
                    raise ValueError(f"Completed stage artifacts changed: {stage}")
                if until == stage:
                    break
                continue
            remaining = config["resources"]["max_active_seconds"] - manifest["active_seconds"]
            if os.environ.get("EFA_STOP_AT_UNIX"):
                remaining = min(remaining, float(os.environ["EFA_STOP_AT_UNIX"]) - time.time())
            if remaining <= 0:
                raise RuntimeError("Study wall-time budget exhausted; no further stages were started")
            started = time.monotonic()
            manifest["status"] = "running"
            manifest["stages"][stage] = {"status": "running", "started_unix": time.time()}
            atomic_json(manifest_path, manifest)
            log_path = root / f"{stage}.log"
            command = [sys.executable, "-m", "exec_feedback.manage", "_stage", "--config", str(root / "resolved-config.json"), "--name", stage]
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    # Each stage gets a fresh process; GPU memory is released between training/evaluation.
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=False)
                if process.returncode:
                    raise RuntimeError(f"Stage {stage} failed (exit {process.returncode}); details: {log_path}")
                manifest["stages"][stage].update(status="complete", artifacts=_artifacts(root, stage))
            except BaseException as exc:
                manifest["status"] = "interrupted"
                manifest["stages"][stage].update(status="interrupted", error=type(exc).__name__)
                raise
            finally:
                manifest["active_seconds"] += time.monotonic() - started
                atomic_json(manifest_path, manifest)
            if config["resources"]["export_stages"]:
                from .handoff import export_results
                export_results(root, Path(config["resources"]["export_root"]), label=stage, lock=False)
            if until == stage:
                break
        manifest["status"] = "complete" if all(manifest["stages"].get(s, {}).get("status") == "complete" for s in stage_names(config)) else "paused"
        atomic_json(manifest_path, manifest)
        if config["resources"]["export_stages"]:
            from .handoff import export_results
            export_results(root, Path(config["resources"]["export_root"]), label=manifest["status"], lock=False)
    return {"study": str(root), "status": manifest["status"], "reports": [str(root / f"report_{s}" / "report.html") for s in config["experiment"]["seeds"]],
            "serve": f"python -m exec_feedback serve --study {root}"}
