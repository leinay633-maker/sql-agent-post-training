"""Paired database/seed uncertainty, without treating 3 x N as independent N."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ..core import digest
from ..storage import atomic_json, fingerprint, read_json, recover_records
from .study import load_run


def aligned(tasks: list[Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {r["task_id"]: r for r in rows}
    expected = {t.id for t in tasks}
    if len(lookup) != len(rows) or not set(lookup) <= expected:
        raise ValueError("Duplicate/foreign row; refuse denominator change")
    for task in tasks:
        row = lookup.get(task.id)
        if row is None:
            continue
        if row["db_id"] != task.db_id or type(row["correct"]) is not bool:
            raise ValueError("Invalid per-task database/correctness record")
        if row["correct"] != (row["status"] == "correct"):
            raise ValueError("Status/correctness mismatch")
    return [lookup.get(t.id, {"task_id": t.id, "db_id": t.db_id,
                             "correct": False, "status": "not_attempted"}) for t in tasks]


def paired_report(tasks: list[Any], baseline: list[dict[str, Any]],
                  runs: dict[str, list[dict[str, Any]]], *, draws: int = 2000,
                  seed: int = 20261009) -> dict[str, Any]:
    if not tasks or not runs or draws < 1:
        raise ValueError("Need a fixed task set, RL seeds and bootstrap draws")
    base = aligned(tasks, baseline)
    scored = {name: aligned(tasks, rows) for name, rows in runs.items()}
    complete = all(r["status"] != "not_attempted" for rows in [base, *scored.values()] for r in rows)
    fixture = any(t.fixture for t in tasks)
    per_seed = {}
    vectors = []
    for name, rows in scored.items():
        delta = [int(r["correct"]) - int(b["correct"]) for r, b in zip(rows, base)]
        vectors.append(delta)
        per_seed[name] = {"correct": sum(bool(r["correct"]) for r in rows),
                          "success_rate": None if fixture else mean(bool(r["correct"]) for r in rows),
                          "delta": None if fixture else mean(delta),
                          "improved_ids": [t.id for t, d in zip(tasks, delta) if d == 1],
                          "regressed_ids": [t.id for t, d in zip(tasks, delta) if d == -1],
                          "statuses": dict(Counter(r["status"] for r in rows))}
    groups = defaultdict(list)
    for index, task in enumerate(tasks):
        groups[task.db_id].append(index)
    interval = None
    if complete and not fixture and len(groups) >= 2 and len(vectors) >= 2:
        rng, values = random.Random(seed), []
        databases = sorted(groups)
        for _ in range(draws):
            columns = [i for db in rng.choices(databases, k=len(databases)) for i in groups[db]]
            chosen = rng.choices(vectors, k=len(vectors))
            values.append(mean(mean(row[i] for i in columns) for row in chosen))
        values.sort()
        interval = [values[int(0.025 * (draws - 1))], values[int(0.975 * (draws - 1))]]
    differences = [mean(v) for v in vectors]
    return {
        "tasks_per_condition": len(tasks), "databases": len(groups),
        "seeds": len(runs), "complete": complete, "fixture": fixture,
        "baseline_statuses": dict(Counter(r["status"] for r in base)),
        "baseline_success_rate": None if fixture else mean(bool(r["correct"]) for r in base),
        "per_seed": per_seed,
        "mean_delta": None if fixture else mean(differences),
        "seed_delta_sd": None if fixture or len(differences) < 2 else stdev(differences),
        "paired_database_seed_bootstrap_95": interval,
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "denominator": "all registered tasks, including errors and not_attempted",
        "scope": ("conditional on one frozen SFT warm start; "
                  "not independent SFT seeds or official BIRD scoring"),
        "warning": ("Three RL seeds and a larger validation subset do not guarantee "
                    "statistical power. No held-out test has run."),
    }


def report(run: Path, out: Path) -> dict[str, Any]:
    frozen, tasks = load_run(run, training=False)
    if frozen["settings"]["phase"] != "study":
        raise ValueError("Pilot is not a validation report")
    names = ["S4"] + [f"E4_seed_{s}" for s in frozen["settings"]["seeds"]]
    records, sealed = {}, []
    for name in names:
        root = run / f"eval_{name}"
        records[name] = []
        if not root.exists():
            continue
        meta = read_json(root / "manifest.json")
        signature = meta.pop("identity")
        if fingerprint(meta) != signature:
            raise ValueError("Evaluation manifest was modified")
        if meta["protocol"] != frozen["protocol"] or meta["run_identity"] != frozen["identity"]:
            raise ValueError("Cannot mix v0.3/foreign runs into v0.4")
        if meta["condition"] != name or meta["task_ids"] != [t.id for t in tasks]:
            raise ValueError("Evaluation task/condition identity mismatch")
        settings = frozen["settings"]
        expected = {"backend": "hf-v04-bucketed", "greedy": True,
                    "batch": settings["generation_batch"], "slots": settings["slots"],
                    "max_steps": 2, "max_tokens": settings["max_tokens"]}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError("Unequal v0.4 decoding budgets/backends")
        expected_hash = frozen["warm_adapter_hash"]
        if name != "S4":
            trained = read_json(run / name / "result.json")
            expected_hash = trained["adapter_hash"]
            if trained["run_identity"] != frozen["identity"]:
                raise ValueError("RL adapter is from another run")
        if meta["adapter_hash"] != expected_hash:
            raise ValueError("Evaluation used a different frozen adapter")
        if (root / "done.json").exists():
            done = read_json(root / "done.json")
            if done["identity"] != signature:
                raise ValueError("Completed evaluation identity changed")
            if set(done["files"]) != {"episodes.jsonl", "summary.json"}:
                raise ValueError("Incomplete evaluation evidence manifest")
            sealed.append(name)
            for path, expected in done["files"].items():
                if digest(root / path) != expected:
                    raise ValueError("Completed evaluation evidence changed")
        if (root / "episodes.jsonl").exists():
            records[name] = recover_records(root / "episodes.jsonl")
    result = paired_report(tasks, records.pop("S4"), records)
    if len(sealed) != len(names):
        result["complete"] = False
        result["paired_database_seed_bootstrap_95"] = None
    result["sealed_conditions"] = sealed
    result.update(protocol=frozen["protocol"], run_identity=frozen["identity"],
                  warm_corpus_origins=frozen["warm_corpus_origins"],
                  new_rl_origins={"natural_model": "all online trajectories", "rule_perturbation": 0})
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "comparison.json", result)
    return result
