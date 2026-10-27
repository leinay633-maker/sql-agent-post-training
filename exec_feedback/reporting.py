"""Paired statistics and self-contained reports; no placeholder experiment scores."""
from __future__ import annotations

import csv
import html
import itertools
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import read_jsonl, digest
from .storage import atomic_json, read_json


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0, center-radius), min(1, center+radius)]


def paired_difference(left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed: int = 42,
                      resamples: int = 2000) -> dict[str, Any]:
    a, b = {r["task_id"]:r for r in left}, {r["task_id"]:r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b) or not a:
        raise ValueError("Paired comparison requires identical, unique nonempty task IDs")
    if any(a[k]["db_id"] != b[k]["db_id"] for k in a):
        raise ValueError("Task database identity differs")
    # Cluster bootstrap by database, not IID questions from one database.
    groups: dict[str, list[int]] = defaultdict(list)
    for key in sorted(a):
        groups[a[key]["db_id"]].append(int(b[key]["correct"]) - int(a[key]["correct"]))
    values = list(groups.values())
    delta = sum(sum(g) for g in values) / len(a)
    ci = None
    if len(values) >= 2 and resamples > 0:
        rng = random.Random(seed)
        samples = []
        for _ in range(resamples):
            sampled = [rng.choice(values) for _ in values]
            samples.append(sum(sum(g) for g in sampled) / sum(len(g) for g in sampled))
        samples.sort()
        ci = [samples[int(.025*(len(samples)-1))], samples[int(.975*(len(samples)-1))]]
    return {"delta_right_minus_left": delta, "database_cluster_bootstrap_95ci": ci,
            "databases": len(values), "tasks": len(a), "resamples": resamples, "seed": seed,
            "right_only_correct": sum(not a[k]["correct"] and b[k]["correct"] for k in a),
            "left_only_correct": sum(a[k]["correct"] and not b[k]["correct"] for k in a),
            "note": "Database-cluster bootstrap is not a multiple-testing-adjusted significance claim."}


def compare_runs(runs: dict[str, Path], out: Path, screen: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("At least two real evaluation runs are required")
    out.mkdir(parents=True, exist_ok=True)
    all_records, manifests, scores, breakdowns = {}, {}, {}, {}
    task_ids = None
    for label, path in runs.items():
        summary = read_json(path / "summary.json")
        manifest = read_json(path / "manifest.json")
        if not summary["complete"] or summary["run_kind"] != "model_evaluation":
            raise ValueError("Incomplete or engineering-fixture runs cannot produce a research comparison")
        completed = read_json(path / "completed.json")
        if (digest(path / "episodes.jsonl") != completed["episodes_sha256"] or
                digest(path / "summary.json") != completed["summary_sha256"]):
            raise ValueError("Evaluation artifacts were modified after completion")
        records = read_jsonl(path / "episodes.jsonl")
        if any(type(r.get("correct")) is not bool for r in records):
            raise ValueError("Correctness values must be booleans")
        if any(r.get("fixture") for r in records) or len(records) != summary["tasks"]:
            raise ValueError("Invalid research records")
        current = set(manifest["task_ids"])
        if task_ids is not None and current != task_ids:
            raise ValueError("Runs do not share the same held-out tasks")
        if len(records) != len(current) or set(r["task_id"] for r in records) != current:
            raise ValueError("Manifest and episode tasks differ")
        task_ids = current
        all_records[label], manifests[label] = records, manifest
        successes = sum(bool(r["correct"]) for r in records)
        if summary["correct"] != successes or summary["strict_success_rate"] != successes / len(records):
            raise ValueError("Summary disagrees with per-task records")
        scores[label] = {**summary, "wilson_95ci_descriptive": wilson(successes, len(records))}
        groups = {}
        for dimension in ("db_id", "difficulty", "status"):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in records:
                grouped[row.get(dimension, "unknown")].append(row)
            groups[dimension] = {key: {"n":len(rows), "success_rate":sum(bool(r["correct"]) for r in rows)/len(rows)}
                                 for key, rows in grouped.items()}
        breakdowns[label] = groups
    pairs = {}
    for a, b in itertools.combinations(runs, 2):
        if (manifests[a]["dataset_id"] != manifests[b]["dataset_id"] or manifests[a]["limits"] != manifests[b]["limits"]
                or manifests[a]["code"] != manifests[b]["code"]):
            raise ValueError("Dataset or SQL limits differ across conditions")
        pa, pb = manifests[a]["provider"], manifests[b]["provider"]
        compare_keys = ("id", "revision", "local_snapshot", "temperature", "max_tokens", "max_context_tokens", "quantization", "device", "seed", "backend", "endpoint", "batch_size")
        if any(pa.get(k) != pb.get(k) for k in compare_keys):
            raise ValueError("Base model or generation settings differ across conditions")
        paired = paired_difference(all_records[a], all_records[b])
        paired["same_interaction_budget"] = (manifests[a]["max_steps"], manifests[a]["feedback"]) == (manifests[b]["max_steps"], manifests[b]["feedback"])
        paired["same_interaction_budget"] = paired["same_interaction_budget"] and manifests[a].get("review_success", False) == manifests[b].get("review_success", False) and manifests[a].get("batch_size", 1) == manifests[b].get("batch_size", 1)
        paired["training_attribution_allowed_by_budget"] = paired["same_interaction_budget"]
        pairs[f"{a} -> {b}"] = paired
    report = {"format": "efa-research-report-v2", "conditions": scores, "paired": pairs,
              "breakdowns": breakdowns, "tasks": len(task_ids),
              "primary_comparison": "C_success_sft -> D_repair_sft (pre-specified; others descriptive)",
              "precision_note": "256 questions is not a power guarantee; database clustering and few discordant pairs can leave wide intervals. One training seed gives exploratory evidence only.",
              "warning": "Custom execution matching, not an official BIRD leaderboard score. Direct vs feedback differences include extra inference."}
    screen_note = ""
    if screen is not None:
        # The denominator is the screened registry; say what was removed before sampling.
        report["gold_screen"] = screen
        screen_note = (f"Reference screen before sampling: {len(screen['excluded'])} of {screen['candidates']} "
                       f"candidate tasks excluded because their reference SQL failed ({screen['statuses']}); ids in comparison.json.")
    atomic_json(out / "comparison.json", report)
    with (out / "per-task.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "task_id", "database", "difficulty", "status", "correct", "model_steps", "elapsed_seconds", "sql"])
        for label, records in all_records.items():
            for row in records:
                writer.writerow([label, row["task_id"], row["db_id"], row.get("difficulty"), row["status"],
                                 row["correct"], len(row["steps"]), row.get("elapsed_seconds"), row.get("final_sql")])
    lines = ["# Execution-feedback experiment", "", report["warning"], "", "| Condition | Tasks | Correct | Success rate |", "|---|---:|---:|---:|"]
    body = []
    for label, score in scores.items():
        lines.append(f"| {label} | {score['tasks']} | {score['correct']} | {score['strict_success_rate']:.2%} |")
        body.append(f"<tr><td>{html.escape(label)}</td><td>{score['tasks']}</td><td>{score['correct']}</td><td>{score['strict_success_rate']:.2%}</td></tr>")
    if screen_note: lines.extend(["", screen_note])
    lines.extend(["", "## Paired differences (database-cluster bootstrap)", ""])
    for label, pair in pairs.items():
        lines.append(f"- {label}: delta={pair['delta_right_minus_left']:.4f}; CI={pair['database_cluster_bootstrap_95ci']}; same budget={pair['same_interaction_budget']}")
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    failures = []
    for label, records in all_records.items():
        for r in records:
            if not r["correct"]:
                failures.append(f"<details><summary>{html.escape(label+' / '+r['task_id']+' / '+r['status'])}</summary><pre>{html.escape(str(r.get('final_sql')))}</pre><pre>{html.escape(str(r['steps']))}</pre></details>")
    note_html = f"<p>{html.escape(screen_note)}</p>" if screen_note else ""
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>实验报告</title>
<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#17222a}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px}details{border-bottom:1px solid #ddd;padding:12px}small{color:#555}</style>
<h1>执行反馈 Agent · 实验报告</h1><p>逐题结果、失败分类和固定预算对照。未把工程测试当作模型效果。</p>
<table><tr><th>条件</th><th>题目</th><th>正确</th><th>成功率</th></tr>""" + "".join(body) + "</table>" + note_html + "<h2>配对差异</h2><pre>" + html.escape(str(pairs)) + "</pre><h2>失败案例</h2>" + "".join(failures) + "<p><small>" + html.escape(report["warning"]) + "</small></p></html>"
    (out / "report.html").write_text(page, encoding="utf-8")
    # Human review queue contains both discordant directions, not only improvements.
    if "C_success_sft" in all_records and "D_repair_sft" in all_records:
        left = {r["task_id"]: r for r in all_records["C_success_sft"]}
        with (out / "review-cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["task_id", "database", "C_correct", "D_correct", "C_SQL", "D_SQL", "manual_notes"])
            for row in all_records["D_repair_sft"]:
                prior = left[row["task_id"]]
                if prior["correct"] != row["correct"]:
                    writer.writerow([row["task_id"], row["db_id"], prior["correct"], row["correct"], prior.get("final_sql"), row.get("final_sql"), ""])
    return report
