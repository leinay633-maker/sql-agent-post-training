"""Registered v0.5 comparisons; every seed and every failure stays in the report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import digest
from ..reporting import paired_difference, wilson
from ..rl.reporting import aligned, paired_report
from ..storage import atomic_json, fingerprint, read_json, recover_records
from .evaluation import CONDITION
from .study import load_run

SCOPE = ("Three LoRA seeds share one sampled/selected RFT dataset; the interval resamples databases and "
         "seeds, not independent data collections. Project execution matching, not official BIRD EX.")


def completed_runs(run: Path, role: str) -> dict[str, list[dict[str, Any]]]:
    found = {}
    for folder in sorted((run / "eval" / role).glob("*")):
        done = folder / "completed.json"
        if not done.exists():
            continue
        saved = read_json(done)
        if (read_json(folder / "manifest.json")["signature"] != saved["signature"]
                or digest(folder / "episodes.jsonl") != saved["episodes_sha256"]):
            raise ValueError(f"Evaluation artifacts changed: {folder.name}")
        found[folder.name] = recover_records(folder / "episodes.jsonl")
    return found


def behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The tool-use signals behind the v0.3.1 diagnosis, per condition."""
    def action(step: dict[str, Any]) -> Any:
        return (step.get("action") or {}).get("action")

    def failed(step: dict[str, Any]) -> bool:
        observation = step.get("observation")
        return isinstance(observation, dict) and observation.get("status", "ok") != "ok"

    attempted = [r for r in rows if "steps" in r]
    errored = [r for r in attempted if any(failed(s) for s in r["steps"])]
    finishes = [[s["action"]["sql"] for s in r["steps"] if action(s) == "finish"] for r in attempted]
    return {"episodes": len(attempted),
            "tasks_using_query": sum(any(action(s) == "query" for s in r["steps"]) for r in attempted),
            "tasks_with_error_feedback": len(errored),
            "error_tasks_finally_correct": sum(bool(r["correct"]) for r in errored),
            "repeated_identical_finish": sum(len(f) >= 2 and len(set(f)) == 1 for f in finishes),
            "incorrect_wrong_columns": sum(r.get("mismatch") == "columns" for r in attempted),
            "incorrect_wrong_rows": sum(r.get("mismatch") == "rows" for r in attempted),
            "mean_assistant_steps": sum(len(r["steps"]) for r in attempted) / max(1, len(attempted))}


def section(tasks: list[Any], runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids = {t.id for t in tasks}
    runs = {name: [r for r in rows if r["task_id"] in ids] for name, rows in runs.items()}
    result: dict[str, Any] = {"tasks": len(tasks), "databases": len({t.db_id for t in tasks}), "conditions": {}}
    for name, rows in sorted(runs.items()):
        rows = aligned(tasks, rows)
        correct = sum(r["correct"] for r in rows)
        result["conditions"][name] = {"correct": correct, "success_rate": correct / len(tasks),
                                      "wilson_95": wilson(correct, len(tasks)),
                                      "not_attempted": sum(r["status"] == "not_attempted" for r in rows),
                                      "behavior": behavior(rows)}
    if "B" not in runs:
        return result
    for prefix in ("R_r1", "R_r2", "P_r1"):
        seeds = {name: rows for name, rows in runs.items() if name.startswith(prefix + "_seed")}
        if not seeds:
            continue
        pooled = paired_report(tasks, runs["B"], seeds)
        pooled.update(scope=SCOPE, warning="Exploratory unless all registered seeds are complete.")
        for name in pooled["per_seed"]:
            pooled["per_seed"][name]["database_cluster"] = paired_difference(
                aligned(tasks, runs["B"]), aligned(tasks, seeds[name]))
        result[f"{prefix}_minus_B"] = pooled
    for name in runs:
        match = CONDITION.fullmatch(name)
        if match and match.group(1) == "P":
            partner = f"R_r1_seed{match.group(3)}"
            if partner in runs:
                result[f"{name}_minus_{partner}"] = paired_difference(aligned(tasks, runs[partner]),
                                                                      aligned(tasks, runs[name]))
        if match and match.group(1) == "Rclean":
            partner = f"R_r{match.group(2)}_seed{match.group(3)}"
            if partner in runs:
                result[f"{partner}_minus_{name}"] = paired_difference(aligned(tasks, runs[name]),
                                                                      aligned(tasks, runs[partner]))
    return result


def markdown(result: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "—" if x is None else f"{100 * x:.1f}%"

    def pts(x: float | None) -> str:
        return "—" if x is None else f"{100 * x:+.1f}"

    def ci(pair: list[float] | None) -> str:
        return "—" if not pair else f"[{100 * pair[0]:+.1f}, {100 * pair[1]:+.1f}]"

    lines = ["# v0.5 报告", "", "执行匹配是本项目判分（比 BIRD 官方 EX 严格），不是官方榜单分数；差值单位为百分点。", ""]
    for role, part in result["roles"].items():
        lines += [f"## {role}（{part['tasks']} 题，{part['databases']} 库）", "",
                  "| 条件 | 正确 | 正确率 | 用 query | 报错题 | 报错后修好 | 重复提交 | 平均步数 | 错因：列数 | 错因：行 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, c in part["conditions"].items():
            b = c["behavior"]
            lines.append(f"| {name} | {c['correct']} | {pct(c['success_rate'])} | {b['tasks_using_query']} | "
                         f"{b['tasks_with_error_feedback']} | {b['error_tasks_finally_correct']} | "
                         f"{b['repeated_identical_finish']} | {b['mean_assistant_steps']:.2f} | "
                         f"{b['incorrect_wrong_columns']} | {b['incorrect_wrong_rows']} |")
        lines.append("")
        for key, value in part.items():
            if key.endswith("_minus_B"):
                lines.append(f"- {key}：seed 平均 {pts(value['mean_delta'])}，数据库×seed bootstrap 95% "
                             f"{ci(value['paired_database_seed_bootstrap_95'])}")
                for seed, s in value["per_seed"].items():
                    lines.append(f"  - {seed}：{pts(s['delta'])}，数据库聚类 95% "
                                 f"{ci(s['database_cluster']['database_cluster_bootstrap_95ci'])}")
            elif "_minus_" in key:
                lines.append(f"- {key}：{pts(value['delta_right_minus_left'])}，数据库聚类 95% "
                             f"{ci(value['database_cluster_bootstrap_95ci'])}")
        lines.append("")
    return "\n".join(lines)


def report(run: Path, out: Path) -> dict[str, Any]:
    frozen, _ = load_run(run)
    result: dict[str, Any] = {"protocol": frozen["protocol"], "run": frozen["identity"], "roles": {}}
    for role in ("dev", "validation", "test"):
        _, tasks = load_run(run, role)
        runs = completed_runs(run, role)
        if not tasks or not runs:
            continue
        result["roles"][role] = section(tasks, runs)
        if role == "validation":
            legacy = set(frozen["legacy_validation_ids"])
            subset = [t for t in tasks if t.id in legacy]
            result["roles"]["validation_v031_256"] = section(subset, runs)
    for round_ in (1, 2):
        path = run / f"round{round_}" / "selection.json"
        if path.exists():
            result[f"round{round_}_selection"] = read_json(path)
    result["identity"] = fingerprint(result)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_bytes(markdown(result).encode("utf-8"))
    return result
