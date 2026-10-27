"""Error analysis of any v0.8/v0.9 run directory: what kind of wrong answer each condition gives, accuracy by BIRD
difficulty and by gold-SQL feature, agent behaviour (queries, reviews, revisions), and the tasks a condition fixed or
broke against a reference condition.

Reads only the run's own frozen files (task lists, gold caches, evaluation journals) and checks their recorded
hashes; there is no code-fingerprint check, so it also runs on an older version's pulled results.

python -m exec_feedback.v9.errors <run dir> --role test --reference B --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ..core import digest, read_jsonl
from ..storage import read_json, recover_records

NO_ANSWER = {"budget_exhausted", "protocol_error", "provider_error", "schema_error"}
OUTCOMES = ("correct", "wrong_values", "wrong_row_count", "empty_result", "wrong_columns", "execution_failed",
            "no_answer", "grader_limit", "not_attempted")
# Patterns on the gold SQL with string literals removed; a task can carry several.
FEATURES = {
    "join": r"\bJOIN\b",
    "aggregate": r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(",
    "group_by": r"\bGROUP\s+BY\b",
    "order_limit": r"\bORDER\s+BY\b[\s\S]*\bLIMIT\b",
    "subquery": r"\(\s*SELECT\b",
    "case_iif": r"\b(CASE|IIF)\b",
    "division": r"/",
    "date_func": r"\b(STRFTIME|DATE|DATETIME|JULIANDAY)\s*\(",
    "like": r"\bLIKE\b",
    "distinct": r"\bDISTINCT\b",
    "set_op": r"\b(UNION|INTERSECT|EXCEPT)\b",
}


def strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def features(sql: str) -> list[str]:
    bare = strip_literals(sql)
    return [name for name, pattern in FEATURES.items() if re.search(pattern, bare, re.I)]


def outcome(row: dict[str, Any] | None, gold: dict[str, Any] | None) -> str:
    """Why a final answer is wrong, from the recorded result preview (columns, full row count) and the cached gold."""
    if row is None:
        return "not_attempted"
    status = row["status"]
    if status == "correct":
        return "correct"
    if status in NO_ANSWER:
        return "no_answer"
    if status in ("execution_failed", "grader_limit"):
        return status
    preview = row.get("answer_preview") or {}
    if preview.get("status") not in (None, "ok"):
        return "execution_failed"
    if status != "incorrect" or not preview or not gold or gold.get("status") != "ok":
        return "wrong_values"
    if len(preview.get("columns") or []) != len(gold["columns"]):
        return "wrong_columns"
    rows = preview.get("row_count", len(preview.get("rows") or []))
    if rows != len(gold["rows"]):
        return "empty_result" if rows == 0 else "wrong_row_count"
    return "wrong_values"


def behaviour(row: dict[str, Any]) -> dict[str, Any]:
    steps = row.get("steps") or []
    actions = [(s.get("action") or {}).get("action") for s in steps]
    reviewed = revised = False
    for i, step in enumerate(steps):
        observation = step.get("observation")
        if isinstance(observation, dict) and "review" in observation:
            reviewed = True
            before = (step.get("action") or {}).get("sql")
            after = next(((s.get("action") or {}).get("sql") for s in steps[i + 1:]
                          if (s.get("action") or {}).get("action") == "finish"), None)
            revised = after is not None and after.strip() != (before or "").strip()
    return {"turns": len(steps), "queries": actions.count("query"), "schema_calls": actions.count("schema"),
            "protocol_errors": actions.count(None), "reviewed": reviewed, "revised": revised}


def run_files(run: Path, role: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(run / "manifest.json")
    tasks_path, gold_path = run / f"{role}.jsonl", run / "gold" / f"{role}.json"
    if digest(tasks_path) != manifest["sha256"][role] or digest(gold_path) != manifest["gold"][role]["sha256"]:
        raise ValueError(f"{role} task list or gold cache differs from the run manifest")
    return read_jsonl(tasks_path), read_json(gold_path)["results"]


def episodes(run: Path, role: str, condition: str) -> list[dict[str, Any]] | None:
    folder = run / "eval" / role / condition
    if not (folder / "summary.json").exists():
        return None
    summary = read_json(folder / "summary.json")
    if digest(folder / "episodes.jsonl") != summary["episodes_sha256"]:
        raise ValueError(f"Evaluation journal changed: {role}/{condition}")
    return recover_records(folder / "episodes.jsonl")


def condition_stats(tasks: list[dict[str, Any]], rows: list[dict[str, Any]], gold: dict[str, Any]) -> dict[str, Any]:
    by_id = {r["task_id"]: r for r in rows}
    kinds = Counter(outcome(by_id.get(t["id"]), gold.get(t["id"])) for t in tasks)
    acts = [behaviour(r) for r in rows]
    reviewed = [(a, r) for a, r in zip(acts, rows) if a["reviewed"]]
    revised = [r for a, r in reviewed if a["revised"]]
    by_difficulty, by_feature = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for t in tasks:
        ok = bool(by_id.get(t["id"], {}).get("correct"))
        level = t.get("difficulty") or "unknown"
        by_difficulty[level][0] += ok
        by_difficulty[level][1] += 1
        for f in features(t["gold_sql"]):
            by_feature[f][0] += ok
            by_feature[f][1] += 1
    return {"tasks": len(tasks), "correct": kinds["correct"],
            "outcomes": {k: kinds[k] for k in OUTCOMES if kinds[k]},
            "mean_turns": mean(a["turns"] for a in acts) if acts else None,
            "with_query": sum(a["queries"] > 0 for a in acts) / max(1, len(acts)),
            "mean_queries": mean(a["queries"] for a in acts) if acts else None,
            "protocol_error_turns": sum(a["protocol_errors"] for a in acts),
            "reviewed": len(reviewed), "revised_after_review": len(revised),
            "revised_correct": sum(bool(r["correct"]) for r in revised),
            "difficulty": {k: {"correct": v[0], "tasks": v[1]} for k, v in sorted(by_difficulty.items())},
            "features": {k: {"correct": v[0], "tasks": v[1]} for k, v in sorted(by_feature.items())}}


def transitions(tasks: list[dict[str, Any]], reference: list[dict[str, Any]], rows: list[dict[str, Any]],
                gold: dict[str, Any]) -> dict[str, Any]:
    """Tasks the condition fixed (reference wrong, condition right) or broke, by feature and by the error involved."""
    ref, cur = {r["task_id"]: r for r in reference}, {r["task_id"]: r for r in rows}
    fixed, broken, fixed_from, broken_to = Counter(), Counter(), Counter(), Counter()
    for t in tasks:
        a, b = bool(ref.get(t["id"], {}).get("correct")), bool(cur.get(t["id"], {}).get("correct"))
        if a == b:
            continue
        target = fixed if b else broken
        target["all"] += 1
        for f in features(t["gold_sql"]) or ["plain"]:
            target[f] += 1
        if b:
            fixed_from[outcome(ref.get(t["id"]), gold.get(t["id"]))] += 1
        else:
            broken_to[outcome(cur.get(t["id"]), gold.get(t["id"]))] += 1
    return {"fixed": dict(fixed), "broken": dict(broken), "fixed_from": dict(fixed_from),
            "broken_to": dict(broken_to)}


def analyse(run: Path, role: str, *, reference: str = "B", conditions: list[str] | None = None) -> dict[str, Any]:
    tasks, gold = run_files(run, role)
    folder = run / "eval" / role
    names = conditions or (sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else [])
    rows = {name: r for name in names if (r := episodes(run, role, name)) is not None}
    result: dict[str, Any] = {"role": role, "tasks": len(tasks), "reference": reference,
                              "feature_counts": dict(Counter(f for t in tasks for f in features(t["gold_sql"]))),
                              "difficulty_counts": dict(Counter(t.get("difficulty") or "unknown" for t in tasks)),
                              "conditions": {n: condition_stats(tasks, r, gold) for n, r in rows.items()}}
    if reference in rows:
        result["transitions"] = {n: transitions(tasks, rows[reference], r, gold)
                                 for n, r in rows.items() if n != reference}
    return result


def view(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    """One condition, or the seed average of a family (e.g. PTR over PTR_seed42/43/44)."""
    members = [c for n, c in result["conditions"].items() if n == name or re.fullmatch(rf"{name}_seed\d+", n)]
    if not members:
        return None
    avg = lambda f: mean(f(c) for c in members)
    return {"seeds": len(members), "rate": avg(lambda c: c["correct"] / c["tasks"]),
            "outcomes": {k: avg(lambda c: c["outcomes"].get(k, 0)) for k in OUTCOMES},
            "difficulty": {d: avg(lambda c: c["difficulty"][d]["correct"] / c["difficulty"][d]["tasks"])
                           for d in members[0]["difficulty"]},
            "features": {f: avg(lambda c: c["features"][f]["correct"] / c["features"][f]["tasks"])
                         for f in members[0]["features"]},
            **{k: avg(lambda c, k=k: c[k]) for k in ("mean_turns", "with_query", "reviewed", "revised_after_review",
                                                     "revised_correct")}}


LABELS = {"wrong_values": "值不对（行数列数都对）", "wrong_row_count": "行数不对", "empty_result": "返回空结果",
          "wrong_columns": "列数不对", "execution_failed": "最终 SQL 执行失败", "no_answer": "没有提交（步数用完等）",
          "grader_limit": "超出判分上限"}


def markdown(result: dict[str, Any], families: list[str]) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    views = {name: v for name in families if (v := view(result, name)) is not None}
    if not views:
        return ""
    lines = [f"### 错误分析：{result['role']}（{result['tasks']} 题；多 seed 的家族取 seed 均值）", "",
             "| 项目 | " + " | ".join(views) + " |", "|---|" + "---|" * len(views),
             "| 项目判分 | " + " | ".join(pct(v["rate"]) for v in views.values()) + " |"]
    for key, label in LABELS.items():
        if any(v["outcomes"].get(key) for v in views.values()):
            lines.append(f"| 错因：{label} | " + " | ".join(f"{v['outcomes'].get(key, 0):.0f}"
                                                         for v in views.values()) + " |")
    for d in sorted({d for v in views.values() for d in v["difficulty"]}):
        lines.append(f"| 难度 {d}（{result['difficulty_counts'].get(d, 0)} 题） | "
                     + " | ".join(pct(v["difficulty"].get(d)) for v in views.values()) + " |")
    for f in sorted(result["feature_counts"], key=lambda f: -result["feature_counts"][f]):
        lines.append(f"| 金标含 {f}（{result['feature_counts'][f]} 题） | "
                     + " | ".join(pct(v["features"].get(f)) for v in views.values()) + " |")
    lines.append("| 平均轮数 | " + " | ".join(f"{v['mean_turns']:.2f}" for v in views.values()) + " |")
    lines.append("| 至少查一次数据 | " + " | ".join(pct(v["with_query"]) for v in views.values()) + " |")
    lines.append("| 复核后改了 SQL / 改后对 | " + " | ".join(
        f"{v['revised_after_review']:.0f} / {v['revised_correct']:.0f}" for v in views.values()) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--role", default="test", choices=["dev", "validation", "test"])
    ap.add_argument("--reference", default="B")
    ap.add_argument("--families", default="B,PTR,RAFT,RL,RLhi")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = analyse(args.run, args.role, reference=args.reference)
    text = markdown(result, [f for f in args.families.split(",") if f])
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
