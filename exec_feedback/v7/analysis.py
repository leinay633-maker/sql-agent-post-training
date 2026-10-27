"""v0.7 report: project execution matching (primary) and official BIRD EX side by side, preregistered comparisons
and dev-only selections."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from ..storage import atomic_json, read_json
from ..v6.analysis import _ex_task, official_ex, pooled
from .freeze import load
from .rollouts import merged


def official(tasks: list[Any], root: Path, runs: dict[str, list[dict[str, Any]]], procs: int) -> dict[str, list[dict[str, Any]]]:
    """v0.6 official-EX re-execution; sequential where fork is unavailable (CPU tests on Windows)."""
    import multiprocessing as mp

    if procs > 1 and "fork" in mp.get_all_start_methods():
        return official_ex(tasks, root, runs, procs)
    preds: dict[str, set[str]] = {}
    for rows in runs.values():
        for r in rows:
            if r.get("final_sql"):
                preds.setdefault(r["task_id"], set()).add(r["final_sql"])
    results = {t.id: _ex_task((str(t.database(root)), t.gold_sql, sorted(preds.get(t.id, ())))) for t in tasks}
    return {name: [{"task_id": r["task_id"], "db_id": r["db_id"],
                    "correct": bool(r.get("final_sql")) and results[r["task_id"]].get(r["final_sql"], 0) == 1,
                    "status": "correct" if r.get("final_sql") and results[r["task_id"]].get(r["final_sql"], 0) == 1
                    else "incorrect"} for r in rows] for name, rows in runs.items()}

FAMILIES = ("R", "P", "T", "TR", "Traw", "PT", "PTR", "R14")
FINAL_CANDIDATES = ("R", "P", "T", "TR", "PT", "PTR")
COMPARISONS = (  # (label, left, right): right - left, seed-matched when both are families
    ("T-R", "R", "T"), ("TR-R", "R", "TR"), ("TR-T", "T", "TR"), ("T-B", "B", "T"), ("R-B", "B", "R"),
    ("TR-B", "B", "TR"), ("P-B", "B", "P"), ("P-R", "R", "P"), ("PT-T", "T", "PT"), ("PTR-TR", "TR", "PTR"),
    ("PT-B", "B", "PT"), ("PTR-B", "B", "PTR"), ("T-Traw", "Traw", "T"), ("Traw-B", "B", "Traw"),
    ("R14-B14", "B14", "R14"), ("B32-B", "B", "B32"), ("B14-B", "B", "B14"),
)


def family(runs: dict[str, Any], name: str) -> dict[str, Any]:
    if name in ("B", "B14", "B32"):
        return {name: runs[name]} if name in runs else {}
    return {n: r for n, r in runs.items() if re.fullmatch(rf"{name}_seed\d+", n)}


def restrict(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Seed-matched pairs only when both sides are seeded families (T-Traw uses the shared seed 42)."""
    seeds = lambda d: {n.rsplit("_seed", 1)[1]: n for n in d if "_seed" in n}
    ls, rs = seeds(left), seeds(right)
    if ls and rs:
        shared = sorted(set(ls) & set(rs))
        return {ls[x]: left[ls[x]] for x in shared}, {rs[x]: right[rs[x]] for x in shared}
    return left, right


def final_flags(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Contamination check (v0.6 lesson): final-SQL timeouts and dialogues with a query timeout."""
    return {"final_sql_timeouts": sum((r.get("answer_preview") or {}).get("status") == "timeout" for r in rows),
            "query_timeout_dialogues": sum(any(isinstance(s.get("observation"), dict)
                                               and s["observation"].get("status") == "timeout"
                                               for s in r.get("steps") or []) for r in rows)}


def section(tasks: list[Any], runs: dict[str, list[dict[str, Any]]], flags: dict[str, Any] | None = None) -> dict[str, Any]:
    from ..rl.reporting import aligned

    part: dict[str, Any] = {"tasks": len(tasks), "databases": len({t.db_id for t in tasks}), "conditions": {}}
    for name, rows in sorted(runs.items()):
        al = aligned(tasks, rows)
        c = sum(r["correct"] for r in al)
        part["conditions"][name] = {"correct": c, "rate": c / len(tasks),
                                    "attempted": sum(r["status"] != "not_attempted" for r in al),
                                    **((flags or {}).get(name) or {})}
    deltas = {}
    for label, left, right in COMPARISONS:
        lf, rf = restrict(family(runs, left), family(runs, right))
        if lf and rf:
            deltas[label] = pooled(tasks, lf, rf)
    for name in runs:
        if name.endswith("+vote") and name[:-5] in runs:
            deltas[f"{name}-{name[:-5]}"] = pooled(tasks, {name[:-5]: runs[name[:-5]]}, {name: runs[name]})
    part["deltas"] = deltas
    return part


def family_means(conditions: dict[str, Any], tasks: int, seeds: list[int], families: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for f in families:
        names = [f"{f}_seed{s}" for s in seeds]
        if all(n in conditions and conditions[n]["attempted"] == tasks for n in names):
            out[f] = mean(conditions[n]["rate"] for n in names)
    return out


def dev_conditions(run: Path) -> tuple[dict[str, Any], int]:
    """Project-score rates of every complete dev evaluation (dev is the only selection set)."""
    from ..rl.reporting import aligned

    _, tasks = load(run, "dev")
    folder = run / "eval" / "dev"
    conditions = {}
    for cond in sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.exists() else []:
        rows, _ = merged(run, "dev", cond)
        if rows is not None:
            al = aligned(tasks, rows)
            conditions[cond] = {"rate": sum(r["correct"] for r in al) / len(tasks),
                                "attempted": sum(r["status"] != "not_attempted" for r in al)}
    return conditions, len(tasks)


def select_x(run: Path) -> dict[str, Any]:
    """Preregistered: the second stage goes on T or TR, whichever has the higher dev 3-seed mean."""
    frozen, _ = load(run)
    if (run / "select_x.json").exists():
        return read_json(run / "select_x.json")
    conditions, n = dev_conditions(run)
    means = family_means(conditions, n, frozen["settings"]["seeds"], ("T", "TR"))
    if not means:
        raise ValueError("Neither T nor TR has all seeds evaluated on dev")
    x = max(sorted(means), key=lambda f: means[f])
    result = {"run": frozen["identity"], "x": x, "dev_means": means, "rule": "argmax dev 3-seed mean of T, TR"}
    atomic_json(run / "select_x.json", result)
    return result


def select_final(run: Path) -> dict[str, Any]:
    """Preregistered: final family = argmax dev 3-seed mean over the complete candidate families."""
    frozen, _ = load(run)
    if (run / "select_final.json").exists():
        return read_json(run / "select_final.json")
    seeds = frozen["settings"]["seeds"]
    conditions, n = dev_conditions(run)
    means = family_means(conditions, n, seeds, FINAL_CANDIDATES)
    if not means:
        raise ValueError("No candidate family has all seeds evaluated on dev")
    best = max(sorted(means), key=lambda f: means[f])
    result = {"run": frozen["identity"], "family": best, "head": f"{best}_seed{seeds[0]}", "dev_means": means,
              "rule": "argmax dev 3-seed mean over R, P, T, TR, P_X (complete families only)"}
    atomic_json(run / "select_final.json", result)
    return result


def report(run: Path, *, procs: int = 64) -> dict[str, Any]:
    frozen, _ = load(run)
    root = Path(frozen["raw_root"])
    result: dict[str, Any] = {
        "protocol": frozen["protocol"], "run": frozen["identity"], "smoke_limit": frozen["smoke_limit"],
        "models": {k: f"{v['spec']['id']}@{v['spec']['revision'][:8]}" for k, v in frozen["models"].items()},
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "roles": {},
        "notes": "Project execution matching is primary (stricter than official EX); official EX re-executes the "
                 "recorded final SQL only. Dev is the only selection set; validation and test were used for no decision."}
    for role in ("test", "validation", "dev"):
        folder = run / "eval" / role
        if not folder.exists():
            continue
        _, tasks = load(run, role)
        runs, info, flags = {}, {}, {}
        for cond in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            rows, meta = merged(run, role, cond)
            if rows is not None:
                runs[cond], info[cond], flags[cond] = rows, meta, final_flags(rows)
        if not runs:
            continue
        ex = official(tasks, root, runs, procs)
        result["roles"][role] = {"project": section(tasks, runs, flags), "official_ex": section(tasks, ex),
                                 "timing": info}
        if role == "test":
            full = (frozen["registration"].get("test_screen") or {}).get("candidates")
            if full:
                result["roles"][role]["official_ex_lower_bound_all_dev"] = {
                    n: c["correct"] / full for n, c in result["roles"][role]["official_ex"]["conditions"].items()}
    for name in ("select_x.json", "select_final.json"):
        if (run / name).exists():
            result[name[:-5]] = read_json(run / name)
    for kind in ("teacher", "scale"):
        p = run / "data" / f"{kind}.json"
        if p.exists():
            d = read_json(p)
            result[f"data_{kind}"] = {"sampling": d["sampling"], "datasets": {
                k: {x: v.get(x) for x in ("examples", "origins", "frontier_tasks", "repair_trajectories",
                                          "supervised_tokens", "skipped_overlong", "student_examples",
                                          "teacher_gap_examples", "wrong_final_answers") if x in v}
                for k, v in d["datasets"].items()}}
    result["training"] = {}
    for p in sorted((run / "train").glob("*/result.json")) if (run / "train").exists() else []:
        r = read_json(p)
        result["training"][p.parent.name] = {k: r.get(k) for k in ("training_completed", "adapter_reload_verified",
                                                                     "reload_max_abs_delta")}
        result["training"][p.parent.name]["active_seconds"] = (r.get("metrics") or {}).get("active_seconds")
    out = run / "report"
    out.mkdir(exist_ok=True)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8")
    return result


def markdown(result: dict[str, Any]) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda p: "—" if not p else f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    lines = [f"# v0.7 报告（{result['models']}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += ["项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed bootstrap 95%。"
              "选择只看 dev。", ""]
    if "select_x" in result:
        lines.append(f"- 第二阶段家族（dev）：{result['select_x']['x']}，dev 均值 {result['select_x']['dev_means']}")
    if "select_final" in result:
        lines.append(f"- 最终家族（dev）：{result['select_final']['family']}，dev 均值 {result['select_final']['dev_means']}")
    lines.append("")
    for role, part in result["roles"].items():
        lines += [f"## {role}（{part['project']['tasks']} 题，{part['project']['databases']} 库）", "",
                  "| 条件 | 项目判分 | 官方 EX | 完成 | 最终 SQL 超时 | 对话内查询超时 |", "|---|---|---|---|---|---|"]
        for name, c in part["project"]["conditions"].items():
            e = part["official_ex"]["conditions"][name]
            lines.append(f"| {name} | {c['correct']} ({pct(c['rate'])}) | {e['correct']} ({pct(e['rate'])}) | "
                         f"{c['attempted']}/{part['project']['tasks']} | {c.get('final_sql_timeouts', '—')} | "
                         f"{c.get('query_timeout_dialogues', '—')} |")
        lines.append("")
        for key, d in part["project"]["deltas"].items():
            e = part["official_ex"]["deltas"].get(key, {})
            if d:
                lines.append(f"- {key}：项目 {pts(d['mean_delta'])} {ci(d['ci95'])}；官方 EX {pts(e.get('mean_delta'))} "
                             f"{ci(e.get('ci95'))}{'' if d['complete'] else '（不完整）'}（{len(d['pairs'])} 对）")
        lb = part.get("official_ex_lower_bound_all_dev")
        if lb:
            lines.append("- 官方 EX 下界（全部 BIRD dev 题，预检排除题记错）：" + "，".join(f"{n} {pct(v)}" for n, v in lb.items()))
        lines.append("")
    for kind in ("teacher", "scale"):
        if f"data_{kind}" in result:
            lines.append(f"- {kind} 采样与数据：{result[f'data_{kind}']}")
    if result.get("training"):
        lines.append(f"- 训练：{result['training']}")
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    shards = 64
    try:
        m, _ = load(run, check_code=False)
        shards = m["settings"]["sample_shards"]
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），题目 {m['counts']}，"
                     f"金标缓存 { {k: v['statuses'] for k, v in m['gold'].items()} }")
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
    from .rollouts import sampling_state
    for kind in ("scale", "teacher"):
        if (run / "samples" / kind).exists():
            st = sampling_state(run, kind, shards)
            facts.append(f"{kind} 采样 {st['trajectories']} 条、对 {st['correct']}，完成 {st['complete']}/{st['shards']} 片")
        p = run / "data" / f"{kind}.json"
        if p.exists():
            d = read_json(p)
            facts.append(f"{kind} 选样：{ {k: v['examples'] for k, v in d['datasets'].items()} } 例")
    for p in sorted((run / "train").glob("*/result.json")) if (run / "train").exists() else []:
        r = read_json(p)
        facts.append(f"{p.parent.name}：完成 {r.get('training_completed')}，重载 {r.get('adapter_reload_verified')}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, part in r["roles"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['project']['tasks']}" for n, c in part["project"]["conditions"].items())
            facts.append(f"{role}（项目判分）：{cells}")
    for name in ("select_x.json", "select_final.json"):
        if (run / name).exists():
            facts.append(f"{name[:-5]}：{read_json(run / name).get('x') or read_json(run / name).get('family')}")
    counts = Counter(p.parent.parent.name for p in (run / "eval").glob("*/*/shard*.summary.json")) if (run / "eval").exists() else {}
    if counts:
        facts.append(f"已完成评测分片：{dict(counts)}")
    return (f"\n### {now} · GPU 机（v0.7 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_V07_RUNBOOK.md。\n")
