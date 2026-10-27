"""v0.8 report: the teacher gate, project execution matching (primary) and official BIRD EX side by side,
preregistered comparisons and the dev-only final selection."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from ..storage import atomic_json, read_json
from ..v6.analysis import pooled
from ..v7.analysis import final_flags, official
from .freeze import load
from .rollouts import merged

FAMILIES = ("R", "P", "TR", "HR", "PTR", "PHR")
COMPARISONS = (  # (label, left, right): right - left, seed-matched when both are families
    ("TR-R", "R", "TR"), ("HR-R", "R", "HR"), ("TR-HR", "HR", "TR"), ("PTR-P", "P", "PTR"), ("PHR-P", "P", "PHR"),
    ("PTR-TR", "TR", "PTR"), ("PHR-HR", "HR", "PHR"), ("R-B", "B", "R"), ("P-B", "B", "P"), ("TR-B", "B", "TR"),
    ("HR-B", "B", "HR"), ("PTR-B", "B", "PTR"), ("PHR-B", "B", "PHR"), ("B235-B", "B", "B235"),
)


def family(runs: dict[str, Any], name: str) -> dict[str, Any]:
    if name in ("B", "B235"):
        return {name: runs[name]} if name in runs else {}
    return {n: r for n, r in runs.items() if re.fullmatch(rf"{name}_seed\d+", n)}


def restrict(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = lambda d: {n.rsplit("_seed", 1)[1]: n for n in d if "_seed" in n}
    ls, rs = seeds(left), seeds(right)
    if ls and rs:
        shared = sorted(set(ls) & set(rs))
        return {ls[x]: left[ls[x]] for x in shared}, {rs[x]: right[rs[x]] for x in shared}
    return left, right


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
    part["deltas"] = deltas
    return part


def gate(run: Path) -> dict[str, Any]:
    """Preregistered: the teacher is used only if its greedy dev score beats the 4B's by >= gate_delta (project
    scoring) and the database-bootstrap 95% lower bound is above zero. The smoke run never decides anything."""
    frozen, tasks = load(run, "dev")
    if (run / "gate.json").exists():
        return read_json(run / "gate.json")
    if frozen["smoke_limit"]:
        result = {"run": frozen["identity"], "passed": True, "smoke": True,
                  "rule": "smoke run: engineering path only, the gate is not evaluated"}
    else:
        rows = {name: merged(run, "dev", name)[0] for name in ("B", "B235")}
        if any(r is None for r in rows.values()):
            raise ValueError("The gate needs complete dev evaluations of B and B235")
        delta = pooled(tasks, {"B": rows["B"]}, {"B235": rows["B235"]})
        threshold = frozen["settings"]["gate_delta"]
        from ..rl.reporting import aligned
        rates = {n: sum(r["correct"] for r in aligned(tasks, v)) / len(tasks) for n, v in rows.items()}
        result = {"run": frozen["identity"], "passed": bool(delta["mean_delta"] >= threshold and delta["ci95"][0] > 0),
                  "smoke": False, "dev_rates": rates, "delta": delta["mean_delta"], "ci95": delta["ci95"],
                  "threshold": threshold,
                  "rule": f"B235 - B on dev (project scoring) >= {threshold:.2f} and bootstrap 95% lower bound > 0"}
    atomic_json(run / "gate.json", result)
    return result


def dev_conditions(run: Path) -> tuple[dict[str, Any], int]:
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


def select_final(run: Path) -> dict[str, Any] | None:
    """Preregistered: final family = argmax dev 3-seed mean over the complete families among R, P, TR, HR, PTR, PHR.
    Computed in the report; validation and test decide nothing."""
    frozen, _ = load(run)
    seeds = frozen["settings"]["seeds"]
    conditions, n = dev_conditions(run)
    means = {}
    for f in FAMILIES:
        names = [f"{f}_seed{s}" for s in seeds]
        if all(x in conditions and conditions[x]["attempted"] == n for x in names):
            means[f] = mean(conditions[x]["rate"] for x in names)
    if not means:
        return None
    best = max(sorted(means), key=lambda f: means[f])
    return {"family": best, "dev_means": means, "rule": "argmax dev 3-seed mean over complete families"}


def report(run: Path, *, procs: int = 64) -> dict[str, Any]:
    frozen, _ = load(run)
    root = Path(frozen["raw_root"])
    result: dict[str, Any] = {
        "protocol": frozen["protocol"], "run": frozen["identity"], "smoke_limit": frozen["smoke_limit"],
        "models": {k: f"{v['spec']['id']}@{v['spec']['revision'][:8]}" for k, v in frozen["models"].items()},
        "unsolved_tasks": len(frozen["unsolved_tasks"]),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "roles": {},
        "notes": "Project execution matching is primary (stricter than official EX); official EX re-executes the "
                 "recorded final SQL only. Dev is the only selection set; validation and test were used for no decision."}
    if (run / "gate.json").exists():
        result["gate"] = read_json(run / "gate.json")
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
    final = select_final(run)
    if final:
        result["select_final"] = final
    for sampler in ("teacher", "hint"):
        p = run / "data" / f"{sampler}.json"
        if p.exists():
            d = read_json(p)
            result[f"data_{sampler}"] = {k: d[k] for k in ("unsolved_tasks", "sampled_tasks", "sampling")}
            result[f"data_{sampler}"]["datasets"] = {k: {x: v.get(x) for x in (
                "examples", "origins", "student_examples", "gap_examples", "frontier_tasks", "repair_trajectories",
                "supervised_tokens", "skipped_overlong", "turns")} for k, v in d["datasets"].items()}
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
    lines = [f"# v0.8 报告（{result['models']}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += ["项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed bootstrap 95%。"
              f"选择只看 dev。4B 第一轮全错的 pool 题：{result['unsolved_tasks']}。", ""]
    g = result.get("gate")
    if g and not g.get("smoke"):
        lines.append(f"- 老师门槛：{'通过' if g['passed'] else '未通过'}；dev B235−B {pts(g['delta'])} {ci(g['ci95'])}"
                     f"（门槛 {pts(g['threshold'])}，dev 判分 {g['dev_rates']}）")
    elif not g:
        lines.append("- 老师门槛：未判定（老师评测没有完成）")
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
    for sampler in ("teacher", "hint"):
        if f"data_{sampler}" in result:
            lines.append(f"- {sampler} 采样与数据：{result[f'data_{sampler}']}")
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
                     f"4B 全错题 {len(m['unsolved_tasks'])}，金标缓存 { {k: v['statuses'] for k, v in m['gold'].items()} }")
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
    from .rollouts import sampling_state
    for sampler in ("hint", "teacher"):
        if (run / "samples" / sampler).exists():
            st = sampling_state(run, sampler, shards)
            facts.append(f"{sampler} 采样 {st['trajectories']} 条、对 {st['correct']}，完成 {st['complete']}/{st['shards']} 片")
        p = run / "data" / f"{sampler}.json"
        if p.exists():
            d = read_json(p)
            facts.append(f"{sampler} 选样：{ {k: (v['examples'], v['gap_examples']) for k, v in d['datasets'].items()} }（总例数, 新增）")
    if (run / "gate.json").exists():
        g = read_json(run / "gate.json")
        facts.append("老师门槛：" + ("冒烟不判定" if g.get("smoke") else
                                 f"{'通过' if g['passed'] else '未通过'}，dev 差 {100 * g['delta']:+.1f} 点，"
                                 f"区间 [{100 * g['ci95'][0]:+.1f}, {100 * g['ci95'][1]:+.1f}]"))
    for p in sorted((run / "train").glob("*/result.json")) if (run / "train").exists() else []:
        r = read_json(p)
        facts.append(f"{p.parent.name}：完成 {r.get('training_completed')}，重载 {r.get('adapter_reload_verified')}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, part in r["roles"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['project']['tasks']}" for n, c in part["project"]["conditions"].items())
            facts.append(f"{role}（项目判分）：{cells}")
        if r.get("select_final"):
            facts.append(f"最终家族（dev）：{r['select_final']['family']}")
    counts = Counter(p.parent.parent.name for p in (run / "eval").glob("*/*/summary.json")) if (run / "eval").exists() else {}
    if counts:
        facts.append(f"已完成评测：{dict(counts)}")
    return (f"\n### {now} · GPU 机（v0.8 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_V08_RUNBOOK.md。\n")
