"""v0.9 report: project execution matching (primary) and official BIRD EX side by side, the preregistered RL-PTR and
RL-RAFT decisions, the dev-only final family, training dynamics and dev learning curves of every online run, and the
error analysis."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from ..storage import atomic_json, read_json, recover_records
from ..v6.analysis import pooled
from ..v7.analysis import final_flags, official
from . import METHODS, START
from .evaluate import merged
from .freeze import load
from .online import registered, run_name

FINAL_CANDIDATES = (START, "RL", "RAFT")
COMPARISONS = (  # (label, left, right): right - left, seed-matched when both are families
    ("RL-PTR", START, "RL"), ("RL-RAFT", "RAFT", "RL"), ("RAFT-PTR", START, "RAFT"), ("RLhi-PTR", START, "RLhi"),
    ("RLhi-RL", "RL", "RLhi"), ("PTR-B", "B", START), ("RL-B", "B", "RL"), ("RAFT-B", "B", "RAFT"),
)
DECISIONS = (("RL-PTR", "主比较"), ("RL-RAFT", "次比较"))


def family(runs: dict[str, Any], name: str) -> dict[str, Any]:
    if name == "B":
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


def decide(roles: dict[str, Any], label: str) -> dict[str, Any]:
    """Preregistered: test mean > 0 with its 95% lower bound > 0, and validation mean > 0 (project scoring)."""
    test = ((roles.get("test") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    validation = ((roles.get("validation") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    if not test or not validation or not test.get("complete") or not validation.get("complete"):
        return {"decided": False, "holds": None, "reason": "missing or incomplete evaluation"}
    holds = test["mean_delta"] > 0 and test["ci95"][0] > 0 and validation["mean_delta"] > 0
    return {"decided": True, "holds": bool(holds), "test": [test["mean_delta"], test["ci95"]],
            "validation": [validation["mean_delta"], validation["ci95"]]}


def select_final(run: Path) -> dict[str, Any] | None:
    """Preregistered: argmax dev 3-seed mean among PTR, RL and RAFT with complete dev evaluations."""
    from ..rl.reporting import aligned

    frozen, tasks = load(run, "dev")
    means = {}
    for name in FINAL_CANDIDATES:
        rates = []
        for seed in frozen["settings"]["seeds"]:
            rows, _ = merged(run, "dev", f"{name}_seed{seed}")
            if rows is None:
                break
            al = aligned(tasks, rows)
            if any(r["status"] == "not_attempted" for r in al):
                break
            rates.append(sum(r["correct"] for r in al) / len(tasks))
        if len(rates) == len(frozen["settings"]["seeds"]):
            means[name] = mean(rates)
    if not means:
        return None
    best = max(sorted(means), key=lambda f: means[f])
    return {"family": best, "dev_means": means, "rule": "argmax dev 3-seed mean over PTR, RL, RAFT"}


def dynamics(run: Path) -> dict[str, Any]:
    """Per online run: what the reward, the informative groups, entropy, lengths and the train/inference gap did."""
    frozen, _ = load(run)
    out = {}
    for fam in METHODS:
        for seed in registered(frozen["settings"], fam):
            folder = run / "train" / run_name(fam, seed)
            rows = recover_records(folder / "metrics.jsonl") if (folder / "metrics.jsonl").exists() else []
            if not rows:
                continue
            head, tail = rows[:10], rows[-10:]
            avg = lambda part, key: mean(r[key] for r in part if r.get(key) is not None)
            groups = lambda r: sum(r.get(k, 0) for k in ("informative_groups", "all_correct_groups", "all_failed_groups"))
            result = read_json(folder / "result.json") if (folder / "result.json").exists() else {}
            curve = [r for r in recover_records(folder / "curve.jsonl") if "correct" in r] \
                if (folder / "curve.jsonl").exists() else []
            out[run_name(fam, seed)] = {
                "steps": rows[-1]["step"], "optimizer_updates": rows[-1]["optimizer_updates"],
                "stopped_by": result.get("stopped_by"), "last_milestone": result.get("last_milestone"),
                "reward_first10": avg(head, "reward_mean"), "reward_last10": avg(tail, "reward_mean"),
                "informative_share": sum(r.get("informative_groups", 0) for r in rows) / max(1, sum(groups(r) for r in rows)),
                "entropy_first10": avg(head, "entropy"), "entropy_last10": avg(tail, "entropy"),
                "turns_first10": avg(head, "mean_turns"), "turns_last10": avg(tail, "mean_turns"),
                "completion_tokens_first10": avg(head, "mean_completion_tokens"),
                "completion_tokens_last10": avg(tail, "mean_completion_tokens"),
                "mean_abs_logp_gap": avg(rows, "mean_abs_logp_gap"),
                "max_abs_logp_gap": max(r.get("max_abs_logp_gap", 0.0) for r in rows),
                "tis_clipped_fraction": avg(rows, "tis_clipped_fraction"),
                "unusable": dict(sum((Counter(r.get("unusable") or {}) for r in rows), Counter())),
                "context_rejections": sum(r.get("context_rejections", 0) for r in rows),
                "step_seconds": avg(rows, "step_seconds"), "generate_seconds": avg(rows, "generate_seconds"),
                "train_seconds": avg(rows, "train_seconds"),
                "peak_gpu_gb": max((r.get("peak_gpu_bytes") or 0) for r in rows) / 1e9,
                "lora_check": result.get("lora_check"),
                "dev_curve": {r["step"]: r["correct"] / r["tasks"] for r in curve}}
    return out


def report(run: Path, *, procs: int = 64) -> dict[str, Any]:
    from .errors import analyse

    frozen, _ = load(run)
    root = Path(frozen["raw_root"])
    result: dict[str, Any] = {
        "protocol": frozen["protocol"], "run": frozen["identity"], "smoke_limit": frozen["smoke_limit"],
        "v08_run": frozen["imports"]["v08_run"], "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "roles": {}, "notes": "Project execution matching is primary (stricter than official EX); official EX "
                              "re-executes the recorded final SQL only. Dev is the only selection set."}
    if (run / "steps.json").exists():
        result["steps"] = read_json(run / "steps.json")
    for role in ("test", "validation", "dev"):
        folder = run / "eval" / role
        if not folder.exists():
            continue
        _, tasks = load(run, role)
        runs, info, flags = {}, {}, {}
        for cond in sorted(p.name for p in folder.iterdir() if p.is_dir() and "@" not in p.name):
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
    result["decisions"] = {label: decide(result["roles"], label) for label, _ in DECISIONS}
    final = select_final(run)
    if final:
        result["select_final"] = final
    result["dynamics"] = dynamics(run)
    result["errors"] = {}
    for role in ("test", "validation"):
        if (run / "eval" / role).exists():
            result["errors"][role] = analyse(run, role, reference="B")
    out = run / "report"
    out.mkdir(exist_ok=True)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8")
    return result


def markdown(result: dict[str, Any]) -> str:
    from .errors import markdown as errors_markdown

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda p: "—" if not p else f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    lines = [f"# v0.9 报告（在线 RL，起点 v0.8 PTR，run {result['run'][:12]}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += ["项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed "
              "bootstrap 95%。选择只看 dev。", ""]
    steps = result.get("steps")
    if steps:
        lines.append(f"- 共同里程碑 K = {steps['k']} 步；各 run 最后里程碑 {steps['last_milestones']}；"
                     f"过短的主 run：{steps['short'] or '无'}")
    for label, name in DECISIONS:
        d = result["decisions"][label]
        if d["decided"]:
            lines.append(f"- {name} {label}：{'成立' if d['holds'] else '不成立'}；test {pts(d['test'][0])} {ci(d['test'][1])}，"
                         f"validation {pts(d['validation'][0])} {ci(d['validation'][1])}")
        else:
            lines.append(f"- {name} {label}：未判定（{d['reason']}）")
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
    if result.get("dynamics"):
        lines += ["## 训练过程（每个在线 run）", "",
                  "| run | 步数 | 奖励 前10→后10 | 有信号组占比 | 熵 前10→后10 | 轮数 前10→后10 | 生成 token 前10→后10 | "
                  "训推 logp 差 均值/最大 | TIS 截断 | 每步秒（生成/训练） | 显存峰值 | dev 曲线 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in result["dynamics"].items():
            curve = "，".join(f"{k}:{pct(v)}" for k, v in sorted(d["dev_curve"].items(), key=lambda x: int(x[0])))
            lines.append(f"| {name} | {d['steps']}（{d['stopped_by']}） | {pct(d['reward_first10'])}→{pct(d['reward_last10'])} | "
                         f"{pct(d['informative_share'])} | {d['entropy_first10']:.3f}→{d['entropy_last10']:.3f} | "
                         f"{d['turns_first10']:.2f}→{d['turns_last10']:.2f} | "
                         f"{d['completion_tokens_first10']:.0f}→{d['completion_tokens_last10']:.0f} | "
                         f"{d['mean_abs_logp_gap']:.4f}/{d['max_abs_logp_gap']:.2f} | {pct(d['tis_clipped_fraction'])} | "
                         f"{d['step_seconds']:.0f}（{d['generate_seconds']:.0f}/{d['train_seconds']:.0f}） | "
                         f"{d['peak_gpu_gb']:.1f} GB | {curve or '—'} |")
        lines.append("")
    for role, analysis in (result.get("errors") or {}).items():
        lines.append(errors_markdown(analysis, ["B", START, "RAFT", "RL", "RLhi"]))
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    try:
        m, _ = load(run, check_code=False)
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），题目 {m['counts']}，"
                     f"起点 v0.8 run {m['imports']['v08_run'][:12]} 的 PTR，金标缓存 "
                     f"{ {k: v['statuses'] for k, v in m['gold'].items()} }")
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
    for path in sorted((run / "train").glob("*/result.json")) if (run / "train").exists() else []:
        r = read_json(path)
        facts.append(f"{path.parent.name}：{r.get('steps_completed')} 步（{r.get('stopped_by')}），更新 "
                     f"{r.get('optimizer_updates')} 次，最后里程碑 {r.get('last_milestone')}，"
                     f"LoRA 检查 {'通过' if (r.get('lora_check') or {}).get('passed') else r.get('lora_check')}")
    if (run / "steps.json").exists():
        facts.append(f"共同里程碑 K = {read_json(run / 'steps.json')['k']}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, part in r["roles"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['project']['tasks']}" for n, c in part["project"]["conditions"].items())
            facts.append(f"{role}（项目判分）：{cells}")
        for label, d in r.get("decisions", {}).items():
            if d.get("decided"):
                facts.append(f"{label}：{'成立' if d['holds'] else '不成立'}，test {100 * d['test'][0]:+.1f} "
                             f"[{100 * d['test'][1][0]:+.1f}, {100 * d['test'][1][1]:+.1f}]")
        if r.get("select_final"):
            facts.append(f"最终家族（dev）：{r['select_final']['family']}")
    counts = Counter(p.parent.parent.name for p in (run / "eval").glob("*/*/summary.json")) if (run / "eval").exists() else {}
    if counts:
        facts.append(f"已完成评测：{dict(counts)}")
    return (f"\n### {now} · GPU 机（v0.9 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_V09_RUNBOOK.md。\n")
