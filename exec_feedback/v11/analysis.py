"""v0.11 report: project execution matching (primary) and official BIRD EX side by side, the preregistered RLb-PTR and
RLb-RL9 decisions, the exploratory continuation (RLb at explore_steps, named RLb400 for 400 steps), the pre-pass (how
often each start solved each pool task), training dynamics with v0.10's three degenerate behaviours per 20 steps, the
same behaviours at evaluation, and the error analysis.

Two reports: `main` once the evaluations at K are done (the preregistered result, kept as written), `final` after the
exploratory continuation; report.json/report.md are the final one once it exists, else the main one."""
from __future__ import annotations

import datetime as dt
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from ..storage import atomic_json, read_json, recover_records
from ..v6.analysis import pooled
from ..v7.analysis import final_flags, official
from ..v9.analysis import family, restrict
from ..v10.analysis import compact
from ..v10.analysis import summarise as run_summary
from . import ARM, FAMILY, START
from .evaluate import merged
from .freeze import load
from .online import behaviour, ended, paused, run_name
from .prepass import prepass_dir

DECISIONS = ((f"{FAMILY}-{START}", "主比较"), (f"{FAMILY}-{ARM}", "次比较"))
BEHAVIOURS = {"unparsed": "首个输出不可解析", "no_query": "不查数据就提交", "repeat": "连续两次查同一 SQL"}
BLOCK = 20  # training steps per row of the behaviour table
REPORTS = ("main", "final")


def explore_name(settings: dict[str, Any]) -> str:
    return f"{FAMILY}{settings['explore_steps']}"


def comparisons(settings: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """(label, left, right): right - left, seed-matched when both are families."""
    x = explore_name(settings)
    return ((f"{FAMILY}-{START}", START, FAMILY), (f"{FAMILY}-{ARM}", ARM, FAMILY), (f"{ARM}-{START}", START, ARM),
            (f"{x}-{START}", START, x), (f"{x}-{FAMILY}", FAMILY, x), (f"{START}-B", "B", START),
            (f"{FAMILY}-B", "B", FAMILY), (f"{ARM}-B", "B", ARM))


def conditions(folder: Path, settings: dict[str, Any]) -> dict[str, str]:
    """Evaluation folders -> report names: every condition at its registered point, and RLb at explore_steps as
    RLb400 (other '@' evaluations are ad hoc and stay out of the report)."""
    suffix = f"@{settings['explore_steps']}"
    out = {}
    for p in sorted(folder.iterdir()) if folder.exists() else []:
        if not p.is_dir():
            continue
        if "@" not in p.name:
            out[p.name] = p.name
        elif re.fullmatch(rf"{FAMILY}_seed\d+{re.escape(suffix)}", p.name):
            out[p.name] = explore_name(settings) + p.name[len(FAMILY):-len(suffix)]
    return out


def section(tasks: list[Any], runs: dict[str, list[dict[str, Any]]], comps: tuple[tuple[str, str, str], ...],
            flags: dict[str, Any] | None = None) -> dict[str, Any]:
    from ..rl.reporting import aligned

    part: dict[str, Any] = {"tasks": len(tasks), "databases": len({t.db_id for t in tasks}), "conditions": {}}
    for name, rows in sorted(runs.items()):
        al = aligned(tasks, rows)
        c = sum(r["correct"] for r in al)
        part["conditions"][name] = {"correct": c, "rate": c / len(tasks),
                                    "attempted": sum(r["status"] != "not_attempted" for r in al),
                                    **((flags or {}).get(name) or {})}
    part["deltas"] = {}
    for label, left, right in comps:
        lf, rf = restrict(family(runs, left), family(runs, right))
        if lf and rf:
            part["deltas"][label] = pooled(tasks, lf, rf)
    return part


def decide(roles: dict[str, Any], label: str) -> dict[str, Any]:
    """Preregistered (v0.9's standard): test mean > 0 with its 95% lower bound > 0, and validation mean > 0."""
    test = ((roles.get("test") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    validation = ((roles.get("validation") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    if not test or not validation or not test.get("complete") or not validation.get("complete"):
        return {"decided": False, "holds": None, "reason": "missing or incomplete evaluation"}
    holds = test["mean_delta"] > 0 and test["ci95"][0] > 0 and validation["mean_delta"] > 0
    return {"decided": True, "holds": bool(holds), "seeds": len(test["pairs"]),
            "test": [test["mean_delta"], test["ci95"]], "validation": [validation["mean_delta"], validation["ci95"]]}


def prepass_summary(run: Path, seeds: list[int]) -> dict[str, Any]:
    out = {}
    for seed in seeds:
        path = prepass_dir(run, seed) / "result.json"
        if path.exists():
            r = read_json(path)
            out[f"{START}_seed{seed}"] = {
                "tasks": r["tasks"], "scorable": r["scorable"], "invalid": len(r["invalid"]), "boundary": len(r["boundary"]),
                "boundary_share": r["boundary_share"], "histogram": r["histogram"], "gate": r["gate"],
                "statuses": r["statuses"], "mean_turns": r["mean_turns"],
                "wall_seconds": (r.get("last_attempt") or {}).get("wall_seconds")}
    return out


def blocks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per BLOCK training steps: the share of dialogues showing each behaviour, the reward and the informative groups."""
    out = []
    for start in range(0, len(rows), BLOCK):
        part = rows[start:start + BLOCK]
        total = sum((Counter(r.get("behaviour") or {}) for r in part), Counter())
        groups = sum(r.get(k, 0) for r in part for k in ("informative_groups", "all_correct_groups", "all_failed_groups"))
        n = total.get("dialogues", 0)
        out.append({"first": part[0]["step"], "last": part[-1]["step"], "dialogues": n,
                    **{k: total.get(k, 0) / n if n else None for k in BEHAVIOURS},
                    "reward": mean(r["reward_mean"] for r in part),
                    "informative_share": sum(r.get("informative_groups", 0) for r in part) / max(1, groups)})
    return out


def dynamics(run: Path) -> dict[str, Any]:
    """v0.9's RL (imported records) and every boundary run: v0.10's per-run summary, plus where a boundary run stands,
    how often it went over its boundary tasks and the behaviours per BLOCK steps."""
    frozen, _ = load(run, check_code=False)
    s = frozen["settings"]
    out = {}
    for seed in s["seeds"]:
        if (d := run_summary(run / "imports" / "v09" / "train" / f"RL_seed{seed}")) is not None:
            out[f"{ARM}_seed{seed}"] = d
        folder = run / "train" / run_name(seed)
        d = run_summary(folder)
        if d is None:
            continue
        record, held = ended(folder), paused(folder)
        d["finished"] = bool(record)
        d["stopped_by"] = record["stopped_by"] if record else "paused" if held else None
        boundary = read_json(folder / "manifest.json")["plan"]["boundary_tasks"]
        d["boundary_tasks"], d["passes"] = boundary, d["steps"] * s["tasks_per_step"] / boundary
        rows = recover_records(folder / "metrics.jsonl")
        d["blocks"] = blocks(rows)
        curve = recover_records(folder / "curve.jsonl") if (folder / "curve.jsonl").exists() else []
        d["curve_behaviour"] = {r["step"]: r["behaviour"] for r in curve if "behaviour" in r}
        out[run_name(seed)] = d
    return out


def report(run: Path, *, name: str = "main", procs: int = 64) -> dict[str, Any]:
    from ..v9.errors import analyse

    if name not in REPORTS:
        raise ValueError(f"Reports are {REPORTS}")
    frozen, _ = load(run)
    s = frozen["settings"]
    root = Path(frozen["raw_root"])
    comps = comparisons(s)
    out = run / "report"
    out.mkdir(exist_ok=True)
    result: dict[str, Any] = {
        "protocol": frozen["protocol"], "run": frozen["identity"], "report": name, "smoke_limit": frozen["smoke_limit"],
        "v08_run": frozen["imports"]["v08_run"], "v09_run": frozen["imports"]["v09"]["v09_run"],
        "planned_steps": s["planned_steps"], "explore_steps": s["explore_steps"], "explore_name": explore_name(s),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "roles": {},
        "notes": "Project execution matching is primary (stricter than official EX); official EX re-executes the "
                 "recorded final SQL only. Only RLb-PTR and RLb-RL9 at K are decisions; everything at explore_steps "
                 "is exploratory."}
    if (run / "steps.json").exists():
        result["steps"] = read_json(run / "steps.json")
    for role in ("test", "validation", "dev"):
        folder = run / "eval" / role
        names = conditions(folder, s)
        if not names:
            continue
        _, tasks = load(run, role)
        runs, info, flags = {}, {}, {}
        for disk, label in names.items():
            rows, meta = merged(run, role, disk)
            if rows is not None:
                runs[label], info[label] = rows, meta
                flags[label] = {**final_flags(rows), **behaviour(rows)}
        if not runs:
            continue
        ex = official(tasks, root, runs, procs)
        result["roles"][role] = {"project": section(tasks, runs, comps, flags), "official_ex": section(tasks, ex, comps),
                                 "timing": info}
        if role == "test":
            full = (frozen["registration"].get("test_screen") or {}).get("candidates")
            if full:
                result["roles"][role]["official_ex_lower_bound_all_dev"] = {
                    n: c["correct"] / full for n, c in result["roles"][role]["official_ex"]["conditions"].items()}
        atomic_json(out / f"rows-{role}.json", {"project": compact(runs), "official_ex": compact(ex), "flags": flags})
    result["decisions"] = {label: decide(result["roles"], label) for label, _ in DECISIONS}
    result["prepass"] = prepass_summary(run, s["seeds"])
    result["dynamics"] = dynamics(run)
    result["errors"] = {}
    for role in ("test", "validation"):
        names = conditions(run / "eval" / role, s)
        if names:
            analysis = analyse(run, role, reference="B", conditions=sorted(names))
            analysis["conditions"] = {names[n]: c for n, c in analysis["conditions"].items()}
            analysis["transitions"] = {names[n]: t for n, t in (analysis.get("transitions") or {}).items()}
            result["errors"][role] = analysis
    atomic_json(out / f"{name}.json", result)
    (out / f"{name}.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    if name == "final" or not (out / "final.json").exists():  # report.* is the fuller of the two once there is one
        for suffix in ("json", "md"):
            shutil.copyfile(out / f"{name}.{suffix}", out / f"report.{suffix}")
    return result


def markdown(result: dict[str, Any]) -> str:
    from ..v9.errors import markdown as errors_markdown

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda p: "—" if not p else f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    num = lambda x, f="{:.1f}": "—" if x is None else f.format(x)
    x, k = result["explore_name"], result["planned_steps"]
    title = "预注册报告" if result["report"] == "main" else "完整报告（含探索续跑）"
    lines = [f"# v0.11 {title}（只在起点的边界题上做在线 RL，run {result['run'][:12]}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += [f"项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed "
              f"bootstrap 95%，按 seed 配对。{FAMILY} 在预注册比较点 K 评测；{ARM} 是 v0.9 的 RL 在它登记的第 {k} 步，"
              f"同晚重评；{x} 是 {FAMILY} 续训到第 {result['explore_steps']} 步，只作探索。", ""]
    steps = result.get("steps")
    if steps:
        lines.append(f"- 比较点 K = {steps['k']} 步；各 run 最后里程碑 {steps['last_milestones']}；"
                     f"过短的 run：{steps['short'] or '无'}")
    for label, role_name in DECISIONS:
        d = result["decisions"][label]
        if d["decided"]:
            lines.append(f"- {role_name} {label}（{d['seeds']} 个 seed）：{'成立' if d['holds'] else '不成立'}；"
                         f"test {pts(d['test'][0])} {ci(d['test'][1])}，validation {pts(d['validation'][0])} "
                         f"{ci(d['validation'][1])}")
        else:
            lines.append(f"- {role_name} {label}：未判定（{d['reason']}）")
    lines.append("")
    if result.get("prepass"):
        lines += ["## 预采样（起点在训练题池上每题采样，做对次数的分布）", "",
                  "| 起点 | 题数 | 可判分 | 金标失败 | 边界题 | 占可判分 | 门禁 | 做对次数分布 | 平均轮数 | 用时 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, p in result["prepass"].items():
            hist = " ".join(f"{c}:{n}" for c, n in sorted(p["histogram"].items(), key=lambda kv: int(kv[0])))
            lines.append(f"| {name} | {p['tasks']} | {p['scorable']} | {p['invalid']} | {p['boundary']} | "
                         f"{pct(p['boundary_share'])} | {'通过' if p['gate']['passed'] else '未通过'} | {hist} | "
                         f"{p['mean_turns']:.2f} | {num((p['wall_seconds'] or 0) / 60, '{:.0f}')} 分钟 |")
        lines.append("")
    for role, part in result["roles"].items():
        lines += [f"## {role}（{part['project']['tasks']} 题，{part['project']['databases']} 库）", "",
                  "| 条件 | 项目判分 | 官方 EX | 完成 | 最终 SQL 超时 | 对话内查询超时 | 解析失败 / 不查就交 / 重复查询 |",
                  "|---|---|---|---|---|---|---|"]
        for name, c in part["project"]["conditions"].items():
            e = part["official_ex"]["conditions"][name]
            shares = "/".join(pct(c.get(b, 0) / c["dialogues"]) if c.get("dialogues") else "—" for b in BEHAVIOURS)
            lines.append(f"| {name} | {c['correct']} ({pct(c['rate'])}) | {e['correct']} ({pct(e['rate'])}) | "
                         f"{c['attempted']}/{part['project']['tasks']} | {c.get('final_sql_timeouts', '—')} | "
                         f"{c.get('query_timeout_dialogues', '—')} | {shares} |")
        lines.append("")
        for key, d in part["project"]["deltas"].items():
            e = part["official_ex"]["deltas"].get(key, {})
            if d:
                tag = "（预注册）" if key in dict(DECISIONS) else "（探索）" if key.startswith(x) else ""
                lines.append(f"- {key}{tag}：项目 {pts(d['mean_delta'])} {ci(d['ci95'])}；官方 EX {pts(e.get('mean_delta'))} "
                             f"{ci(e.get('ci95'))}{'' if d['complete'] else '（不完整）'}（{len(d['pairs'])} 对）")
        lb = part.get("official_ex_lower_bound_all_dev")
        if lb:
            lines.append("- 官方 EX 下界（全部 BIRD dev 题，预检排除题记错）：" + "，".join(f"{n} {pct(v)}" for n, v in lb.items()))
        lines.append("")
    dyn = result.get("dynamics") or {}
    if dyn:
        lines += [f"## 训练过程（{ARM} 为导入的 v0.9 记录）", "",
                  "| run | 步数 | 边界题 / 遍数 | 奖励 前10→后10 | 有信号组占比 | 每步训练轨迹 | 熵 前10→后10 | "
                  "生成 token 前10→后10 | 训推 logp 差 均值/最大 | TIS 截断 | 每步秒（生成/训练） | dev 曲线 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in dyn.items():
            curve = "，".join(f"{step}:{pct(v)}" for step, v in d["dev_curve"].items())
            where = f"{d['boundary_tasks']} / {d['passes']:.1f}" if "passes" in d else "—"
            lines.append(f"| {name} | {d['steps']}（{d['stopped_by'] or '进行中'}） | {where} | "
                         f"{pct(d['reward_first10'])}→{pct(d['reward_last10'])} | {pct(d['informative_share'])} | "
                         f"{num(d['trained_per_step'])} | {num(d['entropy_first10'], '{:.3f}')}→"
                         f"{num(d['entropy_last10'], '{:.3f}')} | {num(d['completion_tokens_first10'], '{:.0f}')}→"
                         f"{num(d['completion_tokens_last10'], '{:.0f}')} | {num(d['mean_abs_logp_gap'], '{:.4f}')}/"
                         f"{num(d['max_abs_logp_gap'], '{:.2f}')} | {pct(d['tis_clipped_fraction'])} | "
                         f"{num(d['step_seconds'], '{:.0f}')}（{num(d['generate_seconds'], '{:.0f}')}/"
                         f"{num(d['train_seconds'], '{:.0f}')}） | {curve or '—'} |")
        lines.append("")
        runs = [n for n, d in dyn.items() if "blocks" in d]
        if runs:
            lines += [f"### 训练中三种行为的占比（每 {BLOCK} 步；{' / '.join(BEHAVIOURS.values())}）", "",
                      "| 步 | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
            depth = max(len(dyn[n]["blocks"]) for n in runs)
            for i in range(depth):
                cells = []
                for n in runs:
                    b = dyn[n]["blocks"][i] if i < len(dyn[n]["blocks"]) else None
                    cells.append("/".join(pct(b[key]) for key in BEHAVIOURS) + f"（信号 {pct(b['informative_share'])}）"
                                 if b else "")
                first = next(dyn[n]["blocks"][i] for n in runs if i < len(dyn[n]["blocks"]))
                lines.append(f"| {first['first']}–{first['last']} | " + " | ".join(cells) + " |")
            lines.append("")
    for role, analysis in (result.get("errors") or {}).items():
        lines.append(errors_markdown(analysis, ["B", START, ARM, FAMILY, x]))
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    try:
        m, _ = load(run, check_code=False)
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），题目 {m['counts']}，起点 v0.8 run "
                     f"{m['imports']['v08_run'][:12]} 的 PTR，{ARM} 为 v0.9 run {m['imports']['v09']['v09_run'][:12]}，"
                     f"金标缓存 { {k: v['statuses'] for k, v in m['gold'].items()} }")
        seeds = m["settings"]["seeds"]
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
        seeds = []
    for name, p in prepass_summary(run, seeds).items():
        facts.append(f"预采样 {name}：边界题 {p['boundary']}/{p['scorable']}（{100 * p['boundary_share']:.1f}%），门禁 "
                     f"{'通过' if p['gate']['passed'] else '未通过'}")
    for folder in sorted((run / "train").glob(f"{FAMILY}_seed*")) if (run / "train").exists() else []:
        record = ended(folder) or paused(folder)
        if record is None:
            continue
        check = record.get("lora_check") or {}
        facts.append(f"{folder.name}：{record.get('steps_completed')} 步（{record.get('stopped_by')}），更新 "
                     f"{record.get('optimizer_updates')} 次，最后里程碑 {record.get('last_milestone')}，LoRA 检查 "
                     f"{'通过' if check.get('passed') else check or '无'}")
    if (run / "steps.json").exists():
        facts.append(f"比较点 K = {read_json(run / 'steps.json')['k']}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, part in r["roles"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['project']['tasks']}" for n, c in part["project"]["conditions"].items())
            facts.append(f"{role}（项目判分）：{cells}")
        for label, d in r.get("decisions", {}).items():
            if d.get("decided"):
                facts.append(f"{label}：{'成立' if d['holds'] else '不成立'}（{d['seeds']} 个 seed），test "
                             f"{100 * d['test'][0]:+.1f} [{100 * d['test'][1][0]:+.1f}, {100 * d['test'][1][1]:+.1f}]")
    counts = Counter(p.parent.parent.name for p in (run / "eval").glob("*/*/summary.json")) if (run / "eval").exists() else {}
    if counts:
        facts.append(f"已完成评测：{dict(counts)}")
    return (f"\n### {now} · GPU 机（v0.11 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_V11_RUNBOOK.md。\n")
