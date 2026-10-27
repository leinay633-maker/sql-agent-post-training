"""v0.12 report: project execution matching (primary) and official BIRD EX side by side; the preregistered OPD-TR and
OPD-B decisions; exploratory comparisons (OPD-PTR, the teacher, TROPD); cross-night references from v0.8's own
evaluations (R and the teacher on validation, per-task outcomes the bundle carried); training dynamics with the
per-token reverse KL to the teacher and v0.10's three behaviours per 20 steps; the teacher service's load and timing;
the error analysis. report.json/report.md; rebuilt from the exported evaluations at any time."""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..storage import atomic_json, read_json, recover_records
from ..v6.analysis import pooled
from ..v7.analysis import final_flags, official
from ..v9.analysis import restrict
from ..v10.analysis import compact
from ..v10.analysis import summarise as run_summary
from . import ARMS, EXPLORE, FAMILY, TEACHER
from .evaluate import merged
from .freeze import load
from .online import behaviour, ended, paused, run_name

DECISIONS = ((f"{FAMILY}-TR", "主比较"), (f"{FAMILY}-B", "次比较"))
COMPARISONS = ((f"{FAMILY}-TR", "TR", FAMILY), (f"{FAMILY}-B", "B", FAMILY), (f"{FAMILY}-PTR", "PTR", FAMILY),
               ("TR-B", "B", "TR"), ("PTR-B", "B", "PTR"), (f"{FAMILY}-{TEACHER}", TEACHER, FAMILY),
               (f"{TEACHER}-B", "B", TEACHER), (f"{EXPLORE}-TR", "TR", EXPLORE), (f"{EXPLORE}-B", "B", EXPLORE),
               (f"{EXPLORE}-{FAMILY}", FAMILY, EXPLORE))
REFERENCE_TEACHER = f"{TEACHER}v08"   # the teacher as evaluated on v0.8's night
REFERENCES = ((f"{FAMILY}-R", "R", FAMILY), (f"{FAMILY}-{REFERENCE_TEACHER}", REFERENCE_TEACHER, FAMILY),
              ("TR-R", "R", "TR"))
BEHAVIOURS = {"unparsed": "首个输出不可解析", "no_query": "不查数据就提交", "repeat": "连续两次查同一 SQL"}
BLOCK = 20


def family(runs: dict[str, Any], name: str) -> dict[str, Any]:
    if name in ("B", TEACHER, REFERENCE_TEACHER):
        return {name: runs[name]} if name in runs else {}
    return {n: r for n, r in runs.items() if re.fullmatch(rf"{name}_seed\d+", n)}


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
    """Preregistered (v0.7-v0.11's standard): test mean > 0 with its 95% lower bound > 0, and validation mean > 0."""
    test = ((roles.get("test") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    validation = ((roles.get("validation") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    if not test or not validation or not test.get("complete") or not validation.get("complete"):
        return {"decided": False, "holds": None, "reason": "missing or incomplete evaluation"}
    holds = test["mean_delta"] > 0 and test["ci95"][0] > 0 and validation["mean_delta"] > 0
    return {"decided": True, "holds": bool(holds), "seeds": len(test["pairs"]),
            "test": [test["mean_delta"], test["ci95"]], "validation": [validation["mean_delta"], validation["ci95"]]}


def references(run: Path, role: str, tasks: list[Any]) -> dict[str, list[dict[str, Any]]]:
    """v0.8's per-task outcomes for R and the teacher on this role (cross-night, exploratory), restricted to this run's
    tasks (the smoke subset is a part of them)."""
    folder = run / "imports" / "v08" / "eval-rows" / role
    ids = {t.id for t in tasks}
    out = {}
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        name = REFERENCE_TEACHER if path.stem == TEACHER else path.stem
        out[name] = [r for r in read_json(path)["rows"] if r["task_id"] in ids]
    return out


def blocks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per BLOCK training steps: each behaviour's share of the sampled dialogues, the reverse KL to the teacher and the
    execution reward (recorded, never trained on)."""
    out = []
    for start in range(0, len(rows), BLOCK):
        part = rows[start:start + BLOCK]
        total = sum((Counter(r.get("behaviour") or {}) for r in part), Counter())
        n = total.get("dialogues", 0)
        out.append({"first": part[0]["step"], "last": part[-1]["step"], "dialogues": n,
                    **{k: total.get(k, 0) / n if n else None for k in BEHAVIOURS},
                    "reverse_kl": mean(r.get("reverse_kl", 0.0) for r in part),
                    "reward": mean(r["reward_mean"] for r in part)})
    return out


def dynamics(run: Path) -> dict[str, Any]:
    frozen, _ = load(run, check_code=False)
    s = frozen["settings"]
    out = {}
    for fam, seeds in ((FAMILY, s["seeds"]), (EXPLORE, s["explore_seeds"])):
        for seed in seeds:
            folder = run / "train" / run_name(fam, seed)
            d = run_summary(folder)
            if d is None:
                continue
            record, held = ended(folder), paused(folder)
            d["finished"] = bool(record)
            d["stopped_by"] = record["stopped_by"] if record else "paused" if held else None
            rows = recover_records(folder / "metrics.jsonl")
            head, tail = rows[:10], rows[-10:]
            avg = lambda part, key: mean(r[key] for r in part if r.get(key) is not None) if any(
                r.get(key) is not None for r in part) else None
            mid = lambda key: median(r[key] for r in rows if r.get(key) is not None) if any(
                r.get(key) is not None for r in rows) else None
            d.update({"reverse_kl_first10": avg(head, "reverse_kl"), "reverse_kl_last10": avg(tail, "reverse_kl"),
                      "teacher_logp_last10": avg(tail, "teacher_logp"), "sampler_logp_last10": avg(tail, "sampler_logp"),
                      "advantage_low_fraction": avg(rows, "advantage_low_fraction"),
                      "teacher_seconds": mid("teacher_seconds"), "teacher_wait_seconds": mid("teacher_wait_seconds"),
                      "teacher_score_seconds": mid("teacher_score_seconds"),
                      "passes": rows[-1].get("pass", 0) + 1 if rows else 0, "blocks": blocks(rows)})
            curve = recover_records(folder / "curve.jsonl") if (folder / "curve.jsonl").exists() else []
            d["curve_behaviour"] = {r["step"]: r["behaviour"] for r in curve if "behaviour" in r}
            out[run_name(fam, seed)] = d
    return out


def service(home: Path) -> dict[str, Any] | None:
    """The teacher service: its gate, how long it served, and the per-request load and timing."""
    if not home.exists():
        return None
    gate = read_json(home / "gate.json") if (home / "gate.json").exists() else None
    served = read_json(home / "served.json") if (home / "served.json").exists() else None
    rows = recover_records(home / "requests.jsonl") if (home / "requests.jsonl").exists() else []
    by_priority = {}
    for priority in sorted({r["priority"] for r in rows}):
        part = [r for r in rows if r["priority"] == priority]
        by_priority[str(priority)] = {
            "requests": len(part), "sequences": sum(r["sequences"] for r in part),
            "tokens": sum(r["tokens"] for r in part),
            "score_seconds_median": median(r["score_seconds"] for r in part),
            "wait_seconds_median": median(r["wait_seconds"] for r in part),
            "tokens_per_second": sum(r["tokens"] for r in part) / max(1e-9, sum(r["score_seconds"] for r in part))}
    attempts = recover_records(home / "gate-attempts.jsonl") if (home / "gate-attempts.jsonl").exists() else []
    peaks = [max(r["gpu_used_mib"]) for r in rows if r.get("gpu_used_mib")]
    return {"gate": gate, "gate_attempts": sum(1 for a in attempts if "kept" not in a),
            "restarts_after_gate": sum(1 for a in attempts if "kept" in a), "served": served,
            "incomplete_rows": sum(r.get("incomplete", 0) for r in rows), "by_priority": by_priority,
            "gpu_used_mib_after_gate": max(gate["gpu_used_mib"]) if gate and gate.get("gpu_used_mib") else None,
            "gpu_used_mib_max": max(peaks) if peaks else None}


def report(run: Path, *, procs: int = 64, teacher_home: Path | None = None) -> dict[str, Any]:
    from ..v9.errors import analyse

    frozen, _ = load(run)
    s = frozen["settings"]
    root = Path(frozen["raw_root"])
    out = run / "report"
    out.mkdir(exist_ok=True)
    result: dict[str, Any] = {
        "protocol": frozen["protocol"], "run": frozen["identity"], "smoke_limit": frozen["smoke_limit"],
        "v08_run": frozen["imports"]["v08_run"], "planned_steps": s["planned_steps"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "roles": {}, "references": {},
        "notes": "Project execution matching is primary (stricter than official EX); official EX re-executes the "
                 "recorded final SQL only. Only OPD-TR and OPD-B at K are decisions; TROPD (one seed), the teacher "
                 "comparisons and v0.8's cross-night references are exploratory."}
    if (run / "steps.json").exists():
        result["steps"] = read_json(run / "steps.json")
    for role in ("test", "validation", "dev"):
        folder = run / "eval" / role
        names = [p.name for p in sorted(folder.iterdir()) if p.is_dir() and "@" not in p.name] if folder.exists() else []
        _, tasks = load(run, role)
        runs, info, flags = {}, {}, {}
        for name in names:
            rows, meta = merged(run, role, name)
            if rows is not None:
                runs[name], info[name] = rows, meta
                flags[name] = {**final_flags(rows), **behaviour(rows)}
        if runs:
            ex = official(tasks, root, runs, procs)
            result["roles"][role] = {"project": section(tasks, runs, COMPARISONS, flags),
                                     "official_ex": section(tasks, ex, COMPARISONS), "timing": info}
            if role == "test":
                full = (frozen["registration"].get("test_screen") or {}).get("candidates")
                if full:
                    result["roles"][role]["official_ex_lower_bound_all_dev"] = {
                        n: c["correct"] / full for n, c in result["roles"][role]["official_ex"]["conditions"].items()}
            atomic_json(out / f"rows-{role}.json", {"project": compact(runs), "official_ex": compact(ex),
                                                    "flags": flags})
        refs = references(run, role, tasks)
        if refs and runs:
            mixed = {**{n: r for n, r in runs.items() if family({n: r}, FAMILY) or family({n: r}, "TR")}, **refs}
            result["references"][role] = section(tasks, mixed, REFERENCES)
    result["decisions"] = {label: decide(result["roles"], label) for label, _ in DECISIONS}
    result["dynamics"] = dynamics(run)
    result["teacher_service"] = service(teacher_home if teacher_home is not None else run.parent / "teacher")
    result["errors"] = {}
    for role in ("test", "validation"):
        folder = run / "eval" / role
        names = sorted(p.name for p in folder.iterdir() if p.is_dir() and "@" not in p.name) if folder.exists() else []
        if names:
            result["errors"][role] = analyse(run, role, reference="B", conditions=names)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


def markdown(result: dict[str, Any]) -> str:
    from ..v9.errors import markdown as errors_markdown

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda p: "—" if not p else f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    num = lambda x, f="{:.1f}": "—" if x is None else f.format(x)
    lines = [f"# v0.12 报告：同一个 235B 老师，在线蒸馏 OPD 对照离线蒸馏 TR（run {result['run'][:12]}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += ["项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed "
              f"bootstrap 95%，按 seed 配对。{FAMILY} 在预注册比较点 K 评测；{EXPLORE}（从 TR 接着做 OPD，1 个 seed）"
              f"在 min(K, 它的最后里程碑) 评测，只作探索；{TEACHER} 是老师，当晚只评 dev 和 test。", ""]
    steps = result.get("steps")
    if steps:
        lines.append(f"- 比较点 K = {steps['k']} 步；各 run 最后里程碑 {steps['last_milestones']}；"
                     f"过短的 run：{steps['short'] or '无'}；评测点 { {n: v['step'] for n, v in steps['evaluate'].items()} }")
    for label, role_name in DECISIONS:
        d = result["decisions"][label]
        if d["decided"]:
            lines.append(f"- {role_name} {label}（{d['seeds']} 个 seed）：{'成立' if d['holds'] else '不成立'}；"
                         f"test {pts(d['test'][0])} {ci(d['test'][1])}，validation {pts(d['validation'][0])} "
                         f"{ci(d['validation'][1])}")
        else:
            lines.append(f"- {role_name} {label}：未判定（{d['reason']}）")
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
                tag = "（预注册）" if key in dict(DECISIONS) else "（探索）"
                lines.append(f"- {key}{tag}：项目 {pts(d['mean_delta'])} {ci(d['ci95'])}；官方 EX {pts(e.get('mean_delta'))} "
                             f"{ci(e.get('ci95'))}{'' if d['complete'] else '（不完整）'}（{len(d['pairs'])} 对）")
        lb = part.get("official_ex_lower_bound_all_dev")
        if lb:
            lines.append("- 官方 EX 下界（全部 BIRD dev 题，预检排除题记错）：" + "，".join(f"{n} {pct(v)}" for n, v in lb.items()))
        lines.append("")
    if result.get("references"):
        lines += ["## 隔晚参照（v0.8 当晚的 R 和老师，只看项目判分，只作探索）", ""]
        for role, part in result["references"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['tasks']}" for n, c in part["conditions"].items()
                             if n == REFERENCE_TEACHER or n.startswith("R_"))
            lines.append(f"- {role}：{cells}")
            for key, d in part["deltas"].items():
                if d:
                    lines.append(f"  - {key}：{pts(d['mean_delta'])} {ci(d['ci95'])}（{len(d['pairs'])} 对，隔晚）")
        lines.append("")
    dyn = result.get("dynamics") or {}
    if dyn:
        lines += ["## 训练过程", "",
                  "| run | 步数 | 遍数 | 反向 KL 前10→后10 | 奖励（只记录）前10→后10 | 熵 前10→后10 | 生成 token 前10→后10 | "
                  "训推 logp 差 均值/最大 | TIS 截断 | A<−5 的 token | 每步秒（生成/老师/训练） | 老师排队秒 | dev 曲线 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in dyn.items():
            curve = "，".join(f"{step}:{pct(v)}" for step, v in d["dev_curve"].items())
            lines.append(f"| {name} | {d['steps']}（{d['stopped_by'] or '进行中'}） | {d['passes']} | "
                         f"{num(d['reverse_kl_first10'], '{:.3f}')}→{num(d['reverse_kl_last10'], '{:.3f}')} | "
                         f"{pct(d['reward_first10'])}→{pct(d['reward_last10'])} | "
                         f"{num(d['entropy_first10'], '{:.3f}')}→{num(d['entropy_last10'], '{:.3f}')} | "
                         f"{num(d['completion_tokens_first10'], '{:.0f}')}→{num(d['completion_tokens_last10'], '{:.0f}')} | "
                         f"{num(d['mean_abs_logp_gap'], '{:.4f}')}/{num(d['max_abs_logp_gap'], '{:.2f}')} | "
                         f"{pct(d['tis_clipped_fraction'])} | {pct(d['advantage_low_fraction'])} | "
                         f"{num(d['step_seconds'], '{:.0f}')}（{num(d['generate_seconds'], '{:.0f}')}/"
                         f"{num(d['teacher_seconds'], '{:.0f}')}/{num(d['train_seconds'], '{:.0f}')}） | "
                         f"{num(d['teacher_wait_seconds'], '{:.0f}')} | {curve or '—'} |")
        lines.append("")
        runs = [n for n, d in dyn.items() if d.get("blocks")]
        if runs:
            lines += [f"### 每 {BLOCK} 步：{' / '.join(BEHAVIOURS.values())}（反向 KL）", "",
                      "| 步 | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
            depth = max(len(dyn[n]["blocks"]) for n in runs)
            for i in range(depth):
                cells = []
                for n in runs:
                    b = dyn[n]["blocks"][i] if i < len(dyn[n]["blocks"]) else None
                    cells.append("/".join(pct(b[key]) for key in BEHAVIOURS) + f"（{b['reverse_kl']:.3f}）" if b else "")
                first = next(dyn[n]["blocks"][i] for n in runs if i < len(dyn[n]["blocks"]))
                lines.append(f"| {first['first']}–{first['last']} | " + " | ".join(cells) + " |")
            lines.append("")
    svc = result.get("teacher_service")
    if svc:
        g = svc.get("gate") or {}
        restarts = svc.get("restarts_after_gate") or 0
        kept = f"，之后服务重启 {restarts} 次、沿用这次门禁" if restarts else ""
        lines += ["## 老师打分服务", "",
                  f"- 门禁：{'通过' if g.get('passed') else '未通过' if g else '无记录'}（{svc['gate_attempts']} 次{kept}）；"
                  f"{g.get('sequences', '—')} 段、{g.get('sampled_tokens', '—')} 个采样 token，平均绝对差 "
                  f"{num(g.get('mean_abs_gap'), '{:.4f}')}，最大 {num(g.get('max_abs_gap'), '{:.3f}')}，缺失 {g.get('missing', '—')}"]
        dropped = {n: (d.get("unusable") or {}).get("teacher_missing", 0) for n, d in (result.get("dynamics") or {}).items()}
        if any(dropped.values()):  # the training side's count: what actually stayed out of the gradient
            lines.append("- 老师 log 概率缺失或 token 对不上、没进梯度的对话："
                         + "，".join(f"{n} {c} 条" for n, c in dropped.items() if c))
        served = svc.get("served") or {}
        if served:
            lines.append(f"- 服务 {num((served.get('serving_seconds') or 0) / 3600, '{:.2f}')} 小时，停止原因 "
                         f"{served.get('stopped_by')}，请求 {served.get('requests')} 次、{served.get('sequences')} 段、"
                         f"{served.get('tokens')} token，打分忙 {num((served.get('busy_seconds') or 0) / 3600, '{:.2f}')} 小时")
        for priority, p in svc.get("by_priority", {}).items():
            lines.append(f"- 优先级 {priority}：{p['requests']} 次请求，中位打分 {p['score_seconds_median']:.1f} 秒、"
                         f"排队 {p['wait_seconds_median']:.1f} 秒，{p['tokens_per_second']:.0f} token/秒")
        if svc.get("gpu_used_mib_after_gate") or svc.get("gpu_used_mib_max"):
            gib = lambda v: num(v / 1024 if v else None, "{:.1f}")
            lines.append(f"- 老师单卡显存占用（nvidia-smi，含所有进程）：门禁后 {gib(svc.get('gpu_used_mib_after_gate'))} GiB，"
                         f"打分期间最高 {gib(svc.get('gpu_used_mib_max'))} GiB")
        lines.append("")
    for role, analysis in (result.get("errors") or {}).items():
        lines.append(errors_markdown(analysis, ["B", "TR", "PTR", FAMILY, EXPLORE, TEACHER]))
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str, teacher_home: Path | None = None) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    try:
        m, _ = load(run, check_code=False)
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），题目 {m['counts']}，对照臂 v0.8 run "
                     f"{m['imports']['v08_run'][:12]} 的 {'/'.join(ARMS)}，金标缓存 "
                     f"{ {k: v['statuses'] for k, v in m['gold'].items()} }")
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
    svc = service(teacher_home if teacher_home is not None else run.parent / "teacher")
    if svc and svc.get("gate"):
        g = svc["gate"]
        facts.append(f"老师打分门禁：{'通过' if g['passed'] else '未通过'}，平均绝对差 {g['mean_abs_gap']:.4f} nat/token"
                     f"（{g['sampled_tokens']} 个 token）")
    for folder in sorted((run / "train").glob("*_seed*")) if (run / "train").exists() else []:
        record = ended(folder) or paused(folder)
        if record is None:
            continue
        check = record.get("lora_check") or {}
        facts.append(f"{folder.name}：{record.get('steps_completed')} 步（{record.get('stopped_by')}），更新 "
                     f"{record.get('optimizer_updates')} 次，最后里程碑 {record.get('last_milestone')}，LoRA 检查 "
                     f"{check.get('verdict', '无')}")
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
    return (f"\n### {now} · GPU 机（v0.12 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_V12_RUNBOOK.md。\n")
