"""v0.10 report: project execution matching (primary) and official BIRD EX side by side, the preregistered RLF-RL
(full parameters against v0.9's LoRA, same steps and samples) and RLF-PTR decisions, the exploratory one-step-stale
family, training dynamics and dev learning curves, the system's measured costs (where each step's time went, sync
against one-step async, the trainer-only layout benchmark, the weights gate) and the error analysis."""
from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..storage import atomic_json, read_json, recover_records
from ..v6.analysis import pooled
from ..v7.analysis import final_flags, official
from ..v9.analysis import family, restrict
from . import FAMILIES, LORA, START
from .evaluate import merged
from .freeze import load
from .trainer import run_name

COMPARISONS = (  # (label, left, right): right - left, seed-matched when both are families
    ("RLF-RL", LORA, "RLF"), ("RLF-PTR", START, "RLF"), ("RLFa-RLF", "RLF", "RLFa"), ("RL-PTR", START, LORA),
    ("RLFa-PTR", START, "RLFa"), ("PTR-B", "B", START), ("RLF-B", "B", "RLF"),
)
DECISIONS = (("RLF-RL", "主比较"), ("RLF-PTR", "次比较"))
EQUIVALENCE = 0.02  # "no difference" is said only when the whole test 95% interval lies within +-2 points
NAMES = ["B", START, LORA, "RLF", "RLFa"]


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
    """Preregistered: holds if the test mean > 0 with its 95% lower bound > 0 and the validation mean > 0 (project
    scoring); reversed if the mirror image holds; equivalent if the test interval lies within +-EQUIVALENCE."""
    test = ((roles.get("test") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    validation = ((roles.get("validation") or {}).get("project") or {}).get("deltas", {}).get(label) or {}
    if not test or not validation or not test.get("complete") or not validation.get("complete"):
        return {"decided": False, "holds": None, "reason": "missing or incomplete evaluation"}
    holds = test["mean_delta"] > 0 and test["ci95"][0] > 0 and validation["mean_delta"] > 0
    reversed_ = test["mean_delta"] < 0 and test["ci95"][1] < 0 and validation["mean_delta"] < 0
    return {"decided": True, "holds": bool(holds), "reversed": bool(reversed_),
            "equivalent": bool(-EQUIVALENCE < test["ci95"][0] and test["ci95"][1] < EQUIVALENCE),
            "seeds": len(test["pairs"]), "test": [test["mean_delta"], test["ci95"]],
            "validation": [validation["mean_delta"], validation["ci95"]]}


def _last_per_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Curve points by step; a step evaluated again after a resume keeps its latest record."""
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("step") is not None and "correct" in row:
            out[int(row["step"])] = row
    return out


def summarise(folder: Path) -> dict[str, Any] | None:
    """Training dynamics and system costs of one online run directory (v0.10's, or v0.9's imported LoRA run)."""
    rows = recover_records(folder / "metrics.jsonl") if (folder / "metrics.jsonl").exists() else []
    if not rows:
        return None
    head, tail = rows[:10], rows[-10:]
    avg = lambda part, key: mean(r[key] for r in part if r.get(key) is not None) if any(
        r.get(key) is not None for r in part) else None
    mid = lambda key: median(r[key] for r in rows if r.get(key) is not None) if any(
        r.get(key) is not None for r in rows) else None
    groups = lambda r: sum(r.get(k, 0) for k in ("informative_groups", "all_correct_groups", "all_failed_groups"))
    result = next((read_json(folder / n) for n in ("result.json", "stopped.json") if (folder / n).exists()), {})
    curve = _last_per_step(recover_records(folder / "curve.jsonl")) if (folder / "curve.jsonl").exists() else {}
    worst = lambda key: median(max(r[key]) for r in rows if r.get(key)) if any(r.get(key) for r in rows) else None
    peak = max((r.get("peak_trainer_gpu_bytes") or r.get("peak_gpu_bytes") or 0) for r in rows) / 1e9
    return {
        "steps": rows[-1]["step"], "optimizer_updates": rows[-1]["optimizer_updates"],
        "finished": result.get("finished"), "stopped_by": result.get("stopped_by"),
        "last_milestone": result.get("last_milestone"),
        "reward_first10": avg(head, "reward_mean"), "reward_last10": avg(tail, "reward_mean"),
        "informative_share": sum(r.get("informative_groups", 0) for r in rows) / max(1, sum(groups(r) for r in rows)),
        "trained_per_step": avg(rows, "trained_trajectories"),
        "entropy_first10": avg(head, "entropy"), "entropy_last10": avg(tail, "entropy"),
        "turns_first10": avg(head, "mean_turns"), "turns_last10": avg(tail, "mean_turns"),
        "completion_tokens_first10": avg(head, "mean_completion_tokens"),
        "completion_tokens_last10": avg(tail, "mean_completion_tokens"),
        "mean_abs_logp_gap": avg(rows, "mean_abs_logp_gap"),
        "max_abs_logp_gap": max(r.get("max_abs_logp_gap", 0.0) for r in rows),
        "tis_clipped_fraction": avg(rows, "tis_clipped_fraction"), "staleness": avg(rows, "staleness"),
        "grad_norm": mid("grad_norm"),
        "unusable": dict(sum((Counter(r.get("unusable") or {}) for r in rows), Counter())),
        "context_rejections": sum(r.get("context_rejections", 0) for r in rows),
        # system (medians per step; v0.9's single-GPU runs only have generate/train seconds)
        "step_seconds": mid("step_seconds"), "wait_seconds": mid("wait_seconds"),
        "generate_seconds": mid("generate_seconds"), "rollout_seconds": worst("rollout_seconds"),
        "engine_load_seconds": worst("load_seconds"), "train_seconds": mid("train_seconds"),
        "optimizer_seconds": mid("optimizer_seconds"), "gather_seconds": mid("gather_seconds"),
        "write_seconds": mid("write_seconds"), "train_tokens": mid("train_tokens"),
        "train_tokens_per_second": mid("train_tokens_per_second"), "mfu": mid("mfu"), "peak_gpu_gb": peak,
        "active_hours": (rows[-1].get("active_seconds") or 0.0) / 3600,
        "weights_check": result.get("weights_check") or result.get("lora_check"),
        "dev_curve": {step: r["correct"] / r["tasks"] for step, r in sorted(curve.items())}}


def dynamics(run: Path) -> dict[str, Any]:
    frozen, _ = load(run, check_code=False)
    out = {}
    for seed in frozen["settings"]["seeds"]:
        lora = run / "imports" / "v09" / "train" / f"{LORA}_seed{seed}"
        if (d := summarise(lora)) is not None:
            out[f"{LORA}_seed{seed}"] = d
        for fam in FAMILIES:
            if (d := summarise(run / "train" / run_name(fam, seed))) is not None:
                out[run_name(fam, seed)] = d
    return out


def systems(run: Path, runs: dict[str, Any], smoke: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frozen, _ = load(run, check_code=False)
    speed = {}
    for seed in frozen["settings"]["seeds"]:
        sync, stale = runs.get(run_name("RLF", seed)), runs.get(run_name("RLFa", seed))
        if sync and stale and sync.get("step_seconds") and stale.get("step_seconds"):
            speed[str(seed)] = sync["step_seconds"] / stale["step_seconds"]
    out["async_speedup"] = speed
    if (run / "bench" / "summary.json").exists():
        out["bench"] = read_json(run / "bench" / "summary.json")
    for where in (run, smoke):
        if where is not None and (where / "gate" / "weights.json").exists():
            out["weights_gate"] = read_json(where / "gate" / "weights.json")
            break
    return out


def compact(runs: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Per-task outcomes only: what a report over several bookings pools (the dialogues stay in eval/)."""
    return {name: [{k: r[k] for k in ("task_id", "db_id", "correct", "status")} for r in rows]
            for name, rows in runs.items()}


def report(run: Path, *, procs: int = 64, smoke: Path | None = None) -> dict[str, Any]:
    from ..v9.errors import analyse

    frozen, _ = load(run)
    root = Path(frozen["raw_root"])
    out = run / "report"
    out.mkdir(exist_ok=True)
    result: dict[str, Any] = {
        "protocol": frozen["protocol"], "run": frozen["identity"], "smoke_limit": frozen["smoke_limit"],
        "v08_run": frozen["imports"]["v08_run"], "v09_run": frozen["imports"]["v09"]["v09_run"],
        "k": frozen["settings"]["planned_steps"], "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "roles": {}, "notes": "Project execution matching is primary (stricter than official EX); official EX "
                              "re-executes the recorded final SQL only. Every condition is a full-weight model on one "
                              "inference path; LoRA adapters were merged before evaluation."}
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
        atomic_json(out / f"rows-{role}.json", {"project": compact(runs), "official_ex": compact(ex), "flags": flags})
    result["decisions"] = {label: decide(result["roles"], label) for label, _ in DECISIONS}
    result["dynamics"] = dynamics(run)
    result["systems"] = systems(run, result["dynamics"], smoke)
    result["errors"] = {}
    for role in ("test", "validation"):
        if (run / "eval" / role).exists():
            result["errors"][role] = analyse(run, role, reference="B")
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


def _agreement(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    x, y = ({r["task_id"]: r["correct"] for r in rows} for rows in (a, b))
    common = set(x) & set(y)
    return sum(x[t] == y[t] for t in common) / max(1, len(common))


def combined(runs: list[Path], out: Path) -> dict[str, Any]:
    """One report over the bookings. Each booking is a fresh host that froze its own run (same registration, imports
    and settings) and trained its own seeds; per-task outcomes come from each run's report (rows-<role>.json), so no
    database is needed here. Each condition is taken from the first run that has it; B from the first run, and every
    later run's B is compared with it task by task (same model, same greedy decoding, another host). The layout
    benchmark comes from the first booking that measured every layout, the weights gate from the first booking's report."""
    from ..core import Task, digest, read_jsonl

    frozen = [load(r, check_code=False)[0] for r in runs]
    if len({f["identity"] for f in frozen}) != len(frozen) or any(f["smoke_limit"] for f in frozen):
        raise ValueError("Expected distinct main (non-smoke) runs")
    for f in frozen[1:]:
        for key in ("protocol", "settings", "registration", "sha256", "counts"):
            if f[key] != frozen[0][key]:
                raise ValueError(f"Runs differ in {key}; they cannot be pooled")
        if f["models"]["student"]["files_sha256"] != frozen[0]["models"]["student"]["files_sha256"] or any(
                f["imports"][k] != frozen[0]["imports"][k] for k in ("v08_run", "adapters", "v09")):
            raise ValueError("Runs start from other models or imports; they cannot be pooled")
    first = frozen[0]
    result: dict[str, Any] = {
        "protocol": first["protocol"], "run": "+".join(f["identity"][:8] for f in frozen),
        "runs": [{"identity": f["identity"], "code": f["code"]} for f in frozen], "smoke_limit": 0,
        "v08_run": first["imports"]["v08_run"], "v09_run": first["imports"]["v09"]["v09_run"],
        "k": first["settings"]["planned_steps"], "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "title": f"# v0.10 合并报告（{len(runs)} 次预约：" + "、".join(f["identity"][:8] for f in frozen) + "）",
        "roles": {}, "errors": {}}
    for role in ("test", "validation", "dev"):
        if digest(runs[0] / f"{role}.jsonl") != first["sha256"][role]:
            raise ValueError("Registered task file changed")
        tasks = [Task(**row) for row in read_jsonl(runs[0] / f"{role}.jsonl")]
        project, ex, flags, sources, b_agreement = {}, {}, {}, {}, {}
        for run, f in zip(runs, frozen):
            path = run / "report" / f"rows-{role}.json"
            if not path.exists():
                continue
            saved = read_json(path)
            for name, rows in saved["project"].items():
                if name in project:
                    if name == "B":
                        b_agreement[f["identity"][:8]] = _agreement(project["B"], rows)
                    continue
                project[name], ex[name], flags[name] = rows, saved["official_ex"][name], saved["flags"].get(name)
                sources[name] = f["identity"][:8]
        if project:
            result["roles"][role] = {"project": section(tasks, project, flags), "official_ex": section(tasks, ex),
                                     "sources": sources, "b_agreement": b_agreement}
    result["decisions"] = {label: decide(result["roles"], label) for label, _ in DECISIONS}
    result["dynamics"], speed, benches, gates = {}, {}, [], []
    for run in runs:
        dyn = dynamics(run)
        result["dynamics"].update(dyn)
        found = systems(run, dyn, None)
        speed.update(found["async_speedup"])
        if "bench" in found:
            benches.append(found["bench"])
        own = run / "report" / "report.json"  # the gate ran on that booking's smoke run; its own report carries it
        if own.exists() and (gate := (read_json(own).get("systems") or {}).get("weights_gate")):
            gates.append(gate)
    result["systems"] = {"async_speedup": speed}
    if benches:  # a booking whose main runs ended late has none; the first that measured every layout, else any
        result["systems"]["bench"] = next((b for b in benches if b.get("complete")), benches[0])
    if gates:
        result["systems"]["weights_gate"] = gates[0]
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


def markdown(result: dict[str, Any]) -> str:
    from ..v9.errors import markdown as errors_markdown

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda p: "—" if not p else f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    num = lambda x, f="{:.1f}": "—" if x is None else f.format(x)
    lines = [result.get("title") or f"# v0.10 报告（分布式全参在线 RL，对照 v0.9 LoRA，run {result['run'][:12]}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += [f"项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed "
              f"bootstrap 95%。所有条件都按全参权重评测（LoRA 先合并），比较点 K = {result['k']} 步。", ""]
    for label, name in DECISIONS:
        d = result["decisions"][label]
        if d["decided"]:
            verdict = "成立" if d["holds"] else "反向成立" if d["reversed"] else "不成立"
            extra = f"；test 区间在 ±{100 * EQUIVALENCE:.0f} 点内（可称无差别）" if d["equivalent"] else ""
            if d["seeds"] < 3:  # one seed per booking; the preregistered decision waits for all three
                extra += "；初步，预注册的判定在三个 seed 都完成后做"
            lines.append(f"- {name} {label}（{d['seeds']} 个 seed）：{verdict}；test {pts(d['test'][0])} "
                         f"{ci(d['test'][1])}，validation {pts(d['validation'][0])} {ci(d['validation'][1])}{extra}")
        else:
            lines.append(f"- {name} {label}：未判定（{d['reason']}）")
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
        if part.get("sources"):
            lines.append("- 各条件来自的 run：" + "，".join(f"{n} {r}" for n, r in part["sources"].items()))
        if part.get("b_agreement"):
            lines.append("- B 在其他机器上重评，与第一次逐题一致的比例：" +
                         "，".join(f"{r} {pct(v)}" for r, v in part["b_agreement"].items()))
        for key, d in part["project"]["deltas"].items():
            e = part["official_ex"]["deltas"].get(key, {})
            if d:
                lines.append(f"- {key}：项目 {pts(d['mean_delta'])} {ci(d['ci95'])}；官方 EX {pts(e.get('mean_delta'))} "
                             f"{ci(e.get('ci95'))}{'' if d['complete'] else '（不完整）'}（{len(d['pairs'])} 对）")
        lines.append("")
    if result.get("dynamics"):
        lines += ["## 训练过程（RL 为导入的 v0.9 LoRA run）", "",
                  "| run | 步数 | 奖励 前10→后10 | 有信号组占比 | 每步训练轨迹 | 熵 前10→后10 | 生成 token 前10→后10 | "
                  "训推 logp 差 均值/最大 | TIS 截断 | 陈旧步数 | dev 曲线 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in result["dynamics"].items():
            curve = "，".join(f"{k}:{pct(v)}" for k, v in d["dev_curve"].items())
            lines.append(f"| {name} | {d['steps']}（{d['stopped_by'] or '进行中'}） | {pct(d['reward_first10'])}→"
                         f"{pct(d['reward_last10'])} | {pct(d['informative_share'])} | {num(d['trained_per_step'])} | "
                         f"{num(d['entropy_first10'], '{:.3f}')}→{num(d['entropy_last10'], '{:.3f}')} | "
                         f"{num(d['completion_tokens_first10'], '{:.0f}')}→{num(d['completion_tokens_last10'], '{:.0f}')} | "
                         f"{num(d['mean_abs_logp_gap'], '{:.4f}')}/{num(d['max_abs_logp_gap'], '{:.2f}')} | "
                         f"{pct(d['tis_clipped_fraction'])} | {num(d['staleness'], '{:.2f}')} | {curve or '—'} |")
        lines += ["", "## 系统：每步时间去向（中位数，秒）", "",
                  "| run | 每步 | 等采样 | 采样（最慢 worker） | 引擎换权重 | 训练 | 优化器 | 收集权重 | 写权重 | 训练 token/s | "
                  "MFU | 训练端显存峰值 | 训练小时 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in result["dynamics"].items():
            lines.append(f"| {name} | {num(d['step_seconds'])} | {num(d['wait_seconds'])} | "
                         f"{num(d['rollout_seconds'] if d['rollout_seconds'] is not None else d['generate_seconds'])} | "
                         f"{num(d['engine_load_seconds'])} | {num(d['train_seconds'])} | {num(d['optimizer_seconds'], '{:.2f}')} | "
                         f"{num(d['gather_seconds'])} | {num(d['write_seconds'])} | {num(d['train_tokens_per_second'], '{:.0f}')} | "
                         f"{pct(d['mfu'])} | {d['peak_gpu_gb']:.1f} GB | {d['active_hours']:.2f} |")
        lines.append("")
    systems = result.get("systems") or {}
    if systems.get("async_speedup"):
        lines.append("- 一步异步相对同步的每步加速（同步每步中位数 ÷ 异步）：" +
                     "，".join(f"seed{k} {v:.2f}×" for k, v in systems["async_speedup"].items()))
    bench = (systems.get("bench") or {}).get("layouts") or {}
    if bench:
        lines += ["", f"### 只测训练端（{systems['bench'].get('family')} seed{systems['bench'].get('seed')} 前几步的同一批轨迹；"
                      "首步梯度范数在各布局间应相同，否则 loss 缩放没抵消 FSDP 的平均）", "",
                  "| 布局 | 训练秒 | 优化器秒 | 收集秒 | 写秒 | token/s | MFU | 显存峰值 | 首步梯度范数 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, b in bench.items():
            if b.get("exit") == 0 and "train_seconds" in b:
                norms = b.get("first_grad_norms") or []
                lines.append(f"| {name} | {b['train_seconds']:.1f} | {b['optimizer_seconds']:.2f} | {b['gather_seconds']:.1f} | "
                             f"{b['write_seconds']:.1f} | {b['tokens_per_second']:.0f} | {pct(b['mfu'])} | "
                             f"{b['peak_gpu_bytes'] / 1e9:.1f} GB | {num(norms[0], '{:.4f}') if norms else '—'} |")
            else:
                lines.append(f"| {name} | 失败（exit {b.get('exit')}） | | | | | | | |")
    gate = systems.get("weights_gate")
    if gate:
        lines += ["", f"- 权重门禁：{'通过' if gate['passed'] else '未通过'}；{gate.get('checks')}；整份权重换入 "
                      f"{num((gate.get('seconds') or {}).get('full_load'))} 秒，写出 {num((gate.get('seconds') or {}).get('write_full'))} 秒"]
    lines.append("")
    for role, analysis in (result.get("errors") or {}).items():
        lines.append(errors_markdown(analysis, NAMES))
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    try:
        m, _ = load(run, check_code=False)
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），题目 {m['counts']}，"
                     f"K = {m['settings']['planned_steps']}，LoRA 臂 v0.9 run {m['imports']['v09']['v09_run'][:12]}")
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
    for folder in sorted((run / "train").glob("*_seed*")) if (run / "train").exists() else []:
        r = next((read_json(folder / n) for n in ("result.json", "stopped.json") if (folder / n).exists()), None)
        if r is None:
            continue
        d = summarise(folder) or {}
        facts.append(f"{folder.name}：{r.get('steps_completed')} 步（{r.get('stopped_by')}），更新 {r.get('optimizer_updates')} 次，"
                     f"每步中位 {d.get('step_seconds') or 0:.1f} 秒，权重检查 "
                     f"{'通过' if (r.get('weights_check') or {}).get('passed') else r.get('weights_check')}")
    gate = run / "gate" / "weights.json"
    if gate.exists():
        facts.append(f"权重门禁 {'通过' if read_json(gate)['passed'] else '未通过'}")
    if (run / "bench" / "summary.json").exists():
        b = read_json(run / "bench" / "summary.json")
        facts.append("训练端基准 " + "，".join(f"{k} {v.get('train_seconds', 0):.1f}s" if v.get("exit") == 0 else
                                         f"{k} 失败" for k, v in b["layouts"].items()))
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, part in r["roles"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['project']['tasks']}" for n, c in part["project"]["conditions"].items())
            facts.append(f"{role}（项目判分）：{cells}")
        for label, d in r.get("decisions", {}).items():
            if d.get("decided"):
                facts.append(f"{label}：{'成立' if d['holds'] else '反向成立' if d['reversed'] else '不成立'}"
                             f"（{d['seeds']} 个 seed{'，初步' if d['seeds'] < 3 else ''}），test "
                             f"{100 * d['test'][0]:+.1f} [{100 * d['test'][1][0]:+.1f}, {100 * d['test'][1][1]:+.1f}]")
    counts = Counter(p.parent.parent.name for p in (run / "eval").glob("*/*/summary.json")) if (run / "eval").exists() else {}
    if counts:
        facts.append(f"已完成评测：{dict(counts)}")
    return (f"\n### {now} · GPU 机（v0.10 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_V10_RUNBOOK.md。\n")
