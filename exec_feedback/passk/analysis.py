"""pass@k report (docs/RESEARCH_PASSK.md): per-task unbiased pass@k of every finished model, family means over seeds,
the registered comparisons at every k with v0.6's database x seed cluster bootstrap, the decisions at k = 1 and at the
largest k, the single-cell / multi-cell strata, the decoding control, descriptives (distinct final SQL, the greedy
scores of the models' own runs, throughput) and the lucky-match audit list (tasks only one side ever solved)."""
from __future__ import annotations

import datetime as dt
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..storage import atomic_json, read_json
from . import BASE, OPD, RL, START
from .freeze import CONTROL, control_tasks, gold, load, parse
from .sample import finished, sample_dir

# (name, left family, right family, role), reported as right - left: B is broadcast to every seed of the right family,
# the trained families pair by seed
COMPARISONS = (("RLb-PTR", START, RL, "primary"), ("PTR-B", BASE, START, "secondary"),
               ("OPD-B", BASE, OPD, "secondary"), ("RLb-B", BASE, RL, "secondary"),
               ("OPD-PTR", START, OPD, "exploratory"))
AUDITED = ("RLb-PTR", "RLb-B")
DECIDED = {"higher": "更高", "lower": "更低", "inseparable": "分不开"}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator 1 - C(n-c, k) / C(n, k): the chance that k of the n dialogues, drawn without replacement,
    include a correct one (Chen et al. 2021)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Need 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curves(rows: list[dict[str, Any]], task_ids: list[str], samples: int, ks: list[int]) -> np.ndarray:
    """(len(ks), tasks): each task's pass@k from its correct count out of `samples`."""
    table = np.array([[pass_at_k(samples, c, k) for k in ks] for c in range(samples + 1)])
    correct = Counter(r["task_id"] for r in rows if r["correct"])
    return table[[correct.get(t, 0) for t in task_ids]].T


def normalise(sql: str) -> str:
    return " ".join(sql.split()).rstrip(";").strip().lower()


def diversity(rows: list[dict[str, Any]], task_ids: list[str]) -> dict[str, float]:
    """Mean distinct final SQL per task (whitespace and case normalised; dialogues without one left out) and the share
    of dialogues without a final SQL."""
    seen: dict[str, set[str]] = {t: set() for t in task_ids}
    for r in rows:
        if r.get("final_sql"):
            seen[r["task_id"]].add(normalise(r["final_sql"]))
    return {"distinct_sql": sum(len(v) for v in seen.values()) / len(task_ids),
            "no_final_share": sum(not r.get("final_sql") for r in rows) / len(rows)}


def pairs(left: str, right: str, models: dict[str, Any], seeds: list[int]) -> list[tuple[str, str]]:
    if left == BASE:
        return [(BASE, f"{right}_seed{s}") for s in seeds if BASE in models and f"{right}_seed{s}" in models]
    return [(f"{left}_seed{s}", f"{right}_seed{s}") for s in seeds
            if f"{left}_seed{s}" in models and f"{right}_seed{s}" in models]


def resample(db_ids: list[str], n_pairs: int, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """v0.6's scheme: each draw takes as many databases as there are, with replacement (a task weighs as often as its
    database was drawn), and as many model pairs as there are, with replacement. Same seed, same draws."""
    rng = np.random.default_rng(seed)
    dbs = sorted(set(db_ids))
    lookup = {d: i for i, d in enumerate(dbs)}
    index = np.array([lookup[d] for d in db_ids])
    weights = rng.multinomial(len(dbs), np.full(len(dbs), 1 / len(dbs)), size=draws)[:, index]
    chosen = rng.multinomial(n_pairs, np.full(n_pairs, 1 / n_pairs), size=draws)
    return weights, chosen


def decide(ci: list[float]) -> str:
    return "higher" if ci[0] > 0 else "lower" if ci[1] < 0 else "inseparable"


def compare(curve: dict[str, np.ndarray], pair_list: list[tuple[str, str]], db_ids: list[str], ks: list[int],
            draws: int, seed: int, columns: np.ndarray | None = None) -> dict[str, Any]:
    """right - left at every k: the point estimate (mean over pairs of the mean over tasks), its 95% interval and the
    decisions at k = 1 and the largest k; `columns` restricts it to a subset of the tasks."""
    cols = np.arange(len(db_ids)) if columns is None else columns
    deltas = np.stack([curve[b][:, cols] - curve[a][:, cols] for a, b in pair_list])  # (pairs, ks, tasks)
    weights, chosen = resample([db_ids[i] for i in cols], len(pair_list), draws, seed)
    per_pair = np.einsum("dt,pkt->dpk", weights, deltas) / weights.sum(1)[:, None, None]
    values = np.sort(np.einsum("dp,dpk->dk", chosen, per_pair) / len(pair_list), axis=0)
    low, high = int(0.025 * (draws - 1)), int(0.975 * (draws - 1))
    point = deltas.mean(axis=(0, 2))
    by_k = {str(k): {"delta": float(point[i]), "ci95": [float(values[low, i]), float(values[high, i])],
                     "per_pair": [float(x) for x in deltas[:, i, :].mean(1)]} for i, k in enumerate(ks)}
    return {"pairs": [list(p) for p in pair_list], "tasks": int(len(cols)), "by_k": by_k,
            "decision": {"k1": decide(by_k[str(ks[0])]["ci95"]), "kmax": decide(by_k[str(ks[-1])]["ci95"])}}


def crossover(dense: dict[str, np.ndarray], pair_list: list[tuple[str, str]],
              columns: np.ndarray | None = None) -> int | None:
    """Descriptive: the smallest k, over every integer up to the sample count, at which right - left's point estimate
    (mean over pairs and tasks) is <= 0; `dense` holds each model's (samples, tasks) pass@k at k = 1 .. samples."""
    cols = slice(None) if columns is None else columns
    point = np.mean([dense[b][:, cols] - dense[a][:, cols] for a, b in pair_list], axis=(0, 2))
    hit = np.flatnonzero(point <= 0)
    return int(hit[0]) + 1 if hit.size else None


def agreement(a: float, b: float) -> str:
    """The decoding control's reading of one difference under both decodings: the same sign, opposite signs, or at
    least one exactly zero (read neither way)."""
    product = np.sign(a) * np.sign(b)
    return "same" if product > 0 else "opposite" if product < 0 else "zero"


def verdict(primary: dict[str, Any] | None, kmax: int) -> dict[str, Any]:
    """Preregistered: RL narrows the boundary iff RLb-PTR is higher at k = 1 and lower at the largest k."""
    if not primary:
        return {"decided": False, "text": "主比较没有可用的模型对"}
    first, last = primary["decision"]["k1"], primary["decision"]["kmax"]
    narrows = first == "higher" and last == "lower"
    if narrows:
        text = f"RL 收窄了能力边界：k=1 更高，k={kmax} 更低"
    elif first == "higher" and last == "inseparable":  # an interval that holds 0 is no evidence of a tie
        text = f"k=1 更高，k={kmax} 未检出差异：无法判断是否追平，既不能说收窄，也不能说扩大"
    elif first == "higher" and last == "higher":
        text = f"k=1 和 k={kmax} 都更高：到 k={kmax} 也没被追上"
    else:
        text = f"k=1 {DECIDED[first]}，k={kmax} {DECIDED[last]}"
    return {"decided": True, "narrows": narrows, "k1": first, "kmax": last, "text": text}


def robust(full: dict[str, Any], multi: dict[str, Any], kmax: int) -> bool | None:
    """A decided largest-k direction holds on the multi-cell tasks if their point estimate has the same sign."""
    direction = full["decision"]["kmax"]
    if direction == "inseparable":
        return None
    value = multi["by_k"][str(kmax)]["delta"]
    return value > 0 if direction == "higher" else value < 0


def audit(frozen: dict[str, Any], tasks: list[Any], rows: dict[str, list[dict[str, Any]]],
          settings: dict[str, Any]) -> dict[str, Any]:
    """Per audited comparison and direction: tasks one family solved at least once (over all its finished models) and
    the other never did; up to audit_per_direction of them drawn with a fixed seed, each with the solving side's first
    correct final SQL (in sampling order), for labelling by hand as logic correct / lucky match / unclear. The items are
    shuffled with a fixed seed and numbered, so the list does not show which side solved a task."""
    by_id = {t.id: t for t in tasks}
    family_rows: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for condition in frozen["conditions"]:
        family, _, control = parse(condition)
        if not control and condition in rows:
            family_rows.setdefault(family, []).extend((condition, r) for r in rows[condition])
    out: dict[str, Any] = {"per_direction": settings["audit_per_direction"], "seed": settings["audit_seed"],
                           "directions": {}, "items": []}
    for name, left, right, _ in COMPARISONS:
        if name not in AUDITED or left not in family_rows or right not in family_rows:
            continue
        first: dict[str, dict[str, tuple[str, dict[str, Any]]]] = {}
        solved: dict[str, Counter] = {}
        for family in (left, right):
            first[family], solved[family] = {}, Counter()
            for condition, r in family_rows[family]:
                if r["correct"]:
                    solved[family][r["task_id"]] += 1
                    first[family].setdefault(r["task_id"], (condition, r))
        for side, other in ((right, left), (left, right)):
            only = sorted(set(first[side]) - set(first[other]))
            drawn = random.Random(f"{settings['audit_seed']}:{name}:{side}").sample(
                only, min(len(only), settings["audit_per_direction"]))
            out["directions"][f"{name}:{side}-only"] = {"tasks": len(only), "drawn": len(drawn),
                                                        "dialogues": {side: len(family_rows[side]) // len(tasks),
                                                                      other: len(family_rows[other]) // len(tasks)}}
            for task_id in sorted(drawn):
                task, (model, row) = by_id[task_id], first[side][task_id]
                out["items"].append({"comparison": name, "solved_by": side, "not_by": other, "task_id": task_id,
                                     "db_id": task.db_id, "question": task.question, "evidence": task.evidence,
                                     "gold_sql": task.gold_sql, "sql": row["final_sql"], "model": model,
                                     "dialogue": row["key"], "solved": solved[side][task_id], "label": None})
    random.Random(f"{settings['audit_seed']}:order").shuffle(out["items"])
    for i, item in enumerate(out["items"], 1):
        item["id"] = i
    return out


def report(run: Path, *, name: str = "report", databases: bool = True) -> dict[str, Any]:
    frozen, tasks = load(run, tasks=True, check_code=False, databases=databases)
    s = frozen["settings"]
    seeds, samples, ks = s["seeds"], s["samples"], s["ks"]
    kmax = ks[-1]
    task_ids, db_ids = [t.id for t in tasks], [t.db_id for t in tasks]
    answers = gold(run, frozen)
    multi = np.array([i for i, t in enumerate(task_ids)
                      if len(answers[t]["rows"]) * len(answers[t]["columns"]) >= 2], dtype=int)
    single = np.array(sorted(set(range(len(task_ids))) - set(multi.tolist())), dtype=int)
    rows, models, missing = {}, {}, []
    for condition in frozen["conditions"]:
        got = finished(run, frozen, condition)
        if got is None:
            missing.append(condition)
        else:
            rows[condition] = got
    main = {c: r for c, r in rows.items() if not parse(c)[2]}
    curve = {c: curves(r, task_ids, samples, ks) for c, r in main.items()}
    dense = {c: curves(r, task_ids, samples, list(range(1, samples + 1))) for c, r in main.items()}  # crossover only
    for condition, r in main.items():
        result = read_json(sample_dir(run, condition) / "result.json")
        meta = result.get("generator") or {}
        models[condition] = {
            "pass": {str(k): float(curve[condition][i].mean()) for i, k in enumerate(ks)},
            "solved_tasks": result["solved_tasks"], "correct_dialogues": result["correct_dialogues"],
            "statuses": result["statuses"], **diversity(r, task_ids),
            "sampling_seconds": result["sampling_seconds"], "attempts": result["attempts"],
            "dialogues_per_second": result["dialogues"] / result["sampling_seconds"] if result["sampling_seconds"] else None,
            "tokens": {k: meta.get(k) for k in ("prompt_tokens", "cached_prompt_tokens", "completion_tokens",
                                                "length_stops", "context_rejections")}}
    families = {}
    for family in (BASE, START, RL, OPD):
        members = [c for c in main if parse(c)[0] == family]
        if members:
            families[family] = {"models": members, "pass": {str(k): float(np.mean([curve[c][i].mean() for c in members]))
                                                            for i, k in enumerate(ks)},
                                "distinct_sql": float(np.mean([models[c]["distinct_sql"] for c in members]))}
    draws, seed = s["bootstrap_draws"], s["bootstrap_seed"]
    comparisons, strata = {}, {"multi": {}, "single": {}}
    for label, left, right, role in COMPARISONS:
        pair_list = pairs(left, right, main, seeds)
        if not pair_list:
            continue
        comparisons[label] = {"role": role, "complete": len(pair_list) == len(seeds),
                              **compare(curve, pair_list, db_ids, ks, draws, seed),
                              "crossover_k": crossover(dense, pair_list)}
        for stratum, cols in (("multi", multi), ("single", single)):
            if len(cols):
                strata[stratum][label] = compare(curve, pair_list, db_ids, ks, draws, seed, columns=cols)
        if label in strata["multi"]:
            comparisons[label]["robust_multi_cell"] = robust(comparisons[label], strata["multi"][label], kmax)
    result = {"run": frozen["identity"], "protocol": frozen["protocol"], "smoke_limit": frozen["smoke_limit"],
              "tasks": len(tasks), "databases": len(set(db_ids)), "samples": samples, "ks": ks, "seeds": seeds,
              "strata_tasks": {"multi": int(len(multi)), "single": int(len(single))},
              "missing": missing, "models": models, "families": families, "comparisons": comparisons,
              "strata": strata, "verdict": verdict(comparisons.get("RLb-PTR"), kmax),
              "control": control_section(frozen, tasks, rows, curve, task_ids, db_ids),
              "greedy": {c: {**g, "rate": g["correct"] / g["tasks"],
                             "pass1": models[c]["pass"]["1"] if c in models else None}
                         for c, g in frozen["imports"].get("greedy", {}).items()},
              "audit": None}
    listing = audit(frozen, tasks, main, s)
    result["audit"] = {k: v for k, v in listing.items() if k != "items"}
    out = run / "report"
    out.mkdir(exist_ok=True)
    atomic_json(out / "audit.json", listing)
    (out / "audit.md").write_text(audit_markdown(listing), encoding="utf-8", newline="\n")
    atomic_json(out / f"{name}.json", result)
    (out / f"{name}.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


def control_section(frozen: dict[str, Any], tasks: list[Any], rows: dict[str, list[dict[str, Any]]],
                    curve: dict[str, np.ndarray], task_ids: list[str], db_ids: list[str]) -> dict[str, Any]:
    """The decoding control on its fixed subset: each control model at both decodings, and (with two models) the later
    minus the earlier at both, on the same tasks."""
    s = frozen["settings"]
    names = s["control"]["conditions"]
    subset = {t.id for t in control_tasks(tasks, s["salt"], s["control"]["tasks"])}
    cols = np.array([i for i, t in enumerate(task_ids) if t in subset], dtype=int)
    ids = [task_ids[i] for i in cols]
    ks, samples = s["ks"], s["samples"]
    out: dict[str, Any] = {"sampling": s["control"]["sampling"], "tasks": int(len(cols)), "models": {}}
    rec = {}
    for name in names:
        if f"{name}{CONTROL}" in rows:
            rec[name] = curves(rows[f"{name}{CONTROL}"], ids, samples, ks)
        out["models"][name] = {
            "recommended": {str(k): float(rec[name][i].mean()) for i, k in enumerate(ks)} if name in rec else None,
            "online": {str(k): float(curve[name][i, cols].mean()) for i, k in enumerate(ks)} if name in curve else None}
    if len(names) == 2 and all(n in rec and n in curve for n in names):
        pair = [(names[0], names[1])]
        sub_db = [db_ids[i] for i in cols]
        draws, seed = s["bootstrap_draws"], s["bootstrap_seed"]
        out["online"] = compare({n: curve[n][:, cols] for n in names}, pair, sub_db, ks, draws, seed)
        out["recommended"] = compare(rec, pair, sub_db, ks, draws, seed)
        out["signs"] = {f"k{k}": agreement(out["online"]["by_k"][str(k)]["delta"],
                                           out["recommended"]["by_k"][str(k)]["delta"]) for k in (ks[0], ks[-1])}
        # preregistered reading at the largest k: opposite signs make the main conclusion depend on the decoding; a
        # zero under either decoding is read neither way
        sign = out["signs"][f"k{ks[-1]}"]
        out["depends_on_decoding"] = None if sign == "zero" else sign == "opposite"
    return out


def audit_markdown(listing: dict[str, Any]) -> str:
    lines = ["# 蒙对抽查清单", "",
             "每题标一个：逻辑正确 / 碰巧对上 / 说不清。只看问题、提示、金标 SQL 和模型的 SQL，不看是哪一方。", ""]
    for key, d in listing["directions"].items():
        lines.append(f"- {key}：{d['tasks']} 题，抽 {d['drawn']} 题（每题对话数 {d['dialogues']}）")
    for item in listing["items"]:
        lines += ["", f"## {item['id']}. {item['task_id']}（{item['db_id']}）", "", f"问题：{item['question']}",
                  f"提示：{item['evidence'] or '无'}", "", "金标 SQL：", "```sql", item["gold_sql"], "```", "",
                  "模型 SQL：", "```sql", item["sql"] or "", "```", "", "标注："]
    return "\n".join(lines) + "\n"


def markdown(result: dict[str, Any]) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: f"{100 * x:+.1f}"
    ci = lambda p: f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    ks, kmax = result["ks"], result["ks"][-1]
    lines = [f"# pass@k 能力边界报告（run {result['run'][:12]}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（{result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += [f"test {result['tasks']} 题（{result['databases']} 个数据库），每个模型每题采 {result['samples']} 段对话。"
              f"pass@k 用无偏估计；差值单位为百分点，区间为数据库×seed 聚类 bootstrap 95%。B 对每个 seed 广播，"
              f"其余按 seed 配对。下界 > 0 判更高，上界 < 0 判更低，否则分不开。", ""]
    v = result["verdict"]
    lines.append(f"- **主比较 RLb−PTR：{v['text']}**")
    if result["control"].get("depends_on_decoding"):
        lines.append(f"- **解码对照在 k={kmax} 方向相反：主结论依赖解码设置**")
    if result["missing"]:
        lines.append(f"- 没采完、不参与比较的模型：{', '.join(result['missing'])}")
    lines.append("")
    lines += ["## 各模型 pass@k", "", "| 模型 | " + " | ".join(f"k={k}" for k in ks) + " | 不同 SQL/题 | 无最终 SQL |",
              "|---|" + "---|" * (len(ks) + 2)]
    for name, m in result["models"].items():
        lines.append(f"| {name} | " + " | ".join(pct(m["pass"][str(k)]) for k in ks)
                     + f" | {m['distinct_sql']:.1f} | {pct(m['no_final_share'])} |")
    for name, f in result["families"].items():
        lines.append(f"| **{name} 均值** | " + " | ".join(pct(f["pass"][str(k)]) for k in ks)
                     + f" | {f['distinct_sql']:.1f} | |")
    lines += ["", "## 比较（后者 − 前者）", "",
              "| 比较 | 性质 | 模型对 | " + " | ".join(f"k={k}" for k in ks) + f" | k=1 | k={kmax} | 交叉点 | 多格子题方向 |",
              "|---|---|---|" + "---|" * (len(ks) + 4)]
    for label, c in result["comparisons"].items():
        role = {"primary": "主", "secondary": "次", "exploratory": "探索"}[c["role"]]
        cells = " | ".join(f"{pts(c['by_k'][str(k)]['delta'])} {ci(c['by_k'][str(k)]['ci95'])}" for k in ks)
        holds = c.get("robust_multi_cell")
        lines.append(f"| {label} | {role} | {len(c['pairs'])}{'' if c['complete'] else '（缺 seed）'} | {cells} | "
                     f"{DECIDED[c['decision']['k1']]} | {DECIDED[c['decision']['kmax']]} | {c['crossover_k'] or '—'} | "
                     f"{'—' if holds is None else '一致' if holds else '不一致（可能受单值答案影响）'} |")
    for stratum, title in (("multi", "金标至少 2 个格子"), ("single", "金标只有 1 个格子")):
        part = result["strata"][stratum]
        if not part:
            continue
        lines += ["", f"## 分层：{title}（{result['strata_tasks'][stratum]} 题）", "",
                  "| 比较 | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for label, c in part.items():
            lines.append(f"| {label} | " + " | ".join(f"{pts(c['by_k'][str(k)]['delta'])} {ci(c['by_k'][str(k)]['ci95'])}"
                                                    for k in ks) + " |")
    control = result["control"]
    lines += ["", f"## 解码对照（{control['tasks']} 题，推荐解码 {control['sampling']}）", ""]
    for name, m in control["models"].items():
        if m["recommended"] and m["online"]:
            lines.append(f"- {name}：温度 1 的 pass@1 {pct(m['online']['1'])}、pass@{kmax} {pct(m['online'][str(kmax)])}；"
                         f"推荐解码 {pct(m['recommended']['1'])}、{pct(m['recommended'][str(kmax)])}")
    if "signs" in control:
        a, b = control["online"]["by_k"], control["recommended"]["by_k"]
        lines.append(f"- 同一批题上后者 − 前者：温度 1 k=1 {pts(a['1']['delta'])} {ci(a['1']['ci95'])}，k={kmax} "
                     f"{pts(a[str(kmax)]['delta'])} {ci(a[str(kmax)]['ci95'])}；推荐解码 k=1 {pts(b['1']['delta'])} "
                     f"{ci(b['1']['ci95'])}，k={kmax} {pts(b[str(kmax)]['delta'])} {ci(b[str(kmax)]['ci95'])}")
        reading = {"same": "两种解码方向相同", "opposite": "两种解码方向相反：主结论依赖解码设置",
                   "zero": "至少一种解码下差值恰为 0，不判是否依赖解码"}[control["signs"][f"k{kmax}"]]
        lines.append(f"- k={kmax}：{reading}")
    else:
        lines.append("- 对照没有采完，不判")
    if result["greedy"]:
        lines += ["", "## 温度 1 的 pass@1 对比贪心（贪心分数取自各模型所在 run 的评测，不是同一晚）", "",
                  "| 模型 | 贪心 | 温度 1 pass@1 |", "|---|---|---|"]
        for name, g in result["greedy"].items():
            lines.append(f"| {name} | {pct(g['rate'])} | {pct(g['pass1'])} |")
    lines += ["", "## 吞吐", "", "| 模型 | 采样秒数 | 对话/秒 | 尝试次数 | 超长被拒 | 达到 max_tokens |", "|---|---|---|---|---|---|"]
    for name, m in result["models"].items():
        rate = m["dialogues_per_second"]
        lines.append(f"| {name} | {m['sampling_seconds']:.0f} | {'—' if rate is None else f'{rate:.2f}'} | "
                     f"{m['attempts']} | {m['tokens'].get('context_rejections')} | {m['tokens'].get('length_stops')} |")
    if result["audit"]:
        lines += ["", "## 蒙对抽查（清单在 audit.md，拉回本机后标注）", ""]
        for key, d in result["audit"]["directions"].items():
            lines.append(f"- {key}：{d['tasks']} 题，抽 {d['drawn']} 题")
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    try:
        m, _ = load(run, check_code=False)
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），test {m['counts']['test']} 题，"
                     f"分层 {m['strata']}，金标 {m['gold']['test']['statuses']}")
        conditions = m["conditions"]
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
        conditions = []
    done = [c for c in conditions if (sample_dir(run, c) / "result.json").exists()]
    if conditions:
        facts.append(f"采完 {len(done)}/{len(conditions)}：{', '.join(done) or '无'}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        facts.append(f"主比较：{r['verdict']['text']}")
        for label, c in r["comparisons"].items():
            a, b = c["by_k"]["1"], c["by_k"][str(r["ks"][-1])]
            facts.append(f"{label}：k=1 {100 * a['delta']:+.1f} [{100 * a['ci95'][0]:+.1f}, {100 * a['ci95'][1]:+.1f}]，"
                         f"k={r['ks'][-1]} {100 * b['delta']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]")
        if "signs" in r["control"]:
            word = {"same": "相同", "opposite": "相反", "zero": "含 0，不判"}[r["control"]["signs"][f"k{r['ks'][-1]}"]]
            facts.append(f"解码对照：k={r['ks'][-1]} 方向{word}")
    return (f"\n### {now} · GPU 机（pass@k 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_PASSK_RUNBOOK.md。\n")
