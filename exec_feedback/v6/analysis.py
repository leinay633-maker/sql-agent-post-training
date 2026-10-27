"""v0.6 report: project execution matching (primary) and official BIRD EX (set equality, 30 s) side by side."""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import os
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ..storage import atomic_json, read_json
from .freeze import load
from .rollouts import merged

EX_TIMEOUT = 30.0
FAMILIES = ("R_r1", "R_r2", "Rclean_r1")


def _ex_init() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3,) * 2)
    except (ImportError, ValueError, OSError):
        pass


def _ex_run(path: str, sql: str) -> tuple[str, Any]:
    started, con = time.monotonic(), None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        con.execute("PRAGMA query_only=ON")
        con.set_progress_handler(lambda: int(time.monotonic() - started > EX_TIMEOUT), 10000)
        return "ok", set(con.execute(sql).fetchall())
    except sqlite3.OperationalError as exc:
        return ("timeout" if "interrupted" in str(exc) else "error"), None
    except Exception:  # noqa: BLE001 - the official script scores any exception as 0
        return "error", None
    finally:
        if con is not None:
            con.close()


def _ex_task(job: tuple[str, str, list[str]]) -> dict[str, int]:
    path, gold_sql, preds = job
    status, gold = _ex_run(path, gold_sql)
    out = {}
    for sql in preds:
        p_status, pred = _ex_run(path, sql)
        out[sql] = int(status == "ok" and p_status == "ok" and pred == gold)
    return out


def official_ex(tasks: list[Any], root: Path, runs: dict[str, list[dict[str, Any]]], procs: int) -> dict[str, list[dict[str, Any]]]:
    preds: dict[str, set[str]] = defaultdict(set)
    for rows in runs.values():
        for r in rows:
            if r.get("final_sql"):
                preds[r["task_id"]].add(r["final_sql"])
    order = [t.id for t in tasks]
    by_id = {t.id: t for t in tasks}
    jobs = [(str(by_id[t].database(root)), by_id[t].gold_sql, sorted(preds.get(t, ()))) for t in order]
    with mp.get_context("fork").Pool(procs, initializer=_ex_init, maxtasksperchild=50) as pool:
        results = dict(zip(order, pool.imap(_ex_task, jobs, chunksize=1)))
    return {name: [{"task_id": r["task_id"], "db_id": r["db_id"],
                    "correct": bool(r.get("final_sql")) and results[r["task_id"]].get(r["final_sql"], 0) == 1,
                    "status": "correct" if r.get("final_sql") and results[r["task_id"]].get(r["final_sql"], 0) == 1
                    else "incorrect"} for r in rows] for name, rows in runs.items()}


def pooled(tasks: list[Any], left: dict[str, list[dict[str, Any]]], right: dict[str, list[dict[str, Any]]],
           draws: int = 2000, seed: int = 20261012) -> dict[str, Any]:
    """Seed-matched (or broadcast) right - left; resamples databases and pairs."""
    from ..rl.reporting import aligned

    pairs = []
    for name in right:
        partner = name if name in left else next(iter(left)) if len(left) == 1 else None
        if partner is None:
            seed_id = name.rsplit("_seed", 1)[-1].split("+")[0]
            partner = next((n for n in left if n.rsplit("_seed", 1)[-1].split("+")[0] == seed_id), None)
        if partner is not None:
            pairs.append((partner, name))
    if not pairs:
        return {}
    vectors, complete = [], True
    for a, b in pairs:
        la, lb = aligned(tasks, left[a]), aligned(tasks, right[b])
        complete &= all(r["status"] != "not_attempted" for r in la + lb)
        vectors.append([int(y["correct"]) - int(x["correct"]) for x, y in zip(la, lb)])
    groups = defaultdict(list)
    for i, t in enumerate(tasks):
        groups[t.db_id].append(i)
    dbs, rng, values = sorted(groups), random.Random(seed), []
    for _ in range(draws):
        cols = [i for db in rng.choices(dbs, k=len(dbs)) for i in groups[db]]
        chosen = rng.choices(vectors, k=len(vectors))
        values.append(mean(mean(v[i] for i in cols) for v in chosen))
    values.sort()
    return {"pairs": pairs, "mean_delta": mean(mean(v) for v in vectors), "per_pair": [mean(v) for v in vectors],
            "ci95": [values[int(0.025 * (draws - 1))], values[int(0.975 * (draws - 1))]], "complete": complete}


def family(runs: dict[str, Any], prefix: str, suffix: str = "") -> dict[str, Any]:
    return {n: r for n, r in runs.items() if re.fullmatch(rf"{prefix}_seed\d+{re.escape(suffix)}", n)}


def section(tasks: list[Any], runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from ..rl.reporting import aligned

    part: dict[str, Any] = {"tasks": len(tasks), "databases": len({t.db_id for t in tasks}), "conditions": {}}
    for name, rows in sorted(runs.items()):
        al = aligned(tasks, rows)
        c = sum(r["correct"] for r in al)
        part["conditions"][name] = {"correct": c, "rate": c / len(tasks),
                                    "attempted": sum(r["status"] != "not_attempted" for r in al)}
    deltas = {}
    if "B" in runs:
        for f in FAMILIES:
            for suffix in ("", "+vote"):
                fam = family(runs, f, suffix)
                if fam:
                    deltas[f"{f}{suffix}-B"] = pooled(tasks, {"B": runs["B"]}, fam)
        if "B+vote" in runs:
            deltas["B+vote-B"] = pooled(tasks, {"B": runs["B"]}, {"B+vote": runs["B+vote"]})
    if family(runs, "R_r1") and family(runs, "R_r2"):
        deltas["R_r2-R_r1"] = pooled(tasks, family(runs, "R_r1"), family(runs, "R_r2"))
    if family(runs, "R_r1") and family(runs, "Rclean_r1"):
        deltas["R_r1-Rclean_r1"] = pooled(tasks, family(runs, "Rclean_r1"), family(runs, "R_r1"))
    for name in runs:
        if name.endswith("+vote") and name[:-5] in runs and name != "B+vote":
            deltas[f"{name}-{name[:-5]}"] = pooled(tasks, {name[:-5]: runs[name[:-5]]}, {name: runs[name]})
    part["deltas"] = deltas
    return part


def selection(roles: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """Preregistered: family by validation 3-seed mean (R_r2 only if all seeds complete); vote if it helps on validation."""
    val = roles.get("validation", {}).get("project", {}).get("conditions", {})
    def fam_mean(f: str) -> float | None:
        names = [f"{f}_seed{s}" for s in seeds]
        if not all(n in val and val[n]["attempted"] == roles["validation"]["project"]["tasks"] for n in names):
            return None
        return mean(val[n]["rate"] for n in names)
    r1, r2 = fam_mean("R_r1"), fam_mean("R_r2")
    fam = "R_r2" if r2 is not None and (r1 is None or r2 > r1) else "R_r1" if r1 is not None else None
    if fam is None:
        return {"family": None, "reason": "no complete trained family on validation"}
    lead = f"{fam}_seed{seeds[0]}"
    use_vote = (f"{lead}+vote" in val and val[f"{lead}+vote"]["attempted"] == roles["validation"]["project"]["tasks"]
                and val[f"{lead}+vote"]["rate"] > val[lead]["rate"])
    return {"family": fam, "validation_means": {"R_r1": r1, "R_r2": r2}, "vote": use_vote,
            "headline_condition": f"{lead}+vote" if use_vote else lead,
            "rule": "family = argmax validation 3-seed mean (R_r2 needs all seeds); vote iff it beats greedy on validation"}


def report(run: Path, *, procs: int = 64) -> dict[str, Any]:
    frozen, _ = load(run)
    root = Path(frozen["raw_root"])
    result: dict[str, Any] = {"protocol": frozen["protocol"], "run": frozen["identity"], "model": frozen["model"]["id"],
                              "revision": frozen["model"]["revision"], "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "roles": {}, "notes": "Project execution matching is primary (stricter than official EX); "
                              "official EX re-executes the recorded final SQL only. Test was used for no decision."}
    for role in ("validation", "test", "dev"):
        folder = run / "eval" / role
        if not folder.exists():
            continue
        _, tasks = load(run, role)
        runs, info = {}, {}
        for cond in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            rows, meta = merged(run, role, cond)
            if rows is not None:
                runs[cond], info[cond] = rows, meta
        if not runs:
            continue
        ex = official_ex(tasks, root, runs, procs)
        result["roles"][role] = {"project": section(tasks, runs), "official_ex": section(tasks, ex), "timing": info}
        if role == "test":
            full = (frozen["registration"].get("source_test_screen") or {}).get("candidates")
            if full:
                result["roles"][role]["official_ex_lower_bound_all_dev"] = {
                    n: c["correct"] / full for n, c in result["roles"][role]["official_ex"]["conditions"].items()}
    result["selection"] = selection(result["roles"], frozen["settings"]["seeds"])
    for r in (1, 2):
        p = run / f"round{r}" / "selection.json"
        if p.exists():
            s = read_json(p)
            result[f"round{r}"] = {k: s.get(k) for k in ("trajectories", "sampled_tasks", "attempts_per_task",
                                   "sampled_accuracy", "pass_at_attempts", "frontier_tasks", "easy_tasks", "unsolved_tasks")}
            result[f"round{r}"]["datasets"] = {v: {k: d.get(k) for k in ("examples", "frontier_tasks", "repair_trajectories",
                                               "supervised_tokens", "skipped_overlong")} for v, d in s["datasets"].items()}
    out = run / "report"
    out.mkdir(exist_ok=True)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8")
    return result


def markdown(result: dict[str, Any]) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    pts = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda p: "—" if not p else f"[{100 * p[0]:+.1f}, {100 * p[1]:+.1f}]"
    sel = result["selection"]
    lines = [f"# v0.6 报告（{result['model']} @ {result['revision'][:8]}）", "",
             "项目判分为主指标（比官方 EX 严格）；官方 EX 只重跑记录的最终 SQL。差值单位为百分点，区间为数据库×seed bootstrap 95%。", "",
             f"**预注册选择**：{sel.get('rule')} → 家族 `{sel.get('family')}`，推理 `{sel.get('headline_condition')}`；"
             f"validation 均值 {sel.get('validation_means')}", ""]
    for role, part in result["roles"].items():
        lines += [f"## {role}（{part['project']['tasks']} 题，{part['project']['databases']} 库）", "",
                  "| 条件 | 项目判分 | 官方 EX | 完成 |", "|---|---|---|---|"]
        for name, c in part["project"]["conditions"].items():
            e = part["official_ex"]["conditions"][name]
            lines.append(f"| {name} | {c['correct']} ({pct(c['rate'])}) | {e['correct']} ({pct(e['rate'])}) | "
                         f"{c['attempted']}/{part['project']['tasks']} |")
        lines.append("")
        for key, d in part["project"]["deltas"].items():
            e = part["official_ex"]["deltas"].get(key, {})
            if d:
                lines.append(f"- {key}：项目 {pts(d['mean_delta'])} {ci(d['ci95'])}；官方 EX {pts(e.get('mean_delta'))} "
                             f"{ci(e.get('ci95'))}{'' if d['complete'] else '（不完整）'}")
        lb = part.get("official_ex_lower_bound_all_dev")
        if lb:
            lines.append(f"- 官方 EX 下界（全部 BIRD dev 题，预检排除题记错）：" + "，".join(f"{n} {pct(v)}" for n, v in lb.items()))
        lines.append("")
    for r in (1, 2):
        if f"round{r}" in result:
            lines.append(f"- 第 {r} 轮采样/选样：{result[f'round{r}']}")
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    manifest = run / "manifest.json"
    if manifest.exists():
        m = read_json(manifest)
        facts.append(f"run {m['identity'][:12]}，题目 {m['counts']}，列描述分级 {m['column_notes']['tiers']}，"
                     f"金标缓存 { {k: v['statuses'] for k, v in m['gold'].items()} }")
    for r in (1, 2):
        sel = run / f"round{r}" / "selection.json"
        status = sorted((run / f"round{r}" / "samples").glob("shard*.status.json"))
        if status:
            st = [read_json(p) for p in status]
            done, reg = sum(x.get("completed", 0) for x in st), sum(x.get("registered_items", 0) for x in st)
            facts.append(f"第 {r} 轮采样 {done}/{reg} 条（{len(st)} 片），正确 {sum(x.get('correct', 0) for x in st)}")
        if sel.exists():
            s = read_json(sel)
            facts.append(f"第 {r} 轮选样：边界题 {s['frontier_tasks']}、全对 {s['easy_tasks']}、全错 {s['unsolved_tasks']}；"
                         f"R {s['datasets']['R']['examples']} 例（修复 {s['datasets']['R']['repair_trajectories']}）")
        for t in sorted((run / f"round{r}").glob("train_*_seed*/result.json")):
            x = read_json(t)
            facts.append(f"{t.parent.name}（第 {r} 轮）：完成 {x.get('training_completed')}，重载 {x.get('adapter_reload_verified')}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, part in r["roles"].items():
            cells = "，".join(f"{n} {c['correct']}/{part['project']['tasks']}" for n, c in part["project"]["conditions"].items())
            facts.append(f"{role}（项目判分）：{cells}")
        facts.append(f"预注册选择：{r['selection'].get('headline_condition')}")
    return (f"\n### {now} · GPU 机（v0.6 自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 v06_run.sh 阶段表。\n")
