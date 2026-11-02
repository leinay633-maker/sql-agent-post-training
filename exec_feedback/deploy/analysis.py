"""Deployment report (docs/RESEARCH_DEPLOY.md): what merging and FP8 cost on test and validation (greedy, project
grader, v0.6's bootstrap over databases); the merge record; per deployment and concurrency the load test's latency,
throughput, token rates, prefix-cache hit rate, peak KV use and GPU memory, and the service answers scored against the
gold; and the alignment of the serving path with the evaluation path on the load test's tasks."""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any

from ..core import Limits, compare, execute
from ..storage import atomic_json, read_json, recover_records
from . import CONDITIONS, MODEL
from .bench import failed
from .evaluate import rows_of
from .freeze import gold, load, merged_model

DELTAS = (("merged", "lora", "merge"), ("fp8", "merged", "FP8"), ("fp8", "lora", "merge + FP8"))


def quantile(values: list[float], q: float) -> float | None:
    """Nearest rank."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def attempts_of(folder: Path) -> list[Path]:
    return sorted(folder.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]))


def attempt_of(folder: Path) -> Path | None:
    """The load-test attempt to report: the one a deployment's result names, else the latest that measured a level,
    else the latest (none: not run)."""
    if (folder / "result.json").exists():
        return folder / f"attempt-{read_json(folder / 'result.json')['attempt']}"
    attempts = attempts_of(folder)
    measured = [a for a in attempts if any(a.glob("level-*.json"))]
    return (measured or attempts or [None])[-1]


def ending(attempt: Path) -> dict[str, Any]:
    """How an attempt ended; one without a final record was killed outright (its journals are partial evidence)."""
    record = read_json(attempt / "attempt.json") if (attempt / "attempt.json").exists() else {}
    status = record.get("status", "running")
    return {"attempt": int(attempt.name.split("-")[1]), "status": "killed" if status == "running" else status,
            "complete": bool(record.get("complete")), "failure": record.get("failure"), "stopped": record.get("stopped")}


def scored(tasks: dict[str, Any], rows: list[dict[str, Any]], answers: dict[str, Any], root: Path,
           limits: Limits) -> list[bool]:
    """Each service answer's final SQL scored by the project grader against the frozen gold (no SQL: wrong)."""
    def one(row: dict[str, Any]) -> bool:
        task = tasks[row["task_id"]]
        if row.get("code") != 200 or not row.get("final_sql"):
            return False
        return bool(compare(execute(task.database(root), row["final_sql"], limits), answers[task.id],
                            task.ordered)["correct"])
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, rows))


def level_stats(data: dict[str, Any], correct: list[bool]) -> dict[str, Any]:
    """One concurrency level. Served: the service answered with its model (HTTP 200, no provider or schema error);
    latency and throughput count those, the all-request latency is beside them, accuracy is over every request."""
    rows, wall = data["rows"], data["wall_seconds"]
    served = [r for r in rows if not failed(r)]
    seconds, every = [r["seconds"] for r in served], [r["seconds"] for r in rows]
    calls = [c for r in served for c in (r.get("calls") or [])]
    before, after = data["metrics_before"], data["metrics_after"]
    delta = lambda name: after.get(name, 0.0) - before.get(name, 0.0) if name in after else None
    queries, hits = delta("vllm:prefix_cache_queries"), delta("vllm:prefix_cache_hits")
    prompt, generated = delta("vllm:prompt_tokens"), delta("vllm:generation_tokens")
    return {"concurrency": data["concurrency"], "requests": len(rows), "served": len(served),
            "failures": len(rows) - len(served), "stopped": data.get("stopped", "done"),
            "failure_reasons": dict(Counter(f"{r.get('code')}:{r.get('status')}:{r.get('reason') or r.get('error')}"
                                            for r in rows if failed(r))),
            "p50_seconds": quantile(seconds, 0.5), "p95_seconds": quantile(seconds, 0.95),
            "p50_all_seconds": quantile(every, 0.5), "p95_all_seconds": quantile(every, 0.95),
            "mean_seconds": fmean(seconds) if seconds else None, "wall_seconds": wall,
            "tasks_per_second": len(served) / wall if wall else None,
            "model_calls": len(calls), "calls_per_task": len(calls) / len(served) if served else None,
            "service_tokens_per_second": sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) for c in calls)
            / wall if wall else None,
            "generated_tokens_per_second": generated / wall if generated is not None and wall else None,
            "prompt_tokens_per_second": prompt / wall if prompt is not None and wall else None,
            "prefix_cache_hit_rate": hits / queries if queries else None,
            "prefix_cache_reset": data["prefix_cache_reset"], "peaks": data["peaks"],
            "correct": sum(correct), "accuracy": sum(correct) / len(rows) if rows else None}


def agreement(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, Any]:
    """Two greedy paths on the same tasks: final SQL compared verbatim (case and spacing inside literals matter), two
    missing SQL counted apart, not as a match; service rows the infrastructure failed are left out and counted."""
    usable = lambda row: not failed({"code": 200, **row})  # evaluation rows have no HTTP code
    present = [i for i in ids if i in left and i in right]
    both = [i for i in present if usable(left[i]) and usable(right[i])]
    sql = lambda row: row.get("final_sql")
    return {"tasks": len(both), "left_out": len(present) - len(both),
            "same_final_sql": sum(sql(left[i]) is not None and sql(left[i]) == sql(right[i]) for i in both),
            "both_without_sql": sum(sql(left[i]) is None and sql(right[i]) is None for i in both),
            "same_correctness": sum(left[i]["correct"] == right[i]["correct"] for i in both),
            "left_correct": sum(left[i]["correct"] for i in both), "right_correct": sum(right[i]["correct"] for i in both)}


def first_prompt_tokens(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, int]:
    """The evaluation path's first-turn prompt length per task: its system and user messages through the chat template,
    as the offline engine received them."""
    out = {}
    for row in rows:
        messages = row.get("messages") or []
        if len(messages) >= 2 and tokenizer is not None:
            out[row["task_id"]] = len(tokenizer.apply_chat_template(messages[:2], tokenize=True,
                                                                    add_generation_prompt=True))
    return out


def report(run: Path, *, tokenizer: Any = None) -> dict[str, Any]:
    from ..v6.analysis import pooled

    frozen, _ = load(run, check_code=False)
    s = frozen["settings"]
    root, limits = Path(frozen["raw_root"]), Limits(**s["limits"])
    result: dict[str, Any] = {"run": frozen["identity"], "protocol": frozen["protocol"], "model": MODEL,
                              "adapter": frozen["imports"]["v11"]["adapters"][MODEL], "smoke_limit": frozen["smoke_limit"],
                              "accuracy": {}, "deltas": {}, "bench": {}, "alignment": {}}
    try:
        result["merge"] = {k: v for k, v in merged_model(run, frozen)[1].items() if k != "adapter"}
    except ValueError as exc:
        result["merge"] = {"missing": str(exc)}
    evals: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for role in s["roles"]:
        _, tasks = load(run, role, check_code=False)
        evals[role] = {c: r for c in CONDITIONS if (r := rows_of(run, role, c)) is not None}
        result["accuracy"][role] = {c: {"correct": sum(x["status"] == "correct" for x in r), "tasks": len(tasks),
                                        "rate": sum(x["status"] == "correct" for x in r) / len(tasks)}
                                    for c, r in evals[role].items()}
        for c in CONDITIONS:  # an evaluation cut short (or killed): how far its current journal got, never a rate
            journal = run / "eval" / role / c / "episodes.jsonl"
            if c not in evals[role] and journal.exists():
                result["accuracy"][role][c] = {"incomplete": True, "episodes": len(recover_records(journal)),
                                               "tasks": len(tasks)}
        for right, left, label in DELTAS:
            if left in evals[role] and right in evals[role]:
                d = pooled(tasks, {left: evals[role][left]}, {right: evals[role][right]},
                           draws=s["bootstrap_draws"], seed=s["bootstrap_seed"])
                result["deltas"].setdefault(role, {})[f"{right}-{left}"] = {
                    "label": label, "delta": d["mean_delta"], "ci95": d["ci95"], "complete": d["complete"]}
    _, test = load(run, "test", check_code=False)
    by_id, answers = {t.id: t for t in test}, gold(run, frozen, "test")
    ids = frozen["bench"]["tasks"]
    if tokenizer is None:
        try:
            from ..models import load_tokenizer
            tokenizer = load_tokenizer(frozen["models"]["student"]["spec"])
        except Exception:  # noqa: BLE001 - no tokenizer here: the prompt-length check is reported as not run
            tokenizer = None
    for condition in CONDITIONS:
        attempt = attempt_of(run / "bench" / condition)
        if attempt is None:
            continue
        done = sorted(attempt.glob("level-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        levels, service = {}, {}
        for path in done:
            data = read_json(path)
            correct = scored(by_id, data["rows"], answers, root, limits)
            levels[str(data["concurrency"])] = level_stats(data, correct)
            if data.get("stopped", "done") == "done":  # a level cut short or incomplete is reported, not aligned
                service[data["concurrency"]] = {r["task_id"]: {**r, "correct": c} for r, c in zip(data["rows"], correct)}
        record = read_json(attempt / "attempt.json") if (attempt / "attempt.json").exists() else {}
        result["bench"][condition] = {**ending(attempt), "load_seconds": record.get("load_seconds"),
                                      "warmup": record.get("warmup"), "levels": levels,
                                      "attempts": [ending(a) for a in attempts_of(run / "bench" / condition)]}
        if not service:
            continue
        low, high = min(service), max(service)
        offline = {r["task_id"]: {**r, "correct": r["status"] == "correct"}
                   for r in evals.get("test", {}).get(condition, []) if r["task_id"] in set(ids)}
        part: dict[str, Any] = {}
        if low != high:  # the noise baseline needs two complete levels
            part["service_low_vs_high"] = {"levels": [low, high], **agreement(service[low], service[high], ids)}
        if offline:
            part["evaluation_vs_service_low"] = agreement(offline, service[low], ids)
            if low != high:
                part["evaluation_vs_service_high"] = agreement(offline, service[high], ids)
            lengths = first_prompt_tokens(list(offline.values()), tokenizer)
            served = {i: (r.get("calls") or [{}])[0].get("prompt_tokens") for i, r in service[low].items()}
            compared = [i for i in ids if i in lengths and served.get(i) is not None]
            part["first_prompt_tokens"] = {"tasks": len(compared),
                                           "identical": sum(lengths[i] == served[i] for i in compared),
                                           "tokenizer": tokenizer is not None,
                                           "differences": {i: [lengths[i], served[i]] for i in compared
                                                           if lengths[i] != served[i]}}
        result["alignment"][condition] = part
    out = run / "report"
    out.mkdir(exist_ok=True)
    atomic_json(out / "report.json", result)
    (out / "report.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


def markdown(result: dict[str, Any]) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    num = lambda x, f="{:.2f}": "—" if x is None else f.format(x)
    lines = [f"# 部署研究报告（{result['model']}@{result['adapter']['step']}，run {result['run'][:12]}）", ""]
    if result["smoke_limit"]:
        lines += [f"**冒烟子集（每个角色 {result['smoke_limit']} 题），数字只用于工程检查，不是结果。**", ""]
    lines += ["描述性研究，不做判定。差值单位为百分点，区间为数据库 bootstrap 95%（只有一个模型，不重采 seed）。", ""]
    merge = result["merge"]
    if "missing" not in merge:
        lines += [f"- 合并：fp32 合并后探针 logit 最大差 {merge['fp32_merge_max_logit_delta']:.2e}，取整到 bf16 后最大差 "
                  f"{merge['bf16_rounding_max_logit_delta']:.2e}", ""]
    lines += ["## 准确率（贪心，项目判分）", "", "| 角色 | " + " | ".join(CONDITIONS) + " | 合并 | FP8 | 合并+FP8 |",
              "|---|" + "---|" * (len(CONDITIONS) + 3)]
    for role, acc in result["accuracy"].items():
        cells = ["—" if c not in acc else f"未完成（{acc[c]['episodes']}/{acc[c]['tasks']} 题）" if acc[c].get("incomplete")
                 else f"{acc[c]['correct']}/{acc[c]['tasks']}（{pct(acc[c]['rate'])}）" for c in CONDITIONS]
        deltas = result["deltas"].get(role, {})
        for key in ("merged-lora", "fp8-merged", "fp8-lora"):
            d = deltas.get(key)
            cells.append("—" if not d else f"{100 * d['delta']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]")
        lines.append(f"| {role} | " + " | ".join(cells) + " |")
    lines += ["", "## 压测（分析服务 /api/analyze，完整 Agent 对话）", "",
              "延迟和题/秒只算服务正常答完的请求；“全部 P95”算上基础设施失败的请求；服务端答对的分母是全部请求。",
              "前缀缓存在每档开始前、vLLM 显示没有在跑和排队的请求后重置（vLLM 不确认重置是否生效）。", "",
              "| 部署 | 并发 | 请求 | 基础设施失败 | P50 秒 | P95 秒 | 全部 P95 秒 | 题/秒 | 生成 token/秒 | 前缀命中 | KV 峰值 | "
              "显存峰值 MiB | 服务端答对 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for condition, b in result["bench"].items():
        for c, x in b["levels"].items():
            peaks = x["peaks"] or {}
            kv = peaks.get("vllm:kv_cache_usage_perc", peaks.get("vllm:gpu_cache_usage_perc"))
            cut = "" if x["stopped"] == "done" else "（截止前没跑完）" if x["stopped"] == "deadline" else "（失败过多，中止）"
            lines.append(f"| {condition} | {c}{cut} | {x['requests']} | {x['failures']} | {num(x['p50_seconds'])} | "
                         f"{num(x['p95_seconds'])} | {num(x['p95_all_seconds'])} | {num(x['tasks_per_second'])} | "
                         f"{num(x['generated_tokens_per_second'], '{:.0f}')} | {pct(x['prefix_cache_hit_rate'])} | "
                         f"{pct(kv)} | {peaks.get('gpu_memory_mib', '—')} | {pct(x['accuracy'])} |")
        for a in b["attempts"]:  # every attempt that did not complete, including the one shown
            if not a["complete"]:
                why = {"killed": "没有结束记录（进程被强杀），只有逐请求日志", "stopped": "到截止时间"}.get(
                    a["status"], a["failure"] or a["status"])
                lines.append(f"| {condition} | 第 {a['attempt']} 次尝试未完成：{why} | | | | | | | | | | | |")
    lines += ["", "## 对齐（压测的固定题）", "",
              "最终 SQL 逐字比较；两边都没有最终 SQL 的题单独计数，不算相同；服务端基础设施失败的题不参加比较。", ""]
    for condition, part in result["alignment"].items():
        a = part.get("service_low_vs_high")
        if a is None:
            lines.append(f"- {condition}：只有一个完整的并发档，没有跨并发比较")
        else:
            lines.append(f"- {condition}：服务端并发 {a['levels'][0]} 对 {a['levels'][1]}（数值噪声基线）最终 SQL 相同 "
                         f"{a['same_final_sql']}/{a['tasks']}，都没有 SQL {a['both_without_sql']}，对错相同 "
                         f"{a['same_correctness']}/{a['tasks']}")
        for key, label in (("evaluation_vs_service_low", "低并发"), ("evaluation_vs_service_high", "高并发")):
            if key in part:
                e = part[key]
                lines.append(f"  - 评测路径对服务端{label}：最终 SQL 相同 {e['same_final_sql']}/{e['tasks']}，都没有 SQL "
                             f"{e['both_without_sql']}，对错相同 {e['same_correctness']}/{e['tasks']}，答对 "
                             f"{e['left_correct']} 对 {e['right_correct']}")
        if "first_prompt_tokens" in part:
            f = part["first_prompt_tokens"]
            lines.append(f"  - 首轮 prompt token 数逐题相同 {f['identical']}/{f['tasks']}"
                         + ("" if f["tokenizer"] else "（本机没有分词器，未比较）"))
    return "\n".join(lines) + "\n"


def progress_entry(run: Path, stage: str, commit: str) -> str:
    """One entry in the project's progress format, from measured artifacts only."""
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    facts: list[str] = []
    try:
        m, _ = load(run, check_code=False)
        facts.append(f"run {m['identity'][:12]}（{'冒烟' if m['smoke_limit'] else '正式'}），题目 {m['counts']}，"
                     f"{MODEL}@{m['imports']['v11']['adapters'][MODEL]['step']}")
    except (OSError, ValueError, KeyError) as exc:
        facts.append(f"manifest 不可读：{type(exc).__name__}")
    rep = run / "report" / "report.json"
    if rep.exists():
        r = read_json(rep)
        for role, deltas in r["deltas"].items():
            for key, d in deltas.items():
                facts.append(f"{role} {key}：{100 * d['delta']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]")
        for condition, b in r["bench"].items():
            top = b["levels"].get(str(max(map(int, b["levels"])))) if b["levels"] else None
            if top and top["served"]:
                facts.append(f"压测 {condition} 并发 {top['concurrency']}：{top['tasks_per_second']:.2f} 题/秒，"
                             f"P95 {top['p95_seconds']:.1f} 秒，基础设施失败 {top['failures']}")
            elif top:
                facts.append(f"压测 {condition} 并发 {top['concurrency']}：{top['requests']} 个请求都没有正常答完")
            for a in b["attempts"]:
                if not a["complete"]:
                    facts.append(f"压测 {condition} 第 {a['attempt']} 次尝试未完成（{a['status']}）：{a['failure'] or '无失败原因'}")
    return (f"\n### {now} · GPU 机（部署研究自动记录） · 源码包 {commit}\n"
            f"- 改动：无代码改动（阶段 `{stage}` 完成）。\n"
            f"- 验证：{'；'.join(facts) or '无产物'}。\n"
            f"- 未完成 / 下一步：见 STATUS.txt 与 docs/H20_DEPLOY_RUNBOOK.md。\n")
