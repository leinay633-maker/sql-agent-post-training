"""Why v0.10's full-parameter runs fell behind v0.9's LoRA RL: training on the very same tasks (each seed samples the
same tasks at every step in both versions), then the behaviour at evaluation. Reads the unpacked final results of
both versions and prints markdown; stdlib only, no GPU.

  python -I scripts/v10_diagnose.py --run <v10 final>/outputs/v10/run --v09-run <v09 final>/outputs/v09/run
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

SEEDS = (42, 43, 44)
BLOCK = 20


def records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def kind(step: dict[str, Any]) -> str | None:
    action = step.get("action")
    return action.get("action") if isinstance(action, dict) else None  # None: the output did not parse


def shape(steps: list[dict[str, Any]]) -> dict[str, bool]:
    """unparsed: the first output is not a valid action; no_query: submitted before any query (or never queried);
    repeat: the same SQL queried twice in a row. A dialogue rejected for its context length has no steps: all False."""
    kinds = [kind(s) for s in steps]
    before = kinds[:kinds.index("finish")] if "finish" in kinds else kinds
    sqls = [s["action"].get("sql") for s in steps if kind(s) == "query"]
    return {"unparsed": bool(kinds) and kinds[0] is None, "no_query": bool(kinds) and "query" not in before,
            "repeat": any(a == b for a, b in zip(sqls, sqls[1:]))}


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}" if whole else "—"


def run_dir(root: Path, family: str, seed: int) -> Path:
    return root / "train" / f"{family}_seed{seed}"


def training(v10: Path, v09: Path) -> list[str]:
    out = ["## 训练：与 v0.9 RL 同题逐步配对（每 20 步一段）", "",
           "奖励是每步 128 条轨迹的项目判分正确率，差值单位为百分点。行为占比的分母是该段有对话的训练轨迹，"
           "不含因题目过长被拒的轨迹（约 7%，三者题目相同，被拒的也相同）。", ""]
    for seed in SEEDS:
        dirs = {"RL": run_dir(v09, "RL", seed), "RLF": run_dir(v10, "RLF", seed), "RLFa": run_dir(v10, "RLFa", seed)}
        metrics = {f: list(records(d / "metrics.jsonl")) for f, d in dirs.items()}
        same = sum(a["tasks"] == b["tasks"] == c["tasks"] for a, b, c in zip(*metrics.values()))
        out += [f"### seed {seed}（三者同题的步数 {same}/{len(metrics['RL'])}）", "",
                "| 步 | RL 奖励 | RLF−RL | RLFa−RL | 输出 token RL/RLF/RLFa | 轮数 RL/RLF/RLFa "
                "| 首个输出不可解析 % RL/RLF/RLFa | 不查数据就提交 % | 连续重复同一 SQL % |",
                "|---|---|---|---|---|---|---|---|---|"]
        for start in range(0, len(metrics["RL"]), BLOCK):
            rows = {f: m[start:start + BLOCK] for f, m in metrics.items()}
            mean = {f: {k: statistics.fmean(r[k] for r in rs) for k in ("reward_mean", "mean_completion_tokens",
                                                                          "mean_turns")} for f, rs in rows.items()}
            shares = {}
            for f, d in dirs.items():
                count, total = collections.Counter(), 0
                for step in range(start, start + BLOCK):
                    path = d / "rollouts" / f"step-{step:04d}.jsonl"
                    if not path.exists():
                        continue
                    for row in records(path):
                        if not row.get("steps"):
                            continue
                        total += 1
                        count.update(k for k, v in shape(row["steps"]).items() if v)
                shares[f] = (count, total)
            three = lambda key: "/".join(pct(shares[f][0][key], shares[f][1]) for f in dirs)
            r = lambda f: 100 * mean[f]["reward_mean"]
            out.append(f"| {start + 1}–{start + BLOCK} | {r('RL'):.1f} | {r('RLF') - r('RL'):+.1f} | "
                       f"{r('RLFa') - r('RL'):+.1f} | "
                       + "/".join(f"{mean[f]['mean_completion_tokens']:.0f}" for f in dirs) + " | "
                       + "/".join(f"{mean[f]['mean_turns']:.2f}" for f in dirs)
                       + f" | {three('unparsed')} | {three('no_query')} | {three('repeat')} |")
        out.append("")
    return out


def error_kind(text: str) -> str:
    """'no such column: T1.x' -> 'no such column'; keeps the SQLite message without names and numbers."""
    text = re.sub(r"\{'error': ['\"]|['\"], 'status'.*$", "", text)
    return re.split(r"[:(]| near ", text, maxsplit=1)[0].strip() or text[:60]


def evaluation(v10: Path, role: str) -> list[str]:
    conditions = ["B"] + [f"{f}_seed{s}" for f in ("PTR", "RL", "RLF", "RLFa") for s in SEEDS]
    episodes = {c: {e["task_id"]: e for e in records(v10 / "eval" / role / c / "episodes.jsonl")} for c in conditions}
    out = [f"## 评测行为：{role}（{len(episodes['B'])} 题）", "",
           "| 条件 | 项目判分 % | 最终 SQL 执行失败 | 步数用完 | 输出 token 中位数 | 首个输出不可解析 % "
           "| 不查数据就提交 % | 连续重复同一 SQL % | 与同 seed PTR 比：修好/改坏 | 改坏的题里执行失败 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conditions:
        es = list(episodes[c].values())
        status = collections.Counter(e["status"] for e in es)
        shapes = collections.Counter(k for e in es for k, v in shape(e.get("steps") or []).items() if v)
        tokens = statistics.median(e["usage"]["completion_tokens"] for e in es)
        moved = "—"
        broken_failed = "—"
        if "_seed" in c and not c.startswith("PTR"):
            ptr = episodes["PTR_seed" + c.split("_seed")[1]]
            fixed = sum(1 for t, e in episodes[c].items() if e["correct"] and not ptr[t]["correct"])
            broken = [e for t, e in episodes[c].items() if ptr[t]["correct"] and not e["correct"]]
            moved = f"{fixed}/{len(broken)}"
            broken_failed = str(sum(e["status"] == "execution_failed" for e in broken))
        out.append(f"| {c} | {pct(status['correct'], len(es))} | {status['execution_failed']} | "
                   f"{status['budget_exhausted']} | {tokens:.0f} | {pct(shapes['unparsed'], len(es))} | "
                   f"{pct(shapes['no_query'], len(es))} | {pct(shapes['repeat'], len(es))} | {moved} | {broken_failed} |")
    out += ["", f"最终 SQL 执行失败的报错（{role}，按家族合计三个 seed，前 5 类）：", ""]
    for family in ("PTR", "RL", "RLF", "RLFa"):
        errors = collections.Counter(error_kind(str(e.get("answer_preview", "")))
                                     for s in SEEDS for e in episodes[f"{family}_seed{s}"].values()
                                     if e["status"] == "execution_failed")
        listed = "；".join(f"{k} {v}" for k, v in errors.most_common(5)) or "无"
        out.append(f"- {family}（{sum(errors.values())}）：{listed}")
    return out + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="outputs/v10/run of the unpacked final results")
    ap.add_argument("--v09-run", type=Path, required=True, help="outputs/v09/run of v0.9's unpacked final results")
    args = ap.parse_args()
    lines = training(args.run, args.v09_run)
    for role in ("test", "validation"):
        lines += evaluation(args.run, role)
    sys.stdout.reconfigure(encoding="utf-8")  # a Windows console redirect would otherwise encode as GBK
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
