"""Paired SFT data: identical targets, audited origins, actual execution feedback."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .core import compare, SCHEMA_SQL, Limits, Task, audit, execute, read_jsonl, write_jsonl
from .datasets import sql_tokens
from .pipeline import build_sft, episode, messages_for
from .storage import atomic_json, fingerprint


class PerturbationReplay:
    """Training augmentation only. Never presented as a model evaluation."""
    kind = "data_augmentation"
    metadata = {"provider": "rule_perturbation", "model": None}
    def __init__(self, bad_sql: str, target_sql: str, review_success: bool = False) -> None:
        self.outputs = iter([{"action": "query", "sql": bad_sql}, {"action": "finish", "sql": target_sql}] + ([{"action": "finish", "sql": target_sql}] if review_success else []))
    def __call__(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.outputs))


def perturbations(sql: str, task_id: str, semantic: bool = False) -> list[tuple[str, str]]:
    tokens = sql_tokens(sql)
    candidates = []
    if semantic:
        swaps = {"SUM": "AVG", "AVG": "SUM", "MIN": "MAX", "MAX": "MIN"}
        for i, token in enumerate(tokens):
            if token.upper() in swaps and i+1 < len(tokens) and tokens[i+1] == "(":
                changed = tokens.copy(); changed[i] = swaps[token.upper()]
                candidates.append(("semantic_aggregate", " ".join(changed)))
            if token.upper() == "DISTINCT":
                changed = tokens.copy(); changed.pop(i)
                candidates.append(("semantic_distinct", " ".join(changed)))
            if token == "=" and i and i+1 < len(tokens):
                changed = tokens.copy(); changed[i] = "!="
                candidates.append(("semantic_filter", " ".join(changed)))
        return sorted(candidates, key=lambda x: hashlib.sha256((task_id + x[0] + x[1]).encode()).digest())
    for i, token in enumerate(tokens):
        if i and tokens[i-1].upper() in {"FROM", "JOIN"} and token not in {"(", ")"}:
            changed = tokens.copy()
            changed[i] = "__efa_missing_table__"
            candidates.append(("missing_table", " ".join(changed)))
        if token.upper() in {"SUM", "AVG", "COUNT", "MIN", "MAX"} and i+1 < len(tokens) and tokens[i+1] == "(":
            changed = tokens.copy()
            changed[i] = "__efa_missing_function__"
            candidates.append(("unknown_function", " ".join(changed)))
        if i and tokens[i-1].upper() in {"WHERE", "BY", "ON"} and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token):
            changed = tokens.copy()
            changed[i] = "__efa_missing_column__"
            candidates.append(("missing_column", " ".join(changed)))
    candidates.append(("syntax_parenthesis", sql.rstrip().rstrip(";") + ")"))
    # Fixed ordering per task; this is not sampling from test results.
    return sorted(candidates, key=lambda x: hashlib.sha256((task_id + x[0] + x[1]).encode()).digest())


def injected_trace(task: Task, root: Path, limits: Limits, max_steps: int, *, semantic: bool = False, review_success: bool = False) -> tuple[dict[str, Any], str]:
    if task.split != "train":
        raise ValueError("Gold-based augmentation is training-only")
    database = task.database(root)
    gold = execute(database, task.gold_sql, limits)
    for kind, bad_sql in perturbations(task.gold_sql, task.id, semantic):
        observed = execute(database, bad_sql, limits)
        if semantic:
            if observed["status"] != "ok" or compare(observed, gold, task.ordered)["status"] != "incorrect":
                continue
        elif observed["status"] != "sql_error":
            continue
        trace = episode(task, root, PerturbationReplay(bad_sql, task.gold_sql, review_success),
                        max_steps=max_steps, feedback=True, limits=limits, review_success=review_success)
        if trace["correct"]:
            trace["origin"] = "rule_perturbation"
            trace["error_type"] = kind
            return trace, kind
    raise ValueError(f"No executable correction pair for {task.id}")


def create_pairs(tasks: list[Task], root: Path, out: Path, *, traces: list[dict[str, Any]] | None = None,
                 mode: str = "mixed", limits: Limits = Limits(), max_steps: int = 4,
                 tokenizer: Any = None, max_length: int = 8192, min_pairs: int = 1,
                 allow_fixture: bool = False, semantic: bool = False, review_success: bool = False) -> dict[str, Any]:
    audit(tasks, root)
    if any(t.split != "train" for t in tasks):
        raise ValueError("Only preselected training tasks may enter the paired corpus")
    if any(t.fixture for t in tasks) and not allow_fixture:
        raise ValueError("Engineering fixtures are not research data")
    if mode not in {"natural", "injected", "mixed"}:
        raise ValueError("Invalid collection mode")
    trace_map = {}
    for trace in traces or []:
        if trace["task_id"] in trace_map:
            raise ValueError("Duplicate collection trace")
        trace_map[trace["task_id"]] = trace
    success_rows, repair_rows, selected_traces, rejected = [], [], [], []
    origins: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    out.mkdir(parents=True, exist_ok=False)
    for task in tasks:
        natural = trace_map.get(task.id)
        candidate = None
        origin, error_type = "natural_model", "observed_execution_error"
        if mode != "injected" and natural:
            try:
                candidate = build_sft([task], root, "repair", [natural], allow_fixture=allow_fixture)[0]
                if not any(step.get("observation", {}).get("status") in {"sql_error", "protocol_error", "timeout", "rejected"} for step in natural["steps"]):
                    error_type = "observed_final_result_mismatch"
            except ValueError:
                candidate = None
        trace = natural
        if candidate is None and mode in {"injected", "mixed"}:
            try:
                try:
                    if not semantic: raise ValueError("Execution-only collection")
                    trace, error_type = injected_trace(task, root, limits, max_steps, semantic=True, review_success=review_success)
                except ValueError:
                    trace, error_type = injected_trace(task, root, limits, max_steps, review_success=review_success)
                candidate = build_sft([task], root, "repair", [trace], allow_fixture=allow_fixture)[0]
                origin = "rule_perturbation"
            except ValueError as exc:
                rejected.append({"task_id": task.id, "reason": str(exc)})
                continue
        if candidate is None:
            rejected.append({"task_id": task.id, "reason": "no verified natural repair"})
            continue
        schema = execute(task.database(root), SCHEMA_SQL, limits)
        if schema["status"] != "ok":
            rejected.append({"task_id": task.id, "reason": "schema_error"})
            continue
        # C and D differ only in context, not in question, final SQL, or target style.
        ordinary = {**candidate, "mode": "success", "prompt": messages_for(task, schema, max_steps, True, review_success=review_success)}
        candidate = {**candidate, "mode": "repair"}
        length_info = {}
        if tokenizer is not None:
            from .train import tokenize_row
            try:
                a = tokenize_row(ordinary, tokenizer, max_length)
                b = tokenize_row(candidate, tokenizer, max_length)
                length_info = {"success_tokens": len(a["input_ids"]), "repair_tokens": len(b["input_ids"]),
                               "supervised_tokens": sum(x != -100 for x in b["labels"])}
            except ValueError as exc:
                rejected.append({"task_id": task.id, "reason": str(exc)})
                continue
        provenance = {"origin": origin, "error_type": error_type,
                      "trajectory_sha256": fingerprint({"messages": trace["messages"], "steps": trace["steps"]}),
                      "lengths": length_info}
        ordinary.update(provenance)
        candidate.update(provenance)
        success_rows.append(ordinary)
        repair_rows.append(candidate)
        selected_traces.append({**trace, "origin": origin, "error_type": error_type})
        origins[origin] += 1
        errors[error_type] += 1
    write_jsonl(out / "success.jsonl", success_rows)
    write_jsonl(out / "repair.jsonl", repair_rows)
    write_jsonl(out / "selected-traces.jsonl", selected_traces)
    write_jsonl(out / "rejected.jsonl", rejected)
    summary = {"eligible_train_tasks": len(tasks), "paired_tasks": len(success_rows),
               "selection_rate": len(success_rows) / len(tasks), "origins": dict(origins),
               "error_types": dict(errors), "rejected": len(rejected),
               "tokenizer_checked": tokenizer is not None, "max_length": max_length,
               "identical_target_check": all(a["completion"] == b["completion"] for a,b in zip(success_rows, repair_rows)),
               "task_ids": [r["task_id"] for r in success_rows],
               "research_ready": len(success_rows) >= min_pairs and tokenizer is not None and not any(t.fixture for t in tasks),
               "minimum_pairs_met": len(success_rows) >= min_pairs,
               "semantic_augmentation": semantic, "review_success": review_success,
               "note": "Rule-perturbed training data is augmentation, not real-model success or natural repair evidence."}
    atomic_json(out / "summary.json", summary)
    return summary
