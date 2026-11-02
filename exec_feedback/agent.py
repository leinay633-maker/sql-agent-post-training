"""Gold-free state machine shared by scalar, batched and service inference."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

from .core import SCHEMA_SQL, Limits, execute
from .pipeline import messages_for, parse_action
from .storage import BudgetExceeded


def preview(answer: dict[str, Any], max_rows: int = 20, cell_chars: int = 256) -> dict[str, Any]:
    if answer["status"] != "ok":
        return answer
    rows, shortened = [], False
    for row in answer["rows"][:max_rows]:
        values = []
        for value in row:
            if isinstance(value, str) and len(value) > cell_chars:
                value, shortened = value[:cell_chars] + "…[preview]", True
            elif isinstance(value, dict):
                value, shortened = {k: str(v)[:cell_chars] for k, v in value.items()}, True
            values.append(value)
        rows.append(values)
    return {**answer, "rows": rows, "row_count": len(answer["rows"]),
            "preview_truncated": shortened or len(answer["rows"]) > max_rows}


def dialogue_stream(database: Path, question: str, evidence: str, *, max_steps: int = 4,
                    feedback: bool = True, limits: Limits = Limits(),
                    review_success: bool = False, extra: dict[str, Any] | None = None, fenced_json: bool = False,
                    ) -> Generator[list[dict[str, str]], Any, dict[str, Any]]:
    """Yield public messages and receive model text/usage or an exception.

    Successful execution can receive one review turn, without using a gold query
    or correctness oracle. Every yielded response consumes one of max_steps.
    """
    if not 1 <= max_steps <= 16 or (not feedback and max_steps != 1):
        raise ValueError("Invalid agent step budget; direct mode uses one step")
    schema = execute(database, SCHEMA_SQL, limits)
    record: dict[str, Any] = {"steps": [], "messages": [], "final_sql": None,
                              "status": "budget_exhausted", "correct": False, "usage": {}}
    if schema["status"] != "ok":
        return {**record, "status": "schema_error", "reason": schema["status"]}
    messages = messages_for(SimpleNamespace(question=question, evidence=evidence), schema,
                            max_steps, feedback, review_success=review_success, extra=extra)
    record["messages"] = messages
    started, reviewed = time.monotonic(), False
    for index in range(max_steps):
        received = yield copy.deepcopy(messages)
        if isinstance(received, BudgetExceeded):
            raise received
        if isinstance(received, Exception):
            record.update(status="provider_error", reason=type(received).__name__)
            break
        raw = received.get("text") if isinstance(received, dict) else received
        if not isinstance(raw, str):
            record.update(status="protocol_error", reason="Provider output is not text")
            break
        messages.append({"role": "assistant", "content": raw[:65536]})
        step: dict[str, Any] = {"output": raw[:65536]}
        if isinstance(received, dict):
            for key, value in received.get("usage", {}).items():
                if type(value) is int:
                    record["usage"][key] = record["usage"].get(key, 0) + value
            # Usage does not enter replay identity (different providers expose different counters).
            if "trace" in received:  # v0.9 online RL: the turn's prompt/sampled token ids and log-probs
                step["trace"] = received["trace"]
        record["steps"].append(step)
        try:
            action = parse_action(raw, fenced_json=fenced_json)
        except (ValueError, TypeError) as exc:
            observation = {"status": "protocol_error", "error": str(exc)[:300]}
            step["observation"] = observation
            messages.append({"role": "user", "content": json.dumps({"tool_result": observation})})
            continue
        step["action"] = action
        if action["action"] == "finish":
            answer = execute(database, action["sql"], limits)
            if feedback and index + 1 < max_steps:
                if answer["status"] != "ok":
                    reviewed = True
                    step["observation"] = answer
                    messages.append({"role": "user", "content": json.dumps({"tool_result": answer}, ensure_ascii=False)})
                    continue
                if review_success and not reviewed:
                    reviewed = True
                    observation = {**preview(answer), "review": "Check this result against the question, filters, joins, aggregation, NULL and duplicates. Submit finish again, correcting SQL if needed. No correctness verdict is available."}
                    step["observation"] = observation
                    messages.append({"role": "user", "content": json.dumps({"tool_result": observation}, ensure_ascii=False)})
                    continue
            record.update(final_sql=action["sql"], result=answer,
                          status="submitted" if answer["status"] == "ok" else "execution_failed")
            break
        if not feedback:
            record.update(status="protocol_error", reason="Tools are disabled in direct baseline")
            break
        observation = schema if action["action"] == "schema" else preview(execute(database, action["sql"], limits))
        if action["action"] == "query":
            reviewed = True
        step["observation"] = observation
        messages.append({"role": "user", "content": json.dumps({"tool_result": observation}, ensure_ascii=False)})
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def dialogue(database: Path, question: str, evidence: str, policy: Any, *,
             max_steps: int = 4, feedback: bool = True, limits: Limits = Limits(),
             review_success: bool = False, fenced_json: bool = False) -> dict[str, Any]:
    engine = dialogue_stream(database, question, evidence, max_steps=max_steps,
                             feedback=feedback, limits=limits, review_success=review_success, fenced_json=fenced_json)
    # a provider that reports each call's own usage (HTTPPolicy.complete) stays right when other dialogues share it;
    # the metadata difference below would count their calls too
    complete = getattr(policy, "complete", None)
    try:
        messages = next(engine)
        while True:
            before = copy.deepcopy(policy.metadata.get("usage", {}))
            try:
                if complete is not None:
                    raw, usage = complete(messages)
                    received = {"text": raw, "usage": usage}
                else:
                    raw = policy(messages)
                    after = policy.metadata.get("usage", {})
                    received = {"text": raw,
                                "usage": {k: v - before.get(k, 0) for k, v in after.items() if type(v) is int}}
            except BudgetExceeded:
                raise
            except Exception as exc:
                received = exc
            messages = engine.send(received)
    except StopIteration as done:
        return done.value
    finally:
        engine.close()
