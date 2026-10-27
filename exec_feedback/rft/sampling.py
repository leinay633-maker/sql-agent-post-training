"""Round-k trajectory sampling on registered pool tasks, scored by a train-only oracle."""
from __future__ import annotations

import importlib.metadata
import threading
import time
from pathlib import Path
from typing import Any

from ..core import Limits, Task, compare, execute
from ..storage import append_record, atomic_json, exclusive_lock, fingerprint, read_json, recover_records
from .study import load_run


class Oracle:
    """Train gold stays here; it never reaches messages, prompts or tool results."""

    def __init__(self, tasks: list[Task], root: Path, limits: Limits) -> None:
        if not tasks or any(t.split != "train" or t.source != "bird-train" or t.fixture for t in tasks):
            raise ValueError("The RFT oracle accepts audited BIRD train tasks only")
        self.root, self.limits = root, limits
        self.tasks = {t.id: t for t in tasks}
        self._gold: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def gold(self, task: Task) -> dict[str, Any] | None:
        """None if a screened reference fails twice under load; that sample is unusable, not dropped."""
        with self._lock:
            if task.id in self._gold:
                return self._gold[task.id]
        answer = execute(task.database(self.root), task.gold_sql, self.limits)
        if answer["status"] != "ok":
            answer = execute(task.database(self.root), task.gold_sql, self.limits)
        cached = answer if answer["status"] == "ok" else None
        with self._lock:
            self._gold[task.id] = cached
        return cached

    def score(self, task: Task, record: dict[str, Any]) -> dict[str, Any]:
        """Grade the final answer and every reviewed-then-revised finish (for loss masks)."""
        if self.tasks.get(task.id) is not task:
            raise ValueError("Unregistered task")
        gold = self.gold(task)
        prediction = record.pop("result", None)
        verdict = {"status": record["status"], "correct": False}
        if gold is None:
            return {"verdict": {"status": "oracle_error", "correct": False}, "finish_checks": []}
        if record["status"] == "submitted" and record.get("final_sql") and prediction is not None:
            verdict = compare(prediction, gold, task.ordered)
        checks = []
        steps = record["steps"]
        for index, step in enumerate(steps[:-1]):
            action, observation = step.get("action") or {}, step.get("observation")
            if (action.get("action") == "finish" and isinstance(observation, dict)
                    and observation.get("status") == "ok" and "review" in observation):
                answer = execute(task.database(self.root), action["sql"], self.limits)
                checks.append({"step": index, "correct": compare(answer, gold, task.ordered)["correct"]})
        return {"verdict": verdict, "finish_checks": checks}


def sample_items(pool: list[Task], samples: int) -> list[tuple[str, Task]]:
    """k-major: every task gets sample 0 before any task gets sample 1."""
    return [(f"{task.id}#{k}", task) for k in range(samples) for task in pool]


def sample_shard(run: Path, *, shard: int, shards: int, round_: int,
                 adapter: Path | None = None) -> dict[str, Any]:
    import torch
    from ..models import load_model, model_identity
    from .runner import Generator, run_dialogues

    frozen, pool = load_run(run, "pool")
    settings = frozen["settings"]
    if not 0 <= shard < shards <= 64 or round_ < 1:
        raise ValueError("Invalid shard/round")
    if (round_ == 1) != (adapter is None):
        raise ValueError("Round 1 samples the base model; later rounds sample a frozen v0.5 adapter")
    out = run / f"round{round_}" / "samples"
    spec = frozen["model"]
    policy = {"base": frozen["model_identity"],
              "adapter_hash": model_identity(spec, adapter)["adapter_hash"] if adapter else None}
    identity = fingerprint({"run": frozen["identity"], "round": round_, "shards": shards, "policy": policy})
    journal = out / f"shard{shard:02d}.jsonl"
    with exclusive_lock(out / f".shard{shard:02d}.lock"):
        status_path = out / f"shard{shard:02d}.status.json"
        if status_path.exists() and read_json(status_path)["identity"] != identity:
            raise ValueError("Sampling policy/run changed; use a new round directory")
        done = {r["key"] for r in recover_records(journal)}
        items = [item for i, item in enumerate(sample_items(pool, settings["samples_per_task"]))
                 if i % shards == shard and item[0] not in done]
        root, limits = Path(frozen["raw_root"]), Limits(**settings["limits"])
        oracle = Oracle(pool, root, limits)
        torch.manual_seed(round_ * 1000 + shard)
        model, tokenizer = load_model(spec, adapter)
        model.eval()
        generator = Generator(model, tokenizer, batch_size=settings["sample_batch"],
                              max_context=settings["max_context"], max_tokens=settings["max_tokens"],
                              sampling={k: settings[k] for k in ("temperature", "top_p", "top_k")},
                              seed=round_ * 1_000_003 + shard * 10_007)
        started = time.monotonic()
        prior = read_json(status_path)["active_seconds"] if status_path.exists() else 0.0
        deadline = started + max(0.0, settings["sample_seconds"] - prior)
        counts = {"completed": len(done), "correct": sum(r["correct"] for r in recover_records(journal))}

        def finish(key: str, task: Task, record: dict[str, Any]) -> dict[str, Any]:
            scored = oracle.score(task, record)
            return {**record, "key": key, "task_id": task.id, "db_id": task.db_id,
                    "k": int(key.rsplit("#", 1)[1]), "round": round_, "shard": shard,
                    "correct": bool(scored["verdict"]["correct"]), "verdict": scored["verdict"]["status"],
                    "finish_checks": scored["finish_checks"]}

        def save_status(final: bool) -> None:
            atomic_json(status_path, {
                "identity": identity, "policy": policy, "final": final, **counts,
                "registered_items": len(sample_items(pool, settings["samples_per_task"])) // shards,
                "active_seconds": prior + time.monotonic() - started, "generator": generator.metadata,
                "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}})

        try:
            for row in run_dialogues(items, root, generator, slots=settings["sample_slots"],
                                     max_steps=settings["max_steps"], review_success=True, limits=limits,
                                     workers=settings["sql_workers"], finish=finish, deadline=deadline):
                append_record(journal, row)
                counts["completed"] += 1
                counts["correct"] += row["correct"]
                if counts["completed"] % 200 == 0:
                    save_status(False)
        finally:
            save_status(True)
        return read_json(status_path)
