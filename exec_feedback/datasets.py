"""Local BIRD ingestion, database-group splits and exhaustive gold preflight."""
from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .core import Limits, Task, audit, digest, execute, read_jsonl, write_jsonl
from .storage import append_record, atomic_json, exclusive_lock, fingerprint, read_json, recover_records

_TOKEN = re.compile(r"--[^\n]*(?:\n|$)|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|[A-Za-z_][A-Za-z_0-9$]*|\s+|.", re.S)


def sql_tokens(sql: str) -> list[str]:
    return [m.group() for m in _TOKEN.finditer(sql)
            if not m.group().isspace() and not m.group().startswith(("--", "/*"))]


def ordered_sql(sql: str) -> bool:
    tokens = sql_tokens(sql)
    depth = 0
    for index, token in enumerate(tokens):
        if token == "(": depth += 1
        elif token == ")": depth -= 1
        elif depth == 0 and token.upper() == "ORDER" and index + 1 < len(tokens) and tokens[index + 1].upper() == "BY":
            return True
    return False


def dataset_id(tasks: list[Task]) -> str:
    return fingerprint([dataclasses.asdict(t) for t in tasks])


def choose_tasks(tasks: list[Task], limit: int, seed: int, *, stratified: bool = False) -> list[Task]:
    if type(limit) is not int or limit < 0:
        raise ValueError("Task limit must be nonnegative (0 = all)")
    ordered = sorted(tasks, key=lambda t: hashlib.sha256(f"{seed}:{t.id}".encode()).digest())
    if stratified and limit and limit < len(tasks):
        from collections import defaultdict, deque
        groups = defaultdict(list)
        for task in ordered:
            groups[task.db_id].append(task)
        queues = {}
        for db, rows in groups.items():
            levels = defaultdict(deque)
            for task in rows: levels[task.difficulty].append(task)
            balanced = []
            while any(levels.values()):
                for level in sorted(levels):
                    if levels[level]: balanced.append(levels[level].popleft())
            queues[db] = deque(balanced)
        picked = []
        db_order = sorted(queues, key=lambda db: hashlib.sha256(f"{seed}:{db}".encode()).digest())
        while len(picked) < limit:
            for db in db_order:
                if queues[db] and len(picked) < limit: picked.append(queues[db].popleft())
        ordered = picked
    return sorted(ordered[:limit] if limit else ordered, key=lambda t: t.id)


def _database_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for file in sorted(root.rglob("*.sqlite")):
        path = file.resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Database symlink escapes the raw data root")
        db_id = path.stem
        if db_id in index and digest(index[db_id]) != digest(path):
            raise ValueError(f"Different snapshots share database id {db_id}")
        index[db_id] = path
    if not index:
        raise ValueError(f"No .sqlite databases under {root}; unpack database archives first")
    return index


def discover_bird_json(root: Path, kind: str) -> Path:
    names = ("train.json", "train_filter.json", "train_filtered.json") if kind == "train" else ("mini_dev_sqlite.json", "dev.json")
    found = [p for name in names for p in root.rglob(name)]
    if len(found) != 1:
        raise ValueError(f"Expected one {kind} JSON under {root}; found {len(found)}. Set the explicit JSON path in config.")
    return found[0]


def prepare_data(settings: dict[str, Any], out: Path) -> dict[str, Any]:
    """Only train data is split; supplied holdout is never sampled for training."""
    if not settings.get("terms_accepted", False):
        raise ValueError("Set data.terms_accepted=true after reviewing upstream data terms")
    root = Path(settings["raw_root"]).expanduser().resolve(strict=True)
    train_path = Path(settings["train_json"]).expanduser().resolve(strict=True) if settings.get("train_json") else discover_bird_json(root, "train")
    holdout_path = Path(settings["holdout_json"]).expanduser().resolve(strict=True) if settings.get("holdout_json") else None
    if holdout_path and holdout_path == train_path:
        raise ValueError("Training and holdout inputs are the same file")
    ratio, seed = settings.get("validation_fraction", 0.2), settings.get("seed", 42)
    if not 0 < ratio < 1 or type(seed) is not int:
        raise ValueError("Invalid database split ratio/seed")
    source_state = {"settings": settings, "train_hash": digest(train_path),
                    "holdout_hash": digest(holdout_path) if holdout_path else None}
    key = fingerprint(source_state)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(out / ".prepare.lock"):
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            saved = read_json(manifest_path)
            if saved["input_fingerprint"] != key:
                raise ValueError("Data/config changed; use a new prepared output directory")
            tasks = load_prepared(out, root)
            return saved
        if (out / "tasks.jsonl").exists():
            raise ValueError("Incomplete data preparation found; choose a new output directory")
        databases = _database_index(root)
        raw_train = read_json(train_path)
        raw_test = read_json(holdout_path) if holdout_path else []
        if not isinstance(raw_train, list) or not raw_train or not isinstance(raw_test, list):
            raise ValueError("BIRD source must be an array; training data must not be empty")
        train_dbs = {r["db_id"] for r in raw_train}
        test_dbs = {r["db_id"] for r in raw_test}
        if train_dbs & test_dbs:
            raise ValueError("Training/holdout database overlap")
        if len(train_dbs) < 2:
            raise ValueError("At least two training databases are required for database-held-out validation")
        groups = sorted(train_dbs, key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).digest())
        validation_dbs = set(groups[:max(1, min(len(groups) - 1, round(len(groups) * ratio)))])
        tasks: list[Task] = []
        for source, rows, path in (("bird-train", raw_train, train_path), ("bird-heldout", raw_test, holdout_path)):
            for i, row in enumerate(rows):
                db_id = row["db_id"]
                if not isinstance(db_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", db_id):
                    raise ValueError("Invalid BIRD database id")
                if db_id not in databases:
                    raise ValueError(f"Missing database: {db_id}")
                sql = row.get("SQL", row.get("sql", row.get("query")))
                if not isinstance(sql, str):
                    raise ValueError(f"Missing SQL in {source} record {i}")
                tokens = sql_tokens(sql)
                if not tokens or tokens[0].upper() not in {"SELECT", "WITH"}:
                    raise ValueError(f"Non-SELECT source record {source}:{i}; curate a separate SELECT-only file")
                split = "test" if source == "bird-heldout" else "validation" if db_id in validation_dbs else "train"
                task = Task(id=f"{source}:{row.get('question_id', i)}", db_id=db_id, split=split,
                            question=row["question"], gold_sql=sql,
                            db_file=databases[db_id].relative_to(root).as_posix(), db_sha256=digest(databases[db_id]),
                            source=source, source_revision="sha256:" + source_state["holdout_hash" if source == "bird-heldout" else "train_hash"], evidence=row.get("evidence") or "",
                            difficulty=row.get("difficulty") or "unknown", ordered=ordered_sql(sql),
                            fixture=bool(settings.get("engineering_fixture", False)))
                tasks.append(task)
        audit_report = audit(tasks, root)
        # Whitespace/comments are ignored, but quoted literal values retain case.
        seen: dict[str, str] = {}
        for task in tasks:
            key_qa = fingerprint([" ".join(task.question.split()).casefold(), sql_tokens(task.gold_sql)])
            previous = seen.setdefault(key_qa, task.split)
            if previous != task.split:
                raise ValueError("Normalized question+SQL crosses splits")
        write_jsonl(out / "tasks.jsonl", [dataclasses.asdict(t) for t in tasks])
        for split in ("train", "validation", "test"):
            write_jsonl(out / f"{split}.jsonl", [dataclasses.asdict(t) for t in tasks if t.split == split])
        manifest = {"format": "efa-dataset-v2", "input_fingerprint": key, "raw_root": str(root),
                    "task_file_sha256": digest(out / "tasks.jsonl"), "dataset_id": dataset_id(tasks),
                    "source_state": source_state, "validation_databases": sorted(validation_dbs),
                    "order_policy": "explicit top-level ORDER BY; review task semantics before final test",
                    "audit": audit_report}
        atomic_json(manifest_path, manifest)
        return manifest


def load_prepared(prepared: Path, root: Path | None = None) -> list[Task]:
    saved = read_json(prepared / "manifest.json")
    if digest(prepared / "tasks.jsonl") != saved["task_file_sha256"]:
        raise ValueError("Prepared task file changed")
    tasks = [Task(**r) for r in read_jsonl(prepared / "tasks.jsonl")]
    if dataset_id(tasks) != saved["dataset_id"]:
        raise ValueError("Dataset identity mismatch")
    audit(tasks, root or Path(saved["raw_root"]))
    return tasks


def validate_gold(tasks: list[Task], root: Path, out: Path, limits: Limits = Limits(), workers: int = 2,
                  *, lock_timeout: float = 0.0) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("CPU workers must be 1..8")
    audit(tasks, root)
    out.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"tasks": dataset_id(tasks), "limits": dataclasses.asdict(limits)})
    with exclusive_lock(out / ".lock", timeout=lock_timeout):
        meta_path = out / "manifest.json"
        if meta_path.exists() and read_json(meta_path)["signature"] != signature:
            raise ValueError("Gold preflight inputs changed")
        atomic_json(meta_path, {"signature": signature, "limits": dataclasses.asdict(limits), "total": len(tasks)})
        records = recover_records(out / "gold.jsonl")
        seen = {r["task_id"] for r in records}
        if len(seen) != len(records) or not seen <= {t.id for t in tasks}:
            raise ValueError("Duplicate or unknown gold-preflight record")
        def check(task: Task) -> dict[str, Any]:
            answer = execute(task.database(root), task.gold_sql, limits)
            return {"task_id": task.id, "db_id": task.db_id, "split": task.split,
                    "status": answer["status"], "reason": answer.get("error"),
                    "rows": len(answer.get("rows", [])), "ordered": task.ordered,
                    "result_sha256": fingerprint(answer) if answer["status"] == "ok" else None}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for item in pool.map(check, [t for t in tasks if t.id not in seen]):
                append_record(out / "gold.jsonl", item)
                records.append(item)
        summary = {"tasks": len(tasks), "valid": sum(r["status"] == "ok" for r in records),
                   "statuses": dict(Counter(r["status"] for r in records)),
                   "invalid_ids": [r["task_id"] for r in records if r["status"] != "ok"],
                   "all_rows_retained": True, "signature": signature}
        atomic_json(out / "summary.json", summary)
        return summary


_INFRASTRUCTURE = ("database is locked", "disk i/o error", "unable to open database")


def _infrastructure(status: str, reason: str | None) -> bool:
    return status == "worker_error" or (status == "sql_error" and any(s in (reason or "").lower() for s in _INFRASTRUCTURE))


def screen_references(tasks: list[Task], root: Path, out: Path, limits: Limits = Limits(), workers: int = 2) -> dict[str, Any]:
    """Pre-selection screen: exclude tasks whose reference SQL itself fails.

    Timeouts and machine failures are retried once serially, because parallel
    load can push a slow but valid reference over the wall-clock limit. A
    machine failure that persists stops the screen instead of dropping the task.
    """
    summary = validate_gold(tasks, root, out, limits, workers)
    by_id = {t.id: t for t in tasks}
    final = {r["task_id"]: (r["status"], r["reason"]) for r in recover_records(out / "gold.jsonl")}
    retried = {}
    for task_id in sorted(summary["invalid_ids"]):
        status, reason = final[task_id]
        if status == "timeout" or _infrastructure(status, reason):
            answer = execute(by_id[task_id].database(root), by_id[task_id].gold_sql, limits)
            final[task_id] = (answer["status"], answer.get("error"))
            retried[task_id] = answer["status"]
    broken = sorted(i for i, (s, r) in final.items() if _infrastructure(s, r))
    if broken:
        raise RuntimeError(f"Reference screen hit machine failures, not task failures: {broken[:10]}")
    excluded = sorted(i for i, (s, _) in final.items() if s != "ok")
    result = {"candidates": len(tasks), "excluded": excluded,
              "statuses": dict(Counter(final[i][0] for i in excluded)),
              "reasons": {i: {"status": final[i][0], "reason": final[i][1]} for i in excluded},
              "retried_serially": retried}
    atomic_json(out / "excluded.json", result)
    return result
