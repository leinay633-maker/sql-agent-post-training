"""Freeze a v0.6 run from the v0.5 task registration (re-verified on this host), plus column notes and gold cache."""
from __future__ import annotations

import csv
import dataclasses
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core import SCHEMA_SQL, Limits, Task, audit, digest, execute, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from . import MODEL_ID, MODEL_REVISION, PROTOCOL
from .config import validate_settings

ROLES = ("pool", "dev", "validation", "test")
GOLD_ROLES = ("validation", "test")


def column_notes(database: Path) -> dict[str, list[list[str]]]:
    """BIRD database_description/<table>.csv, shipped beside every BIRD database. Public metadata, not gold."""
    folder = database.parent / "database_description"
    out: dict[str, list[list[str]]] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.csv")):
        raw = path.read_bytes()
        text = raw.decode("latin-1")
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        rows = []
        for r in csv.DictReader(io.StringIO(text)):
            r = {(k or "").strip().lower().lstrip("\ufeff"): " ".join(v.split()) for k, v in r.items()
                 if k is not None and isinstance(v, str)}
            col, name = r.get("original_column_name", ""), r.get("column_name", "")
            desc, values = r.get("column_description", ""), r.get("value_description", "")
            if col and (values or (desc and desc.lower() not in (col.lower(), name.lower()))
                        or (name and name.lower() != col.lower())):
                rows.append([col, name, desc, values])
        if rows:
            out[path.stem] = rows
    return out


def render(notes: dict[str, list[list[str]]], desc_chars: int, value_chars: int) -> dict[str, list[str]]:
    rendered = {}
    for table, rows in notes.items():
        lines = []
        for col, name, desc, values in rows:
            line = col + (f" ({name})" if name and name.lower() != col.lower() else "")
            if desc and desc_chars and desc.lower() not in (col.lower(), name.lower()):
                line += f": {desc[:desc_chars]}"
            if values and value_chars:
                line += f" | values: {values[:value_chars]}"
            if line != col:
                lines.append(line)
        if lines:
            rendered[table] = lines
    return rendered


TIERS = ((200, 300), (100, 0), (50, 0))


def choose_notes(database: Path, limits: Limits, tokenizer: Any, budget: int, max_steps: int) -> dict[str, Any]:
    """Largest tier whose first prompt (empty question) fits `budget` tokens; questions add < 600 tokens."""
    from ..pipeline import messages_for
    from types import SimpleNamespace

    schema = execute(database, SCHEMA_SQL, limits)
    if schema["status"] != "ok":
        return {"tier": None, "notes": None, "reason": f"schema {schema['status']}"}
    raw = column_notes(database)
    base = messages_for(SimpleNamespace(question="", evidence=""), schema, max_steps, True, review_success=True)
    base_tokens = len(tokenizer.apply_chat_template(base, tokenize=True, add_generation_prompt=True))
    if not raw:
        return {"tier": None, "notes": None, "reason": "no database_description", "base_tokens": base_tokens}
    for tier, (d, v) in enumerate(TIERS):
        notes = render(raw, d, v)
        msgs = messages_for(SimpleNamespace(question="", evidence=""), schema, max_steps, True, review_success=True,
                            extra={"column_notes": notes})
        tokens = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if tokens <= budget:
            return {"tier": tier, "notes": notes, "tokens": tokens, "base_tokens": base_tokens}
    return {"tier": None, "notes": None, "reason": "over budget at every tier", "base_tokens": base_tokens}


def gold_cache(tasks: list[Task], root: Path, limits: Limits, workers: int = 12) -> dict[str, Any]:
    """Execute every reference SQL once, off the evaluation hot path, retrying timeouts (v0.5 dev defect)."""
    databases = {t.db_id: t.database(root) for t in tasks}

    def one(task: Task) -> tuple[str, dict[str, Any], list[list[Any]]]:
        attempts = []
        for _ in range(3):
            started = time.monotonic()
            result = execute(databases[task.db_id], task.gold_sql, limits)
            attempts.append([result["status"], round(time.monotonic() - started, 3)])
            if result["status"] == "ok":
                break
        return task.id, result, attempts

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, tasks))
    from collections import Counter
    return {"tasks": len(tasks), "statuses": dict(Counter(r["status"] for _, r, _ in results)),
            "retried": sum(len(a) > 1 for _, _, a in results), "attempts": {tid: a for tid, _, a in results},
            "results": {tid: r for tid, r, _ in results if r["status"] == "ok"}}


def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, model_path: Path,
           smoke_limit: int = 0) -> dict[str, Any]:
    from ..models import load_tokenizer, model_identity

    validate_settings(settings)
    registration, raw_root = registration.resolve(strict=True), raw_root.resolve(strict=True)
    source = read_json(registration / "manifest.json")
    source_identity = source.pop("identity")
    if fingerprint(source) != source_identity:
        raise ValueError("Registration manifest identity mismatch")
    for role in ROLES:
        if digest(registration / f"{role}.jsonl") != source["sha256"][role]:
            raise ValueError(f"Registered {role} task file changed")
    roles = {role: [Task(**row) for row in read_jsonl(registration / f"{role}.jsonl")] for role in ROLES}
    if smoke_limit:  # engineering smoke only; the manifest says so and the numbers are never results
        import hashlib
        roles = {r: sorted(ts, key=lambda t: hashlib.sha256(f"smoke:{t.id}".encode()).digest())[:smoke_limit]
                 for r, ts in roles.items()}
    audit([t for role in ROLES for t in roles[role]], raw_root)  # database hashes re-verified on this host
    limits = Limits(**settings["limits"])
    spec = {"backend": "transformers", "id": MODEL_ID, "revision": MODEL_REVISION, "device": "cuda",
            "quantization": "none", "local_path": str(model_path.resolve(strict=True)), "allow_download": False,
            "max_context_tokens": settings["max_context"], "max_tokens": settings["max_tokens"],
            "temperature": 0.0, "seed": 42, "batch_size": 16, "endpoint": "", "allow_remote": False}
    tokenizer = load_tokenizer(spec)
    out.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        write_jsonl(out / f"{role}.jsonl", [dataclasses.asdict(t) for t in roles[role]])
    databases = {t.db_id: t.database(raw_root) for role in ROLES for t in roles[role]}
    notes = {db: choose_notes(path, limits, tokenizer, settings["note_budget"], settings["max_steps"])
             for db, path in sorted(databases.items())}
    atomic_json(out / "column_notes.json", notes)
    (out / "gold").mkdir()
    gold_meta = {}
    for role in GOLD_ROLES:
        cache = gold_cache(roles[role], raw_root, limits)
        atomic_json(out / "gold" / f"{role}.json", cache)
        gold_meta[role] = {"statuses": cache["statuses"], "retried": cache["retried"],
                           "sha256": digest(out / "gold" / f"{role}.json")}
    from collections import Counter
    frozen = {"protocol": PROTOCOL, "settings": settings, "model": spec, "model_identity": model_identity(spec),
              "raw_root": str(raw_root),
              "registration": {"source_protocol": source["protocol"], "source_identity": source_identity,
                               "source_counts": source["counts"], "source_test_screen": source.get("test_screen"),
                               "dev_databases": source.get("dev_databases")},
              "counts": {role: len(roles[role]) for role in ROLES},
              "databases": {role: len({t.db_id for t in roles[role]}) for role in ROLES},
              "sha256": {role: digest(out / f"{role}.jsonl") for role in ROLES},
              "column_notes": {"sha256": digest(out / "column_notes.json"),
                               "tiers": dict(Counter(str(v["tier"]) for v in notes.values()))},
              "gold": gold_meta, "smoke_limit": smoke_limit, "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load(run: Path, role: str | None = None) -> tuple[dict[str, Any], list[Task]]:
    """exec_feedback.rft.study.load_run plus v0.6 artifact checks."""
    from ..rft.study import load_run

    frozen, tasks = load_run(run, role)
    if frozen["protocol"] != PROTOCOL:
        raise ValueError("Not a v0.6 run")
    if digest(run / "column_notes.json") != frozen["column_notes"]["sha256"]:
        raise ValueError("Column notes changed")
    return frozen, tasks


def notes_lookup(run: Path) -> Any:
    notes = read_json(run / "column_notes.json")

    def extra_for(task: Task) -> dict[str, Any] | None:
        entry = notes.get(task.db_id) or {}
        return {"column_notes": entry["notes"]} if entry.get("notes") else None
    return extra_for


def gold_for(run: Path, frozen: dict[str, Any], role: str) -> dict[str, Any]:
    path = run / "gold" / f"{role}.json"
    if role not in frozen["gold"] or digest(path) != frozen["gold"][role]["sha256"]:
        raise ValueError(f"Gold cache for {role} missing or changed")
    return read_json(path)["results"]
