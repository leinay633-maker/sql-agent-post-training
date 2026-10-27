"""Freeze v0.5 task pools from a completed v0.3.1 study and an optional BIRD dev final test."""
from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import Any

from ..core import Limits, Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from . import MODEL_ID, MODEL_REVISION, PROTOCOL
from .config import validate_settings

ROLES = ("pool", "dev", "validation", "test")


def rank(salt: str, key: str) -> bytes:
    return hashlib.sha256(f"{salt}:{key}".encode()).digest()


def source_tasks(source: Path, raw_root: Path) -> tuple[dict[str, Any], list[Task]]:
    """Reuse the v0.3.1 split and its recorded pre-selection screen, never rescreen it."""
    from ..datasets import load_prepared

    old = read_json(source / "study.json")
    config = old["config"]
    if old.get("status") != "complete" or config.get("version") != 4 or not config["experiment"].get("gold_screen"):
        raise ValueError("Source must be a completed v0.3.1 study (version 4 with gold_screen)")
    recorded = old["stages"]["prepare"]["artifacts"]
    for name in ("selection.json", "prepared/manifest.json", "prepared/tasks.jsonl",
                 "prepared/gold_screen/excluded.json"):
        if recorded.get(name) != digest(source / name):
            raise ValueError(f"Source artifact changed after the study: {name}")
    excluded = set(read_json(source / "prepared/gold_screen/excluded.json")["excluded"])
    selection = read_json(source / "selection.json")
    if selection["gold_screen_excluded"] != sorted(excluded):
        raise ValueError("Source selection and screen disagree")
    tasks = [t for t in load_prepared(source / "prepared", raw_root) if t.id not in excluded]
    meta = {"source_study": str(source), "source_study_sha256": digest(source / "study.json"),
            "source_name": config["name"], "source_excluded": len(excluded),
            "legacy_validation_ids": selection["evaluation"]}
    return meta, tasks


def bird_dev_tasks(dev_json: Path, raw_root: Path) -> list[Task]:
    """BIRD dev as a held-out final test; database files must live under raw_root."""
    from ..datasets import _database_index, ordered_sql, sql_tokens

    rows = read_json(dev_json)
    if not isinstance(rows, list) or not rows:
        raise ValueError("BIRD dev JSON must be a nonempty array")
    databases = _database_index(dev_json.parent)
    revision = "sha256:" + digest(dev_json)
    tasks = []
    for i, row in enumerate(rows):
        db_id, sql = row["db_id"], row.get("SQL")
        if not isinstance(db_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", db_id) or db_id not in databases:
            raise ValueError(f"Missing or invalid BIRD dev database: {db_id}")
        tokens = sql_tokens(sql) if isinstance(sql, str) else []
        if not tokens or tokens[0].upper() not in {"SELECT", "WITH"}:
            raise ValueError(f"Non-SELECT BIRD dev record {i}")
        path = databases[db_id]
        if not path.is_relative_to(raw_root):
            raise ValueError("BIRD dev databases must be under the audited raw root")
        tasks.append(Task(id=f"bird-dev:{row.get('question_id', i)}", db_id=db_id, split="test",
                          question=row["question"], gold_sql=sql, db_file=path.relative_to(raw_root).as_posix(),
                          db_sha256=digest(path), source="bird-dev", source_revision=revision,
                          evidence=row.get("evidence") or "", difficulty=row.get("difficulty") or "unknown",
                          ordered=ordered_sql(sql)))
    return tasks


def prepare(source: Path, raw_root: Path, out: Path, settings: dict[str, Any], *,
            model_path: Path, dev_json: Path | None = None) -> dict[str, Any]:
    from ..datasets import choose_tasks, screen_references
    from ..models import model_identity

    validate_settings(settings)
    source, raw_root = source.resolve(strict=True), raw_root.resolve(strict=True)
    if settings["phase"] == "study" and dev_json is None:
        raise ValueError("The registered study includes the BIRD dev final test")
    meta, tasks = source_tasks(source, raw_root)
    salt, seed = settings["salt"], int(settings["salt"])
    train = [t for t in tasks if t.split == "train"]
    databases = sorted({t.db_id for t in train}, key=lambda db: rank(salt, db))
    dev_dbs = set(databases[:settings["dev_databases"]])
    if len(dev_dbs) >= len(databases):
        raise ValueError("Dev databases would leave no sampling pool")
    roles = {"pool": [t for t in train if t.db_id not in dev_dbs],
             "dev": [t for t in train if t.db_id in dev_dbs],
             "validation": [t for t in tasks if t.split == "validation"] if settings["phase"] == "study" else [],
             "test": []}
    for role, limit in (("pool", settings["pool_limit"]), ("dev", settings["dev_limit"])):
        if limit:
            roles[role] = choose_tasks(roles[role], limit, seed, stratified=True)
    out.mkdir(parents=True, exist_ok=False)
    screen = None
    if settings["phase"] == "study":
        candidates = bird_dev_tasks(dev_json.resolve(strict=True), raw_root)
        screen = screen_references(candidates, raw_root, out / "test_screen", Limits(**settings["limits"]), workers=8)
        roles["test"] = [t for t in candidates if t.id not in set(screen["excluded"])]
    roles["pool"].sort(key=lambda t: rank(salt, t.id))
    audit([t for role in ROLES for t in roles[role]], raw_root)
    for role in ROLES:
        write_jsonl(out / f"{role}.jsonl", [dataclasses.asdict(t) for t in roles[role]])
    spec = {"backend": "transformers", "id": MODEL_ID, "revision": MODEL_REVISION, "device": "cuda",
            "quantization": settings["quantization"], "local_path": str(model_path.resolve(strict=True)),
            "allow_download": False, "max_context_tokens": settings["max_context"],
            "max_tokens": settings["max_tokens"], "temperature": 0.0, "seed": 42,
            "batch_size": settings["eval_batch"], "endpoint": "", "allow_remote": False}
    frozen = {"protocol": PROTOCOL, "settings": settings, "model": spec,
              "model_identity": model_identity(spec), "raw_root": str(raw_root), **meta,
              "dev_databases": sorted(dev_dbs), "test_screen": None if screen is None else
              {"candidates": screen["candidates"], "excluded": len(screen["excluded"]), "statuses": screen["statuses"]},
              "counts": {role: len(roles[role]) for role in ROLES},
              "databases": {role: len({t.db_id for t in roles[role]}) for role in ROLES},
              "sha256": {role: digest(out / f"{role}.jsonl") for role in ROLES},
              "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load_run(run: Path, role: str | None = None) -> tuple[dict[str, Any], list[Task]]:
    from ..models import model_identity

    frozen = read_json(run / "manifest.json")
    identity = frozen.pop("identity")
    from ..v6 import PROTOCOL as V6_PROTOCOL
    if fingerprint(frozen) != identity or frozen["protocol"] not in (PROTOCOL, V6_PROTOCOL):
        raise ValueError("v0.5 manifest identity mismatch")
    frozen["identity"] = identity
    if frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new v0.5 run")
    if model_identity(frozen["model"]) != frozen["model_identity"]:
        raise ValueError("Base model snapshot changed")
    if role is None:
        return frozen, []
    if role not in ROLES:
        raise ValueError(f"Unknown task role: {role}")
    path = run / f"{role}.jsonl"
    if digest(path) != frozen["sha256"][role]:
        raise ValueError("Registered task file changed")
    tasks = [Task(**row) for row in read_jsonl(path)]
    if tasks:
        audit(tasks, Path(frozen["raw_root"]))
    return frozen, tasks
