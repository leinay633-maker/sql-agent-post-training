"""Freeze a v0.11 run: the v0.5 task registration (re-verified), the pinned 4B, gold caches for dev/validation/test,
v0.8's PTR adapters (the start points) and v0.9's RL adapters at v0.9's registered milestone (the comparison arm), each
checked against the run that produced it; the training settings checked against v0.9's, item by item."""
from __future__ import annotations

import dataclasses
import hashlib
import shutil
from pathlib import Path
from typing import Any

from ..core import Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from ..v8.freeze import expected_identity, model_spec
from ..v9.freeze import verify_imports as verify_v08
from ..v10.freeze import lora_adapter, model_ready, start_adapter, verify_v09  # noqa: F401 - re-exported
from . import MODELS, PROTOCOL, V08_RUN, V09_RUN
from .config import SAME_AS_V09_RL, validate_settings

ROLES = ("pool", "dev", "validation", "test")
GOLD_ROLES = ("dev", "validation", "test")


def models_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for model_id, revision in MODELS.values():
        if manifest.get(model_id, {}).get("revision") != revision:
            raise ValueError(f"Model manifest does not pin {model_id}@{revision}")
    return manifest


def same_training(imports: Path, settings: dict[str, Any]) -> None:
    """The registered study changes the training tasks only: every other training setting is v0.9's."""
    if settings["phase"] != "study":
        return
    v09 = read_json(imports / "v09" / "manifest.json")["settings"]
    differ = [k for k in SAME_AS_V09_RL if v09.get(k) != settings[k]]
    if differ:
        raise ValueError(f"v0.11 must train exactly as v0.9's RL did; differs in {differ}")


def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, models_root: Path,
           manifest_path: Path, imports: Path, smoke_limit: int = 0, v08_run: str = V08_RUN,
           v09_run: str = V09_RUN) -> dict[str, Any]:
    from ..core import Limits
    from ..v6.freeze import gold_cache

    validate_settings(settings)
    if (settings["phase"] == "smoke") != bool(smoke_limit):
        raise ValueError("Smoke settings go with a smoke subset and only with it")
    registration, raw_root = registration.resolve(strict=True), raw_root.resolve(strict=True)
    imports, models_root = imports.resolve(strict=True), models_root.resolve()
    source = read_json(registration / "manifest.json")
    source_identity = source.pop("identity")
    if fingerprint(source) != source_identity:
        raise ValueError("Registration manifest identity mismatch")
    for role in ROLES:
        if digest(registration / f"{role}.jsonl") != source["sha256"][role]:
            raise ValueError(f"Registered {role} task file changed")
    roles = {role: [Task(**row) for row in read_jsonl(registration / f"{role}.jsonl")] for role in ROLES}
    if smoke_limit:  # engineering smoke only; the manifest says so and its numbers are never results
        roles = {r: sorted(ts, key=lambda t: hashlib.sha256(f"smoke:{t.id}".encode()).digest())[:smoke_limit]
                 for r, ts in roles.items()}
    audit([t for role in ROLES for t in roles[role]], raw_root)
    manifest = models_manifest(manifest_path)
    model_id, _ = MODELS["student"]
    spec = model_spec("student", models_root / "student", {"max_context": settings["max_context"],
                                                            "max_tokens": settings["max_tokens"]})
    files = manifest[model_id]["files"]
    models = {"student": {"spec": spec, "identity": expected_identity(spec, files), "files_sha256": fingerprint(files)}}
    snapshot = models["student"]["identity"]["local_snapshot"]
    starts = verify_v08(imports, registration_identity=source_identity, student_snapshot=snapshot,
                        seeds=settings["seeds"], v08_run=v08_run)
    arm = verify_v09(imports, registration_identity=source_identity, student_snapshot=snapshot, settings=settings,
                     v09_run=v09_run, v08_run=v08_run)
    same_training(imports, settings)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(imports, out / "imports", copy_function=shutil.copy2)
    for role in ROLES:
        write_jsonl(out / f"{role}.jsonl", [dataclasses.asdict(t) for t in roles[role]])
    limits = Limits(**settings["limits"])
    (out / "gold").mkdir()
    gold_meta = {}
    for role in GOLD_ROLES:
        cache = gold_cache(roles[role], raw_root, limits)
        atomic_json(out / "gold" / f"{role}.json", cache)
        gold_meta[role] = {"statuses": cache["statuses"], "retried": cache["retried"],
                           "sha256": digest(out / "gold" / f"{role}.json")}
    frozen = {"protocol": PROTOCOL, "settings": settings, "models": models, "raw_root": str(raw_root),
              "models_root": str(models_root),
              "registration": {"source_protocol": source["protocol"], "source_identity": source_identity,
                               "source_counts": source["counts"], "test_screen": source.get("test_screen")},
              "counts": {role: len(roles[role]) for role in ROLES},
              "databases": {role: len({t.db_id for t in roles[role]}) for role in ROLES},
              "sha256": {role: digest(out / f"{role}.jsonl") for role in ROLES},
              "imports": {**starts, "v09": arm,
                          "files": {p.relative_to(out / "imports").as_posix(): digest(p)
                                    for p in sorted((out / "imports").rglob("*")) if p.is_file()}},
              "gold": gold_meta, "smoke_limit": smoke_limit, "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load(run: Path, role: str | None = None, *, check_code: bool = True) -> tuple[dict[str, Any], list[Task]]:
    frozen = read_json(run / "manifest.json")
    identity = frozen.pop("identity")
    if fingerprint(frozen) != identity or frozen["protocol"] != PROTOCOL:
        raise ValueError("v0.11 manifest identity mismatch")
    frozen["identity"] = identity
    if check_code and frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new v0.11 run")
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
