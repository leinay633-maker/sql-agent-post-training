"""Freeze a v0.9 run: the v0.5 task registration (re-verified), the pinned 4B, gold caches for dev/validation/test,
and v0.8's PTR adapters, each checked against the v0.8 run that trained them."""
from __future__ import annotations

import dataclasses
import hashlib
import shutil
from pathlib import Path
from typing import Any

from ..core import Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from ..v7.freeze import _adapter_hash, _stats_match
from ..v8.freeze import MARKER, expected_identity, model_spec
from . import MODELS, PROTOCOL, START, V08_PROTOCOL, V08_RUN
from .config import validate_settings

ROLES = ("pool", "dev", "validation", "test")
GOLD_ROLES = ("dev", "validation", "test")


def models_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for model_id, revision in MODELS.values():
        if manifest.get(model_id, {}).get("revision") != revision:
            raise ValueError(f"Model manifest does not pin {model_id}@{revision}")
    return manifest


def verify_imports(imports: Path, *, registration_identity: str, student_snapshot: str, seeds: list[int],
                   v08_run: str = V08_RUN) -> dict[str, Any]:
    """v0.8's manifest and its PTR adapters. The adapters' own efa-base.json records the v0.8 machine's snapshot
    fingerprint, which includes that machine's download marker (inode, mtime), so the base is proven through the v0.8
    manifest instead: same registration, same pinned 4B snapshot, adapters trained by that run with these hashes."""
    v08 = read_json(imports / "v08" / "manifest.json")
    identity = v08.pop("identity")
    if fingerprint(v08) != identity or identity != v08_run or v08["protocol"] != V08_PROTOCOL:
        raise ValueError("Imported v0.8 manifest is not the registered v0.8 run")
    if v08["registration"]["source_identity"] != registration_identity:
        raise ValueError("v0.8 used another task registration")
    if v08["models"]["student"]["identity"]["local_snapshot"] != student_snapshot:
        raise ValueError("v0.8 trained on another 4B snapshot")
    adapters = {}
    for seed in seeds:
        name = f"{START}_seed{seed}"
        folder = imports / "adapters" / name
        result, manifest = read_json(folder / "result.json"), read_json(folder / "manifest.json")
        if not (result.get("training_completed") and result.get("adapter_reload_verified")):
            raise ValueError(f"Imported {name} was not a finished, reload-verified model")
        if result["identity"] != manifest["identity"] or manifest["run"] != identity \
                or manifest["family"] != START or manifest["seed"] != seed:
            raise ValueError(f"Imported {name} belongs to another run, family or seed")
        if _adapter_hash(folder / "adapter") != result["adapter_hash"]:
            raise ValueError(f"Imported {name} adapter files changed")
        adapters[name] = result["adapter_hash"]
    return {"v08_run": identity, "adapters": adapters}


def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, models_root: Path,
           manifest_path: Path, imports: Path, smoke_limit: int = 0, v08_run: str = V08_RUN) -> dict[str, Any]:
    from ..v6.freeze import gold_cache
    from ..core import Limits

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
    imported = verify_imports(imports, registration_identity=source_identity,
                              student_snapshot=models["student"]["identity"]["local_snapshot"],
                              seeds=settings["seeds"], v08_run=v08_run)
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
              "registration": {"source_protocol": source["protocol"], "source_identity": source_identity,
                               "source_counts": source["counts"], "test_screen": source.get("test_screen")},
              "counts": {role: len(roles[role]) for role in ROLES},
              "databases": {role: len({t.db_id for t in roles[role]}) for role in ROLES},
              "sha256": {role: digest(out / f"{role}.jsonl") for role in ROLES},
              "imports": {**imported, "files": {p.relative_to(out / "imports").as_posix(): digest(p)
                                                for p in sorted((out / "imports").rglob("*")) if p.is_file()}},
              "gold": gold_meta, "smoke_limit": smoke_limit, "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load(run: Path, role: str | None = None, *, check_code: bool = True) -> tuple[dict[str, Any], list[Task]]:
    frozen = read_json(run / "manifest.json")
    identity = frozen.pop("identity")
    if fingerprint(frozen) != identity or frozen["protocol"] != PROTOCOL:
        raise ValueError("v0.9 manifest identity mismatch")
    frozen["identity"] = identity
    if check_code and frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new v0.9 run")
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


def model_ready(frozen: dict[str, Any]) -> dict[str, Any]:
    """Spec of the downloaded, hash-verified 4B; raises if the download is missing or differs from the pin."""
    spec = frozen["models"]["student"]["spec"]
    path = Path(spec["local_path"])
    if not (path / MARKER).exists():
        raise ValueError(f"The 4B is not downloaded/verified yet: {path}")
    saved = read_json(path / MARKER)
    if saved["manifest"] != frozen["models"]["student"]["files_sha256"] or not _stats_match(path, saved):
        raise ValueError(f"The 4B on disk is not the verified pinned snapshot: {path}")
    return spec


def start_adapter(run: Path, frozen: dict[str, Any], seed: int) -> tuple[Path, str]:
    """The imported PTR adapter an online run of this seed starts from, re-hashed."""
    name = f"{START}_seed{seed}"
    expected = frozen["imports"]["adapters"].get(name)
    if expected is None:
        raise ValueError(f"{name} was not imported")
    folder = run / "imports" / "adapters" / name / "adapter"
    if _adapter_hash(folder) != expected:
        raise ValueError(f"Imported adapter changed: {name}")
    return folder, expected
