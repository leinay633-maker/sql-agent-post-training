"""Freeze a v0.12 run: the v0.5 task registration (re-verified), the pinned 4B and 235B (their tokenizer files byte-identical,
so the teacher scores the student's token ids as they are), gold caches for dev/validation/test, and v0.8's TR and PTR
adapters (the arms), each checked against the v0.8 run that trained them."""
from __future__ import annotations

import dataclasses
import hashlib
import shutil
from pathlib import Path
from typing import Any

from ..core import Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from ..v7.freeze import _adapter_hash
from ..v8.freeze import expected_identity, model_spec
from ..v8.freeze import model_ready as _model_ready
from . import ARMS, MODELS, PROTOCOL, SHARED_TOKENIZER, V08_PROTOCOL, V08_RUN
from .config import validate_settings

ROLES = ("pool", "dev", "validation", "test")
GOLD_ROLES = ("dev", "validation", "test")


def models_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for model_id, revision in MODELS.values():
        if manifest.get(model_id, {}).get("revision") != revision:
            raise ValueError(f"Model manifest does not pin {model_id}@{revision}")
    student, teacher = (manifest[MODELS[k][0]]["files"] for k in ("student", "teacher"))
    differ = [n for n in SHARED_TOKENIZER if student.get(n, {}).get("sha256") != teacher.get(n, {}).get("sha256")]
    if differ:
        raise ValueError(f"The teacher would score other token ids: tokenizer files differ: {differ}")
    return manifest


def verify_v08(imports: Path, *, registration_identity: str, student_snapshot: str, seeds: list[int],
               v08_run: str = V08_RUN) -> dict[str, Any]:
    """v0.8's manifest and its TR and PTR adapters (v0.9's check, for both families). The adapters' own efa-base.json
    records the v0.8 machine's download marker, so the base is proven through v0.8's manifest instead."""
    v08 = read_json(imports / "v08" / "manifest.json")
    identity = v08.pop("identity")
    if fingerprint(v08) != identity or identity != v08_run or v08["protocol"] != V08_PROTOCOL:
        raise ValueError("Imported v0.8 manifest is not the registered v0.8 run")
    if v08["registration"]["source_identity"] != registration_identity:
        raise ValueError("v0.8 used another task registration")
    if v08["models"]["student"]["identity"]["local_snapshot"] != student_snapshot:
        raise ValueError("v0.8 trained on another 4B snapshot")
    adapters = {}
    for family in ARMS:
        for seed in seeds:
            name = f"{family}_seed{seed}"
            folder = imports / "adapters" / name
            result, manifest = read_json(folder / "result.json"), read_json(folder / "manifest.json")
            if not (result.get("training_completed") and result.get("adapter_reload_verified")):
                raise ValueError(f"Imported {name} was not a finished, reload-verified model")
            if result["identity"] != manifest["identity"] or manifest["run"] != identity \
                    or manifest["family"] != family or manifest["seed"] != seed:
                raise ValueError(f"Imported {name} belongs to another run, family or seed")
            if _adapter_hash(folder / "adapter") != result["adapter_hash"]:
                raise ValueError(f"Imported {name} adapter files changed")
            adapters[name] = result["adapter_hash"]
    return {"v08_run": identity, "adapters": adapters}


def reference_rows(imports: Path, v08_run: str) -> dict[str, Any]:
    """v0.8's own per-task outcomes for R and the teacher (cross-night references, exploratory only), as the bundle
    compacted them from v0.8's evaluations: each file names the run and the episodes it came from."""
    out = {}
    for path in sorted((imports / "v08" / "eval-rows").glob("*/*.json")) if (imports / "v08").exists() else []:
        rows = read_json(path)
        if rows.get("run") != v08_run or rows.get("role") != path.parent.name or rows.get("condition") != path.stem:
            raise ValueError(f"Reference rows {path.name} belong to another v0.8 evaluation")
        out[f"{path.parent.name}/{path.stem}"] = {"tasks": len(rows["rows"]), "episodes_sha256": rows["episodes_sha256"]}
    return out


def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, models_root: Path,
           manifest_path: Path, imports: Path, smoke_limit: int = 0, v08_run: str = V08_RUN) -> dict[str, Any]:
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
    models = {}
    for kind in ("student", "teacher"):
        model_id, _ = MODELS[kind]
        spec = model_spec(kind, models_root / kind, {"max_context": settings["max_context"],
                                                     "max_tokens": settings["max_tokens"]})
        files = manifest[model_id]["files"]
        models[kind] = {"spec": spec, "identity": expected_identity(spec, files), "files_sha256": fingerprint(files)}
    arms = verify_v08(imports, registration_identity=source_identity,
                      student_snapshot=models["student"]["identity"]["local_snapshot"], seeds=settings["seeds"],
                      v08_run=v08_run)
    references = reference_rows(imports, arms["v08_run"])
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
              "imports": {**arms, "references": references,
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
        raise ValueError("v0.12 manifest identity mismatch")
    frozen["identity"] = identity
    if check_code and frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new v0.12 run")
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


def model_ready(frozen: dict[str, Any], kind: str = "student") -> dict[str, Any]:
    """Spec of a downloaded, hash-verified model of this run; raises if missing or not the pinned snapshot."""
    return _model_ready(frozen, kind)


def imported_adapter(run: Path, frozen: dict[str, Any], name: str) -> tuple[Path, str]:
    """An imported v0.8 adapter (TR_seed<n>, PTR_seed<n>), re-hashed against the freeze."""
    expected = frozen["imports"]["adapters"].get(name)
    if expected is None:
        raise ValueError(f"{name} was not imported")
    folder = run / "imports" / "adapters" / name / "adapter"
    if _adapter_hash(folder) != expected:
        raise ValueError(f"Imported adapter changed: {name}")
    return folder, expected
