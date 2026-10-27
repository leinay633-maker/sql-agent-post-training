"""Freeze a v0.7 run: v0.5 task registration (re-verified), model manifests, gold caches and the v0.5 imports."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import Limits, Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from . import HYBRID, MODELS, NON_THINKING_TEMPLATE, PROTOCOL
from .config import validate_settings

ROLES = ("pool", "dev", "validation", "test")
GOLD_ROLES = ("dev", "validation", "test")
SNAPSHOT_SUFFIXES = {".json", ".safetensors", ".bin", ".txt"}
IMPORTED = ("R", "P")
MARKER = ".efa-verified.json"


# ----------------------------------------------------------------------------------------------- models
def model_spec(kind: str, local_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    model_id, revision = MODELS[kind]
    spec = {"backend": "transformers", "id": model_id, "revision": revision, "device": "cuda", "quantization": "none",
            "local_path": str(local_path), "allow_download": False, "max_context_tokens": settings["max_context"],
            "max_tokens": settings["max_tokens"], "temperature": 0.0, "seed": 42, "batch_size": 16, "endpoint": "",
            "allow_remote": False}
    if kind in HYBRID:
        spec["chat_template"] = NON_THINKING_TEMPLATE
    return spec


def expected_identity(spec: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
    """exec_feedback.models.model_identity of the finished download, computed from the pinned manifest."""
    identity = {k: spec.get(k) for k in ("backend", "id", "revision", "device", "quantization",
                                        "max_context_tokens", "max_tokens", "temperature", "seed", "batch_size")}
    if spec.get("chat_template"):
        identity["chat_template"] = hashlib.sha256(spec["chat_template"].encode()).hexdigest()
    identity["local_snapshot"] = fingerprint({name: f["sha256"] for name, f in files.items()
                                              if Path(name).suffix in SNAPSHOT_SUFFIXES})
    return identity


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_dir(path: Path, files: dict[str, Any], *, rehash: bool = False) -> None:
    """Every pinned file present with its manifest sha256. The full hash runs once (download); later checks compare
    the recorded size/inode/mtime of each verified file."""
    marker = path / MARKER
    expected = fingerprint(files)
    if not rehash and marker.exists() and read_json(marker).get("manifest") == expected and _stats_match(path, read_json(marker)):
        return
    for name, meta in sorted(files.items()):
        target = path / name
        if not target.is_file() or target.stat().st_size != meta["size"] or file_sha256(target) != meta["sha256"]:
            raise ValueError(f"Model file missing or different from the pinned manifest: {target}")
    mark_verified(path, files)


def mark_verified(path: Path, files: dict[str, Any]) -> None:
    """Record size/inode/mtime of files whose sha256 the caller has just checked against the manifest."""
    stats = {}
    for name in sorted(files):
        st = (path / name).stat()
        stats[name] = [st.st_size, st.st_ino, st.st_mtime_ns]
    atomic_json(path / MARKER, {"manifest": fingerprint(files), "stat": stats})


def _stats_match(path: Path, saved: dict[str, Any]) -> bool:
    for name, recorded in saved["stat"].items():
        target = path / name
        if not target.is_file():
            return False
        st = target.stat()
        if [st.st_size, st.st_ino, st.st_mtime_ns] != recorded:
            return False
    return True


def models_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for kind, (model_id, revision) in MODELS.items():
        if manifest.get(model_id, {}).get("revision") != revision:
            raise ValueError(f"Model manifest does not pin {model_id}@{revision}")
    return manifest


# ----------------------------------------------------------------------------------------------- v0.5 imports
def _adapter_hash(adapter: Path) -> str:
    files = sorted(p for p in adapter.rglob("*") if p.is_file())
    if not (adapter / "adapter_config.json").exists() or not files:
        raise ValueError(f"Not a saved PEFT adapter: {adapter}")
    return fingerprint({p.relative_to(adapter).as_posix(): digest(p) for p in files})


def verify_imports(imports: Path, v05_identity: str, student_snapshot: str, pool_ids: set[str],
                   seeds: list[int]) -> dict[str, Any]:
    """v0.5 sft_R / pairs / round-1 outcomes / R and P adapters, each checked against v0.5's own records."""
    selection = read_json(imports / "selection.json")
    if selection["run"] != v05_identity:
        raise ValueError("Imported v0.5 selection belongs to another run")
    if digest(imports / "sft_R.jsonl") != selection["datasets"]["R"]["sha256"]:
        raise ValueError("Imported sft_R.jsonl differs from v0.5's frozen dataset")
    pairs = read_json(imports / "pairs.json")
    if pairs["run"] != v05_identity or digest(imports / "pairs.jsonl") != pairs["sha256"]:
        raise ValueError("Imported v0.5 preference pairs changed")
    outcomes = read_json(imports / "student_round1.json")
    if outcomes["run"] != v05_identity:
        raise ValueError("Imported round-1 outcomes belong to another run")
    tasks = outcomes["tasks"]
    classes = Counter("frontier" if 0 < c < n else "easy" if c else "unsolved" for n, c in tasks.values())
    if (sum(n for n, _ in tasks.values()) != selection["trajectories"] or len(tasks) != selection["sampled_tasks"]
            or classes["frontier"] != selection["frontier_tasks"] or classes["easy"] != selection["easy_tasks"]
            or classes["unsolved"] != selection["unsolved_tasks"] or not set(tasks) <= pool_ids):
        raise ValueError("Imported round-1 outcomes disagree with v0.5's selection summary")
    adapters = {}
    for variant in IMPORTED:
        for seed in seeds:
            name = f"{variant}_seed{seed}"
            folder = imports / "adapters" / name
            result, manifest = read_json(folder / "result.json"), read_json(folder / "manifest.json")
            if not (result.get("training_completed") and result.get("adapter_reload_verified")):
                raise ValueError(f"Imported {name} was not a frozen, reload-verified model")
            if result["identity"] != manifest["identity"] or manifest["run"] != v05_identity:
                raise ValueError(f"Imported {name} belongs to another run")
            if _adapter_hash(folder / "adapter") != result["adapter_hash"]:
                raise ValueError(f"Imported {name} adapter files changed")
            if read_json(folder / "adapter" / "efa-base.json").get("local_snapshot") != student_snapshot:
                raise ValueError(f"Imported {name} was trained on another base snapshot")
            adapters[name] = result["adapter_hash"]
    return {"v05_run": v05_identity, "sft_R_sha256": selection["datasets"]["R"]["sha256"],
            "sft_R_examples": selection["datasets"]["R"]["examples"], "pairs": pairs["pairs"],
            "pairs_sha256": pairs["sha256"], "student_round1_sha256": digest(imports / "student_round1.json"),
            "student_round1_classes": dict(classes), "adapters": adapters}


# ----------------------------------------------------------------------------------------------- freeze
def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, models_root: Path,
           manifest_path: Path, imports: Path, smoke_limit: int = 0) -> dict[str, Any]:
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
    full_pool = {t.id for t in roles["pool"]}
    if smoke_limit:  # engineering smoke only; the manifest says so and its numbers are never results
        roles = {r: sorted(ts, key=lambda t: hashlib.sha256(f"smoke:{t.id}".encode()).digest())[:smoke_limit]
                 for r, ts in roles.items()}
    audit([t for role in ROLES for t in roles[role]], raw_root)
    manifest = models_manifest(manifest_path)
    models = {}
    for kind, (model_id, _) in MODELS.items():
        spec = model_spec(kind, models_root / kind, settings)
        files = manifest[model_id]["files"]
        models[kind] = {"spec": spec, "identity": expected_identity(spec, files), "files_sha256": fingerprint(files)}
    imported = verify_imports(imports, source_identity, models["student"]["identity"]["local_snapshot"], full_pool,
                              settings["seeds"])
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(imports, out / "imports", copy_function=shutil.copy2)
    for role in ROLES:
        write_jsonl(out / f"{role}.jsonl", [dataclasses.asdict(t) for t in roles[role]])
    pool_ids = {t.id for t in roles["pool"]}
    r_in_pool = sum(1 for row in read_jsonl(out / "imports" / "sft_R.jsonl") if row["task_id"] in pool_ids)
    if not smoke_limit and r_in_pool != settings["train_examples"]:
        raise ValueError(f"v0.5 R has {r_in_pool} pool examples, settings register {settings['train_examples']}")
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
              "n_examples": r_in_pool, "gold": gold_meta, "smoke_limit": smoke_limit, "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load(run: Path, role: str | None = None, *, check_code: bool = True) -> tuple[dict[str, Any], list[Task]]:
    frozen = read_json(run / "manifest.json")
    identity = frozen.pop("identity")
    if fingerprint(frozen) != identity or frozen["protocol"] != PROTOCOL:
        raise ValueError("v0.7 manifest identity mismatch")
    frozen["identity"] = identity
    if check_code and frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new v0.7 run")
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


def imported(run: Path, frozen: dict[str, Any], name: str) -> Path:
    path = run / "imports" / name
    if digest(path) != frozen["imports"]["files"][name]:
        raise ValueError(f"Imported file changed: {name}")
    return path


def model_ready(frozen: dict[str, Any], kind: str) -> dict[str, Any]:
    """Spec of a downloaded, hash-verified model; raises if the download is missing or differs from the pin."""
    spec = frozen["models"][kind]["spec"]
    path = Path(spec["local_path"])
    marker = path / MARKER
    if not marker.exists():
        raise ValueError(f"Model {kind} not downloaded/verified yet: {path}")
    saved = read_json(marker)
    if saved["manifest"] != frozen["models"][kind]["files_sha256"] or not _stats_match(path, saved):
        raise ValueError(f"Model {kind} on disk is not the verified pinned snapshot: {path}")
    return spec
