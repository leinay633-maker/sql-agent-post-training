"""Freeze a deployment-study run: the v0.5 task registration (all four files re-verified; the evaluation roles kept),
the pinned 4B, gold caches, the load test's fixed task subset, and v0.11's RLb_seed44 adapter at the milestone v0.11
registered for it (checked against v0.11's manifest, milestone record and adapter files). The merged bf16 model is
built on the host after the freeze and verified against its record every time it is used."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..core import Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, exclusive_lock, fingerprint, read_json
from ..v7.freeze import _adapter_hash
from ..v8.freeze import expected_identity, model_spec
from ..v10.freeze import model_ready  # noqa: F401 - re-exported
from ..v10.merge import merge_adapter, record_of
from ..v10.weights import dir_identity
from ..passk.freeze import at_registered_milestones, milestone_adapters, verify_run
from . import FAMILY, MODEL, MODELS, PROTOCOL, SEED, V11_PROTOCOL, V11_RUN
from .config import SAME_AS_V11, validate_settings

REGISTERED = ("pool", "dev", "validation", "test")


def models_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for model_id, revision in MODELS.values():
        if manifest.get(model_id, {}).get("revision") != revision:
            raise ValueError(f"Model manifest does not pin {model_id}@{revision}")
    return manifest


def bench_tasks(tasks: list[Task], salt: str, count: int) -> list[Task]:
    """The load test's fixed test tasks, in the order the clients submit them."""
    return sorted(tasks, key=lambda t: hashlib.sha256(f"{salt}:bench:{t.id}".encode()).digest())[:count]


def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, models_root: Path,
           manifest_path: Path, imports: Path, smoke_limit: int = 0, v11_run: str = V11_RUN) -> dict[str, Any]:
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
    for role in REGISTERED:
        if digest(registration / f"{role}.jsonl") != source["sha256"][role]:
            raise ValueError(f"Registered {role} task file changed")
    roles = {role: [Task(**row) for row in read_jsonl(registration / f"{role}.jsonl")] for role in settings["roles"]}
    if smoke_limit:  # engineering smoke only; the manifest says so and its numbers are never results
        roles = {r: sorted(ts, key=lambda t: hashlib.sha256(f"smoke:{t.id}".encode()).digest())[:smoke_limit]
                 for r, ts in roles.items()}
    if settings["bench_tasks"] + settings["warmup_tasks"] > len(roles["test"]):
        raise ValueError("The load test and its warm-up need more test tasks than there are")
    audit([t for ts in roles.values() for t in ts], raw_root)
    manifest = models_manifest(manifest_path)
    model_id, _ = MODELS["student"]
    spec = model_spec("student", models_root / "student", {"max_context": settings["max_context"],
                                                            "max_tokens": settings["max_tokens"]})
    files = manifest[model_id]["files"]
    models = {"student": {"spec": spec, "identity": expected_identity(spec, files), "files_sha256": fingerprint(files)}}
    run = verify_run(imports, "v11", protocol=V11_PROTOCOL, identity=v11_run, registration_identity=source_identity,
                     student_snapshot=models["student"]["identity"]["local_snapshot"])
    if settings["phase"] == "study":
        differ = [k for k in SAME_AS_V11 if run["settings"].get(k) != settings[k]]
        if differ:
            raise ValueError(f"The deployment must keep v0.11's agent protocol; differs in {differ}")
    v11 = milestone_adapters(imports, "v11", run, FAMILY, [SEED])
    if settings["phase"] == "study":
        at_registered_milestones("v11", FAMILY, v11)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(imports, out / "imports", copy_function=shutil.copy2)
    for role, tasks in roles.items():
        write_jsonl(out / f"{role}.jsonl", [dataclasses.asdict(t) for t in tasks])
    (out / "gold").mkdir()
    gold_meta = {}
    for role, tasks in roles.items():
        cache = gold_cache(tasks, raw_root, Limits(**settings["limits"]))
        if cache["statuses"] != {"ok": len(tasks)}:
            raise ValueError(f"Every registered {role} reference must execute: {cache['statuses']}")
        atomic_json(out / "gold" / f"{role}.json", cache)
        gold_meta[role] = {"statuses": cache["statuses"], "retried": cache["retried"],
                           "sha256": digest(out / "gold" / f"{role}.json")}
    bench = [t.id for t in bench_tasks(roles["test"], settings["salt"], settings["bench_tasks"])]
    frozen = {"protocol": PROTOCOL, "settings": settings, "models": models, "raw_root": str(raw_root),
              "models_root": str(models_root),
              "registration": {"source_protocol": source["protocol"], "source_identity": source_identity,
                               "source_counts": source["counts"]},
              "counts": {r: len(ts) for r, ts in roles.items()},
              "databases": {r: len({t.db_id for t in ts}) for r, ts in roles.items()},
              "sha256": {r: digest(out / f"{r}.jsonl") for r in roles},
              "imports": {"v11": v11, "files": {p.relative_to(out / "imports").as_posix(): digest(p)
                                                for p in sorted((out / "imports").rglob("*")) if p.is_file()}},
              "gold": gold_meta, "bench": {"tasks": bench, "sha256": hashlib.sha256(json.dumps(bench).encode()).hexdigest()},
              "smoke_limit": smoke_limit, "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load(run: Path, role: str | None = None, *, check_code: bool = True) -> tuple[dict[str, Any], list[Task]]:
    frozen = read_json(run / "manifest.json")
    identity = frozen.pop("identity")
    if fingerprint(frozen) != identity or frozen["protocol"] != PROTOCOL:
        raise ValueError("Deployment manifest identity mismatch")
    frozen["identity"] = identity
    if check_code and frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new deployment run")
    if role is None:
        return frozen, []
    if role not in frozen["counts"]:
        raise ValueError(f"Unknown task role: {role}")
    path = run / f"{role}.jsonl"
    if digest(path) != frozen["sha256"][role]:
        raise ValueError("Registered task file changed")
    tasks = [Task(**row) for row in read_jsonl(path)]
    audit(tasks, Path(frozen["raw_root"]))
    return frozen, tasks


def gold(run: Path, frozen: dict[str, Any], role: str) -> dict[str, Any]:
    path = run / "gold" / f"{role}.json"
    if role not in frozen["gold"] or digest(path) != frozen["gold"][role]["sha256"]:
        raise ValueError(f"Gold cache for {role} missing or changed")
    return read_json(path)["results"]


def adapter(run: Path, frozen: dict[str, Any]) -> tuple[Path, str, int]:
    """RLb_seed44 at v0.11's registered milestone, re-hashed against the freeze."""
    chosen = frozen["imports"]["v11"]["adapters"][MODEL]
    folder = run / "imports" / "v11" / "adapters" / MODEL / "adapter"
    if _adapter_hash(folder) != chosen["adapter_hash"]:
        raise ValueError(f"Imported adapter changed: {MODEL}")
    return folder, chosen["adapter_hash"], chosen["step"]


def merged_dir(frozen: dict[str, Any]) -> Path:
    step = frozen["imports"]["v11"]["adapters"][MODEL]["step"]
    return Path(frozen["models_root"]) / "merged" / f"{MODEL}-{step:04d}"


def merged_model(run: Path, frozen: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """The verified merged directory (raises if missing, built from other inputs or changed on disk)."""
    _, adapter_hash, _ = adapter(run, frozen)
    folder = merged_dir(frozen)
    if not record_of(folder).exists():
        raise ValueError(f"{MODEL} has not been merged yet: {folder}")
    record = read_json(record_of(folder))
    if record["inputs"]["base"] != frozen["models"]["student"]["identity"]["local_snapshot"] \
            or record["inputs"]["adapter"] != adapter_hash:
        raise ValueError(f"Merged {MODEL} was built from other inputs")
    if dir_identity(folder) != record["identity"]:
        raise ValueError(f"Merged {MODEL} changed on disk")
    return folder, record


def merge(run: Path, *, threads: int = 16) -> dict[str, Any]:
    """v0.10's merge (fp32, probe check, one rounding to bf16) of the deployed adapter into the base 4B."""
    import torch

    frozen, _ = load(run)
    base = Path(model_ready(frozen)["local_path"])
    folder, adapter_hash, _ = adapter(run, frozen)
    out = merged_dir(frozen)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    with exclusive_lock(out.parent / f".{out.name}.lock", blocking=True):
        if record_of(out).exists():
            try:
                return merged_model(run, frozen)[1]
            except ValueError:  # built from other inputs or changed: kept aside, built again
                stale = out.with_name(out.name + f".stale-{os.getpid()}")
                if out.exists():
                    os.replace(out, stale)
                os.replace(record_of(out), record_of(stale))
        return merge_adapter(base, folder, out, name=MODEL,
                             inputs={"base": frozen["models"]["student"]["identity"]["local_snapshot"],
                                     "adapter": adapter_hash})


def stop_ids(spec: dict[str, Any]) -> list[int]:
    """Every eos the model card's generation_config lists, as the evaluation engines stop on (v7.engine.stop_ids);
    vLLM's server is started with --generation-config vllm, so the service sends them with every request."""
    config = json.loads((Path(spec["local_path"]) / "generation_config.json").read_text(encoding="utf-8"))
    eos = config.get("eos_token_id")
    ids = eos if isinstance(eos, list) else [eos]
    return sorted(i for i in ids if isinstance(i, int) and i >= 0)
