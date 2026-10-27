"""Freeze new selections and a verified v0.3 warm start, without rewriting v0.3."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ..core import Task, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from . import MODEL_ID, MODEL_REVISION, PROTOCOL
from .config import validate_settings


def prepare(source: Path, raw_root: Path, out: Path,
            settings: dict[str, Any]) -> dict[str, Any]:
    from ..datasets import choose_tasks, load_prepared, validate_gold
    from ..core import Limits
    from ..models import model_identity

    validate_settings(settings)
    source, raw_root = source.resolve(strict=True), raw_root.resolve(strict=True)
    old = read_json(source / "study.json")
    if old["status"] != "complete":
        raise ValueError("v0.3 study must finish and be recovered before v0.4")
    source_protocol = {3: "efa-v3", 4: "efa-v3.1-gold-screen"}.get(old["config"].get("version"))
    if source_protocol is None:
        raise ValueError("Only a completed v0.3 or v0.3.1 study is a supported warm start")
    warm = source / "train_repair_42"
    result = read_json(warm / "result.json")
    manifest = read_json(warm / "manifest.json")
    if not result.get("training_completed") or not result.get("adapter_reload_verified"):
        raise ValueError("Warm-start adapter is not verified complete")
    if result["identity"] != manifest["identity"] or manifest["fixture"]:
        raise ValueError("Warm-start identity/fixture mismatch")
    if (manifest["model"]["id"], manifest["model"]["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Only the pinned Qwen3-4B base is allowed")
    pairs = source / "pairs" / "repair.jsonl"
    if digest(pairs) != manifest["data_sha256"]:
        raise ValueError("Warm-start training corpus changed")
    corpus = read_jsonl(pairs)
    if not corpus or any(r["split"] != "train" or r["fixture"] for r in corpus):
        raise ValueError("Warm start must be trained exclusively on real train data")
    tasks = load_prepared(source / "prepared", raw_root)
    task_map = {t.id: t for t in tasks}
    for row in corpus:
        task = task_map.get(row["task_id"])
        if task is None or task.split != "train" or row["db_sha256"] != task.db_sha256:
            raise ValueError("Warm-start corpus disagrees with the audited train split")
    spec = dict(old["config"]["model"])
    spec.update(id=MODEL_ID, revision=MODEL_REVISION, backend="transformers",
                device="cuda", quantization="none", allow_download=False,
                local_path="", max_tokens=settings["max_tokens"],
                max_context_tokens=settings["max_context"],
                batch_size=settings["generation_batch"], temperature=0.0)
    adapter_hash = model_identity(spec, warm / "adapter")["adapter_hash"]
    if adapter_hash != result["adapter_hash"]:
        raise ValueError("Warm-start adapter hash mismatch")
    # Reuse the source study's recorded pre-selection screen; never rescreen
    # to a different pool silently.
    screened = set()
    if old["config"]["experiment"].get("gold_screen"):
        recorded = old["stages"]["prepare"]["artifacts"]
        for name in ("prepared/gold_screen/excluded.json", "selection.json"):
            if recorded.get(name) != digest(source / name):
                raise ValueError(f"Source screen record changed after the study: {name}")
        screened = set(read_json(source / "prepared" / "gold_screen" / "excluded.json")["excluded"])
        if read_json(source / "selection.json")["gold_screen_excluded"] != sorted(screened):
            raise ValueError("Source selection and screen disagree")
        if any(row["task_id"] in screened for row in corpus):
            raise ValueError("Warm-start corpus contains screened-out tasks")
    tasks = [t for t in tasks if t.id not in screened]
    training = choose_tasks([t for t in tasks if t.split == "train"],
                            settings["train_tasks"], settings["selection_seed"],
                            stratified=True)
    validation = []
    if settings["phase"] == "study":
        validation = choose_tasks([t for t in tasks if t.split == "validation"],
                                  settings["validation_tasks"], settings["selection_seed"],
                                  stratified=True)
    if len(training) != settings["train_tasks"] or len(validation) != settings["validation_tasks"]:
        raise ValueError("Insufficient tasks; do not silently shrink the registered study")
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "train.jsonl", [dataclasses.asdict(t) for t in training])
    write_jsonl(out / "validation.jsonl", [dataclasses.asdict(t) for t in validation])
    frozen = {
        "protocol": PROTOCOL, "settings": settings, "model": spec,
        "raw_root": str(raw_root), "source_study": str(source),
        "warm_adapter": str(warm / "adapter"), "warm_adapter_hash": adapter_hash,
        "warm_corpus_origins": manifest["origins"],
        "source_protocol": source_protocol,
        "source_gold_screen_excluded": sorted(screened),
        "source_study_sha256": digest(source / "study.json"),
        "source_pairs_sha256": digest(pairs),
        "code": code_fingerprint(),
        "train_sha256": digest(out / "train.jsonl"),
        "validation_sha256": digest(out / "validation.jsonl"),
        "train_databases": sorted({t.db_id for t in training}),
        "validation_databases": sorted({t.db_id for t in validation}),
        "conditions": ["S4_frozen_repair_sft", "E4_online_rloo"],
        "comparison": "E4 minus S4; conditional on one frozen SFT warm start",
    }
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    # Gold preflight here is TRAIN ONLY. Validation is not scored until every
    # registered training seed is frozen. No dev/test import or adaptive stop.
    checked = validate_gold(training, raw_root, out / "gold_train",
                            Limits(**settings["limits"]), workers=2)
    if checked["invalid_ids"]:
        raise ValueError("Train gold preflight failed; all task ids retained")
    atomic_json(out / "prepared.json", {"identity": frozen["identity"]})
    return frozen


def load_run(out: Path, *, training: bool) -> tuple[dict[str, Any], list[Task]]:
    from ..core import audit
    from ..models import model_identity

    frozen = read_json(out / "manifest.json")
    identity = frozen.pop("identity")
    if fingerprint(frozen) != identity or frozen["protocol"] != PROTOCOL:
        raise ValueError("v0.4 manifest identity mismatch")
    frozen["identity"] = identity
    if read_json(out / "prepared.json")["identity"] != identity:
        raise ValueError("Preparation did not complete")
    if frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; use a new v0.4 study")
    current = model_identity(frozen["model"], Path(frozen["warm_adapter"]))
    if current["adapter_hash"] != frozen["warm_adapter_hash"]:
        raise ValueError("Warm-start adapter changed")
    split = "train" if training else "validation"
    path = out / f"{split}.jsonl"
    if digest(path) != frozen[f"{split}_sha256"]:
        raise ValueError("Registered task denominator changed")
    tasks = [Task(**row) for row in read_jsonl(path)]
    if any(task.split != split or task.fixture for task in tasks):
        raise ValueError("Wrong split or fixture in registered task file")
    if tasks:
        audit(tasks, Path(frozen["raw_root"]))
    return frozen, tasks
