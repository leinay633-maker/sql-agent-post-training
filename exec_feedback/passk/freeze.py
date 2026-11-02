"""Freeze a pass@k run: the v0.5 task registration (all four files re-verified, only test kept), the pinned 4B, the test
gold cache and its single-cell / multi-cell strata, and the adapters of the sampled models: v0.8's PTR (checked against
v0.8's run), v0.11's RLb and v0.12's OPD at the milestones those runs registered (each checked against its run's
manifest, milestone record and adapter files; RLb's start points must be these PTR adapters). The agent protocol and
temperature are checked against both runs' frozen settings."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..core import Task, audit, digest, read_jsonl, write_jsonl
from ..storage import atomic_json, code_fingerprint, fingerprint, read_json
from ..v7.freeze import _adapter_hash
from ..v8.freeze import expected_identity, model_spec
from ..v9.freeze import verify_imports as verify_v08
from ..v10.freeze import model_ready  # noqa: F401 - re-exported
from . import (BASE, MILESTONES, MODELS, OPD, PROTOCOL, RL, START, V08_RUN, V11_PROTOCOL, V11_RUN, V12_PROTOCOL,
               V12_RUN)
from .config import SAME_AS_ONLINE, validate_settings

ROLES = ("pool", "dev", "validation", "test")
ROLE = "test"
CONTROL = "+rec"  # suffix of a decoding-control condition: the model card's recommended decoding on the control subset
CONDITION = re.compile(rf"(?:(?P<base>{BASE})|(?P<family>{START}|{RL}|{OPD})_seed(?P<seed>\d+))(?P<control>\+rec)?")


def models_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for model_id, revision in MODELS.values():
        if manifest.get(model_id, {}).get("revision") != revision:
            raise ValueError(f"Model manifest does not pin {model_id}@{revision}")
    return manifest


def conditions(seeds: list[int], control: list[str]) -> list[str]:
    """Sampling order: the first seed's PTR and RLb (the main comparison), B and OPD, then the decoding control; then
    the other seeds' PTR and RLb pairs, and their OPD last, so a short booking drops OPD before a main-comparison pair."""
    first, rest = seeds[0], seeds[1:]
    order = [f"{START}_seed{first}", f"{RL}_seed{first}", BASE, f"{OPD}_seed{first}"]
    order += [f"{c}{CONTROL}" for c in control]
    for seed in rest:
        order += [f"{START}_seed{seed}", f"{RL}_seed{seed}"]
    return order + [f"{OPD}_seed{seed}" for seed in rest]


def parse(condition: str) -> tuple[str, int | None, bool]:
    """'B' -> ('B', None, False); 'RLb_seed42' -> ('RLb', 42, False); 'PTR_seed42+rec' -> ('PTR', 42, True)."""
    match = CONDITION.fullmatch(condition)
    if match is None:
        raise ValueError(f"Unknown condition: {condition}")
    control = bool(match.group("control"))
    return (BASE, None, control) if match.group("base") else (match.group("family"), int(match.group("seed")), control)


def control_tasks(tasks: list[Task], salt: str, count: int) -> list[Task]:
    """The decoding control's fixed subset of the frozen test tasks."""
    return sorted(tasks, key=lambda t: hashlib.sha256(f"{salt}:control:{t.id}".encode()).digest())[:count]


def verify_run(imports: Path, name: str, *, protocol: str, identity: str, registration_identity: str,
               student_snapshot: str) -> dict[str, Any]:
    """An imported run manifest: its own identity, the registered run, the same registration and 4B snapshot."""
    run = read_json(imports / name / "manifest.json")
    own = run.pop("identity")
    if fingerprint(run) != own or own != identity or run["protocol"] != protocol:
        raise ValueError(f"Imported {name} manifest is not the registered {name} run")
    if run["registration"]["source_identity"] != registration_identity:
        raise ValueError(f"{name} used another task registration")
    if run["models"]["student"]["identity"]["local_snapshot"] != student_snapshot:
        raise ValueError(f"{name} trained on another 4B snapshot")
    return {**run, "identity": own}


def milestone_adapters(imports: Path, name: str, run: dict[str, Any], family: str, seeds: list[int]) -> dict[str, Any]:
    """Per seed, the adapter at the milestone the run registered for it, re-hashed against its milestone record."""
    steps = read_json(imports / name / "steps.json")
    if steps["run"] != run["identity"]:
        raise ValueError(f"Imported {name} milestone record belongs to another run")
    out = {}
    for seed in seeds:
        condition = f"{family}_seed{seed}"
        chosen = steps["evaluate"].get(condition)
        if chosen is None:
            raise ValueError(f"{name} registered no milestone for {condition}")
        folder = imports / name / "adapters" / condition
        manifest, marks = read_json(folder / "manifest.json"), read_json(folder / "milestones.json")
        if manifest["run"] != run["identity"] or manifest["family"] != family or manifest["seed"] != seed:
            raise ValueError(f"Imported {condition} belongs to another run, family or seed")
        if marks.get(str(chosen["step"])) != chosen["adapter_hash"]:
            raise ValueError(f"Imported {condition} milestones disagree with {name}'s milestone record")
        if _adapter_hash(folder / "adapter") != chosen["adapter_hash"]:
            raise ValueError(f"Imported {condition} adapter files changed")
        out[condition] = {"step": chosen["step"], "adapter_hash": chosen["adapter_hash"], "short": bool(chosen["short"]),
                          "start_hash": manifest.get("start_hash")}
    return {"run": run["identity"], "k": steps["k"], "adapters": out}


def at_registered_milestones(name: str, family: str, checked: dict[str, Any]) -> None:
    """The study samples RLb and OPD at their run's registered K: a run registered at another K, or a seed whose
    milestone fell short of it (training ended early and its last step stood in), is refused."""
    k = MILESTONES[family]
    if checked["k"] != k:
        raise ValueError(f"{name} registered K={checked['k']}; the study compares at K={k}")
    for condition, adapter in checked["adapters"].items():
        if adapter["step"] != k or adapter["short"]:
            raise ValueError(f"{condition} is at step {adapter['step']} (short={adapter['short']}), not at K={k}")


def same_protocol(run: dict[str, Any], name: str, settings: dict[str, Any]) -> None:
    if settings["phase"] != "study":
        return
    differ = [k for k in SAME_AS_ONLINE if run["settings"].get(k) != settings[k]]
    if differ:
        raise ValueError(f"The samplers must use {name}'s agent protocol and temperature; differs in {differ}")


def greedy_reference(imports: Path, test_count: int, runs: tuple[str, ...]) -> dict[str, Any]:
    """Each model's greedy test score on its own run's night (descriptive only), as the bundle compacted it."""
    path = imports / "greedy.json"
    if not path.exists():
        return {}
    rows = read_json(path)["conditions"]
    for condition, row in rows.items():
        if parse(condition)[2] or row["run"] not in runs or row["tasks"] != test_count \
                or not 0 <= row["correct"] <= test_count:
            raise ValueError(f"Greedy reference for {condition} is not a complete test evaluation of v0.11 or v0.12")
    return rows


def strata(gold: dict[str, Any], tasks: list[Task]) -> dict[str, Any]:
    """Single-cell gold results (one row, one column) vs at least two cells; empty results would be a third group."""
    cells = {t.id: len(gold["results"][t.id]["rows"]) * len(gold["results"][t.id]["columns"]) for t in tasks}
    multi = sorted(i for i, c in cells.items() if c >= 2)
    return {"single": sum(c == 1 for c in cells.values()), "multi": len(multi),
            "empty": sum(c == 0 for c in cells.values()),
            "multi_sha256": hashlib.sha256(json.dumps(multi).encode()).hexdigest()}


def verify_imports(imports: Path, *, registration_identity: str, student_snapshot: str, settings: dict[str, Any],
                   registered: int, v08_run: str = V08_RUN, v11_run: str = V11_RUN,
                   v12_run: str = V12_RUN) -> dict[str, Any]:
    """Every import, checked the same way here and in the bundle script on the local machine."""
    seeds = settings["seeds"]
    starts = verify_v08(imports, registration_identity=registration_identity, student_snapshot=student_snapshot,
                        seeds=seeds, v08_run=v08_run)
    runs = {}
    for name, protocol, identity, family in (("v11", V11_PROTOCOL, v11_run, RL), ("v12", V12_PROTOCOL, v12_run, OPD)):
        run = verify_run(imports, name, protocol=protocol, identity=identity, registration_identity=registration_identity,
                         student_snapshot=student_snapshot)
        if run["imports"]["v08_run"] != starts["v08_run"]:
            raise ValueError(f"{name} imported another v0.8 run")
        same_protocol(run, name, settings)
        runs[name] = {**milestone_adapters(imports, name, run, family, seeds),
                      "ptr": {k: v for k, v in run["imports"]["adapters"].items() if k.startswith(f"{START}_")}}
        if settings["phase"] == "study":
            at_registered_milestones(name, family, runs[name])
    for seed in seeds:
        ptr = f"{START}_seed{seed}"
        if runs["v11"]["adapters"][f"{RL}_seed{seed}"]["start_hash"] != starts["adapters"][ptr] \
                or runs["v11"]["ptr"].get(ptr) != starts["adapters"][ptr]:
            raise ValueError(f"v0.11's {RL}_seed{seed} did not start from the imported {ptr}")
    greedy = greedy_reference(imports, registered, (runs["v11"]["run"], runs["v12"]["run"]))
    return {**starts, "v11": runs["v11"], "v12": runs["v12"], "greedy": greedy}


def freeze(registration: Path, raw_root: Path, out: Path, settings: dict[str, Any], *, models_root: Path,
           manifest_path: Path, imports: Path, smoke_limit: int = 0, v08_run: str = V08_RUN, v11_run: str = V11_RUN,
           v12_run: str = V12_RUN) -> dict[str, Any]:
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
    tasks = [Task(**row) for row in read_jsonl(registration / f"{ROLE}.jsonl")]
    registered = len(tasks)
    if smoke_limit:  # engineering smoke only; the manifest says so and its numbers are never results
        tasks = sorted(tasks, key=lambda t: hashlib.sha256(f"smoke:{t.id}".encode()).digest())[:smoke_limit]
    audit(tasks, raw_root)
    manifest = models_manifest(manifest_path)
    model_id, _ = MODELS["student"]
    spec = model_spec("student", models_root / "student", {"max_context": settings["max_context"],
                                                            "max_tokens": settings["max_tokens"]})
    files = manifest[model_id]["files"]
    models = {"student": {"spec": spec, "identity": expected_identity(spec, files), "files_sha256": fingerprint(files)}}
    seeds = settings["seeds"]
    checked = verify_imports(imports, registration_identity=source_identity,
                             student_snapshot=models["student"]["identity"]["local_snapshot"], settings=settings,
                             registered=registered, v08_run=v08_run, v11_run=v11_run, v12_run=v12_run)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(imports, out / "imports", copy_function=shutil.copy2)
    write_jsonl(out / f"{ROLE}.jsonl", [dataclasses.asdict(t) for t in tasks])
    (out / "gold").mkdir()
    cache = gold_cache(tasks, raw_root, Limits(**settings["limits"]))
    if cache["statuses"] != {"ok": len(tasks)}:
        raise ValueError(f"Every registered test reference must execute: {cache['statuses']}")
    atomic_json(out / "gold" / f"{ROLE}.json", cache)
    subset = sorted(t.id for t in control_tasks(tasks, settings["salt"], settings["control"]["tasks"]))
    frozen = {"protocol": PROTOCOL, "settings": settings, "models": models, "raw_root": str(raw_root),
              "models_root": str(models_root),
              "registration": {"source_protocol": source["protocol"], "source_identity": source_identity,
                               "source_counts": source["counts"], "test_screen": source.get("test_screen")},
              "counts": {ROLE: len(tasks)}, "databases": {ROLE: len({t.db_id for t in tasks})},
              "sha256": {ROLE: digest(out / f"{ROLE}.jsonl")},
              "imports": {**checked, "files": {p.relative_to(out / "imports").as_posix(): digest(p)
                                    for p in sorted((out / "imports").rglob("*")) if p.is_file()}},
              "gold": {ROLE: {"statuses": cache["statuses"], "retried": cache["retried"],
                              "sha256": digest(out / "gold" / f"{ROLE}.json")}},
              "strata": strata(cache, tasks), "conditions": conditions(seeds, settings["control"]["conditions"]),
              "control": {"tasks": len(subset), "ids_sha256": hashlib.sha256(json.dumps(subset).encode()).hexdigest()},
              "smoke_limit": smoke_limit, "code": code_fingerprint()}
    frozen["identity"] = fingerprint(frozen)
    atomic_json(out / "manifest.json", frozen)
    return frozen


def load(run: Path, *, tasks: bool = False, check_code: bool = True,
         databases: bool = True) -> tuple[dict[str, Any], list[Task]]:
    """The frozen run, re-checked; with tasks, the task file and (unless databases=False, for reading results on a PC
    without the databases) every database re-hashed under the frozen raw_root."""
    frozen = read_json(run / "manifest.json")
    identity = frozen.pop("identity")
    if fingerprint(frozen) != identity or frozen["protocol"] != PROTOCOL:
        raise ValueError("pass@k manifest identity mismatch")
    frozen["identity"] = identity
    if check_code and frozen["code"] != code_fingerprint():
        raise ValueError("Code changed; start a new pass@k run")
    if not tasks:
        return frozen, []
    path = run / f"{ROLE}.jsonl"
    if digest(path) != frozen["sha256"][ROLE]:
        raise ValueError("Registered task file changed")
    rows = [Task(**row) for row in read_jsonl(path)]
    if databases:
        audit(rows, Path(frozen["raw_root"]))
    return frozen, rows


def gold(run: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    path = run / "gold" / f"{ROLE}.json"
    if digest(path) != frozen["gold"][ROLE]["sha256"]:
        raise ValueError("Gold cache changed")
    return read_json(path)["results"]


def adapter_for(run: Path, frozen: dict[str, Any], condition: str) -> tuple[Path | None, str | None]:
    """The condition's adapter folder, re-hashed against the freeze (None for the base model)."""
    family, seed, _ = parse(condition)
    if condition not in frozen["conditions"]:
        raise ValueError(f"{condition} is not a condition of this run")
    if family == BASE:
        return None, None
    model = f"{family}_seed{seed}"
    if family == START:
        folder, expected = run / "imports" / "adapters" / model / "adapter", frozen["imports"]["adapters"][model]
    else:
        name = "v11" if family == RL else "v12"
        folder = run / "imports" / name / "adapters" / model / "adapter"
        expected = frozen["imports"][name]["adapters"][model]["adapter_hash"]
    if _adapter_hash(folder) != expected:
        raise ValueError(f"Imported adapter changed: {condition}")
    return folder, expected
