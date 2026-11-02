"""Merge a pass@k run sampled on two hosts (passk_run.sh --only) into one run for the report, on the PC.

Usage: python scripts/passk_merge.py <run A> <run B> <out>
  run A, run B: outputs/passk/run from each host's final results package; out: a new folder (starts as a copy of A).
Both hosts froze the same bundle, but v6.freeze.gold_cache records every reference query's timing and retries, so the
two run identities differ. The merge goes ahead only if everything else frozen is the same: the manifests without
their identity and the gold file's hash and retry count, the gold caches (each first checked against its own freeze)
without timings and retry counts, and the task file. Each model that run B finished is then copied in with its
manifest and result re-keyed to run A's identity (rows untouched). A model that both hosts touched, finished or not,
stops the merge. out/merged.json keeps the original identities, the checks and any unfinished attempts.
Then: python -m exec_feedback.passk report --run <out> --offline   (the results come without the databases)"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exec_feedback.core import Task, digest  # noqa: E402
from exec_feedback.passk.freeze import ROLE, control_tasks, gold, load, parse  # noqa: E402
from exec_feedback.storage import atomic_json, fingerprint, read_json  # noqa: E402


def comparable_manifest(frozen: dict[str, Any]) -> dict[str, Any]:
    m = copy.deepcopy(frozen)
    m.pop("identity")
    for key in ("sha256", "retried"):
        m["gold"][ROLE].pop(key)
    return m


def comparable_gold(run: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    gold(run, frozen)  # the file still matches its own freeze
    return {k: v for k, v in read_json(run / "gold" / f"{ROLE}.json").items() if k not in ("attempts", "retried")}


def sample_identity(frozen: dict[str, Any], tasks: list[Task], condition: str, manifest: dict[str, Any]) -> str:
    """sample.sample's identity of a model's samples, for the given run."""
    s = frozen["settings"]
    subset = control_tasks(tasks, s["salt"], s["control"]["tasks"]) if parse(condition)[2] else tasks
    return fingerprint({"run": frozen["identity"], "condition": condition, "adapter_hash": manifest["adapter_hash"],
                        "samples": manifest["samples"], "sampling": manifest["sampling"],
                        "tasks": sorted(t.id for t in subset), "model": frozen["models"]["student"]["identity"]})


def touched(run: Path) -> tuple[set[str], set[str]]:
    """Models with a samples folder, and those of them finished."""
    folders = [p for p in (run / "samples").iterdir() if p.is_dir()] if (run / "samples").exists() else []
    return {p.name for p in folders}, {p.name for p in folders if (p / "result.json").exists()}


def merge(a: Path, b: Path, out: Path) -> dict[str, Any]:
    fa, tasks = load(a, tasks=True, check_code=False, databases=False)
    fb, _ = load(b, tasks=True, check_code=False, databases=False)
    checks = {"manifests": comparable_manifest(fa) == comparable_manifest(fb),
              "gold": comparable_gold(a, fa) == comparable_gold(b, fb),
              "tasks": digest(a / f"{ROLE}.jsonl") == digest(b / f"{ROLE}.jsonl")}
    if not all(checks.values()):
        raise SystemExit(f"not the same frozen run apart from gold timings: {checks}; report the hosts separately")
    (touched_a, finished_a), (touched_b, finished_b) = touched(a), touched(b)
    both = sorted(touched_a & touched_b)
    if both:
        raise SystemExit(f"attempted on both hosts: {both}")
    shutil.copytree(a, out)  # refuses an existing out
    moved = {}
    for condition in sorted(finished_b):
        folder, target = b / "samples" / condition, out / "samples" / condition
        manifest, result = read_json(folder / "manifest.json"), read_json(folder / "result.json")
        if manifest["run"] != fb["identity"] or result["identity"] != manifest["identity"] \
                or sample_identity(fb, tasks, condition, manifest) != manifest["identity"]:
            raise SystemExit(f"{condition}: run B's sampling records do not check out")
        if digest(folder / "rows.jsonl") != result["rows_sha256"]:
            raise SystemExit(f"{condition}: rows differ from run B's result")
        shutil.copytree(folder, target)
        identity = sample_identity(fa, tasks, condition, manifest)
        atomic_json(target / "manifest.json", {**manifest, "run": fa["identity"], "identity": identity})
        atomic_json(target / "result.json", {**result, "identity": identity})
        moved[condition] = {"identity_on_b": manifest["identity"], "rows_sha256": result["rows_sha256"]}
    record = {"run_a": fa["identity"], "run_b": fb["identity"], "checks": checks, "from_a": sorted(finished_a),
              "from_b": moved, "unfinished": {"a": sorted(touched_a - finished_a), "b": sorted(touched_b - finished_b)}}
    atomic_json(out / "merged.json", record)
    return record


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    result = merge(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"run A {result['run_a'][:12]} + run B {result['run_b'][:12]}: from A {result['from_a']}, "
          f"from B {sorted(result['from_b'])}, unfinished {result['unfinished']}")
