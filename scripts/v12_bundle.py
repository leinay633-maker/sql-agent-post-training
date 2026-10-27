#!/usr/bin/env python3
"""Pack the v0.12 upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/          the v0.5 task registration, as shipped in the v0.11 bundle
  imports/v08/manifest.json   v0.8's run manifest, as shipped in the v0.11 bundle
  imports/adapters/      v0.8's TR_seed42/43/44 and PTR_seed42/43/44 (adapter/, result.json, manifest.json), from
                         v0.8's own adapter export
  imports/v08/eval-rows/ v0.8's per-task outcomes of R_seed42/43/44 (dev, validation, test) and of the teacher B235 (dev,
                         validation), compacted from v0.8's final results: the cross-night references
  BUNDLE.json            sha256 of every file, the commit, the v0.8 run identity

The imports are checked here as the H20 freeze will check them (models_manifest, verify_v08, reference_rows, both
configs), and further: every compacted evaluation is complete, matches its summary's episode hash and covers exactly its
registered tasks; each imported adapter is the one v0.8 evaluated on test. A mismatch fails here, not on the host.

python scripts/v12_bundle.py --v11-bundle <efa-v11-bundle-*.tgz> --v08-adapters <v08-adapters-*.tar>
                             --v08-results <v08-results-*-final.tgz> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

SEEDS = (42, 43, 44)
V08_ROOT = "outputs/v08/run"
# the cross-night references (docs/RESEARCH_V12.md): R is not re-evaluated, the teacher was never evaluated on test
REFERENCES = [(role, f"R_seed{seed}") for role in ("dev", "validation", "test") for seed in SEEDS] \
    + [(role, "B235") for role in ("dev", "validation")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def inside(stage: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts:
        raise SystemExit(f"Unsafe archive member: {name}")
    return stage.joinpath(*parts)


def copy_members(archive: Path, mapping: dict[str, str], stage: Path) -> int:
    """Regular files under each source prefix, written below stage under the target prefix."""
    count = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            for source, target in mapping.items():
                if member.name.startswith(source):
                    path = inside(stage, target + member.name[len(source):])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src:
                        path.write_bytes(src.read())
                    count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11-bundle", type=Path, required=True)
    ap.add_argument("--v08-adapters", type=Path, required=True, help="v0.8's exported adapters (v08-adapters-*.tar)")
    ap.add_argument("--v08-results", type=Path, required=True, help="v0.8's final results (v08-results-*-final.tgz)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise SystemExit("Commit first: the bundle is built from HEAD")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    sys.path.insert(0, str(repo))
    from exec_feedback.core import digest, read_jsonl
    from exec_feedback.storage import fingerprint, read_json, recover_records
    from exec_feedback.v8.freeze import expected_identity, model_spec
    from exec_feedback.v12 import ARMS, MODELS, TEACHER, V08_PROTOCOL, V08_RUN
    from exec_feedback.v12.config import load_settings
    from exec_feedback.v12.freeze import models_manifest, reference_rows, verify_v08

    manifest = models_manifest(repo / "configs" / "v12-models.json")
    for config in ("h20-v12.json", "v12-smoke.json"):
        load_settings(repo / "configs" / config)
    spec = model_spec("student", Path("/models/student"), {"max_context": 8192, "max_tokens": 512})
    snapshot = expected_identity(spec, manifest[MODELS["student"][0]]["files"])["local_snapshot"]

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        source = stage / "exec-feedback-agent"
        source.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        if copy_members(args.v11_bundle, {"registration/": "registration/",
                                          "imports/v08/manifest.json": "imports/v08/manifest.json"}, stage) != 6:
            raise SystemExit("The v0.11 bundle should carry registration/ (five files) and imports/v08/manifest.json")
        names = [f"{family}_seed{seed}" for family in ARMS for seed in SEEDS]
        wanted = {f"{V08_ROOT}/train/{name}/{part}": f"imports/adapters/{name}/{part}"
                  for name in names for part in ("adapter/", "result.json", "manifest.json")}
        copy_members(args.v08_adapters, wanted, stage)
        registration = read_json(stage / "registration" / "manifest.json")
        reg_identity = registration.pop("identity")
        if fingerprint(registration) != reg_identity:
            raise SystemExit("Registration manifest identity mismatch")
        imports = stage / "imports"
        arms = verify_v08(imports, registration_identity=reg_identity, student_snapshot=snapshot, seeds=list(SEEDS),
                          v08_run=V08_RUN)
        if sorted(arms["adapters"]) != sorted(names):
            raise SystemExit(f"Expected the adapters {names}, verified {sorted(arms['adapters'])}")

        # v0.8's evaluations: the references' rows, and the test summaries that name each imported adapter's hash
        evals = {f"{V08_ROOT}/eval/{role}/{c}/{f}": (role, c, f) for role, c in REFERENCES
                 for f in ("summary.json", "episodes.jsonl")}
        evals.update({f"{V08_ROOT}/eval/test/{name}/summary.json": ("test", name, "summary.json") for name in names})
        found: dict[tuple[str, str, str], Path] = {}
        scratch = Path(tmp) / "v08-eval"
        with tarfile.open(args.v08_results, "r:*") as tar:
            for member in tar:
                if member.isfile() and member.name in evals:
                    key = evals[member.name]
                    found[key] = inside(scratch, "/".join(key))
                    found[key].parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src:
                        found[key].write_bytes(src.read())
        if len(found) != len(evals):
            raise SystemExit(f"v0.8's results lack {sorted(set(evals.values()) - set(found))}")
        for name in names:
            summary = read_json(found["test", name, "summary.json"])
            if summary["run"] != V08_RUN or summary.get("adapter_hash") != arms["adapters"][name]:
                raise SystemExit(f"{name} is not the adapter v0.8 evaluated on test")
        teacher_id, teacher_revision = MODELS["teacher"]
        for role, condition in REFERENCES:
            summary = read_json(found[role, condition, "summary.json"])
            episodes = found[role, condition, "episodes.jsonl"]
            if summary["run"] != V08_RUN or summary["protocol"] != V08_PROTOCOL or summary["role"] != role \
                    or summary["condition"] != condition or not summary["complete"] \
                    or digest(episodes) != summary["episodes_sha256"]:
                raise SystemExit(f"v0.8's {role}/{condition} is not a complete evaluation of the registered run")
            model = summary["model"]
            if condition == TEACHER:
                if (model["id"], model["revision"]) != (teacher_id, teacher_revision):
                    raise SystemExit(f"v0.8's {role}/{condition} was not the pinned teacher")
            elif model["local_snapshot"] != snapshot:
                raise SystemExit(f"v0.8's {role}/{condition} ran another 4B snapshot")
            tasks = {row["id"]: row["db_id"] for row in read_jsonl(stage / "registration" / f"{role}.jsonl")}
            rows = [{"task_id": r["task_id"], "db_id": r["db_id"], "correct": r["correct"], "status": r["status"]}
                    for r in recover_records(episodes)]
            ids = [r["task_id"] for r in rows]
            if len(set(ids)) != len(ids) or set(ids) != set(tasks) or len(ids) != summary["registered_tasks"] \
                    or any(tasks[r["task_id"]] != r["db_id"] or type(r["correct"]) is not bool
                           or r["correct"] != (r["status"] == "correct") for r in rows) \
                    or sum(r["correct"] for r in rows) != summary["correct"]:
                raise SystemExit(f"v0.8's {role}/{condition} rows do not cover exactly its registered tasks")
            target = imports / "v08" / "eval-rows" / role / f"{condition}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"run": V08_RUN, "role": role, "condition": condition,
                                          "episodes_sha256": summary["episodes_sha256"], "correct": summary["correct"],
                                          "rows": rows}, indent=0), encoding="utf-8", newline="\n")
        references = reference_rows(imports, V08_RUN)
        if len(references) != len(REFERENCES):
            raise SystemExit(f"reference_rows saw {sorted(references)}")

        files = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = {"format": "efa-v12-bundle-v1", "bundle": f"v12-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v08_run": V08_RUN, "registration": reg_identity, "adapters": arms["adapters"],
                  "references": references, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-v12-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest_ = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest_}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest_, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size, "id": bundle["bundle"],
                      "adapters": sorted(arms["adapters"]),
                      "references": {k: v["tasks"] for k, v in references.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
