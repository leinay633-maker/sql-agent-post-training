#!/usr/bin/env python3
"""Pack the v0.8 upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/          the v0.5 run's manifest and task files (v0.8 re-verifies them on the host)
  imports/               v0.5 sft_R, pairs, round-1 outcomes and the R/P adapters, as v0.5 recorded them
  BUNDLE.json            sha256 of every file, the commit and the source run identity

python scripts/v08_bundle.py --v05-run <v0.5 run dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROLES = ("pool", "dev", "validation", "test")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v05-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise SystemExit("Commit first: the bundle is built from HEAD")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    v05 = args.v05_run.resolve(strict=True)
    sys.path.insert(0, str(repo))
    from exec_feedback.storage import fingerprint

    manifest = json.loads((v05 / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest.pop("identity")
    if fingerprint(manifest) != identity:
        raise SystemExit("v0.5 manifest identity mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        source = stage / "exec-feedback-agent"
        source.mkdir()
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        reg = stage / "registration"
        reg.mkdir()
        shutil.copy2(v05 / "manifest.json", reg / "manifest.json")
        for role in ROLES:
            shutil.copy2(v05 / f"{role}.jsonl", reg / f"{role}.jsonl")
        imports = stage / "imports"
        imports.mkdir()
        round1 = v05 / "round1"
        for name in ("sft_R.jsonl", "selection.json", "pairs.jsonl", "pairs.json"):
            shutil.copy2(round1 / name, imports / name)
        tasks: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        sample_files = {}
        for path in sorted((round1 / "samples").glob("shard*.jsonl")):
            sample_files[path.name] = sha256(path)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        tasks[row["task_id"]][0] += 1
                        tasks[row["task_id"]][1] += int(bool(row["correct"]))
        (imports / "student_round1.json").write_text(json.dumps(
            {"run": identity, "sample_files_sha256": sample_files, "trajectories": sum(n for n, _ in tasks.values()),
             "tasks": dict(sorted(tasks.items()))}, indent=0), encoding="utf-8", newline="\n")
        for variant in ("R", "P"):
            for seed in SEEDS:
                src = round1 / f"train_{variant}_seed{seed}"
                dst = imports / "adapters" / f"{variant}_seed{seed}"
                dst.mkdir(parents=True)
                shutil.copy2(src / "result.json", dst / "result.json")
                shutil.copy2(src / "manifest.json", dst / "manifest.json")
                shutil.copytree(src / "adapter", dst / "adapter")
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = {"format": "efa-v08-bundle-v1", "bundle": f"v08-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v05_run": identity, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-v08-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
