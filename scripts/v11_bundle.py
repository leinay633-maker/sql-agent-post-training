#!/usr/bin/env python3
"""Pack the v0.11 upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/, imports/   the v0.5 task registration; v0.8's manifest and PTR_seed42/43/44 (the start points); v0.9's
                         run manifest, milestone record, RL_seed42/43/44 at K and their metrics and curves (the RL9 arm):
                         exactly as shipped in the v0.10 bundle
  v10-final/             from v0.10's final results, what the 4% replay reads: the run manifest, RLF_seed42's manifest
                         and kept benchmark items, the layouts' measurements (bench/z*.json)
  BUNDLE.json            sha256 of every file, the commit, the v0.8, v0.9 and v0.10 run identities and K

The imports are checked here exactly as the H20 freeze will check them (verify_v08, verify_v09 and same_training against
the registered configs/h20-v11.json), so a mismatch fails here, not on the host.

python scripts/v11_bundle.py --v10-bundle <efa-v10-bundle-*.tgz> --v10-final <v0.10 final>/outputs/v10/run --out <dir>
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
from pathlib import Path

SEEDS = (42, 43, 44)
REPLAY = "RLF_seed42"  # the v0.10 run whose first kept step the 4% replay splits


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive: Path, prefix: str, target: Path) -> int:
    """Members under `prefix`, written below `target` with their names unchanged."""
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith(prefix):
                tar.extract(member, target, filter="data")
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10-bundle", type=Path, required=True)
    ap.add_argument("--v10-final", type=Path, required=True, help="the run directory of v0.10's final results")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise SystemExit("Commit first: the bundle is built from HEAD")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    sys.path.insert(0, str(repo))
    from exec_feedback.storage import fingerprint, read_json
    from exec_feedback.v8.freeze import expected_identity, model_spec
    from exec_feedback.v9.freeze import verify_imports as verify_v08
    from exec_feedback.v10.freeze import load as load_v10
    from exec_feedback.v10.freeze import verify_v09
    from exec_feedback.v11 import MODELS, V08_RUN, V09_RUN
    from exec_feedback.v11.config import load_settings
    from exec_feedback.v11.freeze import same_training

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        source = stage / "exec-feedback-agent"
        source.mkdir()
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        if extract(args.v10_bundle, "registration/", stage) != 5:
            raise SystemExit("The v0.10 bundle should carry manifest.json and four task files under registration/")
        if extract(args.v10_bundle, "imports/", stage) < 1 + 3 * len(SEEDS):
            raise SystemExit("The v0.10 bundle should carry imports/ (v0.8, the PTR adapters, v0.9)")
        # the 4% replay's inputs from v0.10's final results, checked against that run's own manifest
        final, replay = args.v10_final.resolve(strict=True), stage / "v10-final"
        v10, _ = load_v10(final, check_code=False)
        wanted = [final / "manifest.json", final / "train" / REPLAY / "manifest.json",
                  *sorted((final / "train" / REPLAY / "bench-items").glob("step-*.json")),
                  *sorted((final / "bench").glob("z*.json"))]
        if not any(p.parent.name == "bench-items" for p in wanted) or not (final / "bench" / "z3-4.json").exists() \
                or not (final / "bench" / "z3-8.json").exists():
            raise SystemExit(f"{final} lacks {REPLAY}'s benchmark items or the 4- and 8-rank measurements")
        if read_json(final / "train" / REPLAY / "manifest.json")["run"] != v10["identity"]:
            raise SystemExit(f"{REPLAY} belongs to another v0.10 run")
        for path in wanted:
            target = replay / path.relative_to(final)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        registration = json.loads((stage / "registration" / "manifest.json").read_text(encoding="utf-8"))
        reg_identity = registration.pop("identity")
        if fingerprint(registration) != reg_identity:
            raise SystemExit("Registration manifest identity mismatch")
        files = json.loads((repo / "configs" / "v11-models.json").read_text(encoding="utf-8"))[MODELS["student"][0]]["files"]
        spec = model_spec("student", Path("/models/student"), {"max_context": 8192, "max_tokens": 512})
        snapshot = expected_identity(spec, files)["local_snapshot"]
        imports = stage / "imports"
        starts = verify_v08(imports, registration_identity=reg_identity, student_snapshot=snapshot, seeds=list(SEEDS),
                            v08_run=V08_RUN)
        settings = load_settings(repo / "configs" / "h20-v11.json")
        arm = verify_v09(imports, registration_identity=reg_identity, student_snapshot=snapshot, settings=settings,
                         v09_run=V09_RUN, v08_run=V08_RUN)
        same_training(imports, settings)
        if v10["registration"]["source_identity"] != reg_identity:
            raise SystemExit("v0.10's final run used another task registration")
        files_ = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files_}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = {"format": "efa-v11-bundle-v1", "bundle": f"v11-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v08_run": V08_RUN, "v09_run": V09_RUN, "v10_run": v10["identity"], "k": arm["k"],
                  "registration": reg_identity, "start_adapters": starts["adapters"], "rl": arm["rl"],
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-v11-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files_, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size, "k": arm["k"], "rl": arm["rl"],
                      "v10_run": v10["identity"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
