#!/usr/bin/env python3
"""Pack the v0.9 upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/          the v0.5 task registration, as shipped in the v0.8 bundle
  imports/v08/           the v0.8 run manifest (from its final results package)
  imports/adapters/      v0.8's PTR_seed42/43/44 (result, manifest, adapter), from its adapters package
  BUNDLE.json            sha256 of every file, the commit and the v0.8 run identity

The imports are checked here exactly as the H20 freeze will check them (v0.9's verify_imports).

python scripts/v09_bundle.py --v08-bundle <efa-v08-bundle-*.tgz> --v08-results <v08-results-*-final.tgz>
                             --v08-adapters <v08-adapters-*.tar> --out <dir>
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
from pathlib import Path

SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive: Path, prefix: str, strip: str, target: Path) -> int:
    """Members under `prefix`, written below `target` with `strip` removed from their names."""
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.startswith(prefix):
                member.name = member.name[len(strip):]
                tar.extract(member, target, filter="data")
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v08-bundle", type=Path, required=True)
    ap.add_argument("--v08-results", type=Path, required=True)
    ap.add_argument("--v08-adapters", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True,
                      text=True, check=True).stdout.strip():
        raise SystemExit("Commit first: the bundle is built from HEAD")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    sys.path.insert(0, str(repo))
    from exec_feedback.storage import fingerprint
    from exec_feedback.v8.freeze import expected_identity, model_spec
    from exec_feedback.v9 import MODELS, V08_RUN
    from exec_feedback.v9.freeze import verify_imports

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        source = stage / "exec-feedback-agent"
        source.mkdir()
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        if extract(args.v08_bundle, "registration/", "", stage) != 5:
            raise SystemExit("The v0.8 bundle should carry manifest.json and four task files under registration/")
        imports = stage / "imports"
        if extract(args.v08_results, "outputs/v08/run/manifest.json", "outputs/v08/run/", imports / "v08") != 1:
            raise SystemExit("No outputs/v08/run/manifest.json in the v0.8 results package")
        for seed in SEEDS:
            prefix = f"outputs/v08/run/train/PTR_seed{seed}/"
            if extract(args.v08_adapters, prefix, "outputs/v08/run/train/", imports / "adapters") < 3:
                raise SystemExit(f"PTR_seed{seed} missing from the v0.8 adapters package")
        registration = json.loads((stage / "registration" / "manifest.json").read_text(encoding="utf-8"))
        reg_identity = registration.pop("identity")
        if fingerprint(registration) != reg_identity:
            raise SystemExit("Registration manifest identity mismatch")
        files = json.loads((repo / "configs" / "v09-models.json").read_text(encoding="utf-8"))[MODELS["student"][0]]["files"]
        spec = model_spec("student", Path("/models/student"), {"max_context": 8192, "max_tokens": 512})
        verified = verify_imports(imports, registration_identity=reg_identity,
                                  student_snapshot=expected_identity(spec, files)["local_snapshot"], seeds=list(SEEDS))
        if verified["v08_run"] != V08_RUN:
            raise SystemExit("Imported v0.8 run is not the registered one")
        files_ = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files_}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = {"format": "efa-v09-bundle-v1", "bundle": f"v09-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v08_run": V08_RUN, "registration": reg_identity, "start_adapters": verified["adapters"],
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-v09-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files_, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size, "start_adapters": verified["adapters"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
