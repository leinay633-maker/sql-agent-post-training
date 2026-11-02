#!/usr/bin/env python3
"""Pack the deployment-study upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/          the v0.5 task registration, as shipped in the v0.12 bundle
  imports/v11/           v0.11's run manifest and milestone record; RLb_seed44's run records and its adapter at the
                         registered milestone (from v0.11's final results and adapter package)
  BUNDLE.json            sha256 of every file, the commit, the run identity and the adapter

The imports are checked here exactly as the H20 freeze will check them, so a mismatch fails here, not on the host.

python scripts/deploy_bundle.py --v12-bundle <efa-v12-bundle-*.tgz> --v11-final <v0.11 final>/outputs/v11/run
    --v11-adapters <v11-adapters-*.tar> --out <dir>
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

RUN_FILES = ("manifest.json", "milestones.json", "result.json", "paused.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive: Path, wanted, target: Path) -> int:
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and wanted(member.name):
                tar.extract(member, target, filter="data")
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v12-bundle", type=Path, required=True)
    ap.add_argument("--v11-final", type=Path, required=True, help="the run directory of v0.11's final results")
    ap.add_argument("--v11-adapters", type=Path, required=True)
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
    from exec_feedback.deploy import FAMILY, MODEL, MODELS, V11_PROTOCOL, V11_RUN
    from exec_feedback.deploy.config import SAME_AS_V11, load_settings
    from exec_feedback.passk.freeze import at_registered_milestones, milestone_adapters, verify_run

    with tempfile.TemporaryDirectory() as tmp:
        stage, scratch = Path(tmp) / "stage", Path(tmp) / "scratch"
        source = stage / "exec-feedback-agent"
        source.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        if extract(args.v12_bundle, lambda n: n.startswith("registration/"), stage) != 5:
            raise SystemExit("The v0.12 bundle should carry manifest.json and four task files under registration/")
        final = args.v11_final.resolve(strict=True)
        out = stage / "imports" / "v11"
        out.mkdir(parents=True)
        shutil.copy2(final / "manifest.json", out / "manifest.json")
        shutil.copy2(final / "steps.json", out / "steps.json")
        step = json.loads((final / "steps.json").read_text(encoding="utf-8"))["evaluate"][MODEL]["step"]
        prefix = f"outputs/v11/run/train/{MODEL}/"
        wanted = lambda n: n in {prefix + f for f in RUN_FILES} or n.startswith(f"{prefix}milestones/step-{step:04d}/adapter/")
        if extract(args.v11_adapters, wanted, scratch) < 4:
            raise SystemExit(f"{args.v11_adapters} lacks {MODEL}'s records or its adapter at step {step}")
        target = out / "adapters" / MODEL
        target.mkdir(parents=True)
        for f in RUN_FILES:
            if (scratch / prefix / f).exists():
                shutil.copy2(scratch / prefix / f, target / f)
        shutil.copytree(scratch / prefix / "milestones" / f"step-{step:04d}" / "adapter", target / "adapter")
        registration = json.loads((stage / "registration" / "manifest.json").read_text(encoding="utf-8"))
        reg_identity = registration.pop("identity")
        if fingerprint(registration) != reg_identity:
            raise SystemExit("Registration manifest identity mismatch")
        files = json.loads((repo / "configs" / "v11-models.json").read_text(encoding="utf-8"))[MODELS["student"][0]]["files"]
        snapshot = expected_identity(model_spec("student", Path("/models/student"),
                                                {"max_context": 8192, "max_tokens": 512}), files)["local_snapshot"]
        run = verify_run(stage / "imports", "v11", protocol=V11_PROTOCOL, identity=V11_RUN,
                         registration_identity=reg_identity, student_snapshot=snapshot)
        settings = load_settings(repo / "configs" / "h20-deploy.json")
        differ = [k for k in SAME_AS_V11 if run["settings"].get(k) != settings[k]]
        if differ:
            raise SystemExit(f"The deployment must keep v0.11's agent protocol; differs in {differ}")
        v11 = milestone_adapters(stage / "imports", "v11", run, FAMILY, [44])
        at_registered_milestones("v11", FAMILY, v11)  # as the host's study freeze checks it
        files_ = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files_}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = {"format": "efa-deploy-bundle-v1", "bundle": f"deploy-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v11_run": V11_RUN, "registration": reg_identity, "adapter": v11["adapters"][MODEL],
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-deploy-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files_, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size, "adapter": bundle["adapter"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
