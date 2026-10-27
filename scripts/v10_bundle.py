#!/usr/bin/env python3
"""Pack the v0.10 upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/          the v0.5 task registration          } as shipped in the v0.9 bundle
  imports/v08/, imports/adapters/   v0.8's manifest and PTR_seed42/43/44 (the start points)
  imports/v09/           v0.9's run manifest and milestone record (steps.json: K and the step evaluated per run), from its
                         final results package; RL_seed42/43/44 at that step (adapter, result, manifest, milestones) from
                         its adapters package; the RL runs' metrics and curves (v0.10's report sets them beside its own)
  BUNDLE.json            sha256 of every file, the commit, the v0.8 and v0.9 run identities and K

The imports are checked here exactly as the H20 freeze will check them (verify_v08 and verify_v09 against the registered
configs/h20-v10.json), so a K that differs from planned_steps fails here, not on the host.

python scripts/v10_bundle.py --v09-bundle <efa-v09-bundle-*.tgz> --v09-results <v09-results-*-final.tgz>
                             --v09-adapters <v09-adapters-*.tar> --out <dir>
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
V09_RUN_DIR = "outputs/v09/run/"


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
    ap.add_argument("--v09-bundle", type=Path, required=True)
    ap.add_argument("--v09-results", type=Path, required=True)
    ap.add_argument("--v09-adapters", type=Path, required=True)
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
    from exec_feedback.v9.freeze import verify_imports as verify_v08
    from exec_feedback.v10 import LORA, MODELS, V08_RUN, V09_RUN
    from exec_feedback.v10.config import load_settings
    from exec_feedback.v10.freeze import verify_v09

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        source = stage / "exec-feedback-agent"
        source.mkdir()
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        if extract(args.v09_bundle, "registration/", "", stage) != 5:
            raise SystemExit("The v0.9 bundle should carry manifest.json and four task files under registration/")
        if extract(args.v09_bundle, "imports/", "", stage) < 1 + 3 * len(SEEDS):
            raise SystemExit("The v0.9 bundle should carry imports/v08/manifest.json and the PTR adapters")
        v09 = stage / "imports" / "v09"
        for name in ("manifest.json", "steps.json"):
            if extract(args.v09_results, V09_RUN_DIR + name, V09_RUN_DIR, v09) != 1:
                raise SystemExit(f"No {V09_RUN_DIR}{name} in the v0.9 results package (the final one has steps.json)")
        steps = json.loads((v09 / "steps.json").read_text(encoding="utf-8"))
        for seed in SEEDS:
            name = f"{LORA}_seed{seed}"
            run_dir = f"{V09_RUN_DIR}train/{name}/"
            for record in ("metrics.jsonl", "curve.jsonl", "result.json", "stopped.json"):
                extract(args.v09_results, run_dir + record, run_dir, v09 / "train" / name)
            chosen = steps["evaluate"].get(name)
            if chosen is None:
                raise SystemExit(f"v0.9 registered no milestone for {name}")
            milestone = f"{run_dir}milestones/step-{chosen['step']:04d}/"
            if extract(args.v09_adapters, milestone + "adapter/", milestone, v09 / "adapters" / name) < 2:
                raise SystemExit(f"{name} at step {chosen['step']} is missing from the v0.9 adapters package")
            for record in ("result.json", "manifest.json", "milestones.json"):
                if extract(args.v09_adapters, run_dir + record, run_dir, v09 / "adapters" / name) != 1:
                    raise SystemExit(f"{name}/{record} is missing from the v0.9 adapters package")
        registration = json.loads((stage / "registration" / "manifest.json").read_text(encoding="utf-8"))
        reg_identity = registration.pop("identity")
        if fingerprint(registration) != reg_identity:
            raise SystemExit("Registration manifest identity mismatch")
        files = json.loads((repo / "configs" / "v10-models.json").read_text(encoding="utf-8"))[MODELS["student"][0]]["files"]
        spec = model_spec("student", Path("/models/student"), {"max_context": 8192, "max_tokens": 512})
        snapshot = expected_identity(spec, files)["local_snapshot"]
        imports = stage / "imports"
        starts = verify_v08(imports, registration_identity=reg_identity, student_snapshot=snapshot, seeds=list(SEEDS),
                            v08_run=V08_RUN)
        settings = load_settings(repo / "configs" / "h20-v10.json")
        arm = verify_v09(imports, registration_identity=reg_identity, student_snapshot=snapshot, settings=settings,
                         v09_run=V09_RUN, v08_run=V08_RUN)
        files_ = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files_}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        bundle = {"format": "efa-v10-bundle-v1", "bundle": f"v10-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v08_run": V08_RUN, "v09_run": V09_RUN, "k": arm["k"], "registration": reg_identity,
                  "start_adapters": starts["adapters"], "rl": arm["rl"],
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-v10-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files_, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size, "k": arm["k"], "rl": arm["rl"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
