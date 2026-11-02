#!/usr/bin/env python3
"""Pack the pass@k upload bundle on the local machine (no GPU, no credentials).

  exec-feedback-agent/   committed source of HEAD (git archive), so the bundle matches a commit
  registration/          the v0.5 task registration, as shipped in the v0.12 bundle
  imports/v08/, imports/adapters/PTR_seed*/   v0.8's manifest and the PTR adapters, as shipped in the v0.12 bundle
  imports/v11/           v0.11's run manifest, milestone record and, per seed, RLb's run records and its adapter at the
                         registered milestone (from v0.11's final results and adapter package)
  imports/v12/           the same for v0.12's OPD
  imports/greedy.json    each model's greedy test score from its own run's evaluation (descriptive)
  BUNDLE.json            sha256 of every file, the commit, the run identities

The imports are checked here exactly as the H20 freeze will check them (verify_imports against the registered
configs/h20-passk.json), so a mismatch fails here, not on the host.

python scripts/passk_bundle.py --v12-bundle <efa-v12-bundle-*.tgz> --v11-final <v0.11 final>/outputs/v11/run
    --v11-adapters <v11-adapters-*.tar> --v12-final <v0.12 final>/outputs/v12/run --v12-adapters <v12-adapters-*.tar>
    --out <dir>
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
RUN_FILES = ("manifest.json", "milestones.json", "result.json", "paused.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive: Path, wanted, target: Path) -> int:
    """Members whose name `wanted` accepts, written below `target` with their names unchanged."""
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.isfile() and wanted(member.name):
                tar.extract(member, target, filter="data")
                count += 1
    return count


def milestone_run(final: Path, adapters: Path, name: str, family: str, stage: Path, scratch: Path) -> dict:
    """A run's manifest and milestone record from its final results; per seed, its run records and the adapter at the
    registered milestone from its adapter package."""
    manifest, steps = json.loads((final / "manifest.json").read_text(encoding="utf-8")), \
        json.loads((final / "steps.json").read_text(encoding="utf-8"))
    out = stage / "imports" / name
    out.mkdir(parents=True)
    shutil.copy2(final / "manifest.json", out / "manifest.json")
    shutil.copy2(final / "steps.json", out / "steps.json")
    prefix = f"outputs/{name}/run/train/"
    chosen = {f"{family}_seed{s}": steps["evaluate"][f"{family}_seed{s}"]["step"] for s in SEEDS}
    wanted = lambda n: n.startswith(prefix) and any(
        n == f"{prefix}{c}/{f}" or n.startswith(f"{prefix}{c}/milestones/step-{k:04d}/adapter/")
        for c, k in chosen.items() for f in RUN_FILES)
    if extract(adapters, wanted, scratch) < len(SEEDS) * 4:
        raise SystemExit(f"{adapters} lacks the {family} runs' records or adapters at the registered milestones")
    for condition, step in chosen.items():
        source, target = scratch / prefix / condition, out / "adapters" / condition
        target.mkdir(parents=True)
        for f in RUN_FILES:
            if (source / f).exists():
                shutil.copy2(source / f, target / f)
        shutil.copytree(source / "milestones" / f"step-{step:04d}" / "adapter", target / "adapter")
    return {"identity": manifest["identity"], "k": steps["k"], "chosen": chosen}


def greedy(final: Path, run: dict, names: list[str]) -> dict:
    """Each condition's complete greedy test evaluation in that run (the registered milestone for trained runs)."""
    out = {}
    for condition in names:
        summary = json.loads((final / "eval" / "test" / condition / "summary.json").read_text(encoding="utf-8"))
        if summary["run"] != run["identity"] or not summary["complete"] or summary["condition"] != condition \
                or summary["step"] != run["chosen"].get(condition, summary["step"]):
            raise SystemExit(f"{final}: the test evaluation of {condition} is not a complete one of that run")
        out[condition] = {"run": summary["run"], "tasks": summary["registered_tasks"], "correct": summary["correct"],
                          "step": summary["step"], "episodes_sha256": summary["episodes_sha256"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v12-bundle", type=Path, required=True)
    ap.add_argument("--v11-final", type=Path, required=True, help="the run directory of v0.11's final results")
    ap.add_argument("--v11-adapters", type=Path, required=True)
    ap.add_argument("--v12-final", type=Path, required=True, help="the run directory of v0.12's final results")
    ap.add_argument("--v12-adapters", type=Path, required=True)
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
    from exec_feedback.passk import MODELS, V08_RUN, V11_RUN, V12_RUN
    from exec_feedback.passk.config import load_settings
    from exec_feedback.passk.freeze import verify_imports

    with tempfile.TemporaryDirectory() as tmp:
        stage, scratch = Path(tmp) / "stage", Path(tmp) / "scratch"
        source = stage / "exec-feedback-agent"
        source.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", "--format=tar", commit], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
        if extract(args.v12_bundle, lambda n: n.startswith("registration/"), stage) != 5:
            raise SystemExit("The v0.12 bundle should carry manifest.json and four task files under registration/")
        ptr = lambda n: n == "imports/v08/manifest.json" or any(n.startswith(f"imports/adapters/PTR_seed{s}/")
                                                               for s in SEEDS)
        if extract(args.v12_bundle, ptr, stage) < 1 + 3 * len(SEEDS):
            raise SystemExit("The v0.12 bundle should carry v0.8's manifest and the PTR adapters under imports/")
        v11 = milestone_run(args.v11_final.resolve(strict=True), args.v11_adapters, "v11", "RLb", stage, scratch)
        v12 = milestone_run(args.v12_final.resolve(strict=True), args.v12_adapters, "v12", "OPD", stage, scratch)
        reference = {**greedy(args.v11_final, v11, ["B", *(f"{f}_seed{s}" for f in ("PTR", "RLb") for s in SEEDS)]),
                     **greedy(args.v12_final, v12, [f"OPD_seed{s}" for s in SEEDS])}
        (stage / "imports" / "greedy.json").write_text(json.dumps({"conditions": reference}, indent=1),
                                                       encoding="utf-8", newline="\n")
        registration = json.loads((stage / "registration" / "manifest.json").read_text(encoding="utf-8"))
        reg_identity = registration.pop("identity")
        if fingerprint(registration) != reg_identity:
            raise SystemExit("Registration manifest identity mismatch")
        test_count = sum(1 for line in (stage / "registration" / "test.jsonl").read_text(encoding="utf-8").splitlines()
                         if line.strip())
        files = json.loads((repo / "configs" / "v11-models.json").read_text(encoding="utf-8"))[MODELS["student"][0]]["files"]
        spec = model_spec("student", Path("/models/student"), {"max_context": 8192, "max_tokens": 512})
        snapshot = expected_identity(spec, files)["local_snapshot"]
        checked = verify_imports(stage / "imports", registration_identity=reg_identity, student_snapshot=snapshot,
                                 settings=load_settings(repo / "configs" / "h20-passk.json"), registered=test_count)
        files_ = sorted(p for p in stage.rglob("*") if p.is_file())
        hashes = {p.relative_to(stage).as_posix(): sha256(p) for p in files_}
        tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        adapters = {**checked["adapters"], **{c: a["adapter_hash"] for run in ("v11", "v12")
                                              for c, a in checked[run]["adapters"].items()}}
        bundle = {"format": "efa-passk-bundle-v1", "bundle": f"passk-{commit[:8]}-{tree[:8]}", "commit": commit,
                  "v08_run": V08_RUN, "v11_run": V11_RUN, "v12_run": V12_RUN, "k": {"v11": v11["k"], "v12": v12["k"]},
                  "registration": reg_identity, "test_tasks": test_count, "adapters": adapters,
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "credentials_included": False, "files": hashes}
        (stage / "BUNDLE.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8", newline="\n")
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"efa-passk-bundle-{commit[:8]}.tgz"
        with tarfile.open(args.out / name, "w:gz", compresslevel=6) as tar:
            for path in [*files_, stage / "BUNDLE.json"]:
                tar.add(path, arcname=path.relative_to(stage).as_posix())
    digest = sha256(args.out / name)
    (args.out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle": str(args.out / name), "sha256": digest, "files": len(hashes), "commit": commit,
                      "bytes": (args.out / name).stat().st_size, "k": bundle["k"], "adapters": adapters,
                      "greedy": {c: g["correct"] for c, g in reference.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
