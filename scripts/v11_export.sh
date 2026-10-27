#!/usr/bin/env bash
# Pack v0.11 results: bash scripts/v11_export.sh <label> [adapters]
# Results (always): run + smoke directories without checkpoints, policy/milestone adapters and imports (pre-pass rows,
# online metrics and rollouts, dev curves, evaluations, reports); the 4% replay; logs, environment.
# Adapters (final only): each boundary run's milestone at K, at explore_steps and its last milestone.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(dirname "$REPO")
EXPORTS=$HERE/exports
mkdir -p "$EXPORTS"
cd "$REPO" || exit 1
stamp=$(TZ=CST-8 date +%m%d-%H%M)
list=$(mktemp)
for run in outputs/v11/run outputs/v11/smoke outputs/v11/regroup; do
  [[ -d $run ]] && find "$run" -type f ! -path '*/ckpt*' ! -path '*/policy/*' ! -path '*/milestones/*' \
    ! -path '*/imports/*' ! -name '*.safetensors' ! -name '*.lock' ! -name '*.tmp*' >> "$list"
done
find outputs/logs outputs/environment -type f 2>/dev/null >> "$list"
cp "$HERE/STATUS.txt" "$EXPORTS/STATUS.txt" 2>/dev/null
name=v11-results-$stamp-$1.tgz
# exit 1 only says a file grew while it was read (metrics and logs during a run; the archive is still complete), and a
# file removed since the listing is skipped; anything else, or a failed rename or hash, publishes nothing
tar --warning=no-file-changed --ignore-failed-read -czf "$EXPORTS/$name.part" -T "$list"
rc=$?
if (( rc > 1 )) || ! mv "$EXPORTS/$name.part" "$EXPORTS/$name" || ! (cd "$EXPORTS" && sha256sum "$name" > "$name.sha256"); then
  rm -f "$EXPORTS/$name.part" "$list"; echo "packing $name failed (tar exit $rc)" >&2; exit 1
fi
ls -t "$EXPORTS"/v11-results-*.tgz 2>/dev/null | tail -n +4 | while read -r old; do rm -f "$old" "$old.sha256"; done
if [[ ${2:-} == adapters ]]; then
  # a run that stopped before its first milestone has none to pack; one that fails here must not pass for "no adapters"
  python3 - "$REPO/outputs/v11/run" > "$list" <<'PY' || { rm -f "$list"; echo "listing the adapters failed" >&2; exit 1; }
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
settings = json.loads((run / "manifest.json").read_text())["settings"]
chosen = json.loads((run / "steps.json").read_text())["evaluate"] if (run / "steps.json").exists() else {}
for folder in sorted((run / "train").glob("*_seed*")):
    marks = {int(s) for s in json.loads((folder / "milestones.json").read_text())} \
        if (folder / "milestones.json").exists() else set()
    wanted = {settings["explore_steps"]} | ({max(marks)} if marks else set()) \
        | ({chosen[folder.name]["step"]} if folder.name in chosen else set())
    for step in sorted(marks & wanted):
        for f in sorted((folder / "milestones" / f"step-{step:04d}").rglob("*")):
            if f.is_file():
                print(f.relative_to(run.parents[2]))
    for name in ("result.json", "paused.json", "manifest.json", "milestones.json"):
        if (folder / name).exists():
            print((folder / name).relative_to(run.parents[2]))
PY
  if [[ -s $list ]] && ! { tar -cf "$EXPORTS/v11-adapters-$stamp.tar.part" -T "$list" \
      && mv "$EXPORTS/v11-adapters-$stamp.tar.part" "$EXPORTS/v11-adapters-$stamp.tar" \
      && (cd "$EXPORTS" && sha256sum "v11-adapters-$stamp.tar" > "v11-adapters-$stamp.tar.sha256"); }; then
    rm -f "$EXPORTS/v11-adapters-$stamp.tar.part" "$list"; echo "packing the adapters failed" >&2; exit 1
  fi
fi
rm -f "$list"
echo "$EXPORTS/$name"
