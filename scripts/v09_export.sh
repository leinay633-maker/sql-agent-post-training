#!/usr/bin/env bash
# Pack v0.9 results: bash scripts/v09_export.sh <label> [adapters]
# Results (always): run + smoke directories without checkpoints, policy/milestone adapters and imports; logs, environment.
# Adapters (final only): each online run's milestone at its evaluated step and its last milestone.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(dirname "$REPO")
EXPORTS=$HERE/exports
mkdir -p "$EXPORTS"
cd "$REPO" || exit 1
stamp=$(TZ=CST-8 date +%m%d-%H%M)
list=$(mktemp)
for run in outputs/v09/run outputs/v09/smoke; do
  [[ -d $run ]] && find "$run" -type f ! -path '*/ckpt*' ! -path '*/policy/*' ! -path '*/milestones/*' \
    ! -path '*/imports/*' ! -name '*.lock' ! -name '*.tmp*' >> "$list"
done
find outputs/logs outputs/environment -type f 2>/dev/null >> "$list"
cp "$HERE/STATUS.txt" "$EXPORTS/STATUS.txt" 2>/dev/null
name=v09-results-$stamp-$1.tgz
tar -czf "$EXPORTS/$name.part" -T "$list" && mv "$EXPORTS/$name.part" "$EXPORTS/$name" \
  && (cd "$EXPORTS" && sha256sum "$name" > "$name.sha256")
ls -t "$EXPORTS"/v09-results-*.tgz 2>/dev/null | tail -n +4 | while read -r old; do rm -f "$old" "$old.sha256"; done
if [[ ${2:-} == adapters ]]; then
  python3 - "$REPO/outputs/v09/run" > "$list" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
chosen = json.loads((run / "steps.json").read_text())["evaluate"] if (run / "steps.json").exists() else {}
for folder in sorted((run / "train").glob("*_seed*")):
    marks = json.loads((folder / "milestones.json").read_text()) if (folder / "milestones.json").exists() else {}
    steps = {int(s) for s in marks} and {max(int(s) for s in marks)} | ({chosen[folder.name]["step"]} if folder.name in chosen else set())
    for step in sorted(steps or ()):
        for f in sorted((folder / "milestones" / f"step-{step:04d}").rglob("*")):
            if f.is_file():
                print(f.relative_to(run.parents[2]))
    for name in ("result.json", "manifest.json", "milestones.json"):
        if (folder / name).exists():
            print((folder / name).relative_to(run.parents[2]))
PY
  [[ -s $list ]] && tar -cf "$EXPORTS/v09-adapters-$stamp.tar.part" -T "$list" \
    && mv "$EXPORTS/v09-adapters-$stamp.tar.part" "$EXPORTS/v09-adapters-$stamp.tar" \
    && (cd "$EXPORTS" && sha256sum "v09-adapters-$stamp.tar" > "v09-adapters-$stamp.tar.sha256")
fi
rm -f "$list"
echo "$EXPORTS/$name"
