#!/usr/bin/env bash
# Pack v0.10 results: bash scripts/v10_export.sh <label>
# Run + smoke directories without weights (checkpoints, milestones, merged models, the weight channel, scratch) and
# imports; logs, environment. The models at K travel separately: `python -m exec_feedback.v10 pack`, run by the scheduler
# as each main run finishes, leaves v10-model-*.tar.part-* and a manifest beside these packages.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(dirname "$REPO")
EXPORTS=$HERE/exports
mkdir -p "$EXPORTS"
cd "$REPO" || exit 1
stamp=$(TZ=CST-8 date +%m%d-%H%M)
list=$(mktemp)
for run in outputs/v10/run outputs/v10/smoke; do
  [[ -d $run ]] && find "$run" -type f ! -path '*/ckpt*' ! -path '*/milestones/*' ! -path '*/imports/*' \
    ! -path '*/scratch/*' ! -path '*/channel/*' ! -name '*.safetensors' ! -name '*.lock' ! -name '*.tmp*' >> "$list"
done
find outputs/logs outputs/environment -type f 2>/dev/null >> "$list"
cp "$HERE/STATUS.txt" "$EXPORTS/STATUS.txt" 2>/dev/null
name=v10-results-$stamp-$1.tgz
# exit 1 only says a file grew while it was read (metrics and logs during a run; the archive is still complete), and a
# file removed since the listing is skipped; anything else, or a failed rename or hash, publishes nothing
tar --warning=no-file-changed --ignore-failed-read -czf "$EXPORTS/$name.part" -T "$list"
rc=$?
rm -f "$list"
if (( rc > 1 )) || ! mv "$EXPORTS/$name.part" "$EXPORTS/$name" || ! (cd "$EXPORTS" && sha256sum "$name" > "$name.sha256"); then
  rm -f "$EXPORTS/$name.part"; echo "packing $name failed (tar exit $rc)" >&2; exit 1
fi
ls -t "$EXPORTS"/v10-results-*.tgz 2>/dev/null | tail -n +4 | while read -r old; do rm -f "$old" "$old.sha256"; done
echo "$EXPORTS/$name"
