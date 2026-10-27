#!/usr/bin/env bash
# Pack v0.7 results after a milestone: bash scripts/v07_export.sh <label> [adapters]
# Results (small, always): run + smoke directories without checkpoints/adapters/tokenized SFT data, logs, environment.
# Adapters (final only): every adapter directory and the tokenized SFT sets.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(dirname "$REPO")
EXPORTS=$HERE/exports
mkdir -p "$EXPORTS"
cd "$REPO" || exit 1
stamp=$(TZ=CST-8 date +%m%d-%H%M)
list=$(mktemp)
for run in outputs/v07/run outputs/v07/smoke; do
  [[ -d $run ]] && find "$run" -type f ! -path '*/checkpoints/*' ! -path '*/adapter/*' ! -path '*/claims/*' \
    ! -path '*/imports/*' ! -name '*.lock' ! -name '*.tmp' ! -name 'sft_*.jsonl' >> "$list"
done
find outputs/logs outputs/environment -type f 2>/dev/null >> "$list"
cp "$HERE/STATUS.txt" "$EXPORTS/STATUS.txt" 2>/dev/null
name=v07-results-$stamp-$1.tgz
tar -czf "$EXPORTS/$name" -T "$list" && (cd "$EXPORTS" && sha256sum "$name" > "$name.sha256")
ls -t "$EXPORTS"/v07-results-*.tgz 2>/dev/null | tail -n +4 | while read -r old; do rm -f "$old" "$old.sha256"; done
if [[ ${2:-} == adapters ]]; then
  find outputs/v07/run -type f \( -path '*/train/*/adapter/*' -o -name 'sft_*.jsonl' -o -path '*/train/*/result.json' \
    -o -path '*/train/*/manifest.json' \) > "$list" 2>/dev/null
  [[ -s $list ]] && tar -cf "$EXPORTS/v07-adapters-$stamp.tar" -T "$list" \
    && (cd "$EXPORTS" && sha256sum "v07-adapters-$stamp.tar" > "v07-adapters-$stamp.tar.sha256")
fi
rm -f "$list"
echo "$EXPORTS/$name"
