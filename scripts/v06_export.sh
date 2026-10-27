#!/usr/bin/env bash
# Pack v0.6 results after a stage: bash scripts/v06_export.sh <label> [adapters]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(dirname "$REPO")
EXPORTS=$HERE/exports
RUN=outputs/v06/run
mkdir -p "$EXPORTS"
cd "$REPO" || exit 1
stamp=$(TZ=CST-8 date +%m%d-%H%M)
list=$(mktemp)
{ [[ -d $RUN ]] && find "$RUN" -type f ! -path '*/checkpoints/*' ! -path '*/adapter/*' ! -path '*/claims/*' \
    ! -name '*.lock' ! -name '*.tmp' ! -name 'sft_*.jsonl'
  find outputs/logs outputs/environment -type f 2>/dev/null; ls outputs/model-path.txt 2>/dev/null; } > "$list"
cp "$HERE/STATUS.txt" "$EXPORTS/STATUS.txt" 2>/dev/null
name=v06-results-$stamp-$1.tgz
tar -czf "$EXPORTS/$name" -T "$list" && (cd "$EXPORTS" && sha256sum "$name" > "$name.sha256")
ls -t "$EXPORTS"/v06-results-*.tgz 2>/dev/null | tail -n +4 | while read -r old; do rm -f "$old" "$old.sha256"; done
if [[ ${2:-} == adapters ]]; then
  find "$RUN" -type f \( -path '*/adapter/*' -o -name 'sft_*.jsonl' \) > "$list"
  [[ -s $list ]] && tar -cf "$EXPORTS/v06-adapters-$stamp.tar" -T "$list" \
    && (cd "$EXPORTS" && sha256sum "v06-adapters-$stamp.tar" > "v06-adapters-$stamp.tar.sha256")
fi
rm -f "$list"
echo "$EXPORTS/$name"
