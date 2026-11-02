#!/usr/bin/env bash
# Pack deployment-study results: bash scripts/deploy_export.sh <label>
# Run + smoke directories without the imported adapter (evaluations, load-test levels and service answers, reports,
# demo record), logs, environment. No weights: the adapter came in the bundle; the merged model stays on the host.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$(dirname "$REPO")
EXPORTS=$HERE/exports
mkdir -p "$EXPORTS"
cd "$REPO" || exit 1
stamp=$(TZ=CST-8 date +%m%d-%H%M)
list=$(mktemp)
for run in outputs/deploy/run outputs/deploy/smoke; do
  [[ -d $run ]] && find "$run" -type f ! -path '*/imports/*' ! -name '*.safetensors' ! -name '*.lock' ! -name '*.tmp*' >> "$list"
done
find outputs/logs outputs/environment -type f 2>/dev/null >> "$list"
cp "$HERE/STATUS.txt" "$EXPORTS/STATUS.txt" 2>/dev/null
name=deploy-results-$stamp-$1.tgz
# exit 1 only says a file grew while it was read (logs during a run; the archive is still complete), and a file removed
# since the listing is skipped; anything else, or a failed rename or hash, publishes nothing
tar --warning=no-file-changed --ignore-failed-read -czf "$EXPORTS/$name.part" -T "$list"
rc=$?
if (( rc > 1 )) || ! mv "$EXPORTS/$name.part" "$EXPORTS/$name" || ! (cd "$EXPORTS" && sha256sum "$name" > "$name.sha256"); then
  rm -f "$EXPORTS/$name.part" "$list"; echo "packing $name failed (tar exit $rc)" >&2; exit 1
fi
ls -t "$EXPORTS"/deploy-results-*.tgz 2>/dev/null | tail -n +4 | while read -r old; do rm -f "$old" "$old.sha256"; done
rm -f "$list"
echo "$EXPORTS/$name"
