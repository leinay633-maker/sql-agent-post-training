#!/usr/bin/env bash
# Demo for a screen recording, after the study: vLLM's server (the merged model by default) and the analysis service,
# the service on every interface, both behind a key, for a fixed number of minutes; then both stop and the demo record
# is packed into exports/ (deploy-results-*-demo.tgz).
# Usage:  bash exec-feedback-agent/scripts/deploy_demo.sh [--minutes 60] [--port 8082] [--condition merged] [--force]
#         The key is in /root/efa-deploy/demo.key (owner-only; never printed or logged); paste it into the page.
set -uo pipefail
SELF=$(readlink -f "${BASH_SOURCE[0]}")
HERE=$(cd "$(dirname "$SELF")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
RUN=$REPO/outputs/deploy/run
STATUS=$HERE/STATUS.txt
KEY=$HERE/demo.key
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONPATH=$REPO

if [[ ${1:-} == --worker ]]; then  # detached, under the locks: the demo, then its export
  cd "$REPO" || exit 1
  "$REPO/.venv/bin/python" -m exec_feedback.deploy demo --run "$RUN" --condition "$2" --minutes "$3" --port "$4" \
    --vllm-py "$REPO/.venv-vllm/bin/python" --key-file "$KEY"
  rc=$?
  if bash "$REPO/scripts/deploy_export.sh" demo > /dev/null 2>> "$REPO/outputs/logs/demo.log"; then
    note "demo ended (exit $rc); its record is in exports/deploy-results-*-demo.tgz"
  else
    note "demo ended (exit $rc); WARNING: packing its record failed (outputs/logs/demo.log)"
  fi
  exit $rc
fi

MINUTES=60 PORT=8082 CONDITION=merged FORCE=0
while (( $# )); do
  case $1 in
    --minutes) MINUTES=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --condition) CONDITION=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "usage: bash $0 [--minutes 60] [--port 8082] [--condition merged|fp8|lora] [--force]"; exit 2 ;;
  esac
done
flock -n "$HERE/.demo.lock" true || { echo "A demo is already running"; exit 1; }
[[ -f $RUN/manifest.json ]] || { echo "No frozen run at $RUN"; exit 1; }
# The study's worker holds .deploy.lock while it runs, and the demo holds it for its whole life (unless --force): the
# demo would take the GPU from the evaluations and load tests, and a restarted study would take it from the demo.
if (( FORCE )); then
  LOCKS=(flock -n "$HERE/.demo.lock")
else
  flock -n "$HERE/.deploy.lock" true || { echo "The study is still running (cat $STATUS); wait for ALL DONE, or pass --force"; exit 1; }
  LOCKS=(flock -n "$HERE/.deploy.lock" flock -n "$HERE/.demo.lock")
fi
GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
mkdir -p "$REPO/outputs/logs"
setsid nohup "${LOCKS[@]}" env CUDA_VISIBLE_DEVICES="$GPU" bash "$SELF" --worker "$CONDITION" "$MINUTES" "$PORT" \
  >> "$REPO/outputs/logs/demo.log" 2>&1 < /dev/null &
note "demo: $CONDITION on port $PORT for $MINUTES min (GPU $GPU); key in $KEY; log outputs/logs/demo.log"
echo "Started. In about 3 minutes open http://<this host's IP>:$PORT/ , paste the key from $KEY, then pick a database."
