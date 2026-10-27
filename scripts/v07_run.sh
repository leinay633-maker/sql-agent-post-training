#!/usr/bin/env bash
# v0.7 unattended run on a fresh 8xH20 host. Layout after `tar -xzf efa-v07-bundle-*.tgz` in /root/efa-v07:
#   exec-feedback-agent/  source (git HEAD)   registration/  v0.5 task files   imports/  v0.5 R/P adapters, data
#   BUNDLE.json           sha256 of every file
# Usage:  bash exec-feedback-agent/scripts/v07_run.sh --stop 2026-10-20T01:30:00+08:00
#         (absolute compute stop; keep >= 30 min before the booking ends for downloading results)
# Progress: cat /root/efa-v07/STATUS.txt ; results: /root/efa-v07/exports/ (re-packed at every milestone)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
SELF=$REPO/scripts/v07_run.sh
EXPORTS=$HERE/exports
STATUS=$HERE/STATUS.txt
PROGRESS=$EXPORTS/进度-v07.md
MODELS=$HERE/models
RUN=$REPO/outputs/v07/run
SMOKE=$REPO/outputs/v07/smoke
BIRD_TRAIN_SHA=66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc
BIRD_DEV_SHA=cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630

if [[ ${1:-} != --worker ]]; then
  [[ ${1:-} == --stop && -n ${2:-} ]] || { echo "usage: bash $0 --stop <ISO time with +08:00>"; exit 2; }
  date -d "$2" > /dev/null 2>&1 || { echo "bad --stop $2"; exit 2; }
  (( $(date -d "$2" +%s) > $(date +%s) + 3 * 3600 )) || echo "WARNING: less than 3 h before --stop"
  flock -n "$HERE/.v07.lock" true || { echo "Already running: cat $STATUS"; exit 1; }
  setsid nohup bash "$SELF" --worker "$2" >> "$HERE/v07.log" 2>&1 < /dev/null &
  echo "Started (pid $!). Closing the terminal does not stop it. Progress: cat $STATUS"
  exit 0
fi
STOP=$2
exec 9> "$HERE/.v07.lock"; flock -n 9 || exit 1
mkdir -p "$EXPORTS" "$MODELS"
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
die() { note "STOPPED: $*"; bash "$REPO/scripts/v07_export.sh" stopped > /dev/null 2>&1; exit 1; }
BUNDLE=$(python3 -c "import json;print(json.load(open('$HERE/BUNDLE.json'))['bundle'])" 2>/dev/null || echo unknown)
note "v0.7 worker on $(hostname); compute stop $STOP; bundle $BUNDLE"

# ---- 0. bundle integrity
python3 - "$HERE" <<'PY' || die "bundle files differ from BUNDLE.json; upload again"
import hashlib, json, sys
from pathlib import Path
here = Path(sys.argv[1]); b = json.loads((here / "BUNDLE.json").read_text())
bad = [n for n, h in b["files"].items() if hashlib.sha256((here / n).read_bytes()).hexdigest() != h]
print("\n".join(bad[:20])); sys.exit(1 if bad else 0)
PY
note "ok: bundle verified"

# ---- 1. hardware
cd "$REPO" || die "no repo"
mkdir -p outputs/environment outputs/logs
LOGS=$REPO/outputs/logs
{ date -Is; nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  free -h; nproc; ldd --version | head -1; df -h . /tmp; cat /etc/os-release | head -3; } > outputs/environment/hardware.txt 2>&1
GPUS=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$2 >= 80000 {printf "%s%s", s, $1; s=","}')
[[ -n $GPUS ]] || die "no GPU with >= 80 GB"
FIRST=${GPUS%%,*}
note "hardware: GPUs [$GPUS], driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), glibc $(ldd --version | head -1 | awk '{print $NF}'), $(nproc) CPUs"

# ---- 2. main environment (uv, Python 3.12, torch, gate tests)
[[ -f /etc/pki/tls/certs/ca-bundle.crt ]] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
export HF_HOME=$REPO/.hf HF_ENDPOINT=https://huggingface.co PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONPATH=$REPO
note "setup: main venv (about 6 min)"
if [[ ! -f outputs/environment/.bootstrap-ok ]]; then
  CUDA_VISIBLE_DEVICES=$FIRST bash scripts/bootstrap_h20.sh > "$LOGS/bootstrap.log" 2>&1 || die "bootstrap failed ($LOGS/bootstrap.log)"
  touch outputs/environment/.bootstrap-ok
fi
PY=$REPO/.venv/bin/python
UV=$REPO/.runtime/bin/uv
VPY=$REPO/.venv-vllm/bin/python
note "ok: main venv; starting in parallel: vLLM venv, BIRD train/dev, 4B -> 14B and 32B downloads"

# ---- 3. everything that only needs the network, in parallel
download() {  # kind repo workers
  [[ -f $MODELS/$1/.efa-verified.json ]] && { echo 0 > "$MODELS/$1.download.rc"; return; }
  rm -f "$MODELS/$1.download.rc"
  "$PY" -m exec_feedback.v7.download --manifest configs/v07-models.json --model "$2" --out "$MODELS/$1" \
    --workers "$3" > "$LOGS/download-$1.log" 2>&1
  echo $? > "$MODELS/$1.download.rc"
}
fetch() { [[ -f data/raw/$1.source.json ]] && return 0
  "$PY" -m exec_feedback fetch-bird --part "$1" --out data/raw --accept-terms > "$LOGS/fetch-$1.log" 2>&1; }
( fetch train; echo "train $?" > "$LOGS/fetch-train.rc" ) &
( fetch dev; echo "dev $?" > "$LOGS/fetch-dev.rc" ) &
( [[ -x $VPY ]] || { "$UV" venv .venv-vllm --python "$(readlink -f .venv/bin/python)" -q \
    && "$UV" pip install --python "$VPY" -r configs/vllm-0.11.2-lock.txt; } > "$LOGS/vllm-install.log" 2>&1
  echo "vllm $?" > "$LOGS/vllm.rc" ) &
( download student Qwen/Qwen3-4B-Instruct-2507 8; download scale Qwen/Qwen3-14B 8 ) &
( download teacher Qwen/Qwen3-32B 6 ) &
for part in fetch-train fetch-dev; do
  while [[ ! -f $LOGS/$part.rc ]]; do sleep 10; done
  read -r what code < "$LOGS/$part.rc"; (( code == 0 )) || die "BIRD $what download failed ($LOGS/$part.log)"
done
python3 -c "import json,sys;sys.exit(0 if json.load(open('data/raw/train.source.json'))['sha256']=='$BIRD_TRAIN_SHA' else 1)" \
  || die "BIRD train archive differs from the registered one"
python3 -c "import json,sys;sys.exit(0 if json.load(open('data/raw/dev.source.json'))['sha256']=='$BIRD_DEV_SHA' else 1)" \
  || die "BIRD dev archive differs from the registered one"
note "ok: BIRD train/dev (registered hashes)"

# ---- 4. CPU gates and freeze (smoke subset + main run)
"$PY" -m pytest -q -p no:cacheprovider tests/test_v07.py tests/test_v06.py tests/test_rft_v05.py tests/test_core.py \
  tests/test_pipeline.py > "$LOGS/v07-tests.log" 2>&1 || die "CPU tests failed ($LOGS/v07-tests.log)"
note "ok: CPU tests ($(tail -1 "$LOGS/v07-tests.log"))"
freeze() {  # out config smoke_limit
  if [[ -f $1/manifest.json ]]; then
    "$PY" -m exec_feedback.v7 status --run "$1" > /dev/null 2>&1 && return 0
    # Code changed since this freeze (a fix after a failed smoke path): a new run under the new code. The old one is
    # kept beside it, never deleted; finished trainings are too expensive to drop silently, so those need a person.
    compgen -G "$1/train/*/result.json" > /dev/null && die "$1 was frozen by other code and has finished trainings; decide by hand"
    mv "$1" "$1.stale-$(date +%s)"
    note "code changed: $(basename "$1") moved aside and frozen again"
  fi
  "$PY" -m exec_feedback.v7 freeze --registration "$HERE/registration" --raw-root data/raw --models-root "$MODELS" \
    --models-manifest configs/v07-models.json --imports "$HERE/imports" --config "$2" --out "$1" --smoke-limit "$3" \
    > "$LOGS/freeze-$(basename "$1").log" 2>&1
}
freeze "$SMOKE" configs/v07-smoke.json 16 || die "smoke freeze failed ($LOGS/freeze-smoke.log)"
freeze "$RUN" configs/h20-v07.json 0 || die "freeze failed ($LOGS/freeze-run.log)"
counts=$("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['counts'], m['n_examples'])")
[[ $counts == "{'pool': 6682, 'dev': 615, 'validation': 1612, 'test': 1501} 3435" ]] || die "registration counts differ: $counts"
note "ok: frozen run $("$PY" -c "import json;print(json.load(open('$RUN/manifest.json'))['identity'][:12])"), $counts"
"$PY" -m exec_feedback.v7 progress --run "$RUN" --stage freeze --commit "$BUNDLE" --file "$PROGRESS" > /dev/null
bash scripts/v07_export.sh freeze > /dev/null 2>&1

# ---- 5. vLLM gate: a real generation on one GPU with the 4B
while [[ ! -f $LOGS/vllm.rc ]]; do sleep 10; done
read -r what code < "$LOGS/vllm.rc"; (( code == 0 )) || die "vLLM venv install failed ($LOGS/vllm-install.log)"
while [[ ! -f $MODELS/student.download.rc ]]; do sleep 10; done
[[ $(cat "$MODELS/student.download.rc") == 0 ]] || die "4B download failed ($LOGS/download-student.log)"
CUDA_VISIBLE_DEVICES=$FIRST timeout 900 "$VPY" -c "
from vllm import LLM, SamplingParams
llm = LLM(model='$MODELS/student', dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=16)
print(llm.generate(['SELECT 1 is a'], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0].text)" \
  > "$LOGS/vllm-gate.log" 2>&1 || die "vLLM cannot run here ($LOGS/vllm-gate.log)"
"$UV" pip freeze --python "$VPY" > outputs/environment/vllm-packages.txt 2>&1
note "ok: vLLM $("$VPY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) generates on GPU $FIRST; scheduler starts (smoke run first, main run follows each smoke path)"

# ---- 6. schedule everything else (14B/32B downloads keep running; their jobs wait for the verified files)
"$PY" -m exec_feedback.v7.orchestrate --run "$RUN" --smoke-run "$SMOKE" --models-root "$MODELS" --stop "$STOP" \
  --gpus "$GPUS" --repo "$REPO" --main-py "$PY" --vllm-py "$VPY" --status "$STATUS" --progress "$PROGRESS" \
  --commit "$BUNDLE" --export-hook "$REPO/scripts/v07_export.sh" --log-dir "$LOGS/v07" > "$LOGS/orchestrate.log" 2>&1
note "scheduler exit $?"
for kind in scale teacher; do
  [[ -f $MODELS/$kind.download.rc && $(cat "$MODELS/$kind.download.rc") != 0 ]] && note "WARNING: $kind download failed ($LOGS/download-$kind.log)"
done
bash scripts/v07_export.sh final adapters > /dev/null 2>&1
note "ALL DONE. Download exports/v07-results-*.tgz (+ .sha256) and exports/进度-v07.md; adapters .tar optional."
touch "$HERE/DONE"
