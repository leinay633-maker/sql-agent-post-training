#!/usr/bin/env bash
# v0.6 unattended run on a fresh 8xH20 host. Layout after `tar -xzf efa-v06-*.tgz` in /root/efa-v06:
#   exec-feedback-agent/   v0.6 source       registration/   v0.5 task registration (verified by hash)
#   vllm-lock.txt          exact vLLM venv   BUNDLE.json     file hashes of this bundle
# Usage:  bash exec-feedback-agent/scripts/v06_run.sh --stop 2026-10-12T08:45:00+08:00
#         (the absolute compute stop; keep >= 30 min before the booking ends for download)
# Progress: cat /root/efa-v06/STATUS.txt ; results: /root/efa-v06/exports/ (re-packed after every stage)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
SELF=$REPO/scripts/v06_run.sh
EXPORTS=$HERE/exports
STATUS=$HERE/STATUS.txt
PROGRESS=$EXPORTS/进度-v06.md
RUN=$REPO/outputs/v06/run
BIRD_TRAIN_SHA=66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc
BIRD_DEV_SHA=cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
MODEL_REV=c03e6d358207e414f1eca0bb1891e29f1db0e242

if [[ ${1:-} != --worker ]]; then
  [[ ${1:-} == --stop && -n ${2:-} ]] || { echo "usage: bash $0 --stop <ISO time with +08:00>"; exit 2; }
  date -d "$2" > /dev/null 2>&1 || { echo "bad --stop $2"; exit 2; }
  (( $(date -d "$2" +%s) > $(date +%s) + 4 * 3600 )) || echo "WARNING: less than 4 h before --stop; round 2 will be skipped"
  flock -n "$HERE/.v06.lock" true || { echo "Already running: cat $STATUS"; exit 1; }
  setsid nohup bash "$SELF" --worker "$2" >> "$HERE/v06.log" 2>&1 < /dev/null &
  echo "Started (pid $!). Closing the terminal does not stop it. Progress: cat $STATUS"
  exit 0
fi
STOP=$2
exec 9> "$HERE/.v06.lock"; flock -n 9 || exit 1
mkdir -p "$EXPORTS"
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
die() { note "STOPPED: $*"; bash "$REPO/scripts/v06_export.sh" stopped > /dev/null 2>&1; exit 1; }
COMMIT=$(python3 -c "import json;print(json.load(open('$HERE/BUNDLE.json'))['bundle'])" 2>/dev/null || echo unknown)
note "v0.6 worker on $(hostname); compute stop $STOP; bundle $COMMIT"

# ---- 0. bundle integrity
python3 - "$HERE" <<'PY' || die "bundle files differ from BUNDLE.json; upload again"
import hashlib, json, sys
from pathlib import Path
here = Path(sys.argv[1]); b = json.loads((here / "BUNDLE.json").read_text())
bad = [n for n, h in b["files"].items() if hashlib.sha256((here / n).read_bytes()).hexdigest() != h]
sys.exit(1 if bad else 0)
PY
note "ok: bundle verified ($(python3 -c "import json;print(len(json.load(open('$HERE/BUNDLE.json'))['files']))") files)"

# ---- 1. hardware
cd "$REPO" || die "no repo"
mkdir -p outputs/environment outputs/logs
{ date -Is; nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  free -h; nproc; ldd --version | head -1; df -h . /tmp; cat /etc/os-release | head -3; } > outputs/environment/hardware.txt 2>&1
GPUS=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$2 >= 80000 {printf "%s%s", s, $1; s=","}')
[[ -n $GPUS ]] || die "no GPU with >= 80 GB"
note "hardware: GPUs [$GPUS], driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), $(ldd --version | head -1 | awk '{print $NF}') glibc"

# ---- 2. environments, data, model (in parallel)
[[ -f /etc/pki/tls/certs/ca-bundle.crt ]] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
export HF_HOME=$REPO/.hf HF_ENDPOINT=https://huggingface.co PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONPATH=$REPO
note "setup: main venv (gate tests), vLLM venv, BIRD train/dev, $MODEL@${MODEL_REV:0:8} (about 25 min)"
if [[ ! -f outputs/environment/.bootstrap-ok ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} bash scripts/bootstrap_h20.sh > outputs/logs/bootstrap.log 2>&1 \
    || die "bootstrap failed (outputs/logs/bootstrap.log)"
  touch outputs/environment/.bootstrap-ok
fi
PY=$REPO/.venv/bin/python
UV=$REPO/.runtime/bin/uv
note "ok: main venv"
fetch() { [[ -f data/raw/$1.source.json ]] && return 0
  "$PY" -m exec_feedback fetch-bird --part "$1" --out data/raw --accept-terms > "outputs/logs/fetch-$1.log" 2>&1; }
( fetch train; echo "train $?" > outputs/logs/fetch-train.rc ) &
( fetch dev; echo "dev $?" > outputs/logs/fetch-dev.rc ) &
( "$PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', revision='$MODEL_REV', allow_patterns=['*.json','*.safetensors','*.txt','LICENSE']))" \
    > outputs/model-path.txt 2> outputs/logs/model.log; echo "model $?" > outputs/logs/model.rc ) &
( [[ -x .venv-vllm/bin/python ]] || { "$UV" venv .venv-vllm --python "$(readlink -f .venv/bin/python)" -q \
    && "$UV" pip install --python .venv-vllm/bin/python -r "$HERE/vllm-lock.txt"; } > outputs/logs/vllm-install.log 2>&1
  echo "vllm $?" > outputs/logs/vllm.rc ) &
wait
for rc in outputs/logs/{fetch-train,fetch-dev,model,vllm}.rc; do
  read -r what code < "$rc"; (( code == 0 )) || die "$what setup failed (outputs/logs/)"; done
VPY=$REPO/.venv-vllm/bin/python
MODEL_PATH=$(tail -1 outputs/model-path.txt)
python3 -c "import json,sys;sys.exit(0 if json.load(open('data/raw/train.source.json'))['sha256']=='$BIRD_TRAIN_SHA' else 1)" \
  || die "BIRD train archive differs from the registered one"
python3 -c "import json,sys;sys.exit(0 if json.load(open('data/raw/dev.source.json'))['sha256']=='$BIRD_DEV_SHA' else 1)" \
  || die "BIRD dev archive differs from the registered one"
"$UV" pip freeze --python "$VPY" > outputs/environment/vllm-packages.txt 2>&1
"$PY" -m pip freeze > outputs/environment/main-packages.txt 2>&1 || "$UV" pip freeze --python "$PY" > outputs/environment/main-packages.txt
note "ok: data (BIRD hashes match), model $MODEL_PATH, vLLM $("$VPY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---- 3. gates: CPU tests and a real vLLM generation on one GPU
"$PY" -m pytest -q -p no:cacheprovider tests/test_v06.py tests/test_rft_v05.py tests/test_core.py tests/test_pipeline.py \
  > outputs/logs/v06-tests.log 2>&1 || die "v0.6 CPU tests failed (outputs/logs/v06-tests.log)"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} timeout 900 "$VPY" -c "
from vllm import LLM, SamplingParams
llm = LLM(model='$MODEL_PATH', dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=16)
print(llm.generate(['SELECT 1 is a'], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0].text)" \
  > outputs/logs/vllm-gate.log 2>&1 || die "vLLM cannot run here (outputs/logs/vllm-gate.log)"
note "ok: gates ($(tail -1 outputs/logs/v06-tests.log)); vLLM generates on GPU ${GPUS%%,*}"

# ---- 4. freeze
if [[ ! -f $RUN/manifest.json ]]; then
  [[ -e $RUN ]] && mv "$RUN" "$RUN.incomplete-$(date +%s)"
  "$PY" -m exec_feedback.v6 freeze --registration "$HERE/registration" --raw-root data/raw --model-path "$MODEL_PATH" \
    --config configs/h20-v06.json --out "$RUN" > outputs/logs/freeze.log 2>&1 || die "freeze failed (outputs/logs/freeze.log)"
fi
counts=$("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['counts'])")
[[ $counts == "{'pool': 6682, 'dev': 615, 'validation': 1612, 'test': 1501}" ]] || die "registration counts differ: $counts"
note "ok: frozen run $("$PY" -c "import json;print(json.load(open('$RUN/manifest.json'))['identity'][:12])"), $counts"
"$PY" -m exec_feedback.v6 progress --run "$RUN" --stage freeze --commit "$COMMIT" --file "$PROGRESS" > /dev/null
bash scripts/v06_export.sh freeze > /dev/null 2>&1

# ---- 5. schedule everything else
"$PY" -m exec_feedback.v6.orchestrate --run "$RUN" --stop "$STOP" --gpus "$GPUS" --repo "$REPO" --main-py "$PY" \
  --vllm-py "$VPY" --status "$STATUS" --progress "$PROGRESS" --commit "$COMMIT" \
  --export-hook "$REPO/scripts/v06_export.sh" > outputs/logs/orchestrate.log 2>&1
note "scheduler exit $?"
bash scripts/v06_export.sh final adapters > /dev/null 2>&1
note "ALL DONE. Download exports/v06-results-*.tgz (+ .sha256) and exports/进度-v06.md; adapters .tar optional."
touch "$HERE/DONE"
