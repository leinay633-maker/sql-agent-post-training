#!/usr/bin/env bash
# v0.8 unattended run on a fresh 8xH20 host. Layout after `tar -xzf efa-v08-bundle-*.tgz` in /root/efa-v08:
#   exec-feedback-agent/  source (git HEAD)   registration/  v0.5 task files   imports/  v0.5 R/P adapters, data
#   BUNDLE.json           sha256 of every file
# Usage:  bash exec-feedback-agent/scripts/v08_run.sh --stop 2026-10-13T01:35:00+08:00
#         (absolute compute stop; keep >= 25 min before the booking ends for downloading results)
# Progress: cat /root/efa-v08/STATUS.txt ; results: /root/efa-v08/exports/ (re-packed at every milestone)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
SELF=$REPO/scripts/v08_run.sh
EXPORTS=$HERE/exports
STATUS=$HERE/STATUS.txt
PROGRESS=$EXPORTS/进度-v08.md
MODELS=$HERE/models
RUN=$REPO/outputs/v08/run
SMOKE=$REPO/outputs/v08/smoke
BIRD_TRAIN_SHA=66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc
BIRD_DEV_SHA=cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630
TEACHER=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8

if [[ ${1:-} != --worker ]]; then
  [[ ${1:-} == --stop && -n ${2:-} ]] || { echo "usage: bash $0 --stop <ISO time with +08:00>"; exit 2; }
  date -d "$2" > /dev/null 2>&1 || { echo "bad --stop $2"; exit 2; }
  (( $(date -d "$2" +%s) > $(date +%s) + 3 * 3600 )) || echo "WARNING: less than 3 h before --stop"
  flock -n "$HERE/.v08.lock" true || { echo "Already running: cat $STATUS"; exit 1; }
  setsid nohup bash "$SELF" --worker "$2" >> "$HERE/v08.log" 2>&1 < /dev/null &
  echo "Started (pid $!). Closing the terminal does not stop it. Progress: cat $STATUS"
  exit 0
fi
STOP=$2
exec 9> "$HERE/.v08.lock"; flock -n 9 || exit 1
mkdir -p "$EXPORTS" "$MODELS"
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
die() { note "STOPPED: $*"; bash "$REPO/scripts/v08_export.sh" stopped > /dev/null 2>&1; exit 1; }
BUNDLE=$(python3 -c "import json;print(json.load(open('$HERE/BUNDLE.json'))['bundle'])" 2>/dev/null || echo unknown)
note "v0.8 worker on $(hostname); compute stop $STOP; bundle $BUNDLE"

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
{ date -Is; nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nvidia-smi topo -m
  free -h; nproc; ldd --version | head -1; df -h . /tmp /dev/shm; cat /etc/os-release | head -3; } \
  > outputs/environment/hardware.txt 2>&1
GPUS=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$2 >= 80000 {printf "%s%s", s, $1; s=","}')
[[ -n $GPUS ]] || die "no GPU with >= 80 GB"
FIRST=${GPUS%%,*}
SHM_MB=$(df -m /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [[ -n $SHM_MB ]] && (( SHM_MB < 1024 )); then
  export NCCL_SHM_DISABLE=1  # tensor-parallel teacher: NCCL then uses NVLink P2P instead of a small /dev/shm
fi
note "hardware: GPUs [$GPUS], driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), glibc $(ldd --version | head -1 | awk '{print $NF}'), $(nproc) CPUs, /dev/shm ${SHM_MB:-?} MB"

# ---- 2. main environment (uv, Python 3.12, torch, gate tests)
[[ -f /etc/pki/tls/certs/ca-bundle.crt ]] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
export HF_HOME=$REPO/.hf HF_ENDPOINT=https://huggingface.co PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONPATH=$REPO
# The 236 GB teacher is the long pole: start it (stdlib only) before torch is installed. uv and its Python 3.12 are
# fetched exactly as scripts/bootstrap_h20.sh does, which then finds and reuses both.
export UV_PYTHON_INSTALL_DIR=$REPO/.runtime/python UV_CACHE_DIR=$REPO/.runtime/cache
early_python() {
  if [[ ! -x .runtime/bin/uv ]]; then
    local tmp asset=uv-x86_64-unknown-linux-musl.tar.gz
    tmp=$(mktemp -d)
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
      "https://github.com/astral-sh/uv/releases/download/0.10.0/$asset" -o "$tmp/$asset" || return 1
    (cd "$tmp" && printf '%s  %s\n' 312d37f31b6f2c3bfc65668ba0efea9f1f9eaf7bc3209fe1a109e5cf861b95fa "$asset" \
      | sha256sum --check) || return 1
    tar -xzf "$tmp/$asset" -C "$tmp" uv-x86_64-unknown-linux-musl/uv && mkdir -p .runtime/bin \
      && cp "$tmp/uv-x86_64-unknown-linux-musl/uv" .runtime/bin/uv && chmod +x .runtime/bin/uv || return 1
    rm -rf "$tmp"
  fi
  .runtime/bin/uv python install 3.12 > /dev/null 2>&1 && .runtime/bin/uv python find 3.12
}
PY3=$(early_python 2>> "$LOGS/early-python.log" || true)
download() {  # kind repo workers python
  [[ -f $MODELS/$1/.efa-verified.json ]] && { echo 0 > "$MODELS/$1.download.rc"; return; }
  rm -f "$MODELS/$1.download.rc"
  "$4" -m exec_feedback.v8.download --manifest configs/v08-models.json --model "$2" --out "$MODELS/$1" \
    --workers "$3" >> "$LOGS/download-$1.log" 2>&1
  echo $? > "$MODELS/$1.download.rc"
}
EARLY=0
if [[ -n $PY3 ]] && "$PY3" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2> /dev/null \
    && "$PY3" -c "import exec_feedback.v8.download" 2> /dev/null; then
  ( download teacher "$TEACHER" 24 "$PY3" ) &
  EARLY=1
  note "teacher download started early with $PY3 (24 connections, 1 GiB ranges)"
fi
note "setup: main venv (about 6-8 min)"
if [[ ! -f outputs/environment/.bootstrap-ok ]]; then
  CUDA_VISIBLE_DEVICES=$FIRST bash scripts/bootstrap_h20.sh > "$LOGS/bootstrap.log" 2>&1 || die "bootstrap failed ($LOGS/bootstrap.log)"
  touch outputs/environment/.bootstrap-ok
fi
PY=$REPO/.venv/bin/python
UV=$REPO/.runtime/bin/uv
VPY=$REPO/.venv-vllm/bin/python
note "ok: main venv; starting in parallel: vLLM venv (+ MoE glibc patch), BIRD train/dev, 4B download"

# ---- 3. everything that only needs the network, in parallel
fetch() { [[ -f data/raw/$1.source.json ]] && return 0
  "$PY" -m exec_feedback fetch-bird --part "$1" --out data/raw --accept-terms > "$LOGS/fetch-$1.log" 2>&1; }
( fetch train; echo "train $?" > "$LOGS/fetch-train.rc" ) &
( fetch dev; echo "dev $?" > "$LOGS/fetch-dev.rc" ) &
( { [[ -x $VPY ]] || { "$UV" venv .venv-vllm --python "$(readlink -f .venv/bin/python)" -q \
      && "$UV" pip install --python "$VPY" -r configs/vllm-0.11.2-lock.txt; }; } > "$LOGS/vllm-install.log" 2>&1
  rc=$?
  if (( rc == 0 )); then
    MOE=$("$VPY" -c "import importlib.util as u, os; print(os.path.join(os.path.dirname(u.find_spec('vllm').origin), '_moe_C.abi3.so'))" 2>> "$LOGS/vllm-install.log")
    "$VPY" scripts/patch_moe_glibc.py "$MOE" >> "$LOGS/vllm-install.log" 2>&1 || echo "moe patch failed" >> "$LOGS/vllm-install.log"
  fi
  echo "vllm $rc" > "$LOGS/vllm.rc" ) &
( download student Qwen/Qwen3-4B-Instruct-2507 8 "$PY" ) &
(( EARLY )) || { ( download teacher "$TEACHER" 24 "$PY" ) & }
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
gate_tests() { "$PY" -m pytest -q -p no:cacheprovider tests/test_v08.py tests/test_v07.py tests/test_rft_v05.py \
  tests/test_core.py tests/test_pipeline.py > "$LOGS/v08-tests$1.log" 2>&1; }
if ! gate_tests ""; then  # one rerun: the end-to-end scheduler tests are timing-dependent subprocess runs
  note "CPU tests failed once ($(tail -1 "$LOGS/v08-tests.log")); running them again"
  gate_tests "-rerun" || die "CPU tests failed twice ($LOGS/v08-tests.log, $LOGS/v08-tests-rerun.log)"
  TESTLOG=$LOGS/v08-tests-rerun.log
fi
note "ok: CPU tests ($(tail -1 "${TESTLOG:-$LOGS/v08-tests.log}"))"
freeze() {  # out config smoke_limit
  if [[ -f $1/manifest.json ]]; then
    "$PY" -m exec_feedback.v8 status --run "$1" > /dev/null 2>&1 && return 0
    # Code changed since this freeze (a fix after a failed smoke path): a new run under the new code. The old one is
    # kept beside it, never deleted; finished trainings are too expensive to drop silently, so those need a person.
    compgen -G "$1/train/*/result.json" > /dev/null && die "$1 was frozen by other code and has finished trainings; decide by hand"
    mv "$1" "$1.stale-$(date +%s)"
    note "code changed: $(basename "$1") moved aside and frozen again"
  fi
  "$PY" -m exec_feedback.v8 freeze --registration "$HERE/registration" --raw-root data/raw --models-root "$MODELS" \
    --models-manifest configs/v08-models.json --imports "$HERE/imports" --config "$2" --out "$1" --smoke-limit "$3" \
    > "$LOGS/freeze-$(basename "$1").log" 2>&1
}
freeze "$SMOKE" configs/v08-smoke.json 16 || die "smoke freeze failed ($LOGS/freeze-smoke.log)"
freeze "$RUN" configs/h20-v08.json 0 || die "freeze failed ($LOGS/freeze-run.log)"
counts=$("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['counts'], m['n_examples'], len(m['unsolved_tasks']))")
[[ $counts == "{'pool': 6682, 'dev': 615, 'validation': 1612, 'test': 1501} 3435 3247" ]] || die "registration counts differ: $counts"
note "ok: frozen run $("$PY" -c "import json;print(json.load(open('$RUN/manifest.json'))['identity'][:12])"), $counts (pool, R examples, never-solved tasks)"
"$PY" -m exec_feedback.v8 progress --run "$RUN" --stage freeze --commit "$BUNDLE" --file "$PROGRESS" > /dev/null
bash scripts/v08_export.sh freeze > /dev/null 2>&1

# ---- 5. vLLM gate: a real generation on one GPU with the 4B; MoE kernels must import after the glibc patch
while [[ ! -f $LOGS/vllm.rc ]]; do sleep 10; done
read -r what code < "$LOGS/vllm.rc"; (( code == 0 )) || die "vLLM venv install failed ($LOGS/vllm-install.log)"
if "$VPY" -c "import torch, vllm._moe_C; assert hasattr(torch.ops._moe_C, 'topk_softmax')" >> "$LOGS/vllm-install.log" 2>&1; then
  note "ok: vLLM MoE kernels import ($(grep -E '^(patched|nothing to do)' "$LOGS/vllm-install.log" | tail -1))"
else
  note "WARNING: vLLM MoE kernels do not import ($LOGS/vllm-install.log); the teacher will fail and HR carries the run"
fi
while [[ ! -f $MODELS/student.download.rc ]]; do sleep 10; done
[[ $(cat "$MODELS/student.download.rc") == 0 ]] || die "4B download failed ($LOGS/download-student.log)"
CUDA_VISIBLE_DEVICES=$FIRST timeout 900 "$VPY" -c "
from vllm import LLM, SamplingParams
llm = LLM(model='$MODELS/student', dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=16)
print(llm.generate(['SELECT 1 is a'], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0].text)" \
  > "$LOGS/vllm-gate.log" 2>&1 || die "vLLM cannot run here ($LOGS/vllm-gate.log)"
"$UV" pip freeze --python "$VPY" > outputs/environment/vllm-packages.txt 2>&1
note "ok: vLLM $("$VPY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) generates on GPU $FIRST; teacher download: $(tail -1 "$LOGS/download-teacher.log" 2>/dev/null | cut -c1-120); scheduler starts"

# ---- 6. schedule everything else (the teacher download keeps running; its jobs wait for the verified files)
"$PY" -m exec_feedback.v8.orchestrate --run "$RUN" --smoke-run "$SMOKE" --models-root "$MODELS" --stop "$STOP" \
  --gpus "$GPUS" --repo "$REPO" --main-py "$PY" --vllm-py "$VPY" --status "$STATUS" --progress "$PROGRESS" \
  --commit "$BUNDLE" --export-hook "$REPO/scripts/v08_export.sh" --log-dir "$LOGS/v08" > "$LOGS/orchestrate.log" 2>&1
note "scheduler exit $?"
[[ -f $MODELS/teacher.download.rc && $(cat "$MODELS/teacher.download.rc") != 0 ]] && note "WARNING: teacher download failed ($LOGS/download-teacher.log)"
bash scripts/v08_export.sh final adapters > /dev/null 2>&1
note "ALL DONE. Download exports/v08-results-*.tgz (+ .sha256) and exports/进度-v08.md; adapters .tar optional."
touch "$HERE/DONE"
