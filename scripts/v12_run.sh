#!/usr/bin/env bash
# v0.12 unattended run on a fresh 8xH20 host (the teacher takes four GPUs, every other GPU one student run). Layout after
# `tar -xzf efa-v12-bundle-*.tgz` in /root/efa-v12:
#   exec-feedback-agent/  source (git HEAD)   registration/  v0.5 task files
#   imports/  v0.8 manifest; TR and PTR adapters (seeds 42-44); v0.8's per-task rows of R and the teacher (references)
#   BUNDLE.json  sha256 of every file
# Usage:  bash exec-feedback-agent/scripts/v12_run.sh --stop <ISO time with +08:00>
#         (absolute compute stop; keep >= 25 min before the booking ends for downloading results)
# Progress: cat /root/efa-v12/STATUS.txt ; results: /root/efa-v12/exports/ (packed at milestones and every 30 min)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
SELF=$REPO/scripts/v12_run.sh
EXPORTS=$HERE/exports
STATUS=$HERE/STATUS.txt
PROGRESS=$EXPORTS/进度-v12.md
MODELS=$HERE/models
RUN=$REPO/outputs/v12/run
SMOKE=$REPO/outputs/v12/smoke
TEACHER_HOME=$REPO/outputs/v12/teacher
BIRD_TRAIN_SHA=66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc
BIRD_DEV_SHA=cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630
STUDENT=Qwen/Qwen3-4B-Instruct-2507
TEACHER=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8

if [[ ${1:-} != --worker ]]; then
  [[ ${1:-} == --stop && -n ${2:-} ]] || { echo "usage: bash $0 --stop <ISO time with +08:00>"; exit 2; }
  date -d "$2" > /dev/null 2>&1 || { echo "bad --stop $2"; exit 2; }
  # setup, the teacher's evaluation and the smoke take about 2 h; training ends 45 min before the stop
  (( $(date -d "$2" +%s) > $(date +%s) + 6 * 3600 )) \
    || echo "WARNING: less than 6 h before --stop: the OPD runs get few steps (planned: about 8 h of training in 11 h)"
  flock -n "$HERE/.v12.lock" true || { echo "Already running: cat $STATUS"; exit 1; }
  setsid nohup bash "$SELF" --worker "$2" >> "$HERE/v12.log" 2>&1 < /dev/null &
  echo "Started (pid $!). Closing the terminal does not stop it. Progress: cat $STATUS"
  exit 0
fi
STOP=$2
exec 9> "$HERE/.v12.lock"; flock -n 9 || exit 1
mkdir -p "$EXPORTS" "$MODELS"
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
die() { note "STOPPED: $*"; bash "$REPO/scripts/v12_export.sh" stopped > /dev/null 2>&1; exit 1; }
BUNDLE=$(python3 -c "import json;print(json.load(open('$HERE/BUNDLE.json'))['bundle'])" 2>/dev/null || echo unknown)
note "v0.12 worker on $(hostname); compute stop $STOP; bundle $BUNDLE"

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
NGPU=$(tr ',' '\n' <<< "$GPUS" | wc -l)
(( NGPU >= 5 )) || die "$NGPU GPUs with >= 80 GB: the teacher needs four and every student run one more"
(( NGPU >= 8 )) || note "WARNING: $NGPU GPUs, not 8: the scheduler gives the OPD seeds the student GPUs in seed order and TROPD only a spare one"
FIRST=${GPUS%%,*}
SHM_MB=$(df -m /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [[ -n $SHM_MB ]] && (( SHM_MB < 1024 )); then
  export NCCL_SHM_DISABLE=1  # tensor-parallel teacher: NCCL then uses NVLink P2P instead of a small /dev/shm
fi
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
(( ${FREE_GB:-0} >= 400 )) || note "WARNING: only ${FREE_GB} GB free disk (teacher 236 GB, BIRD ~50 GB extracted, 4B 8 GB, venvs ~25 GB, runs ~20 GB)"
MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
[[ $LIMIT =~ ^[0-9]+$ ]] && (( LIMIT / 1073741824 < MEM_GB )) && MEM_GB=$(( LIMIT / 1073741824 ))
note "hardware: GPUs [$GPUS], driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), glibc $(ldd --version | head -1 | awk '{print $NF}'), $(nproc) CPUs, ${MEM_GB} GB RAM, ${FREE_GB} GB free disk, /dev/shm ${SHM_MB:-?} MB"

# ---- 2. stdlib-only work first (the 236 GB teacher is the long pole; BIRD and the 4B over parallel ranges), then the
# main environment
[[ -f /etc/pki/tls/certs/ca-bundle.crt ]] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
export HF_HOME=$REPO/.hf HF_ENDPOINT=https://huggingface.co PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONPATH=$REPO
unset VLLM_FLAT_LOGPROBS  # the teacher's scoring reads prompt log-probs as vLLM's default list of dicts
export UV_PYTHON_INSTALL_DIR=$REPO/.runtime/python UV_CACHE_DIR=$REPO/.runtime/cache
early_python() {
  if [[ ! -x .runtime/bin/uv ]]; then
    local tmp asset=uv-x86_64-unknown-linux-musl.tar.gz
    tmp=$(mktemp -d)
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
      "https://github.com/astral-sh/uv/releases/download/0.10.0/$asset" -o "$tmp/$asset" || return 1
    # the check's "<file>: OK" goes to the log: stdout is the interpreter path the caller captures
    (cd "$tmp" && printf '%s  %s\n' 312d37f31b6f2c3bfc65668ba0efea9f1f9eaf7bc3209fe1a109e5cf861b95fa "$asset" \
      | sha256sum --check >&2) || return 1
    tar -xzf "$tmp/$asset" -C "$tmp" uv-x86_64-unknown-linux-musl/uv && mkdir -p .runtime/bin \
      && cp "$tmp/uv-x86_64-unknown-linux-musl/uv" .runtime/bin/uv && chmod +x .runtime/bin/uv || return 1
    rm -rf "$tmp"
  fi
  .runtime/bin/uv python install 3.12 > /dev/null && .runtime/bin/uv python find 3.12
}
PY3=$(early_python 2>> "$LOGS/early-python.log" || true)
download() {  # kind repo workers python
  [[ -f $MODELS/$1/.efa-verified.json ]] && { echo 0 > "$MODELS/$1.download.rc"; return; }
  rm -f "$MODELS/$1.download.rc"
  "$4" -m exec_feedback.v8.download --manifest configs/v12-models.json --model "$2" --out "$MODELS/$1" \
    --workers "$3" >> "$LOGS/download-$1.log" 2>&1
  echo $? > "$MODELS/$1.download.rc"
}
early_fetch() {  # part sha python
  "$3" -m exec_feedback.v9.fetch --part "$1" --sha256 "$2" --out data/raw --workers 16 >> "$LOGS/fetch-$1.log" 2>&1
  echo "$1 $?" > "$LOGS/early-fetch-$1.rc"
}
EARLY=0
if [[ -n $PY3 ]] && "$PY3" -c "import exec_feedback.v9.fetch, exec_feedback.v8.download" 2> /dev/null; then
  ( download teacher "$TEACHER" 24 "$PY3" ) &
  ( early_fetch train "$BIRD_TRAIN_SHA" "$PY3" ) &
  ( early_fetch dev "$BIRD_DEV_SHA" "$PY3" ) &
  ( download student "$STUDENT" 8 "$PY3" ) &
  EARLY=1
  note "teacher (24 connections), BIRD train/dev (16 ranges each) and the 4B downloads started early with $PY3"
else
  note "early downloads not started (python '${PY3:-none}', see $LOGS/early-python.log); they start after the main venv"
fi
note "setup: main venv (about 6-8 min)"
if [[ ! -f outputs/environment/.bootstrap-ok ]]; then
  CUDA_VISIBLE_DEVICES=$FIRST bash scripts/bootstrap_h20.sh > "$LOGS/bootstrap.log" 2>&1 || die "bootstrap failed ($LOGS/bootstrap.log)"
  touch outputs/environment/.bootstrap-ok
fi
PY=$REPO/.venv/bin/python
UV=$REPO/.runtime/bin/uv
VPY=$REPO/.venv-vllm/bin/python
note "ok: main venv; vLLM venv (+ peft for the online trainer, + the MoE glibc patch for the teacher) in parallel"

# ---- 3. vLLM venv; the downloads (single-connection BIRD and plain model downloads if the early start failed)
( { [[ -x $VPY ]] || { "$UV" venv .venv-vllm --python "$(readlink -f .venv/bin/python)" -q \
      && "$UV" pip install --python "$VPY" -r configs/vllm-0.11.2-lock.txt; }; } > "$LOGS/vllm-install.log" 2>&1
  rc=$?
  # the online trainer runs vLLM and PEFT in one process: the same pins as the main venv, no dependency changes
  (( rc == 0 )) && { "$UV" pip install --python "$VPY" --no-deps peft==0.18.0 accelerate==1.12.0 >> "$LOGS/vllm-install.log" 2>&1; rc=$?; }
  if (( rc == 0 )); then
    MOE=$("$VPY" -c "import importlib.util as u, os; print(os.path.join(os.path.dirname(u.find_spec('vllm').origin), '_moe_C.abi3.so'))" 2>> "$LOGS/vllm-install.log")
    "$VPY" scripts/patch_moe_glibc.py "$MOE" >> "$LOGS/vllm-install.log" 2>&1 || echo "moe patch failed" >> "$LOGS/vllm-install.log"
  fi
  echo "vllm $rc" > "$LOGS/vllm.rc" ) &
if (( ! EARLY )); then
  ( download teacher "$TEACHER" 24 "$PY" ) &
  ( download student "$STUDENT" 8 "$PY" ) &
fi
fetch() { [[ -f data/raw/$1.source.json ]] && return 0
  "$PY" -m exec_feedback fetch-bird --part "$1" --out data/raw --accept-terms > "$LOGS/fetch-$1-single.log" 2>&1; }
for part in train dev; do
  if (( EARLY )); then
    while [[ ! -f $LOGS/early-fetch-$part.rc ]]; do sleep 10; done
  fi
  fetch "$part" || die "BIRD $part download failed ($LOGS/fetch-$part*.log)"
done
python3 -c "import json,sys;sys.exit(0 if json.load(open('data/raw/train.source.json'))['sha256']=='$BIRD_TRAIN_SHA' else 1)" \
  || die "BIRD train archive differs from the registered one"
python3 -c "import json,sys;sys.exit(0 if json.load(open('data/raw/dev.source.json'))['sha256']=='$BIRD_DEV_SHA' else 1)" \
  || die "BIRD dev archive differs from the registered one"
note "ok: BIRD train/dev (registered hashes)"

# ---- 4. CPU gates and freeze (smoke subset + main run); the freeze needs the model manifest, not the weights
gate_tests() { CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider tests/test_v12.py tests/test_v11.py \
  tests/test_v09.py tests/test_v08.py tests/test_rft_v05.py tests/test_core.py tests/test_pipeline.py \
  > "$LOGS/v12-tests$1.log" 2>&1; }
if ! gate_tests ""; then  # one rerun: the end-to-end scheduler tests are timing-dependent subprocess runs
  note "CPU tests failed once ($(tail -1 "$LOGS/v12-tests.log")); running them again"
  gate_tests "-rerun" || die "CPU tests failed twice ($LOGS/v12-tests.log, $LOGS/v12-tests-rerun.log)"
  TESTLOG=$LOGS/v12-tests-rerun.log
fi
note "ok: CPU tests ($(tail -1 "${TESTLOG:-$LOGS/v12-tests.log}"))"
freeze() {  # out config smoke_limit
  if [[ -f $1/manifest.json ]]; then
    "$PY" -m exec_feedback.v12 status --run "$1" > /dev/null 2>&1 && return 0
    # Code changed since this freeze (a fix after a failed smoke path): a new run under the new code. The old one is
    # kept beside it, never deleted; online steps cost hours, so those need a person.
    if [[ $1 == "$RUN" ]] && compgen -G "$1/train/*/metrics.jsonl" > /dev/null; then
      die "$1 was frozen by other code and has online steps; decide by hand"
    fi
    local stamp
    stamp=$(date +%s)
    mv "$1" "$1.stale-$stamp"
    # the teacher service's records (gate, requests, served) belong to the runs it served: set aside with them
    [[ -d $TEACHER_HOME ]] && mv "$TEACHER_HOME" "$TEACHER_HOME.stale-$stamp"
    note "code changed: $(basename "$1") moved aside and frozen again"
  fi
  "$PY" -m exec_feedback.v12 freeze --registration "$HERE/registration" --raw-root data/raw --models-root "$MODELS" \
    --models-manifest configs/v12-models.json --imports "$HERE/imports" --config "$2" --out "$1" --smoke-limit "$3" \
    > "$LOGS/freeze-$(basename "$1").log" 2>&1
}
freeze "$SMOKE" configs/v12-smoke.json 16 || die "smoke freeze failed ($LOGS/freeze-smoke.log)"
freeze "$RUN" configs/h20-v12.json 0 || die "freeze failed ($LOGS/freeze-run.log)"
counts=$("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['counts'], sorted(m['imports']['adapters']), len(m['imports']['references']))")
[[ $counts == "{'pool': 6682, 'dev': 615, 'validation': 1612, 'test': 1501} ['PTR_seed42', 'PTR_seed43', 'PTR_seed44', 'TR_seed42', 'TR_seed43', 'TR_seed44'] 11" ]] \
  || die "registration counts or imports differ: $counts"
note "ok: frozen run $("$PY" -c "import json;print(json.load(open('$RUN/manifest.json'))['identity'][:12])"), $counts (counts, v0.8 arms, v0.8 reference evaluations)"
"$PY" -m exec_feedback.v12 progress --run "$RUN" --stage freeze --commit "$BUNDLE" --file "$PROGRESS" > /dev/null
bash scripts/v12_export.sh freeze > /dev/null 2>&1

# ---- 5. vLLM gates: MoE kernels import after the glibc patch (the teacher); the 4B samples with the imported TR LoRA
# and returns log-probs, and peft imports beside vLLM (the online trainer)
while [[ ! -f $LOGS/vllm.rc ]]; do sleep 10; done
read -r what code < "$LOGS/vllm.rc"; (( code == 0 )) || die "vLLM venv install failed ($LOGS/vllm-install.log)"
if "$VPY" -c "import torch, vllm._moe_C; assert hasattr(torch.ops._moe_C, 'topk_softmax')" >> "$LOGS/vllm-install.log" 2>&1; then
  note "ok: vLLM MoE kernels import ($(grep -E '^(patched|nothing to do)' "$LOGS/vllm-install.log" | tail -1))"
else
  # preregistered: no teacher means no OPD, the comparator evaluations go on (docs/RESEARCH_V12.md)
  note "WARNING: vLLM MoE kernels do not import ($LOGS/vllm-install.log); the teacher will fail: no OPD, only B/TR/PTR evaluations"
fi
while [[ ! -f $MODELS/student.download.rc ]]; do sleep 10; done
[[ $(cat "$MODELS/student.download.rc") == 0 ]] || die "4B download failed ($LOGS/download-student.log)"
CUDA_VISIBLE_DEVICES=$FIRST timeout 900 "$VPY" -c "
import traceback
from exec_feedback.v9.shutdown import hard_exit
try:
    import peft, accelerate
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model='$MODELS/student', dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.4, enable_lora=True, max_lora_rank=16, max_loras=1)
    out = llm.generate(['SELECT 1 is a'], SamplingParams(max_tokens=8, temperature=1.0, logprobs=0), lora_request=LoRARequest('tr', 1, '$RUN/imports/adapters/TR_seed42/adapter'))[0].outputs[0]
    assert out.logprobs and all(t in d for t, d in zip(out.token_ids, out.logprobs)), out
    print(out.text, [round(d[t].logprob, 3) for t, d in zip(out.token_ids, out.logprobs)], 'peft', peft.__version__)
    code = 0
except BaseException:
    traceback.print_exc()
    code = 1
hard_exit(code)" \
  > "$LOGS/vllm-gate.log" 2>&1 || die "vLLM cannot run here (exit $?; $LOGS/vllm-gate.log)"
"$UV" pip freeze --python "$VPY" > outputs/environment/vllm-packages.txt 2>&1
"$UV" pip freeze --python "$PY" > outputs/environment/main-packages.txt 2>&1
note "ok: vLLM $("$VPY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) samples on GPU $FIRST with the TR LoRA and returns log-probs; teacher download: $(tail -1 "$LOGS/download-teacher.log" 2>/dev/null | cut -c1-120)"

# ---- 6. schedule everything else (the teacher download keeps running; its jobs wait for the verified files)
# Nothing of this worker runs now (the lock), so a service record left by a killed earlier attempt is stale.
rm -f "$TEACHER_HOME/serving.json"
note "scheduler starts"
"$PY" -m exec_feedback.v12.orchestrate --run "$RUN" --smoke-run "$SMOKE" --teacher-home "$TEACHER_HOME" \
  --models-root "$MODELS" --stop "$STOP" --gpus "$GPUS" --repo "$REPO" --main-py "$PY" --vllm-py "$VPY" \
  --status "$STATUS" --progress "$PROGRESS" --commit "$BUNDLE" --export-hook "$REPO/scripts/v12_export.sh" \
  --log-dir "$LOGS/v12" > "$LOGS/orchestrate.log" 2>&1
rc=$?
(( rc == 0 )) || note "scheduler exit $rc ($LOGS/orchestrate.log)"
[[ -f $MODELS/teacher.download.rc && $(cat "$MODELS/teacher.download.rc") != 0 ]] \
  && note "WARNING: teacher download failed ($LOGS/download-teacher.log)"
bash scripts/v12_export.sh final adapters > /dev/null 2>> "$LOGS/export-final.log" \
  || note "WARNING: the final export failed ($LOGS/export-final.log); the newest earlier results package stands"
note "ALL DONE. Download exports/v12-results-*-final.tgz (+ .sha256), exports/进度-v12.md and exports/v12-adapters-*.tar."
touch "$HERE/DONE"
