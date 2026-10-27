#!/usr/bin/env bash
# v0.10 unattended run on a fresh 8xH20 host. Layout after `tar -xzf efa-v10-bundle-*.tgz` in /root/efa-v10:
#   exec-feedback-agent/  source (git HEAD)   registration/  v0.5 task files
#   imports/  v0.8 manifest + PTR adapters; v0.9 manifest, milestone record (K), RL adapters at K and their metrics
#   BUNDLE.json           sha256 of every file
# Usage:  bash exec-feedback-agent/scripts/v10_run.sh --stop <ISO time with +08:00> --seeds 42,43,44
#         (absolute compute stop; keep >= 25 min before the booking ends for downloading results; a later booking that
#         finishes a seed passes only that seed)
# Progress: cat /root/efa-v10/STATUS.txt ; results: /root/efa-v10/exports/ (packed at milestones and every 30 min).
# Each finished main run's model at K is packed there as soon as the run ends: v10-model-*.tar.part-* and, last, .json.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
SELF=$REPO/scripts/v10_run.sh
EXPORTS=$HERE/exports
STATUS=$HERE/STATUS.txt
PROGRESS=$EXPORTS/进度-v10.md
MODELS=$HERE/models
RUN=$REPO/outputs/v10/run
SMOKE=$REPO/outputs/v10/smoke
BIRD_TRAIN_SHA=66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc
BIRD_DEV_SHA=cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630
STUDENT=Qwen/Qwen3-4B-Instruct-2507

if [[ ${1:-} != --worker ]]; then
  [[ ${1:-} == --stop && -n ${2:-} && ${3:-} == --seeds && -n ${4:-} ]] \
    || { echo "usage: bash $0 --stop <ISO time with +08:00> --seeds <42,43,44 or some of them>"; exit 2; }
  date -d "$2" > /dev/null 2>&1 || { echo "bad --stop $2"; exit 2; }
  [[ $4 =~ ^(42|43|44)(,(42|43|44))*$ ]] || { echo "bad --seeds $4"; exit 2; }
  [[ -z $(tr ',' '\n' <<< "$4" | sort | uniq -d) ]] || { echo "bad --seeds $4 (a seed twice)"; exit 2; }
  n_seeds=$(tr ',' '\n' <<< "$4" | grep -c .)
  # setup with the smoke about 1.2 h, the end about 1.2 h, each seed's sync run about 3 h (RLFa only where time is left)
  need=$((150 + 180 * n_seeds))
  (( $(date -d "$2" +%s) > $(date +%s) + need * 60 )) \
    || echo "WARNING: less than $((need / 60)) h $((need % 60)) min before --stop for $n_seeds seed(s)"
  flock -n "$HERE/.v10.lock" true || { echo "Already running: cat $STATUS"; exit 1; }
  setsid nohup bash "$SELF" --worker "$2" "$4" >> "$HERE/v10.log" 2>&1 < /dev/null &
  echo "Started (pid $!). Closing the terminal does not stop it. Progress: cat $STATUS"
  exit 0
fi
STOP=$2
SEEDS=$3
exec 9> "$HERE/.v10.lock"; flock -n 9 || exit 1
mkdir -p "$EXPORTS" "$MODELS"
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
die() { note "STOPPED: $*"; bash "$REPO/scripts/v10_export.sh" stopped > /dev/null 2>&1; exit 1; }
BUNDLE=$(python3 -c "import json;print(json.load(open('$HERE/BUNDLE.json'))['bundle'])" 2>/dev/null || echo unknown)
note "v0.10 worker on $(hostname); compute stop $STOP; seeds $SEEDS; bundle $BUNDLE"

# ---- 0. bundle integrity
python3 - "$HERE" <<'PY' || die "bundle files differ from BUNDLE.json; upload again"
import hashlib, json, sys
from pathlib import Path
here = Path(sys.argv[1]); b = json.loads((here / "BUNDLE.json").read_text())
bad = [n for n, h in b["files"].items() if hashlib.sha256((here / n).read_bytes()).hexdigest() != h]
print("\n".join(bad[:20])); sys.exit(1 if bad else 0)
PY
note "ok: bundle verified"

# ---- 1. hardware: eight GPUs (4 trainer + 4 rollout), room on disk for milestones and in /dev/shm for the weight channel
cd "$REPO" || die "no repo"
mkdir -p outputs/environment outputs/logs
LOGS=$REPO/outputs/logs
{ date -Is; nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nvidia-smi topo -m
  free -h; nproc; ldd --version | head -1; df -h . /tmp /dev/shm; cat /etc/os-release | head -3; } \
  > outputs/environment/hardware.txt 2>&1
GPUS=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$2 >= 80000 {printf "%s%s", s, $1; s=","}')
(( $(tr ',' '\n' <<< "$GPUS" | grep -c .) == 8 )) || die "v0.10 needs eight GPUs with >= 80 GB (found [$GPUS])"
FIRST=${GPUS%%,*}
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
(( ${FREE_GB:-0} >= 400 )) || note "WARNING: only ${FREE_GB} GB free disk (milestones 8 GB each, checkpoints ~100 GB, BIRD ~50 GB)"
SHM_GB=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
(( ${SHM_GB:-0} >= 64 )) || die "/dev/shm has ${SHM_GB} GB free; the weight channel needs about 50 GB (two smoke runs)"
note "hardware: GPUs [$GPUS], driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), glibc $(ldd --version | head -1 | awk '{print $NF}'), $(nproc) CPUs, ${FREE_GB} GB free disk, ${SHM_GB} GB /dev/shm"

# ---- 2. stdlib-only work first (BIRD and the 4B over parallel ranges), then the main environment
[[ -f /etc/pki/tls/certs/ca-bundle.crt ]] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
[[ -f /etc/ssl/certs/ca-certificates.crt ]] && export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
export HF_HOME=$REPO/.hf HF_ENDPOINT=https://huggingface.co PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONPATH=$REPO
export UV_PYTHON_INSTALL_DIR=$REPO/.runtime/python UV_CACHE_DIR=$REPO/.runtime/cache
early_python() {
  if [[ ! -x .runtime/bin/uv ]]; then
    local tmp asset=uv-x86_64-unknown-linux-musl.tar.gz
    tmp=$(mktemp -d)
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 600 \
      "https://github.com/astral-sh/uv/releases/download/0.10.0/$asset" -o "$tmp/$asset" || return 1
    # the check's "<file>: OK" goes to the log: stdout is the interpreter path the caller captures. On stdout it
    # made PY3 two lines, so on a fresh host the early fetch never started (v0.8-v0.10; it did after a restart)
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
  "$4" -m exec_feedback.v8.download --manifest configs/v10-models.json --model "$2" --out "$MODELS/$1" \
    --workers "$3" >> "$LOGS/download-$1.log" 2>&1
  echo $? > "$MODELS/$1.download.rc"
}
early_fetch() {  # part sha python
  "$3" -m exec_feedback.v9.fetch --part "$1" --sha256 "$2" --out data/raw --workers 16 >> "$LOGS/fetch-$1.log" 2>&1
  echo "$1 $?" > "$LOGS/early-fetch-$1.rc"
}
EARLY=0
if [[ -n $PY3 ]] && "$PY3" -c "import exec_feedback.v9.fetch, exec_feedback.v8.download" 2> /dev/null; then
  ( early_fetch train "$BIRD_TRAIN_SHA" "$PY3" ) &
  ( early_fetch dev "$BIRD_DEV_SHA" "$PY3" ) &
  ( download student "$STUDENT" 8 "$PY3" ) &
  EARLY=1
  note "BIRD train/dev (16 ranges each) and the 4B download started early with $PY3"
else
  note "early downloads not started (python '${PY3:-none}', see $LOGS/early-python.log); BIRD goes single-connection after the main venv"
fi
note "setup: main venv (about 6-8 min)"
if [[ ! -f outputs/environment/.bootstrap-ok ]]; then
  CUDA_VISIBLE_DEVICES=$FIRST bash scripts/bootstrap_h20.sh > "$LOGS/bootstrap.log" 2>&1 || die "bootstrap failed ($LOGS/bootstrap.log)"
  touch outputs/environment/.bootstrap-ok
fi
PY=$REPO/.venv/bin/python
UV=$REPO/.runtime/bin/uv
VPY=$REPO/.venv-vllm/bin/python
note "ok: main venv; vLLM venv in parallel"

# ---- 3. vLLM venv; NCCL across the trainer's four GPUs; BIRD and the 4B (single-connection fallback)
( { [[ -x $VPY ]] || { "$UV" venv .venv-vllm --python "$(readlink -f .venv/bin/python)" -q \
      && "$UV" pip install --python "$VPY" -r configs/vllm-0.11.2-lock.txt; }; } > "$LOGS/vllm-install.log" 2>&1
  rc=$?
  # v0.9's pins: the evaluation code it shares imports peft beside vLLM
  (( rc == 0 )) && { "$UV" pip install --python "$VPY" --no-deps peft==0.18.0 accelerate==1.12.0 >> "$LOGS/vllm-install.log" 2>&1; rc=$?; }
  echo "vllm $rc" > "$LOGS/vllm.rc" ) &
cat > "$LOGS/nccl_check.py" <<'PY'
import os, torch, torch.distributed as dist
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", device_id=torch.device("cuda", local))
x = torch.full((1 << 24,), float(dist.get_rank() + 1), device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
assert float(x[0]) == sum(range(1, dist.get_world_size() + 1)), float(x[0])
if dist.get_rank() == 0:
    print("nccl ok", torch.__version__, torch.cuda.nccl.version(), dist.get_world_size())
dist.barrier()
dist.destroy_process_group()
PY
CUDA_VISIBLE_DEVICES=$(cut -d, -f1-4 <<< "$GPUS") timeout 300 "$PY" -m torch.distributed.run --standalone --nnodes 1 \
  --nproc-per-node 4 "$LOGS/nccl_check.py" > "$LOGS/nccl-check.log" 2>&1 || die "NCCL all-reduce over four GPUs failed ($LOGS/nccl-check.log)"
note "ok: $(grep -h '^nccl ok' "$LOGS/nccl-check.log" | head -1) (main venv, four GPUs)"
(( EARLY )) || { ( download student "$STUDENT" 8 "$PY" ) & }
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

# ---- 4. CPU gates (no GPU visible: the tests run FSDP2 on a CPU process group) and freeze (smoke subset + main run)
gate_tests() { CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider tests/test_v10.py tests/test_v09.py \
  tests/test_v08.py tests/test_rft_v05.py tests/test_core.py tests/test_pipeline.py > "$LOGS/v10-tests$1.log" 2>&1; }
if ! gate_tests ""; then  # one rerun: the end-to-end scheduler tests are timing-dependent subprocess runs
  note "CPU tests failed once ($(tail -1 "$LOGS/v10-tests.log")); running them again"
  gate_tests "-rerun" || die "CPU tests failed twice ($LOGS/v10-tests.log, $LOGS/v10-tests-rerun.log)"
  TESTLOG=$LOGS/v10-tests-rerun.log
fi
note "ok: CPU tests ($(tail -1 "${TESTLOG:-$LOGS/v10-tests.log}"))"
freeze() {  # out config smoke_limit
  if [[ -f $1/manifest.json ]]; then
    "$PY" -m exec_feedback.v10 status --run "$1" > /dev/null 2>&1 && return 0
    # Code changed since this freeze (a fix after a failed smoke path): a new run under the new code. The old one is
    # kept beside it, never deleted; finished online runs are too expensive to drop silently, so those need a person.
    compgen -G "$1/train/*/result.json" > /dev/null && [[ $1 == "$RUN" ]] && die "$1 was frozen by other code and has finished online runs; decide by hand"
    mv "$1" "$1.stale-$(date +%s)"
    note "code changed: $(basename "$1") moved aside and frozen again"
  fi
  "$PY" -m exec_feedback.v10 freeze --registration "$HERE/registration" --raw-root data/raw --models-root "$MODELS" \
    --models-manifest configs/v10-models.json --imports "$HERE/imports" --config "$2" --out "$1" --smoke-limit "$3" \
    > "$LOGS/freeze-$(basename "$1").log" 2>&1
}
freeze "$SMOKE" configs/v10-smoke.json 16 || die "smoke freeze failed ($LOGS/freeze-smoke.log)"
freeze "$RUN" configs/h20-v10.json 0 || die "freeze failed ($LOGS/freeze-run.log)"
counts=$("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['counts'], sorted(m['imports']['adapters']), sorted(m['imports']['v09']['rl']))")
[[ $counts == "{'pool': 6682, 'dev': 615, 'validation': 1612, 'test': 1501} ['PTR_seed42', 'PTR_seed43', 'PTR_seed44'] ['RL_seed42', 'RL_seed43', 'RL_seed44']" ]] \
  || die "registration counts or imports differ: $counts"
note "ok: frozen run $("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['identity'][:12], 'K =', m['imports']['v09']['k'])"), $counts"
"$PY" -m exec_feedback.v10 progress --run "$RUN" --stage freeze --commit "$BUNDLE" --file "$PROGRESS" > /dev/null
bash scripts/v10_export.sh freeze > /dev/null 2>&1

# ---- 5. vLLM gate: the 4B generates on one GPU and returns log-probs (the weight-loading gate is a scheduler job)
while [[ ! -f $LOGS/vllm.rc ]]; do sleep 10; done
read -r what code < "$LOGS/vllm.rc"; (( code == 0 )) || die "vLLM venv install failed ($LOGS/vllm-install.log)"
while [[ ! -f $MODELS/student.download.rc ]]; do sleep 10; done
[[ $(cat "$MODELS/student.download.rc") == 0 ]] || die "4B download failed ($LOGS/download-student.log)"
CUDA_VISIBLE_DEVICES=$FIRST timeout 900 "$VPY" -c "
import traceback
from exec_feedback.v9.shutdown import hard_exit
try:
    from vllm import LLM, SamplingParams
    llm = LLM(model='$MODELS/student', dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=0.4)
    out = llm.generate(['SELECT 1 is a'], SamplingParams(max_tokens=8, temperature=1.0, logprobs=0))[0].outputs[0]
    assert out.logprobs and all(t in d for t, d in zip(out.token_ids, out.logprobs)), out
    print(out.text, [round(d[t].logprob, 3) for t, d in zip(out.token_ids, out.logprobs)])
    code = 0
except BaseException:
    traceback.print_exc()
    code = 1
hard_exit(code)" \
  > "$LOGS/vllm-gate.log" 2>&1 || die "vLLM cannot run here (exit $?; $LOGS/vllm-gate.log)"
"$UV" pip freeze --python "$VPY" > outputs/environment/vllm-packages.txt 2>&1
"$UV" pip freeze --python "$PY" > outputs/environment/main-packages.txt 2>&1
note "ok: vLLM $("$VPY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) samples on GPU $FIRST and returns log-probs; scheduler starts"

# ---- 6. schedule everything else
"$PY" -m exec_feedback.v10.orchestrate --run "$RUN" --smoke-run "$SMOKE" --models-root "$MODELS" --stop "$STOP" \
  --seeds "$SEEDS" --gpus "$GPUS" --repo "$REPO" --main-py "$PY" --vllm-py "$VPY" --status "$STATUS" \
  --progress "$PROGRESS" --commit "$BUNDLE" --export-hook "$REPO/scripts/v10_export.sh" --exports "$EXPORTS" \
  --log-dir "$LOGS/v10" > "$LOGS/orchestrate.log" 2>&1
rc=$?
(( rc == 0 )) || die "scheduler exit $rc ($LOGS/orchestrate.log)"
bash scripts/v10_export.sh final > /dev/null 2>&1 || note "WARNING: the final results package failed; the newest earlier one stands"
note "ALL DONE. Download exports/v10-results-*.tgz (+ .sha256), exports/进度-v10.md and every exports/v10-model-* file."
touch "$HERE/DONE"
