#!/usr/bin/env bash
# Deployment study, unattended, on a fresh H20 host (the first GPU). Layout after
# `tar -xzf efa-deploy-bundle-*.tgz` in /root/efa-deploy:
#   exec-feedback-agent/  source (git HEAD)   registration/  v0.5 task files
#   imports/v11/  v0.11's manifest, milestone record, RLb_seed44's records and adapter at K
#   BUNDLE.json  sha256 of every file
# Usage:  bash exec-feedback-agent/scripts/deploy_run.sh --stop <ISO time with +08:00>
#         (absolute compute stop; keep >= 25 min before the booking ends for downloading results)
# Progress: cat /root/efa-deploy/STATUS.txt ; results: /root/efa-deploy/exports/ ; demo: scripts/deploy_demo.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$HERE/exec-feedback-agent
SELF=$REPO/scripts/deploy_run.sh
EXPORTS=$HERE/exports
STATUS=$HERE/STATUS.txt
PROGRESS=$EXPORTS/进度-deploy.md
MODELS=$HERE/models
RUN=$REPO/outputs/deploy/run
SMOKE=$REPO/outputs/deploy/smoke
BIRD_TRAIN_SHA=66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc
BIRD_DEV_SHA=cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630
STUDENT=Qwen/Qwen3-4B-Instruct-2507

if [[ ${1:-} != --worker ]]; then
  [[ ${1:-} == --stop && -n ${2:-} ]] || { echo "usage: bash $0 --stop <ISO time with +08:00>"; exit 2; }
  date -d "$2" > /dev/null 2>&1 || { echo "bad --stop $2"; exit 2; }
  # setup and smoke about 1.5 h, the evaluations about 1 h, the load tests about 2 h, the report minutes
  (( $(date -d "$2" +%s) > $(date +%s) + 5 * 3600 )) \
    || echo "WARNING: less than 5 h before --stop: the last load tests may not start (they go last)"
  flock -n "$HERE/.deploy.lock" true || { echo "Already running: cat $STATUS"; exit 1; }
  setsid nohup bash "$SELF" --worker "$2" >> "$HERE/deploy.log" 2>&1 < /dev/null &
  echo "Started (pid $!). Closing the terminal does not stop it. Progress: cat $STATUS"
  exit 0
fi
STOP=$2
exec 9> "$HERE/.deploy.lock"; flock -n 9 || exit 1
mkdir -p "$EXPORTS" "$MODELS"
note() { echo "$(TZ=CST-8 date '+%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
die() { note "STOPPED: $*"; bash "$REPO/scripts/deploy_export.sh" stopped > /dev/null 2>&1; exit 1; }
BUNDLE=$(python3 -c "import json;print(json.load(open('$HERE/BUNDLE.json'))['bundle'])" 2>/dev/null || echo unknown)
note "deployment worker on $(hostname); compute stop $STOP; bundle $BUNDLE"

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
rm -f "$LOGS"/early-fetch-*.rc "$LOGS/vllm.rc"  # a restart after STOPPED must not read the last attempt's exit codes
{ date -Is; nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nvidia-smi topo -m
  free -h; nproc; ldd --version | head -1; df -h . /tmp /dev/shm; cat /etc/os-release | head -3; } \
  > outputs/environment/hardware.txt 2>&1
GPUS=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits | awk -F', ' '$2 >= 80000 {printf "%s%s", s, $1; s=","}')
[[ -n $GPUS ]] || die "no GPU with >= 80 GB"
FIRST=${GPUS%%,*}
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
(( ${FREE_GB:-0} >= 100 )) || note "WARNING: only ${FREE_GB} GB free disk (BIRD train and dev ~50 GB extracted, 4B 8 GB, merged 4B 8 GB)"
MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
[[ $LIMIT =~ ^[0-9]+$ ]] && (( LIMIT / 1073741824 < MEM_GB )) && MEM_GB=$(( LIMIT / 1073741824 ))
note "hardware: GPUs [$GPUS], driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), glibc $(ldd --version | head -1 | awk '{print $NF}'), $(nproc) CPUs, ${MEM_GB} GB RAM, ${FREE_GB} GB free disk"
(( ${MEM_GB:-0} >= 40 )) || note "WARNING: ${MEM_GB} GB memory; the CPU merge needs about 20 GB besides the system"

# ---- 2. stdlib-only work first (BIRD dev and the 4B over parallel ranges), then the main environment
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
  "$4" -m exec_feedback.v8.download --manifest configs/v11-models.json --model "$2" --out "$MODELS/$1" \
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
note "ok: main venv (with peft for the merge); vLLM venv in parallel"

# ---- 3. vLLM venv; BIRD train/dev and the 4B (single-connection fallback if the early fetch failed)
# the marker is written only after a complete install: a half-installed venv from a failed attempt is rebuilt
( { [[ -f .venv-vllm/.efa-installed ]] || { rm -rf .venv-vllm && "$UV" venv .venv-vllm \
      --python "$(readlink -f .venv/bin/python)" -q && "$UV" pip install --python "$VPY" \
      -r configs/vllm-0.11.2-lock.txt && touch .venv-vllm/.efa-installed; }; } > "$LOGS/vllm-install.log" 2>&1
  echo "vllm $?" > "$LOGS/vllm.rc" ) &
(( EARLY )) || { ( download student "$STUDENT" 8 "$PY" ) & }
fetch() { [[ -f data/raw/$1.source.json ]] && return 0
  "$PY" -m exec_feedback fetch-bird --part "$1" --out data/raw --accept-terms > "$LOGS/fetch-$1-single.log" 2>&1; }
for part in train dev; do  # the validation tasks use BIRD train's databases
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

# ---- 4. CPU gates and freeze (smoke subset + main run)
gate_tests() { CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider tests/test_deploy.py tests/test_v08.py \
  tests/test_core.py tests/test_pipeline.py tests/test_service.py > "$LOGS/deploy-tests$1.log" 2>&1; }
if ! gate_tests ""; then  # one rerun: the end-to-end scheduler tests are timing-dependent subprocess runs
  note "CPU tests failed once ($(tail -1 "$LOGS/deploy-tests.log")); running them again"
  gate_tests "-rerun" || die "CPU tests failed twice ($LOGS/deploy-tests.log, $LOGS/deploy-tests-rerun.log)"
  TESTLOG=$LOGS/deploy-tests-rerun.log
fi
note "ok: CPU tests ($(tail -1 "${TESTLOG:-$LOGS/deploy-tests.log}"))"
freeze() {  # out config smoke_limit
  if [[ -d $1 && ! -f $1/manifest.json ]]; then  # a freeze cut short: kept aside, frozen again
    mv "$1" "$1.interrupted-$(date +%s)"
  fi
  if [[ -f $1/manifest.json ]]; then
    "$PY" -m exec_feedback.deploy status --run "$1" > /dev/null 2>&1 && return 0
    # Code changed since this freeze (a fix after a failed smoke path): a new run under the new code. The old one is
    # kept beside it, never deleted; evaluations and load tests cost hours, so those need a person.
    if [[ $1 == "$RUN" ]] && compgen -G "$1/eval/*/*/episodes.jsonl" > /dev/null; then
      die "$1 was frozen by other code and has evaluations; decide by hand"
    fi
    mv "$1" "$1.stale-$(date +%s)"
    note "code changed: $(basename "$1") moved aside and frozen again"
  fi
  "$PY" -m exec_feedback.deploy freeze --registration "$HERE/registration" --raw-root data/raw --models-root "$MODELS" \
    --models-manifest configs/v11-models.json --imports "$HERE/imports" --config "$2" --out "$1" --smoke-limit "$3" \
    > "$LOGS/freeze-$(basename "$1").log" 2>&1
}
freeze "$SMOKE" configs/deploy-smoke.json 16 || die "smoke freeze failed ($LOGS/freeze-smoke.log)"
freeze "$RUN" configs/h20-deploy.json 0 || die "freeze failed ($LOGS/freeze-run.log)"
counts=$("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['counts'], len(m['bench']['tasks']), sorted(m['imports']['v11']['adapters']))")
[[ $counts == "{'test': 1501, 'validation': 1612} 256 ['RLb_seed44']" ]] || die "registration counts or imports differ: $counts"
note "ok: frozen run $("$PY" -c "import json;m=json.load(open('$RUN/manifest.json'));a=m['imports']['v11']['adapters']['RLb_seed44'];print(m['identity'][:12], 'RLb_seed44 @', a['step'], a['adapter_hash'][:12])"), $counts"
"$PY" -m exec_feedback.deploy progress --run "$RUN" --stage freeze --commit "$BUNDLE" --file "$PROGRESS" > /dev/null
bash scripts/deploy_export.sh freeze > /dev/null 2>&1

# ---- 5. vLLM gate: the evaluation engine (the whole GPU, the base 4B with the deployed LoRA) answers greedily; vLLM's
# OpenAI-compatible server and FP8 are first exercised by the smoke run
while [[ ! -f $LOGS/vllm.rc ]]; do sleep 10; done
read -r what code < "$LOGS/vllm.rc"; (( code == 0 )) || die "vLLM venv install failed ($LOGS/vllm-install.log)"
while [[ ! -f $MODELS/student.download.rc ]]; do sleep 10; done
[[ $(cat "$MODELS/student.download.rc") == 0 ]] || die "4B download failed ($LOGS/download-student.log)"
CUDA_VISIBLE_DEVICES=$FIRST timeout 900 "$VPY" -c "
import traceback
from exec_feedback.v9.shutdown import hard_exit
try:
    import vllm.entrypoints.openai.api_server  # noqa: F401 - the load tests' server
    from pathlib import Path
    from exec_feedback.deploy.evaluate import engine_for, vllm_factory
    from exec_feedback.deploy.freeze import load
    run = Path('$RUN')
    frozen, _ = load(run)
    spec, adapter, quantization, _ = engine_for(run, frozen, 'lora')
    gen = vllm_factory(spec, frozen['settings'], adapter=adapter, quantization=quantization, name='gate')
    out = gen.generate([[{'role': 'user', 'content': 'Reply with one SQL query that returns the number 1.'}]])
    assert isinstance(out[0], dict) and out[0]['text'], out
    print(out[0]['text'][:80], gen.metadata)
    code = 0
except BaseException:
    traceback.print_exc()
    code = 1
hard_exit(code)" \
  > "$LOGS/vllm-gate.log" 2>&1 || die "vLLM cannot run here (exit $?; $LOGS/vllm-gate.log)"
"$UV" pip freeze --python "$VPY" > outputs/environment/vllm-packages.txt 2>&1
"$UV" pip freeze --python "$PY" > outputs/environment/main-packages.txt 2>&1
note "ok: vLLM $("$VPY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null) answers on GPU $FIRST with RLb_seed44's LoRA"

# ---- 6. every step in order on the first GPU: merge, evaluations, load tests, report (the smoke run first)
note "driver starts"
CUDA_VISIBLE_DEVICES=$FIRST "$PY" -m exec_feedback.deploy.orchestrate --run "$RUN" --smoke-run "$SMOKE" --stop "$STOP" \
  --repo "$REPO" --main-py "$PY" --vllm-py "$VPY" --status "$STATUS" --progress "$PROGRESS" --commit "$BUNDLE" \
  --export-hook "$REPO/scripts/deploy_export.sh" --log-dir "$LOGS/deploy" > "$LOGS/orchestrate.log" 2>&1
rc=$?
(( rc == 0 )) || note "driver exit $rc ($LOGS/orchestrate.log)"
bash scripts/deploy_export.sh final > /dev/null 2>> "$LOGS/export-final.log" \
  || note "WARNING: the final export failed ($LOGS/export-final.log); the newest earlier results package stands"
note "ALL DONE. Download exports/deploy-results-*-final.tgz (+ .sha256) and exports/进度-deploy.md. Demo: bash exec-feedback-agent/scripts/deploy_demo.sh"
touch "$HERE/DONE"
