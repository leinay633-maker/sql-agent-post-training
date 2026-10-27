#!/usr/bin/env bash
# Installs an isolated Python without using system Python, conda, root or Git auth.
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in curl tar sha256sum getconf; do command -v "$tool" >/dev/null || { echo "Missing command: $tool" >&2; exit 2; }; done
[[ "$(uname -m)" == x86_64 ]] || { echo 'This profile targets x86_64 H20 Linux.' >&2; exit 2; }
glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
awk -v v="$glibc" 'BEGIN {split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=28))}' || {
  echo 'glibc >= 2.28 required by this torch wheel. Select an Ubuntu 22.04+ container image; replacing Python alone cannot fix glibc. Do not change host drivers.' >&2; exit 2;
}
export PYTHONUTF8=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_ENDPOINT=https://huggingface.co
export HF_HOME="$PWD/.hf"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
export UV_CACHE_DIR="$PWD/.runtime/cache"
mkdir -p .runtime/bin outputs/environment
if [[ ! -x .runtime/bin/uv ]]; then
  release=0.10.0
  asset=uv-x86_64-unknown-linux-musl.tar.gz
  url="https://github.com/astral-sh/uv/releases/download/$release/$asset"
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 1800 "$url" -o "$tmp/$asset"
  # Fixed upstream GitHub release asset digest (uv 0.10.0, x86_64 musl).
  hash=312d37f31b6f2c3bfc65668ba0efea9f1f9eaf7bc3209fe1a109e5cf861b95fa
  (cd "$tmp"; printf '%s  %s\n' "$hash" "$asset" | sha256sum --check)
  tar -xzf "$tmp/$asset" -C "$tmp" uv-x86_64-unknown-linux-musl/uv
  cp "$tmp/uv-x86_64-unknown-linux-musl/uv" .runtime/bin/uv
  chmod +x .runtime/bin/uv
fi
uv="$PWD/.runtime/bin/uv"
"$uv" --version | tee outputs/environment/uv.txt
"$uv" python install 3.12
if [[ ! -x .venv/bin/python ]]; then "$uv" venv --python 3.12 --seed .venv; fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use a fresh Python 3.12 venv"'
# Install only torch from the PyTorch index, everything else from PyPI.
"$uv" pip install --python .venv/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
"$uv" pip install --python .venv/bin/python -e '.[dev,server,ml]'
"$uv" pip check --python .venv/bin/python
"$uv" pip freeze --python .venv/bin/python > outputs/environment/resolved-requirements.txt
.venv/bin/python - <<'PY'
import json, platform, subprocess, sys, torch
from pathlib import Path
assert torch.cuda.is_available(), 'CUDA unavailable; inspect host driver, do not install a driver inside this container'
assert torch.cuda.device_count() == 1, 'Set CUDA_VISIBLE_DEVICES to one GPU'
assert torch.cuda.is_bf16_supported(), 'This profile needs BF16'
x = torch.ones((64, 64), device='cuda', dtype=torch.bfloat16, requires_grad=True)
(x @ x).float().mean().backward()
torch.cuda.synchronize()
p = torch.cuda.get_device_properties(0)
result = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
          'cuda_runtime': torch.version.cuda, 'gpu': p.name, 'vram_bytes': p.total_memory,
          'bf16_matmul_backward': True,
          'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}
Path('outputs/environment/gpu.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
PY
.venv/bin/python scripts/verify.py --out "outputs/environment/acceptance-$(date +%s)" --require-ml
echo 'Environment gate passed. This is NOT Qwen4B training or BIRD accuracy. See docs/H20_RUNBOOK.md next.'
