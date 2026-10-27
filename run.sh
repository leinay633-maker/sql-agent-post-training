#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then echo 'Run ./setup.sh first (or ./setup.sh --gpu on the GPU host).' >&2; exit 2; fi
exec .venv/bin/python -m exec_feedback "$@"
