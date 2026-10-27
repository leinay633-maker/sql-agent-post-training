#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
python3 -m venv .venv
extras='dev,server'
if [[ "${1:-}" == '--gpu' ]]; then extras='dev,server,ml,gpu'; elif [[ "${1:-}" == '--ml-cpu' ]]; then extras='dev,server,ml'; elif [[ -n "${1:-}" ]]; then echo 'Usage: ./setup.sh [--gpu|--ml-cpu]' >&2; exit 2; fi
.venv/bin/python -m pip install -e ".[${extras}]"
.venv/bin/python -m exec_feedback check
