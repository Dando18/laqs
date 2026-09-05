#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/run-search.py ]]; then
    echo "error: submit with the RELAY repository root as CWD" >&2
    exit 1
fi

module load "${RELAY_FINAL_ROCM_MODULE:-rocm/7.0.2}"
export PYTHONUNBUFFERED=1
export RELAY_TRITON_PYTHON_ROOT="${PWD}/triton/triton-lang/python"
export TRITON_HOME="${RELAY_SEARCH_TUOLUMNE_CACHE_DIR:-${TMPDIR:-/tmp}/relay-search-triton-${USER:?}}"
export MPLCONFIGDIR="${RELAY_SEARCH_MPLCONFIGDIR:-${TMPDIR:-/tmp}/relay-search-matplotlib-${USER:?}}"
unset CUDA_VISIBLE_DEVICES

exec "${PWD}/triton/.venv/bin/python" triton/experiments/run-search.py "$@"
