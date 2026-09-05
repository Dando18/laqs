#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/run-search.py ]]; then
    echo "error: submit with the RELAY repository root as CWD" >&2
    exit 1
fi

module load "${RELAY_FINAL_CUDA_MODULE:-cuda/13.1.1}"
module load "${RELAY_FINAL_NCU_MODULE:-nsight-compute/2025.3.0}"
command -v ncu >/dev/null
export PYTHONUNBUFFERED=1
export TRITON_HOME="${RELAY_SEARCH_MATRIX_CACHE_DIR:-${TMPDIR:-/tmp}/relay-search-triton-${USER:?}}"
export MPLCONFIGDIR="${RELAY_SEARCH_MPLCONFIGDIR:-${TMPDIR:-/tmp}/relay-search-matplotlib-${USER:?}}"
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES

exec "${PWD}/triton/.venv-matrix/bin/python" triton/experiments/run-search.py "$@"
