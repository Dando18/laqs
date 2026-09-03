#!/usr/bin/env bash
set -euo pipefail

# Slurm stages jobs separately; the experiment expects the RELAY root as CWD.
if [[ ! -f triton/experiments/run.py ]]; then
    echo "error: submit with the RELAY repository root as the job CWD" >&2
    exit 1
fi

module load "${RELAY_FINAL_CUDA_MODULE:-cuda/13.1.1}"
module load "${RELAY_FINAL_NCU_MODULE:-nsight-compute/2025.3.0}"
if ! command -v ncu >/dev/null 2>&1; then
    echo "error: Nsight Compute CLI is unavailable" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TRITON_HOME="${RELAY_FINAL_MATRIX_CACHE_DIR:-${TMPDIR:-/tmp}/relay-final-triton-${USER:?}}"
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES

exec "${PWD}/triton/.venv-matrix/bin/python" triton/experiments/run.py "$@"
