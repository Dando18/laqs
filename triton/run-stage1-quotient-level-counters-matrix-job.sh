#!/usr/bin/env bash

set -euo pipefail

# Slurm stages job scripts separately; this job expects the RELAY root as CWD.
if [[ ! -f triton/run-stage1-quotient-level-counters-matrix.py ]]; then
    echo "error: run this job with the RELAY repository root as its CWD" >&2
    exit 1
fi

relay_cuda_module="${RELAY_TRITON_MATRIX_CUDA_MODULE:-cuda/13.1.1}"
relay_ncu_module="${RELAY_TRITON_MATRIX_NCU_MODULE:-nsight-compute/2025.3.0}"
relay_python="${PWD}/triton/.venv-matrix/bin/python"

module load "${relay_cuda_module}" "${relay_ncu_module}"

if [[ ! -x "${relay_python}" ]]; then
    echo "error: Matrix Triton is not installed; run triton/install-matrix.sh" >&2
    exit 1
fi
if ! command -v ncu >/dev/null 2>&1; then
    echo "error: ncu is unavailable after loading ${relay_ncu_module}" >&2
    exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
    echo "error: the counter experiment requires exactly one visible GPU" >&2
    exit 1
fi

export TRITON_HOME="${RELAY_TRITON_MATRIX_CACHE_DIR:-${TMPDIR:-/tmp}/relay-triton-matrix-${USER:?}}"
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES

exec "${relay_python}" \
    triton/run-stage1-quotient-level-counters-matrix.py \
    "$@"
