#!/usr/bin/env bash
set -euo pipefail
# Schedulers relocate scripts; launch from the repository root.
test -f triton/packet_layout/debug-suite.py
module load "${RELAY_FINAL_CUDA_MODULE:-cuda/13.1.1}"
for relay_packet_arg in "$@"; do
    if [[ "$relay_packet_arg" == --profile-addresses ]]; then
        module load "${RELAY_NCU_MODULE:-nsight-compute/2025.3.0}"
        ncu --version >/dev/null
        break
    fi
done
export TRITON_HOME="${TMPDIR:-/tmp}/relay-packet-triton-${USER:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES
exec triton/.venv-matrix/bin/python triton/packet_layout/debug-suite.py --platform matrix "$@"
