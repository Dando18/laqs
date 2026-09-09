#!/usr/bin/env bash
set -euo pipefail
# Schedulers relocate scripts; launch from the repository root.
test -f triton/packet_layout/copy_service_probe.cu
module load "${RELAY_FINAL_CUDA_MODULE:-cuda/13.1.1}"
module load "${RELAY_NCU_MODULE:-nsight-compute/2025.3.0}"
ncu --version >/dev/null
export PYTHONUNBUFFERED=1
exec .venv/bin/python triton/packet_layout/copy_service_probe.py "$@"
