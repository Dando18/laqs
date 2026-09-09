#!/usr/bin/env bash
set -euo pipefail
# Schedulers relocate scripts; launch from the repository root.
test -f triton/packet_layout/debug-suite.py
module load "${RELAY_FINAL_ROCM_MODULE:-rocm/7.0.2}"
export TRITON_HOME="${TMPDIR:-/tmp}/relay-packet-triton-${USER:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec triton/.venv/bin/python triton/packet_layout/debug-suite.py --platform tuolumne "$@"
