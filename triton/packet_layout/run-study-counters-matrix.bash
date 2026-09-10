#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root after the GEMM study completes. Repeat to resume.
# srun -n1 -G1 -c8 -p pdebug -t 00:05:00 bash triton/packet_layout/run-study-counters-matrix.bash
test -f triton/packet_layout/run.py
module load "${RELAY_FINAL_CUDA_MODULE:-cuda/13.1.1}"
module load "${RELAY_NCU_MODULE:-nsight-compute/2025.3.0}"
export TRITON_HOME="${TMPDIR:-/tmp}/relay-packet-triton-${USER:?}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES
exec timeout --signal=TERM --kill-after=5s 270s triton/.venv-matrix/bin/python triton/packet_layout/run.py \
    --stage study_profile --platform matrix --case gemm--asymmetric \
    --root triton/experiments/results/layout-study-gemm-v1 --warmup 3 --iterations 20 "$@"
