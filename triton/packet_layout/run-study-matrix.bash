#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Repeat the identical command to resume.
# srun -n1 -G1 -c8 -p pdebug -t 00:05:00 bash triton/packet_layout/run-study-matrix.bash
test -f triton/packet_layout/debug-suite.py
exec bash triton/packet_layout/run-debug-matrix.bash \
    --study --root triton/experiments/results/layout-study-gemm-v1 \
    --address-reference triton/experiments/results/packet-debug-split --cases gemm--asymmetric \
    --minutes 4.5 --samples 11 --iterations 20 --warmup 3 "$@"
