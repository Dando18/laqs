#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; repeat to resume completed stages/processes.
# srun -n1 -G1 -c8 -p pdebug -t 00:05:00 bash triton/packet_layout/run-manual-matrix.bash
test -f triton/packet_layout/debug-suite.py
exec bash triton/packet_layout/run-debug-matrix.bash \
    --manual --root triton/experiments/results/manual-layout-debug --minutes 4.5 \
    --cases row_column--small row_column--large gemm--asymmetric --samples 11 --iterations 20 --warmup 3 "$@"
