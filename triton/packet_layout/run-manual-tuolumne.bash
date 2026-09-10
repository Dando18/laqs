#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root, including when the scheduler relocates this script.
# flux run -n1 -g1 -c8 -t 5m -q pdebug bash triton/packet_layout/run-manual-tuolumne.bash
# Repeat that command to continue the same checkpointed experiment.
test -f triton/packet_layout/debug-suite.py
exec bash triton/packet_layout/run-debug-tuolumne.bash \
    --manual --root triton/experiments/results/manual-layout-debug --minutes 4.5 \
    --cases row_column--small row_column--large sum--small --samples 11 --iterations 20 --warmup 3 "$@"
