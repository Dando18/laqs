#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Repeat the identical command to resume.
# flux run -n1 -g1 -c8 -t 5m -q pdebug bash triton/packet_layout/run-study-tuolumne.bash [row_column|sum]
test -f triton/packet_layout/debug-suite.py
relay_study_case="${1:-row_column}"
if [[ $# -gt 0 ]]; then shift; fi
case "$relay_study_case" in
    row_column) relay_study_reference=triton/experiments/results/packet-debug-20260908-215841 ;;
    sum) relay_study_reference=triton/experiments/results/packet-debug-split ;;
    *) echo 'First argument must be row_column or sum' >&2; exit 2 ;;
esac
exec bash triton/packet_layout/run-debug-tuolumne.bash \
    --study --root "triton/experiments/results/layout-study-${relay_study_case}-v1" \
    --address-reference "$relay_study_reference" --cases "${relay_study_case}--small" \
    --minutes 4.5 --samples 11 --iterations 20 --warmup 3 "$@"
