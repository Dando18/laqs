#!/usr/bin/env bash
set -euo pipefail
# Run from the RELAY repository root, including under Flux.
test -f tests/test_packet_layout_gpu.py
module load rocm/7.0.2
export TRITON_HOME="${TMPDIR:-/tmp}/relay-packet-smoke-${USER:?}"
exec triton/.venv/bin/python -m unittest discover -s tests -p test_packet_layout_gpu.py
