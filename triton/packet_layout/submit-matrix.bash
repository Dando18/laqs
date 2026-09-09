#!/usr/bin/env bash
set -euo pipefail
# Run from the RELAY repository root.
test -f triton/packet_layout/run.py
triton/packet_layout/build-matrix.bash
exec .venv/bin/python triton/packet_layout/submit.py --platform matrix "$@"
