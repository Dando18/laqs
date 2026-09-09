#!/usr/bin/env bash
set -euo pipefail
# Schedulers relocate scripts; run from the repository root.
test -f triton/packet_layout/run.py
module load "${RELAY_FINAL_ROCM_MODULE:-rocm/7.0.2}"
export TRITON_HOME="${TMPDIR:-/tmp}/relay-packet-triton-${USER:?}"
export PYTHONUNBUFFERED=1
exec triton/.venv/bin/python triton/packet_layout/run.py "$@"
