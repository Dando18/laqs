#!/usr/bin/env bash
set -euo pipefail

# Flux stages jobs separately; the experiment expects the RELAY root as CWD.
if [[ ! -f triton/experiments/run.py ]]; then
    echo "error: submit with the RELAY repository root as the job CWD" >&2
    exit 1
fi

module load "${RELAY_FINAL_ROCM_MODULE:-rocm/7.0.2}"
export PYTHONUNBUFFERED=1
export TRITON_HOME="${RELAY_FINAL_TUOLUMNE_CACHE_DIR:-${TMPDIR:-/tmp}/relay-final-triton-${USER:?}}"
unset CUDA_VISIBLE_DEVICES

exec "${PWD}/triton/.venv/bin/python" triton/experiments/run.py "$@"
