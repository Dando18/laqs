#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/appendix_common.py ]]; then
    echo "error: submit with the RELAY repository root as CWD" >&2
    exit 1
fi
if [[ "$#" -lt 1 ]]; then
    echo "usage: run-appendix-tuolumne-job.bash PYTHON-SCRIPT [arguments ...]" >&2
    exit 2
fi

module load "${RELAY_FINAL_ROCM_MODULE:-rocm/7.0.2}"
export PYTHONUNBUFFERED=1
export RELAY_TRITON_PYTHON_ROOT="${PWD}/triton/triton-lang/python"
export TRITON_HOME="${RELAY_APPENDIX_TUOLUMNE_CACHE_DIR:-${TMPDIR:-/tmp}/relay-appendix-triton-${USER:?}}"
export MPLCONFIGDIR="${RELAY_APPENDIX_MPLCONFIGDIR:-${TMPDIR:-/tmp}/relay-appendix-matplotlib-${USER:?}}"
unset CUDA_VISIBLE_DEVICES

relay_appendix_program="$1"
shift
exec "${PWD}/triton/.venv/bin/python" "${relay_appendix_program}" "$@"
