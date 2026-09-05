#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/appendix_common.py ]]; then
    echo "error: submit with the RELAY repository root as CWD" >&2
    exit 1
fi
if [[ "$#" -lt 1 ]]; then
    echo "usage: run-appendix-matrix-job.bash PYTHON-SCRIPT [arguments ...]" >&2
    exit 2
fi

module load "${RELAY_FINAL_CUDA_MODULE:-cuda/13.1.1}"
export PYTHONUNBUFFERED=1
export TRITON_HOME="${RELAY_APPENDIX_MATRIX_CACHE_DIR:-${TMPDIR:-/tmp}/relay-appendix-triton-${USER:?}}"
export MPLCONFIGDIR="${RELAY_APPENDIX_MPLCONFIGDIR:-${TMPDIR:-/tmp}/relay-appendix-matplotlib-${USER:?}}"
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES

relay_appendix_program="$1"
shift
exec "${PWD}/triton/.venv-matrix/bin/python" "${relay_appendix_program}" "$@"
