#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f triton/run-stage1-quotient-level-counters-matrix-job.sh ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi
if [[ ! -x triton/.venv-matrix/bin/python ]]; then
    echo "error: Matrix Triton is not installed; run triton/install-matrix.sh" >&2
    exit 1
fi

KERNELS=(
    bias_relu
    softmax_bias
    embedding_bag
    gemv
    mvt
    gesummv
    stencil5
)

declare -A TILE_SHAPES=(
    [bias_relu]="256"
    [softmax_bias]="64 256"
    [embedding_bag]="64 128"
    [gemv]="64 64"
    [mvt]="64 64"
    [gesummv]="64 64"
    [stencil5]="64 64"
)

relay_log_dir="${PWD}/triton/results/stage1-quotient-level-counter-logs-q32b-matrix"
relay_profile_launches="${RELAY_QUOTIENT_COUNTER_PROFILE_LAUNCHES:-3}"
relay_partition="${RELAY_TRITON_MATRIX_COUNTER_PARTITION:-pdebug}"
relay_wall_time="${RELAY_TRITON_MATRIX_COUNTER_WALL_TIME:-00:05:00}"
mkdir -p "${relay_log_dir}"

for kernel in "${KERNELS[@]}"; do
    read -r -a tile_shape <<< "${TILE_SHAPES[$kernel]}"
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --gpus=1 \
        --partition="${relay_partition}" \
        --time="${relay_wall_time}" \
        --chdir="${PWD}" \
        --job-name="laqs-ncu-q32-${kernel}" \
        --output="${relay_log_dir}/%x-%j.log" \
        triton/run-stage1-quotient-level-counters-matrix-job.sh \
        --case "${kernel}" \
        --tile-shape "${tile_shape[@]}" \
        --transaction-bytes 32 \
        --profile-launches "${relay_profile_launches}" \
        --quiet
done
