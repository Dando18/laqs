#!/usr/bin/env bash
set -euo pipefail

# Run from the RELAY repository root; Flux may stage submitted jobs elsewhere.
if [[ ! -f triton/run-stage1-quotient-level-counters.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
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

WALL_TIME="${RELAY_QUOTIENT_COUNTER_WALL_TIME:-30m}"

for kernel in "${KERNELS[@]}"; do
    read -r -a tile_shape <<< "${TILE_SHAPES[$kernel]}"
    flux submit \
        -N 1 -x \
        -t "${WALL_TIME}" \
        --job-name="laqs-quotient-${kernel}" \
        triton/.venv/bin/python \
        triton/run-stage1-quotient-level-counters.py \
        --case "${kernel}" \
        --tile-shape "${tile_shape[@]}" \
        --profile-launches 3 \
        --rerun \
        --quiet
done
