#!/usr/bin/env bash
set -euo pipefail

# Run from the RELAY repository root; Flux may stage submitted jobs elsewhere.
if [[ ! -f triton/run-stage1-random-layout-counters.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi

kernels=(
    bias_relu
    softmax_bias
    embedding_bag
    gemv
    mvt
    gesummv
    stencil5
)
read -r -a byte_levels <<< "${RELAY_RANDOM_LAYOUT_BYTE_LEVELS:-32 64 128}"

layout_count="${RELAY_RANDOM_LAYOUT_COUNT:-100}"
seed="${RELAY_RANDOM_LAYOUT_SEED:-0}"
profile_launches="${RELAY_RANDOM_LAYOUT_PROFILE_LAUNCHES:-1}"
wall_time="${RELAY_RANDOM_LAYOUT_WALL_TIME:-2h}"
queue="${RELAY_RANDOM_LAYOUT_QUEUE:-}"
log_dir="${RELAY_RANDOM_LAYOUT_LOG_DIR:-${PWD}/triton/results/stage1-random-layout-counter-logs}"
mkdir -p "${log_dir}"

flux_options=(-N 1 -n 1 -g 1 -x -t "${wall_time}")
if [[ -n "${queue}" ]]; then
    flux_options+=(-q "${queue}")
fi

for byte_level in "${byte_levels[@]}"; do
    for kernel in "${kernels[@]}"; do
        name="laqs-random-q${byte_level}-${kernel}"
        flux submit \
            "${flux_options[@]}" \
            --job-name="${name}" \
            --output="${log_dir}/${name}.out" \
            --error="${log_dir}/${name}.err" \
            triton/.venv/bin/python \
            triton/run-stage1-random-layout-counters.py \
            --kernel "${kernel}" \
            --byte-level "${byte_level}" \
            --layouts "${layout_count}" \
            --seed "${seed}" \
            --profile-launches "${profile_launches}" \
            --quiet
    done
done
