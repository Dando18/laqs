#!/usr/bin/env bash
set -euo pipefail

# This is a submission helper and must be run from the RELAY repository root.
if [[ ! -f triton/experiments/run.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi
if [[ "$#" -lt 2 ]]; then
    echo "usage: submit.bash EXPERIMENT {tuolumne|matrix} [run.py arguments ...]" >&2
    exit 2
fi

experiment="$1"
platform="$2"
shift 2
if [[ ! "${experiment}" =~ ^[123]$ ]]; then
    echo "error: experiment must be 1, 2, or 3" >&2
    exit 2
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
layout_count="${RELAY_FINAL_LAYOUTS:-100}"
seed="${RELAY_FINAL_SEED:-0}"
profile_launches="${RELAY_FINAL_PROFILE_LAUNCHES:-3}"
log_dir="${PWD}/triton/experiments/results/experiment-${experiment}/${platform}/logs"
mkdir -p "${log_dir}"

if [[ "${platform}" == "tuolumne" ]]; then
    queue="${RELAY_FINAL_TUOLUMNE_QUEUE:-pbatch}"
    wall_time="${RELAY_FINAL_TUOLUMNE_WALL_TIME:-8h}"
    for kernel in "${kernels[@]}"; do
        name="relay-e${experiment}-${kernel}-mi300a"
        flux submit \
            -N 1 -n 1 -g 1 -q "${queue}" -t "${wall_time}" \
            --cwd="${PWD}" \
            --job-name="${name}" \
            --output="${log_dir}/${name}.out" \
            --error="${log_dir}/${name}.err" \
            triton/experiments/run-tuolumne-job.bash \
            --experiment "${experiment}" \
            --platform tuolumne \
            --case "${kernel}" \
            --layouts "${layout_count}" \
            --seed "${seed}" \
            --profile-launches "${profile_launches}" \
            "$@"
    done
elif [[ "${platform}" == "matrix" ]]; then
    partition="${RELAY_FINAL_MATRIX_PARTITION:-pbatch}"
    wall_time="${RELAY_FINAL_MATRIX_WALL_TIME:-06:00:00}"
    for kernel in "${kernels[@]}"; do
        name="relay-e${experiment}-${kernel}-h100"
        sbatch \
            --nodes=1 \
            --ntasks=1 \
            --gpus=1 \
            --partition="${partition}" \
            --time="${wall_time}" \
            --chdir="${PWD}" \
            --job-name="${name}" \
            --output="${log_dir}/%x-%j.log" \
            triton/experiments/run-matrix-job.bash \
            --experiment "${experiment}" \
            --platform matrix \
            --case "${kernel}" \
            --layouts "${layout_count}" \
            --seed "${seed}" \
            --profile-launches "${profile_launches}" \
            "$@"
    done
else
    echo "error: platform must be tuolumne or matrix" >&2
    exit 2
fi
