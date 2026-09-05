#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/rescore.py ]]; then
    echo "error: run from the RELAY repository root" >&2
    exit 1
fi

counter_source="${RELAY_FINAL_COUNTER_SOURCE:-triton/experiments/results}"
pool_multiplier="${RELAY_FINAL_POOL_MULTIPLIER:-20}"
read -r -a stratifications <<< "${RELAY_FINAL_STRATIFICATIONS:-all issue temporal}"
partition="${RELAY_FINAL_RESCORE_PARTITION:-pbatch}"
walltime="${RELAY_FINAL_RESCORE_TIME:-00:30:00}"
embedding_walltime="${RELAY_FINAL_EMBEDDING_RESCORE_TIME:-${RELAY_FINAL_RESCORE_TIME:-01:00:00}}"
log_dir="${PWD}/triton/experiments/rescore-logs/matrix"
mkdir -p "${log_dir}"
for stratification in "${stratifications[@]}"; do
    for case in bias_relu softmax_bias embedding_bag gemv mvt gesummv stencil5; do
        name="relay-rescore-${stratification}-${case}-h100"
        case_walltime="${walltime}"
        if [[ "${case}" == "embedding_bag" ]]; then
            case_walltime="${embedding_walltime}"
        fi
        sbatch --nodes=1 --ntasks=1 --gpus=1 \
            --partition="${partition}" --time="${case_walltime}" \
            --chdir="${PWD}" --job-name="${name}" \
            --output="${log_dir}/%x-%j.log" \
            triton/experiments/rescore-matrix-job.bash \
            --case "${case}" --counter-source "${counter_source}" \
            --stratification "${stratification}" \
            --pool-multiplier "${pool_multiplier}" --skip-existing
    done
done
