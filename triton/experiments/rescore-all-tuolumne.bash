#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/rescore.py ]]; then
    echo "error: run from the RELAY repository root" >&2
    exit 1
fi

counter_source="${RELAY_FINAL_COUNTER_SOURCE:-triton/experiments/results}"
pool_multiplier="${RELAY_FINAL_POOL_MULTIPLIER:-20}"
read -r -a stratifications <<< "${RELAY_FINAL_STRATIFICATIONS:-all issue temporal}"
queue="${RELAY_FINAL_RESCORE_QUEUE:-pbatch}"
walltime="${RELAY_FINAL_RESCORE_TIME:-30m}"
embedding_walltime="${RELAY_FINAL_EMBEDDING_RESCORE_TIME:-${RELAY_FINAL_RESCORE_TIME:-60m}}"
log_dir="${PWD}/triton/experiments/rescore-logs/tuolumne"
mkdir -p "${log_dir}"
for stratification in "${stratifications[@]}"; do
    for case in bias_relu softmax_bias embedding_bag gemv mvt gesummv stencil5; do
        name="relay-rescore-${stratification}-${case}-mi300a"
        case_walltime="${walltime}"
        if [[ "${case}" == "embedding_bag" ]]; then
            case_walltime="${embedding_walltime}"
        fi
        flux submit -N1 -n1 -g1 -t "${case_walltime}" -q "${queue}" \
            --cwd="${PWD}" --job-name="${name}" \
            --output="${log_dir}/${name}.out" \
            --error="${log_dir}/${name}.err" \
            triton/experiments/rescore-tuolumne-job.bash \
            --case "${case}" --counter-source "${counter_source}" \
            --stratification "${stratification}" \
            --pool-multiplier "${pool_multiplier}" --skip-existing
    done
done
