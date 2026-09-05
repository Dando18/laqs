#!/usr/bin/env bash
set -euo pipefail

# This submission helper must be run from the RELAY repository root.
if [[ ! -f triton/experiments/run.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi
if [[ "$#" -lt 1 ]]; then
    echo "usage: submit-gemv-sweep.bash {tuolumne|matrix} [run.py arguments ...]" >&2
    exit 2
fi

platform="$1"
shift
experiment="${RELAY_GEMV_SWEEP_EXPERIMENT:-1}"
layout_count="${RELAY_GEMV_SWEEP_LAYOUTS:-100}"
seed="${RELAY_GEMV_SWEEP_SEED:-0}"
profile_launches="${RELAY_GEMV_SWEEP_PROFILE_LAUNCHES:-3}"
pool_multiplier="${RELAY_GEMV_SWEEP_POOL_MULTIPLIER:-20}"
read -r -a k_values <<< "${RELAY_GEMV_SWEEP_K_VALUES:-64 128 256 512 1024 2048 4096}"
read -r -a stratifications <<< "${RELAY_GEMV_SWEEP_STRATIFICATIONS:-all}"
results_root="${RELAY_GEMV_SWEEP_RESULTS_ROOT:-${PWD}/triton/experiments/results/gemv-sweep}"
plots_root="${RELAY_GEMV_SWEEP_PLOTS_ROOT:-${PWD}/triton/experiments/plots/gemv-sweep}"

if [[ ! "${experiment}" =~ ^[123]$ ]]; then
    echo "error: RELAY_GEMV_SWEEP_EXPERIMENT must be 1, 2, or 3" >&2
    exit 2
fi
for stratification in "${stratifications[@]}"; do
    case "${stratification}" in
        all|issue|temporal) ;;
        *)
            echo "error: invalid stratification ${stratification}" >&2
            exit 2
            ;;
    esac
done

if [[ "${platform}" == "tuolumne" ]]; then
    queue="${RELAY_GEMV_SWEEP_TUOLUMNE_QUEUE:-pbatch}"
    wall_time="${RELAY_GEMV_SWEEP_TUOLUMNE_WALL_TIME:-12h}"
    for stratification in "${stratifications[@]}"; do
        for k in "${k_values[@]}"; do
            k_root="${results_root}/k-${k}"
            log_dir="${k_root}/experiment-${experiment}/${platform}/stratified-${stratification}/logs"
            mkdir -p "${log_dir}"
            name="relay-gemv-k${k}-${stratification}-mi300a"
            flux submit \
                -N 1 -n 1 -g 1 -q "${queue}" -t "${wall_time}" \
                --cwd="${PWD}" --job-name="${name}" \
                --output="${log_dir}/${name}.out" \
                --error="${log_dir}/${name}.err" \
                triton/experiments/run-tuolumne-job.bash \
                --experiment "${experiment}" --platform tuolumne --case gemv \
                --gemv-k "${k}" --layouts "${layout_count}" --seed "${seed}" \
                --stratification "${stratification}" \
                --pool-multiplier "${pool_multiplier}" \
                --profile-launches "${profile_launches}" \
                --results-root "${k_root}" --plots-root "${plots_root}/k-${k}" \
                "$@"
        done
    done
elif [[ "${platform}" == "matrix" ]]; then
    partition="${RELAY_GEMV_SWEEP_MATRIX_PARTITION:-pbatch}"
    wall_time="${RELAY_GEMV_SWEEP_MATRIX_WALL_TIME:-12:00:00}"
    for stratification in "${stratifications[@]}"; do
        for k in "${k_values[@]}"; do
            k_root="${results_root}/k-${k}"
            log_dir="${k_root}/experiment-${experiment}/${platform}/stratified-${stratification}/logs"
            mkdir -p "${log_dir}"
            name="relay-gemv-k${k}-${stratification}-h100"
            sbatch \
                --nodes=1 --ntasks=1 --gpus=1 \
                --partition="${partition}" --time="${wall_time}" \
                --chdir="${PWD}" --job-name="${name}" \
                --output="${log_dir}/%x-%j.log" \
                triton/experiments/run-matrix-job.bash \
                --experiment "${experiment}" --platform matrix --case gemv \
                --gemv-k "${k}" --layouts "${layout_count}" --seed "${seed}" \
                --stratification "${stratification}" \
                --pool-multiplier "${pool_multiplier}" \
                --profile-launches "${profile_launches}" \
                --results-root "${k_root}" --plots-root "${plots_root}/k-${k}" \
                "$@"
        done
    done
else
    echo "error: platform must be tuolumne or matrix" >&2
    exit 2
fi
