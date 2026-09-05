#!/usr/bin/env bash
set -euo pipefail

# Run this submission helper from the RELAY repository root on Matrix.
if [[ ! -f triton/experiments/rescore.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi

read -r -a k_values <<< "${RELAY_GEMV_SWEEP_K_VALUES:-64 128 256 512 1024 2048 4096}"
read -r -a stratifications <<< "${RELAY_GEMV_SWEEP_STRATIFICATIONS:-all issue temporal}"
read -r -a temporal_windows <<< "${RELAY_GEMV_TEMPORAL_WINDOWS:-4 16 64 128 256 512 1024}"
source_root="${RELAY_GEMV_SWEEP_RESULTS_ROOT:-${PWD}/triton/experiments/results/gemv-sweep}"
study="temporal-$(IFS=-; echo "${temporal_windows[*]}")"
results_root="${RELAY_GEMV_WINDOW_RESULTS_ROOT:-${PWD}/triton/experiments/results/window-rescore/${study}/gemv-sweep}"
plots_root="${RELAY_GEMV_WINDOW_PLOTS_ROOT:-${PWD}/triton/experiments/plots/window-rescore/${study}/gemv-sweep}"
partition="${RELAY_GEMV_WINDOW_PARTITION:-pbatch}"
walltime="${RELAY_GEMV_WINDOW_TIME:-00:30:00}"

for stratification in "${stratifications[@]}"; do
    case "${stratification}" in
        all|issue|temporal) ;;
        *)
            echo "error: invalid stratification ${stratification}" >&2
            exit 2
            ;;
    esac
    for k in "${k_values[@]}"; do
        source_k="${source_root}/k-${k}"
        result_k="${results_root}/k-${k}"
        plot_k="${plots_root}/k-${k}"
        log_dir="${result_k}/experiment-1/matrix/stratified-${stratification}/logs"
        mkdir -p "${log_dir}"
        name="relay-gemv-window-k${k}-${stratification}-h100"
        sbatch \
            --nodes=1 --ntasks=1 --gpus=1 \
            --partition="${partition}" --time="${walltime}" \
            --chdir="${PWD}" --job-name="${name}" \
            --output="${log_dir}/%x-%j.log" \
            triton/experiments/rescore-matrix-job.bash \
            --case gemv --gemv-k "${k}" --experiment 1 \
            --preserve-panel \
            --temporal-windows "${temporal_windows[@]}" \
            --counter-source "${source_k}" \
            --results-root "${result_k}" --plots-root "${plot_k}" \
            --stratification "${stratification}"
    done
done
