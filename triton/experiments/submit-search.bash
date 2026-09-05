#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/run-search.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi
if [[ "$#" -lt 2 ]]; then
    echo "usage: submit-search.bash EXPERIMENT {tuolumne|matrix} [run-search.py arguments ...]" >&2
    exit 2
fi
relay_search_experiment="$1"
relay_search_platform="$2"
shift 2
if [[ ! "${relay_search_experiment}" =~ ^[456]$ ]]; then
    echo "error: experiment must be 4, 5, or 6" >&2
    exit 2
fi

relay_search_cases=(
    vector_add--small vector_add--large
    vector_exp--small vector_exp--large
    low_mem_dropout--small low_mem_dropout--large
    softmax--small softmax--large
    sum--small sum--large
    layer_norm--small layer_norm--large
    gemm--square gemm--asymmetric
    bf16xint16_gemm--projection bf16xint16_gemm--output
    int4_gemm--decode int4_gemm--prefill
    fp8_gemm--small fp8_gemm--large
    gather_gemv--small gather_gemv--large
    template_attention--default
    jagged_sum--small jagged_sum--large
    jagged_mean--small jagged_mean--large
    jagged_softmax--small jagged_softmax--large
)
if [[ -n "${RELAY_SEARCH_CASES:-}" ]]; then
    read -r -a relay_search_cases <<< "${RELAY_SEARCH_CASES}"
fi
relay_search_results="${RELAY_FINAL_RESULTS_ROOT:-${PWD}/triton/experiments/results}"
relay_search_plots="${RELAY_FINAL_PLOTS_ROOT:-${PWD}/triton/experiments/plots}"
relay_search_profiles="${RELAY_SEARCH_PROFILE_LAUNCHES:-3}"

for relay_search_case in "${relay_search_cases[@]}"; do
    relay_search_operator="${relay_search_case%%--*}"
    relay_search_config="${relay_search_case#*--}"
    relay_search_log_dir="${relay_search_results}/experiment-${relay_search_experiment}/${relay_search_platform}/logs"
    mkdir -p "${relay_search_log_dir}"
    if [[ "${relay_search_platform}" == "tuolumne" ]]; then
        relay_search_queue="${RELAY_SEARCH_TUOLUMNE_QUEUE:-pbatch}"
        case "${relay_search_experiment}" in
            4) relay_search_time="${RELAY_SEARCH_TUOLUMNE_TIME_E4:-1h}" ;;
            5) relay_search_time="${RELAY_SEARCH_TUOLUMNE_TIME_E5:-1h}" ;;
            6) relay_search_time="${RELAY_SEARCH_TUOLUMNE_TIME_E6:-90m}" ;;
        esac
        relay_search_name="relay-e${relay_search_experiment}-${relay_search_case}-mi300a"
        flux submit -N 1 -n 1 -g 1 -q "${relay_search_queue}" -t "${relay_search_time}" \
            --cwd="${PWD}" --job-name="${relay_search_name}" \
            --output="${relay_search_log_dir}/${relay_search_name}.out" \
            --error="${relay_search_log_dir}/${relay_search_name}.err" \
            triton/experiments/run-search-tuolumne-job.bash \
            --experiment "${relay_search_experiment}" --platform tuolumne \
            --operator "${relay_search_operator}" --config "${relay_search_config}" \
            --profile-launches "${relay_search_profiles}" \
            --results-root "${relay_search_results}" --plots-root "${relay_search_plots}" "$@"
    elif [[ "${relay_search_platform}" == "matrix" ]]; then
        relay_search_partition="${RELAY_SEARCH_MATRIX_PARTITION:-pbatch}"
        case "${relay_search_experiment}" in
            4) relay_search_time="${RELAY_SEARCH_MATRIX_TIME_E4:-00:45:00}" ;;
            5) relay_search_time="${RELAY_SEARCH_MATRIX_TIME_E5:-00:45:00}" ;;
            6) relay_search_time="${RELAY_SEARCH_MATRIX_TIME_E6:-01:00:00}" ;;
        esac
        relay_search_name="relay-e${relay_search_experiment}-${relay_search_case}-h100"
        sbatch --nodes=1 --ntasks=1 --gpus=1 --partition="${relay_search_partition}" \
            --time="${relay_search_time}" --chdir="${PWD}" --job-name="${relay_search_name}" \
            --output="${relay_search_log_dir}/%x-%j.log" \
            triton/experiments/run-search-matrix-job.bash \
            --experiment "${relay_search_experiment}" --platform matrix \
            --operator "${relay_search_operator}" --config "${relay_search_config}" \
            --profile-launches "${relay_search_profiles}" \
            --results-root "${relay_search_results}" --plots-root "${relay_search_plots}" "$@"
    else
        echo "error: platform must be tuolumne or matrix" >&2
        exit 2
    fi
done
