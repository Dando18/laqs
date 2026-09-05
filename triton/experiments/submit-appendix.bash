#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/experiments/appendix_common.py ]]; then
    echo "error: run this script from the RELAY repository root" >&2
    exit 1
fi
if [[ "$#" -lt 2 ]]; then
    echo "usage: submit-appendix.bash {10|12} {tuolumne|matrix} [runner arguments ...]" >&2
    exit 2
fi
relay_appendix_experiment="$1"
relay_appendix_platform="$2"
shift 2
if [[ "${relay_appendix_experiment}" != "10" && "${relay_appendix_experiment}" != "12" ]]; then
    echo "error: experiment must be 10 or 12" >&2
    exit 2
fi
if [[ "${relay_appendix_platform}" != "tuolumne" && "${relay_appendix_platform}" != "matrix" ]]; then
    echo "error: platform must be tuolumne or matrix" >&2
    exit 2
fi

relay_appendix_cases=(
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
relay_appendix_case_override="${RELAY_APPENDIX_CASES:-}"
if [[ "${relay_appendix_experiment}" == "10" ]]; then
    relay_appendix_program="triton/experiments/run-sensitivity.py"
    relay_appendix_grammars="${RELAY_SENSITIVITY_SEARCH_EXPERIMENTS:-5}"
    relay_appendix_case_override="${RELAY_SENSITIVITY_CASES:-${relay_appendix_case_override}}"
else
    relay_appendix_program="triton/experiments/run-solve-times.py"
    relay_appendix_grammars="${RELAY_SOLVE_TIMES_SEARCH_EXPERIMENTS:-4}"
    relay_appendix_case_override="${RELAY_SOLVE_TIMES_CASES:-${relay_appendix_case_override}}"
fi
if [[ -n "${relay_appendix_case_override}" ]]; then
    read -r -a relay_appendix_cases <<< "${relay_appendix_case_override}"
fi
read -r -a relay_appendix_search_experiments <<< "${relay_appendix_grammars}"

relay_appendix_results="${RELAY_FINAL_RESULTS_ROOT:-${PWD}/triton/experiments/results}"
for relay_appendix_search_experiment in "${relay_appendix_search_experiments[@]}"; do
    if [[ ! "${relay_appendix_search_experiment}" =~ ^[456]$ ]]; then
        echo "error: search experiments must be selected from 4, 5, and 6" >&2
        exit 2
    fi
    for relay_appendix_case in "${relay_appendix_cases[@]}"; do
        relay_appendix_operator="${relay_appendix_case%%--*}"
        relay_appendix_config="${relay_appendix_case#*--}"
        relay_appendix_log_dir="${relay_appendix_results}/experiment-${relay_appendix_experiment}/${relay_appendix_platform}/grammar-e${relay_appendix_search_experiment}/logs"
        mkdir -p "${relay_appendix_log_dir}"
        if [[ "${relay_appendix_platform}" == "tuolumne" ]]; then
            relay_appendix_queue="${RELAY_APPENDIX_TUOLUMNE_QUEUE:-pbatch}"
            if [[ "${relay_appendix_experiment}" == "10" ]]; then
                relay_appendix_time="${RELAY_SENSITIVITY_TUOLUMNE_TIME:-45m}"
            else
                relay_appendix_time="${RELAY_SOLVE_TIMES_TUOLUMNE_TIME:-20m}"
            fi
            relay_appendix_name="relay-e${relay_appendix_experiment}-g${relay_appendix_search_experiment}-${relay_appendix_case}-mi300a"
            flux submit -N 1 -n 1 -g 1 -q "${relay_appendix_queue}" \
                -t "${relay_appendix_time}" --cwd="${PWD}" \
                --job-name="${relay_appendix_name}" \
                --output="${relay_appendix_log_dir}/${relay_appendix_name}.out" \
                --error="${relay_appendix_log_dir}/${relay_appendix_name}.err" \
                triton/experiments/run-appendix-tuolumne-job.bash \
                "${relay_appendix_program}" --platform tuolumne \
                --operator "${relay_appendix_operator}" \
                --config "${relay_appendix_config}" \
                --search-experiment "${relay_appendix_search_experiment}" \
                --results-root "${relay_appendix_results}" "$@"
        else
            relay_appendix_partition="${RELAY_APPENDIX_MATRIX_PARTITION:-pdebug}"
            if [[ "${relay_appendix_experiment}" == "10" ]]; then
                relay_appendix_time="${RELAY_SENSITIVITY_MATRIX_TIME:-00:20:00}"
            else
                relay_appendix_time="${RELAY_SOLVE_TIMES_MATRIX_TIME:-00:15:00}"
            fi
            relay_appendix_name="relay-e${relay_appendix_experiment}-g${relay_appendix_search_experiment}-${relay_appendix_case}-h100"
            sbatch --nodes=1 --ntasks=1 --gpus=1 \
                --partition="${relay_appendix_partition}" \
                --time="${relay_appendix_time}" --chdir="${PWD}" \
                --job-name="${relay_appendix_name}" \
                --output="${relay_appendix_log_dir}/%x-%j.log" \
                triton/experiments/run-appendix-matrix-job.bash \
                "${relay_appendix_program}" --platform matrix \
                --operator "${relay_appendix_operator}" \
                --config "${relay_appendix_config}" \
                --search-experiment "${relay_appendix_search_experiment}" \
                --results-root "${relay_appendix_results}" "$@"
        fi
    done
done
