#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f pyproject.toml || ! -f triton/tritonbench/run.py ]]; then
    echo "Run this script from the RELAY repository root." >&2
    exit 1
fi

relay_triton_cuda_module="${RELAY_TRITON_MATRIX_CUDA_MODULE:-cuda/13.1.1}"
relay_triton_venv="${PWD}/triton/.venv-matrix"
relay_triton_cache="${RELAY_TRITON_MATRIX_CACHE_DIR:-${TMPDIR:-/tmp}/relay-triton-matrix-${USER:?}}"
relay_triton_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
relay_triton_results="${RELAY_TRITON_MATRIX_RESULTS_DIR:-${PWD}/triton/results/baseline-matrix-${relay_triton_timestamp}}"
relay_python="${relay_triton_venv}/bin/python"

module load "${relay_triton_cuda_module}"

if [[ ! -x "${relay_python}" ]]; then
    echo "Matrix Triton is not installed. Run triton/install-matrix.sh first." >&2
    exit 1
fi

relay_triton_scheduler_devices="${CUDA_VISIBLE_DEVICES:-}"
relay_triton_gpu="${RELAY_TRITON_MATRIX_GPU:-${relay_triton_scheduler_devices:-0}}"
if [[ "${relay_triton_gpu}" == *,* ]]; then
    echo "The Matrix baseline must have exactly one GPU in CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${relay_triton_gpu}"
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES

mkdir -p "${relay_triton_results}"
export TRITON_HOME="${relay_triton_cache}"
export PYTHONUNBUFFERED=1

{
    echo "timestamp_utc=${relay_triton_timestamp}"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-not-set}"
    echo "cuda_module=${relay_triton_cuda_module}"
    echo "cuda_home=${CUDA_HOME:-${CUDA_PATH:-not-set}}"
    echo "scheduler_visible_devices=${relay_triton_scheduler_devices:-not-set}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "triton_commit=$(git -C triton/triton-lang rev-parse HEAD)"
    echo "tritonbench_commit=$(git -C triton/tritonbench rev-parse HEAD)"
    echo
    module list 2>&1
    echo
    nvidia-smi
    echo
    "${relay_python}" - <<'PY'
import platform
from importlib import metadata

import torch
import triton

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"triton={metadata.version('triton')}")
print(f"triton.path={triton.__file__}")
print(f"gpu_available={torch.cuda.is_available()}")
if torch.version.cuda is None:
    raise RuntimeError("The installed PyTorch build does not have CUDA support")
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible; run this script through Slurm")

print(f"gpu_name={torch.cuda.get_device_name(0)}")
print(f"gpu_count={torch.cuda.device_count()}")
if torch.cuda.device_count() != 1:
    raise RuntimeError("The baseline must run with exactly one visible GPU")

print("\npackages:")
packages = sorted(
    (dist.metadata["Name"], dist.version)
    for dist in metadata.distributions()
    if dist.metadata["Name"]
)
for name, version in packages:
    print(f"{name}=={version}")
PY
} > "${relay_triton_results}/environment.txt"

cd triton/tritonbench

"${relay_python}" run.py \
    --op vector_add \
    --only torch_add,triton_add \
    --baseline torch_add \
    --metrics latency,speedup,gbps,accuracy \
    --input-id 8 \
    --num-inputs 3 \
    --output "${relay_triton_results}/vector-add.csv" \
    --output-json "${relay_triton_results}/vector-add.json" \
    2>&1 | tee "${relay_triton_results}/vector-add.log"

"${relay_python}" run.py \
    --op softmax \
    --only naive_softmax,triton_softmax \
    --baseline naive_softmax \
    --metrics latency,speedup,gbps,accuracy \
    --precision fp16 \
    --output "${relay_triton_results}/softmax.csv" \
    --output-json "${relay_triton_results}/softmax.json" \
    --M 4096 \
    --N 1024 \
    2>&1 | tee "${relay_triton_results}/softmax.log"

"${relay_python}" run.py \
    --op gemm \
    --only aten_matmul,triton_tutorial_matmul \
    --baseline aten_matmul \
    --metrics latency,speedup,tflops,accuracy \
    --precision fp16 \
    --output "${relay_triton_results}/gemm.csv" \
    --output-json "${relay_triton_results}/gemm.json" \
    --shapes 1024x1024x1024,2048x2048x2048,4096x4096x4096 \
    2>&1 | tee "${relay_triton_results}/gemm.log"

echo "Matrix baseline results: ${relay_triton_results}"
