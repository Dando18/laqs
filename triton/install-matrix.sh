#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f pyproject.toml || ! -f triton/triton-lang/setup.py || ! -f triton/tritonbench/run.py ||
      ! -f triton/automatic_frontend/CMakeLists.txt ]]; then
    echo "Run this script from the RELAY repository root after initializing the submodules." >&2
    exit 1
fi

relay_laqs_plugin_source="${PWD}/triton/automatic_frontend"
relay_triton_expected_commit="b1233aa326fa485b08de8593da2d08cb853c346b"
relay_triton_hook_patch="${PWD}/triton/patches/post-coalesce-hook.patch"
if [[ "${relay_laqs_plugin_source}" =~ [[:space:]] ]]; then
    echo "The LAQS plugin source path cannot contain whitespace: ${relay_laqs_plugin_source}" >&2
    exit 1
fi

relay_triton_cuda_module="${RELAY_TRITON_MATRIX_CUDA_MODULE:-cuda/13.1.1}"
relay_triton_torch_index="${RELAY_TRITON_MATRIX_TORCH_INDEX:-cu130}"
relay_triton_max_jobs="${RELAY_TRITON_MATRIX_MAX_JOBS:-8}"
relay_triton_venv="${PWD}/triton/.venv-matrix"
relay_triton_storage="${RELAY_TRITON_MATRIX_STORAGE:-/usr/WS1/${USER:?}/record-replay/relay/triton}"
relay_triton_venv_target="${RELAY_TRITON_MATRIX_VENV_TARGET:-${relay_triton_storage}/.venv-matrix}"
relay_triton_cache="${RELAY_TRITON_MATRIX_CACHE_DIR:-${TMPDIR:-/tmp}/relay-triton-matrix-${USER}}"
relay_triton_uv_cache="${RELAY_TRITON_MATRIX_UV_CACHE_DIR:-${relay_triton_storage}/uv-cache-matrix}"
relay_triton_uv="${RELAY_TRITON_MATRIX_UV:-uv}"

module load "${relay_triton_cuda_module}"

if ! command -v "${relay_triton_uv}" >/dev/null 2>&1; then
    echo "uv is missing. Set RELAY_TRITON_MATRIX_UV to its executable path." >&2
    exit 1
fi
relay_triton_uv="$(command -v "${relay_triton_uv}")"

if [[ ! -x .venv/bin/python ]]; then
    echo "The RELAY .venv Python interpreter is missing." >&2
    exit 1
fi

relay_triton_venv_target="$(realpath -m "${relay_triton_venv_target}")"
relay_triton_tuolumne_venv="$(realpath -m "${PWD}/triton/.venv")"
if [[ "${relay_triton_venv_target}" == "${relay_triton_tuolumne_venv}" ]]; then
    echo "The Matrix and Tuolumne environments must not share a target." >&2
    exit 1
fi

if [[ -e "${relay_triton_venv}" && ! -L "${relay_triton_venv}" ]]; then
    echo "The Matrix environment path must be a symlink: ${relay_triton_venv}" >&2
    exit 1
fi
if [[ -L "${relay_triton_venv}" && "$(realpath -m "${relay_triton_venv}")" != "${relay_triton_venv_target}" ]]; then
    echo "The Matrix environment symlink points somewhere unexpected: ${relay_triton_venv}" >&2
    exit 1
fi
if [[ -e "${relay_triton_venv_target}" && ! -d "${relay_triton_venv_target}" ]]; then
    echo "The Matrix environment target is not a directory: ${relay_triton_venv_target}" >&2
    exit 1
fi
if [[ -d "${relay_triton_venv_target}" && ! -f "${relay_triton_venv_target}/pyvenv.cfg" ]] &&
        [[ -n "$(find "${relay_triton_venv_target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to use a non-empty directory that is not a virtual environment: ${relay_triton_venv_target}" >&2
    exit 1
fi

mkdir -p "$(dirname "${relay_triton_venv_target}")"
if [[ ! -x "${relay_triton_venv_target}/bin/python" ]]; then
    "${relay_triton_uv}" venv \
        --python .venv/bin/python \
        "${relay_triton_venv_target}"
fi
if [[ ! -L "${relay_triton_venv}" ]]; then
    ln -s "${relay_triton_venv_target}/" "${relay_triton_venv}"
fi
relay_python="${relay_triton_venv}/bin/python"

# An editable Triton build writes libtriton.so into its source tree. Use a
# Matrix-only clone so the CUDA build cannot replace Tuolumne's ROCm library.
relay_triton_commit="$(git -C triton/triton-lang rev-parse HEAD)"
if [[ "${relay_triton_commit}" != "${relay_triton_expected_commit}" ]]; then
    echo "The Triton checkout is not at the pinned LAQS revision ${relay_triton_expected_commit}." >&2
    exit 1
fi
relay_triton_commit_short="${relay_triton_commit:0:12}"
relay_triton_source="${RELAY_TRITON_MATRIX_SOURCE_DIR:-${relay_triton_storage}/triton-lang-source-matrix-${relay_triton_commit_short}}"
relay_triton_source="$(realpath -m "${relay_triton_source}")"
if [[ "${relay_triton_source}" == "$(realpath -m "${PWD}/triton/triton-lang")" ]]; then
    echo "The Matrix Triton source must not be the shared Tuolumne checkout." >&2
    exit 1
fi
if [[ ! -e "${relay_triton_source}" ]]; then
    git clone --no-local --no-checkout \
        "$(realpath triton/triton-lang)" \
        "${relay_triton_source}"
    git -C "${relay_triton_source}" checkout --detach "${relay_triton_commit}"
elif [[ ! -d "${relay_triton_source}/.git" ]]; then
    echo "The Matrix Triton source exists but is not a Git clone: ${relay_triton_source}" >&2
    exit 1
elif [[ "$(git -C "${relay_triton_source}" rev-parse HEAD)" != "${relay_triton_commit}" ]]; then
    echo "The Matrix Triton source is not at the pinned commit: ${relay_triton_source}" >&2
    exit 1
fi

# Apply the tracked hook patch to the commit-pinned Matrix clone. A reverse
# check makes this idempotent; any other edit to the hook files is refused.
if git -C "${relay_triton_source}" apply --reverse --check "${relay_triton_hook_patch}" >/dev/null 2>&1; then
    : # The exact patch is already present.
elif git -C "${relay_triton_source}" diff --quiet -- \
        CMakeLists.txt \
        python/src/ir.cc \
        python/triton/compiler/compiler.py \
        python/triton/knobs.py \
        python/triton/runtime/jit.py \
        third_party/amd/backend/compiler.py \
        third_party/nvidia/backend/compiler.py && \
        git -C "${relay_triton_source}" apply --check "${relay_triton_hook_patch}"; then
    git -C "${relay_triton_source}" apply "${relay_triton_hook_patch}"
else
    echo "The Matrix Triton hook files diverge from triton/patches/post-coalesce-hook.patch." >&2
    exit 1
fi
if ! cmp -s \
        <(git -C "${relay_triton_source}" diff --binary -- \
            CMakeLists.txt \
            python/src/ir.cc \
            python/triton/compiler/compiler.py \
            python/triton/knobs.py \
            python/triton/runtime/jit.py \
            third_party/amd/backend/compiler.py \
            third_party/nvidia/backend/compiler.py) \
        "${relay_triton_hook_patch}"; then
    echo "The Matrix Triton hook diff is not exactly triton/patches/post-coalesce-hook.patch." >&2
    exit 1
fi
for relay_triton_hook_marker in \
    "python/triton/knobs.py:post_coalesce_hook" \
    "python/src/ir.cc:ModuleOp &self" \
    "third_party/amd/backend/compiler.py:post_coalesce_hook" \
    "third_party/nvidia/backend/compiler.py:post_coalesce_hook" \
    "CMakeLists.txt:TRITON_PASS_PLUGIN_DIRS"; do
    IFS=: read -r relay_triton_hook_file relay_triton_hook_text <<< "${relay_triton_hook_marker}"
    if ! grep -q "${relay_triton_hook_text}" "${relay_triton_source}/${relay_triton_hook_file}"; then
        echo "The Matrix Triton source is missing the LAQS hook in ${relay_triton_hook_file}." >&2
        exit 1
    fi
done

relay_triton_build_dir="${RELAY_TRITON_MATRIX_BUILD_DIR:-${relay_triton_storage}/triton-lang-build-matrix-${relay_triton_commit_short}}"
relay_triton_ccache_dir="${RELAY_TRITON_MATRIX_CCACHE_DIR:-${relay_triton_storage}/ccache-matrix}"
relay_triton_ccache_tmp="${RELAY_TRITON_MATRIX_CCACHE_TEMPDIR:-${relay_triton_storage}/ccache-tmp-matrix}"
relay_triton_clang_lld="${RELAY_TRITON_MATRIX_BUILD_WITH_CLANG_LLD:-ON}"
relay_triton_c_compiler="${RELAY_TRITON_MATRIX_C_COMPILER:-/usr/bin/clang}"
relay_triton_cxx_compiler="${RELAY_TRITON_MATRIX_CXX_COMPILER:-/usr/bin/clang++}"
relay_triton_linker="${RELAY_TRITON_MATRIX_LINKER:-/usr/bin/ld.lld}"

relay_triton_tuolumne_storage="$(dirname "${relay_triton_tuolumne_venv}")"
for relay_triton_pair in \
    "build:${relay_triton_build_dir}:${relay_triton_tuolumne_storage}/triton-lang-build" \
    "ccache:${relay_triton_ccache_dir}:${relay_triton_tuolumne_storage}/ccache" \
    "ccache temporary:${relay_triton_ccache_tmp}:${relay_triton_tuolumne_storage}/ccache-tmp" \
    "runtime cache:${relay_triton_cache}:${TMPDIR:-/tmp}/relay-triton-${USER}"; do
    IFS=: read -r relay_triton_label relay_triton_matrix_path relay_triton_tuolumne_path <<< "${relay_triton_pair}"
    if [[ "$(realpath -m "${relay_triton_matrix_path}")" == "$(realpath -m "${relay_triton_tuolumne_path}")" ]]; then
        echo "The Matrix ${relay_triton_label} must not share Tuolumne's path." >&2
        exit 1
    fi
done

mkdir -p \
    "${relay_triton_build_dir}" \
    "${relay_triton_ccache_dir}" \
    "${relay_triton_ccache_tmp}" \
    "${relay_triton_uv_cache}"

export UV_LINK_MODE="${RELAY_TRITON_MATRIX_UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${relay_triton_uv_cache}"
export UV_NATIVE_TLS="${RELAY_TRITON_MATRIX_UV_NATIVE_TLS:-true}"
export CCACHE_DIR="${relay_triton_ccache_dir}"
export CCACHE_TEMPDIR="${relay_triton_ccache_tmp}"

if [[ "${relay_triton_clang_lld^^}" =~ ^(ON|1|YES|TRUE|Y)$ ]]; then
    for relay_triton_tool in \
        "${relay_triton_c_compiler}" \
        "${relay_triton_cxx_compiler}" \
        "${relay_triton_linker}"; do
        if [[ ! -x "${relay_triton_tool}" ]]; then
            echo "Required Triton build tool is missing: ${relay_triton_tool}" >&2
            exit 1
        fi
    done
    export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DCMAKE_C_COMPILER=${relay_triton_c_compiler} -DCMAKE_CXX_COMPILER=${relay_triton_cxx_compiler} -DCMAKE_LINKER=${relay_triton_linker}"
fi
export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DTRITON_PASS_PLUGIN_DIRS=${relay_laqs_plugin_source}"

# Remove any wheel bundled with PyTorch before installing the pinned checkout.
"${relay_triton_uv}" pip uninstall \
    --python "${relay_python}" \
    --no-cache \
    torch triton triton-rocm pytorch-triton pytorch-triton-rocm
"${relay_triton_uv}" pip install \
    --python "${relay_python}" \
    --prerelease allow \
    --no-cache \
    torch \
    --default-index "https://download.pytorch.org/whl/nightly/${relay_triton_torch_index}"
"${relay_triton_uv}" pip uninstall \
    --python "${relay_python}" \
    --no-cache \
    triton triton-rocm pytorch-triton pytorch-triton-rocm
"${relay_triton_uv}" pip install \
    --python "${relay_python}" \
    -r "${relay_triton_source}/python/requirements.txt"

TRITON_BUILD_PROTON=OFF \
TRITON_EXT_ENABLED=ON \
TRITON_BUILD_WITH_CLANG_LLD="${relay_triton_clang_lld}" \
TRITON_BUILD_DIR="${relay_triton_build_dir}" \
TRITON_HOME="${relay_triton_cache}" \
MAX_JOBS="${relay_triton_max_jobs}" \
    "${relay_triton_uv}" pip install \
        --python "${relay_python}" \
        --no-build-isolation \
        -e "${relay_triton_source}"

relay_laqs_plugin_library="${relay_triton_source}/python/triton/plugins/libLAQSTritonAccessManifest.so"
if [[ ! -f "${relay_laqs_plugin_library}" ]]; then
    echo "The Triton build did not produce the LAQS plugin: ${relay_laqs_plugin_library}" >&2
    exit 1
fi
relay_laqs_plugin_link_dir="${PWD}/triton/plugins/matrix"
relay_laqs_plugin_link="${relay_laqs_plugin_link_dir}/libLAQSTritonAccessManifest.so"
mkdir -p "${relay_laqs_plugin_link_dir}"
ln -sfn "${relay_laqs_plugin_library}" "${relay_laqs_plugin_link}"

(
    cd triton/tritonbench
    "${relay_triton_uv}" pip install \
        --python "${relay_python}" \
        --group dev-numpy \
        --group dev-nvidia
    "${relay_triton_uv}" pip install \
        --python "${relay_python}" \
        --no-deps \
        -e .
)

"${relay_python}" - "${relay_laqs_plugin_link}" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

import torch
import triton
from triton._C.libtriton import passes

plugin = Path(sys.argv[1]).resolve()
passes.plugin.extend_with(str(plugin))
if not hasattr(passes.plugin, "add_laqs_access_manifest"):
    raise RuntimeError(f"LAQS plugin did not register its pass: {plugin}")

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"triton={metadata.version('triton')}")
print(f"triton.path={triton.__file__}")
print(f"laqs.plugin={plugin}")

if torch.version.cuda is None:
    raise RuntimeError("The installed PyTorch build does not have CUDA support")
PY

echo "Matrix Triton and TritonBench are installed in ${relay_triton_venv}."
echo "Matrix environment target: ${relay_triton_venv_target}"
echo "Matrix Triton source: ${relay_triton_source}"
echo "Matrix Triton build directory: ${relay_triton_build_dir}"
echo "Matrix Triton ccache directory: ${relay_triton_ccache_dir}"
echo "LAQS access-manifest plugin: ${relay_laqs_plugin_link}"
