#!/usr/bin/env bash

set -euo pipefail

if [[ ! -f pyproject.toml || ! -f triton/triton-lang/setup.py || ! -f triton/tritonbench/run.py ||
      ! -f triton/automatic_frontend/CMakeLists.txt ]]; then
    echo "Run this script from the RELAY repository root after initializing the submodules." >&2
    exit 1
fi

relay_laqs_plugin_source="${PWD}/triton/automatic_frontend"
relay_laqs_plugin_library="${PWD}/triton/triton-lang/python/triton/plugins/libLAQSTritonAccessManifest.so"
relay_triton_expected_commit="b1233aa326fa485b08de8593da2d08cb853c346b"
relay_triton_hook_patch="${PWD}/triton/patches/post-coalesce-hook.patch"
if [[ "${relay_laqs_plugin_source}" =~ [[:space:]] ]]; then
    echo "The LAQS plugin source path cannot contain whitespace: ${relay_laqs_plugin_source}" >&2
    exit 1
fi
if [[ "$(git -C triton/triton-lang rev-parse HEAD)" != "${relay_triton_expected_commit}" ]]; then
    echo "The Triton checkout is not at the pinned LAQS revision ${relay_triton_expected_commit}." >&2
    exit 1
fi
if git -C triton/triton-lang apply --reverse --check "${relay_triton_hook_patch}" >/dev/null 2>&1; then
    : # The exact patch is already present.
elif git -C triton/triton-lang diff --quiet -- \
        CMakeLists.txt \
        python/src/ir.cc \
        python/triton/compiler/compiler.py \
        python/triton/knobs.py \
        python/triton/runtime/jit.py \
        third_party/amd/backend/compiler.py \
        third_party/nvidia/backend/compiler.py && \
        git -C triton/triton-lang apply --check "${relay_triton_hook_patch}"; then
    git -C triton/triton-lang apply "${relay_triton_hook_patch}"
else
    echo "The pinned Triton hook files diverge from triton/patches/post-coalesce-hook.patch." >&2
    exit 1
fi
if ! cmp -s \
        <(git -C triton/triton-lang diff --binary -- \
            CMakeLists.txt \
            python/src/ir.cc \
            python/triton/compiler/compiler.py \
            python/triton/knobs.py \
            python/triton/runtime/jit.py \
            third_party/amd/backend/compiler.py \
            third_party/nvidia/backend/compiler.py) \
        "${relay_triton_hook_patch}"; then
    echo "The pinned Triton hook diff is not exactly triton/patches/post-coalesce-hook.patch." >&2
    exit 1
fi
for relay_triton_hook_marker in \
    "python/triton/knobs.py:post_coalesce_hook" \
    "python/src/ir.cc:ModuleOp &self" \
    "third_party/amd/backend/compiler.py:post_coalesce_hook" \
    "third_party/nvidia/backend/compiler.py:post_coalesce_hook" \
    "CMakeLists.txt:TRITON_PASS_PLUGIN_DIRS"; do
    IFS=: read -r relay_triton_hook_file relay_triton_hook_text <<< "${relay_triton_hook_marker}"
    if ! grep -q "${relay_triton_hook_text}" "triton/triton-lang/${relay_triton_hook_file}"; then
        echo "The pinned Triton checkout is missing the LAQS hook in ${relay_triton_hook_file}." >&2
        exit 1
    fi
done

relay_triton_rocm_module="${RELAY_TRITON_ROCM_MODULE:-rocm/7.2.1}"
relay_triton_torch_index="${RELAY_TRITON_TORCH_INDEX:-rocm7.2}"
relay_triton_max_jobs="${RELAY_TRITON_MAX_JOBS:-8}"
relay_triton_venv="${RELAY_TRITON_VENV:-${PWD}/triton/.venv}"
relay_triton_cache="${RELAY_TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/relay-triton-${USER:?}}"
relay_triton_uv="${RELAY_TRITON_UV:-uv}"
relay_python="${relay_triton_venv}/bin/python"

module load "${relay_triton_rocm_module}"

if ! command -v "${relay_triton_uv}" >/dev/null 2>&1; then
    echo "uv is missing. Set RELAY_TRITON_UV to its executable path." >&2
    exit 1
fi
relay_triton_uv="$(command -v "${relay_triton_uv}")"

if [[ ! -x .venv/bin/python ]]; then
    echo "The RELAY .venv Python interpreter is missing." >&2
    exit 1
fi

if [[ -L "${relay_triton_venv}" && ! -e "${relay_triton_venv}" ]]; then
    echo "The Triton environment symlink is broken: ${relay_triton_venv}" >&2
    exit 1
fi

if [[ ! -x "${relay_python}" ]]; then
    "${relay_triton_uv}" venv \
        --python .venv/bin/python \
        "${relay_triton_venv}"
fi

relay_triton_venv_real="$(readlink -f "${relay_triton_venv}")"
relay_triton_storage="$(dirname "${relay_triton_venv_real}")"
relay_triton_build_dir="${RELAY_TRITON_BUILD_DIR:-${relay_triton_storage}/triton-lang-build}"
relay_triton_ccache_dir="${RELAY_TRITON_CCACHE_DIR:-${relay_triton_storage}/ccache}"
relay_triton_ccache_tmp="${RELAY_TRITON_CCACHE_TEMPDIR:-${relay_triton_storage}/ccache-tmp}"
relay_triton_clang_lld="${RELAY_TRITON_BUILD_WITH_CLANG_LLD:-ON}"
relay_triton_c_compiler="${RELAY_TRITON_C_COMPILER:-/usr/bin/clang}"
relay_triton_cxx_compiler="${RELAY_TRITON_CXX_COMPILER:-/usr/bin/clang++}"
relay_triton_linker="${RELAY_TRITON_LINKER:-/usr/bin/ld.lld}"
relay_triton_legacy_build="${PWD}/triton/triton-lang/build"

# Triton defaults to a large in-tree CMake build. Move a partial legacy build
# off the project filesystem before starting a fresh build in workspace storage.
if [[ -d "${relay_triton_legacy_build}" && "${relay_triton_build_dir}" != "${relay_triton_legacy_build}" ]]; then
    relay_triton_legacy_archive="${relay_triton_storage}/triton-lang-build-from-project-fs-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Moving the in-tree Triton build to ${relay_triton_legacy_archive}."
    mv "${relay_triton_legacy_build}" "${relay_triton_legacy_archive}"
    echo "The archived partial build may be removed after installation succeeds."
fi

mkdir -p \
    "${relay_triton_build_dir}" \
    "${relay_triton_ccache_dir}" \
    "${relay_triton_ccache_tmp}"
if [[ ! -w "${relay_triton_build_dir}" ]]; then
    echo "The Triton build directory is not writable: ${relay_triton_build_dir}" >&2
    exit 1
fi

export UV_LINK_MODE="${RELAY_TRITON_UV_LINK_MODE:-copy}"
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

"${relay_triton_uv}" pip install \
    --python "${relay_python}" \
    --prerelease allow \
    --no-cache \
    torch \
    --default-index "https://download.pytorch.org/whl/nightly/${relay_triton_torch_index}"

# PyTorch nightlies may bundle their own Triton build. Replace it with this
# repository's pinned, editable triton-lang checkout.
"${relay_triton_uv}" pip uninstall \
    --python "${relay_python}" \
    --no-cache \
    triton triton-rocm pytorch-triton-rocm
"${relay_triton_uv}" pip install \
    --python "${relay_python}" \
    -r triton/triton-lang/python/requirements.txt

TRITON_BUILD_PROTON=OFF \
TRITON_EXT_ENABLED=ON \
TRITON_BUILD_WITH_CLANG_LLD="${relay_triton_clang_lld}" \
TRITON_BUILD_DIR="${relay_triton_build_dir}" \
TRITON_HOME="${relay_triton_cache}" \
MAX_JOBS="${relay_triton_max_jobs}" \
    "${relay_triton_uv}" pip install \
        --python "${relay_python}" \
        --no-build-isolation \
        -e triton/triton-lang

if [[ ! -f "${relay_laqs_plugin_library}" ]]; then
    echo "The Triton build did not produce the LAQS plugin: ${relay_laqs_plugin_library}" >&2
    exit 1
fi

(
    cd triton/tritonbench
    "${relay_triton_uv}" pip install \
        --python "${relay_python}" \
        --group dev-numpy \
        --group dev-amd
    "${relay_triton_uv}" pip install \
        --python "${relay_python}" \
        --no-deps \
        -e .
)

"${relay_python}" - "${relay_laqs_plugin_library}" <<'PY'
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
print(f"torch.version.hip={torch.version.hip}")
print(f"triton={metadata.version('triton')}")
print(f"triton.path={triton.__file__}")
print(f"laqs.plugin={plugin}")

if torch.version.hip is None:
    raise RuntimeError("The installed PyTorch build does not have ROCm support")
PY

echo "Triton and TritonBench are installed in ${relay_triton_venv}."
echo "Triton build directory: ${relay_triton_build_dir}"
echo "Triton ccache directory: ${relay_triton_ccache_dir}"
echo "Triton clang/lld build: ${relay_triton_clang_lld}"
echo "LAQS access-manifest plugin: ${relay_laqs_plugin_library}"
