#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/layout_rewrite/LayoutRewritePlugin.cpp || ! -x triton/.venv/bin/python ]]; then
    echo "error: run from the RELAY repository root after installing Triton" >&2
    exit 1
fi

relay_search_venv_real="$(readlink -f triton/.venv)"
relay_search_build="${RELAY_TRITON_BUILD_DIR:-$(dirname "${relay_search_venv_real}")/triton-lang-build}"
cmake --build "${relay_search_build}" --target LAQSTritonLayoutRewrite -j "${RELAY_TRITON_MAX_JOBS:-8}"

relay_search_library="${PWD}/triton/triton-lang/python/triton/plugins/libLAQSTritonLayoutRewrite.so"
test -f "${relay_search_library}"
echo "Layout rewrite plugin: ${relay_search_library}"
