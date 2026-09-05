#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f triton/layout_rewrite/LayoutRewritePlugin.cpp || ! -x triton/.venv-matrix/bin/python ]]; then
    echo "error: run from the RELAY repository root on Matrix after installing Triton" >&2
    exit 1
fi

relay_search_commit="$(git -C triton/triton-lang rev-parse HEAD)"
relay_search_short="${relay_search_commit:0:12}"
relay_search_storage="${RELAY_TRITON_MATRIX_STORAGE:-/usr/WS1/${USER:?}/record-replay/relay/triton}"
relay_search_build="${RELAY_TRITON_MATRIX_BUILD_DIR:-${relay_search_storage}/triton-lang-build-matrix-${relay_search_short}}"
cmake --build "${relay_search_build}" --target LAQSTritonLayoutRewrite -j "${RELAY_TRITON_MATRIX_MAX_JOBS:-8}"

triton/.venv-matrix/bin/python - <<'PY'
from importlib import metadata
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

direct = metadata.distribution("triton").read_text("direct_url.json")
source = Path(unquote(urlparse(json.loads(direct)["url"]).path))
library = source / "python/triton/plugins/libLAQSTritonLayoutRewrite.so"
if not library.is_file():
    raise SystemExit(f"layout rewrite plugin was not produced: {library}")
print(f"Layout rewrite plugin: {library}")
PY
