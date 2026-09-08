#!/usr/bin/env bash
set -euo pipefail
# Scheduler submissions start in the repository root.
if [[ ! -f triton/experiments/run-search-suite.py || $# -lt 2 ]]; then
    echo "usage (from repository root): submit-search.bash {4|5|6} {tuolumne|matrix} [suite arguments]" >&2
    exit 2
fi
relay_search_experiment="$1"
relay_search_platform="$2"
shift 2
case "${relay_search_experiment}/${relay_search_platform}" in
    [456]/tuolumne|[456]/matrix) ;;
    *) echo "invalid experiment/platform" >&2; exit 2 ;;
esac
exec "triton/experiments/submit-experiments-4-6-${relay_search_platform}.bash" \
    --experiments "${relay_search_experiment}" "$@"
