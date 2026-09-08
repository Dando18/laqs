#!/usr/bin/env bash
set -euo pipefail
# Scheduler jobs must start in the repository root.
test -f triton/experiments/run-search-suite.py
export RELAY_SEARCH_DRIVER=triton/experiments/run-search-suite.py
exec triton/experiments/run-search-matrix-job.bash "$@"
