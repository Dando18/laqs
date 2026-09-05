#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit-search.bash 4 matrix "$@"
