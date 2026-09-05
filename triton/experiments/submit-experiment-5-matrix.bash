#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit-search.bash 5 matrix "$@"
