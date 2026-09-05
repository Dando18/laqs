#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit-search.bash 6 matrix "$@"
