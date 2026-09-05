#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit-gemv-sweep.bash matrix "$@"
