#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit-appendix.bash 12 matrix "$@"
