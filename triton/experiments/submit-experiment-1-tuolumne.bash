#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit.bash 1 tuolumne "$@"
