#!/usr/bin/env bash
set -euo pipefail
exec triton/experiments/submit.bash 3 tuolumne "$@"
