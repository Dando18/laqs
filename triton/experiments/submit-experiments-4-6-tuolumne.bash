#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; schedulers relocate submitted scripts.
test -f triton/experiments/run-search-suite.py
triton/experiments/build-layout-plugin-tuolumne.bash
exec .venv/bin/python triton/experiments/submit-search-suite.py --platform tuolumne "$@"
