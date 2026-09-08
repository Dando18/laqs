#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; schedulers relocate submitted scripts.
test -f triton/experiments/run-search-suite.py
# Use the GPU allocation accepted by Matrix for CPU stages as well. Keep this
# scheduler setting outside Python sources to preserve active suite fingerprints.
export SBATCH_GPUS=1
triton/experiments/build-layout-plugin-matrix.bash
exec .venv/bin/python triton/experiments/submit-search-suite.py --platform matrix "$@"
