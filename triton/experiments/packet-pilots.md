# Packet Experiment 1 and legacy Experiment 3

Run from the repository root. Experiment 1 now compiles ordinary-address pilot
kernels and applies the packet plugin after coalescing. Its grammar is the
whole-tensor canonical grammar restricted to layouts preserving the ordinary
low element-address bits required by every target load's native packet. It is
not limited to the packet search workflow's split/chunk templates.

The native packet depth is measured, not assumed from dtype or warp width.
The ordinary anchor is retained; the column-major anchor is retained only when
packet-compatible. The panel is stratified before compiler validation and
counter collection. Every proposed mapping undergoes numerical, packet-site,
execution-layout, final memory primitive/count, spill, and shared-memory checks.
Rejected mappings are recorded in `packet-preflight.json` and the report's
`panel.rejected_candidates`; they are not replaced using counter observations.
The original proposal count and stratification metadata remain recorded.

Old Experiments 1–3 are not overwritten. New jobs use
`triton/experiments/results/pilot-v2`, a directory beneath the shared scratch
results root. A packet source/compiler/plugin/device identity prevents stale
resumes. Use a fresh result root after changing implementation sources or the
compiler. Repeating a command with unchanged sources resumes profiler launches.

## Minimum new collection: six jobs per device

On Tuolumne:

```bash
triton/packet_layout/build-tuolumne.bash
.venv/bin/python triton/experiments/submit-pilots.py --platform tuolumne
```

On Matrix:

```bash
triton/packet_layout/build-matrix.bash
.venv/bin/python triton/experiments/submit-pilots.py --platform matrix
```

Each command submits only Experiment 1, with GEMV, GESUMMV, MVT, embedding bag,
softmax+bias, and stencil5. Bias+ReLU is excluded. Defaults are 100 mappings,
seed 0, all-scope stratification, a 20x candidate pool, and three profiler
launches of 20 steady-state dispatches. Jobs request one task and one GPU, with
an eight-hour pbatch allocation for the user-submitted full collection. The
usual five-minute maximum still applies to agent-submitted smoke tests.

Inspect without submitting with `--dry-run`. Select individual cases with
`--cases`, or override `--queue`, `--wall-time`, `--results-root`, and
`--plots-root`. Short debug tests should try pdebug first. The script neither
builds the plugin nor submits extra jobs per tau profile.

After collection, rescore all tau profiles on the CPU, replacing PLATFORM with
`tuolumne` or `matrix`:

```bash
.venv/bin/python triton/experiments/rescore-tau-profiles.py \
  --platform PLATFORM --experiments 1 --stratifications all \
  --cases gemv gesummv mvt embedding_bag softmax_bias stencil5 \
  --counter-source triton/experiments/results/pilot-v2 \
  --results-root triton/experiments/results/pilot-v2 \
  --plots-root triton/experiments/plots/pilot-v2
```

These commands apply the existing frozen tau profiles; they do not refit them.
Counter-only collection is not a replacement for unprofiled timing data needed
to recalibrate the speedup profile.

## Experiment 3

Experiment 3 continues to use the original element-level GL(p,2) inner-tile
maps, ordinary outer tiles, and the original Stage-1 kernels. No packet plugin
or packet restriction is enabled for this experiment.

The historical kernel source is unchanged, so adding the packet plugin alone
does not invalidate the old Experiment 3 counter measurements. However, the
saved all-scope reports predate corrections to dynamic-operation numbering and
workgroup versus wave temporal scopes. Their J_area correlations must not be
presented as results of the current graph without rescoring. The old reports
also lack source/compiler fingerprints, so they cannot certify equivalence to
the current compiler build. Retain them as historical results.

To record a fresh Experiment 3 panel under the current build, append
`--experiments 3` to the submit command (six additional jobs per device). To
collect both experiments, use `--experiments 1 3` (12 jobs per device).
Use the new root; do not resume old profiler checkpoints. The tau-rescoring
command above then accepts `--experiments 1 3`.
