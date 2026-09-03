# Final Triton experiments

This directory implements memory-counter Experiments 1--3 from
[`notes/final-experiments.md`](../../notes/final-experiments.md). Each run uses
one of the seven pilot kernels and one GPU. The six `submit-experiment-*`
scripts submit all seven kernels as independent batch jobs, so a failed or
preempted kernel can resume without repeating completed profiles.

## What the experiments do

All three experiments use the same kernels, score construction, profiler
protocol, and analysis. They differ only in the sampled layout grammar:

1. `submit-experiment-1-*`: sample canonical (`G_C`) words over every bit of
   the target tensor.
2. `submit-experiment-2-*`: choose uniformly from the kernel's declared inner
   tile shapes, sample a canonical word within the tile, and keep the outer
   tiles row-major.
3. `submit-experiment-3-*`: choose a declared inner tile shape, draw the inner
   matrix uniformly from `GL(p,2)`, and keep the outer tiles row-major
   (`G_OC`). This is random sampling, not the bounded exact `G_OC` search used
   by later search experiments.

Sampling is deterministic for a given seed and is without replacement after
full physical mappings are materialized. A grammar can contain fewer distinct
mappings than the requested sample count. In particular, the rank-one bias
operand has only one `G_C` mapping; the report records this as a sample
shortfall rather than duplicating the same observation.

The panel is scored before profiling. The ordinary kernel is launched once
under the post-coalescing access-manifest plugin. The automatic frontend
reconstructs the exact active memory events and ordered trace classes, then
builds every nonempty universal-v1 issue, lane-window, SIMD-window,
workgroup-step/window, and phase family. The scorer materializes the complete
family vector at 32, 64, 128, and 256 bytes for the operand varied by the
experiment. Other operations remain in the ordered trace, but their layouts
are fixed and are not part of this one-operand panel. Every component contains
its raw `Q`, packing lower bound, normalized excess, whole-kernel useful-byte
exposure, and weighted contribution.
The aggregate is exactly

```text
J_area = sum_component tau * bytes * (Q - packing_bound) / useful_byte_exposure
```

The MI300A and H100 `tau` profiles live in `tau-profiles.json`. They are fit
only on the seven pilot kernels, using the mean within-kernel rank of relevant
L1 and L2 work counters. A nonnegative ridge fit over the automatic component
features initializes a deterministic pair-grid and greedy refinement that
selects tau by training-set macro Spearman. TritonBench and real kernels are
not part of this tuning set. Derived aliases of the same native counter
contribute only once to the tuning target; on H100, `l2_read_work` remains in
analysis output but aliases
the same native counter as `l1_to_l2_read_traffic` and is excluded from the
fit. Reports record the selected components, weights, fitting method, tuning
counters, excluded aliases, and training correlation. The native fine
components are `issue.g64.stream.load.64B` on MI300A and
`issue.g32.stream.load.32B` on H100.

This is the automatic construction introduced by commit
`8f40ee8e0a5528740ade0ba0f0ab229a27129215`, not the earlier manual Stage-1
issue/instruction/lane/temporal graph. The independent oracle comparison in
`tests/test_triton_stage1_automatic.py` remains a frontend validation test.

Each mapping is packed into a fresh target operand, checked for numerical
correctness, compiled, and structurally compared with the row-major kernel.
Mappings that change the execution layout or load/store instruction structure,
or introduce spills, are rejected. Counter processes use a warm-cache protocol
and rotate mapping order between independent profiler launches.

## Submit the full experiments

Run submission commands from the repository root.

Tuolumne (MI300A, Flux `pbatch`, 8 hours per kernel by default):

```bash
triton/experiments/submit-experiment-1-tuolumne.bash
triton/experiments/submit-experiment-2-tuolumne.bash
triton/experiments/submit-experiment-3-tuolumne.bash
```

Matrix (H100, Slurm `pbatch`, 6 hours per kernel by default):

```bash
triton/experiments/submit-experiment-1-matrix.bash
triton/experiments/submit-experiment-2-matrix.bash
triton/experiments/submit-experiment-3-matrix.bash
```

The defaults are 100 mappings, seed 0, three independent profiler launches,
five warm-up dispatches, and 20 measured dispatches per profiler launch. The
following environment variables change submission-wide settings:

```bash
RELAY_FINAL_LAYOUTS=200
RELAY_FINAL_SEED=17
RELAY_FINAL_PROFILE_LAUNCHES=5
RELAY_FINAL_TUOLUMNE_QUEUE=pbatch
RELAY_FINAL_TUOLUMNE_WALL_TIME=12h
RELAY_FINAL_MATRIX_PARTITION=pbatch
RELAY_FINAL_MATRIX_WALL_TIME=08:00:00
```

Extra command-line arguments to a submit script are forwarded to every
`run.py` invocation. For example, this submits a short, resumable trial:

```bash
RELAY_FINAL_LAYOUTS=10 RELAY_FINAL_PROFILE_LAUNCHES=1 \
  triton/experiments/submit-experiment-2-tuolumne.bash \
  --profile-iterations 5
```

For a single case inside an existing one-GPU allocation:

```bash
triton/.venv/bin/python triton/experiments/run.py \
  --experiment 1 --platform tuolumne --case gemv
```

On Matrix use `triton/.venv-matrix/bin/python`, load CUDA and Nsight Compute,
and pass `--platform matrix`. The Matrix implementation is present but has not
been exercised on Tuolumne.

## Outputs and resuming

Per-case results have this layout:

```text
results/experiment-<1|2|3>/<platform>/<kernel>/
  report.json          complete panel, score vectors, and aggregate counters
  counter-data.csv     profiler-oriented aggregate table
  raw-data.csv         flat analysis-ready score and counter table
  spearman.csv         every score/counter Spearman correlation
  analysis.json        primary correlations and artifact manifest
  profiles/<mapping>/launch-<n>/
    counters.csv       raw rocprof or Nsight Compute output
    worker.json        numerical and compiled-structure validation
    profile.json       parsed checkpoint for this independent launch
```

Plots are written as multi-page PDFs to
`plots/experiment-<n>/<platform>/<kernel>.pdf`. Every counter gets one page:
the left panel uses its predeclared quotient component and the right panel uses
`J_area`; both show tie-aware Spearman rho.

Existing valid profile checkpoints are reused automatically. `--rerun`
explicitly replaces them. `--max-profiles N` is useful for smoke tests, and
`--analyze-only` rebuilds CSV/PDF analysis without touching a GPU:

```bash
triton/.venv/bin/python triton/experiments/run.py \
  --experiment 3 --platform tuolumne --case mvt --analyze-only
```

After all seven jobs finish, collect their per-kernel Spearman tables into one
suite table (and regenerate any per-kernel analysis) with:

```bash
triton/.venv/bin/python triton/experiments/analyze.py \
  --experiment 1 --platform tuolumne
```

The corresponding Matrix command changes only `--platform matrix` and the
Python environment.

## Rescoring archived counters and tuning tau

`results-old` is the immutable profiler corpus. The rescore workers launch
each default kernel once to reconstruct its automatic graph, generate all
three layout panels in that process, and join the saved counter aggregates by
`mapping_id`. They never invoke rocprof or NCU. Run one command on each
cluster from the repository root:

```bash
triton/experiments/rescore-all-tuolumne.bash
triton/experiments/rescore-all-matrix.bash
```

The worker checks that the automatic manifest plugin belongs to the active
platform-specific Triton checkout before requesting work. If the Matrix
environment predates the automatic frontend, update it once on Matrix with
`triton/setup/install-matrix.sh`; the setup keeps CUDA and ROCm builds in
separate source/build trees.

Each case uses one GPU and a 30-minute `pbatch` allocation by default, except
`embedding_bag`, whose data-dependent full trace gets 60 minutes. Override the
queue/partition or common wall time with `RELAY_FINAL_RESCORE_QUEUE`,
`RELAY_FINAL_RESCORE_PARTITION`, and `RELAY_FINAL_RESCORE_TIME`; override the
embedding exception with `RELAY_FINAL_EMBEDDING_RESCORE_TIME`. A single case
can be regenerated with `rescore-tuolumne-job.bash` or
`rescore-matrix-job.bash`, passing `--case` and `--counter-source`.

After both platforms are present, fit both device profiles and rewrite all
reports, CSV analysis, and PDFs with the fitted `tau` values:

```bash
.venv/bin/python triton/experiments/tune_tau.py
```

Running the rescore commands again later reuses the same archived counters and
automatically applies the checked-in `tau-profiles.json`; rerunning
`tune_tau.py` does not launch a GPU.
