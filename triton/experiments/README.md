# Final Triton experiments

This directory implements Experiments 1--6, 10, and 12 from
[`notes/final-experiments.md`](../../notes/final-experiments.md). Each run uses
one GPU. Experiments 1--3 use the seven pilot kernels; Experiments 4--6 use the
broad portable TritonBench panel. Jobs are independently resumable, including
their timing processes and profiler launches.

## What the experiments do

All three experiments use the same kernels, score construction, profiler
protocol, stratification, and analysis. They differ only in the layout grammar
used to build the candidate pool:

1. `submit-experiment-1-*`: sample canonical (`G_C`) words over every bit of
   the target tensor.
2. `submit-experiment-2-*`: choose uniformly from the kernel's declared inner
   tile shapes, sample a canonical word within the tile, and keep the outer
   tiles row-major.
3. `submit-experiment-3-*`: choose a declared inner tile shape, draw the inner
   matrix uniformly from `GL(p,2)`, and keep the outer tiles row-major
   (`G_OC`). This is a random pool, not the bounded exact `G_OC` search used by
   later search experiments.

The random pool is deterministic for a given seed and is deduplicated after
full physical mappings are materialized. By default it is 20 times the final
panel size. Panel selection happens before any counter is observed. It balances
the marginal `Q / packing_bound` bands `[1,1.5)`, `[1.5,2)`, `[2,4)`, `[4,8)`,
and `[8,infinity)` in one of three component sets: `all`, `issue`, or
`temporal` (all non-issue scopes). Distinct row-major and column-major mappings
for every tile hypothesis are anchors. The three variants use the same seeded
pool, which isolates the effect of the stratification rule. A grammar can
contain fewer distinct mappings than requested; the report records a shortfall
rather than duplicating an observation.

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

MI300A continues to collect the full existing counter set; the primary
downstream field is `l1_miss_demand_to_l2`, backed by `TCP_TCC_READ_REQ`. H100
retains the old counters and additionally collects TEX-source L2 requests with
`lts__t_requests_srcunit_tex_op_read.sum`. The corresponding TEX-source L2
sector counter remains `lts__t_sectors_srcunit_tex_op_read.sum`. On the Matrix
GH100 and Nsight Compute 2025.3 combination,
`lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum` is not emitted
in the raw results and therefore is not requested. H100's primary
`l1_miss_demand_to_l2` alias is backed by the TEX-source request count; request
and sector counts are both focus counters. The remaining L1, L2, HBM, request,
and duration counters stay available as diagnostics.

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

Each command above submits all three stratifications (`all issue temporal`), or
21 jobs: seven kernels times three panels. The defaults are 100 selected
mappings from a 2,000-mapping pool, seed 0, three independent profiler
launches, five warm-up dispatches, and 20 measured dispatches per profiler
launch. The following environment variables change submission-wide settings:

```bash
RELAY_FINAL_LAYOUTS=200
RELAY_FINAL_SEED=17
RELAY_FINAL_PROFILE_LAUNCHES=5
RELAY_FINAL_POOL_MULTIPLIER=50
RELAY_FINAL_STRATIFICATIONS="all issue temporal"
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
results/experiment-<1|2|3>/<platform>/stratified-<all|issue|temporal>/<kernel>/
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

Plots are written as multi-page PDFs under the corresponding
`plots/experiment-<n>/<platform>/stratified-<mode>/` directory. Every counter
gets one page showing `Q_b` at 32, 64, 128, and 256 bytes for a fixed,
counter-matched automatic scope, plus `J_area`; every panel shows tie-aware
Spearman rho. `spearman.csv` still contains every individual automatic
component and excess predictor, not only the plotted scope.

Existing valid profile checkpoints are reused automatically. `--rerun`
explicitly replaces them. `--max-profiles N` is useful for smoke tests, and
`--analyze-only` rebuilds CSV/PDF analysis without touching a GPU:

```bash
triton/.venv/bin/python triton/experiments/run.py \
  --experiment 3 --platform tuolumne --stratification all \
  --case mvt --analyze-only
```

After all seven jobs finish, collect their per-kernel Spearman tables into one
suite table (and regenerate any per-kernel analysis) with:

```bash
triton/.venv/bin/python triton/experiments/analyze.py \
  --experiment 1 --platform tuolumne --stratification all
```

The corresponding Matrix command changes only `--platform matrix` and the
Python environment.

## TritonBench searches (Experiments 4--6)

These experiments use the 15-operator, 29-configuration broad portable panel
specified in `notes/final-experiments.md`. Each configuration is an independent
job. An ordinary Triton launch first selects and freezes its native autotune
configuration and produces the exact automatic access trace. Experiment 4
runs the exact count-grid DP over whole-tensor `G_C`; Experiment 5 runs the
same DP over the largest natural single-operation access tile; Experiment 6
runs the exact `G_OC` search with at most four inner bits and its canonical
outer DP. All eligible read-only ordinary-dense inputs are selected jointly
under the platform's frozen `J_area` profile. Exact score ties keep the
ordinary row-major layout.

The selected persistent layouts are realized by
`libLAQSTritonLayoutRewrite.so`. This is a separate post-coalescing compiler
pass: it neither replaces nor changes `libLAQSTritonAccessManifest.so`, so
building it does not change the frontend or commands used by queued
Experiments 1--3. Build just this new target once on each platform:

```bash
# On Tuolumne
triton/experiments/build-layout-plugin-tuolumne.bash

# On Matrix
triton/experiments/build-layout-plugin-matrix.bash
```

Then submit the three stages from the repository root:

```bash
# Tuolumne / MI300A
triton/experiments/submit-experiment-4-tuolumne.bash
triton/experiments/submit-experiment-5-tuolumne.bash
triton/experiments/submit-experiment-6-tuolumne.bash

# Matrix / H100
triton/experiments/submit-experiment-4-matrix.bash
triton/experiments/submit-experiment-5-matrix.bash
triton/experiments/submit-experiment-6-matrix.bash
```

Each submission launches 29 jobs, allowing all configurations and grammars to
run concurrently. Defaults are intentionally queue-friendly: Experiments 4
and 5 request 1 hour on Tuolumne and 45 minutes on Matrix; Experiment 6
requests 90 minutes and 1 hour, respectively. Override them only if observed
runtimes warrant it:

```bash
RELAY_SEARCH_TUOLUMNE_TIME_E6=2h
RELAY_SEARCH_MATRIX_TIME_E4=01:00:00
RELAY_SEARCH_MATRIX_TIME_E6=01:30:00
RELAY_SEARCH_TUOLUMNE_QUEUE=pbatch
RELAY_SEARCH_MATRIX_PARTITION=pbatch
```

Use `RELAY_SEARCH_CASES` for a smoke test or targeted retry. Extra arguments
are forwarded to every job:

```bash
RELAY_SEARCH_CASES="vector_add--small softmax--small" \
  triton/experiments/submit-experiment-4-tuolumne.bash \
  --timing-processes 1 --timing-samples 3 --timing-iterations 10 \
  --profile-launches 1 --profile-iterations 5
```

The full protocol uses three fresh timing processes, 10 warm-ups, and 21
samples of 50 launches. Counter collection is separate for ordinary and
selected layouts, with three profiler processes per layout, five warm-ups,
and 20 measured dispatches. Packing, analysis, autotuning, compilation, and
search are outside timed regions; their elapsed times remain in `report.json`.
Unsupported traces and failed portability gates are recorded as exclusions
rather than silently replaced.

Per-configuration outputs are:

```text
results/experiment-<4|5|6>/<platform>/<operator>--<config>/
  selection.json
  report.json
  raw-data.csv
  timings/process-<n>.json
  profiles/<baseline|selected>/launch-<n>/
    counters.csv
    profile.json
```

Every completed job also writes a PDF under
`plots/experiment-<n>/<platform>/`. After the jobs finish, aggregate the
per-case raw rows and build the suite PDF with:

```bash
triton/.venv/bin/python triton/experiments/analyze-search.py \
  --experiment 4 --platform tuolumne

triton/.venv-matrix/bin/python triton/experiments/analyze-search.py \
  --experiment 4 --platform matrix
```

Repeat with `--experiment 5` and `6`. The suite CSV is written to the
experiment result directory and `summary.pdf` to its plot directory. Run this
after both platforms finish: the summary includes only configurations marked
complete on both machines and records the counterpart status for every row.
Runtime speedup and the primary L1-to-L2 demand reduction are the principal
figures; the report retains every parsed native counter and derived reduction.

## Appendix hardware-profile sensitivity (Experiment 10)

Experiment 10 reruns exact layout selection after independently multiplying
every positive tau by a deterministic random factor. The default bounds are
plus or minus 10%, 25%, and 50%, with three trials per bound and seed 0. The
same trial profile is used for every case on one platform. The nominal profile
is also included. Physically identical selected mappings are deduplicated and
timed once, then shared by all trials that selected them. Each distinct mapping
is correctness-checked and timed against an ordinary frozen-config Triton
launch with 10 warm-ups and 11 alternating samples of 50 launches.

The default search space is Experiment 5's natural-tile canonical DP. This
keeps the appendix focused while covering all 29 TritonBench configurations.
Set `RELAY_SENSITIVITY_SEARCH_EXPERIMENTS="4 5 6"` to study all three search
spaces. Matrix's fitted profile currently has one positive tau entry, so
multiplying it cannot change the analytical ranking; the recorded trials make
that invariance explicit rather than manufacturing inactive weights.

Experiment 10 needs the layout-rewrite plugin described above. Submit from the
repository root:

```bash
# Tuolumne: 29 jobs, 45 minutes each, pbatch
triton/experiments/submit-experiment-10-tuolumne.bash

# Matrix: 29 jobs, 20 minutes each, pdebug
triton/experiments/submit-experiment-10-matrix.bash
```

The Matrix jobs deliberately try `pdebug` first with a short wall time. If its
two debug nodes remain occupied, submit to the regular partition without
inflating the request:

```bash
RELAY_APPENDIX_MATRIX_PARTITION=pbatch \
  triton/experiments/submit-experiment-10-matrix.bash
```

Useful overrides are:

```bash
RELAY_SENSITIVITY_CASES="vector_add--small softmax--small"
RELAY_SENSITIVITY_SEARCH_EXPERIMENTS="4 5"
RELAY_SENSITIVITY_TUOLUMNE_TIME=30m
RELAY_SENSITIVITY_MATRIX_TIME=00:15:00
```

Runner arguments are forwarded to every case, for example
`--trials-per-magnitude 5 --perturbation-magnitudes 0.1 0.25 0.5`.
Results are independently resumable under
`results/experiment-10/<platform>/grammar-e5/<operator>--<config>/`; timing
checkpoints are keyed by the physical mapping, not the perturbation trial.
Aggregate the reports and create a multi-page PDF with:

```bash
triton/.venv/bin/python triton/experiments/analyze-sensitivity.py \
  --platform tuolumne --search-experiment 5
triton/.venv-matrix/bin/python triton/experiments/analyze-sensitivity.py \
  --platform matrix --search-experiment 5
```

## Appendix construction, scoring, and solve times (Experiment 12)

Experiment 12 captures one exact automatic trace, then separately replays and
times these CPU phases:

1. build universal-v1 scale-free edge families and materialize all byte-scale
   components;
2. compute the full quotient-component vector and `J_area` for one ordinary
   row-major layout assignment; and
3. run exact layout selection end to end, including solver objective
   materialization and final selected-layout scoring.

The ordinary compile/autotune/launch, manifest evaluation, and initial graph
build are retained as `trace_capture_seconds`, but are explicitly setup rather
than one of the three reported phase measurements. The default solve is
Experiment 4's whole-tensor canonical DP, repeated three times; quotient
scoring is repeated five times. Experiments 5 and 6 can be requested with
`RELAY_SOLVE_TIMES_SEARCH_EXPERIMENTS`.

Experiment 12 only needs the existing access-manifest plugin and never loads
the layout-rewrite pass. Submit from the repository root:

```bash
# Tuolumne: 29 jobs, 20 minutes each, pbatch
triton/experiments/submit-experiment-12-tuolumne.bash

# Matrix: 29 jobs, 15 minutes each, pdebug
triton/experiments/submit-experiment-12-matrix.bash
```

If Matrix `pdebug` is stalled, use the same short request on `pbatch`:

```bash
RELAY_APPENDIX_MATRIX_PARTITION=pbatch \
  triton/experiments/submit-experiment-12-matrix.bash
```

Targeted retries and alternate grammars use, for example:

```bash
RELAY_SOLVE_TIMES_CASES="gemm--square softmax--large" \
RELAY_SOLVE_TIMES_SEARCH_EXPERIMENTS="4 5 6" \
  triton/experiments/submit-experiment-12-tuolumne.bash
```

Per-case reports and flat rows live under
`results/experiment-12/<platform>/grammar-e4/<operator>--<config>/`. Build the
suite CSV and paper-ready PDF with:

```bash
triton/.venv/bin/python triton/experiments/analyze-solve-times.py \
  --platform tuolumne --search-experiment 4
triton/.venv-matrix/bin/python triton/experiments/analyze-solve-times.py \
  --platform matrix --search-experiment 4
```

## GEMV working-set sweep

The GEMV sweep keeps `M=1024`, `BLOCK=64`, FP32, and varies `K` over 64, 128,
256, 512, 1024, 2048, and 4096. It uses Experiment 1's whole-tensor grammar and
the all-component stratification by default. Each `K` is an independent,
resumable scheduler job using the same report/CSV/PDF workflow:

```bash
triton/experiments/submit-gemv-sweep-tuolumne.bash
triton/experiments/submit-gemv-sweep-matrix.bash
```

Set `RELAY_GEMV_SWEEP_STRATIFICATIONS="all issue temporal"` to run all three
sampling variants. The `K` list, experiment grammar, panel size, pool
multiplier, and profiling count can also be overridden with the corresponding
`RELAY_GEMV_SWEEP_*` variables in the submission script.

## Rescoring recorded counters and tuning tau

The rescore workers use the current stratified profiler corpus by default.
They launch each default kernel once to reconstruct its automatic graph,
generate all three experiment panels for each stratification, and join the
saved counter aggregates by `mapping_id`. They never invoke rocprof or NCU.
Run one command on each cluster from the repository root:

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
and stratification can be regenerated with `rescore-tuolumne-job.bash` or
`rescore-matrix-job.bash`, passing `--case` and `--counter-source`.

After both platforms are present, fit both device profiles and rewrite all
reports, CSV analysis, and PDFs with the fitted `tau` values:

```bash
.venv/bin/python triton/experiments/tune_tau.py
```

Running the rescore commands again later reuses the same recorded counters and
automatically applies the checked-in `tau-profiles.json`; rerunning
`tune_tau.py` does not launch a GPU. `results-old` remains the historical
uniform-sampling corpus and is intentionally not compatible with the new
stratified panels.
