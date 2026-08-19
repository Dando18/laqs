# Multi-kernel layout score/runtime experiment

`experiments/layout_ranking.py` compares modeled RELAY locality scores with
measured HIP runtimes for five square FP64 kernels: ATAX, GEMM, GESUMMV, MVT,
and SYRK. By default it evaluates all five at `N=256`, `512`, and `1024`.
Repeated `--kernel` and `--size` options select a smaller or different matrix
of experiments.

All score, runtime, and rank values are ascending costs, so lower is better.

## Kernels and layouts

The selected canonical layout is applied uniformly to every target matrix in
a kernel. Context vectors retain fixed contiguous layouts. This keeps the
experiment a reproducible control set instead of taking a cross product of
operand layouts.

| Kernel | Modeled operation | Layout targets | Context arrays | Group shape |
| --- | --- | --- | --- | --- |
| ATAX | `tmp=A*x; y=A^T*tmp` | A | x, tmp, y | one-dimensional |
| GEMM | `C=alpha*A*B+beta*C` | A, B, C | none | two-dimensional |
| GESUMMV | `y=alpha*A*x+beta*B*x` | A, B | x, y | one-dimensional |
| MVT | `x1=beta*x1+alpha*A*y1`; `x2=beta*x2+alpha*A^T*y2` | A | x1, x2, y1, y2 | one-dimensional |
| SYRK | `C=alpha*A*A^T+beta*C` | A, C | none | two-dimensional |

ATAX times both dependent kernels as one operation. MVT computes its opposing
row and transpose products in one kernel. SYRK computes the full result, not
only one triangle, so all output threads follow the modeled access structure.

For every `N x N` matrix the experiment includes:

| Case | Canonical word |
| --- | --- |
| `row_major` | `j^log2(N) i^log2(N)` |
| `column_major` | `i^log2(N) j^log2(N)` |
| `tile8_row_major` | `jjjiii` |
| `tile8_column_major` | `iiijjj` |
| `tile16_row_major` | `jjjjiiii` |
| `tile16_column_major` | `iiiijjjj` |
| `tile32_row_major` | `jjjjjiiiii` |
| `tile32_column_major` | `iiiiijjjjj` |
| `tileIxJ_row_major` | `j^log2(J) i^log2(I)` |
| `tileIxJ_column_major` | `i^log2(I) j^log2(J)` |
| `tile16_interleaved` | `jijijiji` |
| `tile32_interleaved` | `jijijijiji` |
| `tile8x8_canonical_WORD` | every six-bit word with three `i` and three `j` bits |
| `tile8x16_canonical_WORD` | every seven-bit word with three `i` and four `j` bits |

The rectangular `(I, J)` shapes are `(8,16)`, `(16,8)`, `(8,32)`, `(32,8)`,
`(16,32)`, and `(32,16)`. Tile cases larger than `N` are omitted. Tiled words
describe the inner tile; the outer tile grid is row-major. Words list physical
element-address bits from least significant to most significant.

The two exhaustive families contain `C(6,3)=20` and `C(7,3)=35` words. Their
row- and column-inner endpoints already occur among the controls above, so
they add 18 and 33 cases rather than duplicating those endpoints. The complete
family contains 73 cases. Repeat `--layout-case NAME` to select
an ordered subset by the names above, including the concrete rectangular name
such as `tile8x16_row_major`. The first occurrence wins when a name is repeated.

Every score records address-code generation costs. `runs` counts contiguous
source-mode runs in the canonical word; `xors` counts XORs in a general linear
address expression and is zero for this canonical experiment family. These
costs are reported separately and do not change the scalar locality score.

## Kernel objective models

The objective components construct the hyperedges consumed by the unchanged
score equations in [Scoring realized layouts](scoring.md). `grounded` means
the hyperedge membership comes directly from a traced wave memory instruction.
`hypothesis` means the grouping or scale is a proposed model of reuse or cache
locality. A hypothesis label deliberately does not claim that a hardware cache
has exactly that capacity, replacement policy, or sharing boundary.

The current objective definitions and default `tau` weights are below. A
slash-separated lane-weight list is ordered as lanes 8, 16, 32, and 64. Zero
weight keeps a diagnostic component and its hyperedges in the report but
excludes it from all three scalar aggregates.

ATAX uses these scopes for both matrix-vector stages:

| Component | Provenance | Region B | Tau | Hyperedge meaning |
| --- | --- | ---: | ---: | --- |
| `wave_load.64B` | grounded | 64 | 0 | one traced A wave load from either stage |
| `stage1_wave_load.64B` | grounded | 64 | 0.25 | one traced first-stage A wave load |
| `output_store.64B` | grounded | 64 | 0 | one traced tmp or y vector store |
| `wave_lane_group.lane{8,16,32,64}.*` | hypothesis | 64/128/256/512 | 0/4/0/0 | nested contiguous lane groups over both A streams |
| `stage1_wave_neighborhood.256B` | hypothesis | 256 | 1 | first-stage A wave values at a calibrated cache-neighborhood scale |
| `lane_reuse.128B.window16` | hypothesis | 128 | 0 | one lane's 16 consecutive reduction values in either stage |
| `wave_neighborhood.512B` | hypothesis | 512 | 0 | one wave's A values at a broader locality scale |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | A row or column panel shared by all waves at one reduction step |
| `wave_phase.4096B` | hypothesis | 4096 | 2 | one wave's complete row-wise or column-wise A pass |

GEMM uses these scopes:

| Component | Provenance | Region B | Tau | Hyperedge meaning |
| --- | --- | ---: | ---: | --- |
| `wave_load.64B` | grounded | 64 | 4 | one traced A, B, or C wave load |
| `output_store.64B` | grounded | 64 | 0 | one traced C wave store |
| `B.wave_lane_group.lane{8,16,32,64}.*` | hypothesis | 64/128/256/512 | 2/2/0/0 | nested contiguous lane groups for B loads |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | one lane's 16 consecutive A or B k-loop values |
| `wave_neighborhood.512B` | hypothesis | 512 | 1 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0 | unique A or B values reused by a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 0 | 16 consecutive A/B load pairs for one wave |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete k-loop working set |

GESUMMV uses these scopes:

| Component | Provenance | Region B | Tau | Hyperedge meaning |
| --- | --- | ---: | ---: | --- |
| `wave_load.64B` | grounded | 64 | 0 | one traced A or B wave load |
| `output_store.64B` | grounded | 64 | 0 | one traced y wave store |
| `wave_lane_group.lane{8,16,32,64}.*` | hypothesis | 64/128/256/512 | 0/0/0/0.5 | nested contiguous lane groups over matrix loads |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | one lane's 16 consecutive A or B values |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.5 | one wave's 64 FP64 matrix values at a broader scale |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | the A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 4 | one wave's complete matrix-read phase |

MVT uses these scopes for the opposing row and transpose streams:

| Component | Provenance | Region B | Tau | Hyperedge meaning |
| --- | --- | ---: | ---: | --- |
| `wave_load.64B` | grounded | 64 | 0 | one traced row or transpose A wave load |
| `output_store.64B` | grounded | 64 | 0 | one traced x1 or x2 vector store |
| `A.wave_lane_group.lane{8,16,32,64}.*` | hypothesis | 64/128/256/512 | 0/0/0/0 | nested contiguous lane groups over A loads |
| `row_lane_stream.128B.window16` | hypothesis | 128 | 0 | one lane's 16 consecutive `A[i,j]` values |
| `row_lane_stream.512B.window16` | hypothesis | 512 | 0 | the same row-lane window in a broader diagnostic neighborhood |
| `transpose_lane_stream.128B.window16` | hypothesis | 128 | 0 | one lane's 16 consecutive `A[j,i]` values |
| `wave_neighborhood.512B` | hypothesis | 512 | 0 | one row or transpose wave load at a broader scale |
| `transpose_wave_neighborhood.512B` | hypothesis | 512 | 0 | one transpose wave load at the symmetric-wave scale |
| `transpose_wave_neighborhood.{1024,4096,8192}B` | hypothesis | 1024/4096/8192 | 0.0625 each | one transpose wave load at three empirically calibrated cache-neighborhood scales |
| `workgroup_step_cross.2048B` | hypothesis | 2048 | 0 | row and column arms touched by a workgroup at one inner step |
| `wave_pattern_window.4096B` | hypothesis | 4096 | 0 | 16 consecutive loads from one directional stream |
| `wave_pattern_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete row or transpose stream |

SYRK uses these scopes:

| Component | Provenance | Region B | Tau | Hyperedge meaning |
| --- | --- | ---: | ---: | --- |
| `wave_load.64B` | grounded | 64 | 1 | one traced A or C wave load |
| `output_store.64B` | grounded | 64 | 0 | one traced C wave store |
| `A.row_j_lane_group.lane{8,16,32,64}.*` | hypothesis | 64/128/256/512 | 4/0.25/0/0 | nested lane groups for the `A[j,k]` stream |
| `A.paired_row_reuse.128B.window16` | hypothesis | 128 | 0.25 | eight consecutive k steps from both A row streams for one lane |
| `A.wave_neighborhood.512B` | hypothesis | 512 | 0 | one A wave load at a broader locality scale |
| `A.workgroup_k_column.256B` | hypothesis | 256 | 0 | unique A rows used by the workgroup at one k step |
| `A.wave_k_window.4096B` | hypothesis | 4096 | 0 | 16 consecutive k steps from both A streams for one wave |
| `A.wave_inner_phase.32768B` | hypothesis | 32768 | 1 | one wave's complete pair of A row streams |

These are explicit sparse weights, not hyperedge multiplicities or asserted
hardware constants. The reports repeat every expanded component's exact name,
provenance, region size, description, and applied weight so both active and
zero-weight hypotheses remain auditable.

For every kernel and size, the experiment also reports the exact score Pareto
frontier over the locality vector from the notes plus explicit codegen costs:

```text
(Q for wave_load.64B, J_peak, J_area, codegen runs, codegen XORs)
```

All five entries are minimized. A layout is included when no other measured
layout case is no greater in every entry and strictly smaller in at least one.
Exact cost ties are retained. This is a frontier of modeled costs only; runtime
and timing variation are not Pareto objectives.

The report also constructs four fine-locality-gated frontiers. For `delta` in
0%, 1%, 5%, and 10%, it first forms

```text
L_delta = { A : Q_fine(A) <= (1 + delta) Q_fine* }
```

and then Pareto-filters `L_delta` over `(J_peak, J_area, codegen runs,
codegen XORs)`. The gate prioritizes the grounded fine-load quantity before
trading among reuse/locality hypotheses and address-code costs. Both frontier
families remain independent of measured runtime.

## Frontier candidate-generation scorecard

A completed timing report treats the Pareto frontier as a retained candidate
set rather than as a total runtime ordering. For benchmark instance `e`, the
reported oracle regret is

```text
R_e(F_e) = min runtime in F_e / min runtime in L_e - 1.
```

Here `L_e` is the layout family actually enumerated by this experiment, not
the unbounded space of every representable address mapping.

The report aggregates its mean, median, and maximum across instances. It also
records the frontier size and retained fraction `|F_e| / |L_e|` for every
kernel/size pair and renders a retained-fraction-versus-regret scatter plot.

For epsilon values 0%, 0.25%, 0.5%, 1%, 2%, and 5%, an epsilon-optimal layout
has median runtime at most `(1 + epsilon)` times the measured optimum. The
scorecard reports and plots:

- instance coverage: the fraction of frontiers retaining at least one such
  layout;
- purity: the epsilon-optimal fraction of the frontier;
- enrichment: purity divided by epsilon-optimal prevalence in the full tested
  layout set; and
- the exact expected coverage of a uniformly random subset with the same size
  as each frontier.

For exact-winner coverage, the report also gives expected random hits and the
Poisson-binomial probability that independent size-matched random subsets
would obtain at least the observed hit count.

The top-k diagnostic orders layouts by the selected scalar score. Layout name
is used only as a deterministic tie-breaker so each budget contains exactly
`k` candidates. For every `k`, the JSON records best-runtime regret and
epsilon coverage; the plot shows median, mean, and maximum regret. This tests
the scalar ordering, whereas frontier regret tests analytical candidate
generation.

For each cost-vector definition, layouts with exactly equal modeled vectors
are grouped before looking at runtime. The report records

```text
spread(G) = max median runtime in G / min median runtime in G - 1.
```

Large non-singleton spread identifies a runtime discriminator absent from the
quotient/locality/codegen vector; small spread means the vector successfully
collapses layouts that behave similarly. Singleton groups are retained in the
JSON but excluded from the mean, median, and maximum spread summaries.

## Frontier information ladder

The five-cost frontier deliberately compresses every component excess into
`J_peak` and `J_area`. A completed report now tests that compression against a
sequence of increasingly informative, runtime-independent frontiers:

```text
F_agg     = (Q_fine, J_peak, J_area, runs, XORs)
F_active  = (Q_fine, every e_k with tau_k > 0, runs, XORs)
F_all     = (Q_fine, every existing e_k, runs, XORs)
F_split   = F_all plus source-separated component excesses
F_dense-d = (every Q_s(V_d) at every feasible d, runs, XORs)
```

`F_split` retains the existing joint component and adds separate array, ATAX
stage, MVT row/transpose, or SYRK row-i/row-j quantities when hyperedge source
provenance exposes those partitions. These are diagnostic coordinates only:
they receive no fitted tau and do not modify the kernel problem definitions.
`F_dense-d` evaluates each existing target-array edge family for element-region
dimensions from zero through the full target address width. It therefore tests
whether the original sparse choice of byte scales discarded useful quotient
information.

For every representation and kernel/size instance, the JSON records frontier
regret, retained fraction, exact and 1%-winner coverage, exact-vector
equivalence groups and runtime spreads, and score-dominance/runtime-order
violations. A violation is marked confirmed only when the dominated layout's
maximum observed sample is smaller than the dominator's minimum observed
sample.

If a representation misses an exact measured winner, its dominance
certificate lists every analytical dominator and `e_dominator - e_winner` for
every existing component. This distinguishes aggregate compression from a
zero-weight scope, sparse scale sampling, or a limitation that remains even in
the dense quotient representation.

The experiment also repeatedly removes each nondominated set to form Pareto
layers `F_1, F_2, ...`. For every cumulative candidate set
`C_L = union(F_1, ..., F_L)`, it reports regret and retained fraction. The
Pareto-depth plot shows how conservatively retaining additional layers trades
screening power for empirical winner recovery.

The tau-weight robustness ablation independently multiplies each nonzero tau
by a uniformly selected value from `{0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5}`. It
recomputes `J_area`, rebuilds the five-cost frontier, and reports oracle regret
and retained fraction for every trial. `Q_fine`, `J_peak`, runs, and XORs do
not depend on tau and therefore remain unchanged. The defaults are 128 trials
per kernel/size with seed 0; use `--tau-perturbation-trials` and
`--tau-perturbation-seed` to change them reproducibly.

The six plots are written to `<output stem>_plots/` by default:

- `epsilon_optimal_coverage.png`;
- `retained_fraction_vs_regret.png`;
- `purity_and_enrichment.png`; and
- `top_k_regret.png`; and
- `tau_weight_robustness.png`; and
- `pareto_depth_regret.png`.

Use `--plots-dir DIRECTORY` to select another location. Plotting requires the
`experiments` optional dependencies from `pyproject.toml`. Score-only and
incomplete checkpoints contain no aggregate runtime scorecard because the
required medians are not yet available; completed kernel/size groups still
receive their per-group JSON analysis as checkpoints progress.

## Score-only check

Scoring needs no GPU. This example builds ten kernel/size groups—all five
kernels at two sizes—and writes both the detailed JSON and Markdown tables:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --size 256 --size 512 \
  --score-only \
  --output results/layout-ranking-score-only.json
```

The Markdown path defaults to the JSON path with a `.md` suffix. Use
`--markdown PATH` to choose it explicitly.

Omitting both `--kernel` and `--size` selects all five kernels and all three
default sizes. For a quick ordered subset of layouts, repeat `--layout-case`:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --kernel atax --size 256 \
  --layout-case row_major \
  --layout-case tile8x16_row_major \
  --score-only --output results/atax-score-only.json
```

The earlier two-kernel score-only run is retained at
[`../results/layout_ranking_expanded_score_only.md`](../results/layout_ranking_expanded_score_only.md).
It covers 22 layouts for GEMM and GESUMMV at all three sizes and predates the
ATAX, MVT, and SYRK additions.

Problem events and objective components are built once per kernel and size,
then reused for every layout. Repeated
`--component-weight OBJECTIVE=WEIGHT` options override a problem-provided
default wherever that objective exists in the selected kernels. An objective
name that exists in none of the selected kernels is an error. Weight 0 excludes
a component from scalar aggregates while retaining its detailed score. See
[Scoring realized layouts](scoring.md) for score formulas.

## Hardware run

Each evaluator generates, compiles, validates, and times its HIP operation per
layout. Run a checkpointed `N=256` sweep inside a GPU allocation:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --compiler /opt/rocm-7.0.2/bin/hipcc \
  --arch gfx942 \
  --size 256 \
  --samples 5 \
  --iterations 3 \
  --warmup 2 \
  --max-benchmarks 40 \
  --output results/layout-ranking.json
```

The default sizes are `256`, `512`, and `1024`. Supplying any `--size` replaces
that set; repeat the option for several sizes. Supplying any `--kernel`
similarly replaces the default five-kernel set. A complete default sweep is
1,095 separately compiled cases, so use checkpointed chunks that fit the cluster
allocation limit.

The evaluators check five output points before timing and report HIP-event
kernel time only. Packing, layout validation, allocation, copies, and
compilation are excluded. Every timing sample, median, mean, minimum, standard
deviation, and GFLOP/s value is retained.

All kernel/size/layout benchmark jobs are shuffled together using `--seed` to
reduce fixed-order drift. Each layout is still compiled and measured in a
separate process; samples are not interleaved across layouts.

On systems with short allocation limits, split the work into checkpointed
chunks. The initial command saves all scores and then atomically updates both
reports after every completed benchmark:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --size 256 --samples 5 --iterations 3 --warmup 2 \
  --max-benchmarks 20 \
  --output results/layout-ranking.json
```

Resume with exactly the same experiment settings plus `--resume`; the chunk
size may change or be omitted:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --size 256 --samples 5 --iterations 3 --warmup 2 \
  --resume \
  --output results/layout-ranking.json
```

Resume validates the kernel, size, workgroup, score mode, CLI weight overrides,
timing, compiler, architecture, device, and seed settings. It skips score
construction and every layout that already has timing samples. Partial
Markdown tables mark unfinished cells as `pending`; variation-aware metrics
appear once a kernel/size group is complete.

For a size whose score construction itself approaches the allocation limit,
prepare the checkpoint on a CPU node first, then resume it on the GPU:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --size 1024 --samples 5 --iterations 3 --warmup 2 \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --prepare-checkpoint \
  --output results/layout-ranking-n1024.json

flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --size 1024 --samples 5 --iterations 3 --warmup 2 \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --resume \
  --output results/layout-ranking-n1024.json
```

`--prepare-checkpoint` runs no evaluator and deliberately leaves the report
incomplete. Resume validates stored CLI weight overrides but assumes that the
problem objective code and its default weights have not changed since the
checkpoint was prepared.

Workgroup controls are separated by launch geometry:

- `--block-x` and `--block-y` configure the two-dimensional GEMM and SYRK
  groups; and
- `--block-size` configures the one-dimensional ATAX, GESUMMV, and MVT groups.

The shared `--samples`, `--iterations`, `--warmup`, `--device`, `--compiler`,
and `--arch` options are passed to every evaluator.

## Reusing timing samples while revising objectives

`--reuse-timings JSON` scores the current objective definitions and attaches
completed timing records from existing reports. Repeat the option to combine
disjoint kernel or size reports into one result. Matching is exact on kernel,
matrix size, layout-case name, and canonical word. It then recomputes all raw
score ranks and variation-aware metrics and records the absolute source paths
in both reports. No GPU or evaluator process is used.

For example, this reproduces the revised five-kernel report from the baseline
timings:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --size 256 --samples 5 --iterations 3 --warmup 2 \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --reuse-timings results/layout_ranking_five_kernel_baseline.json \
  --output results/layout_ranking_five_kernel_final.json
```

The source must contain a completed record for every selected case. Timing
reuse checks the samples, iterations, warmups, device, compiler, architecture,
and per-kernel workgroup against the requested report. It cannot detect a
changed evaluator implementation, so inspect the retained source commands when
code has changed. It cannot be combined with `--score-only`, `--resume`, or
`--max-benchmarks`.

When expanding a layout family, `--seed-timings JSON` provides partial reuse:
matching completed records are copied into a newly scored checkpoint, while
unmatched layouts remain pending and run normally. It uses the same exact
identity and benchmark-configuration checks as full reuse. Combine it with
`--prepare-checkpoint` on a CPU node, then resume the output inside GPU
allocations. Unlike `--reuse-timings`, a seed source need not cover every
requested layout.

## Objective calibration result

The checked-in calibration set contains all 22 layouts for all five kernels at
`N=256` on an MI300A (`gfx942`), using a 32x32 group for GEMM/SYRK, a 128-thread
group for ATAX/GESUMMV/MVT, five timing samples, three timed iterations per
sample, and two warmups. The initial measurements and objective model are in
[`../results/layout_ranking_five_kernel_baseline.md`](../results/layout_ranking_five_kernel_baseline.md),
with complete raw data in its adjacent JSON file.

The component behavior was inspected against those timings, then the objective
definitions and sparse weights were revised. In particular, ATAX gained an
explicit traced first-stage term and a first-stage cache-neighborhood
hypothesis; MVT gained three explicitly hypothetical transpose-neighborhood
scales. GEMM, GESUMMV, and SYRK retained their hyperedge families with revised
weights. The current tables above are the resulting model. The same raw timing
samples were attached with `--reuse-timings` to produce
[`../results/layout_ranking_five_kernel_final.md`](../results/layout_ranking_five_kernel_final.md).

That file remains the historical N=256 calibration result. After the expanded
three-size sweep exposed the MVT N=1024 miss, the current MVT model added two
zero-weight 512-byte stream diagnostics and disabled both active symmetric
512-byte wave copies. Those two copies encode the same full-wave edge family
and jointly caused the false dominance. A discrete ablation over all three
stored MVT sizes found that activating the new diagnostic scopes recovered the
winner but degraded total-score ranks more sharply; they therefore remain
explicit hypotheses at tau zero. This second adjustment is also in-sample.

For the displayed `weighted-normalized-excess` score, the exact
variation-aware results changed as follows:

| Kernel | Accurate ranks, baseline -> revised | Accuracy, baseline -> revised | Mean rank error, baseline -> revised | Max rank error, baseline -> revised |
| --- | ---: | ---: | ---: | ---: |
| ATAX | 3/22 -> 11/22 | 13.6% -> 50.0% | 2.614 -> 1.773 | 8.5 -> 6.0 |
| GEMM | 16/22 -> 19/22 | 72.7% -> 86.4% | 1.273 -> 0.273 | 12.0 -> 3.5 |
| GESUMMV | 13/22 -> 17/22 | 59.1% -> 77.3% | 0.614 -> 0.341 | 4.5 -> 3.5 |
| MVT | 6/22 -> 20/22 | 27.3% -> 90.9% | 1.977 -> 0.364 | 10.0 -> 5.0 |
| SYRK | 12/22 -> 12/22 | 54.5% -> 54.5% | 1.682 -> 0.500 | 6.5 -> 3.0 |
| All cases | 50/110 -> 79/110 | 45.5% -> 71.8% | 1.632 -> 0.650 | 12.0 -> 6.0 |

This comparison is deliberately exploratory and entirely in-sample: the same
`N=256` layouts and timing samples informed the revisions and measured their
improvement. It is evidence that the revised objectives fit this calibration
set, not an unbiased estimate of reliability. Every cache/reuse term labeled
`hypothesis` remains a modeling proposal even when its fitted weight is
nonzero.

## Fresh post-calibration multi-size result

After fixing the objective definitions and weights, the experiment was run
again at all three default sizes. The complete combined report is
[`../results/layout_ranking.md`](../results/layout_ranking.md), with all 1,095
raw benchmark records in the adjacent JSON. All cases reported correctness
PASS. The sweep preserves 330 exact-configuration measurements from the prior
22-layout run and adds 765 newly measured exhaustive-sweep cases. The JSON
records the three underlying timing reports as seed sources.

For `weighted-normalized-excess`, the fresh variation-aware summary is:

| Kernel | N=256 accuracy / mean error | N=512 accuracy / mean error | N=1024 accuracy / mean error |
| --- | ---: | ---: | ---: |
| ATAX | 20/73 (27.4%) / 8.870 | 18/73 (24.7%) / 8.548 | 12/73 (16.4%) / 11.349 |
| GEMM | 34/73 (46.6%) / 8.062 | 17/73 (23.3%) / 11.295 | 70/73 (95.9%) / 0.315 |
| GESUMMV | 22/73 (30.1%) / 12.514 | 11/73 (15.1%) / 14.541 | 5/73 (6.8%) / 14.021 |
| MVT | 17/73 (23.3%) / 13.027 | 15/73 (20.5%) / 14.219 | 15/73 (20.5%) / 12.295 |
| SYRK | 32/73 (43.8%) / 5.897 | 39/73 (53.4%) / 3.014 | 67/73 (91.8%) / 0.370 |
| All five | 125/365 (34.2%) / 9.674 | 100/365 (27.4%) / 10.323 | 169/365 (46.3%) / 7.670 |

Across all sizes, 394/1,095 score ranks lie in their observed timing-derived
rank ranges (36.0%), with mean rank error 9.222. The exhaustive families expose
many more tied or near-tied score structures and substantially weaken the
complete-rank diagnostic. The MVT entries include the explicitly in-sample
second adjustment described above; the other four kernels have not been
refitted to this expanded sweep. Measurements on other devices and repeated
randomized sweeps remain necessary before treating any weight table as
portable.

The candidate-generation scorecard remains stronger than the complete-rank
diagnostic. The main frontier retains 10.137% of layouts on average, contains
the exact measured winner in 9/15 instances, and reaches 13/15 coverage by
epsilon=1% and 15/15 by epsilon=5%. Oracle regret has median 0%, mean 0.3713%,
and maximum 2.4318%. Each MVT size now retains 9/73 layouts and contains its
exact measured winner; MVT N=1024 regret falls from 9.5120% to 0%.

All four tested fine-locality gates produce the same aggregate result on this
discrete layout family: 7/15 exact winners, median regret 0.1826%, mean regret
3.0336%, maximum regret 22.3181%, and 6.301% mean retention. The equality from
0% through 10% means no additional sampled `Q_fine` level falls inside those
slack thresholds; it is a result, not an implementation shortcut.

Across 128 tau-perturbation trials per instance, aggregate oracle regret has
median 0%, mean 0.3603%, and maximum 2.4318%; retained fraction has median
10.959% and mean 10.700%. Thus the tested weight perturbations do not materially
change the headline frontier regret, although the per-instance JSON and plot
retain the full trial distributions.

The frontier-information refresh reused these exact 1,095 timing records; no
GPU kernels were rerun. Its aggregate results are:

| Representation | Exact winners | Within 1% | Mean regret | Max regret | Mean retained | Max exact-alias spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `F_agg` | 9/15 | 13/15 | 0.3713% | 2.4318% | 10.137% | 57.286% |
| `F_active` | 12/15 | 13/15 | 0.2557% | 2.4318% | 18.356% | 57.286% |
| `F_all` | 10/15 | 13/15 | 0.3263% | 2.4318% | 26.027% | 50.926% |
| `F_split` | 11/15 | 13/15 | 0.2681% | 2.4318% | 37.626% | 50.926% |
| `F_dense-d` | 14/15 | 15/15 | 0.0582% | 0.8736% | 68.767% | 31.817% |

The MVT correction shows that its former miss was caused by activated objective
selection: two copies of the symmetric 512-byte full-wave family made the
winner analytically dominated. Removing those terms recovers the winner in the
aggregate and active-component frontiers. The coarser aggregate now has more
exact aliases, however, raising maximum alias spread from 50.926% to 57.286%.
Dense scale curves recover every winner within 1%, but retain more than two
thirds of the tested layouts. They also leave exact-vector runtime spreads as
large as 31.817%. Thus the revised aggregate is a better candidate screen, not
a better total ordering, and region counts still omit a material runtime
discriminator.

Adding coordinates can break a tie in a coarser projection, so the identities
of Pareto members need not be nested even though a richer representation
contains more information. This is why `F_all` can have lower exact-winner
coverage than `F_active`: zero-weight diagnostic coordinates distinguish an
active-vector tie in the empirically wrong direction for some instances.

Pareto depth provides a cheaper conservative alternative to the very large
dense frontier. The first three aggregate layers retain 51.324% of layouts on
average and reduce maximum regret to 0.9117%, reaching 15/15 instances within
1%. Two dense layers retain 87.671% and contain every exact winner. The checked-
in Markdown report includes every missed-winner dominator and component delta;
the JSON retains all layer memberships and complete score vectors.

### Static MVT code-generation audit

The four MVT N=1024 words proposed in the analysis notes were also compiled
with ROCm 7.0.2 clang for `gfx942`. This was a compiler-only audit: generated
executables were not launched and no GPU timings were recollected. The loop
instruction count spans the backward-branch body. "Integer/address" excludes
loads, stores, waits, floating-point operations, comparisons, and the branch;
it includes address formation and loop/pointer updates. Address depth is the
longest dependent integer-ALU chain feeding either matrix global load.

| Word | Audit role | Stored median ms | Kernel instructions | Loop instructions | Loop integer/address | VGPR / SGPR | Spills / private B | Matrix load form | Address depth |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `iijjjji` | empirical winner | 0.132054 | 77 | 33 | 22 | 22 / 30 | 0 / 0 | 2x `global_load_dwordx2` | 5 |
| `iijjji` | best aggregate-frontier layout | 0.144615 | 75 | 32 | 21 | 22 / 30 | 0 / 0 | 2x `global_load_dwordx2` | 5 |
| `jiiijjj` | low-score three-run layout | 0.188534 | 78 | 34 | 23 | 19 / 30 | 0 / 0 | 2x `global_load_dwordx2` | 5 |
| `jjjiiij` | additional slow three-run layout | 0.175240 | 74 | 32 | 21 | 19 / 30 | 0 / 0 | 2x `global_load_dwordx2` | 5 |

All four use zero LDS, have zero scalar/vector spills, issue the same matrix
load form, and have the same apparent address dependency depth. The slow
layouts actually use fewer VGPRs, while total and loop instruction counts do
not order the measured runtimes. The code object does not record achieved
occupancy, and this static audit did not launch a kernel, so it makes no claim
about dynamic occupancy. It does rule out an obvious spill, load-form, or long
address-chain explanation for the MVT gap and strengthens the case for a
locality or unmodeled microarchitectural discriminator.

The audit can be reproduced without a GPU by emitting each source with
`kernels/mvt/evaluate.py --n 1024 --emit-only`, compiling it with
`hipcc -O3 -std=c++17 --offload-arch=gfx942 --genco`, extracting the gfx942
bundle with `clang-offload-bundler`, and inspecting metadata/disassembly with
`llvm-readelf --notes` and `llvm-objdump -d --mcpu=gfx942`.

## Raw ranks and variation-aware metrics

The tables always show raw values:

- score rank is computed from the exact selected score;
- runtime rank is computed from the exact sample median;
- exact ties receive their average one-based rank; and
- `rank delta = score rank - runtime rank`.

Timing variation does not modify these values. It is used only for summary
metrics.

For layout `l`, let its observed timing interval be

```text
I_l = [minimum raw sample, maximum raw sample].
```

Another layout is definitely faster than `l` only if its entire interval is
below `I_l`; it is definitely slower only if its interval is entirely above
`I_l`. With `K` layouts, the plausible runtime-rank bounds are

```text
lower_l = 1 + number of layouts definitely faster than l
upper_l = K - number of layouts definitely slower than l.
```

Overlapping intervals are deliberately left unordered. A score rank is
variation-aware accurate when it lies in `[lower_l, upper_l]`. Its error is
the distance to that interval, or zero when it lies inside. Each score mode is
summarized with:

- the number and fraction of layout ranks inside their plausible intervals;
- mean rank error; and
- maximum rank error.

This directly handles an adjacent raw-rank swap when both timing sample ranges
overlap. It is intentionally simple and conservative. The interval is an
observed range, not a confidence interval, and it generally widens as more
samples expose more variability. More rigorous resampling or interleaved
measurement can replace this metric later without changing the raw report.

## Reports

The JSON report contains:

- the selected kernels, sizes, workgroups, score mode, kernel-default weights,
  command-line weight overrides, and timing configuration;
- the global randomized benchmark order;
- one run record for every kernel and size;
- each run's Pareto objective definitions, members, objective values, and a
  per-layout membership flag;
- fine-locality gates and frontiers at all four deltas, plus their completed
  runtime scorecards;
- exact score-equivalence groups and runtime spreads for the main and gated
  vectors;
- the five-level frontier-information ladder, complete diagnostic signatures,
  dominance violations and certificates, and cumulative Pareto layers;
- tau-perturbation factors, seeds, every trial result, and aggregate regret and
  retained-fraction summaries;
- per-layout canonical words, component scores, aggregate scores, per-array and
  total codegen costs, exact ranks, timing statistics, and raw samples;
- observed timing intervals and plausible runtime-rank ranges;
- variation-aware metrics for every score mode; and
- evaluator commands with complete stdout and stderr.

The Markdown report contains a cross-kernel/size summary followed by an
objective/provenance table, the score Pareto frontier, one raw score/runtime
table, and one variation-aware metric table for every run.

## Interpretation limits

This experiment measures a bounded canonical layout family on one generated
kernel implementation at a time. Runs and XORs are structural codegen proxies,
not calibrated instruction latency or throughput models. The score does not
model register pressure, cache-set or channel mapping, allocation base phase,
cross-array interference, or layout conversion cost. The min–max variation
metric can say that a modeled order is not contradicted by these samples; it
cannot establish that the order will generalize to another run or system.
