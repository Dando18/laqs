# Multi-kernel layout score/runtime experiment

`experiments/layout_ranking.py` compares modeled RELAY locality scores with
measured HIP runtimes for square FP64 GEMM and GESUMMV. By default it evaluates
both kernels at `N=256`, `512`, and `1024`. Repeated `--kernel` and `--size`
options select a smaller or different matrix of experiments.

All score, runtime, and rank values are ascending costs, so lower is better.

## Kernels and layouts

GEMM applies one layout uniformly to A, B, and C. GESUMMV applies one layout
uniformly to A and B while its x and y context vectors stay contiguous. This
keeps the experiment a small reproducible control set instead of taking a
cross product of operand layouts.

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

Tile cases larger than `N` are omitted. Tiled words describe the inner tile;
the outer tile grid is row-major. Words list physical element-address bits
from least significant to most significant.

## Kernel objective models

The objective components construct the hyperedges consumed by the unchanged
score equations in [Scoring realized layouts](scoring.md). `grounded` means
the hyperedge membership comes directly from a traced wave memory instruction.
`hypothesis` means the grouping or scale is a proposed model of reuse or cache
locality. A hypothesis label deliberately does not claim that a hardware cache
has exactly that capacity, replacement policy, or sharing boundary.

GESUMMV uses these scopes:

| Component | Provenance | Tau | Hyperedge meaning |
| --- | --- | --- | --- |
| `wave_load.64B` | grounded | 1 | one traced A or B wave load |
| `output_store.64B` | grounded | `1/(2N)` | one traced y wave store, frequency-scaled against two inner-loop matrix loads |
| `wave_lane_group.lane{8,16,32,64}.*` | hypothesis | 0.125 each | nested contiguous lane groups; the four correlated levels share total weight 0.5 |
| `lane_reuse.128B.window16` | hypothesis | 2 | one lane's 16 consecutive A or B values |
| `wave_neighborhood.512B` | hypothesis | 0.25 | the broader neighborhood of one 64-value wave load |
| `workgroup_step_panel.1024B` | hypothesis | 0.5 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 0.5 | one wave's complete matrix-read phase |

GEMM uses these scopes:

| Component | Provenance | Tau | Hyperedge meaning |
| --- | --- | --- | --- |
| `wave_load.64B` | grounded | 4 | one traced A, B, or C wave load; the grounded transaction family balances all reuse/cache hypotheses combined |
| `output_store.64B` | grounded | `1/(2N)` | one traced C wave store, frequency-scaled against the inner A/B loads |
| `B.wave_lane_group.lane{8,16,32,64}.*` | hypothesis | 0.125 each | nested B lane groups; the correlated family shares total weight 0.5 |
| `lane_reuse.128B.window16` | hypothesis | 2 | one lane's 16 consecutive A or B k-loop values |
| `wave_neighborhood.512B` | hypothesis | 0.25 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 0.5 | unique A or B values shared by the workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 1 | 16 consecutive A/B load pairs for one wave |
| `wave_inner_phase.32768B` | hypothesis | 0.25 | one wave's complete k-loop working set at a broad scale |

The weights are explicit sparse `tau` choices, not hyperedge multiplicities.
In particular, reducing the four nested lane weights prevents four correlated
views of the same accesses from dominating `J_area`. The reports repeat every
component's exact name, provenance, region size, description, and applied
weight so these hypotheses remain auditable.

For every kernel and size, the experiment also reports the exact score Pareto
frontier over the vector from the notes:

```text
(Q for wave_load.64B, J_peak, J_area)
```

All three entries are minimized. A layout is included when no other measured
layout case is no greater in all three and strictly smaller in at least one.
Exact score ties are retained. This is a frontier of modeled scores only;
runtime and timing variation are not Pareto objectives.

## Score-only check

Scoring needs no GPU. This example builds four kernel/size groups and writes
both the detailed JSON and Markdown tables:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --kernel gemm --kernel gesummv \
  --size 256 --size 512 \
  --score-only \
  --output results/layout-ranking-score-only.json
```

The Markdown path defaults to the JSON path with a `.md` suffix. Use
`--markdown PATH` to choose it explicitly.

Problem events and objective components are built once per kernel and size,
then reused for every layout. Repeated
`--component-weight OBJECTIVE=WEIGHT` options override a problem-provided
default wherever that objective exists in the selected kernels. An objective
name that exists in none of the selected kernels is an error. Weight 0 excludes
a component from scalar aggregates while retaining its detailed score. See
[Scoring realized layouts](scoring.md) for score formulas.

## Hardware run

Each evaluator generates, compiles, validates, and times one HIP kernel per
layout. Run the combined experiment inside a GPU allocation:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --compiler /opt/rocm-7.0.2/bin/hipcc \
  --arch gfx942 \
  --samples 10 \
  --iterations 5 \
  --warmup 3 \
  --output results/layout-ranking.json
```

The default sizes are `256`, `512`, and `1024`. Supplying any `--size` replaces
that set; repeat the option for several sizes. Supplying any `--kernel`
similarly replaces the default pair.

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
  --max-benchmarks 20 \
  --output results/layout-ranking.json
```

Resume with exactly the same experiment settings plus `--resume`; the chunk
size may change or be omitted:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --resume \
  --output results/layout-ranking.json
```

Resume validates the kernel, size, workgroup, score, weight, timing, compiler,
architecture, device, and seed settings. It skips score construction and every
layout that already has timing samples. Partial Markdown tables mark unfinished
cells as `pending`; variation-aware metrics appear once a kernel/size group is
complete.

Kernel-specific workgroup controls are explicitly named:

- `--gemm-block-x` and `--gemm-block-y` configure GEMM's two-dimensional group;
- `--gesummv-block-size` configures GESUMMV's one-dimensional group.

The shared `--samples`, `--iterations`, `--warmup`, `--device`, `--compiler`,
and `--arch` options are passed to both evaluators.

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
- per-layout canonical words, component scores, aggregate scores, exact ranks,
  timing statistics, and raw samples;
- observed timing intervals and plausible runtime-rank ranges;
- variation-aware metrics for every score mode; and
- evaluator commands with complete stdout and stderr.

The Markdown report contains a cross-kernel/size summary followed by an
objective/provenance table, the score Pareto frontier, one raw score/runtime
table, and one variation-aware metric table for every run.

## Interpretation limits

This experiment measures a small traditional layout family on one generated
kernel implementation at a time. The score does not model instruction cost,
register pressure, cache-set or channel mapping, allocation base phase,
cross-array interference, or layout conversion cost. The min–max variation
metric can say that a modeled order is not contradicted by these samples; it
cannot establish that the order will generalize to another run or system.
