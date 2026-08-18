# GEMM layout score/runtime experiment

`experiments/gemm_layout_ranking.py` compares modeled RELAY locality scores
with measured HIP kernel runtimes.  It is deliberately a small first
experiment: square FP64 GEMM, one representative workgroup in the score model,
and conventional canonical matrix layouts.

All score and runtime columns are costs, so lower values rank first.

## Layout cases

The experiment applies the same layout to A, B, and C.  This produces a small,
reproducible control set rather than a cross product of every operand choice.
For an `N x N` matrix it includes:

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

Tile cases larger than `N` are omitted.  Tiled words describe the inner tile;
the outer tile grid is row-major.  Words always list physical element-address
bits from least significant to most significant.

## Score-only check

Scoring needs no GPU and is the quickest way to validate the trace, layout
definitions, weights, and report path:

```bash
.venv/bin/python experiments/gemm_layout_ranking.py \
  --n 256 \
  --score-mode weighted-normalized-excess \
  --score-only \
  --output results/gemm-layout-ranking-score-only.json
```

The problem events and objective components are built once, then reused for
every layout.  The layout scorer also caches repeated logical point offsets
within each candidate; this matters for the overlapping temporal-window
hyperedges in GEMM.

Use repeated `--component-weight OBJECTIVE=WEIGHT` options to change the
component weights.  Unspecified objectives use weight 1; weight 0 excludes an
objective from all scalar aggregates while retaining its detailed score.
The available `--score-mode` values and their formulas are documented in
[Scoring realized layouts](scoring.md).

## Hardware run

The existing evaluator generates, compiles, validates, and times one HIP GEMM
for each layout.  Run the experiment inside a GPU allocation with ROCm loaded.
For the LLNL MI300A environment used during development:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/gemm_layout_ranking.py \
  --n 1024 \
  --arch gfx942 \
  --samples 10 \
  --iterations 5 \
  --warmup 3 \
  --output results/gemm-layout-ranking-1024.json
```

`kernels/gemm/evaluate.py` checks five output points before timing and reports
HIP-event kernel time only.  Packing, validation, allocation, and copies are
outside the measured interval.  The experiment uses median milliseconds for
runtime ranking and retains the mean, minimum, standard deviation, GFLOP/s,
and every timing sample.

The layout benchmark order is shuffled using `--seed` (default 0) to reduce a
fixed ordering bias.  Each layout is still compiled and timed as a separate
process; the experiment does not interleave samples across layouts.

Useful runtime options are:

- `--block-x` and `--block-y` for the workgroup shape used by both trace and
  generated kernel;
- `--device` for the HIP device ordinal;
- `--compiler` for the compiler command passed to the evaluator;
- `--arch` for the HIP offload architecture; and
- `--samples`, `--iterations`, and `--warmup` for timing depth.

## Rank comparison and JSON

Score rank and runtime rank are ascending, one-based ranks.  Exact ties receive
their average rank.  For each layout the terminal report shows

```text
rank delta = score rank - runtime rank.
```

A positive delta means the model ranked that layout worse than hardware did;
a negative delta means the model was more optimistic.  The summary also
reports Spearman's rank correlation using average-tie ranks.  A value near 1
means the score and runtime orders agree, 0 means little monotonic agreement,
and -1 means reversed order.  One hardware pass reports a correlation for all
three supported score modes; `--score-mode` selects the ranking shown in the
main table.  Correlation is reported as undefined when one rank vector is
constant.

The optional JSON report records:

- the complete experiment and benchmark configuration;
- the selected score mode and component-weight overrides;
- randomized benchmark order;
- per-layout canonical words and all component/aggregate score details;
- every aggregate score plus score rank, runtime rank, and rank delta for the
  selected mode;
- Spearman correlation for every supported score mode;
- parsed runtime statistics and raw timing samples; and
- each evaluator command plus its complete stdout and stderr.

Keeping raw process output makes later parsing changes auditable.

### Checked-in baseline

`results/gemm_layout_ranking_256.json` contains an end-to-end MI300A run with
ROCm 7.0.2, `gfx942`, a 32x32 workgroup, 10 samples, 5 launches per sample,
and 3 warmups.  With `weighted-normalized-excess`, six of the eight score
ranks matched median-runtime rank exactly.  The two remaining row-inner tiled
layouts exchanged adjacent ranks:

| Score rank | Runtime rank | Layout | Score | Median ms |
| ---: | ---: | --- | ---: | ---: |
| 1 | 1 | `tile32_row_major` | 16.132368 | 0.065364 |
| 2 | 2 | `row_major` | 17.132368 | 0.065476 |
| 3 | 4 | `tile16_row_major` | 20.132368 | 0.065568 |
| 4 | 3 | `tile8_row_major` | 23.568052 | 0.065492 |
| 5 | 5 | `tile8_column_major` | 69.953146 | 0.096680 |
| 6 | 6 | `tile16_column_major` | 120.517461 | 0.159392 |
| 7 | 7 | `tile32_column_major` | 176.517461 | 0.159548 |
| 8 | 8 | `column_major` | 192.517461 | 0.169048 |

Spearman's `rho` for `weighted-normalized-excess` is `0.976190`.  On the same
timings, `weighted-region-count` gives `0.738095` and
`peak-normalized-excess` gives `0.540062`.  This is an encouraging smoke
result for the weighted normalized mode, not a reliability estimate: the four
row-oriented medians differ by only about 0.0002 ms, and repeated or
interleaved runs are needed to quantify their ordering stability.

## Interpretation limits

This experiment measures correlation for this kernel, trace construction,
hardware, compiler, workgroup, and small layout family.  It does not establish
that a score mode generalizes to other kernels or GPUs.  In particular, the
current score does not model instruction cost, register pressure, cache-set or
channel mapping, base-address phase, cross-array interference, or layout
conversion cost.  Treat low correlation as evidence for revising hypotheses,
and high correlation as a reason to test a broader and independently chosen
layout set.
