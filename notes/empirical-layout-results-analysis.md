# Completed empirical layout study

All three studies completed. The 75 row/column, 114 sum, and 24 GEMM worker
records (213 total, including tuning, held-out evaluation and manual audits) have
valid record and batch hashes and no recorded kernel-validation or primitive
rejections. Each held-out comparison uses three fresh processes and three shared
input placements per process. Confidence intervals below use processes as the
replication unit. Launch schedules are fixed; packing is excluded from timings.

| Case | Panel | Analytical top-1 | Empirical automatic choice | Ordinary / empirical, H (95% CI) | Top-1 / empirical, S (95% CI) |
| --- | ---: | --- | --- | --- | --- |
| MI300A row/column | 91 | 4×8 | 1024×8 | 1.9037 (1.8940–1.9135) | 1.0080 (1.0041–1.0119) |
| MI300A sum | 109 | 2×16 | 64×128 | 1.2065 (0.9345–1.5576) | 1.4716 (0.9910–2.1853) |
| H100 GEMM, A only | 19 | 2×32 | ordinary | 1.0000 (self-comparison) | 1.0192 (1.0137–1.0248) |

Sources: [row/column](../triton/experiments/results/layout-study-row_column-v1/tuolumne/row_column--small/expert/study.json),
[sum](../triton/experiments/results/layout-study-sum-v1/tuolumne/sum--small/expert/study.json),
[GEMM](../triton/experiments/results/layout-study-gemm-v1/matrix/gemm--asymmetric/expert/study.json).
The empirical choice estimates the explored family under its automatic
realization; it is not a global optimum over layouts, schedules or realizations.

## Row/column: strong headroom, little selection loss

The empirical winner runs at about 3.80 µs versus ordinary's 7.24 µs, giving
1.904×. The analytical 4×8 map gives 1.889× and takes only about 0.8% longer than
the empirical winner. Automatic/manual runtime ratios are 1.0011 for the winner
and 1.0144 for top-1. Both maps have J=1.08. This case does not justify a broad
backend rewrite or an expensive exhaustive search as the deployment strategy.
The small within-family difference should also be read alongside the 3.27%
maximum identity-control deviation (same-pointer deviation is 0.31%). The large
layout improvement comfortably exceeds that control budget.

Packing strengthens the practical case for the analytical map: median measured
packing/allocation costs are approximately 0.093 ms for 4×8 and 0.276 ms for
1024×8. Their per-placement break-even counts versus this fixed ordinary schedule
are 27–31 and 68–83 uses, respectively. Neither is a one-use end-to-end win. These
counts are not the prior experiment's comparison against separately tuned ordinary
storage.

## The strongest model diagnostic: equal features and flags, different runtimes

Row/column's ordinary and 1024×64 layouts have identical full component vectors,
identical partial flags at the recorded scored/protected depths, and J=2. Yet
1024×64 gives **1.4245×** held-out speedup (95% CI 1.4133–1.4359). Its manual
realization also improves performance: 1.3932× (1.3302–1.4592). This contrast was
chosen on training data and then independently confirmed.

The captured ordinary and automatic 1024×64 binaries both report 76 registers,
zero spills, 64 shared bytes, and the same memory-primitive counts. Static
instruction counts are 298 and 302. The changed map's automatic/manual comparison
records comparable ownership and no changes in the checked primitive/ownership
fields. These observations narrow the explanation but do not prove identical
instruction scheduling or identify the hardware mechanism.

Reweighting the existing component vector cannot distinguish this pair. Runtime
effects outside that vector matter even when the partial flag is unchanged.
This validates retaining physical maps for empirical diagnostics. It does not
invalidate flag deduplication as an exact optimization of the declared objective.
The ordinary/1024×64 pair is the clearest next profiling contrast; compare address
generation and scheduling alongside memory behavior before assigning a cause.

## Sum: analytical selection is poor, but average headroom is unresolved

Analytical 2×16 achieves only 0.8198× ordinary performance (95% CI
0.7126–0.9431). Manual indexing gives essentially the same result; its
automatic/manual ratio is 1.0001. The empirical choice, 64×128, has worse J=0.3
than top-1's J=0, and has the same full vector and partial flag as ordinary.
Its apparent mean win does not pass independent confirmation: H's interval
includes one. The selection-gap interval also includes one.

The raw held-out timings explain the wide intervals:

| Process | Ordinary times at the three placements (µs) | 64×128 automatic times (µs) |
| --- | --- | --- |
| 0 | 20.642, 20.720, 20.528 | 15.480, 15.486, 15.490 |
| 1 | 21.070, 20.628, 15.172 | 15.432, 15.440, 15.434 |
| 2 | 20.466, 15.150, 15.146 | 15.438, 15.434, 15.432 |

The winner is stable near 15.4 µs; ordinary changes regime. Consequently,
ordinary/winner is about 1.33 on six placements and 0.98 on three. The processes
report the same host/GPU UUID and repeat the same virtual input addresses for
corresponding placements. This is not enough to attribute the difference to
virtual address placement, physical backing, cache state, or another mechanism.
Identity and same-pointer control deviations are small (0.33% and 0.08%); those
controls do not eliminate the layout-dependent regime difference.

Manual 64×128 is about 2.2% faster than automatic but inherits the same uncertainty
versus ordinary. The independent equal-feature 16×8 versus 512×8 contrast is also
inconclusive: ratio 1.277, CI 0.924–1.765. More frozen-map confirmation and a
controlled investigation of the ordinary timing regimes are needed before
claiming reliable sum headroom or changing component weights. Reselecting a new
winner on the held-out measurements would defeat the experimental split.

## H100 GEMM: no confirmed automatic win within this panel

Ordinary won the 19-map A-only tuning study. H=1 is therefore a self-comparison,
not a confidence bound establishing that all possible layouts lack headroom.
Held-out analytical 2×32 gives 0.9811× ordinary performance, approximately a 1.9%
runtime penalty. The manual version gives 1.0184×, and automatic/manual is 1.0380
(CI 1.0320–1.0440). A small realization gap remains, but this experiment has not
exposed a large profitable layout hidden by the automatic backend.

The 1.8% manual gain is below the harness's conservative gain-plus-control
criterion with a 1.15% identity deviation. It should not be promoted to a robust
deployment win. The checked ownership and primitive fields match; registers and
instruction counts differ (automatic: 96 registers/610 instructions; manual:
118/602), with no spills and identical shared-memory usage. The eight poor-map
manual audits do not reveal a substantial win; their best manual speedup is
about 0.823×. This supports retaining ordinary for this workload while preserving
the small 2×32 realization discrepancy as a focused backend regression case.

## Next priorities

1. Profile the confirmed row/column ordinary versus 1024×64 equal-feature contrast.
   It offers stronger evidence about model omissions than another general
   compiler change.
2. Repeat only the frozen sum comparisons with more independent allocations and
   processes, preserving per-placement times and investigating the two ordinary
   timing regimes. Keep selection frozen.
3. Retain ordinary as the practical H100 GEMM choice for this automatic A-only
   panel. The separate conditional GEMM counter script is not triggered by these
   headroom results.

No new GPU jobs or counter collections were launched for this analysis.
