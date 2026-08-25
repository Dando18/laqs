# Solver-frontier speedup experiment

`experiments/solver_frontier.py` connects RELAY's grammar search directly to
the five HIP kernel evaluators. For each kernel it:

1. builds the traced objectives and the kernel's configured component weights;
2. solves `G_S` by exhaustive cut-point enumeration, `G_C` by the exact
   canonical count-grid dynamic program, bounded `G_OC` by exhaustive inner
   realization plus a canonical-suffix DP, and `G_A` by the affine
   access-block count-grid dynamic program;
3. forms distinct joint layouts for every target matrix;
4. retains the ordinary exact Pareto frontier over
   `(Q_fine, J_peak, J_area, codegen runs, codegen XORs)`;
5. correctness-checks and times every retained frontier layout; and
6. reports the fastest measured frontier member relative to a full row-major
   baseline.

`G_A` first verifies that every modeled edge is an exact affine coset. It
constructs the event-direction lattice, verifies distributivity, derives a
sparse fixed adapted basis from its join-irreducible access blocks, and orders
those block directions with the generalized count-grid DP. The emitted layout
is a full-rank binary matrix, so mixed directions are evaluated with the same
correctness and timing harness as canonical words. Its codegen objectives use
contiguous source-field groups and the row-weight XOR proxy described in the
codegen notes.

The `G_OC` experiment is exact for an explicitly bounded language: every
canonical layout plus every block-diagonal outer-canonical layout whose
arbitrary invertible inner map has at most `--goc-max-inner-bits` bits. The
default and current implementation limit are four, which permits exhaustive
realization of every inner matrix. For each cut point, a DP searches every
canonical interleaving of the remaining outer bits. The optional upper-right
coupling block is zero: it leaves every low-address quotient subspace
unchanged, while nonzero entries only add XOR terms, so it cannot improve the
experiment's locality/codegen Pareto objectives.

Exact analytical score ties remain separate for `G_S` and `G_C` because their
generated address expressions can have different runtimes. The much larger
`G_OC` preserves the complete `G_C` tie family so its measured candidate set
respects `G_C` subset `G_OC`, while equivalent noncanonical inner realizations
are represented once. `G_A` retains one deterministic representative for
analytically equivalent score paths. Benchmark commands are deduplicated
across the baseline and all four grammars.

The affine-access lemma requires every active edge to be an exact affine coset
and the resulting access lattice to be distributive. The current universal
scopes violate the first premise for MVT and SYRK, so the experiment records
`G_A` as not applicable rather than aborting or silently substituting a
heuristic grammar.

`--frontier-type fine-gated` selects the alternative formulation from the
notes: first require `Q_fine <= (1 + epsilon) Q_fine*`, then Pareto-filter over
`(J_peak, J_area, runs, XORs)`. Its `epsilon` is controlled by
`--fine-tolerance`, which defaults to 0.05. The ordinary five-cost frontier is
the experiment default because `results/layout_ranking.md` found it to be a
better candidate generator than the fine-locality gates.

## Run

Prepare the exact search checkpoint on a login node:

```bash
.venv/bin/python experiments/solver_frontier.py \
  --size 1024 \
  --prepare-only \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
```

Then time the retained layouts in a single-GPU allocation:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/solver_frontier.py \
  --size 1024 \
  --resume \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
```

Every completed layout is checkpointed immediately. If an allocation ends
early, repeat the same `--resume` command. `--max-benchmarks COUNT` can be used
to split work deliberately across allocations. The CLI defaults to `N=256`;
the explicit `--size 1024` above reproduces the checked-in report.

`--reuse-solvers REPORT` and `--reuse-timings REPORT` can import compatible
analytical results and exact matching raw benchmark records when extending an
existing experiment. Reused evidence is identified in the JSON report.

The default outputs are:

- `results/solver_frontier.json`: configuration, full analytical frontiers,
  scores, raw timing samples, evaluator commands/output, and selected winners;
- `results/solver_frontier_speedup.png`: grouped baseline/solver speedups.

## Current N=1024 MI300A result

The checked-in report was regenerated after the universal access-scope change.
It uses 10 samples, 5 launches per sample, 3 warmups, ROCm 7.0.2, `gfx942`,
and the ordinary five-cost Pareto frontier. All 167 unique benchmark mappings
were freshly correctness-checked and timed on MI300A. `G_OC` retained exactly
the same mappings as `G_C` for every applicable kernel, despite its much larger
analytical search cost.

| Kernel | `G_S` frontier | `G_C` frontier | `G_OC` frontier | `G_A` frontier | `G_S` speedup | `G_C` speedup | `G_OC` speedup | `G_A` speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATAX | 10 | 12 | 12 | 6 | 1.810x | 1.810x | 1.810x | 1.751x |
| GEMM | 13 | 17 | 17 | 15 | 1.056x | 1.056x | 1.056x | 0.610x |
| GESUMMV | 38 | 52 | 52 | 24 | 1.823x | 1.823x | 1.823x | 1.587x |
| MVT | 9 | 11 | 11 | N/A | 2.699x | 2.699x | 2.699x | N/A |
| SYRK | 11 | 23 | 23 | N/A | 2.322x | 2.326x | 2.326x | N/A |

### Why `G_A` underperforms

The regenerated data separates three issues:

- MVT and SYRK are outside the current affine grammar because an active
  universal temporal-window edge is not an affine coset. Their missing bars
  are applicability failures, not measured slowdowns.
- On the three applicable kernels, `G_A` has no exact layout mappings in common
  with `G_S`, `G_C`, or `G_OC`. Its fixed adapted access-block basis and
  deterministic equal-score representative exclude the measured compact-
  grammar winners.
- Relative to the measured cross-grammar union, the best `G_A` layout has
  3.401% regret on ATAX, 73.007% on GEMM, and 14.842% on GESUMMV. Cross-grammar
  analytical comparison already dominates all 6 ATAX affine candidates, 3 of
  15 GEMM candidates, and 20 of 24 GESUMMV candidates with compact-grammar
  candidates. The remaining affine candidates survive mostly through a lower
  predicted peak objective, but that tradeoff does not translate to runtime.

GEMM makes the code-generation gap most visible: the best `G_A` mapping uses
an arbitrary linear descriptor and costs 19 source-field runs plus 9 XORs,
versus 9 runs and no XORs for the measured union winner. The fixed-basis affine
grammar is therefore not a replacement for `G_C`; a useful next design is a
union/hybrid grammar that preserves compact layouts, retains meaningful affine
ties, and decomposes or otherwise handles non-affine edges explicitly.

These are measured best-in-frontier results, not predicted rankings. The
objectives and component weights are the current experimental model, and this
single N=1024 run is not holdout evidence that the model generalizes.
