# Solver-frontier speedup experiment

`experiments/solver_frontier.py` connects RELAY's grammar search directly to
the five HIP kernel evaluators. For each kernel it:

1. builds the traced objectives and the kernel's configured component weights;
2. solves `G_S` by exhaustive cut-point enumeration, `G_C` by the exact
   canonical count-grid dynamic program, and `G_A` by the affine access-block
   count-grid dynamic program;
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

Exact analytical score ties remain separate for `G_S` and `G_C` because their
generated address expressions can have different runtimes. The much larger
`G_A` word language retains one deterministic representative for analytically
equivalent DP paths. Benchmark commands are deduplicated across the baseline
and all three grammars.

The affine-access lemma requires a distributive access lattice. The current
SYRK `A` edge spaces provide a concrete counterexample, so the experiment
records `G_A` as not applicable for SYRK rather than substituting a heuristic
grammar.

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

The checked-in report uses 10 samples, 5 launches per sample, 3 warmups,
ROCm 7.0.2, `gfx942`, and the ordinary five-cost Pareto frontier. It contains
223 unique benchmark mappings after cross-grammar and baseline deduplication.
The 211 existing `G_S`/`G_C` mappings were reused from the preceding report;
only 12 distinct new affine layouts required GPU measurements, and every one
passed correctness validation.

| Kernel | `G_S` frontier | `G_C` frontier | `G_A` frontier | `G_S` speedup | `G_C` speedup | `G_A` speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ATAX | 6 | 10 | 4 | 1.791x | 1.791x | 1.768x |
| GEMM | 12 | 12 | 8 | 1.057x | 1.057x | 1.083x |
| GESUMMV | 88 | 181 | 17 | 2.121x | 2.121x | 2.121x |
| MVT | 2 | 2 | 2 | 1.858x | 1.858x | 1.956x |
| SYRK | 4 | 4 | N/A | 2.334x | 2.334x | N/A |

These are measured best-in-frontier results, not predicted rankings. The
objectives and component weights are the current experimental model, and this
single N=1024 run is not holdout evidence that the model generalizes.
