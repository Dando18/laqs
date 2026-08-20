# Solver-frontier speedup experiment

`experiments/solver_frontier.py` connects RELAY's grammar search directly to
the five HIP kernel evaluators. For each kernel it:

1. builds the traced objectives and the kernel's configured component weights;
2. solves `G_S` by exhaustive cut-point enumeration and `G_C` by the exact
   canonical count-grid dynamic program;
3. forms distinct joint layouts for every target matrix;
4. retains the ordinary exact Pareto frontier over
   `(Q_fine, J_peak, J_area, codegen runs, codegen XORs)`;
6. correctness-checks and times every distinct frontier layout; and
7. reports the fastest measured frontier member relative to a full row-major
   baseline.

Exact analytical score ties remain separate because their generated address
expressions can have different runtimes. Benchmark commands are deduplicated
across the baseline and both grammars.

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
  --prepare-only \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
```

Then time the retained layouts in a single-GPU allocation:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/solver_frontier.py \
  --resume \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
```

Every completed layout is checkpointed immediately. If an allocation ends
early, repeat the same `--resume` command. `--max-benchmarks COUNT` can be used
to split work deliberately across allocations.

The default outputs are:

- `results/solver_frontier.json`: configuration, full analytical frontiers,
  scores, raw timing samples, evaluator commands/output, and selected winners;
- `results/solver_frontier_speedup.png`: grouped baseline/solver speedups.

## Current N=256 MI300A result

The checked-in report uses 10 samples, 5 launches per sample, 3 warmups,
ROCm 7.0.2, `gfx942`, and the ordinary five-cost Pareto frontier. It contains
211 unique benchmark mappings after cross-grammar and baseline deduplication.

| Kernel | `G_S` frontier | `G_C` frontier | `G_S` speedup | `G_C` speedup |
| --- | ---: | ---: | ---: | ---: |
| ATAX | 6 | 10 | 1.416x | 1.416x |
| GEMM | 12 | 12 | 1.001x | 1.001x |
| GESUMMV | 88 | 181 | 2.036x | 2.036x |
| MVT | 2 | 2 | 1.101x | 1.101x |
| SYRK | 3 | 3 | 2.452x | 2.452x |

These are measured best-in-frontier results, not predicted rankings. The
objectives and component weights are the current experimental model, and this
single N=256 run is not holdout evidence that the model generalizes.
