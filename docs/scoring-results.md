# Exhaustive canonical scoring results

`experiments/scoring_results.py` produces the plot-oriented result corpus for
evaluating RELAY's analytical layout selections against an exhaustive runtime
oracle.

The runner defaults to `--grammar canonical`; `--grammar standard` enumerates
the four-form cut-point `G_S` language instead. Use distinct output, raw, and
plan paths for each grammar.

## Experiment scope

For an `N x N` matrix, a full canonical word contains `log2(N)` `i` symbols
and `log2(N)` `j` symbols. The experiment enumerates

```text
|G_C(shared)| = binomial(2 log2(N), log2(N)).
```

The same word is applied to every target matrix in a kernel. Context vectors
retain their fixed contiguous layouts. This matches the shared-layout policy
used by `experiments/layout_ranking.py` and gives 48,620 layouts at N=512 and
184,756 at N=1024.

This scope is intentionally not the independent multi-array Cartesian product
used by `experiments/solver_frontier.py`. With independent words, GEMM would
contain `|G_C|^3` joint layouts and an exhaustive runtime oracle would not be a
practical benchmark.

For each kernel and size, the analytical plan records the exact locality
frontier over ascending costs

```text
(Q_fine, J_peak, J_area)
```

`J_place` is deliberately not a fourth Pareto coordinate. The plan forms a 5%
coordinatewise locality shell around every frontier member, adds the best
placement member from each shell bucket, and orders that bounded union by the
selected global-phase statistic. The fixed budgets are 1, 3, 5, and 10, and
the summary reports regret@K. Codegen runs and XORs remain annotations and
deterministic tie breakers, not dominance coordinates. Exact locality ties are
retained.

and these additional selection mechanisms:

- `lowest_hardware_area`: one minimum-`J_area` layout, with deterministic
  codegen-run and word tie breaks;
- `top5_hardware_area`: the first five layouts under that scalar ordering;
- `placement_rerank_at_{1,3,5,10}`: fixed-budget corrected-`J_place`
  recommendations from the locality shell; and
- `row_major_baseline`: the complete row-major word.

For every completed selection, its best median time is reported. Regret is

```text
best selected median / exhaustive-oracle median - 1.
```

Regret remains `null` until the exhaustive oracle is complete, so a partial
sweep cannot accidentally be presented as oracle evidence.

## Oracle primitive-feature audit

Before changing placement semantics, the complete GESUMMV-1024 `G_S` family
was checked over `Q_fine`, every nonzero locality excess footprint, raw pair
excess, and normalized placement. The measured oracle
`jjiiiiiiiijjjjjjjjii` is componentwise dominated by
`jjiiiiiiijjjjjjjjiii`. The dominator is itself only 0.482% slower than the
oracle, but the certificate proves that no nonnegative reweighting of the old
features can make the oracle optimal.

The checked-in corrected audit expands the vector to 450 coordinates,
including every global phase plus within-array and cross-array placement. The
same word still dominates the oracle. This is a model-information failure, not
a grammar-reachability or coefficient-tuning failure.

Use both flags to attach the complete oracle, reranker-leader, and locality-
frontier vectors to the scoring summary:

```bash
.venv/bin/python experiments/scoring_results.py \
  --kernel gesummv --size 1024 --grammar standard --prepare-only \
  --dump-oracle-components --check-oracle-feature-dominance \
  --seed-raw results/standard_scoring_mi300a.raw.jsonl
```

## Corrected placement reranker

The August 26, 2026 corrected robust rescore reused all 1,640 exhaustive MI300A
timings and launched no GPU work. Allocation phases are global across cohorts,
pair excess is normalized within each cohort, and within/cross placement are
reported separately. The table reports the locality-frontier size, bounded
pool size, and regret at each recommendation budget:

| Kernel | N | Locality | Pool | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATAX | 512 | 4 | 5 | 10.644% | 6.130% | 0.858% | 0.858% |
| ATAX | 1024 | 4 | 5 | 7.867% | 7.867% | 1.169% | 1.169% |
| GEMM | 512 | 10 | 10 | 17.643% | 15.465% | 15.465% | 0.147% |
| GEMM | 1024 | 11 | 11 | 1.210% | 1.210% | 0.921% | 0.501% |
| GESUMMV | 512 | 10 | 10 | 13.913% | 13.913% | 13.913% | 13.913% |
| GESUMMV | 1024 | 11 | 11 | 20.878% | 20.878% | 20.878% | 20.878% |
| MVT | 512 | 4 | 4 | 13.077% | 13.077% | 13.077% | 13.077% |
| MVT | 1024 | 4 | 4 | 12.522% | 12.522% | 12.522% | 12.522% |
| SYRK | 512 | 10 | 10 | 1.197% | 1.161% | 0.663% | 0.663% |
| SYRK | 1024 | 11 | 11 | 3.276% | 2.661% | 2.661% | 2.661% |

Mean regret is 10.223%, 9.488%, 8.213%, and 6.639% at K=1, 3, 5, and
10. The mean bounded pool is 8.1 candidates, versus 24.5 for the superseded
unrestricted four-axis frontier. The corrected reranker therefore fixes the
candidate-set architecture but does not yet have sufficient top-K precision:
at K=5 it retains 4.8 layouts on average but is worse than the old unbounded
frontier's 3.151% mean regret. Expected, robust, and worst-quartile CVaR give
the same GESUMMV-1024 ordering and 20.878% regret.

Corrected artifacts:

- `results/standard_reranked_robust_scoring_mi300a.{plan.json,raw.jsonl,jsonl}`
  contains the ten-case robust reranker plan, imported timings, and summary;
- `results/gesummv1024_oracle_feature_audit_mi300a.{plan.json,raw.jsonl,jsonl}`
  contains the 450-coordinate oracle audit and complete diagnostic panel.

## Historical unrestricted colored rescore

The initial August 26, 2026 mechanism rescore reused all 1,640 MI300A timing
records; it launched no new GPU measurements. The table compares the
locality-only frontier with the same frontier plus independent `J_place`:

| Kernel | N | Locality layouts | Locality regret | Colored layouts | Colored regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATAX | 512 | 4 | 0.858% | 18 | 0.802% |
| ATAX | 1024 | 4 | 1.169% | 32 | 1.169% |
| GEMM | 512 | 10 | 0.147% | 16 | 0.147% |
| GEMM | 1024 | 11 | 0.204% | 26 | 0.204% |
| GESUMMV | 512 | 10 | 13.913% | 23 | 4.667% |
| GESUMMV | 1024 | 11 | 20.878% | 34 | 20.878% |
| MVT | 512 | 4 | 13.077% | 13 | 0.539% |
| MVT | 1024 | 4 | 12.522% | 15 | 0.000% |
| SYRK | 512 | 10 | 0.663% | 27 | 0.596% |
| SYRK | 1024 | 11 | 2.661% | 41 | 2.507% |

`J_place` improves six instances and leaves four unchanged. Mean regret falls
from 6.609% to 3.151%, while mean frontier size grows from 7.9 to 24.5 layouts.
The result supports placement as a useful missing signal, but the unchanged
20.878% GESUMMV-1024 regret and larger candidate sets show that this initial
HBM-stack sketch is not yet sufficient. This four-axis policy is retained as
historical evidence and is superseded by the bounded reranker above.

## Historical sparse flag-fiber materialization

This earlier follow-up run used `--fiber-max-xors 1`. For every shared `G_S` flag, it
constructs the canonical representative and every color-relevant elementary
shear

```text
T_ij: y_i <- y_i XOR y_j, i < j.
```

Only destination bits that feed the resource map are searched: element-offset
bits 9, 10, and 11 for the current MI300A sketch. Each candidate's complete
low-address prefix flag is compared with the source flag. All 41,540 tested
materializations preserve every prefix subspace exactly. Candidates are first
Pareto-filtered within a flag over `(J_place, swizzle XORs)` and then globally
over `(Q_fine, J_peak, J_area, J_place)`.

That global four-axis frontier predates the bounded-reranker correction. The
measurements remain useful, but its selection architecture is not the current
default.

The base oracle remains the exhaustive 146/182 canonical-representative `G_S`
sweep. Negative percentages therefore mean that the enlarged fiber grammar
found a layout faster than every canonical representative; they are not
negative regret against an exhaustive oracle for the enlarged grammar.

| Kernel | N | Base colored regret | Fiber vs. base oracle | Frontier size |
| --- | ---: | ---: | ---: | ---: |
| ATAX | 512 | 0.802% | -4.191% | 18 -> 56 |
| ATAX | 1024 | 1.169% | -3.777% | 32 -> 94 |
| GEMM | 512 | 0.147% | -0.887% | 16 -> 27 |
| GEMM | 1024 | 0.204% | -1.690% | 26 -> 41 |
| GESUMMV | 512 | 4.667% | -1.241% | 23 -> 40 |
| GESUMMV | 1024 | 20.878% | 10.172% | 34 -> 54 |
| MVT | 512 | 0.539% | -4.643% | 13 -> 23 |
| MVT | 1024 | 0.000% | -5.742% | 15 -> 17 |
| SYRK | 512 | 0.596% | -0.554% | 27 -> 48 |
| SYRK | 1024 | 2.507% | -1.087% | 41 -> 95 |

All ten selected times improve, by 4.281% on average relative to the selected
canonical colored frontier. The mean value relative to the canonical `G_S`
oracle changes from 3.151% to -1.364%; nine of ten cases are within 1%. The
main remaining failure is GESUMMV-1024, reduced from 20.878% to 10.172%.
Mean frontier size grows from 24.5 to 49.5 layouts.

Because the canonical timings and fiber timings were collected in separate
sweeps, GEMM-512 and GESUMMV-1024 were also rerun in randomized paired panels.
The fiber winner improved over the best base-colored candidate in all 24
rounds: 0.652% for GEMM-512 (95% bootstrap interval 0.629--0.677%) and 5.499%
for GESUMMV-1024 (5.427--5.565%). Relative to the paired canonical oracle,
their mean differences were -0.538% and 9.197%, respectively. This confirms
that the important improvements are not merely cross-run device drift.

The checked-in artifacts are:

- `results/standard_fiber_scoring_mi300a.plan.json`: analytical fiber search;
- `results/standard_fiber_scoring_mi300a.raw.jsonl`: 1,640 imported identity
  timings plus 347 new selected-fiber measurements;
- `results/standard_fiber_scoring_mi300a.jsonl`: compact summary;
- `results/standard_fiber_interleaved_confirmation_mi300a.json`: randomized
  paired confirmation; and
- `results/plots/gs_flag_fiber_comparison.pdf`: base/fiber comparison figure.

## Files

The default outputs separate compact paper data from the much larger timing
checkpoint:

- `results/canonical_scoring_mi300a.jsonl` contains one summary object per
  kernel/size pair. It includes device identity, layout and frontier sizes, the
  oracle top five, frontier layouts and scores, selection times, and regrets.
- `results/canonical_scoring_mi300a.raw.jsonl` begins with compatible-run
  metadata and then stores one append-only raw timing record per layout. Each
  completed evaluator is flushed and synchronized before the next starts.
- `results/canonical_scoring_mi300a.plan.json` stores exact analytical
  frontiers and selection sets. GPU resumes load this plan rather than
  rebuilding every score.

The summary's `complete` and `oracle.complete` fields become true only when
`base_timed_layout_count == layout_count`. Without a fiber search this is also
`timed_layout_count == layout_count`; fiber runs may contain additional timed
descriptors. `best_observed_*` and `top_observed_layouts` expose partial
progress without labeling it an oracle.
The `frontier.layouts` and every selection's `layouts` contain their analytical
score and their timing when available.

After changing only analytical scoring, `--rescore --prepare-only` rebuilds
and replaces the plan and summary while reusing the append-only raw timing
checkpoint. It never launches a benchmark.

## Cluster workflow

First build the exact analytical plan on a CPU node. The checked-in 73-layout
corpus uses the same 5 samples, 3 iterations, 2 warmups, workgroups, compiler,
and architecture as this experiment's defaults. Its full row- and column-major
descriptors can seed 20 exact evaluator cases across the requested ten groups:

```bash
.venv/bin/python experiments/scoring_results.py \
  --prepare-only \
  --seed-timings results/layout_ranking.json
```

Short tiled descriptors are not imported even when they induce the same
physical permutation as a full word. Their generated address expressions are
different, so treating their timings as full-word evidence would confound the
code-generation part of the experiment.

Then repeatedly run a bounded chunk on one MI300A:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/scoring_results.py \
  --resume \
  --max-benchmarks 40 \
  --compiler /opt/rocm-7.0.2/bin/hipcc \
  --arch gfx942
```

Analytical frontier and comparison layouts are scheduled before the residual
exhaustive sweep. `--max-benchmarks` may be changed between resumes; all other
configuration fields must remain identical. If the allocation ends before the
requested chunk, already appended timing lines remain valid and the same
command can be submitted again.

Use repeated `--kernel` or `--size` arguments with separate output, raw, and
plan paths to split the suite into independently managed corpora. The final
paper file can then be formed by concatenating the completed summary JSONL
files.

## Standard grammar run

The shared `G_S` family has 146 unique words at N=512 and 182 at N=1024. The
analytical plan for all ten groups can be prepared with:

```bash
.venv/bin/python experiments/scoring_results.py \
  --grammar standard --prepare-only \
  --seed-timings results/layout_ranking.json \
  --output results/standard_scoring_mi300a.jsonl \
  --raw-output results/standard_scoring_mi300a.raw.jsonl \
  --plan results/standard_scoring_mi300a.plan.json
```

Then repeat this single-GPU command until every summary record is complete:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/scoring_results.py \
  --grammar standard --resume --max-benchmarks 100 \
  --output results/standard_scoring_mi300a.jsonl \
  --raw-output results/standard_scoring_mi300a.raw.jsonl \
  --plan results/standard_scoring_mi300a.plan.json \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
```

To extend that completed canonical-representative sweep with one sparse
flag-preserving shear, seed a separate checkpoint from the exhaustive raw
timings:

```bash
.venv/bin/python experiments/scoring_results.py \
  --grammar standard --fiber-max-xors 1 --prepare-only \
  --seed-raw results/standard_scoring_mi300a.raw.jsonl \
  --output results/standard_fiber_scoring_mi300a.jsonl \
  --raw-output results/standard_fiber_scoring_mi300a.raw.jsonl \
  --plan results/standard_fiber_scoring_mi300a.plan.json
```

Resume it under Flux with the same three paths and both grammar options. The
runner benchmarks selected swizzled descriptors first; it does not claim an
exhaustive runtime oracle over the enlarged fiber grammar.
