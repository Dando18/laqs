# Simplified RELAY solver

This is a small, readable implementation of the current experimental RELAY layout solver.
It is intended for analysis and iteration, not production deployment.

The library takes:

- metadata for one or more dense arrays;
- logical memory events with lane identities, sites, kinds, local groups, order, weights, and arbitrary metadata;
- local ordered event sequences;
- explicit scope/granularity objective builders; and
- a layout-search configuration.

It returns:

- a bounded candidate family for each target array;
- canonical layout words and/or arbitrary bit-linear inner matrices;
- exact modeled aligned-region scores and packing bounds;
- simple lane-order and indexing proxies;
- bounded joint configurations across target arrays; and
- terminal and JSON reports explaining the search.

The current implementation assumes power-of-two array extents and scalar accesses. The `relay` package deliberately leaves compilation and hardware measurement to companion scripts under `kernels/` and `experiments/`; non-power-of-two fringes and cache/channel response modeling remain out of scope.

## Install and run

The package has no third-party dependencies.

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python examples/row_column_conflict.py
.venv/bin/python examples/gesummv_multi.py --json gesummv.json
.venv/bin/python examples/jacobi_multi.py --json jacobi.json
```

Without installing, run with:

```bash
PYTHONPATH=. .venv/bin/python examples/jacobi_multi.py
```

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Score an existing layout

The public scorer evaluates concrete layouts without running layout search. It
reports each objective's weighted aligned-region count, its capacity-only
packing lower bound, and its normalized excess over that bound. It also
reports per-array and total address-code run/XOR costs, and supports three
explicitly named scalar locality costs:

- `weighted-region-count`;
- `peak-normalized-excess`; and
- `weighted-normalized-excess`.

For example, score a globally row-major layout for every GEMM operand:

```bash
.venv/bin/python bin/score_layout.py kernels/gemm/problem.py \
  --layout all=row-major \
  --score-mode weighted-normalized-excess
```

Each kernel problem supplies explicit default component weights (the notes'
`tau` values). Use `--json` for a machine-readable report,
`--component-weight NAME=VALUE` to override one default, and
`--problem-option problem_size=512` to pass a JSON-valued option into the
problem's `build_config` routine. See
[Scoring realized layouts](docs/scoring.md) for the formulas, API, complete
layout-word convention, CLI contract, and JSON fields.

## Benchmark solver frontiers

The solver-frontier experiment runs the `G_S` exhaustive solver, the `G_C`
canonical dynamic program, the bounded `G_OC` exact search, and the
affine-access `G_A` dynamic program for all five kernels. It benchmarks every
retained layout in their ordinary five-cost Pareto frontiers, selects the
fastest measured member, and plots speedup over full row-major layouts. The
`G_OC` bound is controlled by `--goc-max-inner-bits` and defaults to four.
`G_A` is reported as not applicable when the access lattice is not
distributive, as happens for the current SYRK model:

```bash
.venv/bin/python experiments/solver_frontier.py \
  --size 1024 \
  --prepare-only \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/solver_frontier.py \
  --size 1024 \
  --resume \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942
```

The current N=1024 MI300A results and complete workflow are documented in
[`docs/solver-frontier-experiment.md`](docs/solver-frontier-experiment.md).
The final plot is
[`results/solver_frontier_speedup.png`](results/solver_frontier_speedup.png),
with exact frontiers and raw timing samples retained in the adjacent JSON.

## Compare score and runtime ranks

The combined experiment scores global, square-tiled, rectangular-tiled, and
interleaved canonical layouts for five FP64 kernels: ATAX, GEMM, GESUMMV, MVT,
and SYRK. It runs the matching HIP benchmark for each case and compares
ascending score rank to ascending median-runtime rank. Exact raw numbers and
ranks are written to JSON and Markdown. Variation-aware metrics use observed
timing sample ranges without changing those raw ranks. Each kernel/size report
also includes the exact non-dominated cost frontier over
`(Q_fine, J_peak, J_area, codegen runs, codegen XORs)`; runtime is not a Pareto
objective. It also reports fine-locality-gated frontiers at 0%, 1%, 5%, and
10% slack in `Q_fine`. Completed reports evaluate these frontiers as candidate generators
using best-in-frontier regret, epsilon-optimal coverage, retained fraction,
purity, enrichment, a size-matched random baseline, and top-k scalar-score
regret. Exact-score equivalence groups expose otherwise hidden runtime spread,
and a discrete random tau-weight ablation reports regret/retention robustness.
An information ladder compares the aggregate frontier with active-component,
all-component, source-split, and dense quotient-scale frontiers; it also emits
missed-winner dominance certificates and cumulative Pareto-depth screening
curves. Six PNG plots are written beside the JSON report.

Validate scoring without a GPU:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --size 256 --size 512 --score-only \
  --output results/layout-ranking-score-only.json
```

Run timings from an MI300A allocation:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/layout_ranking.py \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --size 256 --samples 5 --iterations 3 --warmup 2 \
  --max-benchmarks 40 \
  --output results/layout-ranking.json
```

Plot generation uses the optional experiment dependencies:

```bash
.venv/bin/python -m pip install -e '.[experiments]'
```

Repeat the same command with `--resume` to finish a checkpointed hardware run.
The default kernel set contains all five kernels, and the default size set is
256, 512, and 1024. See
[Multi-kernel layout score/runtime experiment](docs/layout-experiment.md) for
the exact layout set, variation-aware rank formula, timing boundary, options,
checkpoint/resume workflow, output fields, and interpretation limits.

The kernel problem files distinguish traced, `grounded` wave scopes from
`hypothesis` reuse and cache-neighborhood scopes. The latter are modeling
assumptions, not claims about a particular cache implementation. Reports list
the provenance, region size, meaning, and applied `tau` for every objective.
The five-kernel `N=256` MI300A measurements before objective calibration are
in
[`results/layout_ranking_five_kernel_baseline.md`](results/layout_ranking_five_kernel_baseline.md).
The revised objective model, rescored against those exact timing samples, is
in
[`results/layout_ranking_five_kernel_final.md`](results/layout_ranking_five_kernel_final.md).
The corresponding JSON files retain complete components, raw samples, and
commands.

The current combined result is
[`results/layout_ranking.md`](results/layout_ranking.md). It contains fresh
post-calibration MI300A measurements for all five kernels at N=256, 512, and
1024. The current experiment family has 73 layouts per group,
including complete 8x8 and 8x16 canonical inner-word sweeps, for 1,095
correctness-checked benchmark cases in total. Its
adjacent JSON retains every raw sample, score component, rank, Pareto member,
frontier-information signature, Pareto depth, and evaluator command.
The current information ladder reaches 9/15, 12/15, 10/15, 11/15, and 14/15
exact-winner coverage from `F_agg` through `F_dense-d`; the dense frontier is
within 1% on all 15 instances but retains 68.8% of the tested candidates. See
the report for the full regret/aliasing tradeoff and dominance certificates.
The experiment documentation also records a compiler-only gfx942 ISA/resource
audit of the four MVT N=1024 words highlighted by that analysis.

The original revised weights were selected by inspecting the same 22-layout,
`N=256` measurements shown in the historical final report. MVT received a
second, explicitly in-sample adjustment after inspecting the three-size sweep:
its duplicated symmetric 512-byte wave terms are now disabled, and two new
stream-specific 512-byte hypotheses remain visible at weight zero. These
changes are not holdout evidence that the model will generalize. The fresh
multi-size sweep continues to expose substantial size sensitivity.
The experiment documentation records both sets of results and a
`--reuse-timings` workflow for reproducibly combining exact timing reports.

A historical GEMM-only `N=256` MI300A run is available at
[`results/gemm_layout_ranking_256.json`](results/gemm_layout_ranking_256.json).
For this sample, the 8x8 and 16x16 row-inner layouts exchanged adjacent raw
ranks while their observed timing ranges overlapped—the situation the new
variation-aware metric is designed to represent.

## Kernel problem/evaluator pairs

Each non-trivial kernel has a `problem.py` that constructs its representative
trace and labeled objectives, plus an `evaluate.py` that generates, validates,
and times a matching HIP implementation:

| Kernel | Operation | Layout targets | Workgroup option |
| --- | --- | --- | --- |
| ATAX | `tmp=A*x; y=A^T*tmp` | A | `--block-size` |
| GEMM | `C=alpha*A*B+beta*C` | A, B, C | `--block-x`, `--block-y` |
| GESUMMV | `y=alpha*A*x+beta*B*x` | A, B | `--block-size` |
| MVT | simultaneous `A*y1` and `A^T*y2` updates | A | `--block-size` |
| SYRK | `C=alpha*A*A^T+beta*C` | A, C | `--block-x`, `--block-y` |

All vector operands use fixed contiguous layouts. During the ranking
experiment, the selected canonical layout is applied uniformly to every target
matrix named in a row.

### GESUMMV example

`kernels/gesummv/problem.py` describes the FP64 kernel

```text
y[i] = alpha * sum_j A[i,j] * x[j] + beta * sum_j B[i,j] * x[j]
```

as one complete representative workgroup trace. A and B are independently
scored layout targets; x and y are fixed contiguous context vectors. The
grounded objectives score traced wave loads and the output store. Explicitly
hypothetical objectives encode 16-access per-lane reuse, nested lane groups,
the two-wave panel reused at each inner-loop step, and broader wave/cache
neighborhoods.

For example, score column-major A and row-major B on a small problem:

```bash
.venv/bin/python bin/score_layout.py kernels/gesummv/problem.py \
  --problem-option problem_size=256 \
  --layout all=row-major --layout A=column-major
```

`kernels/gesummv/evaluate.py` generates a standalone HIP driver, checks the
two generated address mappings and five output values, then reports kernel-only
timings. Its two positional words select A and B. Words list physical address
bits from low to high; omitted coordinate bits address a row-major outer tile
grid. Thus `jjjjiiiiii` means a row-major 64x16 inner tile.

Generate source without a GPU:

```bash
.venv/bin/python kernels/gesummv/evaluate.py \
  jjjjiiiiii jjjjiiiiiii --n 256 --emit-only
```

Compile and time it on an MI300A allocation:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python kernels/gesummv/evaluate.py \
  jjjjiiiiii jjjjiiiiiii --n 256 --arch gfx942
```

Use `--help` for the sample, iteration, warmup, workgroup, compiler, device,
and retained-build-directory controls. Matrix dimensions must be divisible by
the inner tile dimensions selected by each word.

## Minimal DSL example

```python
from relay import *

A = MatrixSpec(
    name="A",
    shape=(256, 256),
    element_bytes=4,
    mode_names=("i", "j"),
    target=True,
    role="read",
)

load = MemoryEvent.make(
    id="load.wave0",
    site="A.load",
    accesses=[
        Access("A", (32, 64 + lane), lane=lane, kind="read")
        for lane in range(64)
    ],
    group="wg0.wave0",
    order=0,
    weight=1024,
    metadata={"workgroup": "wg0", "phase": "main"},
)

problem = RelayProblem(
    matrices=(A,),
    events=(load,),
    sequences=(),
    objectives=(
        SimultaneousRegions("wave128", 128),
        LanePrefixRegions(
            "lane",
            levels=((8, 32), (16, 64), (32, 128), (64, 256)),
        ),
    ),
    config=SolverConfig(
        policy=ScorePolicy(
            kind="pareto",
            order=("wave128", "lane.lane16.64B", "runs", "xors"),
        ),
        tile_shapes={"A": ((4, 4), (8, 8), (16, 8), (16, 16))},
        general_tile_shapes={"A": ((8, 8),)},
    ),
)

result = solve(problem)
print_report(result)
```

Words are shown from the least-significant physical element-address bit to the most-significant bit. For modes `(i, j)`, the canonical word `jjjiii` is an 8x8 row-inner tile: `j0,j1,j2` occupy physical bits 0-2, followed by `i0,i1,i2`.

## Multiple arrays

Each target array is synthesized independently first. The solver retains a small family per array and then builds bounded joint configurations from those families.

Context arrays use fixed row-major layouts. Their events still contribute to report-only and joint objective features.

This first implementation treats aligned regions from separate allocations as separate objects, so most joint region scores are additive. It does not yet model synchronized base residues, aliases, combined compiler register pressure, or cache interference.

## Event representation

`Access` describes one logical element touched by one lane or thread.

`MemoryEvent` describes one dynamic memory-instruction event. It carries:

- a stable id and static site;
- logical accesses;
- lane ids;
- read/write kind;
- a local execution group;
- local order;
- compressed multiplicity; and
- arbitrary string metadata.

`EventSequence` stores a local program-order sequence of event ids. It is used for temporal windows. The library never invents a global order among independent waves.

For regular events, `lane_event(...)`, `lane_accesses(...)`, and `sequence(...)` are shorter convenience constructors. `MemoryEvent.make(...)` remains the more general form and permits accesses to several arrays in one declared event.

## Built-in objective builders

### `SimultaneousRegions`

Each event becomes one hyperedge. With `lane_group=16`, each aligned contiguous 16-lane subset becomes a separate edge.

```python
SimultaneousRegions(
    "lane16_64",
    64,
    lane_group=16,
    event_filter=EventFilter.make(arrays=("A",), kinds=("read",)),
)
```

### `LanePrefixRegions`

Convenience wrapper that creates several contiguous-lane objectives.

```python
LanePrefixRegions(
    "lane",
    levels=((4, 16), (8, 32), (16, 64), (32, 128), (64, 256)),
)
```

Each tuple is `(lane_group_size, aligned_region_bytes)`.

### `TemporalWindowRegions`

Unions accesses from sliding windows in explicit local sequences.

```python
TemporalWindowRegions(
    "five_load_window128",
    128,
    window=5,
    event_filter=EventFilter.make(arrays=("input",), kinds=("read",)),
)
```

Use `window=None` for the complete sequence.

### `PerLaneTemporalRegions`

Builds one component per requested temporal-window length. Within each explicit
sequence, accesses are filtered and ordered independently for every array and
lane, so interleaved accesses to other arrays do not consume window positions.

```python
PerLaneTemporalRegions(
    "lane_window128",
    region_bytes=128,
    windows=(2, 4, 8, 16, 32),
    event_filter=EventFilter.make(arrays=("A", "B"), kinds=("read",)),
)
```

This creates components named `lane_window128.window2` through
`lane_window128.window32`.

### `GroupedRegions`

Unions events that share fields such as `group`, `site`, `order`, or metadata keys.

```python
GroupedRegions(
    "workgroup_panel1024",
    1024,
    group_by=("workgroup", "step"),
)
```

### `ExplicitRegions`

Direct escape hatch for custom hyperedges.

```python
ExplicitRegions(
    "custom64",
    64,
    edges_by_array={
        "A": (
            Hyperedge.make(((0, 0), (1, 3), (7, 2)), weight=10),
        )
    },
)
```

A new objective builder only needs a `build(matrices, events, sequences)` method that returns one or more `ObjectiveComponent` objects.

## Layout grammars

### Canonical words

For each declared tile shape, the exact grid dynamic program searches all low-to-high interleavings of mode bits while preserving low-to-high order inside each mode.

The exact optimum for a lexicographic or weighted policy is preserved even when only a bounded number of alternate histories are kept per state. A capped Pareto frontier is reported as capped.

### Arbitrary bit-linear inner maps

For selected tile shapes, the solver:

1. fragments each edge by outer tile;
2. computes the span of all within-fragment XOR differences;
3. projects into that active span;
4. enumerates all subspaces rank by rank when the active rank is small;
5. runs exact cover-edge dynamic programming; and
6. lifts the chosen flag back into an invertible inner matrix `A_in`.

The default exact active-rank limit is seven. This implementation intentionally returns no approximate general-linear answer above that limit rather than silently switching to a beam.

When the main policy is Pareto, the general-linear branch defaults to the same objective order interpreted lexicographically. Exact Pareto path enumeration on the full subspace lattice grows quickly; set `SolverConfig.general_policy` explicitly to experiment with another policy.

## Search configuration

Important knobs are in `SolverConfig`:

- `tile_shapes`: canonical tile shapes per target array;
- `general_tile_shapes`: tile shapes for arbitrary `A_in` search;
- `outer_orders`: legal canonical outer mode orders;
- `canonical_candidates_per_tile`;
- `general_exact_rank`;
- `primary_tolerance`;
- `per_array_candidates`;
- `joint_beam_width`; and
- `joint_candidates`.

`ScorePolicy` supports:

- `kind="lexicographic"`;
- `kind="weighted"`; and
- `kind="pareto"`.

The names in `ScorePolicy.order` may include objective names plus terminal features such as:

- `runs`;
- `xors`;
- `adj_gap`;
- `adj_breaks`; and
- `max_adj_gap`.

Region objectives guide the dynamic program. Lane-gap and concrete-address features are evaluated after a layout is realized and are used to rank or retain the resulting family.

## Reading the report

For each array, the terminal report shows:

- tile hypotheses considered;
- candidates realized and retained;
- layout grammar;
- canonical word or low physical-bit expressions;
- every requested objective score;
- packing bounds and packing ratios;
- run and XOR proxies;
- exact/capped status;
- grid or subspace search state counts; and
- a small inner-tile offset diagram.

The final table shows joint choices across arrays.

Use `dump_json(result, "result.json")` for machine-readable output.

## Known simplifications

- Array extents must be powers of two.
- Targeted accesses must be scalar and exactly one logical element wide.
- Outer tile order is canonical mixed-radix order; padding and fringes are not implemented.
- The general-linear branch is exact only up to the configured active-rank limit.
- Symbolic code-generation cost is approximate.
- Separate array allocations do not share one physical-region identifier.
- Joint cache interference, base-address phase, aliases, conversion cost, compiler features, and replay measurement are not modeled.
- The model chooses candidates to inspect; it does not claim to predict GPU runtime exactly.
