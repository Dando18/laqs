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
supports three explicitly named scalar costs:

- `weighted-region-count`;
- `peak-normalized-excess`; and
- `weighted-normalized-excess`.

For example, score a globally row-major layout for every GEMM operand:

```bash
.venv/bin/python bin/score_layout.py kernels/gemm/problem.py \
  --layout all=row-major \
  --score-mode weighted-normalized-excess
```

Use `--json` for a machine-readable report, `--component-weight NAME=VALUE`
to set an objective's weight, and `--problem-option problem_size=512` to pass a
JSON-valued option into the problem's `build_config` routine. See
[Scoring realized layouts](docs/scoring.md) for the formulas, API, complete
layout-word convention, CLI contract, and JSON fields.

## Compare GEMM score and runtime ranks

The GEMM experiment scores eight conventional global and tiled layouts, runs
the existing HIP benchmark for each one, and compares ascending score rank to
ascending median-runtime rank. It reports per-layout rank deltas and Spearman's
rank correlation, with complete score and timing detail available as JSON.

Validate scoring without a GPU:

```bash
.venv/bin/python experiments/gemm_layout_ranking.py \
  --n 256 --score-only \
  --output results/gemm-layout-ranking-score-only.json
```

Run timings from an MI300A allocation:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/gemm_layout_ranking.py \
  --n 1024 --arch gfx942 \
  --output results/gemm-layout-ranking-1024.json
```

See [GEMM layout score/runtime experiment](docs/gemm-layout-experiment.md) for
the exact layout set, ranking rules, timing boundary, options, output fields,
and interpretation limits.

A checked-in `N=256` MI300A run is available at
[`results/gemm_layout_ranking_256.json`](results/gemm_layout_ranking_256.json).
For this sample, six of eight positions matched exactly; the 8x8 and 16x16
row-inner layouts exchanged ranks, giving Spearman `rho = 0.976190`.

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
