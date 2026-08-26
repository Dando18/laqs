# Scoring realized layouts

RELAY uses the same locality model for search and for evaluating a layout that
is already known.  The public implementation is in `relay/scoring.py`; the
command-line wrapper is `bin/score_layout.py`.

Every score is a cost. **Lower is better.**

## Component score

An objective component represents one universal access scope at one aligned
byte scale. For example, `issue.g64.stream.load.64B` contains the compressed
hyperedges for 64-lane stream-local loads evaluated with 64-byte regions. For
hyperedge `E`, realized layout `L`, and a region that holds `c` elements,
RELAY computes

```text
q(E, L, c) = number of distinct values of floor(L(x) / c), for x in E.
```

This is the quotient count `|E / V_d|` from the project notes, evaluated
directly from physical element offsets.  A component's raw region count is

```text
Q = sum(edge.weight * q(edge, layout, capacity)).
```

The capacity-only packing lower bound is

```text
LB = sum(edge.weight * ceil(number_of_distinct_points / capacity)).
```

The normalized excess over that lower bound is

```text
e = (Q - LB) / max(LB, 1).
```

An excess of 0 reaches the packing bound.  An excess of 0.25 means 25% more
aligned regions than that bound.  The bound need not be attainable for every
access pattern.

`region_bytes` is converted to elements using each matrix's `element_bytes`.
The current model requires a positive power-of-two number of elements per
region and assumes that each allocation is aligned to those regions.  Arrays
are separate allocations: their `Q` and `LB` contributions are summed, but a
region from one array is never treated as a region from another array.

## Scalar score modes

Each objective component has a nonnegative component weight `tau`. A direct
scorer call without a hardware profile defaults unspecified components to 1.
A hardware profile instead supplies a complete response over the realized
universal cells, assigning 0 to unsupported cells. A weight of 0 retains the
component in the detailed report but removes it from weighted aggregates.

| CLI mode | Definition | Interpretation |
| --- | --- | --- |
| `weighted-region-count` | `sum(tau * Q)` | Raw multiscale region cost. |
| `peak-normalized-excess` | `max(e)` over components with `tau > 0` | Worst relative miss over the packing bound. The magnitude of `tau` does not scale this maximum. |
| `weighted-normalized-excess` | `sum(tau * e)` | Legacy weighted area under normalized-excess components. |
| `hardware-peak` | `max(e / kappa)` over profile-supported peak cells | Worst excess relative to the hardware profile's acceptable tolerance. |
| `hardware-area` | `sum(tau * b * (Q - LB) / B_K)` | Exposure-preserving `J_area` under the hardware response. |
| `hardware-place` | weighted normalized excess same-color pairs | Cross-allocation resource placement cost `J_place`. |

The CLI always reports every component and all six aggregates. `--score-mode`
only chooses the scalar printed as `Selected score` and stored as
`selected_score` in JSON.  This keeps the underlying multi-objective vector
visible rather than hiding it behind one unexplained label.

## Code-generation costs

Concrete scores also report two integer address-code proxies for every target
array and their totals:

- `runs` is the number of contiguous source-mode runs in a canonical bit
  selection. For example, `jjjjiiii` has 2 runs and `jijijiji` has 8.
- `xors` is the number of XOR operations in a general linear inner-layout
  expression. Canonical layouts have 0 XORs.

These costs reuse the layout grammar's code-generation definitions. Non-target
context arrays are omitted because their layout is fixed rather than selected.
Runs and XORs remain separate: RELAY does not assume a conversion factor
between mask/shift work and XOR work. They are not folded into `Q`, `e`,
`J_peak`, `J_area`, `J_place`, or any scalar score mode, and they are
annotations rather than default dominance coordinates.

## Colored quotient placement

A `ResourceMap` deduplicates aligned transactions within each hardware cohort,
maps each transaction address to a small color through sparse GF(2) parity
masks, and counts same-color transaction pairs. `J_place` subtracts the pair
count of an optimally balanced occupancy and normalizes by the range from
balanced to fully contended placement. The `cohort` partition retains array
identity, so transactions from different allocations can contend.

The MI300A proof-of-concept uses `simd_window.t4.cohort.load`, 64-byte
transactions, and an eight-color 4 KiB HBM-stack interleave sketch. Its robust
phase policy maximizes contention over relative allocation colors because raw
timing records do not expose physical allocation bases.

## Pareto frontiers

`pareto_frontier` compares several named `LayoutScore` objects over any ordered
set of numeric score extractors. Every objective is minimized. Point `a`
dominates point `b` exactly when `a` is no greater in every objective and is
strictly smaller in at least one. Exact objective ties are retained as distinct
frontier members.

With no explicit extractors, the function compares all six public aggregate
score modes. A caller can instead construct the multi-objective vector from
the notes, including a particular fine-scale component:

```python
from relay import pareto_frontier

frontier = pareto_frontier(
    {
        "row_major": row_score,
        "tile8_row_major": tile8_score,
        "column_major": column_score,
    },
    objectives={
        "issue.g64.stream.load.64B.raw-region-count": (
            lambda score: score.component(
                "issue.g64.stream.load.64B"
            ).raw_region_count
        ),
        "hardware-peak": lambda score: score.hardware_peak,
        "hardware-area": lambda score: score.hardware_area,
        "hardware-place": lambda score: score.hardware_place,
    },
)

print(frontier.objectives)
for point in frontier.points:
    print(point.name, point.values)
```

The result is deterministic: frontier points are ordered by their objective
tuple and then by name. The routine returns the exact non-dominated set; it
does not apply epsilon tolerances or select one preferred member.

## Library API

```python
from relay import (
    MI300A_V1,
    UniversalScopeObjectives,
    build_resource_cohorts,
    canonical_layout_from_word,
    score_layouts,
)
from relay.objectives import build_objectives

matrices = {matrix.name: matrix for matrix in problem.matrices}
events = {event.id: event for event in problem.events}
components = build_objectives(
    (UniversalScopeObjectives(MI300A_V1.byte_scales),),
    matrices,
    events,
    problem.sequences,
)
resource_cohorts = build_resource_cohorts(
    matrices,
    events,
    problem.sequences,
    (resource_map.cohort_family for resource_map in MI300A_V1.resource_maps),
)

layouts = {
    "A": canonical_layout_from_word(matrices["A"], "jjjiii"),
    "B": canonical_layout_from_word(matrices["B"], "jjjiii"),
    "C": canonical_layout_from_word(matrices["C"], "jjjiii"),
}
score = score_layouts(
    matrices,
    components,
    layouts,
    hardware_profile=MI300A_V1,
    resource_cohorts=resource_cohorts,
)

print(score.component(MI300A_V1.fine_component).normalized_excess)
print(score.value("hardware-area"))
print(score.value("hardware-place"))
```

The main public routines are:

- `quotient_region_count`: unweighted `q` for one hyperedge;
- `weighted_component_region_count`: one array's `Q` contribution;
- `normalized_excess`: computes `e` from `Q` and `LB`;
- `layout_codegen_cost`: computes per-target-array and total run/XOR proxies;
- `score_layouts`: scores materialized objective components without rebuilding
  them, which is best when comparing many layouts;
- `score_problem`: convenience wrapper that builds a `RelayProblem`'s
  components and scores once;
- `pareto_frontier`: returns exact non-dominated named scores for default or
  caller-defined objective extractors; and
- `score_to_dict`: stable JSON-compatible detail for experiments and tools.

## Canonical layout words

`canonical_layout_from_word(matrix, word)` is the shared parser.  A word uses
the matrix's single-character mode names and lists physical element-address
bits from least significant to most significant.

For modes `(i, j)`:

- `jjjiii` is an 8x8 row-major inner tile;
- `iiijjj` is an 8x8 column-major inner tile;
- a complete `j...ji...i` word is globally row-major; and
- a complete `i...ij...j` word is globally column-major.

The symbol counts define the inner tile.  Bits omitted from the word select
outer tiles, whose default ordering is row-major.  Consequently, the word
length may be smaller than the matrix's total coordinate-bit count.

## Score CLI

Problem files use the small Python protocol already used by
`kernels/gemm/problem.py`:

```python
build_config(**options)
get_matrices(config)
get_events_and_sequences(config)
```

The command constructs `UniversalScopeObjectives` from the selected hardware
profile's byte-scale ladder. Kernel modules provide access and schedule facts,
not objective names or hardware weights.

Score global row-major GEMM layouts:

```bash
.venv/bin/python bin/score_layout.py kernels/gemm/problem.py \
  --layout A=row-major \
  --layout B=row-major \
  --layout C=row-major
```

Score one common 8x8 tiled layout and emit JSON:

```bash
.venv/bin/python bin/score_layout.py kernels/gemm/problem.py \
  --layout all=jjjiii \
  --score-mode hardware-area \
  --hardware-profile mi300a \
  --component-weight issue.g64.stream.load.64B=2 \
  --problem-option problem_size=256 \
  --json
```

`--hardware-profile` selects the global byte-scale ladder, `tau` response,
peak tolerances, and resource maps. Repeated `--component-weight NAME=VALUE` options override
individual profile `tau` entries for exploration. A zero override keeps the
component detail but excludes it from weighted aggregates.

`--layout all=...` supplies a default; an array-specific assignment overrides
it. Non-target context arrays default to row-major when they have no explicit
assignment. Layout specs are `row-major`, `column-major`, a canonical word, or
the explicit spelling `word:jjjiii`. `--problem-option NAME=JSON_VALUE` passes
configuration into `build_config`; numbers, strings, booleans, and lists must
therefore use JSON syntax.

The JSON report includes the selected hardware profile, selected score, all
aggregate scores, per-array and total codegen costs, every component's `Q`,
`LB`, normalized excess and weight, plus per-array component contributions.
