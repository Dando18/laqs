# Scoring realized layouts

RELAY uses the same locality model for search and for evaluating a layout that
is already known.  The public implementation is in `relay/scoring.py`; the
command-line wrapper is `bin/score_layout.py`.

Every score is a cost. **Lower is better.**

## Component score

An objective component represents one access scope at one aligned byte scale.
For example, `wave_load.64B` may contain one hyperedge per wave-wide load and
use 64-byte regions. For hyperedge `E`, realized layout `L`, and a region that
holds `c` elements, RELAY computes

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

Each objective component has a nonnegative component weight `tau`.  The
default is 1.  A weight of 0 retains the component in the detailed report but
removes it from every scalar aggregate.

| CLI mode | Definition | Interpretation |
| --- | --- | --- |
| `weighted-region-count` | `sum(tau * Q)` | Raw multiscale region cost. |
| `peak-normalized-excess` | `max(e)` over components with `tau > 0` | Worst relative miss over the packing bound. The magnitude of `tau` does not scale this maximum. |
| `weighted-normalized-excess` | `sum(tau * e)` | Weighted area under the normalized-excess components (`J_area` in the notes). |

The CLI always reports every component and all three aggregates.  `--score-mode`
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
`J_peak`, `J_area`, or any scalar score mode.

## Pareto frontiers

`pareto_frontier` compares several named `LayoutScore` objects over any ordered
set of numeric score extractors. Every objective is minimized. Point `a`
dominates point `b` exactly when `a` is no greater in every objective and is
strictly smaller in at least one. Exact objective ties are retained as distinct
frontier members.

With no explicit extractors, the function compares all three public aggregate
score modes followed by `codegen-runs` and `codegen-xors`. A caller can instead
construct the multi-objective vector from the notes, including a particular
fine-scale component and the codegen proxies:

```python
from relay import pareto_frontier

frontier = pareto_frontier(
    {
        "row_major": row_score,
        "tile8_row_major": tile8_score,
        "column_major": column_score,
    },
    objectives={
        "wave_load.64B.raw-region-count": (
            lambda score: score.component("wave_load.64B").raw_region_count
        ),
        "peak-normalized-excess": (
            lambda score: score.peak_normalized_excess
        ),
        "weighted-normalized-excess": (
            lambda score: score.weighted_normalized_excess
        ),
        "codegen-runs": lambda score: score.codegen.runs,
        "codegen-xors": lambda score: score.codegen.xors,
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
from relay import canonical_layout_from_word, score_layouts
from relay.objectives import build_objectives

matrices = {matrix.name: matrix for matrix in problem.matrices}
events = {event.id: event for event in problem.events}
components = build_objectives(
    problem.objectives, matrices, events, problem.sequences
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
    component_weights={"wave_load.64B": 2.0},
)

print(score.component("wave_load.64B").normalized_excess)
print(score.value("weighted-normalized-excess"))
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
get_objectives(config)
get_component_weights(config)
```

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
  --score-mode peak-normalized-excess \
  --component-weight wave_load.64B=2 \
  --problem-option problem_size=256 \
  --json
```

`get_component_weights(config)` returns the problem's complete mapping from
objective-component name to the notes' `tau` weight. The CLI applies those
defaults; repeated `--component-weight NAME=VALUE` options override individual
entries. A zero override keeps the component detail but excludes it from all
aggregates.

`--layout all=...` supplies a default; an array-specific assignment overrides
it. Non-target context arrays default to row-major when they have no explicit
assignment. Layout specs are `row-major`, `column-major`, a canonical word, or
the explicit spelling `word:jjjiii`. `--problem-option NAME=JSON_VALUE` passes
configuration into `build_config`; numbers, strings, booleans, and lists must
therefore use JSON syntax.

The JSON report includes the selected score, all aggregate scores, per-array
and total codegen costs, every component's `Q`, `LB`, normalized excess and
weight, plus per-array component contributions.
