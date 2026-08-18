"""Score realized layouts against RELAY memory-region objectives.

The search code reasons about low-address quotient subspaces.  Once a layout
is concrete, the same quantity is simpler to compute directly: map every
logical point in a hyperedge to an element offset, divide by the aligned
region capacity, and count distinct region identifiers.

All scores in this module are costs: lower values are better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from .layouts import Layout
from .model import Coord, MatrixSpec
from .objectives import Hyperedge, ObjectiveComponent, build_objectives

if TYPE_CHECKING:
    from .solver import RelayProblem


ScoreMode = Literal[
    "weighted-region-count",
    "peak-normalized-excess",
    "weighted-normalized-excess",
]

SCORE_MODES: tuple[ScoreMode, ...] = (
    "weighted-region-count",
    "peak-normalized-excess",
    "weighted-normalized-excess",
)


@dataclass(frozen=True)
class ArrayComponentScore:
    """One array's ``Q`` and ``LB`` contribution to an objective component.

    ``raw_region_count`` already includes each hyperedge's multiplicity
    weight.  "Raw" distinguishes ``Q`` from lower-bound normalization and the
    separate component weight ``tau``.
    """

    array: str
    raw_region_count: float
    packing_lower_bound: float

    @property
    def normalized_excess(self) -> float:
        return normalized_excess(self.raw_region_count, self.packing_lower_bound)


@dataclass(frozen=True)
class ComponentScore:
    """The quotient-region score for one access scope and byte scale.

    ``raw_region_count`` is the notes' edge-weighted ``Q``.  ``weight`` is the
    independently configured component weight ``tau``.
    """

    name: str
    region_bytes: int
    weight: float
    raw_region_count: float
    packing_lower_bound: float
    normalized_excess: float
    arrays: tuple[ArrayComponentScore, ...]

    @property
    def weighted_region_count(self) -> float:
        return self.weight * self.raw_region_count

    @property
    def weighted_normalized_excess(self) -> float:
        return self.weight * self.normalized_excess


@dataclass(frozen=True)
class LayoutScore:
    """Detailed component scores and the supported scalar aggregates."""

    components: tuple[ComponentScore, ...]
    weighted_region_count: float
    peak_normalized_excess: float
    weighted_normalized_excess: float

    def value(self, mode: ScoreMode) -> float:
        """Return one explicitly named scalar score; lower is better."""

        if mode == "weighted-region-count":
            return self.weighted_region_count
        if mode == "peak-normalized-excess":
            return self.peak_normalized_excess
        if mode == "weighted-normalized-excess":
            return self.weighted_normalized_excess
        raise ValueError(
            f"unknown score mode {mode!r}; expected one of {', '.join(SCORE_MODES)}"
        )

    def component(self, name: str) -> ComponentScore:
        """Look up a component score by its declared objective name."""

        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)


def quotient_region_count(
    matrix: MatrixSpec,
    layout: Layout,
    edge: Hyperedge,
    region_bytes: int,
) -> int:
    """Return ``q(E; V_d)`` for one hyperedge and a realized layout.

    ``region_bytes / element_bytes`` is the number of elements in an aligned
    region.  Layout offsets are element offsets, so their integer quotients
    are exactly the physical region identifiers from the project notes.
    """

    if region_bytes % matrix.element_bytes:
        raise ValueError(
            f"{region_bytes} B is not divisible by {matrix.name}'s "
            f"{matrix.element_bytes} B element width"
        )
    capacity = region_bytes // matrix.element_bytes
    if capacity <= 0 or capacity & (capacity - 1):
        raise ValueError(
            f"region capacity must be a positive power of two in elements; got {capacity}"
        )
    return len({layout.offset(matrix, point) // capacity for point in edge.points})


def weighted_component_region_count(
    matrix: MatrixSpec,
    layout: Layout,
    component: ObjectiveComponent,
    *,
    offset_cache: dict[Coord, int] | None = None,
) -> float:
    """Compute ``Q`` for one matrix within an objective component.

    ``offset_cache`` is useful when the same logical points occur in many
    overlapping hyperedges.  The cache is layout-specific and callers must
    not reuse it with a different layout.
    """

    if offset_cache is None:
        return sum(
            edge.weight
            * quotient_region_count(matrix, layout, edge, component.region_bytes)
            for edge in component.edges_by_array.get(matrix.name, ())
        )

    capacity = component.capacity_elements(matrix)
    if capacity <= 0 or capacity & (capacity - 1):
        raise ValueError(
            f"region capacity must be a positive power of two in elements; got {capacity}"
        )

    total = 0.0
    for edge in component.edges_by_array.get(matrix.name, ()):
        regions: set[int] = set()
        for point in edge.points:
            offset = offset_cache.get(point)
            if offset is None:
                offset = layout.offset(matrix, point)
                offset_cache[point] = offset
            regions.add(offset // capacity)
        total += edge.weight * len(regions)
    return total


def normalized_excess(raw_region_count: float, packing_lower_bound: float) -> float:
    """Return relative excess over the capacity-only packing lower bound."""

    return (raw_region_count - packing_lower_bound) / max(packing_lower_bound, 1.0)


def score_layouts(
    matrices: Mapping[str, MatrixSpec],
    components: Sequence[ObjectiveComponent],
    layouts: Mapping[str, Layout],
    *,
    component_weights: Mapping[str, float] | None = None,
) -> LayoutScore:
    """Score one layout for every participating array.

    Arrays are treated as separate allocations, matching the solver: region
    counts and lower bounds are summed across arrays before normalized excess
    is computed.  Component weights are the notes' ``tau[s,d]`` values.
    Unspecified components have weight 1; weight 0 disables a component from
    all three scalar aggregates while retaining it in the detailed report.
    """

    weights = dict(component_weights or {})
    component_names = {component.name for component in components}
    unknown_weights = sorted(set(weights) - component_names)
    if unknown_weights:
        raise ValueError(
            "weights were supplied for unknown objective components: "
            + ", ".join(unknown_weights)
        )
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("component weights must be nonnegative")

    results: list[ComponentScore] = []
    offset_caches: dict[str, dict[Coord, int]] = {
        name: {} for name in layouts
    }
    for component in components:
        per_array: list[ArrayComponentScore] = []
        for array_name, edges in component.edges_by_array.items():
            if not edges:
                continue
            if array_name not in matrices:
                raise ValueError(
                    f"objective {component.name}: unknown array {array_name!r}"
                )
            if array_name not in layouts:
                raise ValueError(
                    f"objective {component.name}: no layout supplied for array {array_name!r}"
                )
            matrix = matrices[array_name]
            layout = layouts[array_name]
            if layout.matrix_name != array_name:
                raise ValueError(
                    f"layout {layout.name!r} targets {layout.matrix_name!r}, "
                    f"not {array_name!r}"
                )
            raw = weighted_component_region_count(
                matrix,
                layout,
                component,
                offset_cache=offset_caches[array_name],
            )
            bound = component.packing_bound(matrix)
            per_array.append(ArrayComponentScore(array_name, raw, bound))

        raw_total = sum(item.raw_region_count for item in per_array)
        bound_total = sum(item.packing_lower_bound for item in per_array)
        results.append(
            ComponentScore(
                name=component.name,
                region_bytes=component.region_bytes,
                weight=float(weights.get(component.name, 1.0)),
                raw_region_count=raw_total,
                packing_lower_bound=bound_total,
                normalized_excess=normalized_excess(raw_total, bound_total),
                arrays=tuple(per_array),
            )
        )

    active = [component for component in results if component.weight > 0]
    return LayoutScore(
        components=tuple(results),
        weighted_region_count=sum(
            component.weighted_region_count for component in active
        ),
        peak_normalized_excess=max(
            (component.normalized_excess for component in active), default=0.0
        ),
        weighted_normalized_excess=sum(
            component.weighted_normalized_excess for component in active
        ),
    )


def score_problem(
    problem: "RelayProblem",
    layouts: Mapping[str, Layout],
    *,
    component_weights: Mapping[str, float] | None = None,
) -> LayoutScore:
    """Build a :class:`RelayProblem`'s objectives and score given layouts."""

    matrices = {matrix.name: matrix for matrix in problem.matrices}
    events = {event.id: event for event in problem.events}
    components = build_objectives(
        problem.objectives, matrices, events, problem.sequences
    )
    return score_layouts(
        matrices,
        components,
        layouts,
        component_weights=component_weights,
    )


def score_to_dict(score: LayoutScore) -> dict[str, object]:
    """Convert a score to a stable JSON-compatible representation."""

    return {
        "aggregates": {
            "weighted_region_count": score.weighted_region_count,
            "peak_normalized_excess": score.peak_normalized_excess,
            "weighted_normalized_excess": score.weighted_normalized_excess,
        },
        "components": [
            {
                "name": component.name,
                "region_bytes": component.region_bytes,
                "weight": component.weight,
                "raw_region_count": component.raw_region_count,
                "packing_lower_bound": component.packing_lower_bound,
                "normalized_excess": component.normalized_excess,
                "weighted_region_count": component.weighted_region_count,
                "weighted_normalized_excess": (
                    component.weighted_normalized_excess
                ),
                "arrays": [
                    {
                        "name": array.array,
                        "raw_region_count": array.raw_region_count,
                        "packing_lower_bound": array.packing_lower_bound,
                        "normalized_excess": array.normalized_excess,
                    }
                    for array in component.arrays
                ],
            }
            for component in score.components
        ],
    }
