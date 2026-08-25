"""Score realized layouts against RELAY memory-region objectives.

The search code reasons about low-address quotient subspaces.  Once a layout
is concrete, the same quantity is simpler to compute directly: map every
logical point in a hyperedge to an element offset, divide by the aligned
region capacity, and count distinct region identifiers.

All scores in this module are costs: lower values are better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
)

from .layouts import Layout
from .model import Coord, MatrixSpec
from .objectives import Hyperedge, ObjectiveComponent, build_objectives

if TYPE_CHECKING:
    from .hardware import HardwareProfile
    from .solver import RelayProblem


ScoreMode = Literal[
    "weighted-region-count",
    "peak-normalized-excess",
    "weighted-normalized-excess",
    "hardware-peak",
    "hardware-area",
]

SCORE_MODES: tuple[ScoreMode, ...] = (
    "weighted-region-count",
    "peak-normalized-excess",
    "weighted-normalized-excess",
    "hardware-peak",
    "hardware-area",
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
    normalization_bytes: float | None = None
    excess_footprint: float = 0.0
    peak_tolerance: float | None = None

    @property
    def weighted_region_count(self) -> float:
        return self.weight * self.raw_region_count

    @property
    def weighted_normalized_excess(self) -> float:
        return self.weight * self.normalized_excess

    @property
    def weighted_excess_footprint(self) -> float:
        """Return ``tau[s,b] * x[s,b]`` for the hardware area score."""

        return self.weight * self.excess_footprint

    @property
    def peak_excess_ratio(self) -> float | None:
        """Return ``e[s,b] / kappa[s,b]`` when this peak cell is active."""

        if self.peak_tolerance is None:
            return None
        return self.normalized_excess / self.peak_tolerance


@dataclass(frozen=True)
class ArrayCodegenCost:
    """Address-code cost proxies for one target array's realized layout."""

    array: str
    grammar: str
    runs: int
    xors: int


@dataclass(frozen=True)
class CodegenCost:
    """Per-array and total address-code cost proxies.

    A run is one contiguous source-mode run in a canonical bit-selection
    expression, and an XOR is one XOR operation in a linear expression.  The
    measures deliberately remain separate because their machine costs are not
    assumed to be interchangeable.
    """

    arrays: tuple[ArrayCodegenCost, ...]

    @property
    def runs(self) -> int:
        return sum(array.runs for array in self.arrays)

    @property
    def xors(self) -> int:
        return sum(array.xors for array in self.arrays)


@dataclass(frozen=True)
class LayoutScore:
    """Detailed locality scores, codegen costs, and scalar aggregates."""

    components: tuple[ComponentScore, ...]
    codegen: CodegenCost
    weighted_region_count: float
    peak_normalized_excess: float
    weighted_normalized_excess: float
    hardware_peak: float = 0.0
    hardware_area: float = 0.0

    def value(self, mode: ScoreMode) -> float:
        """Return one explicitly named scalar score; lower is better."""

        if mode == "weighted-region-count":
            return self.weighted_region_count
        if mode == "peak-normalized-excess":
            return self.peak_normalized_excess
        if mode == "weighted-normalized-excess":
            return self.weighted_normalized_excess
        if mode == "hardware-peak":
            return self.hardware_peak
        if mode == "hardware-area":
            return self.hardware_area
        raise ValueError(
            f"unknown score mode {mode!r}; expected one of {', '.join(SCORE_MODES)}"
        )

    def component(self, name: str) -> ComponentScore:
        """Look up a component score by its declared objective name."""

        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)


@dataclass(frozen=True)
class ParetoPoint:
    """One non-dominated named score and its ordered objective values."""

    name: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class ParetoFrontier:
    """A deterministic set of points that are not strictly dominated."""

    objectives: tuple[str, ...]
    points: tuple[ParetoPoint, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Return frontier member names in objective-value order."""

        return tuple(point.name for point in self.points)


ScoreExtractor = Callable[[LayoutScore], float]


def pareto_frontier(
    scores: Mapping[str, LayoutScore],
    *,
    objectives: Mapping[str, ScoreExtractor] | None = None,
) -> ParetoFrontier:
    """Return the exact Pareto frontier for named layout scores.

    Every objective is minimized. A score is dominated when another score is
    no greater in every objective and strictly smaller in at least one.
    Objective order follows the supplied mapping; when omitted, all public
    aggregate score modes followed by codegen runs and XORs are used. Exact
    ties remain as distinct frontier members.
    """

    if objectives is None:
        objective_items: tuple[tuple[str, ScoreExtractor], ...] = (
            *tuple(
                (
                    mode,
                    lambda score, selected_mode=mode: score.value(selected_mode),
                )
                for mode in SCORE_MODES
            ),
            ("codegen-runs", lambda score: float(score.codegen.runs)),
            ("codegen-xors", lambda score: float(score.codegen.xors)),
        )
    else:
        objective_items = tuple(objectives.items())
    if not objective_items:
        raise ValueError("a Pareto frontier requires at least one objective")

    candidates = tuple(
        ParetoPoint(
            name,
            tuple(float(extractor(score)) for _, extractor in objective_items),
        )
        for name, score in scores.items()
    )

    def dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
        return all(
            left_value <= right_value
            for left_value, right_value in zip(left.values, right.values)
        ) and any(
            left_value < right_value
            for left_value, right_value in zip(left.values, right.values)
        )

    points = tuple(
        sorted(
            (
                candidate
                for candidate in candidates
                if not any(
                    other is not candidate and dominates(other, candidate)
                    for other in candidates
                )
            ),
            key=lambda point: (point.values, point.name),
        )
    )
    return ParetoFrontier(
        tuple(name for name, _ in objective_items),
        points,
    )


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


def excess_footprint(
    raw_region_count: float,
    packing_lower_bound: float,
    region_bytes: int,
    normalization_bytes: float | None,
) -> float:
    """Return the kernel-exposure-preserving feature ``x[s,b]``.

    Universal objectives provide the common kernel denominator ``B_K``.  The
    normalized-excess fallback keeps legacy hand-built objective components
    usable while they are migrated.
    """

    if normalization_bytes is None:
        return normalized_excess(raw_region_count, packing_lower_bound)
    if normalization_bytes <= 0:
        raise ValueError("normalization_bytes must be positive")
    return (
        region_bytes
        * (raw_region_count - packing_lower_bound)
        / normalization_bytes
    )


def layout_codegen_cost(
    matrices: Mapping[str, MatrixSpec],
    layouts: Mapping[str, Layout],
) -> CodegenCost:
    """Compute address-code proxies for every target matrix layout.

    Non-target matrices describe fixed context data and are excluded, matching
    the solver and generated kernel interfaces.  Costs are summed across
    target arrays because every generated address expression executes in the
    kernel, while the per-array detail remains available for inspection.
    """

    arrays: list[ArrayCodegenCost] = []
    for array_name, matrix in matrices.items():
        if not matrix.target:
            continue
        if array_name not in layouts:
            raise ValueError(f"no layout supplied for target array {array_name!r}")
        layout = layouts[array_name]
        if layout.matrix_name != array_name:
            raise ValueError(
                f"layout {layout.name!r} targets {layout.matrix_name!r}, "
                f"not {array_name!r}"
            )
        arrays.append(
            ArrayCodegenCost(
                array=array_name,
                grammar=layout.grammar,
                runs=layout.runs,
                xors=layout.xor_count,
            )
        )
    return CodegenCost(tuple(arrays))


def score_layouts(
    matrices: Mapping[str, MatrixSpec],
    components: Sequence[ObjectiveComponent],
    layouts: Mapping[str, Layout],
    *,
    component_weights: Mapping[str, float] | None = None,
    peak_tolerances: Mapping[str, float] | None = None,
    hardware_profile: "HardwareProfile | None" = None,
    offset_cache_by_array: Mapping[str, dict[Coord, int]] | None = None,
    array_component_cache: MutableMapping[
        tuple[str, tuple[object, ...], str], tuple[float, float]
    ]
    | None = None,
) -> LayoutScore:
    """Score one layout for every participating array.

    Arrays are treated as separate allocations, matching the solver: region
    counts and lower bounds are summed across arrays before normalized excess
    is computed.  Component weights are the notes' ``tau[s,d]`` values.
    Unspecified components have weight 1; weight 0 disables a component from
    all three scalar aggregates while retaining it in the detailed report.
    Address-code runs and XORs are computed independently of those aggregates.
    ``offset_cache_by_array`` lets callers reuse realized offsets in additional
    diagnostics for the same layouts. ``array_component_cache`` avoids
    rescoring the same concrete array layout when many joint candidates reuse
    it; cached values retain both the raw region count and packing bound.
    """

    if hardware_profile is not None and (
        component_weights is not None or peak_tolerances is not None
    ):
        raise ValueError(
            "hardware_profile cannot be combined with explicit weights or tolerances"
        )
    if hardware_profile is not None:
        weights = hardware_profile.component_weights(components)
        tolerances = hardware_profile.peak_tolerances(components)
    else:
        weights = dict(component_weights or {})
        tolerances = dict(peak_tolerances or {})
    component_names = {component.name for component in components}
    unknown_weights = sorted(set(weights) - component_names)
    if unknown_weights:
        raise ValueError(
            "weights were supplied for unknown objective components: "
            + ", ".join(unknown_weights)
        )
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("component weights must be nonnegative")
    unknown_tolerances = sorted(set(tolerances) - component_names)
    if unknown_tolerances:
        raise ValueError(
            "peak tolerances were supplied for unknown objective components: "
            + ", ".join(unknown_tolerances)
        )
    if any(tolerance <= 0 for tolerance in tolerances.values()):
        raise ValueError("peak tolerances must be positive")

    effective_weights = {
        component.name: float(weights.get(component.name, 1.0))
        for component in components
    }
    if hardware_profile is None and peak_tolerances is None:
        tolerances = {
            name: 1.0
            for name, weight in effective_weights.items()
            if weight > 0
        }

    results: list[ComponentScore] = []
    offset_caches = (
        {name: offset_cache_by_array[name] for name in layouts}
        if offset_cache_by_array is not None
        else {name: {} for name in layouts}
    )
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
            cache_key = (
                array_name,
                layout.signature(),
                component.name,
            )
            cached = (
                array_component_cache.get(cache_key)
                if array_component_cache is not None
                else None
            )
            if cached is None:
                raw = weighted_component_region_count(
                    matrix,
                    layout,
                    component,
                    offset_cache=offset_caches[array_name],
                )
                bound = component.packing_bound(matrix)
                if array_component_cache is not None:
                    array_component_cache[cache_key] = (raw, bound)
            else:
                raw, bound = cached
            per_array.append(ArrayComponentScore(array_name, raw, bound))

        raw_total = sum(item.raw_region_count for item in per_array)
        bound_total = sum(item.packing_lower_bound for item in per_array)
        results.append(
            ComponentScore(
                name=component.name,
                region_bytes=component.region_bytes,
                weight=effective_weights[component.name],
                raw_region_count=raw_total,
                packing_lower_bound=bound_total,
                normalized_excess=normalized_excess(raw_total, bound_total),
                arrays=tuple(per_array),
                normalization_bytes=component.normalization_bytes,
                excess_footprint=excess_footprint(
                    raw_total,
                    bound_total,
                    component.region_bytes,
                    component.normalization_bytes,
                ),
                peak_tolerance=(
                    float(tolerances[component.name])
                    if component.name in tolerances
                    else None
                ),
            )
        )

    active = [component for component in results if component.weight > 0]
    return LayoutScore(
        components=tuple(results),
        codegen=layout_codegen_cost(matrices, layouts),
        weighted_region_count=sum(
            component.weighted_region_count for component in active
        ),
        peak_normalized_excess=max(
            (component.normalized_excess for component in active), default=0.0
        ),
        weighted_normalized_excess=sum(
            component.weighted_normalized_excess for component in active
        ),
        hardware_peak=max(
            (
                component.peak_excess_ratio
                for component in results
                if component.peak_excess_ratio is not None
            ),
            default=0.0,
        ),
        hardware_area=sum(
            component.weighted_excess_footprint for component in active
        ),
    )


def score_problem(
    problem: "RelayProblem",
    layouts: Mapping[str, Layout],
    *,
    component_weights: Mapping[str, float] | None = None,
    peak_tolerances: Mapping[str, float] | None = None,
    hardware_profile: "HardwareProfile | None" = None,
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
        peak_tolerances=peak_tolerances,
        hardware_profile=hardware_profile,
    )


def score_to_dict(score: LayoutScore) -> dict[str, object]:
    """Convert a score to a stable JSON-compatible representation."""

    return {
        "codegen": {
            "runs": score.codegen.runs,
            "xors": score.codegen.xors,
            "arrays": [
                {
                    "name": array.array,
                    "grammar": array.grammar,
                    "runs": array.runs,
                    "xors": array.xors,
                }
                for array in score.codegen.arrays
            ],
        },
        "aggregates": {
            "weighted_region_count": score.weighted_region_count,
            "peak_normalized_excess": score.peak_normalized_excess,
            "weighted_normalized_excess": score.weighted_normalized_excess,
            "hardware_peak": score.hardware_peak,
            "hardware_area": score.hardware_area,
        },
        "components": [
            {
                "name": component.name,
                "region_bytes": component.region_bytes,
                "weight": component.weight,
                "raw_region_count": component.raw_region_count,
                "packing_lower_bound": component.packing_lower_bound,
                "normalized_excess": component.normalized_excess,
                "normalization_bytes": component.normalization_bytes,
                "excess_footprint": component.excess_footprint,
                "peak_tolerance": component.peak_tolerance,
                "peak_excess_ratio": component.peak_excess_ratio,
                "weighted_region_count": component.weighted_region_count,
                "weighted_normalized_excess": (
                    component.weighted_normalized_excess
                ),
                "weighted_excess_footprint": (
                    component.weighted_excess_footprint
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
