r"""Exact frontier solvers for RELAY's structured layout grammars.

The notation follows :mod:`notes/relay.tex`: ``standard`` is
``\mathcal{G}_S``, ``canonical`` is ``\mathcal{G}_C``, and
``outer_canonical`` is a bounded ``\mathcal{G}_{OC}`` language.
``affine`` is the access-induced grammar derived from affine event direction
spaces. The implementations combine exhaustive inner enumeration with exact
count-grid dynamic programs where each grammar permits it.

All costs are minimized. Search and multi-array joins retain a raw component
vector until every target array has been selected. This is important because
``J_peak`` and ``J_area`` are computed after component scores have been summed
across separate allocations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from math import factorial, isfinite, prod
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, TypeVar

from .gf2 import (
    add_vector,
    contains,
    intersection_basis,
    invert_matrix_from_columns,
    is_subspace,
    reduce_vector,
    rref_basis,
)
from .layouts import (
    AffineAccessLayout,
    CanonicalLayout,
    Layout,
    LinearInnerLayout,
    linear_codegen_runs,
    row_major_layout,
)
from .model import EventSequence, MatrixSpec, MemoryEvent
from .objectives import ObjectiveComponent, ObjectiveSpec, build_objectives
from .scoring import (
    LayoutScore,
    excess_footprint,
    normalized_excess,
    score_layouts,
    weighted_component_region_count,
)
from .search import LayoutSeed, ScorePolicy, SearchStats, search_canonical

if TYPE_CHECKING:
    from .hardware import HardwareProfile


Grammar = Literal[
    "standard", "canonical", "outer_canonical", "affine"
]
FrontierType = Literal["pareto", "fine-gated"]
FRONTIER_OBJECTIVES = (
    "fine-region-count",
    "hardware-peak",
    "hardware-area",
    "codegen-runs",
    "codegen-xors",
)


@dataclass(frozen=True)
class SimpleRelayProblem:
    """A kernel problem for one exact grammar-frontier search."""

    matrices: tuple[MatrixSpec, ...]
    events: tuple[MemoryEvent, ...]
    sequences: tuple[EventSequence, ...]
    objectives: tuple[ObjectiveSpec, ...]
    grammar: Grammar
    component_weights: Mapping[str, float] = field(default_factory=dict)
    peak_tolerances: Mapping[str, float] = field(default_factory=dict)
    hardware_profile: "HardwareProfile | None" = None
    frontier_type: FrontierType = "pareto"
    fine_component: str = "issue.g64.stream.load.64B"
    fine_tolerance: float = 0.05
    outer_canonical_max_inner_bits: int = 4
    name: str = "simple_relay_problem"


@dataclass(frozen=True)
class FrontierCost:
    """The notes-aligned costs for one realized joint layout."""

    fine_region_count: float
    hardware_peak: float
    hardware_area: float
    codegen_runs: int
    codegen_xors: int

    @property
    def values(self) -> tuple[float, ...]:
        return (
            self.fine_region_count,
            self.hardware_peak,
            self.hardware_area,
            float(self.codegen_runs),
            float(self.codegen_xors),
        )


@dataclass(frozen=True)
class FrontierMember:
    """One distinct layout mapping retained by the analytical frontier."""

    layouts: Mapping[str, Layout]
    score: LayoutScore
    cost: FrontierCost

    def word_signature(
        self, matrices: Mapping[str, MatrixSpec]
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                name,
                self.layouts[name].word_string(matrices[name])
                if isinstance(self.layouts[name], CanonicalLayout)
                else self.layouts[name].evaluator_descriptor(matrices[name]),
            )
            for name in self.layouts
        )


@dataclass(frozen=True)
class ArraySearchResult:
    """Search size and retained raw frontier for one target array."""

    matrix: str
    grammar_layout_count: int
    raw_frontier_count: int
    search_stats: SearchStats | None = None
    affine_edge_count: int | None = None
    access_lattice_size: int | None = None
    access_block_dimensions: tuple[int, ...] = ()
    active_rank: int | None = None
    inactive_rank: int | None = None
    score_ties_collapsed: bool = False
    tile_hypotheses: int | None = None
    max_inner_bits: int | None = None


@dataclass(frozen=True)
class SimpleRelayResult:
    """An exact joint frontier for one grammar and kernel problem."""

    problem: SimpleRelayProblem
    components: tuple[ObjectiveComponent, ...]
    frontier: tuple[FrontierMember, ...]
    array_searches: tuple[ArraySearchResult, ...]
    joint_raw_frontier_count: int
    context_layouts: Mapping[str, Layout]
    fine_minimum: float
    fine_limit: float | None
    fine_eligible_count: int
    frontier_objectives: tuple[str, ...]
    objectives: tuple[str, ...] = FRONTIER_OBJECTIVES
    exact: bool = True
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class _ArrayCandidate:
    layout: Layout
    raw_scores: Mapping[str, float]


@dataclass(frozen=True)
class _JointCandidate:
    layouts: Mapping[str, Layout]
    raw_scores: Mapping[str, float]


@dataclass(frozen=True)
class _CostedJoint:
    candidate: _JointCandidate
    cost: FrontierCost


_T = TypeVar("_T")


class NonDistributiveAccessError(ValueError):
    """Raised when affine event spaces do not satisfy the grammar premise."""

    def __init__(
        self,
        matrix: str,
        witness: tuple[int, int, int, int, int] | None = None,
    ):
        self.matrix = matrix
        self.witness = witness
        prefix = f"{matrix}: " if matrix else ""
        detail = ""
        if witness is not None:
            x, y, z, left, right = witness
            detail = (
                f" (witness ranks {x}, {y}, {z}; "
                f"left={left}, right={right})"
            )
        super().__init__(
            f"{prefix}affine access lattice is non-distributive{detail}"
        )


class NonAffineAccessError(ValueError):
    """Raised when an objective edge is outside the affine grammar."""

    def __init__(self, matrix: str, objective: str, source: str):
        self.matrix = matrix
        self.objective = objective
        self.source = source
        super().__init__(
            f"{matrix}: objective {objective!r} edge {source!r} "
            "is not an affine coset"
        )


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _pareto(
    items: Sequence[_T], key, *, retain_ties: bool = True
) -> list[_T]:
    """Return a deterministic strict-dominance frontier."""

    groups: dict[tuple[float, ...], list[_T]] = {}
    for item in sorted(items, key=key):
        values = tuple(key(item))
        groups.setdefault(values, []).append(item)
    frontier_values: list[tuple[float, ...]] = []
    for values in groups:
        if any(_dominates(other, values) for other in frontier_values):
            continue
        frontier_values = [
            other
            for other in frontier_values
            if not _dominates(values, other)
        ]
        frontier_values.append(values)
    return [
        item
        for values in frontier_values
        for item in (
            groups[values] if retain_ties else groups[values][:1]
        )
    ]


def _validate_problem(
    problem: SimpleRelayProblem,
) -> tuple[dict[str, MatrixSpec], dict[str, MemoryEvent]]:
    if not problem.matrices:
        raise ValueError("the problem contains no matrices")
    matrices = {matrix.name: matrix for matrix in problem.matrices}
    if len(matrices) != len(problem.matrices):
        raise ValueError("matrix names must be unique")
    events = {event.id: event for event in problem.events}
    if len(events) != len(problem.events):
        raise ValueError("event ids must be unique")
    if not any(matrix.target for matrix in problem.matrices):
        raise ValueError("the problem contains no target matrices")
    if problem.grammar not in (
        "standard",
        "canonical",
        "outer_canonical",
        "affine",
    ):
        raise ValueError(f"unknown grammar {problem.grammar!r}")
    if problem.frontier_type not in ("pareto", "fine-gated"):
        raise ValueError(f"unknown frontier type {problem.frontier_type!r}")
    if not isfinite(problem.fine_tolerance) or problem.fine_tolerance < 0:
        raise ValueError("fine_tolerance must be finite and nonnegative")
    if not 0 <= problem.outer_canonical_max_inner_bits <= 4:
        raise ValueError(
            "outer_canonical_max_inner_bits must be between zero and four"
        )
    for event in problem.events:
        for access in event.accesses:
            if access.array not in matrices:
                raise ValueError(f"event {event.id}: unknown array {access.array}")
            matrices[access.array].validate_coord(access.coord)
    for sequence in problem.sequences:
        unknown = [event_id for event_id in sequence.event_ids if event_id not in events]
        if unknown:
            raise ValueError(
                f"sequence {sequence.name}: unknown event {unknown[0]}"
            )
    return matrices, events


def _effective_weights(
    components: Sequence[ObjectiveComponent], weights: Mapping[str, float]
) -> dict[str, float]:
    component_names = {component.name for component in components}
    unknown = sorted(set(weights) - component_names)
    if unknown:
        raise ValueError(
            "weights were supplied for unknown objective components: "
            + ", ".join(unknown)
        )
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("component weights must be finite and nonnegative")
    return {
        component.name: float(weights.get(component.name, 1.0))
        for component in components
    }


def _effective_tolerances(
    components: Sequence[ObjectiveComponent],
    tolerances: Mapping[str, float],
) -> dict[str, float]:
    component_names = {component.name for component in components}
    unknown = sorted(set(tolerances) - component_names)
    if unknown:
        raise ValueError(
            "peak tolerances were supplied for unknown objective components: "
            + ", ".join(unknown)
        )
    if any(
        not isfinite(tolerance) or tolerance <= 0
        for tolerance in tolerances.values()
    ):
        raise ValueError("peak tolerances must be finite and positive")
    return {name: float(tolerance) for name, tolerance in tolerances.items()}


def _standard_words(matrix: MatrixSpec) -> tuple[tuple[int, ...], ...]:
    """Enumerate the four cut-point forms that define ``G_S``."""

    if matrix.rank != 2:
        raise ValueError("the standard grammar requires rank-2 target matrices")
    first_bits, second_bits = matrix.mode_bits
    words: set[tuple[int, ...]] = set()
    for first_cut in range(first_bits + 1):
        for second_cut in range(second_bits + 1):
            inner = {
                (0,) * first_cut + (1,) * second_cut,
                (1,) * second_cut + (0,) * first_cut,
            }
            outer_first = first_bits - first_cut
            outer_second = second_bits - second_cut
            outer = {
                (0,) * outer_first + (1,) * outer_second,
                (1,) * outer_second + (0,) * outer_first,
            }
            words.update(prefix + suffix for prefix in inner for suffix in outer)
    return tuple(sorted(words))


def _canonical_word_count(matrix: MatrixSpec) -> int:
    total = sum(matrix.mode_bits)
    count = factorial(total)
    for bits in matrix.mode_bits:
        count //= factorial(bits)
    return count


def _canonical_word_scorer(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
):
    """Build a cached full-word scorer using canonical prefix quotients."""

    tile_bits = sum(matrix.mode_bits)
    by_dimension: dict[int, list[ObjectiveComponent]] = {}
    constants: dict[str, float] = {}
    control = row_major_layout(matrix)
    for component in components:
        if (
            component.name not in raw_component_names
            or not component.edges_by_array.get(matrix.name)
        ):
            continue
        dimension = component.dimension(matrix)
        if dimension <= tile_bits:
            by_dimension.setdefault(dimension, []).append(component)
        else:
            constants[component.name] = weighted_component_region_count(
                matrix, control, component
            )

    node_cache: dict[tuple[int, ...], dict[str, float]] = {}

    def node_score(counts: tuple[int, ...]) -> dict[str, float]:
        if counts not in node_cache:
            score: dict[str, float] = {}
            for component in by_dimension.get(sum(counts), ()):
                score[component.name] = sum(
                    edge.weight
                    * len(
                        {
                            tuple(
                                value >> used
                                for value, used in zip(point, counts)
                            )
                            for point in edge.points
                        }
                    )
                    for edge in component.edges_by_array[matrix.name]
                )
            node_cache[counts] = score
        return node_cache[counts]

    def score(layout: CanonicalLayout) -> dict[str, float]:
        counts = [0] * matrix.rank
        result = dict(constants)
        for name, value in node_score(tuple(counts)).items():
            result[name] = result.get(name, 0.0) + value
        for mode in layout.word:
            counts[mode] += 1
            for name, value in node_score(tuple(counts)).items():
                result[name] = result.get(name, 0.0) + value
        result["runs"] = float(layout.runs)
        result["xors"] = float(layout.xor_count)
        return result

    return score


def _raw_key(
    scores: Mapping[str, float], raw_order: Sequence[str]
) -> tuple[float, ...]:
    return tuple(float(scores.get(name, 0.0)) for name in raw_order)


def _standard_array_frontier(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
    raw_order: Sequence[str],
) -> tuple[list[_ArrayCandidate], ArraySearchResult]:
    words = _standard_words(matrix)
    score_word = _canonical_word_scorer(
        matrix, components, raw_component_names
    )
    candidates: list[_ArrayCandidate] = []
    for word in words:
        word_text = "".join(matrix.mode_names[mode] for mode in word)
        layout = CanonicalLayout(
            f"G_S_{word_text}",
            matrix.name,
            matrix.mode_bits,
            word,
            tuple(reversed(range(matrix.rank))),
        )
        layout.validate(matrix)
        candidates.append(
            _ArrayCandidate(
                layout,
                score_word(layout),
            )
        )
    frontier = _pareto(
        candidates,
        lambda candidate: _raw_key(candidate.raw_scores, raw_order),
    )
    return frontier, ArraySearchResult(
        matrix.name,
        len(words),
        len(frontier),
    )


def _canonical_array_frontier(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
    raw_order: Sequence[str],
) -> tuple[list[_ArrayCandidate], ArraySearchResult]:
    grammar_layout_count = _canonical_word_count(matrix)
    stats: list[SearchStats] = []
    seeds: list[LayoutSeed] = search_canonical(
        matrix,
        components,
        matrix.mode_bits,
        (tuple(reversed(range(matrix.rank))),),
        ScorePolicy(
            kind="pareto",
            order=tuple(raw_order),
            paths_per_state=grammar_layout_count,
            frontier_limit=grammar_layout_count,
        ),
        candidates_per_tile=grammar_layout_count,
        stats_sink=stats,
    )
    if not stats or not stats[0].exact or stats[0].truncated:
        raise RuntimeError(
            f"canonical DP for {matrix.name} unexpectedly truncated its frontier"
        )
    score_word = _canonical_word_scorer(
        matrix, components, raw_component_names
    )
    candidates = [
        _ArrayCandidate(
            seed.layout,
            score_word(seed.layout),
        )
        for seed in seeds
        if isinstance(seed.layout, CanonicalLayout)
    ]
    frontier = _pareto(
        candidates,
        lambda candidate: _raw_key(candidate.raw_scores, raw_order),
    )
    return frontier, ArraySearchResult(
        matrix.name,
        grammar_layout_count,
        len(frontier),
        stats[0],
    )


@dataclass(frozen=True)
class _DifferencePattern:
    differences: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class _OuterCanonicalPath:
    counts: tuple[int, ...]
    first_mode: int | None
    last_mode: int | None
    word: tuple[int, ...]
    raw_scores: Mapping[str, float]


@dataclass(frozen=True)
class _OuterCanonicalCandidate:
    word: tuple[int, ...]
    first_mode: int | None
    raw_scores: Mapping[str, float]


@dataclass(frozen=True)
class _LinearInnerCandidate:
    a_rows: tuple[int, ...]
    columns: tuple[int, ...]
    merge_mode: int | None
    raw_scores: Mapping[str, float]


def _component_difference_patterns(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
) -> tuple[
    dict[str, tuple[_DifferencePattern, ...]],
    dict[int, tuple[str, ...]],
]:
    """Aggregate translation-equivalent edges for quotient scoring."""

    patterns: dict[str, tuple[_DifferencePattern, ...]] = {}
    by_dimension: dict[int, list[str]] = {}
    for component in components:
        edges = component.edges_by_array.get(matrix.name, ())
        if component.name not in raw_component_names or not edges:
            continue
        aggregated: dict[tuple[int, ...], float] = {}
        for edge in edges:
            anchor = matrix.coord_to_bits(edge.points[0])
            differences = tuple(
                sorted(
                    matrix.coord_to_bits(point) ^ anchor
                    for point in edge.points
                )
            )
            aggregated[differences] = (
                aggregated.get(differences, 0.0) + edge.weight
            )
        patterns[component.name] = tuple(
            _DifferencePattern(differences, weight)
            for differences, weight in sorted(aggregated.items())
        )
        by_dimension.setdefault(component.dimension(matrix), []).append(
            component.name
        )
    return patterns, {
        dimension: tuple(sorted(names))
        for dimension, names in by_dimension.items()
    }


def _quotient_pattern_score(
    patterns: Sequence[_DifferencePattern], subspace: Sequence[int]
) -> float:
    return sum(
        pattern.weight
        * len(
            {
                reduce_vector(difference, subspace)
                for difference in pattern.differences
            }
        )
        for pattern in patterns
    )


def _embed_inner_vector(
    matrix: MatrixSpec,
    tile_exponents: Sequence[int],
    vector: int,
) -> int:
    compact_offsets = matrix.bit_offsets(tile_exponents)
    global_offsets = matrix.bit_offsets()
    result = 0
    for mode, width in enumerate(tile_exponents):
        for bit in range(width):
            compact = compact_offsets[mode] + bit
            if (vector >> compact) & 1:
                result |= 1 << (global_offsets[mode] + bit)
    return result


def _inner_coordinate_basis(
    matrix: MatrixSpec, tile_exponents: Sequence[int]
) -> tuple[int, ...]:
    offsets = matrix.bit_offsets()
    return tuple(
        1 << (offsets[mode] + bit)
        for mode, width in enumerate(tile_exponents)
        for bit in range(width)
    )


def _outer_node_basis(
    matrix: MatrixSpec,
    tile_exponents: Sequence[int],
    counts: Sequence[int],
) -> tuple[int, ...]:
    offsets = matrix.bit_offsets()
    return (
        *_inner_coordinate_basis(matrix, tile_exponents),
        *(
            1 << (offsets[mode] + tile_exponents[mode] + bit)
            for mode, count in enumerate(counts)
            for bit in range(count)
        ),
    )


def _outer_canonical_paths(
    matrix: MatrixSpec,
    tile_exponents: tuple[int, ...],
    patterns: Mapping[str, Sequence[_DifferencePattern]],
    by_dimension: Mapping[int, Sequence[str]],
    raw_order: Sequence[str],
    stats: SearchStats,
) -> list[_OuterCanonicalCandidate]:
    remaining = tuple(
        bits - exponent
        for bits, exponent in zip(matrix.mode_bits, tile_exponents)
    )
    zero = tuple(0 for _ in remaining)
    initial = _OuterCanonicalPath(zero, None, None, (), {"runs": 0.0})
    layer: dict[
        tuple[tuple[int, ...], int | None, int | None],
        list[_OuterCanonicalPath],
    ] = {(zero, None, None): [initial]}
    stats.states += 1
    node_cache: dict[tuple[int, ...], dict[str, float]] = {}

    def node_score(counts: tuple[int, ...]) -> dict[str, float]:
        if counts not in node_cache:
            dimension = sum(tile_exponents) + sum(counts)
            basis = _outer_node_basis(matrix, tile_exponents, counts)
            node_cache[counts] = {
                name: _quotient_pattern_score(patterns[name], basis)
                for name in by_dimension.get(dimension, ())
            }
        return node_cache[counts]

    for _ in range(sum(remaining)):
        pending: dict[
            tuple[tuple[int, ...], int, int],
            list[_OuterCanonicalPath],
        ] = {}
        for paths in layer.values():
            for path in paths:
                for mode, limit in enumerate(remaining):
                    if path.counts[mode] >= limit:
                        continue
                    counts = list(path.counts)
                    counts[mode] += 1
                    next_counts = tuple(counts)
                    scores = _add_scores(
                        path.raw_scores, node_score(next_counts)
                    )
                    if path.last_mode is None or path.last_mode != mode:
                        scores["runs"] = scores.get("runs", 0.0) + 1.0
                    first_mode = (
                        mode if path.first_mode is None else path.first_mode
                    )
                    candidate = _OuterCanonicalPath(
                        next_counts,
                        first_mode,
                        mode,
                        (*path.word, mode),
                        scores,
                    )
                    pending.setdefault(
                        (next_counts, first_mode, mode), []
                    ).append(candidate)
                    stats.transitions += 1
                    stats.paths_considered += 1
        layer = {}
        for state, candidates in pending.items():
            retained = _pareto(
                candidates,
                lambda candidate: _raw_key(
                    candidate.raw_scores, raw_order
                ),
                retain_ties=False,
            )
            layer[state] = retained
            stats.paths_retained += len(retained)
        stats.states += len(layer)

    terminal = [path for paths in layer.values() for path in paths]
    if not terminal:
        terminal = [initial]
    constants = {
        name: sum(pattern.weight for pattern in component_patterns)
        for name, component_patterns in patterns.items()
        if any(
            name in names and dimension > matrix.total_bits
            for dimension, names in by_dimension.items()
        )
    }
    result = [
        _OuterCanonicalCandidate(
            path.word,
            path.first_mode,
            _add_scores(path.raw_scores, constants),
        )
        for path in terminal
    ]
    grouped: list[_OuterCanonicalCandidate] = []
    for first_mode in {candidate.first_mode for candidate in result}:
        grouped.extend(
            _pareto(
                [
                    candidate
                    for candidate in result
                    if candidate.first_mode == first_mode
                ],
                lambda candidate: _raw_key(
                    candidate.raw_scores, raw_order
                ),
                retain_ties=False,
            )
        )
    return grouped


def _linear_inner_candidates(
    matrix: MatrixSpec,
    tile_exponents: tuple[int, ...],
    patterns: Mapping[str, Sequence[_DifferencePattern]],
    by_dimension: Mapping[int, Sequence[str]],
    raw_order: Sequence[str],
    stats: SearchStats,
) -> list[_LinearInnerCandidate]:
    width = sum(tile_exponents)
    if width == 0:
        return [
            _LinearInnerCandidate((), (), None, {"runs": 0.0, "xors": 0.0})
        ]
    candidates: list[_LinearInnerCandidate] = []
    for columns in permutations(range(1, 1 << width), width):
        if len(rref_basis(columns)) != width:
            continue
        rows = invert_matrix_from_columns(columns, width)
        scores: dict[str, float] = {}
        for dimension, names in by_dimension.items():
            if dimension > width:
                continue
            basis = rref_basis(
                _embed_inner_vector(
                    matrix, tile_exponents, vector
                )
                for vector in columns[:dimension]
            )
            for name in names:
                scores[name] = _quotient_pattern_score(
                    patterns[name], basis
                )
        scores["runs"] = float(
            linear_codegen_runs(rows, tile_exponents)
        )
        scores["xors"] = float(
            sum(max(0, row.bit_count() - 1) for row in rows)
        )
        last_row = rows[-1]
        merge_mode: int | None = None
        if last_row.bit_count() == 1:
            source_bit = last_row.bit_length() - 1
            cursor = 0
            for mode, mode_width in enumerate(tile_exponents):
                if cursor <= source_bit < cursor + mode_width:
                    if source_bit - cursor == mode_width - 1:
                        merge_mode = mode
                    break
                cursor += mode_width
        candidates.append(
            _LinearInnerCandidate(
                rows, tuple(columns), merge_mode, scores
            )
        )
        stats.paths_considered += 1
    retained: list[_LinearInnerCandidate] = []
    for merge_mode in {candidate.merge_mode for candidate in candidates}:
        retained.extend(
            _pareto(
                [
                    candidate
                    for candidate in candidates
                    if candidate.merge_mode == merge_mode
                ],
                lambda candidate: _raw_key(
                    candidate.raw_scores, raw_order
                ),
                retain_ties=False,
            )
        )
    stats.paths_retained += len(retained)
    return retained


def _general_linear_count(width: int) -> int:
    return prod(
        (1 << width) - (1 << rank) for rank in range(width)
    )


def _outer_word_count(remaining: Sequence[int]) -> int:
    count = factorial(sum(remaining))
    for width in remaining:
        count //= factorial(width)
    return count


def _outer_canonical_array_frontier(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
    raw_order: Sequence[str],
    max_inner_bits: int,
) -> tuple[list[_ArrayCandidate], ArraySearchResult]:
    """Search the exact bounded ``G_OC`` language used by the experiment."""

    patterns, by_dimension = _component_difference_patterns(
        matrix, components, raw_component_names
    )
    tile_hypotheses = tuple(
        exponents
        for exponents in product(
            *(range(bits + 1) for bits in matrix.mode_bits)
        )
        if sum(exponents) <= max_inner_bits
    )
    grammar_layout_count = sum(
        _general_linear_count(sum(tile))
        * _outer_word_count(
            tuple(
                bits - exponent
                for bits, exponent in zip(matrix.mode_bits, tile)
            )
        )
        for tile in tile_hypotheses
    )
    stats = SearchStats(
        "outer_canonical",
        matrix.mode_bits,
        note=(
            "exact bounded G_OC search over every invertible inner map and "
            "canonical outer word; equal-score layouts collapsed"
        ),
    )
    candidates: list[_ArrayCandidate] = []
    for tile in tile_hypotheses:
        inner_candidates = _linear_inner_candidates(
            matrix,
            tile,
            patterns,
            by_dimension,
            raw_order,
            stats,
        )
        outer_candidates = _outer_canonical_paths(
            matrix,
            tile,
            patterns,
            by_dimension,
            raw_order,
            stats,
        )
        for inner in inner_candidates:
            for outer in outer_candidates:
                scores = _add_scores(
                    inner.raw_scores, outer.raw_scores
                )
                if (
                    inner.merge_mode is not None
                    and inner.merge_mode == outer.first_mode
                ):
                    scores["runs"] -= 1.0
                if not inner.a_rows:
                    word_text = "".join(
                        matrix.mode_names[mode] for mode in outer.word
                    )
                    layout: Layout = CanonicalLayout(
                        f"G_OC_canonical_{word_text}",
                        matrix.name,
                        matrix.mode_bits,
                        outer.word,
                        tuple(reversed(range(matrix.rank))),
                    )
                else:
                    layout = LinearInnerLayout(
                        "G_OC_"
                        + "x".join(str(1 << exponent) for exponent in tile)
                        + "_"
                        + str(len(candidates)),
                        matrix.name,
                        tile,
                        inner.a_rows,
                        tuple(reversed(range(matrix.rank))),
                        inner.columns,
                        sum(tile),
                        outer.word,
                    )
                layout.validate(matrix)  # type: ignore[attr-defined]
                scores["runs"] = float(layout.runs)
                scores["xors"] = float(layout.xor_count)
                candidates.append(_ArrayCandidate(layout, scores))
    deduplicated: dict[tuple[object, ...], _ArrayCandidate] = {}
    for candidate in candidates:
        signature = candidate.layout.signature()
        incumbent = deduplicated.get(signature)
        if incumbent is None or _raw_key(
            candidate.raw_scores, raw_order
        ) < _raw_key(incumbent.raw_scores, raw_order):
            deduplicated[signature] = candidate
    representatives = _pareto(
        list(deduplicated.values()),
        lambda candidate: _raw_key(candidate.raw_scores, raw_order),
        retain_ties=False,
    )
    canonical, _ = _canonical_array_frontier(
        matrix,
        components,
        raw_component_names,
        raw_order,
    )
    combined: dict[tuple[object, ...], _ArrayCandidate] = {
        candidate.layout.signature(): candidate
        for candidate in (*representatives, *canonical)
    }
    frontier = _pareto(
        list(combined.values()),
        lambda candidate: _raw_key(candidate.raw_scores, raw_order),
    )
    stats.exact = True
    return frontier, ArraySearchResult(
        matrix.name,
        grammar_layout_count,
        len(frontier),
        stats,
        score_ties_collapsed=True,
        tile_hypotheses=len(tile_hypotheses),
        max_inner_bits=max_inner_bits,
    )


@dataclass(frozen=True)
class _AccessBlock:
    join_space: tuple[int, ...]
    lower_cover: tuple[int, ...]
    basis: tuple[int, ...]


@dataclass(frozen=True)
class _AffineEdge:
    rank: int
    blocks: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class _AccessPath:
    counts: tuple[int, ...]
    last_block: int | None
    word: tuple[int, ...]
    raw_scores: Mapping[str, float]


def _join_space(
    left: Sequence[int], right: Sequence[int]
) -> tuple[int, ...]:
    return rref_basis((*left, *right))


def _distributive_witness(
    spaces: Sequence[tuple[int, ...]], width: int
) -> tuple[int, int, int, int, int] | None:
    joins: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]
    ] = {}
    meets: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[int, ...]
    ] = {}

    def cached_join(
        left: tuple[int, ...], right: tuple[int, ...]
    ) -> tuple[int, ...]:
        key = tuple(sorted((left, right)))
        if key not in joins:
            joins[key] = _join_space(left, right)
        return joins[key]

    def cached_meet(
        left: tuple[int, ...], right: tuple[int, ...]
    ) -> tuple[int, ...]:
        key = tuple(sorted((left, right)))
        if key not in meets:
            meets[key] = intersection_basis(left, right, width)
        return meets[key]

    for left in spaces:
        for middle in spaces:
            for right in spaces:
                lhs = cached_meet(
                    left, cached_join(middle, right)
                )
                rhs = cached_join(
                    cached_meet(left, middle),
                    cached_meet(left, right),
                )
                if lhs != rhs:
                    return (
                        len(left),
                        len(middle),
                        len(right),
                        len(lhs),
                        len(rhs),
                    )
    return None


def _access_lattice(
    spaces: Sequence[tuple[int, ...]], width: int
) -> tuple[tuple[int, ...], ...]:
    witness = _distributive_witness(spaces, width)
    if witness is not None:
        raise NonDistributiveAccessError("", witness)
    active = rref_basis(vector for space in spaces for vector in space)
    lattice = {(), active, *spaces}
    while True:
        current = tuple(lattice)
        additions: set[tuple[int, ...]] = set()
        for index, left in enumerate(current):
            for right in current[index:]:
                additions.add(_join_space(left, right))
                additions.add(intersection_basis(left, right, width))
        if additions <= lattice:
            break
        lattice.update(additions)
        if len(lattice) > 4096:
            raise ValueError(
                "affine access lattice exceeded 4096 spaces during closure"
            )
    return tuple(sorted(lattice, key=lambda space: (len(space), space)))


def _relative_sparse_complement(
    lower: Sequence[int], upper: Sequence[int], width: int
) -> tuple[int, ...]:
    selected: list[int] = []
    candidates = (
        *(1 << bit for bit in range(width)),
        *upper,
    )
    for vector in candidates:
        if not contains(upper, vector):
            continue
        extended = add_vector((*lower, *selected), vector)
        if extended is not None:
            selected.append(vector)
        if len(lower) + len(selected) == len(upper):
            break
    if len(lower) + len(selected) != len(upper):
        raise RuntimeError("failed to construct an adapted access-block basis")
    return tuple(selected)


def _access_blocks(
    lattice: Sequence[tuple[int, ...]], width: int
) -> tuple[_AccessBlock, ...]:
    blocks: list[_AccessBlock] = []
    for space in lattice:
        if not space:
            continue
        proper = [
            lower
            for lower in lattice
            if lower != space and is_subspace(lower, space)
        ]
        covers = [
            lower
            for lower in proper
            if not any(
                lower != middle
                and middle != space
                and is_subspace(lower, middle)
                and is_subspace(middle, space)
                for middle in proper
            )
        ]
        if len(covers) == 1:
            lower = covers[0]
            blocks.append(
                _AccessBlock(
                    space,
                    lower,
                    _relative_sparse_complement(lower, space, width),
                )
            )
    blocks.sort(
        key=lambda block: (
            len(block.join_space),
            block.join_space,
            block.basis,
        )
    )
    all_columns = tuple(vector for block in blocks for vector in block.basis)
    active = max(lattice, key=len)
    if len(rref_basis(all_columns)) != len(active):
        raise NonDistributiveAccessError("")
    for space in lattice:
        represented = rref_basis(
            vector
            for block in blocks
            if is_subspace(block.join_space, space)
            for vector in block.basis
        )
        if represented != space:
            raise NonDistributiveAccessError("")
    return tuple(blocks)


def _affine_edge_spaces(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
) -> tuple[
    dict[str, tuple[tuple[tuple[int, ...], float], ...]],
    tuple[tuple[int, ...], ...],
    int,
]:
    by_component: dict[str, tuple[tuple[tuple[int, ...], float], ...]] = {}
    spaces: set[tuple[int, ...]] = set()
    edge_count = 0
    for component in components:
        if component.name not in raw_component_names:
            continue
        aggregated: dict[tuple[int, ...], float] = {}
        for edge in component.edges_by_array.get(matrix.name, ()):
            anchor = matrix.coord_to_bits(edge.points[0])
            differences = {
                matrix.coord_to_bits(point) ^ anchor for point in edge.points
            }
            direction_space = rref_basis(differences)
            if len(differences) != 1 << len(direction_space):
                raise NonAffineAccessError(
                    matrix.name,
                    component.name,
                    edge.source,
                )
            aggregated[direction_space] = (
                aggregated.get(direction_space, 0.0) + edge.weight
            )
            spaces.add(direction_space)
            edge_count += 1
        if aggregated:
            by_component[component.name] = tuple(sorted(aggregated.items()))
    return by_component, tuple(sorted(spaces)), edge_count


def _inactive_complement(
    active_columns: Sequence[int], width: int
) -> tuple[int, ...]:
    selected: list[int] = []
    for bit in range(width):
        vector = 1 << bit
        if add_vector((*active_columns, *selected), vector) is not None:
            selected.append(vector)
        if len(active_columns) + len(selected) == width:
            break
    if len(active_columns) + len(selected) != width:
        raise RuntimeError("failed to complete the affine access basis")
    return tuple(selected)


def _affine_word_count(blocks: Sequence[_AccessBlock]) -> int:
    total = sum(len(block.basis) for block in blocks)
    count = factorial(total)
    for block in blocks:
        count //= factorial(len(block.basis))
    return count


def _affine_array_frontier(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
    raw_order: Sequence[str],
) -> tuple[list[_ArrayCandidate], ArraySearchResult]:
    component_spaces, spaces, edge_count = _affine_edge_spaces(
        matrix, components, raw_component_names
    )
    try:
        lattice = _access_lattice(spaces, matrix.total_bits)
    except NonDistributiveAccessError as error:
        raise NonDistributiveAccessError(matrix.name, error.witness) from error
    try:
        blocks = _access_blocks(lattice, matrix.total_bits)
    except NonDistributiveAccessError as error:
        raise NonDistributiveAccessError(matrix.name, error.witness) from error
    block_dimensions = tuple(len(block.basis) for block in blocks)
    active_columns = tuple(
        vector for block in blocks for vector in block.basis
    )
    inactive = _inactive_complement(active_columns, matrix.total_bits)
    active_rank = len(active_columns)
    reference_columns = (*active_columns, *inactive)
    reference_rows = invert_matrix_from_columns(
        reference_columns, matrix.total_bits
    )
    block_offsets: list[int] = []
    offset = 0
    for dimension in block_dimensions:
        block_offsets.append(offset)
        offset += dimension

    affine_edges: dict[str, tuple[_AffineEdge, ...]] = {}
    for component_name, weighted_spaces in component_spaces.items():
        affine_edges[component_name] = tuple(
            _AffineEdge(
                len(space),
                tuple(
                    index
                    for index, block in enumerate(blocks)
                    if is_subspace(block.join_space, space)
                ),
                weight,
            )
            for space, weight in weighted_spaces
        )
    dimensions: dict[int, list[str]] = {}
    constants: dict[str, float] = {}
    components_by_name = {component.name: component for component in components}
    for name in affine_edges:
        dimension = components_by_name[name].dimension(matrix)
        if dimension <= active_rank:
            dimensions.setdefault(dimension, []).append(name)
        else:
            constants[name] = sum(edge.weight for edge in affine_edges[name])

    node_cache: dict[tuple[int, ...], dict[str, float]] = {}

    def node_score(counts: tuple[int, ...]) -> dict[str, float]:
        if counts not in node_cache:
            result: dict[str, float] = {}
            for name in dimensions.get(sum(counts), ()):
                result[name] = sum(
                    edge.weight
                    * float(
                        1
                        << (
                            edge.rank
                            - sum(counts[index] for index in edge.blocks)
                        )
                    )
                    for edge in affine_edges[name]
                )
            node_cache[counts] = result
        return node_cache[counts]

    zero = tuple(0 for _ in blocks)
    initial_scores = _add_scores(constants, node_score(zero))
    initial_scores["runs"] = 0.0
    initial_scores["xors"] = float(
        sum(max(0, row.bit_count() - 1) for row in reference_rows)
    )
    layer: dict[tuple[tuple[int, ...], int | None], list[_AccessPath]] = {
        (zero, None): [_AccessPath(zero, None, (), initial_scores)]
    }
    stats = SearchStats(
        "affine_access",
        matrix.mode_bits,
        states=1,
        active_rank=active_rank,
        note="exact affine-access count-grid DP; equal-score paths collapsed",
    )
    for _dimension in range(active_rank):
        pending: dict[
            tuple[tuple[int, ...], int], list[_AccessPath]
        ] = {}
        for paths in layer.values():
            for path in paths:
                for block, limit in enumerate(block_dimensions):
                    if path.counts[block] >= limit:
                        continue
                    counts = list(path.counts)
                    counts[block] += 1
                    next_counts = tuple(counts)
                    scores = _add_scores(
                        path.raw_scores, node_score(next_counts)
                    )
                    next_reference = (
                        block_offsets[block] + path.counts[block]
                    )
                    if path.last_block is None:
                        scores["runs"] = scores.get("runs", 0.0) + 1.0
                    else:
                        previous_reference = (
                            block_offsets[path.last_block]
                            + path.counts[path.last_block]
                            - 1
                        )
                        if linear_codegen_runs(
                            (
                                reference_rows[previous_reference],
                                reference_rows[next_reference],
                            ),
                            matrix.mode_bits,
                        ) != 1:
                            scores["runs"] = scores.get("runs", 0.0) + 1.0
                    next_path = _AccessPath(
                        next_counts,
                        block,
                        (*path.word, block),
                        scores,
                    )
                    pending.setdefault((next_counts, block), []).append(
                        next_path
                    )
                    stats.transitions += 1
                    stats.paths_considered += 1
        layer = {}
        for state, paths in pending.items():
            retained = _pareto(
                paths,
                lambda path: _raw_key(path.raw_scores, raw_order),
                retain_ties=False,
            )
            layer[state] = retained
            stats.paths_retained += len(retained)
        stats.states += len(layer)

    terminal = [path for paths in layer.values() for path in paths]
    candidates: list[_ArrayCandidate] = []
    for path in terminal:
        used = [0] * len(blocks)
        ordered_columns: list[int] = []
        for block in path.word:
            ordered_columns.append(blocks[block].basis[used[block]])
            used[block] += 1
        ordered_columns.extend(inactive)
        columns = tuple(ordered_columns)
        reference_order = []
        used = [0] * len(blocks)
        for block in path.word:
            reference_order.append(block_offsets[block] + used[block])
            used[block] += 1
        reference_order.extend(range(active_rank, matrix.total_bits))
        a_rows = tuple(reference_rows[index] for index in reference_order)
        word_text = "".join(chr(ord("a") + block) for block in path.word)
        layout = AffineAccessLayout(
            f"G_A_{word_text}",
            matrix.name,
            matrix.mode_bits,
            a_rows,
            tuple(reversed(range(matrix.rank))),
            columns,
            path.word,
            block_dimensions,
            len(inactive),
        )
        layout.validate(matrix)
        scores = dict(path.raw_scores)
        scores["runs"] = float(layout.runs)
        scores["xors"] = float(layout.xor_count)
        candidates.append(_ArrayCandidate(layout, scores))
    frontier = _pareto(
        candidates,
        lambda candidate: _raw_key(candidate.raw_scores, raw_order),
        retain_ties=False,
    )
    return frontier, ArraySearchResult(
        matrix.name,
        _affine_word_count(blocks),
        len(frontier),
        stats,
        affine_edge_count=edge_count,
        access_lattice_size=len(lattice),
        access_block_dimensions=block_dimensions,
        active_rank=active_rank,
        inactive_rank=len(inactive),
        score_ties_collapsed=True,
    )


def _add_scores(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    result = dict(left)
    for name, value in right.items():
        result[name] = result.get(name, 0.0) + float(value)
    return result


def _context(
    matrices: Sequence[MatrixSpec],
    components: Sequence[ObjectiveComponent],
    raw_component_names: set[str],
) -> tuple[dict[str, Layout], dict[str, float]]:
    layouts: dict[str, Layout] = {}
    scores: dict[str, float] = {}
    for matrix in matrices:
        if matrix.target:
            continue
        layout = row_major_layout(matrix)
        layouts[matrix.name] = layout
        for component in components:
            if (
                component.name in raw_component_names
                and component.edges_by_array.get(matrix.name)
            ):
                scores[component.name] = scores.get(
                    component.name, 0.0
                ) + weighted_component_region_count(
                    matrix, layout, component
                )
    return layouts, scores


def _joint_raw_frontier(
    array_candidates: Sequence[tuple[str, Sequence[_ArrayCandidate]]],
    context_scores: Mapping[str, float],
    raw_order: Sequence[str],
    *,
    retain_ties: bool = True,
) -> list[_JointCandidate]:
    frontier = [_JointCandidate({}, dict(context_scores))]
    for name, candidates in array_candidates:
        expanded = [
            _JointCandidate(
                {**partial.layouts, name: candidate.layout},
                _add_scores(partial.raw_scores, candidate.raw_scores),
            )
            for partial in frontier
            for candidate in candidates
        ]
        frontier = _pareto(
            expanded,
            lambda candidate: _raw_key(candidate.raw_scores, raw_order),
            retain_ties=retain_ties,
        )
    return frontier


def _member_cost(score: LayoutScore, fine_component: str) -> FrontierCost:
    return FrontierCost(
        fine_region_count=score.component(fine_component).raw_region_count,
        hardware_peak=score.hardware_peak,
        hardware_area=score.hardware_area,
        codegen_runs=score.codegen.runs,
        codegen_xors=score.codegen.xors,
    )


def _raw_cost(
    raw_scores: Mapping[str, float],
    matrices: Sequence[MatrixSpec],
    components: Sequence[ObjectiveComponent],
    weights: Mapping[str, float],
    peak_tolerances: Mapping[str, float],
    fine_component: str,
) -> FrontierCost:
    active = [component for component in components if weights[component.name] > 0]
    peak_components = [
        component
        for component in components
        if component.name in peak_tolerances
    ]
    excesses = {
        component.name: normalized_excess(
            raw_scores.get(component.name, 0.0),
            sum(
                component.packing_bound(matrix)
                for matrix in matrices
                if component.edges_by_array.get(matrix.name)
            ),
        )
        for component in (*active, *peak_components)
    }
    return FrontierCost(
        fine_region_count=float(raw_scores.get(fine_component, 0.0)),
        hardware_peak=max(
            (
                excesses[component.name]
                / peak_tolerances[component.name]
                for component in peak_components
            ),
            default=0.0,
        ),
        hardware_area=sum(
            weights[component.name]
            * excess_footprint(
                raw_scores.get(component.name, 0.0),
                sum(
                    component.packing_bound(matrix)
                    for matrix in matrices
                    if component.edges_by_array.get(matrix.name)
                ),
                component.region_bytes,
                component.normalization_bytes,
            )
            for component in active
        ),
        codegen_runs=int(raw_scores.get("runs", 0.0)),
        codegen_xors=int(raw_scores.get("xors", 0.0)),
    )


def _final_frontier(
    items: Sequence[_T],
    frontier_type: FrontierType,
    fine_tolerance: float,
    *,
    retain_ties: bool = True,
) -> list[_T]:
    if frontier_type == "pareto":
        objectives = lambda item: item.cost.values
        eligible = list(items)
    else:
        minimum = min(item.cost.fine_region_count for item in items)
        limit = (1.0 + fine_tolerance) * minimum
        eligible = [
            item
            for item in items
            if item.cost.fine_region_count <= limit
        ]
        objectives = lambda item: item.cost.values[1:]
    return _pareto(eligible, objectives, retain_ties=retain_ties)


def simple_solve(problem: SimpleRelayProblem) -> SimpleRelayResult:
    """Return an exact structured-grammar joint layout frontier.

    ``frontier_type='pareto'`` returns the ordinary five-cost Pareto frontier.
    ``frontier_type='fine-gated'`` first restricts layouts to
    ``Q_fine <= (1 + fine_tolerance) Q_fine*`` and then Pareto-filters over
    ``(J_peak, J_area, runs, xors)``. Exact score ties remain as distinct
    layouts because hardware performance may distinguish them. The
    outer-canonical grammar preserves the complete canonical tie family but
    collapses equivalent noncanonical inner realizations. The affine grammar
    likewise keeps one deterministic representative per equivalent path so
    these exponentially larger languages remain enumerable.
    """

    start = perf_counter()
    matrices, events = _validate_problem(problem)
    components = tuple(
        build_objectives(
            problem.objectives,
            matrices,
            events,
            problem.sequences,
        )
    )
    components_by_name = {component.name: component for component in components}
    if problem.fine_component not in components_by_name:
        raise ValueError(
            f"unknown fine objective component {problem.fine_component!r}"
        )
    if problem.hardware_profile is not None and (
        problem.component_weights or problem.peak_tolerances
    ):
        raise ValueError(
            "hardware_profile cannot be combined with explicit weights or tolerances"
        )
    if problem.hardware_profile is not None:
        weights = problem.hardware_profile.component_weights(components)
        peak_tolerances = problem.hardware_profile.peak_tolerances(components)
    else:
        weights = _effective_weights(components, problem.component_weights)
        peak_tolerances = _effective_tolerances(
            components, problem.peak_tolerances
        )
        if not problem.peak_tolerances:
            peak_tolerances = {
                name: 1.0 for name, weight in weights.items() if weight > 0
            }
    raw_names = {
        problem.fine_component,
        *peak_tolerances,
        *(
            component.name
            for component in components
            if weights[component.name] > 0
        ),
    }
    non_searchable = sorted(
        name for name in raw_names if not components_by_name[name].search
    )
    if non_searchable:
        raise ValueError(
            "frontier objectives must participate in search: "
            + ", ".join(non_searchable)
        )
    raw_order = (*sorted(raw_names), "runs", "xors")

    context_layouts, context_scores = _context(
        problem.matrices, components, raw_names
    )
    array_frontiers: list[tuple[str, Sequence[_ArrayCandidate]]] = []
    array_searches: list[ArraySearchResult] = []
    for matrix in problem.matrices:
        if not matrix.target:
            continue
        if problem.grammar == "standard":
            candidates, search_result = _standard_array_frontier(
                matrix,
                components,
                raw_names,
                raw_order,
            )
        elif problem.grammar == "canonical":
            candidates, search_result = _canonical_array_frontier(
                matrix,
                components,
                raw_names,
                raw_order,
            )
        elif problem.grammar == "outer_canonical":
            candidates, search_result = _outer_canonical_array_frontier(
                matrix,
                components,
                raw_names,
                raw_order,
                problem.outer_canonical_max_inner_bits,
            )
        else:
            candidates, search_result = _affine_array_frontier(
                matrix,
                components,
                raw_names,
                raw_order,
            )
        array_frontiers.append((matrix.name, candidates))
        array_searches.append(search_result)

    joint = _joint_raw_frontier(
        array_frontiers,
        context_scores,
        raw_order,
        retain_ties=problem.grammar != "affine",
    )
    costed = [
        _CostedJoint(
            candidate,
            _raw_cost(
                candidate.raw_scores,
                problem.matrices,
                components,
                weights,
                peak_tolerances,
                problem.fine_component,
            ),
        )
        for candidate in joint
    ]
    fine_minimum = min(item.cost.fine_region_count for item in costed)
    if problem.frontier_type == "pareto":
        fine_limit = None
        fine_eligible_count = len(costed)
        frontier_objectives = FRONTIER_OBJECTIVES
    else:
        fine_limit = (1.0 + problem.fine_tolerance) * fine_minimum
        fine_eligible_count = sum(
            item.cost.fine_region_count <= fine_limit for item in costed
        )
        frontier_objectives = FRONTIER_OBJECTIVES[1:]
    selected = _final_frontier(
        costed,
        problem.frontier_type,
        problem.fine_tolerance,
        retain_ties=problem.grammar != "affine",
    )
    members: list[FrontierMember] = []
    array_component_cache: dict[
        tuple[str, tuple[object, ...], str], tuple[float, float]
    ] = {}
    for item in selected:
        candidate = item.candidate
        layouts: dict[str, Layout] = {
            **context_layouts,
            **candidate.layouts,
        }
        score = score_layouts(
            matrices,
            components,
            layouts,
            component_weights=weights,
            peak_tolerances=peak_tolerances,
            array_component_cache=array_component_cache,
        )
        realized_cost = _member_cost(score, problem.fine_component)
        if any(
            abs(left - right) > 1.0e-9
            for left, right in zip(item.cost.values, realized_cost.values)
        ):
            raise RuntimeError(
                "search cost does not match concrete layout scoring for "
                + ", ".join(candidate.layouts)
            )
        members.append(
            FrontierMember(
                layouts=dict(candidate.layouts),
                score=score,
                cost=realized_cost,
            )
        )
    members.sort(
        key=lambda member: (
            member.cost.values,
            member.word_signature(matrices),
        )
    )
    return SimpleRelayResult(
        problem=problem,
        components=components,
        frontier=tuple(members),
        array_searches=tuple(array_searches),
        joint_raw_frontier_count=len(joint),
        context_layouts=context_layouts,
        fine_minimum=fine_minimum,
        fine_limit=fine_limit,
        fine_eligible_count=fine_eligible_count,
        frontier_objectives=frontier_objectives,
        elapsed_seconds=perf_counter() - start,
    )
