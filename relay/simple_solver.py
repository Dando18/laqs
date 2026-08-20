r"""Exact frontier solvers for the standard and canonical layout grammars.

The notation follows :mod:`notes/relay.tex`: ``standard`` is
``\mathcal{G}_S`` and ``canonical`` is ``\mathcal{G}_C``. Both grammars
produce full-matrix canonical words. The former is small enough to enumerate;
the latter is searched with the canonical count-grid dynamic program.

All costs are minimized. Search and multi-array joins retain a raw component
vector until every target array has been selected. This is important because
``J_peak`` and ``J_area`` are computed after component scores have been summed
across separate allocations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, isfinite
from time import perf_counter
from typing import Literal, Mapping, Sequence, TypeVar

from .layouts import CanonicalLayout, Layout, row_major_layout
from .model import EventSequence, MatrixSpec, MemoryEvent
from .objectives import ObjectiveComponent, ObjectiveSpec, build_objectives
from .scoring import (
    LayoutScore,
    normalized_excess,
    score_layouts,
    weighted_component_region_count,
)
from .search import LayoutSeed, ScorePolicy, SearchStats, search_canonical


Grammar = Literal["standard", "canonical"]
FrontierType = Literal["pareto", "fine-gated"]
FRONTIER_OBJECTIVES = (
    "fine-region-count",
    "peak-normalized-excess",
    "weighted-normalized-excess",
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
    frontier_type: FrontierType = "pareto"
    fine_component: str = "wave_load.64B"
    fine_tolerance: float = 0.05
    name: str = "simple_relay_problem"


@dataclass(frozen=True)
class FrontierCost:
    """The notes-aligned costs for one realized joint layout."""

    fine_region_count: float
    peak_normalized_excess: float
    weighted_normalized_excess: float
    codegen_runs: int
    codegen_xors: int

    @property
    def values(self) -> tuple[float, ...]:
        return (
            self.fine_region_count,
            self.peak_normalized_excess,
            self.weighted_normalized_excess,
            float(self.codegen_runs),
            float(self.codegen_xors),
        )


@dataclass(frozen=True)
class FrontierMember:
    """One distinct layout mapping retained by the analytical frontier."""

    layouts: Mapping[str, CanonicalLayout]
    score: LayoutScore
    cost: FrontierCost

    def word_signature(
        self, matrices: Mapping[str, MatrixSpec]
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, self.layouts[name].word_string(matrices[name]))
            for name in self.layouts
        )


@dataclass(frozen=True)
class ArraySearchResult:
    """Search size and retained raw frontier for one target array."""

    matrix: str
    grammar_layout_count: int
    raw_frontier_count: int
    search_stats: SearchStats | None = None


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
    layout: CanonicalLayout
    raw_scores: Mapping[str, float]


@dataclass(frozen=True)
class _JointCandidate:
    layouts: Mapping[str, CanonicalLayout]
    raw_scores: Mapping[str, float]


@dataclass(frozen=True)
class _CostedJoint:
    candidate: _JointCandidate
    cost: FrontierCost


_T = TypeVar("_T")


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _pareto(items: Sequence[_T], key) -> list[_T]:
    """Return a deterministic strict-dominance frontier, retaining ties."""

    frontier: list[_T] = []
    for item in sorted(items, key=key):
        values = key(item)
        if any(_dominates(key(other), values) for other in frontier):
            continue
        frontier = [
            other
            for other in frontier
            if not _dominates(values, key(other))
        ]
        frontier.append(item)
    return frontier


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
    if problem.grammar not in ("standard", "canonical"):
        raise ValueError(f"unknown grammar {problem.grammar!r}")
    if problem.frontier_type not in ("pareto", "fine-gated"):
        raise ValueError(f"unknown frontier type {problem.frontier_type!r}")
    if not isfinite(problem.fine_tolerance) or problem.fine_tolerance < 0:
        raise ValueError("fine_tolerance must be finite and nonnegative")
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
        )
    return frontier


def _member_cost(score: LayoutScore, fine_component: str) -> FrontierCost:
    return FrontierCost(
        fine_region_count=score.component(fine_component).raw_region_count,
        peak_normalized_excess=score.peak_normalized_excess,
        weighted_normalized_excess=score.weighted_normalized_excess,
        codegen_runs=score.codegen.runs,
        codegen_xors=score.codegen.xors,
    )


def _raw_cost(
    raw_scores: Mapping[str, float],
    matrices: Sequence[MatrixSpec],
    components: Sequence[ObjectiveComponent],
    weights: Mapping[str, float],
    fine_component: str,
) -> FrontierCost:
    active = [component for component in components if weights[component.name] > 0]
    excesses = {
        component.name: normalized_excess(
            raw_scores.get(component.name, 0.0),
            sum(
                component.packing_bound(matrix)
                for matrix in matrices
                if component.edges_by_array.get(matrix.name)
            ),
        )
        for component in active
    }
    return FrontierCost(
        fine_region_count=float(raw_scores.get(fine_component, 0.0)),
        peak_normalized_excess=max(excesses.values(), default=0.0),
        weighted_normalized_excess=sum(
            weights[name] * value for name, value in excesses.items()
        ),
        codegen_runs=int(raw_scores.get("runs", 0.0)),
        codegen_xors=int(raw_scores.get("xors", 0.0)),
    )


def _final_frontier(
    items: Sequence[_T],
    frontier_type: FrontierType,
    fine_tolerance: float,
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
    return _pareto(eligible, objectives)


def simple_solve(problem: SimpleRelayProblem) -> SimpleRelayResult:
    """Return the exact ``G_S`` or ``G_C`` joint layout frontier.

    ``frontier_type='pareto'`` returns the ordinary five-cost Pareto frontier.
    ``frontier_type='fine-gated'`` first restricts layouts to
    ``Q_fine <= (1 + fine_tolerance) Q_fine*`` and then Pareto-filters over
    ``(J_peak, J_area, runs, xors)``. Exact score ties remain as distinct
    layouts because hardware performance may distinguish them.
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
    weights = _effective_weights(components, problem.component_weights)
    raw_names = {
        problem.fine_component,
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
        else:
            candidates, search_result = _canonical_array_frontier(
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
    )
    costed = [
        _CostedJoint(
            candidate,
            _raw_cost(
                candidate.raw_scores,
                problem.matrices,
                components,
                weights,
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
    )
    members: list[FrontierMember] = []
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
        )
        members.append(
            FrontierMember(
                layouts=dict(candidate.layouts),
                score=score,
                cost=_member_cost(score, problem.fine_component),
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
