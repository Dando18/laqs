from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from time import perf_counter
from typing import Iterable, Mapping, Sequence

from .layouts import (
    CanonicalLayout,
    Layout,
    column_major_layout,
    layout_codegen_runs,
    row_major_layout,
    tiled_row_major_layout,
)
from .model import EventSequence, MatrixSpec, MemoryEvent, exact_log2, is_power_of_two
from .objectives import ObjectiveComponent, ObjectiveSpec, build_objectives
from .scoring import weighted_component_region_count
from .search import LayoutSeed, ScorePolicy, SearchStats, search_canonical, search_linear_inner


@dataclass(frozen=True)
class SolverConfig:
    policy: ScorePolicy = field(default_factory=ScorePolicy)
    tile_shapes: Mapping[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    general_tile_shapes: Mapping[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    outer_orders: Mapping[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    candidate_layouts: Mapping[str, tuple[Layout, ...]] = field(default_factory=dict)
    max_inner_bits: int = 8
    max_global_canonical_bits: int = 32
    include_global_canonical: bool = True
    max_tile_hypotheses: int = 128
    canonical_candidates_per_tile: int = 4
    enable_linear_inner: bool = True
    include_row_major_control: bool = True
    include_column_major_control: bool = True
    include_tiled_row_major_control: bool = False
    retain_one_candidate_per_tile: bool = False
    general_max_inner_bits: int = 8
    general_exact_rank: int = 7
    general_candidates_per_tile: int = 2
    general_policy: ScorePolicy | None = None
    primary_tolerance: float | None = 0.05
    per_array_candidates: int = 16
    joint_beam_width: int = 128
    joint_candidates: int = 16
    exhaustive_inner_validation_bits: int = 14


@dataclass(frozen=True)
class RelayProblem:
    matrices: tuple[MatrixSpec, ...]
    events: tuple[MemoryEvent, ...]
    sequences: tuple[EventSequence, ...]
    objectives: tuple[ObjectiveSpec, ...]
    config: SolverConfig = field(default_factory=SolverConfig)
    name: str = "relay_problem"


@dataclass(frozen=True)
class Candidate:
    matrix: str
    layout: Layout
    scores: Mapping[str, float]
    packing_bounds: Mapping[str, float]
    search_scores: Mapping[str, float]
    exact: bool
    note: str
    search_stats: SearchStats

    @property
    def grammar(self) -> str:
        return self.layout.grammar


@dataclass(frozen=True)
class ArrayResult:
    matrix: MatrixSpec
    candidates: tuple[Candidate, ...]
    all_candidate_count: int
    tile_hypotheses: tuple[tuple[int, ...], ...]
    search_stats: tuple[SearchStats, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class JointCandidate:
    layouts: Mapping[str, Candidate]
    scores: Mapping[str, float]


@dataclass(frozen=True)
class RelayResult:
    problem: RelayProblem
    components: tuple[ObjectiveComponent, ...]
    arrays: Mapping[str, ArrayResult]
    joint_candidates: tuple[JointCandidate, ...]
    context_layouts: Mapping[str, Layout]
    elapsed_seconds: float


def _validate_problem(problem: RelayProblem) -> tuple[dict[str, MatrixSpec], dict[str, MemoryEvent]]:
    if not problem.matrices:
        raise ValueError("the problem contains no matrices")
    matrices = {matrix.name: matrix for matrix in problem.matrices}
    if len(matrices) != len(problem.matrices):
        raise ValueError("matrix names must be unique")
    events = {event.id: event for event in problem.events}
    if len(events) != len(problem.events):
        raise ValueError("event ids must be unique")
    for event in problem.events:
        for access in event.accesses:
            if access.array not in matrices:
                raise ValueError(f"event {event.id}: unknown array {access.array}")
            matrix = matrices[access.array]
            matrix.validate_coord(access.coord)
            width = matrix.element_bytes if access.width_bytes is None else access.width_bytes
            if width != matrix.element_bytes:
                raise ValueError(
                    f"event {event.id}: the first implementation supports scalar accesses only; "
                    f"{matrix.name} uses {matrix.element_bytes} B elements but access width is {width} B"
                )
    for sequence in problem.sequences:
        for event_id in sequence.event_ids:
            if event_id not in events:
                raise ValueError(f"sequence {sequence.name}: unknown event {event_id}")
    return matrices, events


def _shape_to_exponents(matrix: MatrixSpec, shape: Sequence[int]) -> tuple[int, ...]:
    if len(shape) != matrix.rank:
        raise ValueError(f"{matrix.name}: tile shape {tuple(shape)} has the wrong rank")
    exponents: list[int] = []
    for tile_extent, matrix_extent in zip(shape, matrix.shape):
        if not is_power_of_two(tile_extent) or tile_extent > matrix_extent:
            raise ValueError(
                f"{matrix.name}: tile extent {tile_extent} must be a power of two <= {matrix_extent}"
            )
        exponents.append(exact_log2(tile_extent))
    return tuple(exponents)


def _auto_tile_exponents(matrix: MatrixSpec, config: SolverConfig) -> tuple[tuple[int, ...], ...]:
    has_explicit = matrix.name in config.tile_shapes
    explicit = config.tile_shapes.get(matrix.name, ())
    result: set[tuple[int, ...]] = set()
    if has_explicit:
        result.update(_shape_to_exponents(matrix, shape) for shape in explicit)
    else:
        ranges = [range(bits + 1) for bits in matrix.mode_bits]
        for exponents in product(*ranges):
            total = sum(exponents)
            if 0 < total <= config.max_inner_bits:
                result.add(tuple(exponents))
    if config.include_global_canonical and matrix.total_bits <= config.max_global_canonical_bits:
        result.add(matrix.mode_bits)
    ordered = sorted(result, key=lambda value: (sum(value), value))
    if len(ordered) > config.max_tile_hypotheses:
        ordered = ordered[: config.max_tile_hypotheses]
    return tuple(ordered)


def _general_tile_exponents(
    matrix: MatrixSpec,
    canonical_tiles: Sequence[tuple[int, ...]],
    config: SolverConfig,
) -> tuple[tuple[int, ...], ...]:
    if matrix.name in config.general_tile_shapes:
        explicit = config.general_tile_shapes[matrix.name]
        values = {_shape_to_exponents(matrix, shape) for shape in explicit}
    else:
        values = {tile for tile in canonical_tiles if sum(tile) <= config.general_max_inner_bits}
    return tuple(sorted(values, key=lambda value: (sum(value), value)))


def _outer_orders(matrix: MatrixSpec, config: SolverConfig) -> tuple[tuple[int, ...], ...]:
    explicit = config.outer_orders.get(matrix.name)
    if not explicit:
        return (tuple(reversed(range(matrix.rank))),)
    index = {name: position for position, name in enumerate(matrix.mode_names)}
    result: list[tuple[int, ...]] = []
    for order in explicit:
        try:
            converted = tuple(index[name] for name in order)
        except KeyError as error:
            raise ValueError(f"{matrix.name}: unknown mode in outer order {order}") from error
        if tuple(sorted(converted)) != tuple(range(matrix.rank)):
            raise ValueError(f"{matrix.name}: outer order {order} is not a permutation")
        result.append(converted)
    return tuple(result)


def _lane_metrics(
    matrix: MatrixSpec,
    layout: Layout,
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
) -> dict[str, float]:
    gap = 0.0
    breaks = 0.0
    pairs = 0.0
    max_gap = 0.0
    if sequences:
        event_occurrences = (
            (events[event_id], sequence.weight)
            for sequence in sequences
            for event_id in sequence.event_ids
        )
    else:
        event_occurrences = ((event, 1.0) for event in events.values())

    for event, sequence_weight in event_occurrences:
        accesses = [
            access
            for access in event.accesses
            if access.array == matrix.name and access.lane is not None
        ]
        accesses.sort(key=lambda access: int(access.lane))
        event_weight = event.weight * sequence_weight
        for left, right in zip(accesses, accesses[1:]):
            if right.lane == left.lane:
                continue
            delta = abs(
                layout.offset(matrix, right.coord)
                - layout.offset(matrix, left.coord)
            )
            gap += event_weight * delta
            breaks += event_weight * float(delta != 1)
            pairs += event_weight
            max_gap = max(max_gap, float(delta))
    return {
        "adj_gap": gap,
        "adj_breaks": breaks,
        "adj_pairs": pairs,
        "max_adj_gap": max_gap,
    }


def _validate_layout(matrix: MatrixSpec, layout: Layout, max_bits: int) -> None:
    if isinstance(layout, CanonicalLayout):
        layout.validate(matrix)
    else:
        layout.validate(matrix)  # type: ignore[attr-defined]
    bits = sum(layout.tile_exponents)
    if bits > max_bits:
        return
    shape = tuple(1 << exponent for exponent in layout.tile_exponents)
    seen: set[int] = set()
    for coord in product(*(range(extent) for extent in shape)):
        inner = layout.offset(matrix, tuple(coord))
        # This is tile zero, so the complete offset equals the inner offset.
        if inner in seen:
            raise ValueError(f"layout {layout.name} is not injective inside its tile")
        seen.add(inner)
    if len(seen) != 1 << bits:
        raise ValueError(f"layout {layout.name} does not cover its inner tile")


def _candidate_from_seed(
    matrix: MatrixSpec,
    seed: LayoutSeed,
    components: Sequence[ObjectiveComponent],
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
    config: SolverConfig,
) -> Candidate:
    _validate_layout(matrix, seed.layout, config.exhaustive_inner_validation_bits)
    direct = {
        component.name: weighted_component_region_count(
            matrix, seed.layout, component
        )
        for component in components
        if component.edges_by_array.get(matrix.name)
    }
    bounds = {
        component.name: component.packing_bound(matrix)
        for component in components
        if component.edges_by_array.get(matrix.name)
    }
    metrics = _lane_metrics(matrix, seed.layout, events, sequences)
    scores = {
        **direct,
        **metrics,
        "runs": float(layout_codegen_runs(matrix, seed.layout)),
        "xors": float(seed.layout.xor_count),
    }
    return Candidate(
        matrix.name,
        seed.layout,
        scores,
        bounds,
        dict(seed.search_scores),
        seed.exact,
        seed.note,
        seed.search_stats,
    )


def _control_seed(layout: Layout, tile_exponents: tuple[int, ...]) -> LayoutSeed:
    stats = SearchStats("control", tile_exponents, exact=True, note="mandatory control")
    return LayoutSeed(layout, {"runs": float(layout.runs), "xors": float(layout.xor_count)}, True, stats, "standard control")


def _provided_seed(layout: Layout) -> LayoutSeed:
    stats = SearchStats(
        "provided",
        layout.tile_exponents,
        exact=True,
        note="provided candidate",
    )
    return LayoutSeed(
        layout,
        {"runs": float(layout.runs), "xors": float(layout.xor_count)},
        True,
        stats,
        "provided candidate",
    )


def _effective_policy(config: SolverConfig, components: Sequence[ObjectiveComponent]) -> ScorePolicy:
    policy = config.policy
    if policy.order:
        return policy
    order = tuple(component.name for component in components) + (
        "adj_breaks",
        "runs",
        "xors",
        "adj_gap",
    )
    return ScorePolicy(
        kind=policy.kind,
        order=order,
        weights=policy.weights,
        tie_order=policy.tie_order,
        paths_per_state=policy.paths_per_state,
        frontier_limit=policy.frontier_limit,
    )


def _pareto_frontier(candidates: Sequence[Candidate], policy: ScorePolicy) -> list[Candidate]:
    frontier: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: policy.key(item.scores)):
        if any(policy.dominates(other.scores, candidate.scores) for other in frontier):
            continue
        frontier = [other for other in frontier if not policy.dominates(candidate.scores, other.scores)]
        frontier.append(candidate)
    return sorted(frontier, key=lambda item: policy.key(item.scores))


def _select_array_candidates(
    candidates: Sequence[Candidate],
    policy: ScorePolicy,
    config: SolverConfig,
) -> tuple[Candidate, ...]:
    ordered = sorted(candidates, key=lambda candidate: policy.key(candidate.scores))
    if not ordered:
        return ()
    pool = ordered
    if config.primary_tolerance is not None and policy.order:
        primary = policy.order[0]
        best = float(ordered[0].scores.get(primary, 0.0))
        if best > 0:
            pool = [
                candidate
                for candidate in ordered
                if float(candidate.scores.get(primary, 0.0)) <= best * (1.0 + config.primary_tolerance)
            ]
    if policy.kind == "pareto":
        selected = _pareto_frontier(pool, policy)
    else:
        selected = list(pool)

    mandatory: list[Candidate] = []
    if config.retain_one_candidate_per_tile:
        seen_tiles: set[tuple[int, ...]] = set()
        for candidate in ordered:
            if candidate.search_stats.grammar != "canonical":
                continue
            tile = candidate.layout.tile_exponents
            if tile not in seen_tiles:
                mandatory.append(candidate)
                seen_tiles.add(tile)
    for grammar in ("canonical", "linear_inner"):
        matches = [candidate for candidate in ordered if candidate.grammar == grammar]
        if matches:
            mandatory.append(matches[0])
    for name in ("row_major", "column_major", "tiled_row_major"):
        mandatory.extend(
            candidate
            for candidate in ordered
            if candidate.layout.name == name
        )

    merged: list[Candidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in [*selected, *mandatory]:
        signature = candidate.layout.signature()
        if signature not in seen:
            seen.add(signature)
            merged.append(candidate)
    merged.sort(key=lambda candidate: policy.key(candidate.scores))
    if len(merged) <= config.per_array_candidates:
        return tuple(merged)

    keep: list[Candidate] = []
    mandatory_signatures = {candidate.layout.signature() for candidate in mandatory}
    for candidate in merged:
        if candidate.layout.signature() in mandatory_signatures:
            keep.append(candidate)
    for candidate in merged:
        if len(keep) >= config.per_array_candidates:
            break
        if candidate not in keep:
            keep.append(candidate)
    keep.sort(key=lambda candidate: policy.key(candidate.scores))
    return tuple(keep)


def _sum_score_maps(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    result = dict(left)
    for name, value in right.items():
        result[name] = result.get(name, 0.0) + float(value)
    return result


def _joint_candidates(
    array_results: Mapping[str, ArrayResult],
    context_scores: Mapping[str, float],
    policy: ScorePolicy,
    config: SolverConfig,
) -> tuple[JointCandidate, ...]:
    target_names = tuple(array_results)
    beam: list[JointCandidate] = [JointCandidate({}, dict(context_scores))]
    for name in target_names:
        expanded: list[JointCandidate] = []
        for partial in beam:
            for candidate in array_results[name].candidates:
                layouts = dict(partial.layouts)
                layouts[name] = candidate
                scores = _sum_score_maps(partial.scores, candidate.scores)
                expanded.append(JointCandidate(layouts, scores))
        if policy.kind == "pareto":
            frontier: list[JointCandidate] = []
            for candidate in sorted(expanded, key=lambda item: policy.key(item.scores)):
                if any(policy.dominates(other.scores, candidate.scores) for other in frontier):
                    continue
                frontier = [other for other in frontier if not policy.dominates(candidate.scores, other.scores)]
                frontier.append(candidate)
            beam = sorted(frontier, key=lambda item: policy.key(item.scores))[: config.joint_beam_width]
        else:
            beam = sorted(expanded, key=lambda item: policy.key(item.scores))[: config.joint_beam_width]
    return tuple(sorted(beam, key=lambda item: policy.key(item.scores))[: config.joint_candidates])


def solve(problem: RelayProblem) -> RelayResult:
    start = perf_counter()
    matrices, events = _validate_problem(problem)
    components = build_objectives(problem.objectives, matrices, events, problem.sequences)
    policy = _effective_policy(problem.config, components)
    config = problem.config
    unknown_candidate_matrices = sorted(
        set(config.candidate_layouts) - set(matrices)
    )
    if unknown_candidate_matrices:
        raise ValueError(
            "provided candidate layouts reference unknown matrices: "
            f"{unknown_candidate_matrices}"
        )

    context_layouts: dict[str, Layout] = {}
    context_scores: dict[str, float] = {}
    for matrix in problem.matrices:
        if matrix.target:
            continue
        layout = row_major_layout(matrix)
        context_layouts[matrix.name] = layout
        for component in components:
            if component.edges_by_array.get(matrix.name):
                context_scores[component.name] = context_scores.get(
                    component.name, 0.0
                ) + weighted_component_region_count(
                    matrix,
                    layout,
                    component,
                )
        context_scores = _sum_score_maps(
            context_scores,
            _lane_metrics(matrix, layout, events, problem.sequences),
        )
        context_scores["runs"] = context_scores.get(
            "runs", 0.0
        ) + layout_codegen_runs(matrix, layout)
        context_scores["xors"] = context_scores.get("xors", 0.0) + layout.xor_count

    array_results: dict[str, ArrayResult] = {}
    for matrix in problem.matrices:
        if not matrix.target:
            continue
        array_start = perf_counter()
        tile_exponents = _auto_tile_exponents(matrix, config)
        outer_orders = _outer_orders(matrix, config)
        seeds: list[LayoutSeed] = []
        search_stats: list[SearchStats] = []
        for tile in tile_exponents:
            seeds.extend(
                search_canonical(
                    matrix,
                    components,
                    tile,
                    outer_orders,
                    policy,
                    candidates_per_tile=config.canonical_candidates_per_tile,
                    stats_sink=search_stats,
                )
            )
        if config.enable_linear_inner:
            general_policy = config.general_policy
            if general_policy is None:
                if policy.kind == "pareto":
                    general_policy = ScorePolicy(
                        kind="lexicographic",
                        order=policy.order,
                        weights=policy.weights,
                        tie_order=policy.tie_order,
                        paths_per_state=min(policy.paths_per_state, 4),
                        frontier_limit=policy.frontier_limit,
                    )
                else:
                    general_policy = policy
            for tile in _general_tile_exponents(matrix, tile_exponents, config):
                seeds.extend(
                    search_linear_inner(
                        matrix,
                        components,
                        tile,
                        outer_orders,
                        general_policy,
                        exact_rank_limit=config.general_exact_rank,
                        candidates_per_tile=config.general_candidates_per_tile,
                        stats_sink=search_stats,
                    )
                )

        seeds.extend(
            _provided_seed(layout)
            for layout in config.candidate_layouts.get(matrix.name, ())
        )

        if config.include_tiled_row_major_control:
            for tile in tile_exponents:
                tiled = tiled_row_major_layout(matrix, tile)
                seeds.append(_control_seed(tiled, tile))
        if config.include_row_major_control:
            row = row_major_layout(matrix)
            seeds.append(_control_seed(row, row.tile_exponents))
        if config.include_column_major_control:
            column = column_major_layout(matrix)
            seeds.append(_control_seed(column, column.tile_exponents))

        deduplicated: dict[tuple[object, ...], LayoutSeed] = {}
        for seed in seeds:
            signature = seed.layout.signature()
            incumbent = deduplicated.get(signature)
            if incumbent is None or policy.key(seed.search_scores) < policy.key(incumbent.search_scores):
                deduplicated[signature] = seed

        candidates = [
            _candidate_from_seed(
                matrix,
                seed,
                components,
                events,
                problem.sequences,
                config,
            )
            for seed in deduplicated.values()
        ]
        selected = _select_array_candidates(candidates, policy, config)
        array_results[matrix.name] = ArrayResult(
            matrix,
            selected,
            len(candidates),
            tile_exponents,
            tuple(search_stats),
            perf_counter() - array_start,
        )

    joint = _joint_candidates(array_results, context_scores, policy, config)
    return RelayResult(
        problem,
        components,
        array_results,
        joint,
        context_layouts,
        perf_counter() - start,
    )
