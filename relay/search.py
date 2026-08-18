from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Iterable, Mapping, Sequence

from .gf2 import (
    codimension_one_subspaces,
    complement_basis,
    coordinate_map,
    enumerate_subspace_layers,
    invert_matrix_from_columns,
    lift_coordinate,
    new_direction,
    reduce_vector,
    rref_basis,
)
from .layouts import CanonicalLayout, LinearInnerLayout, Layout
from .model import MatrixSpec
from .objectives import Hyperedge, ObjectiveComponent


Score = dict[str, float]


@dataclass(frozen=True)
class ScorePolicy:
    """How paths and final candidates are ordered before hardware measurement."""

    kind: str = "lexicographic"
    order: tuple[str, ...] = ()
    weights: Mapping[str, float] = field(default_factory=dict)
    paths_per_state: int = 8
    frontier_limit: int = 32

    def key(self, score: Mapping[str, float]) -> tuple[float, ...]:
        values = tuple(float(score.get(name, 0.0)) for name in self.order)
        if self.kind == "weighted":
            total = sum(float(self.weights.get(name, 0.0)) * float(score.get(name, 0.0)) for name in self.order)
            return (total, *values)
        if self.kind in {"lexicographic", "pareto"}:
            return values
        raise ValueError(f"unknown score policy kind {self.kind!r}")

    def dominates(self, left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        values_left = [float(left.get(name, 0.0)) for name in self.order]
        values_right = [float(right.get(name, 0.0)) for name in self.order]
        return all(a <= b for a, b in zip(values_left, values_right)) and any(
            a < b for a, b in zip(values_left, values_right)
        )


@dataclass
class SearchStats:
    grammar: str
    tile_exponents: tuple[int, ...]
    states: int = 0
    transitions: int = 0
    paths_considered: int = 0
    paths_retained: int = 0
    active_rank: int | None = None
    exact: bool = True
    truncated: bool = False
    note: str = ""


@dataclass(frozen=True)
class LayoutSeed:
    layout: Layout
    search_scores: Mapping[str, float]
    exact: bool
    search_stats: SearchStats
    note: str = ""


@dataclass(frozen=True)
class _CanonicalPath:
    counts: tuple[int, ...]
    last_mode: int | None
    word: tuple[int, ...]
    score: Mapping[str, float]


@dataclass(frozen=True)
class _GeneralPath:
    chain: tuple[tuple[int, ...], ...]
    score: Mapping[str, float]


def add_scores(left: Mapping[str, float], right: Mapping[str, float]) -> Score:
    result = dict(left)
    for name, value in right.items():
        result[name] = result.get(name, 0.0) + float(value)
    return result


def _prune_paths(paths: Iterable[object], policy: ScorePolicy, score_getter) -> tuple[list[object], bool]:
    unique: dict[tuple[object, ...], object] = {}
    for path in paths:
        signature = getattr(path, "word", getattr(path, "chain", ()))
        key = tuple(signature)
        incumbent = unique.get(key)
        if incumbent is None or policy.key(score_getter(path)) < policy.key(score_getter(incumbent)):
            unique[key] = path
    values = list(unique.values())
    truncated = False
    if policy.kind == "pareto":
        frontier: list[object] = []
        for path in sorted(values, key=lambda item: policy.key(score_getter(item))):
            score = score_getter(path)
            if any(policy.dominates(score_getter(other), score) for other in frontier):
                continue
            frontier = [other for other in frontier if not policy.dominates(score, score_getter(other))]
            frontier.append(path)
        values = sorted(frontier, key=lambda item: policy.key(score_getter(item)))
        if len(values) > policy.frontier_limit:
            truncated = True
            values = values[: policy.frontier_limit]
    else:
        values.sort(key=lambda item: policy.key(score_getter(item)))
        if len(values) > policy.paths_per_state:
            truncated = True
            values = values[: policy.paths_per_state]
    return values, truncated


def _component_dimensions(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    tile_bits: int,
) -> dict[int, list[ObjectiveComponent]]:
    result: dict[int, list[ObjectiveComponent]] = {}
    for component in components:
        if not component.search or not component.edges_by_array.get(matrix.name):
            continue
        try:
            dimension = component.dimension(matrix)
        except ValueError:
            continue
        if dimension <= tile_bits:
            result.setdefault(dimension, []).append(component)
    return result


def _canonical_edge_regions(
    edge: Hyperedge,
    matrix: MatrixSpec,
    tile_exponents: Sequence[int],
    prefix_counts: Sequence[int],
) -> int:
    seen: set[tuple[object, ...]] = set()
    masks = tuple((1 << exponent) - 1 for exponent in tile_exponents)
    for coord in edge.points:
        outer = matrix.outer_coord(coord, tile_exponents)
        remainder = tuple(
            ((value & mask) >> used)
            for value, mask, used in zip(coord, masks, prefix_counts)
        )
        seen.add((*outer, *remainder))
    return len(seen)


def _canonical_node_score(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    tile_exponents: Sequence[int],
    counts: Sequence[int],
) -> Score:
    score: Score = {}
    for component in components:
        total = 0.0
        for edge in component.edges_by_array.get(matrix.name, ()):
            total += edge.weight * _canonical_edge_regions(edge, matrix, tile_exponents, counts)
        score[component.name] = total
    return score


def search_canonical(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    tile_exponents: tuple[int, ...],
    outer_orders: Sequence[tuple[int, ...]],
    policy: ScorePolicy,
    *,
    candidates_per_tile: int = 8,
    stats_sink: list[SearchStats] | None = None,
) -> list[LayoutSeed]:
    tile_bits = sum(tile_exponents)
    if tile_bits == 0:
        return []
    dimensions = _component_dimensions(matrix, components, tile_bits)
    stats = SearchStats("canonical", tile_exponents)
    node_cache: dict[tuple[int, ...], Score] = {}

    def cached_node_score(counts: tuple[int, ...]) -> Score:
        if counts not in node_cache:
            node_cache[counts] = _canonical_node_score(
                matrix, dimensions.get(sum(counts), ()), tile_exponents, counts
            )
        return node_cache[counts]

    zero = tuple(0 for _ in tile_exponents)
    initial_score = dict(cached_node_score(zero))
    initial_score["runs"] = 0.0
    initial_score["xors"] = 0.0
    layer: dict[tuple[tuple[int, ...], int | None], list[_CanonicalPath]] = {
        (zero, None): [_CanonicalPath(zero, None, (), initial_score)]
    }
    stats.states = 1

    for dimension in range(tile_bits):
        pending: dict[tuple[tuple[int, ...], int], list[_CanonicalPath]] = {}
        for (_, _), paths in layer.items():
            for path in paths:
                for mode, limit in enumerate(tile_exponents):
                    if path.counts[mode] >= limit:
                        continue
                    counts = list(path.counts)
                    counts[mode] += 1
                    next_counts = tuple(counts)
                    local = cached_node_score(next_counts)
                    score = add_scores(path.score, local)
                    if path.last_mode is None or path.last_mode != mode:
                        score["runs"] = score.get("runs", 0.0) + 1.0
                    next_path = _CanonicalPath(
                        next_counts,
                        mode,
                        (*path.word, mode),
                        score,
                    )
                    pending.setdefault((next_counts, mode), []).append(next_path)
                    stats.transitions += 1
                    stats.paths_considered += 1
        layer = {}
        for state, paths in pending.items():
            retained, truncated = _prune_paths(paths, policy, lambda item: item.score)
            stats.truncated |= truncated
            layer[state] = retained  # type: ignore[assignment]
            stats.paths_retained += len(retained)
        stats.states += len(layer)

    terminal: list[_CanonicalPath] = []
    for paths in layer.values():
        terminal.extend(paths)
    terminal, truncated = _prune_paths(terminal, policy, lambda item: item.score)
    stats.truncated |= truncated
    terminal = terminal[:candidates_per_tile]
    stats.exact = policy.kind != "pareto" or not stats.truncated

    seeds: list[LayoutSeed] = []
    shape = "x".join(str(1 << exponent) for exponent in tile_exponents)
    for path in terminal:
        word_text = "".join(matrix.mode_names[mode] for mode in path.word)
        for outer_order in outer_orders:
            outer_text = "".join(matrix.mode_names[mode] for mode in outer_order)
            name = f"c{shape}_{word_text}_outer{outer_text}"
            layout = CanonicalLayout(name, matrix.name, tile_exponents, path.word, outer_order)
            layout.validate(matrix)
            seeds.append(
                LayoutSeed(
                    layout,
                    dict(path.score),
                    stats.exact,
                    stats,
                    "exact canonical grid DP" if stats.exact else "canonical DP with capped alternatives",
                )
            )
    if stats_sink is not None:
        stats_sink.append(stats)
    return seeds


@dataclass(frozen=True)
class _ProjectedEdge:
    weight: float
    fragments: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _ProjectedComponent:
    name: str
    dimension: int
    edges: tuple[_ProjectedEdge, ...]


def _build_projected_problem(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    tile_exponents: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[_ProjectedComponent, ...]]:
    tile_bits = sum(tile_exponents)
    raw_components: list[tuple[ObjectiveComponent, list[list[int]]]] = []
    all_differences: list[int] = []

    for component in components:
        if not component.search or not component.edges_by_array.get(matrix.name):
            continue
        try:
            dimension = component.dimension(matrix)
        except ValueError:
            continue
        if dimension > tile_bits:
            continue
        component_fragments: list[list[int]] = []
        for edge in component.edges_by_array[matrix.name]:
            by_outer: dict[tuple[int, ...], list[int]] = {}
            for coord in edge.points:
                outer = matrix.outer_coord(coord, tile_exponents)
                by_outer.setdefault(outer, []).append(matrix.inner_bits(coord, tile_exponents))
            flattened: list[int] = []
            for points in by_outer.values():
                unique = sorted(set(points))
                if not unique:
                    continue
                anchor = unique[0]
                for point in unique[1:]:
                    all_differences.append(point ^ anchor)
                # Store one fragment after another; a sentinel -1 separates fragments.
                if flattened:
                    flattened.append(-1)
                flattened.extend(unique)
            component_fragments.append(flattened)
        raw_components.append((component, component_fragments))

    active_basis = rref_basis(all_differences)
    coords = coordinate_map(active_basis)
    projected: list[_ProjectedComponent] = []
    for component, flattened_edges in raw_components:
        edges: list[_ProjectedEdge] = []
        source_edges = component.edges_by_array[matrix.name]
        for edge, flattened in zip(source_edges, flattened_edges):
            fragments: list[tuple[int, ...]] = []
            current: list[int] = []
            anchor: int | None = None
            for point in [*flattened, -1]:
                if point == -1:
                    if current:
                        fragments.append(tuple(current))
                    current = []
                    anchor = None
                    continue
                if anchor is None:
                    anchor = point
                difference = point ^ anchor
                if difference not in coords:
                    raise AssertionError("active-span projection failed")
                current.append(coords[difference])
            edges.append(_ProjectedEdge(edge.weight, tuple(fragments)))
        projected.append(_ProjectedComponent(component.name, component.dimension(matrix), tuple(edges)))
    return active_basis, tuple(projected)


def _projected_component_score(component: _ProjectedComponent, subspace: Sequence[int]) -> float:
    total = 0.0
    for edge in component.edges:
        regions = 0
        for fragment in edge.fragments:
            regions += len({reduce_vector(point, subspace) for point in fragment})
        total += edge.weight * regions
    return total


def search_linear_inner(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    tile_exponents: tuple[int, ...],
    outer_orders: Sequence[tuple[int, ...]],
    policy: ScorePolicy,
    *,
    exact_rank_limit: int = 7,
    candidates_per_tile: int = 4,
    stats_sink: list[SearchStats] | None = None,
) -> list[LayoutSeed]:
    tile_bits = sum(tile_exponents)
    if tile_bits == 0:
        return []
    active_basis, projected = _build_projected_problem(matrix, components, tile_exponents)
    active_rank = len(active_basis)
    stats = SearchStats("linear_inner", tile_exponents, active_rank=active_rank)
    if active_rank > exact_rank_limit:
        stats.exact = False
        stats.note = f"active rank {active_rank} exceeds exact limit {exact_rank_limit}"
        if stats_sink is not None:
            stats_sink.append(stats)
        return []

    if active_rank == 0:
        columns = complement_basis((), tile_bits)
        a_rows = invert_matrix_from_columns(columns, tile_bits)
        seeds: list[LayoutSeed] = []
        for outer_order in outer_orders:
            layout = LinearInnerLayout(
                f"lin{'x'.join(str(1 << e) for e in tile_exponents)}_inactive",
                matrix.name,
                tile_exponents,
                a_rows,
                outer_order,
                columns,
                0,
            )
            seeds.append(LayoutSeed(layout, {"runs": float(layout.runs), "xors": float(layout.xor_count)}, True, stats))
        if stats_sink is not None:
            stats_sink.append(stats)
        return seeds

    layers = enumerate_subspace_layers(active_rank)
    stats.states = sum(len(layer) for layer in layers)
    by_dimension: dict[int, list[_ProjectedComponent]] = {}
    for component in projected:
        by_dimension.setdefault(component.dimension, []).append(component)

    node_cache: dict[tuple[int, tuple[int, ...]], Score] = {}

    def node_score(dimension: int, subspace: Sequence[int]) -> Score:
        canonical = tuple(subspace)
        key = (dimension, canonical)
        if key not in node_cache:
            result: Score = {}
            for component in by_dimension.get(dimension, ()):
                result[component.name] = _projected_component_score(component, canonical)
            node_cache[key] = result
        return node_cache[key]

    initial = add_scores(node_score(0, ()), {"runs": 0.0, "xors": 0.0})
    dp: dict[tuple[int, ...], list[_GeneralPath]] = {(): [_GeneralPath(((),), initial)]}

    for dimension in range(1, active_rank + 1):
        next_dp: dict[tuple[int, ...], list[_GeneralPath]] = {}
        for subspace in layers[dimension]:
            candidates: list[_GeneralPath] = []
            for predecessor in codimension_one_subspaces(subspace):
                for path in dp.get(predecessor, ()):
                    score = add_scores(path.score, node_score(dimension, subspace))
                    candidates.append(_GeneralPath((*path.chain, subspace), score))
                    stats.transitions += 1
                    stats.paths_considered += 1
            retained, truncated = _prune_paths(candidates, policy, lambda item: item.score)
            stats.truncated |= truncated
            next_dp[subspace] = retained  # type: ignore[assignment]
            stats.paths_retained += len(retained)
        dp = next_dp

    terminal_paths: list[_GeneralPath] = []
    for paths in dp.values():
        terminal_paths.extend(paths)

    # Components above the active rank are constant once the whole active span is low-address.
    full_subspace = next(iter(dp))
    completed: list[_GeneralPath] = []
    for path in terminal_paths:
        score = dict(path.score)
        for dimension, component_list in by_dimension.items():
            if dimension > active_rank:
                for component in component_list:
                    score[component.name] = _projected_component_score(component, full_subspace)
        completed.append(_GeneralPath(path.chain, score))
    completed, truncated = _prune_paths(completed, policy, lambda item: item.score)
    stats.truncated |= truncated
    completed = completed[:candidates_per_tile]
    stats.exact = not stats.truncated or policy.kind != "pareto"

    seeds: list[LayoutSeed] = []
    shape = "x".join(str(1 << exponent) for exponent in tile_exponents)
    for index, path in enumerate(completed):
        active_columns: list[int] = []
        previous: tuple[int, ...] = ()
        for subspace in path.chain[1:]:
            direction_coordinate = new_direction(previous, subspace)
            active_columns.append(lift_coordinate(direction_coordinate, active_basis))
            previous = subspace
        columns = tuple(active_columns) + complement_basis(active_columns, tile_bits)
        a_rows = invert_matrix_from_columns(columns, tile_bits)
        for outer_order in outer_orders:
            outer_text = "".join(matrix.mode_names[mode] for mode in outer_order)
            layout = LinearInnerLayout(
                f"lin{shape}_flag{index}_outer{outer_text}",
                matrix.name,
                tile_exponents,
                a_rows,
                outer_order,
                columns,
                active_rank,
            )
            layout.validate(matrix)
            score = dict(path.score)
            score["runs"] = float(layout.runs)
            score["xors"] = float(layout.xor_count)
            seeds.append(
                LayoutSeed(
                    layout,
                    score,
                    stats.exact,
                    stats,
                    f"exact cover-edge DP after active-span reduction to rank {active_rank}",
                )
            )
    seeds.sort(key=lambda seed: policy.key(seed.search_scores))
    if stats_sink is not None:
        stats_sink.append(stats)
    return seeds
