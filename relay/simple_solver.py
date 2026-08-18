
from dataclasses import asdict, dataclass, field
from itertools import product
from time import perf_counter
from typing import Iterable, Literal, Mapping, Sequence

from .layouts import CanonicalLayout, Layout, column_major_layout, row_major_layout
from .model import EventSequence, MatrixSpec, MemoryEvent, exact_log2, is_power_of_two
from .objectives import ObjectiveComponent, ObjectiveSpec, build_objectives
from .search import LayoutSeed, ScorePolicy, SearchStats, search_canonical, search_linear_inner


@dataclass(frozen=True)
class SimpleRelayProblem:
    matrices: tuple[MatrixSpec, ...]
    events: tuple[MemoryEvent, ...]
    sequences: tuple[EventSequence, ...]
    objectives: tuple[ObjectiveSpec, ...]
    grammar: Literal["canonical", "standard"]
    

def collect_matrices(problem: SimpleRelayProblem) -> dict[str, MatrixSpec]:
    return {matrix.name: matrix for matrix in problem.matrices}

def collect_events(problem: SimpleRelayProblem) -> dict[str, MemoryEvent]:
    return {event.id: event for event in problem.events}

def describe_matrices(matrices: Mapping[str, MatrixSpec]) -> None:
    print("Matrix summary:")
    for matrix in matrices.values():
        shape = "x".join(str(extent) for extent in matrix.shape)
        modes = ",".join(matrix.mode_names)
        target = "yes" if matrix.target else "context"
        print(
            f"  {matrix.name}: shape={shape}, element_bytes={matrix.element_bytes}, "
            f"modes={modes}, target={target}, role={matrix.role}"
        )

def describe_events(events: Mapping[str, MemoryEvent]) -> None:
    print("Event summary:")
    if not events:
        print("  0 events")
        return

    by_site: dict[str, list[MemoryEvent]] = {}
    for event in events.values():
        by_site.setdefault(event.site, []).append(event)

    print(f"  {len(events)} events across {len(by_site)} sites")
    for site, site_events in by_site.items():
        accesses = [access for event in site_events for access in event.accesses]
        arrays = sorted({access.array for access in accesses})
        kinds = sorted({access.kind for access in accesses})
        lanes = {access.lane for access in accesses if access.lane is not None}
        lane_summary = str(len(lanes)) if lanes else "-"
        print(
            f"  {site}: {len(site_events)} events, {len(accesses)} accesses, "
            f"arrays={','.join(arrays) or '-'}, kinds={','.join(kinds) or '-'}, "
            f"lanes={lane_summary}"
        )

def describe_hypergraph(components: Iterable[ObjectiveComponent]) -> None:
    """ Print a summary of the hypergraph formed by the objectives. """
    print("Hypergraph summary:")
    for component in components:
        edges = [
            (array, edge)
            for array, array_edges in component.edges_by_array.items()
            for edge in array_edges
        ]

        # A source identifies the access scope that produced an edge (for
        # example, one event or one temporal window).  An explicit edge may
        # omit its source, in which case the edge itself is the only useful
        # region identity.
        regions: set[object] = set()
        for index, (array, edge) in enumerate(edges):
            regions.add(("source", edge.source) if edge.source else ("edge", index))

        # Arrays are separate allocations, so a coordinate is only a vertex
        # together with the array it belongs to.
        vertices = {
            (array, point)
            for array, edge in edges
            for point in edge.points
        }

        print(
            f"  {component.name}: {len(regions)} regions, "
            f"{len(edges)} edges, {len(vertices)} vertices"
        )

def describe_policy(policy: ScorePolicy) -> None:
    """Print the policy used to order layout candidates."""
    descriptions = {
        "lexicographic": "compares objectives in priority order",
        "weighted": "minimizes a weighted sum of objectives",
        "pareto": "keeps non-dominated trade-offs",
    }
    description = descriptions.get(policy.kind, "uses an unrecognized policy kind")
    order = ",".join(policy.order) or "-"
    print("Policy summary:")
    print(f"  kind={policy.kind} ({description})")
    print(f"  order={order}")
    if policy.weights:
        weights = ",".join(
            f"{name}={value:g}" for name, value in sorted(policy.weights.items())
        )
        print(f"  weights={weights}")
    print(
        f"  paths_per_state={policy.paths_per_state}, "
        f"frontier_limit={policy.frontier_limit}"
    )


def _search_standard_canonical(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    policy: ScorePolicy,
) -> LayoutSeed:
    """ Here we simply enumerate all "tiled" layouts, which is a subset of the canonical layouts.
        A tiled layout has word i^n j^m i^p j^q where n, m, p, q are the number of bits in each mode.
        This is of the form ijij, but we also consider the four forms ijij, ijji, jiij, jiji. This
        is all layouts where we have outer row or column major and an inner tile with row or column major.
        Since there are only approx 4 * (n+p+1) * (m+q+1) layouts, we simply enumerate them all.
    """
    if matrix.rank != 2:
        raise ValueError("standard tiled canonical search requires a rank-2 matrix")

    mode_bits = matrix.mode_bits
    total_bits = sum(mode_bits)
    components_by_dimension: dict[int, list[ObjectiveComponent]] = {}
    for component in components:
        if not component.search or not component.edges_by_array.get(matrix.name):
            continue
        dimension = component.dimension(matrix)
        if dimension <= total_bits:
            components_by_dimension.setdefault(dimension, []).append(component)

    node_cache: dict[tuple[int, int], dict[str, float]] = {}

    def node_score(counts: tuple[int, int]) -> dict[str, float]:
        score = node_cache.get(counts)
        if score is not None:
            return score
        score = {}
        for component in components_by_dimension.get(sum(counts), ()):
            total = 0.0
            for edge in component.edges_by_array[matrix.name]:
                cosets = {
                    (point[0] >> counts[0], point[1] >> counts[1])
                    for point in edge.points
                }
                total += edge.weight * len(cosets)
            score[component.name] = total
        node_cache[counts] = score
        return score

    words: dict[tuple[int, ...], None] = {}
    i_bits, j_bits = mode_bits
    for n in range(i_bits + 1):
        p = i_bits - n
        for m in range(j_bits + 1):
            q = j_bits - m
            i_n = (0,) * n
            j_m = (1,) * m
            i_p = (0,) * p
            j_q = (1,) * q
            words.setdefault(i_n + j_m + i_p + j_q, None)  # ijij
            words.setdefault(i_n + j_m + j_q + i_p, None)  # ijji
            words.setdefault(j_m + i_n + i_p + j_q, None)  # jiij
            words.setdefault(j_m + i_n + j_q + i_p, None)  # jiji

    best_word: tuple[int, ...] | None = None
    best_score: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None
    for word in words:
        counts = [0, 0]
        score = dict(node_score((0, 0)))
        for mode in word:
            counts[mode] += 1
            for name, value in node_score((counts[0], counts[1])).items():
                score[name] = score.get(name, 0.0) + value
        score["runs"] = float(
            bool(word) + sum(left != right for left, right in zip(word, word[1:]))
        )
        score["xors"] = 0.0
        key = policy.key(score)
        if best_key is None or key < best_key:
            best_word = word
            best_score = score
            best_key = key

    assert best_word is not None and best_score is not None
    word_text = "".join(matrix.mode_names[mode] for mode in best_word)
    layout = CanonicalLayout(
        f"simple_standard_canonical_{word_text or 'empty'}",
        matrix.name,
        mode_bits,
        best_word,
        tuple(reversed(range(matrix.rank))),
    )
    layout.validate(matrix)

    exact = policy.kind != "pareto"
    note = (
        "exhaustive tiled canonical enumeration"
        if exact
        else "Pareto candidates simplified to one lexicographic representative"
    )
    stats = SearchStats(
        "simple_standard_canonical",
        mode_bits,
        states=len(node_cache),
        transitions=len(words) * total_bits,
        paths_considered=len(words),
        paths_retained=1,
        exact=exact,
        truncated=not exact,
        note=note,
    )
    return LayoutSeed(layout, best_score, exact, stats, note)


def _search_canonical_dp(
    matrix: MatrixSpec,
    components: Sequence[ObjectiveComponent],
    policy: ScorePolicy,
) -> LayoutSeed:
    """Solve the full-matrix canonical DP and return its best layout seed.

    The two-dimensional recurrence is

        D(a_i, a_j) = c(a_i, a_j)
                      + min(D(a_i - 1, a_j), D(a_i, a_j - 1)).

    Here a_i and a_j count the low bits already placed for each mode.  Paths
    through this grid spell canonical words such as ``iijjii``.  The code
    below applies the same idea to matrices of any rank.

    This is a smaller version of :func:`relay.search.search_canonical`.  It
    returns one path rather than a candidate family, so Pareto search is only
    approximated.
    """

    # given a matrix of rank d, the canonical word is a sequence of d letters
    # repeated according to the number of bits in each mode.  The DP grid is
    # d-dimensional, with each axis having length 1 + the number of bits in
    # that mode.  The recurrence is filled in one diagonal at a time, so that
    # all the dependencies of a grid point have already been computed.  The
    # score of a grid point is the sum of the scores of all components that
    # apply to that matrix and that grid point.  The score of a component is
    # the sum of the weights of its edges, where each edge contributes once
    # for each coset of the edge's points in the grid point's subgrid.
    tile_exponents = matrix.mode_bits
    tile_bits = sum(tile_exponents)

    # An objective with a 2^d-element region is scored after d address bits
    # have been chosen.  Components that do not apply to this matrix do not
    # participate in its search.
    components_by_dimension: dict[int, list[ObjectiveComponent]] = {}
    for component in components:
        # filter out components not relevant
        if not component.search or not component.edges_by_array.get(matrix.name):
            continue
        dimension = component.dimension(matrix)
        if dimension <= tile_bits:
            components_by_dimension.setdefault(dimension, []).append(component)

    # c(a_i, a_j, ...) only depends on the grid point, not on how we reached
    # it, so cache it.  Shifting a coordinate right by the number of bits
    # already placed identifies the coset containing that coordinate.
    node_cache: dict[tuple[int, ...], dict[str, float]] = {}

    def node_score(counts: tuple[int, ...]) -> dict[str, float]:
        """ Return the score of a grid point, which is the sum of the scores of all
            components that apply to this matrix and that grid point.  The score of 
            a component is the sum of the weights of its edges, where each edge 
            contributes once for each coset of the edge's points in the grid point's 
            subgrid.
        """
        if counts not in node_cache:
            score: dict[str, float] = {}
            for component in components_by_dimension.get(sum(counts), ()):
                total = 0.0
                for edge in component.edges_by_array[matrix.name]:
                    cosets = {
                        tuple(value >> used for value, used in zip(point, counts))
                        for point in edge.points
                    }
                    total += edge.weight * len(cosets)
                score[component.name] = total
            node_cache[counts] = score
        return node_cache[counts]

    # A path is (score, canonical word).  Keeping the last mode in the state
    # lets us charge one run whenever the next letter differs from the last.
    # For a first sketch we retain only the best path to each state.
    zero = tuple(0 for _ in tile_exponents)
    initial_score = dict(node_score(zero))
    initial_score["runs"] = 0.0
    initial_score["xors"] = 0.0
    layer: dict[
        tuple[tuple[int, ...], int | None],
        tuple[dict[str, float], tuple[int, ...]],
    ] = {(zero, None): (initial_score, ())}

    stats = SearchStats("simple_canonical", tile_exponents, states=1)

    # Fill the grid one diagonal at a time.  A transition appends the next
    # unused low bit of one mode to the canonical word.
    for _ in range(tile_bits):
        next_layer: dict[
            tuple[tuple[int, ...], int],
            tuple[dict[str, float], tuple[int, ...]],
        ] = {}
        for (counts, last_mode), (path_score, word) in layer.items():
            for mode, limit in enumerate(tile_exponents):
                if counts[mode] == limit:
                    continue

                next_counts_list = list(counts)
                next_counts_list[mode] += 1
                next_counts = tuple(next_counts_list)

                score = dict(path_score)
                for name, value in node_score(next_counts).items():
                    score[name] = score.get(name, 0.0) + value
                if last_mode is None or last_mode != mode:
                    score["runs"] += 1.0

                next_word = (*word, mode)
                state = (next_counts, mode)
                incumbent = next_layer.get(state)
                if incumbent is None or policy.key(score) < policy.key(incumbent[0]):
                    next_layer[state] = (score, next_word)

                stats.transitions += 1
                stats.paths_considered += 1

        layer = next_layer
        stats.states += len(layer)
        stats.paths_retained += len(layer)

    best_score, best_word = min(layer.values(), key=lambda path: policy.key(path[0]))

    # The tile covers the whole matrix, so the outer order is immaterial.  A
    # conventional row-major outer order keeps the Layout object well formed.
    word_text = "".join(matrix.mode_names[mode] for mode in best_word)
    layout = CanonicalLayout(
        f"simple_canonical_{word_text or 'empty'}",
        matrix.name,
        tile_exponents,
        best_word,
        tuple(reversed(range(matrix.rank))),
    )
    layout.validate(matrix)

    # A single path per state is exact for lexicographic and weighted search.
    # It is only a convenient representative for a Pareto policy, which would
    # need to retain a frontier of incomparable scores at every state.
    stats.exact = policy.kind != "pareto"
    stats.truncated = policy.kind == "pareto"
    stats.note = (
        "one path per state"
        if stats.exact
        else "Pareto frontier simplified to one lexicographic path per state"
    )
    return LayoutSeed(layout, best_score, stats.exact, stats, stats.note)


def simple_solve(problem: SimpleRelayProblem) -> "RelayResult":
    start = perf_counter()

    matrices = collect_matrices(problem)
    events = collect_events(problem)
    describe_matrices(matrices)
    describe_events(events)

    components = build_objectives(problem.objectives, matrices, events, problem.sequences)
    describe_hypergraph(components)

    component_names = tuple(component.name for component in components)
    policy = ScorePolicy(
        kind = "lexicographic",
        order = (
            *component_names,
            "runs",
            "xors",
        )
    )
    describe_policy(policy)

    matrix = next(iter(matrices.values()))
    if problem.grammar == "canonical":
        seed = _search_canonical_dp(matrix, components, policy)
    elif problem.grammar == "standard":
        seed = _search_standard_canonical(matrix, components, policy)
    print(seed)
