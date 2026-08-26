"""Universal construction of hardware-independent access-scope edge families."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Iterable, Mapping, Sequence

from .gf2 import rank
from .model import Access, Coord, EventSequence, MatrixSpec, MemoryEvent
from .objectives import EdgeFamily, Hyperedge, ObjectiveComponent, ScopeKey


OPERATION_ALIASES = {
    "read": "load",
    "load": "load",
    "write": "store",
    "store": "store",
    "atomic": "atomic",
}


@dataclass(frozen=True)
class UniversalScopeBasis:
    """The fixed v1 scope grammar used for every kernel event trace."""

    issue_lane_groups: tuple[int, ...] = (8, 16, 32, 64)
    temporal_windows: tuple[int, ...] = (4, 16)
    operations: tuple[str, ...] = ("load", "store", "atomic")
    include_phase: bool = True

    def __post_init__(self) -> None:
        for label, values in (
            ("issue lane groups", self.issue_lane_groups),
            ("temporal windows", self.temporal_windows),
        ):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{label} must be nonempty, sorted, and unique")
            if any(value <= 0 or value & (value - 1) for value in values):
                raise ValueError(f"{label} must contain positive powers of two")
        if tuple(dict.fromkeys(self.operations)) != self.operations:
            raise ValueError("operation classes must be unique")
        unknown = set(self.operations) - {"load", "store", "atomic"}
        if unknown:
            raise ValueError(
                "unknown operation classes: " + ", ".join(sorted(unknown))
            )

    def scope_keys(self) -> tuple[ScopeKey, ...]:
        """Return the complete global schema, including potentially empty cells."""

        keys: list[ScopeKey] = []
        for operation in self.operations:
            keys.extend(
                ScopeKey("issue", group, "stream", operation)
                for group in self.issue_lane_groups
            )
            for family in ("lane_window", "simd_window", "workgroup_window"):
                keys.extend(
                    ScopeKey(family, window, partition, operation)
                    for window in self.temporal_windows
                    for partition in ("stream", "array")
                )
            keys.extend(
                ScopeKey("workgroup_step", None, partition, operation)
                for partition in ("stream", "array")
            )
            if self.include_phase:
                keys.extend(
                    ScopeKey("phase", cohort, partition, operation)
                    for cohort in ("lane", "simd", "workgroup")
                    for partition in ("stream", "array")
                )
        return tuple(keys)


UNIVERSAL_V1_BASIS = UniversalScopeBasis()

# Exact all-anchor canonicalization is quadratic in edge cardinality. This
# covers current workgroup-step shapes while bounding larger working sets.
_MAX_NONAFFINE_CANONICAL_POINTS = 256


@dataclass(frozen=True)
class _Occurrence:
    points: tuple[Coord, ...]
    weight: float
    source: str


@dataclass(frozen=True)
class ResourceCohort:
    """Cross-allocation accesses sharing one hardware scoring window."""

    family: str
    accesses: tuple[Access, ...]
    weight: float
    source: str

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("resource cohort family must be nonempty")
        if not self.accesses:
            raise ValueError("resource cohort accesses must be nonempty")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("resource cohort weight must be finite and positive")


def _parse_resource_cohort_family(name: str) -> tuple[int, str]:
    parts = name.split(".")
    if (
        len(parts) != 4
        or parts[0] != "simd_window"
        or not parts[1].startswith("t")
        or parts[2] != "cohort"
        or parts[3] not in {"load", "store", "atomic"}
    ):
        raise ValueError(
            f"unsupported resource cohort family {name!r}; expected "
            "'simd_window.t<window>.cohort.<operation>'"
        )
    try:
        window = int(parts[1][1:])
    except ValueError as error:
        raise ValueError(f"invalid resource cohort window in {name!r}") from error
    if window <= 0 or window & (window - 1):
        raise ValueError("resource cohort window must be a positive power of two")
    return window, parts[3]


def build_resource_cohorts(
    matrices: Mapping[str, MatrixSpec],
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
    cohort_families: Iterable[str],
) -> dict[str, tuple[ResourceCohort, ...]]:
    """Build hardware-only cohorts while retaining allocation identities."""

    families = tuple(dict.fromkeys(cohort_families))
    if not families:
        return {}
    validate_trace_contract(events, sequences)
    result: dict[str, tuple[ResourceCohort, ...]] = {}
    for family in families:
        window, selected_operation = _parse_resource_cohort_family(family)
        cohorts: list[ResourceCohort] = []
        for sequence in sequences:
            by_interval: dict[int, list[tuple[MemoryEvent, Access]]] = {}
            for slot, event_id in enumerate(sequence.event_ids):
                event = events[event_id]
                for access in event.accesses:
                    matrix = matrices.get(access.array)
                    if matrix is None:
                        raise ValueError(
                            f"event {event.id}: unknown array {access.array}"
                        )
                    matrix.validate_coord(access.coord)
                    if operation_class(access) == selected_operation:
                        by_interval.setdefault(slot // window, []).append(
                            (event, access)
                        )
            for interval, entries in sorted(by_interval.items()):
                cohorts.append(
                    ResourceCohort(
                        family=family,
                        accesses=tuple(access for _event, access in entries),
                        weight=min(event.weight for event, _access in entries),
                        source=(
                            f"{sequence.name}:cohort-slots"
                            f"[{interval * window}:{(interval + 1) * window}]"
                        ),
                    )
                )
        result[family] = tuple(cohorts)
    return result


def compress_resource_cohorts(
    matrices: Mapping[str, MatrixSpec],
    cohorts: Sequence[ResourceCohort],
) -> tuple[ResourceCohort, ...]:
    """Merge cohorts equivalent under independent per-array XOR translation.

    This preserves colored occupancy for linear layouts when allocation color
    phases are optimized independently, as in the robust policy. Repeated
    dynamic windows are represented by the sum of their weights.
    """

    groups: dict[
        tuple[tuple[str, tuple[int, ...]], ...],
        tuple[ResourceCohort, float, int],
    ] = {}
    for cohort in cohorts:
        by_array: dict[str, set[int]] = {}
        for access in cohort.accesses:
            matrix = matrices.get(access.array)
            if matrix is None:
                raise ValueError(
                    f"resource cohort {cohort.source}: unknown array "
                    f"{access.array!r}"
                )
            by_array.setdefault(access.array, set()).add(
                matrix.coord_to_bits(access.coord)
            )
        signature = []
        for array, values_set in sorted(by_array.items()):
            values = tuple(sorted(values_set))
            translated = min(
                tuple(sorted(value ^ anchor for value in values))
                for anchor in values
            )
            signature.append((array, translated))
        key = tuple(signature)
        existing = groups.get(key)
        if existing is None:
            groups[key] = (cohort, cohort.weight, 1)
        else:
            representative, weight, count = existing
            groups[key] = (representative, weight + cohort.weight, count + 1)

    return tuple(
        ResourceCohort(
            family=representative.family,
            accesses=representative.accesses,
            weight=weight,
            source=(
                representative.source
                if count == 1
                else f"{representative.source} (+{count - 1} XOR translations)"
            ),
        )
        for representative, weight, count in groups.values()
    )


def operation_class(access: Access) -> str:
    """Map trace spelling to the profile's load/store/atomic classes."""

    try:
        return OPERATION_ALIASES[access.kind]
    except KeyError as error:
        raise ValueError(f"unknown memory operation kind {access.kind!r}") from error


def dynamic_useful_bytes(
    matrices: Mapping[str, MatrixSpec], events: Iterable[MemoryEvent]
) -> float:
    """Count lane-requested dynamic bytes for target-array accesses."""

    total = 0.0
    for event in events:
        for access in event.accesses:
            matrix = matrices.get(access.array)
            if matrix is None:
                raise ValueError(f"event {event.id}: unknown array {access.array}")
            if not matrix.target:
                continue
            width = (
                matrix.element_bytes
                if access.width_bytes is None
                else access.width_bytes
            )
            total += event.weight * width
    if total <= 0:
        raise ValueError("universal scopes require at least one target-array access")
    return total


def validate_trace_contract(
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
) -> None:
    """Validate the complete ordered local-trace contract used by v1 scopes."""

    mismatched_ids = [key for key, event in events.items() if key != event.id]
    if mismatched_ids:
        raise ValueError(
            "event mapping keys must match MemoryEvent.id: "
            + ", ".join(sorted(mismatched_ids))
        )

    owners: dict[str, str] = {}
    for sequence in sequences:
        if sequence.weight != 1.0:
            raise ValueError(
                f"sequence {sequence.name}: universal-v1 requires weight 1"
            )
        previous_order: int | None = None
        for event_id in sequence.event_ids:
            if event_id not in events:
                raise ValueError(
                    f"sequence {sequence.name}: unknown event id {event_id!r}"
                )
            owner = owners.get(event_id)
            if owner is not None:
                raise ValueError(
                    f"event {event_id!r} appears more than once "
                    f"({owner!r} and {sequence.name!r})"
                )
            event = events[event_id]
            if previous_order is not None and event.order < previous_order:
                raise ValueError(
                    f"sequence {sequence.name}: events are not in nondecreasing "
                    "MemoryEvent.order"
                )
            previous_order = event.order
            owners[event_id] = sequence.name

    missing = sorted(set(events) - set(owners))
    if missing:
        raise ValueError(
            "universal-v1 sequences must contain every event exactly once; missing: "
            + ", ".join(missing)
        )

    multiplicities = {event.weight for event in events.values()}
    if len(multiplicities) > 1:
        raise ValueError(
            "universal-v1 requires one common MemoryEvent.weight across the trace"
        )


def _selected(
    event: MemoryEvent,
    matrices: Mapping[str, MatrixSpec],
) -> tuple[tuple[Access, str], ...]:
    result = []
    for access in event.accesses:
        matrix = matrices.get(access.array)
        if matrix is None:
            raise ValueError(f"event {event.id}: unknown array {access.array}")
        if matrix.target:
            result.append((access, operation_class(access)))
    return tuple(result)


def _append_edge(
    edges: dict[ScopeKey, dict[str, list[Hyperedge]]],
    scope: ScopeKey,
    array: str,
    points: Iterable[Coord],
    *,
    weight: float,
    source: str,
) -> None:
    edge = Hyperedge.make(points, weight=weight, source=source)
    edges.setdefault(scope, {}).setdefault(array, []).append(edge)


def _windowed(
    occurrences: Sequence[_Occurrence], window: int
) -> Iterable[tuple[int, tuple[Coord, ...], float]]:
    for start in range(0, len(occurrences), window):
        chunk = occurrences[start : start + window]
        if not chunk:
            continue
        yield (
            start,
            tuple(point for occurrence in chunk for point in occurrence.points),
            min(occurrence.weight for occurrence in chunk),
        )


def _issue_edges(
    edges: dict[ScopeKey, dict[str, list[Hyperedge]]],
    basis: UniversalScopeBasis,
    matrices: Mapping[str, MatrixSpec],
    events: Sequence[MemoryEvent],
) -> None:
    for event in events:
        by_stream: dict[tuple[str, str], list[Access]] = {}
        for access, operation in _selected(event, matrices):
            by_stream.setdefault((access.array, operation), []).append(access)
        for (array, operation), accesses in by_stream.items():
            if operation not in basis.operations:
                continue
            for group_size in basis.issue_lane_groups:
                lane_groups: dict[int, list[Coord]] = {}
                for access in accesses:
                    if access.lane is None:
                        raise ValueError(
                            f"event {event.id}: issue scopes require lane ids"
                        )
                    lane_groups.setdefault(access.lane // group_size, []).append(
                        access.coord
                    )
                scope = ScopeKey("issue", group_size, "stream", operation)
                for lane_group, points in lane_groups.items():
                    _append_edge(
                        edges,
                        scope,
                        array,
                        points,
                        weight=event.weight,
                        source=f"{event.id}:g{group_size}:{lane_group}",
                    )


def _sequence_edges(
    edges: dict[ScopeKey, dict[str, list[Hyperedge]]],
    basis: UniversalScopeBasis,
    matrices: Mapping[str, MatrixSpec],
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
) -> None:
    for sequence in sequences:
        lane_streams: dict[tuple[str, str, str, int], list[_Occurrence]] = {}

        for event_id in sequence.event_ids:
            event = events[event_id]
            lane_buckets: dict[tuple[str, str, str, int], list[Coord]] = {}
            for access, operation in _selected(event, matrices):
                if operation not in basis.operations:
                    continue
                if access.lane is None:
                    raise ValueError(
                        f"event {event.id}: temporal scopes require lane ids"
                    )
                lane_buckets.setdefault(
                    (event.site, access.array, operation, access.lane), []
                ).append(access.coord)

            for (site, array, operation, lane), points in lane_buckets.items():
                occurrence = _Occurrence(
                    tuple(points), event.weight, f"{event.id}:lane{lane}"
                )
                lane_streams.setdefault(
                    (site, array, operation, lane), []
                ).append(occurrence)

        for window in basis.temporal_windows:
            lane_arrays: dict[
                tuple[str, str, int, int], list[_Occurrence]
            ] = {}
            for (site, array, operation, lane), occurrences in lane_streams.items():
                scope = ScopeKey("lane_window", window, "stream", operation)
                for start, points, weight in _windowed(occurrences, window):
                    _append_edge(
                        edges,
                        scope,
                        array,
                        points,
                        weight=weight,
                        source=(
                            f"{sequence.name}:{site}:stream"
                            f"[{start}:{start + window}]"
                        ),
                    )
                for occurrence_index, occurrence in enumerate(occurrences):
                    interval = occurrence_index // window
                    lane_arrays.setdefault(
                        (array, operation, lane, interval), []
                    ).append(occurrence)

            for (array, operation, lane, interval), occurrences in lane_arrays.items():
                start = interval * window
                _grouped_edge(
                    edges,
                    ScopeKey("lane_window", window, "array", operation),
                    array,
                    occurrences,
                    source=(
                        f"{sequence.name}:lane{lane}:array"
                        f"[{start}:{start + window}]"
                    ),
                )

            simd_streams: dict[
                tuple[str, str, str, int], list[_Occurrence]
            ] = {}
            simd_arrays: dict[tuple[str, str, int], list[_Occurrence]] = {}
            for slot, event_id in enumerate(sequence.event_ids):
                event = events[event_id]
                interval = slot // window
                by_site: dict[tuple[str, str, str], list[Coord]] = {}
                by_array: dict[tuple[str, str], list[Coord]] = {}
                for access, operation in _selected(event, matrices):
                    if operation not in basis.operations:
                        continue
                    if access.lane is None:
                        raise ValueError(
                            f"event {event.id}: temporal scopes require lane ids"
                        )
                    by_site.setdefault(
                        (event.site, access.array, operation), []
                    ).append(access.coord)
                    by_array.setdefault((access.array, operation), []).append(
                        access.coord
                    )
                for (site, array, operation), points in by_site.items():
                    simd_streams.setdefault(
                        (site, array, operation, interval), []
                    ).append(_Occurrence(tuple(points), event.weight, event.id))
                for (array, operation), points in by_array.items():
                    simd_arrays.setdefault(
                        (array, operation, interval), []
                    ).append(_Occurrence(tuple(points), event.weight, event.id))

            for (site, array, operation, interval), occurrences in simd_streams.items():
                start = interval * window
                _grouped_edge(
                    edges,
                    ScopeKey("simd_window", window, "stream", operation),
                    array,
                    occurrences,
                    source=(
                        f"{sequence.name}:{site}:stream-slots"
                        f"[{start}:{start + window}]"
                    ),
                )
            for (array, operation, interval), occurrences in simd_arrays.items():
                start = interval * window
                _grouped_edge(
                    edges,
                    ScopeKey("simd_window", window, "array", operation),
                    array,
                    occurrences,
                    source=(
                        f"{sequence.name}:array-slots"
                        f"[{start}:{start + window}]"
                    ),
                )


def _metadata(event: MemoryEvent, key: str, fallback: str = "") -> str:
    value = event.meta(key)
    return fallback if value is None else value


def _grouped_edge(
    edges: dict[ScopeKey, dict[str, list[Hyperedge]]],
    scope: ScopeKey,
    array: str,
    occurrences: Sequence[_Occurrence],
    source: str,
) -> None:
    _append_edge(
        edges,
        scope,
        array,
        (point for occurrence in occurrences for point in occurrence.points),
        weight=min(occurrence.weight for occurrence in occurrences),
        source=source,
    )


def _workgroup_edges(
    edges: dict[ScopeKey, dict[str, list[Hyperedge]]],
    basis: UniversalScopeBasis,
    matrices: Mapping[str, MatrixSpec],
    events: Sequence[MemoryEvent],
) -> None:
    steps: dict[tuple[object, ...], list[_Occurrence]] = {}
    windows: dict[tuple[object, ...], list[_Occurrence]] = {}
    for event in events:
        step_text = event.meta("step")
        if step_text is None:
            continue
        try:
            step = int(step_text)
        except ValueError as error:
            raise ValueError(
                f"event {event.id}: step metadata must be an integer"
            ) from error
        workgroup = _metadata(event, "workgroup", event.group)
        phase = _metadata(event, "phase")
        by_array: dict[tuple[str, str], list[Coord]] = {}
        for access, operation in _selected(event, matrices):
            if operation in basis.operations:
                by_array.setdefault((access.array, operation), []).append(access.coord)
        for (array, operation), points in by_array.items():
            occurrence = _Occurrence(tuple(points), event.weight, event.id)
            steps.setdefault(
                ("stream", workgroup, phase, step, event.site, array, operation), []
            ).append(occurrence)
            steps.setdefault(
                ("array", workgroup, phase, step, array, operation), []
            ).append(occurrence)
            for window in basis.temporal_windows:
                interval = step // window
                windows.setdefault(
                    (
                        "stream",
                        window,
                        workgroup,
                        phase,
                        interval,
                        event.site,
                        array,
                        operation,
                    ),
                    [],
                ).append(occurrence)
                windows.setdefault(
                    (
                        "array",
                        window,
                        workgroup,
                        phase,
                        interval,
                        array,
                        operation,
                    ),
                    [],
                ).append(occurrence)

    for key, occurrences in steps.items():
        partition = str(key[0])
        array = str(key[-2])
        operation = str(key[-1])
        _grouped_edge(
            edges,
            ScopeKey("workgroup_step", None, partition, operation),
            array,
            occurrences,
            source="workgroup-step:" + "/".join(str(value) for value in key[1:-2]),
        )
    for key, occurrences in windows.items():
        partition = str(key[0])
        window = int(key[1])
        array = str(key[-2])
        operation = str(key[-1])
        _grouped_edge(
            edges,
            ScopeKey("workgroup_window", window, partition, operation),
            array,
            occurrences,
            source="workgroup-window:" + "/".join(str(value) for value in key[2:-2]),
        )


def _phase_edges(
    edges: dict[ScopeKey, dict[str, list[Hyperedge]]],
    basis: UniversalScopeBasis,
    matrices: Mapping[str, MatrixSpec],
    events: Sequence[MemoryEvent],
) -> None:
    if not basis.include_phase:
        return
    grouped: dict[tuple[object, ...], list[_Occurrence]] = {}
    for event in events:
        phase = event.meta("phase")
        if phase is None:
            continue
        workgroup = _metadata(event, "workgroup", event.group)
        simd = _metadata(event, "wave", event.group)
        by_lane: dict[tuple[str, str, int], list[Coord]] = {}
        by_array: dict[tuple[str, str], list[Coord]] = {}
        for access, operation in _selected(event, matrices):
            if operation not in basis.operations:
                continue
            if access.lane is None:
                raise ValueError(f"event {event.id}: phase scopes require lane ids")
            by_lane.setdefault((access.array, operation, access.lane), []).append(
                access.coord
            )
            by_array.setdefault((access.array, operation), []).append(access.coord)

        for (array, operation, lane), points in by_lane.items():
            occurrence = _Occurrence(tuple(points), event.weight, event.id)
            grouped.setdefault(
                (
                    "lane",
                    "stream",
                    workgroup,
                    simd,
                    phase,
                    event.site,
                    lane,
                    array,
                    operation,
                ),
                [],
            ).append(occurrence)
            grouped.setdefault(
                (
                    "lane",
                    "array",
                    workgroup,
                    simd,
                    phase,
                    lane,
                    array,
                    operation,
                ),
                [],
            ).append(occurrence)
        for (array, operation), points in by_array.items():
            occurrence = _Occurrence(tuple(points), event.weight, event.id)
            for cohort, identity in (
                ("simd", (workgroup, simd, phase)),
                ("workgroup", (workgroup, phase)),
            ):
                grouped.setdefault(
                    (cohort, "stream", *identity, event.site, array, operation), []
                ).append(occurrence)
                grouped.setdefault(
                    (cohort, "array", *identity, array, operation), []
                ).append(occurrence)

    for key, occurrences in grouped.items():
        cohort = str(key[0])
        partition = str(key[1])
        array = str(key[-2])
        operation = str(key[-1])
        _grouped_edge(
            edges,
            ScopeKey("phase", cohort, partition, operation),
            array,
            occurrences,
            source="phase:" + "/".join(str(value) for value in key[2:-2]),
        )


def _compress_edges(
    matrix: MatrixSpec, edges: Sequence[Hyperedge]
) -> tuple[Hyperedge, ...]:
    """Exactly merge tractable XOR translations and retain their multiplicity.

    Affine cosets have the same normalized set for every member anchor, so one
    anchor is sufficient. Non-affine sets of at most 256 points are
    canonicalized over every possible anchor. Larger non-affine sets remain
    separate to keep edge construction from doing quadratic work.
    """

    groups: dict[tuple[object, ...], tuple[Hyperedge, float, int]] = {}
    nonaffine_orbits: dict[tuple[int, ...], tuple[int, ...]] = {}
    offsets = matrix.bit_offsets()

    def encode(point: Coord) -> int:
        matrix.validate_coord(point)
        value = 0
        for component, shift in zip(point, offsets):
            value |= component << shift
        return value

    for edge_index, edge in enumerate(edges):
        values = tuple(sorted(encode(point) for point in edge.points))
        normalized = tuple(sorted(value ^ values[0] for value in values))
        cardinality = len(normalized)
        affine = (
            (cardinality & (cardinality - 1)) == 0
            and cardinality == 1 << rank(normalized)
        )
        if affine:
            key: tuple[object, ...] = ("translation", normalized)
        elif len(values) <= _MAX_NONAFFINE_CANONICAL_POINTS:
            signature = nonaffine_orbits.get(normalized)
            if signature is None:
                orbit = tuple(
                    tuple(sorted(value ^ anchor for value in values))
                    for anchor in values
                )
                signature = min(orbit)
                for translated_signature in orbit:
                    nonaffine_orbits[translated_signature] = signature
            key = ("translation", signature)
        else:
            key = ("unmerged", edge_index, values)

        existing = groups.get(key)
        if existing is None:
            groups[key] = (edge, edge.weight, 1)
        else:
            representative, weight, count = existing
            groups[key] = (representative, weight + edge.weight, count + 1)
    result = []
    for key in sorted(groups):
        representative, weight, count = groups[key]
        source = representative.source
        if count > 1:
            source = f"{source} (+{count - 1} XOR translations)"
        result.append(Hyperedge(representative.points, weight, source))
    return tuple(result)


def build_edge_families(
    matrices: Mapping[str, MatrixSpec],
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
    *,
    basis: UniversalScopeBasis = UNIVERSAL_V1_BASIS,
) -> tuple[EdgeFamily, ...]:
    """Build the same scale-free scope grammar from any kernel event trace."""

    validate_trace_contract(events, sequences)
    ordered_events = tuple(
        sorted(events.values(), key=lambda event: (event.order, event.id))
    )
    exposure = dynamic_useful_bytes(matrices, ordered_events)
    edge_lists: dict[ScopeKey, dict[str, list[Hyperedge]]] = {}
    _issue_edges(edge_lists, basis, matrices, ordered_events)
    _sequence_edges(edge_lists, basis, matrices, events, sequences)
    _workgroup_edges(edge_lists, basis, matrices, ordered_events)
    _phase_edges(edge_lists, basis, matrices, ordered_events)

    descriptions = {
        "issue": "one aligned contiguous lane group at one dynamic instruction",
        "lane_window": "one lane over an aligned temporal window",
        "simd_window": "one SIMD cohort over an aligned event window",
        "workgroup_step": "one workgroup at one logical schedule step",
        "workgroup_window": "one workgroup over aligned logical schedule steps",
        "phase": "one execution cohort over an explicit kernel phase",
    }
    families = []
    for scope in basis.scope_keys():
        arrays = edge_lists.get(scope)
        if not arrays:
            continue
        families.append(
            EdgeFamily(
                scope=scope,
                edges_by_array={
                    array: _compress_edges(matrices[array], array_edges)
                    for array, array_edges in sorted(arrays.items())
                },
                normalization_bytes=exposure,
                provenance="universal-v1",
                description=descriptions[scope.family],
            )
        )
    return tuple(families)


def materialize_edge_families(
    families: Sequence[EdgeFamily],
    matrices: Mapping[str, MatrixSpec],
    byte_scales: Sequence[int],
) -> tuple[ObjectiveComponent, ...]:
    """Cross scale-free families with a physical byte-scale ladder."""

    scales = tuple(byte_scales)
    if not scales or tuple(sorted(set(scales))) != scales:
        raise ValueError("byte scales must be nonempty, sorted, and unique")
    components: list[ObjectiveComponent] = []
    for family in families:
        for region_bytes in scales:
            component = family.at_scale(region_bytes)
            for array in family.edges_by_array:
                component.dimension(matrices[array])
            components.append(component)
    return tuple(components)


@dataclass(frozen=True)
class UniversalScopeObjectives:
    """Objective adapter that preserves the existing solver problem protocol."""

    byte_scales: tuple[int, ...]
    basis: UniversalScopeBasis = field(default_factory=UniversalScopeBasis)

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        families = build_edge_families(
            matrices,
            events,
            sequences,
            basis=self.basis,
        )
        return list(materialize_edge_families(families, matrices, self.byte_scales))

    def schema_names(self) -> tuple[str, ...]:
        return tuple(
            f"{scope.name}.{region_bytes}B"
            for scope in self.basis.scope_keys()
            for region_bytes in self.byte_scales
        )
