"""Sequence-sensitive quotient-region persistence objectives.

The ordinary RELAY objectives score each access working set independently.
This module derives layout-independent transition relations from the recorded
event schedule and scores how many quotient regions in the later event were
absent from the earlier event. Region identities are tagged by allocation, so
this objective does not assume physical placement relationships between arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable, Mapping, Sequence

from .access_scopes import operation_class, validate_trace_contract
from .layouts import Layout
from .model import Access, Coord, EventSequence, MatrixSpec, MemoryEvent


@dataclass(frozen=True, order=True)
class TransitionKey:
    """One schedule-derived transition family at a fixed event distance."""

    family: str
    delta: int
    operation: str

    @property
    def name(self) -> str:
        return f"{self.family}.d{self.delta}.{self.operation}"


TransitionSide = tuple[tuple[str, tuple[Coord, ...]], ...]


@dataclass(frozen=True)
class QuotientTransition:
    """One weighted ordered pair of logical access working sets."""

    previous: TransitionSide
    current: TransitionSide
    weight: float = 1.0
    multiplicity: int = 1
    source: str = ""

    def __post_init__(self) -> None:
        if not self.previous or not self.current:
            raise ValueError("quotient transitions require two nonempty sides")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("quotient transition weight must be finite and positive")
        if self.multiplicity <= 0:
            raise ValueError("quotient transition multiplicity must be positive")


@dataclass(frozen=True)
class TransitionFamily:
    """A homogeneous collection of ordered access-event relations."""

    key: TransitionKey
    transitions: tuple[QuotientTransition, ...]
    transition_count: int
    transition_weight: float
    description: str = ""

    @property
    def name(self) -> str:
        return self.key.name


@dataclass(frozen=True)
class TemporalPersistenceBasis:
    """Kernel-independent transition grammar used by ``J_persist``."""

    deltas: tuple[int, ...] = (1, 4, 16)
    families: tuple[str, ...] = (
        "simd_stream",
        "lane_stream",
        "simd_schedule",
    )

    def __post_init__(self) -> None:
        if not self.deltas or tuple(sorted(set(self.deltas))) != self.deltas:
            raise ValueError("transition deltas must be nonempty, sorted, and unique")
        if any(delta <= 0 for delta in self.deltas):
            raise ValueError("transition deltas must be positive")
        supported = {"simd_stream", "lane_stream", "simd_schedule"}
        if not self.families or tuple(dict.fromkeys(self.families)) != self.families:
            raise ValueError("transition families must be nonempty and unique")
        unknown = set(self.families) - supported
        if unknown:
            raise ValueError(
                "unknown transition families: " + ", ".join(sorted(unknown))
            )


UNIVERSAL_PERSISTENCE_V1_BASIS = TemporalPersistenceBasis()


@dataclass(frozen=True)
class PersistenceComponentScore:
    """Turnover for one transition family, event distance, and byte scale."""

    name: str
    transition_family: str
    delta: int
    operation: str
    region_bytes: int
    weight: float
    transition_count: int
    transition_weight: float
    weighted_new_demand: float
    normalized_turnover: float

    @property
    def weighted_turnover(self) -> float:
        return self.weight * self.weighted_new_demand


@dataclass(frozen=True)
class TemporalPersistenceScore:
    """Detailed quotient-turnover primitives and aggregate ``J_persist``."""

    components: tuple[PersistenceComponentScore, ...]
    hardware_persist: float

    def component(self, name: str) -> PersistenceComponentScore:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)


@dataclass(frozen=True)
class _Occurrence:
    arrays: TransitionSide
    weight: float
    source: str


def _side(accesses: Iterable[Access]) -> TransitionSide:
    by_array: dict[str, set[Coord]] = {}
    for access in accesses:
        by_array.setdefault(access.array, set()).add(access.coord)
    return tuple(
        (array, tuple(sorted(points)))
        for array, points in sorted(by_array.items())
        if points
    )


def _append_transitions(
    destination: dict[TransitionKey, list[QuotientTransition]],
    family: str,
    operation: str,
    occurrences: Sequence[_Occurrence],
    deltas: Sequence[int],
    sequence_weight: float,
) -> None:
    for delta in deltas:
        key = TransitionKey(family, delta, operation)
        for index in range(len(occurrences) - delta):
            previous = occurrences[index]
            current = occurrences[index + delta]
            destination.setdefault(key, []).append(
                QuotientTransition(
                    previous=previous.arrays,
                    current=current.arrays,
                    weight=min(previous.weight, current.weight) * sequence_weight,
                    source=f"{previous.source}->{current.source}",
                )
            )


def _transition_signature(
    matrices: Mapping[str, MatrixSpec],
    transition: QuotientTransition,
    orbit_cache: dict[
        tuple[str, tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...]
    ],
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    labels_by_array: dict[str, dict[int, int]] = {}
    for state, side in ((1, transition.previous), (2, transition.current)):
        for array, points in side:
            matrix = matrices[array]
            labels = labels_by_array.setdefault(array, {})
            for point in points:
                logical_bits = matrix.coord_to_bits(point)
                labels[logical_bits] = labels.get(logical_bits, 0) | state

    signature = []
    for array, labels in sorted(labels_by_array.items()):
        first_anchor = min(labels)
        normalized = tuple(
            sorted((value ^ first_anchor, state) for value, state in labels.items())
        )
        cache_key = (array, normalized)
        canonical = orbit_cache.get(cache_key)
        if canonical is None:
            orbit = tuple(
                tuple(
                    sorted(
                        (value ^ anchor, state)
                        for value, state in labels.items()
                    )
                )
                for anchor in labels
            )
            canonical = min(orbit)
            for translated in orbit:
                orbit_cache[(array, translated)] = canonical
        signature.append((array, canonical))
    return tuple(signature)


def _compress_transitions(
    matrices: Mapping[str, MatrixSpec],
    transitions: Sequence[QuotientTransition],
) -> tuple[QuotientTransition, ...]:
    """Merge exact independent per-allocation XOR translations."""

    groups: dict[
        tuple[tuple[str, tuple[tuple[int, int], ...]], ...],
        tuple[QuotientTransition, float, int],
    ] = {}
    orbit_cache: dict[
        tuple[str, tuple[tuple[int, int], ...]], tuple[tuple[int, int], ...]
    ] = {}
    for transition in transitions:
        signature = _transition_signature(matrices, transition, orbit_cache)
        existing = groups.get(signature)
        if existing is None:
            groups[signature] = (
                transition,
                transition.weight,
                transition.multiplicity,
            )
        else:
            representative, weight, multiplicity = existing
            groups[signature] = (
                representative,
                weight + transition.weight,
                multiplicity + transition.multiplicity,
            )

    result = []
    for signature in sorted(groups):
        representative, weight, multiplicity = groups[signature]
        source = representative.source
        if multiplicity > representative.multiplicity:
            source = f"{source} (+{multiplicity - 1} XOR translations)"
        result.append(
            QuotientTransition(
                previous=representative.previous,
                current=representative.current,
                weight=weight,
                multiplicity=multiplicity,
                source=source,
            )
        )
    return tuple(result)


def build_transition_families(
    matrices: Mapping[str, MatrixSpec],
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
    *,
    basis: TemporalPersistenceBasis = UNIVERSAL_PERSISTENCE_V1_BASIS,
) -> tuple[TransitionFamily, ...]:
    """Derive ordered target-array relations from complete local schedules.

    ``simd_stream`` follows repeated dynamic issues from the same access site
    and allocation. ``lane_stream`` follows the corresponding per-lane loop
    stream. ``simd_schedule`` follows target-bearing issue events in schedule
    order and therefore includes cross-array relations such as A-load to
    B-load without equating the two allocations' region identifiers.
    """

    validate_trace_contract(events, sequences)
    transition_lists: dict[TransitionKey, list[QuotientTransition]] = {}

    for sequence in sequences:
        simd_streams: dict[tuple[str, str, str], list[_Occurrence]] = {}
        lane_streams: dict[tuple[str, str, str, int], list[_Occurrence]] = {}
        schedule_streams: dict[str, list[_Occurrence]] = {}

        for event_id in sequence.event_ids:
            event = events[event_id]
            selected: list[tuple[Access, str]] = []
            for access in event.accesses:
                matrix = matrices.get(access.array)
                if matrix is None:
                    raise ValueError(f"event {event.id}: unknown array {access.array}")
                matrix.validate_coord(access.coord)
                if matrix.target:
                    selected.append((access, operation_class(access)))

            by_stream: dict[tuple[str, str, str], list[Access]] = {}
            by_lane: dict[tuple[str, str, str, int], list[Access]] = {}
            by_operation: dict[str, list[Access]] = {}
            for access, operation in selected:
                by_stream.setdefault(
                    (event.site, access.array, operation), []
                ).append(access)
                if access.lane is None:
                    raise ValueError(
                        f"event {event.id}: persistence streams require lane ids"
                    )
                by_lane.setdefault(
                    (event.site, access.array, operation, access.lane), []
                ).append(access)
                by_operation.setdefault(operation, []).append(access)

            for key, accesses in by_stream.items():
                simd_streams.setdefault(key, []).append(
                    _Occurrence(_side(accesses), event.weight, event.id)
                )
            for key, accesses in by_lane.items():
                lane_streams.setdefault(key, []).append(
                    _Occurrence(_side(accesses), event.weight, event.id)
                )
            for operation, accesses in by_operation.items():
                schedule_streams.setdefault(operation, []).append(
                    _Occurrence(_side(accesses), event.weight, event.id)
                )

        if "simd_stream" in basis.families:
            for (_site, _array, operation), occurrences in simd_streams.items():
                _append_transitions(
                    transition_lists,
                    "simd_stream",
                    operation,
                    occurrences,
                    basis.deltas,
                    sequence.weight,
                )
        if "lane_stream" in basis.families:
            for (_site, _array, operation, _lane), occurrences in lane_streams.items():
                _append_transitions(
                    transition_lists,
                    "lane_stream",
                    operation,
                    occurrences,
                    basis.deltas,
                    sequence.weight,
                )
        if "simd_schedule" in basis.families:
            for operation, occurrences in schedule_streams.items():
                _append_transitions(
                    transition_lists,
                    "simd_schedule",
                    operation,
                    occurrences,
                    basis.deltas,
                    sequence.weight,
                )

    descriptions = {
        "simd_stream": "successive SIMD issues from one array access site",
        "lane_stream": "successive per-lane accesses from one array access site",
        "simd_schedule": "successive target-bearing issues in recorded schedule order",
    }
    families = []
    for key in sorted(transition_lists):
        raw = transition_lists[key]
        compressed = _compress_transitions(matrices, raw)
        families.append(
            TransitionFamily(
                key=key,
                transitions=compressed,
                transition_count=sum(item.multiplicity for item in compressed),
                transition_weight=sum(item.weight for item in compressed),
                description=descriptions[key.family],
            )
        )
    return tuple(families)


def score_temporal_persistence(
    matrices: Mapping[str, MatrixSpec],
    layouts: Mapping[str, Layout],
    families: Sequence[TransitionFamily],
    region_bytes: Sequence[int],
    *,
    component_weights: Mapping[str, float] | None = None,
) -> TemporalPersistenceScore:
    """Compute normalized newly demanded quotient regions for each relation.

    ``weighted_new_demand`` is the inner sum in the proposed objective,
    ``sum_t w_t nu_d(F|E)``. The default aggregate sums those values with
    ``rho=1``; callers can supply other ``rho`` values through exact
    component-name weights. ``normalized_turnover`` additionally exposes the
    weighted mean for diagnostics and scale/family ablations.
    """

    scales = tuple(region_bytes)
    if not scales or tuple(sorted(set(scales))) != scales:
        raise ValueError("persistence byte scales must be nonempty, sorted, and unique")
    weights = dict(component_weights or {})
    known_names = {
        f"persist.{family.name}.{scale}B"
        for family in families
        for scale in scales
    }
    unknown = set(weights) - known_names
    if unknown:
        raise ValueError(
            "weights were supplied for unknown persistence components: "
            + ", ".join(sorted(unknown))
        )
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("persistence component weights must be nonnegative")

    offset_caches: dict[str, dict[Coord, int]] = {array: {} for array in layouts}
    region_cache: dict[
        tuple[str, tuple[Coord, ...], int], frozenset[tuple[str, int]]
    ] = {}

    def tagged_regions(
        side: TransitionSide, scale: int
    ) -> frozenset[tuple[str, int]]:
        result: set[tuple[str, int]] = set()
        for array, points in side:
            if array not in matrices:
                raise ValueError(
                    f"persistence transition references unknown array {array!r}"
                )
            if array not in layouts:
                raise ValueError(f"no layout supplied for persistence array {array!r}")
            matrix = matrices[array]
            if scale % matrix.element_bytes:
                raise ValueError(
                    f"{scale} B is not divisible by {array}'s "
                    f"{matrix.element_bytes} B element width"
                )
            capacity = scale // matrix.element_bytes
            if capacity <= 0 or capacity & (capacity - 1):
                raise ValueError(
                    "persistence region capacity must be a positive power of two"
                )
            cache_key = (array, points, scale)
            cached = region_cache.get(cache_key)
            if cached is None:
                offsets = offset_caches[array]
                ids = set()
                for point in points:
                    offset = offsets.get(point)
                    if offset is None:
                        offset = layouts[array].offset(matrix, point)
                        offsets[point] = offset
                    ids.add((array, offset // capacity))
                cached = frozenset(ids)
                region_cache[cache_key] = cached
            result.update(cached)
        return frozenset(result)

    components = []
    for family in families:
        for scale in scales:
            name = f"persist.{family.name}.{scale}B"
            new_demand = 0.0
            for transition in family.transitions:
                previous = tagged_regions(transition.previous, scale)
                current = tagged_regions(transition.current, scale)
                new_demand += transition.weight * (
                    len(current - previous) / max(len(current), 1)
                )
            normalized = new_demand / max(family.transition_weight, 1.0)
            components.append(
                PersistenceComponentScore(
                    name=name,
                    transition_family=family.key.family,
                    delta=family.key.delta,
                    operation=family.key.operation,
                    region_bytes=scale,
                    weight=float(weights.get(name, 1.0)),
                    transition_count=family.transition_count,
                    transition_weight=family.transition_weight,
                    weighted_new_demand=new_demand,
                    normalized_turnover=normalized,
                )
            )
    return TemporalPersistenceScore(
        components=tuple(components),
        hardware_persist=sum(
            component.weighted_turnover for component in components
        ),
    )
