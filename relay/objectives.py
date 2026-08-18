from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable, Mapping, Protocol, Sequence

from .model import (
    Access,
    Coord,
    EventFilter,
    EventSequence,
    MatrixSpec,
    MemoryEvent,
    exact_log2,
    is_power_of_two,
)


@dataclass(frozen=True)
class Hyperedge:
    """A weighted set of logical elements from one array."""

    points: tuple[Coord, ...]
    weight: float = 1.0
    source: str = ""

    @classmethod
    def make(
        cls, points: Iterable[Coord], *, weight: float = 1.0, source: str = ""
    ) -> "Hyperedge":
        unique = tuple(sorted(set(tuple(point) for point in points)))
        if not unique:
            raise ValueError("hyperedge cannot be empty")
        if weight <= 0:
            raise ValueError("hyperedge weight must be positive")
        return cls(unique, weight, source)


@dataclass(frozen=True)
class ObjectiveComponent:
    """One access scope evaluated at one aligned byte granularity."""

    name: str
    region_bytes: int
    edges_by_array: Mapping[str, tuple[Hyperedge, ...]]
    provenance: str = "hypothesis"
    search: bool = True
    description: str = ""

    def capacity_elements(self, matrix: MatrixSpec) -> int:
        if self.region_bytes % matrix.element_bytes != 0:
            raise ValueError(
                f"objective {self.name}: {self.region_bytes} B is not divisible by "
                f"{matrix.name}'s {matrix.element_bytes} B element width"
            )
        return self.region_bytes // matrix.element_bytes

    def dimension(self, matrix: MatrixSpec) -> int:
        capacity = self.capacity_elements(matrix)
        if not is_power_of_two(capacity):
            raise ValueError(
                f"objective {self.name}: capacity {capacity} elements is not a power of two"
            )
        return exact_log2(capacity)

    def packing_bound(self, matrix: MatrixSpec) -> float:
        capacity = self.capacity_elements(matrix)
        total = 0.0
        for edge in self.edges_by_array.get(matrix.name, ()):  # distinct scalar elements
            total += edge.weight * ceil(len(edge.points) / capacity)
        return total


class ObjectiveSpec(Protocol):
    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]: ...


def _selected_accesses(event: MemoryEvent, event_filter: EventFilter) -> list[Access]:
    if not event_filter.matches_event(event):
        return []
    return [access for access in event.accesses if event_filter.matches_access(access)]


@dataclass(frozen=True)
class SimultaneousRegions:
    """Use each memory event, or each contiguous lane subgroup, as one edge."""

    name: str
    region_bytes: int
    event_filter: EventFilter = field(default_factory=EventFilter)
    lane_group: int | None = None
    weight: float = 1.0
    provenance: str = "grounded"
    search: bool = True
    description: str = ""

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        del sequences
        edges: dict[str, list[Hyperedge]] = {}
        for event in events.values():
            accesses = _selected_accesses(event, self.event_filter)
            buckets: dict[tuple[str, int], list[Coord]] = {}
            for access in accesses:
                if access.array not in matrices:
                    raise ValueError(f"event {event.id}: unknown array {access.array}")
                if self.lane_group is None:
                    group = 0
                else:
                    if self.lane_group <= 0:
                        raise ValueError("lane_group must be positive")
                    if access.lane is None:
                        raise ValueError(
                            f"event {event.id}: lane_group objective requires lane ids"
                        )
                    group = access.lane // self.lane_group
                buckets.setdefault((access.array, group), []).append(access.coord)
            for (array, group), points in buckets.items():
                source = event.id if self.lane_group is None else f"{event.id}:lanes{group}"
                edge = Hyperedge.make(
                    points,
                    weight=event.weight * self.weight,
                    source=source,
                )
                edges.setdefault(array, []).append(edge)
        return [
            ObjectiveComponent(
                self.name,
                self.region_bytes,
                {name: tuple(items) for name, items in edges.items()},
                self.provenance,
                self.search,
                self.description,
            )
        ]


@dataclass(frozen=True)
class LanePrefixRegions:
    """Convenience builder for several contiguous-lane scope/granularity pairs."""

    name_prefix: str
    levels: tuple[tuple[int, int], ...]
    event_filter: EventFilter = field(default_factory=EventFilter)
    weight: float = 1.0
    provenance: str = "hypothesis"
    search: bool = True

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        components: list[ObjectiveComponent] = []
        for group_size, region_bytes in self.levels:
            spec = SimultaneousRegions(
                name=f"{self.name_prefix}.lane{group_size}.{region_bytes}B",
                region_bytes=region_bytes,
                event_filter=self.event_filter,
                lane_group=group_size,
                weight=self.weight,
                provenance=self.provenance,
                search=self.search,
                description=f"contiguous groups of {group_size} lanes",
            )
            components.extend(spec.build(matrices, events, sequences))
        return components


@dataclass(frozen=True)
class PerLaneTemporalRegions:
    """Use temporal windows from each lane's filtered accesses as edges."""

    name_prefix: str
    region_bytes: int
    windows: tuple[int, ...]
    event_filter: EventFilter = field(default_factory=EventFilter)
    stride: int = 1
    sequence_names: frozenset[str] | None = None
    weight: float = 1.0
    provenance: str = "hypothesis"
    search: bool = True
    description: str = ""

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        if not self.windows:
            raise ValueError("per-lane temporal windows cannot be empty")
        if any(window <= 0 for window in self.windows):
            raise ValueError("per-lane temporal windows must be positive")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("per-lane temporal windows must be unique")
        if self.stride <= 0:
            raise ValueError("per-lane temporal stride must be positive")

        edges_by_window: dict[int, dict[str, list[Hyperedge]]] = {
            window: {} for window in self.windows
        }
        for sequence in sequences:
            if self.sequence_names is not None and sequence.name not in self.sequence_names:
                continue

            accesses_by_lane: dict[tuple[str, int], list[Coord]] = {}
            for event_id in sequence.event_ids:
                event = events[event_id]
                for access in _selected_accesses(event, self.event_filter):
                    if access.array not in matrices:
                        raise ValueError(f"event {event.id}: unknown array {access.array}")
                    if access.lane is None:
                        raise ValueError(
                            f"event {event.id}: per-lane temporal objective requires lane ids"
                        )
                    accesses_by_lane.setdefault((access.array, access.lane), []).append(
                        access.coord
                    )

            for (array, lane), lane_accesses in accesses_by_lane.items():
                for window in self.windows:
                    for start in range(0, len(lane_accesses) - window + 1, self.stride):
                        edge = Hyperedge.make(
                            lane_accesses[start : start + window],
                            weight=sequence.weight * self.weight,
                            source=(
                                f"{sequence.name}:{array}:lane{lane}"
                                f"[{start}:{start + window}]"
                            ),
                        )
                        edges_by_window[window].setdefault(array, []).append(edge)

        return [
            ObjectiveComponent(
                f"{self.name_prefix}.window{window}",
                self.region_bytes,
                {
                    name: tuple(items)
                    for name, items in edges_by_window[window].items()
                },
                self.provenance,
                self.search,
                self.description
                or f"one lane over {window} consecutive filtered accesses",
            )
            for window in self.windows
        ]


@dataclass(frozen=True)
class TemporalWindowRegions:
    """Union accesses from sliding windows in explicitly ordered local sequences."""

    name: str
    region_bytes: int
    window: int | None = None
    stride: int = 1
    sequence_names: frozenset[str] | None = None
    event_filter: EventFilter = field(default_factory=EventFilter)
    weight: float = 1.0
    provenance: str = "hypothesis"
    search: bool = True
    include_partial: bool = False
    description: str = ""

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        if self.stride <= 0:
            raise ValueError("temporal window stride must be positive")
        edges: dict[str, list[Hyperedge]] = {}
        for sequence in sequences:
            if self.sequence_names is not None and sequence.name not in self.sequence_names:
                continue
            ordered = [events[event_id] for event_id in sequence.event_ids]
            window = len(ordered) if self.window is None else self.window
            if window <= 0:
                raise ValueError("temporal window must be positive")
            if self.include_partial:
                starts = range(0, len(ordered), self.stride)
            else:
                starts = range(0, max(0, len(ordered) - window + 1), self.stride)
            for start in starts:
                selected = ordered[start : start + window]
                if not selected:
                    continue
                by_array: dict[str, list[Coord]] = {}
                for event in selected:
                    for access in _selected_accesses(event, self.event_filter):
                        if access.array not in matrices:
                            raise ValueError(f"event {event.id}: unknown array {access.array}")
                        by_array.setdefault(access.array, []).append(access.coord)
                for array, points in by_array.items():
                    edge = Hyperedge.make(
                        points,
                        weight=sequence.weight * self.weight,
                        source=f"{sequence.name}[{start}:{start + len(selected)}]",
                    )
                    edges.setdefault(array, []).append(edge)
        return [
            ObjectiveComponent(
                self.name,
                self.region_bytes,
                {name: tuple(items) for name, items in edges.items()},
                self.provenance,
                self.search,
                self.description,
            )
        ]


@dataclass(frozen=True)
class GroupedRegions:
    """Union events that share metadata keys, group id, site, or order."""

    name: str
    region_bytes: int
    group_by: tuple[str, ...]
    event_filter: EventFilter = field(default_factory=EventFilter)
    weight: float = 1.0
    weight_mode: str = "min"
    provenance: str = "hypothesis"
    search: bool = True
    description: str = ""

    def _field(self, event: MemoryEvent, key: str) -> str:
        if key == "group":
            return event.group
        if key == "site":
            return event.site
        if key == "order":
            return str(event.order)
        value = event.meta(key)
        if value is None:
            raise ValueError(f"event {event.id} lacks metadata key {key!r}")
        return value

    def _group_weight(self, members: Sequence[MemoryEvent]) -> float:
        values = [event.weight for event in members]
        if self.weight_mode == "min":
            return min(values)
        if self.weight_mode == "max":
            return max(values)
        if self.weight_mode == "sum":
            return sum(values)
        if self.weight_mode == "one":
            return 1.0
        raise ValueError(f"unknown grouped weight_mode {self.weight_mode!r}")

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        del sequences
        groups: dict[tuple[str, ...], list[MemoryEvent]] = {}
        for event in events.values():
            if self.event_filter.matches_event(event):
                key = tuple(self._field(event, item) for item in self.group_by)
                groups.setdefault(key, []).append(event)
        edges: dict[str, list[Hyperedge]] = {}
        for key, members in groups.items():
            by_array: dict[str, list[Coord]] = {}
            for event in members:
                for access in event.accesses:
                    if self.event_filter.matches_access(access):
                        if access.array not in matrices:
                            raise ValueError(f"event {event.id}: unknown array {access.array}")
                        by_array.setdefault(access.array, []).append(access.coord)
            group_weight = self._group_weight(members) * self.weight
            for array, points in by_array.items():
                edge = Hyperedge.make(
                    points,
                    weight=group_weight,
                    source=f"group:{'/'.join(key)}",
                )
                edges.setdefault(array, []).append(edge)
        return [
            ObjectiveComponent(
                self.name,
                self.region_bytes,
                {name: tuple(items) for name, items in edges.items()},
                self.provenance,
                self.search,
                self.description,
            )
        ]


@dataclass(frozen=True)
class ExplicitRegions:
    """Escape hatch: provide exact hyperedges directly."""

    name: str
    region_bytes: int
    edges_by_array: Mapping[str, tuple[Hyperedge, ...]]
    provenance: str = "hypothesis"
    search: bool = True
    description: str = ""

    def build(
        self,
        matrices: Mapping[str, MatrixSpec],
        events: Mapping[str, MemoryEvent],
        sequences: Sequence[EventSequence],
    ) -> list[ObjectiveComponent]:
        del events, sequences
        for array in self.edges_by_array:
            if array not in matrices:
                raise ValueError(f"explicit objective {self.name}: unknown array {array}")
        return [
            ObjectiveComponent(
                self.name,
                self.region_bytes,
                self.edges_by_array,
                self.provenance,
                self.search,
                self.description,
            )
        ]


def build_objectives(
    specs: Iterable[ObjectiveSpec],
    matrices: Mapping[str, MatrixSpec],
    events: Mapping[str, MemoryEvent],
    sequences: Sequence[EventSequence],
) -> tuple[ObjectiveComponent, ...]:
    result: list[ObjectiveComponent] = []
    names: set[str] = set()
    for spec in specs:
        for component in spec.build(matrices, events, sequences):
            if component.name in names:
                raise ValueError(f"duplicate objective name {component.name!r}")
            names.add(component.name)
            result.append(component)
    return tuple(result)
