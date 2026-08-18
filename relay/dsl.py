from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Mapping

from .model import Access, Coord, EventSequence, MemoryEvent


def lane_accesses(
    array: str,
    lanes: int | Iterable[int],
    coordinate: Callable[[int], Coord],
    *,
    kind: str = "read",
    width_bytes: int | None = None,
) -> tuple[Access, ...]:
    """Build one logical access per lane from a coordinate function."""

    lane_values = range(lanes) if isinstance(lanes, int) else lanes
    return tuple(
        Access(array, tuple(coordinate(lane)), lane=lane, kind=kind, width_bytes=width_bytes)
        for lane in lane_values
    )


def lane_event(
    id: str,
    site: str,
    array: str,
    lanes: int | Iterable[int],
    coordinate: Callable[[int], Coord],
    *,
    kind: str = "read",
    width_bytes: int | None = None,
    group: str = "",
    order: int = 0,
    weight: float = 1.0,
    metadata: Mapping[str, object] | None = None,
) -> MemoryEvent:
    """Convenience constructor for the common one-array, one-access-per-lane event."""

    return MemoryEvent.make(
        id,
        site,
        lane_accesses(array, lanes, coordinate, kind=kind, width_bytes=width_bytes),
        group=group,
        order=order,
        weight=weight,
        metadata=metadata,
    )


def sequence(
    name: str,
    *event_ids: str,
    weight: float = 1.0,
    metadata: Mapping[str, object] | None = None,
) -> EventSequence:
    return EventSequence.make(name, event_ids, weight=weight, metadata=metadata)
