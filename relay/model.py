from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log2
from numbers import Real
from typing import Iterable, Mapping, Sequence


Coord = tuple[int, ...]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def exact_log2(value: int) -> int:
    if not is_power_of_two(value):
        raise ValueError(f"expected a power of two, got {value}")
    return int(log2(value))


@dataclass(frozen=True)
class MatrixSpec:
    """Logical metadata for one dense array."""

    name: str
    shape: tuple[int, ...]
    element_bytes: int
    mode_names: tuple[str, ...]
    target: bool = True
    role: str = "read_write"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("matrix name cannot be empty")
        if len(self.shape) == 0 or len(self.shape) != len(self.mode_names):
            raise ValueError("shape and mode_names must have the same nonzero rank")
        if len(set(self.mode_names)) != len(self.mode_names):
            raise ValueError(f"mode names must be unique for {self.name}")
        if self.element_bytes <= 0:
            raise ValueError("element_bytes must be positive")
        for extent in self.shape:
            if not is_power_of_two(extent):
                raise ValueError(
                    f"{self.name}: the current solver requires power-of-two extents; got {self.shape}"
                )

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def mode_bits(self) -> tuple[int, ...]:
        return tuple(exact_log2(extent) for extent in self.shape)

    @property
    def total_bits(self) -> int:
        return sum(self.mode_bits)

    @property
    def size(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def validate_coord(self, coord: Coord) -> None:
        if len(coord) != self.rank:
            raise ValueError(f"{self.name}: coordinate {coord} has the wrong rank")
        for value, extent in zip(coord, self.shape):
            if value < 0 or value >= extent:
                raise ValueError(f"{self.name}: coordinate {coord} is out of bounds")

    def bit_offsets(self, widths: Sequence[int] | None = None) -> tuple[int, ...]:
        widths = tuple(widths if widths is not None else self.mode_bits)
        offsets: list[int] = []
        current = 0
        for width in widths:
            offsets.append(current)
            current += width
        return tuple(offsets)

    def coord_to_bits(self, coord: Coord) -> int:
        self.validate_coord(coord)
        value = 0
        for component, shift in zip(coord, self.bit_offsets()):
            value |= component << shift
        return value

    def inner_bits(self, coord: Coord, tile_exponents: Sequence[int]) -> int:
        self.validate_coord(coord)
        exponents = tuple(tile_exponents)
        if len(exponents) != self.rank:
            raise ValueError("tile exponent rank mismatch")
        value = 0
        shift = 0
        for component, exponent in zip(coord, exponents):
            if exponent < 0:
                raise ValueError("tile exponents must be nonnegative")
            mask = (1 << exponent) - 1
            value |= (component & mask) << shift
            shift += exponent
        return value

    def outer_coord(self, coord: Coord, tile_exponents: Sequence[int]) -> Coord:
        self.validate_coord(coord)
        return tuple(value >> exponent for value, exponent in zip(coord, tile_exponents))

    def bit_label(self, flat_bit: int, widths: Sequence[int] | None = None) -> str:
        widths = tuple(widths if widths is not None else self.mode_bits)
        cursor = 0
        for mode, width in zip(self.mode_names, widths):
            if cursor <= flat_bit < cursor + width:
                return f"{mode}{flat_bit - cursor}"
            cursor += width
        raise IndexError(flat_bit)


@dataclass(frozen=True)
class Access:
    """One logical element accessed by one lane/thread."""

    array: str
    coord: Coord
    lane: int | None = None
    kind: str = "read"
    width_bytes: int | None = None


@dataclass(frozen=True)
class MemoryEvent:
    """One dynamic memory-instruction event in logical coordinates."""

    id: str
    site: str
    accesses: tuple[Access, ...]
    group: str = ""
    order: int = 0
    weight: float = 1.0
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("event id cannot be empty")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("event weight must be finite and positive")
        if not self.accesses:
            raise ValueError(f"event {self.id} contains no accesses")
        keys = [key for key, _ in self.metadata]
        if len(keys) != len(set(keys)):
            raise ValueError(f"event {self.id} has duplicate metadata keys")

    @classmethod
    def make(
        cls,
        id: str,
        site: str,
        accesses: Iterable[Access],
        *,
        group: str = "",
        order: int = 0,
        weight: float = 1.0,
        metadata: Mapping[str, object] | None = None,
    ) -> "MemoryEvent":
        items = tuple(sorted((str(k), str(v)) for k, v in (metadata or {}).items()))
        return cls(id, site, tuple(accesses), group, order, weight, items)

    def meta(self, key: str, default: str | None = None) -> str | None:
        for item_key, value in self.metadata:
            if item_key == key:
                return value
        return default


@dataclass(frozen=True)
class EventSequence:
    """One exact local trace class and its dynamic multiplicity.

    ``weight`` is the number of represented executions of ``event_ids``.  An
    event may be referenced by more than one trace class; consumers count each
    reference with effective weight ``event.weight * sequence.weight``.
    """

    name: str
    event_ids: tuple[str, ...]
    weight: float = 1.0
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sequence name cannot be empty")
        if not self.event_ids:
            raise ValueError(f"sequence {self.name} contains no events")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, Real)
            or not isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("sequence weight must be finite and positive")

    @classmethod
    def make(
        cls,
        name: str,
        event_ids: Iterable[str],
        *,
        weight: float = 1.0,
        metadata: Mapping[str, object] | None = None,
    ) -> "EventSequence":
        items = tuple(sorted((str(k), str(v)) for k, v in (metadata or {}).items()))
        return cls(name, tuple(event_ids), weight, items)

    @property
    def multiplicity(self) -> float:
        """Return the exact dynamic multiplicity represented by this class."""

        return self.weight


@dataclass(frozen=True)
class EventFilter:
    """Simple reusable predicate for selecting events and accesses."""

    arrays: frozenset[str] | None = None
    sites: frozenset[str] | None = None
    kinds: frozenset[str] | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def make(
        cls,
        *,
        arrays: Iterable[str] | None = None,
        sites: Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "EventFilter":
        return cls(
            frozenset(arrays) if arrays is not None else None,
            frozenset(sites) if sites is not None else None,
            frozenset(kinds) if kinds is not None else None,
            tuple(sorted((str(k), str(v)) for k, v in (metadata or {}).items())),
        )

    def matches_event(self, event: MemoryEvent) -> bool:
        if self.sites is not None and event.site not in self.sites:
            return False
        for key, value in self.metadata:
            if event.meta(key) != value:
                return False
        return True

    def matches_access(self, access: Access) -> bool:
        if self.arrays is not None and access.array not in self.arrays:
            return False
        if self.kinds is not None and access.kind not in self.kinds:
            return False
        return True
