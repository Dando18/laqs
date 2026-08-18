from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .gf2 import apply_matrix
from .model import Coord, MatrixSpec


class Layout(Protocol):
    name: str
    matrix_name: str
    tile_exponents: tuple[int, ...]
    outer_order: tuple[int, ...]

    def offset(self, matrix: MatrixSpec, coord: Coord) -> int: ...

    @property
    def grammar(self) -> str: ...

    @property
    def runs(self) -> int: ...

    @property
    def xor_count(self) -> int: ...

    def signature(self) -> tuple[object, ...]: ...


def _validate_common(
    matrix: MatrixSpec,
    tile_exponents: Sequence[int],
    outer_order: Sequence[int],
) -> None:
    if len(tile_exponents) != matrix.rank:
        raise ValueError("tile exponent rank mismatch")
    for exponent, bits in zip(tile_exponents, matrix.mode_bits):
        if exponent < 0 or exponent > bits:
            raise ValueError(
                f"{matrix.name}: tile exponent {exponent} is outside [0, {bits}]"
            )
    if tuple(sorted(outer_order)) != tuple(range(matrix.rank)):
        raise ValueError("outer_order must be a permutation of mode indices")


def _outer_rank(
    matrix: MatrixSpec,
    coord: Coord,
    tile_exponents: Sequence[int],
    outer_order: Sequence[int],
) -> int:
    outer = matrix.outer_coord(coord, tile_exponents)
    tile_counts = tuple(extent >> exponent for extent, exponent in zip(matrix.shape, tile_exponents))
    rank = 0
    stride = 1
    for mode in outer_order:
        rank += outer[mode] * stride
        stride *= tile_counts[mode]
    return rank


def _flat_inner_bit_index(tile_exponents: Sequence[int], mode: int, bit: int) -> int:
    return sum(tile_exponents[:mode]) + bit


@dataclass(frozen=True)
class CanonicalLayout:
    name: str
    matrix_name: str
    tile_exponents: tuple[int, ...]
    word: tuple[int, ...]
    outer_order: tuple[int, ...]

    def validate(self, matrix: MatrixSpec) -> None:
        _validate_common(matrix, self.tile_exponents, self.outer_order)
        if self.matrix_name != matrix.name:
            raise ValueError("layout and matrix names differ")
        if len(self.word) != sum(self.tile_exponents):
            raise ValueError("canonical word length does not equal tile bit count")
        counts = [0] * matrix.rank
        for mode in self.word:
            if mode < 0 or mode >= matrix.rank:
                raise ValueError("canonical word contains an invalid mode")
            counts[mode] += 1
        if tuple(counts) != self.tile_exponents:
            raise ValueError(
                f"canonical word counts {tuple(counts)} do not match {self.tile_exponents}"
            )

    @property
    def grammar(self) -> str:
        return "canonical"

    @property
    def runs(self) -> int:
        if not self.word:
            return 0
        return 1 + sum(left != right for left, right in zip(self.word, self.word[1:]))

    @property
    def xor_count(self) -> int:
        return 0

    @property
    def inner_bits(self) -> int:
        return len(self.word)

    @property
    def tile_shape(self) -> tuple[int, ...]:
        return tuple(1 << exponent for exponent in self.tile_exponents)

    def inner_bit_order(self) -> tuple[int, ...]:
        used = [0] * len(self.tile_exponents)
        order: list[int] = []
        for mode in self.word:
            bit = used[mode]
            order.append(_flat_inner_bit_index(self.tile_exponents, mode, bit))
            used[mode] += 1
        return tuple(order)

    def matrix_rows(self) -> tuple[int, ...]:
        return tuple(1 << input_bit for input_bit in self.inner_bit_order())

    def inner_offset(self, matrix: MatrixSpec, coord: Coord) -> int:
        self.validate(matrix)
        used = [0] * matrix.rank
        result = 0
        for physical_bit, mode in enumerate(self.word):
            logical_bit = used[mode]
            result |= ((coord[mode] >> logical_bit) & 1) << physical_bit
            used[mode] += 1
        return result

    def offset(self, matrix: MatrixSpec, coord: Coord) -> int:
        matrix.validate_coord(coord)
        inner = self.inner_offset(matrix, coord)
        outer = _outer_rank(matrix, coord, self.tile_exponents, self.outer_order)
        return (outer << self.inner_bits) | inner

    def word_string(self, matrix: MatrixSpec) -> str:
        return "".join(matrix.mode_names[mode] for mode in self.word)

    def physical_bit_labels(self, matrix: MatrixSpec) -> tuple[str, ...]:
        used = [0] * matrix.rank
        labels: list[str] = []
        for mode in self.word:
            labels.append(f"{matrix.mode_names[mode]}{used[mode]}")
            used[mode] += 1
        return tuple(labels)

    def encode_plan(self, matrix: MatrixSpec) -> tuple[str, ...]:
        """Human-readable mask/shift plan, one item per maximal word run."""

        if not self.word:
            return ()
        plan: list[str] = []
        start = 0
        used_before = [0] * matrix.rank
        position = 0
        while position < len(self.word):
            mode = self.word[position]
            end = position + 1
            while end < len(self.word) and self.word[end] == mode:
                end += 1
            width = end - position
            logical_start = used_before[mode]
            physical_start = position
            plan.append(
                f"{matrix.mode_names[mode]}[{logical_start}:{logical_start + width}] "
                f"-> y[{physical_start}:{physical_start + width}]"
            )
            used_before[mode] += width
            position = end
        return tuple(plan)

    def signature(self) -> tuple[object, ...]:
        return (self.grammar, self.tile_exponents, self.word, self.outer_order)


@dataclass(frozen=True)
class LinearInnerLayout:
    name: str
    matrix_name: str
    tile_exponents: tuple[int, ...]
    a_rows: tuple[int, ...]
    outer_order: tuple[int, ...]
    basis_columns: tuple[int, ...] = ()
    active_rank: int = 0

    def validate(self, matrix: MatrixSpec) -> None:
        _validate_common(matrix, self.tile_exponents, self.outer_order)
        if self.matrix_name != matrix.name:
            raise ValueError("layout and matrix names differ")
        width = sum(self.tile_exponents)
        if len(self.a_rows) != width:
            raise ValueError("A_in row count does not match tile bit count")
        mask = (1 << width) - 1
        if any(row <= 0 or row & ~mask for row in self.a_rows):
            raise ValueError("A_in contains an invalid row mask")

    @property
    def grammar(self) -> str:
        return "linear_inner"

    @property
    def runs(self) -> int:
        # There is no unique run decomposition for a general linear circuit.
        return sum(1 for row in self.a_rows if row)

    @property
    def xor_count(self) -> int:
        return sum(max(0, row.bit_count() - 1) for row in self.a_rows)

    @property
    def inner_bits(self) -> int:
        return len(self.a_rows)

    @property
    def tile_shape(self) -> tuple[int, ...]:
        return tuple(1 << exponent for exponent in self.tile_exponents)

    def inner_offset(self, matrix: MatrixSpec, coord: Coord) -> int:
        self.validate(matrix)
        x = matrix.inner_bits(coord, self.tile_exponents)
        return apply_matrix(self.a_rows, x)

    def offset(self, matrix: MatrixSpec, coord: Coord) -> int:
        matrix.validate_coord(coord)
        inner = self.inner_offset(matrix, coord)
        outer = _outer_rank(matrix, coord, self.tile_exponents, self.outer_order)
        return (outer << self.inner_bits) | inner

    def physical_bit_labels(self, matrix: MatrixSpec) -> tuple[str, ...]:
        width = len(self.a_rows)
        labels: list[str] = []
        for row in self.a_rows:
            terms = [matrix.bit_label(bit, self.tile_exponents) for bit in range(width) if (row >> bit) & 1]
            labels.append("^".join(terms))
        return tuple(labels)

    def encode_plan(self, matrix: MatrixSpec) -> tuple[str, ...]:
        return tuple(f"y{index} = {expr}" for index, expr in enumerate(self.physical_bit_labels(matrix)))

    def signature(self) -> tuple[object, ...]:
        return (self.grammar, self.tile_exponents, self.a_rows, self.outer_order)


def canonical_layout_from_word(
    matrix: MatrixSpec,
    word: str,
    *,
    name: str | None = None,
    outer_order: Sequence[int] | None = None,
) -> CanonicalLayout:
    """Build a canonical layout from low-to-high physical-bit mode names.

    Each character selects the next unused low bit of the matching logical
    mode.  The number of occurrences of each mode therefore defines the inner
    tile shape.  Logical bits not named by ``word`` select outer tiles.
    """

    if not word:
        raise ValueError("canonical layout word cannot be empty")

    mode_indices: dict[str, int] = {}
    for index, mode_name in enumerate(matrix.mode_names):
        if len(mode_name) != 1:
            raise ValueError(
                f"{matrix.name}: canonical layout words require single-character "
                f"mode names; got {mode_name!r}"
            )
        mode_indices[mode_name] = index

    parsed_word: list[int] = []
    counts = [0] * matrix.rank
    for symbol in word:
        if symbol not in mode_indices:
            expected = ", ".join(repr(mode) for mode in matrix.mode_names)
            raise ValueError(
                f"{matrix.name}: unknown mode {symbol!r} in canonical layout word; "
                f"expected one of {expected}"
            )
        mode = mode_indices[symbol]
        counts[mode] += 1
        if counts[mode] > matrix.mode_bits[mode]:
            raise ValueError(
                f"{matrix.name}: canonical layout word uses mode {symbol!r} "
                f"{counts[mode]} times, but the extent provides only "
                f"{matrix.mode_bits[mode]} bits"
            )
        parsed_word.append(mode)

    resolved_outer_order = (
        tuple(reversed(range(matrix.rank)))
        if outer_order is None
        else tuple(outer_order)
    )
    layout = CanonicalLayout(
        name or f"canonical_{word}",
        matrix.name,
        tuple(counts),
        tuple(parsed_word),
        resolved_outer_order,
    )
    layout.validate(matrix)
    return layout


def row_major_layout(matrix: MatrixSpec, name: str = "row_major") -> CanonicalLayout:
    exponents = matrix.mode_bits
    word: list[int] = []
    for mode in reversed(range(matrix.rank)):
        word.extend([mode] * exponents[mode])
    outer_order = tuple(reversed(range(matrix.rank)))
    layout = CanonicalLayout(name, matrix.name, exponents, tuple(word), outer_order)
    layout.validate(matrix)
    return layout


def column_major_layout(matrix: MatrixSpec, name: str = "column_major") -> CanonicalLayout:
    exponents = matrix.mode_bits
    word: list[int] = []
    for mode in range(matrix.rank):
        word.extend([mode] * exponents[mode])
    outer_order = tuple(range(matrix.rank))
    layout = CanonicalLayout(name, matrix.name, exponents, tuple(word), outer_order)
    layout.validate(matrix)
    return layout
