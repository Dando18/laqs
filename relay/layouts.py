from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .gf2 import apply_matrix, invert_matrix_from_columns
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


def _flat_bit_mode(tile_exponents: Sequence[int], flat_bit: int) -> int:
    cursor = 0
    for mode, width in enumerate(tile_exponents):
        if cursor <= flat_bit < cursor + width:
            return mode
        cursor += width
    raise IndexError(flat_bit)


def linear_codegen_runs(
    a_rows: Sequence[int], tile_exponents: Sequence[int]
) -> int:
    """Count contiguous source-field groups in a linear address matrix."""

    runs = 0
    position = 0
    while position < len(a_rows):
        runs += 1
        row = a_rows[position]
        if row.bit_count() != 1:
            position += 1
            continue
        source_bit = row.bit_length() - 1
        mode = _flat_bit_mode(tile_exponents, source_bit)
        end = position + 1
        while end < len(a_rows):
            next_bit = source_bit + end - position
            if (
                a_rows[end] != 1 << next_bit
                or _flat_bit_mode(tile_exponents, next_bit) != mode
            ):
                break
            end += 1
        position = end
    return runs


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
    outer_word: tuple[int, ...] | None = None

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
        if self.outer_word is not None:
            remaining = tuple(
                bits - exponent
                for bits, exponent in zip(
                    matrix.mode_bits, self.tile_exponents
                )
            )
            counts = [0] * matrix.rank
            for mode in self.outer_word:
                if mode < 0 or mode >= matrix.rank:
                    raise ValueError("outer word contains an invalid mode")
                counts[mode] += 1
            if tuple(counts) != remaining:
                raise ValueError(
                    f"outer word counts {tuple(counts)} do not match "
                    f"remaining mode bits {remaining}"
                )

    @property
    def grammar(self) -> str:
        return "linear_inner"

    @property
    def runs(self) -> int:
        inner = linear_codegen_runs(self.a_rows, self.tile_exponents)
        outer_word = self.outer_word or ()
        if not outer_word:
            return inner
        outer = 1 + sum(
            left != right
            for left, right in zip(outer_word, outer_word[1:])
        )
        if self.a_rows and self._merge_mode() == outer_word[0]:
            outer -= 1
        return inner + outer

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
        if self.outer_word is None:
            outer = _outer_rank(
                matrix, coord, self.tile_exponents, self.outer_order
            )
        else:
            outer_coord = matrix.outer_coord(coord, self.tile_exponents)
            used = [0] * matrix.rank
            outer = 0
            for physical_bit, mode in enumerate(self.outer_word):
                outer |= (
                    (outer_coord[mode] >> used[mode]) & 1
                ) << physical_bit
                used[mode] += 1
        return (outer << self.inner_bits) | inner

    def _resolved_outer_word(self, matrix: MatrixSpec) -> tuple[int, ...]:
        if self.outer_word is not None:
            return self.outer_word
        return tuple(
            mode
            for mode in self.outer_order
            for _ in range(
                matrix.mode_bits[mode] - self.tile_exponents[mode]
            )
        )

    def _merge_mode(self) -> int | None:
        if not self.a_rows:
            return None
        row = self.a_rows[-1]
        if row.bit_count() != 1:
            return None
        source_bit = row.bit_length() - 1
        mode = _flat_bit_mode(self.tile_exponents, source_bit)
        mode_offset = sum(self.tile_exponents[:mode])
        if source_bit - mode_offset != self.tile_exponents[mode] - 1:
            return None
        return mode

    def physical_bit_labels(self, matrix: MatrixSpec) -> tuple[str, ...]:
        width = len(self.a_rows)
        labels: list[str] = []
        for row in self.a_rows:
            terms = [matrix.bit_label(bit, self.tile_exponents) for bit in range(width) if (row >> bit) & 1]
            labels.append("^".join(terms))
        return tuple(labels)

    def encode_plan(self, matrix: MatrixSpec) -> tuple[str, ...]:
        return tuple(f"y{index} = {expr}" for index, expr in enumerate(self.physical_bit_labels(matrix)))

    def descriptor(self, matrix: MatrixSpec) -> str:
        exponents = ",".join(str(value) for value in self.tile_exponents)
        rows = ",".join(f"{row:x}" for row in self.a_rows)
        outer = "".join(
            matrix.mode_names[mode]
            for mode in self._resolved_outer_word(matrix)
        )
        return f"linear:{exponents}:{rows}:{outer}"

    def evaluator_descriptor(self, matrix: MatrixSpec) -> str:
        used = [0] * matrix.rank
        symbols: list[str] = []
        offsets = matrix.bit_offsets(self.tile_exponents)
        for row in self.a_rows:
            if row.bit_count() != 1:
                return self.descriptor(matrix)
            flat_bit = row.bit_length() - 1
            mode = _flat_bit_mode(self.tile_exponents, flat_bit)
            logical_bit = flat_bit - offsets[mode]
            if logical_bit != used[mode]:
                return self.descriptor(matrix)
            used[mode] += 1
            symbols.append(matrix.mode_names[mode])
        symbols.extend(
            matrix.mode_names[mode]
            for mode in self._resolved_outer_word(matrix)
        )
        return "".join(symbols)

    def signature(self) -> tuple[object, ...]:
        return (
            self.grammar,
            self.tile_exponents,
            self.a_rows,
            self.outer_order,
            self.outer_word,
        )


@dataclass(frozen=True)
class AffineAccessLayout:
    """A fixed-basis realization of an affine-access grammar word.

    ``basis_columns`` lists the logical difference directions from low to
    high physical address. ``a_rows`` is its inverse and is the matrix used
    by address code generation. The access word contains only active access
    blocks; any inactive complement directions are fixed at the high end.
    """

    name: str
    matrix_name: str
    tile_exponents: tuple[int, ...]
    a_rows: tuple[int, ...]
    outer_order: tuple[int, ...]
    basis_columns: tuple[int, ...]
    access_word: tuple[int, ...]
    access_block_dimensions: tuple[int, ...]
    inactive_rank: int = 0

    def validate(self, matrix: MatrixSpec) -> None:
        _validate_common(matrix, self.tile_exponents, self.outer_order)
        if self.matrix_name != matrix.name:
            raise ValueError("layout and matrix names differ")
        width = sum(self.tile_exponents)
        if len(self.a_rows) != width or len(self.basis_columns) != width:
            raise ValueError("affine-access basis width does not match tile bits")
        if len(self.access_word) != sum(self.access_block_dimensions):
            raise ValueError("access word does not cover every access-block direction")
        counts = [0] * len(self.access_block_dimensions)
        for block in self.access_word:
            if block < 0 or block >= len(counts):
                raise ValueError("access word contains an invalid block")
            counts[block] += 1
        if tuple(counts) != self.access_block_dimensions:
            raise ValueError("access word counts do not match access-block dimensions")
        if self.inactive_rank != width - len(self.access_word):
            raise ValueError("inactive rank does not complete the access basis")
        mask = (1 << width) - 1
        if any(row <= 0 or row & ~mask for row in self.a_rows):
            raise ValueError("affine-access A matrix contains an invalid row mask")
        if invert_matrix_from_columns(self.basis_columns, width) != self.a_rows:
            raise ValueError("affine-access A matrix is not the inverse basis")

    @property
    def grammar(self) -> str:
        return "affine_access"

    @property
    def runs(self) -> int:
        return linear_codegen_runs(self.a_rows, self.tile_exponents)

    @property
    def xor_count(self) -> int:
        return sum(max(0, row.bit_count() - 1) for row in self.a_rows)

    @property
    def inner_bits(self) -> int:
        return len(self.a_rows)

    @property
    def active_rank(self) -> int:
        return len(self.access_word)

    @property
    def tile_shape(self) -> tuple[int, ...]:
        return tuple(1 << exponent for exponent in self.tile_exponents)

    def inner_offset(self, matrix: MatrixSpec, coord: Coord) -> int:
        if self.matrix_name != matrix.name:
            raise ValueError("layout and matrix names differ")
        return apply_matrix(
            self.a_rows, matrix.inner_bits(coord, self.tile_exponents)
        )

    def offset(self, matrix: MatrixSpec, coord: Coord) -> int:
        matrix.validate_coord(coord)
        inner = self.inner_offset(matrix, coord)
        outer = _outer_rank(
            matrix, coord, self.tile_exponents, self.outer_order
        )
        return (outer << self.inner_bits) | inner

    def physical_bit_labels(self, matrix: MatrixSpec) -> tuple[str, ...]:
        labels: list[str] = []
        for row in self.a_rows:
            terms = [
                matrix.bit_label(bit, self.tile_exponents)
                for bit in range(self.inner_bits)
                if (row >> bit) & 1
            ]
            labels.append("^".join(terms))
        return tuple(labels)

    def encode_plan(self, matrix: MatrixSpec) -> tuple[str, ...]:
        return tuple(
            f"y{index} = {expression}"
            for index, expression in enumerate(self.physical_bit_labels(matrix))
        )

    def descriptor(self) -> str:
        exponents = ",".join(str(value) for value in self.tile_exponents)
        rows = ",".join(f"{row:x}" for row in self.a_rows)
        return f"linear:{exponents}:{rows}"

    def evaluator_descriptor(self, matrix: MatrixSpec) -> str:
        """Use a compact canonical word when this matrix is canonical."""

        used = [0] * matrix.rank
        symbols: list[str] = []
        offsets = matrix.bit_offsets(self.tile_exponents)
        for row in self.a_rows:
            if row.bit_count() != 1:
                return self.descriptor()
            flat_bit = row.bit_length() - 1
            mode = _flat_bit_mode(self.tile_exponents, flat_bit)
            logical_bit = flat_bit - offsets[mode]
            if logical_bit != used[mode]:
                return self.descriptor()
            used[mode] += 1
            symbols.append(matrix.mode_names[mode])
        return "".join(symbols)

    def signature(self) -> tuple[object, ...]:
        return (
            self.grammar,
            self.tile_exponents,
            self.a_rows,
            self.outer_order,
            self.access_word,
        )


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
