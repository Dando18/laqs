from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence


def parity(value: int) -> int:
    return value.bit_count() & 1


def highest_bit(value: int) -> int:
    if value == 0:
        return -1
    return value.bit_length() - 1


def rref_basis(vectors: Iterable[int]) -> tuple[int, ...]:
    """Return a unique reduced basis for a binary vector subspace."""

    rows = [int(vector) for vector in vectors if vector]
    if not rows:
        return ()
    max_bit = max(highest_bit(row) for row in rows)
    pivot_row = 0
    for bit in range(max_bit, -1, -1):
        pivot = next((index for index in range(pivot_row, len(rows)) if (rows[index] >> bit) & 1), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row]
        for index in range(len(rows)):
            if index != pivot_row and ((rows[index] >> bit) & 1):
                rows[index] ^= pivot_value
        pivot_row += 1
        if pivot_row == len(rows):
            break
    basis = [row for row in rows[:pivot_row] if row]
    basis.sort(key=highest_bit, reverse=True)
    return tuple(basis)


def reduce_vector(vector: int, basis: Sequence[int]) -> int:
    result = int(vector)
    for row in basis:
        bit = highest_bit(row)
        if bit >= 0 and ((result >> bit) & 1):
            result ^= row
    return result


def rank(vectors: Iterable[int]) -> int:
    return len(rref_basis(vectors))


def contains(basis: Sequence[int], vector: int) -> bool:
    return reduce_vector(vector, basis) == 0


def is_subspace(sub_basis: Sequence[int], super_basis: Sequence[int]) -> bool:
    return all(contains(super_basis, row) for row in sub_basis)


def add_vector(basis: Sequence[int], vector: int) -> tuple[int, ...] | None:
    new_basis = rref_basis((*basis, vector))
    if len(new_basis) == len(basis):
        return None
    return new_basis


def span_vectors(basis: Sequence[int]) -> tuple[int, ...]:
    values = [0]
    for row in basis:
        values += [value ^ row for value in values]
    return tuple(sorted(values))


def coordinate_map(basis: Sequence[int]) -> dict[int, int]:
    """Map every ambient vector in span(basis) to coefficient bits."""

    result = {0: 0}
    for index, row in enumerate(basis):
        additions = {value ^ row: coeff | (1 << index) for value, coeff in result.items()}
        result.update(additions)
    return result


def lift_coordinate(coordinate: int, basis: Sequence[int]) -> int:
    value = 0
    for index, row in enumerate(basis):
        if (coordinate >> index) & 1:
            value ^= row
    return value


def matrix_rows_from_columns(columns: Sequence[int], width: int) -> tuple[int, ...]:
    if len(columns) != width:
        raise ValueError("a square matrix needs exactly width columns")
    rows: list[int] = []
    for row_index in range(width):
        row = 0
        for column_index, column in enumerate(columns):
            if (column >> row_index) & 1:
                row |= 1 << column_index
        rows.append(row)
    return tuple(rows)


def invert_matrix_rows(rows: Sequence[int], width: int) -> tuple[int, ...]:
    if len(rows) != width:
        raise ValueError("a square matrix needs exactly width rows")
    augmented = [int(row) | (1 << (width + index)) for index, row in enumerate(rows)]
    for column in range(width):
        pivot = next((row for row in range(column, width) if (augmented[row] >> column) & 1), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column]
        for row in range(width):
            if row != column and ((augmented[row] >> column) & 1):
                augmented[row] ^= pivot_value
    return tuple((row >> width) & ((1 << width) - 1) for row in augmented)


def invert_matrix_from_columns(columns: Sequence[int], width: int) -> tuple[int, ...]:
    return invert_matrix_rows(matrix_rows_from_columns(columns, width), width)


def apply_matrix(rows: Sequence[int], vector: int) -> int:
    result = 0
    for index, row in enumerate(rows):
        result |= parity(row & vector) << index
    return result


def matrix_multiply_rows(left: Sequence[int], right: Sequence[int], width: int) -> tuple[int, ...]:
    """Return left * right, both represented as row masks."""

    columns_of_right: list[int] = []
    for column in range(width):
        value = 0
        for row_index, row in enumerate(right):
            if (row >> column) & 1:
                value |= 1 << row_index
        columns_of_right.append(value)
    output: list[int] = []
    for left_row in left:
        row = 0
        for column, right_column in enumerate(columns_of_right):
            if parity(left_row & right_column):
                row |= 1 << column
        output.append(row)
    return tuple(output)


def complement_basis(existing: Sequence[int], width: int) -> tuple[int, ...]:
    basis = rref_basis(existing)
    additions: list[int] = []
    for bit in range(width):
        vector = 1 << bit
        candidate = add_vector((*basis, *additions), vector)
        if candidate is not None:
            additions.append(vector)
        if len(basis) + len(additions) == width:
            break
    if len(basis) + len(additions) != width:
        raise ValueError("failed to construct a complement")
    return tuple(additions)


def codimension_one_subspaces(basis: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Enumerate all codimension-one subspaces of span(basis)."""

    dimension = len(basis)
    if dimension == 0:
        return ()
    result: set[tuple[int, ...]] = set()
    for functional in range(1, 1 << dimension):
        vectors: list[int] = []
        for coefficients in range(1, 1 << dimension):
            if parity(coefficients & functional) == 0:
                vectors.append(lift_coordinate(coefficients, basis))
        result.add(rref_basis(vectors))
    return tuple(sorted(result))


def enumerate_subspace_layers(width: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Simple exhaustive layer enumeration; intended for width <= 7 or 8."""

    layers: list[set[tuple[int, ...]]] = [set() for _ in range(width + 1)]
    layers[0].add(())
    nonzero = range(1, 1 << width)
    for dimension in range(width):
        for basis in layers[dimension]:
            for vector in nonzero:
                candidate = add_vector(basis, vector)
                if candidate is not None and len(candidate) == dimension + 1:
                    layers[dimension + 1].add(candidate)
    return tuple(tuple(sorted(layer)) for layer in layers)


def new_direction(previous: Sequence[int], current: Sequence[int]) -> int:
    for vector in span_vectors(current):
        if vector and not contains(previous, vector):
            return vector
    raise ValueError("current subspace does not extend previous subspace")
