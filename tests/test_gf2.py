from __future__ import annotations

import unittest

from relay.gf2 import (
    apply_matrix,
    invert_matrix_from_columns,
    intersection_basis,
    matrix_rows_from_columns,
    nullspace_basis,
    rref_basis,
)


class GF2Tests(unittest.TestCase):
    def test_basis_is_canonical(self) -> None:
        self.assertEqual(rref_basis((0b011, 0b101, 0b110)), rref_basis((0b110, 0b011, 0b101)))
        self.assertEqual(len(rref_basis((0b011, 0b101, 0b110))), 2)

    def test_matrix_inverse(self) -> None:
        columns = (0b001, 0b110, 0b100)
        rows = matrix_rows_from_columns(columns, 3)
        inverse = invert_matrix_from_columns(columns, 3)
        for vector in range(8):
            encoded = apply_matrix(rows, vector)
            decoded = apply_matrix(inverse, encoded)
            self.assertEqual(decoded, vector)

    def test_nullspace_and_intersection(self) -> None:
        self.assertEqual(nullspace_basis((0b011,), 3), (0b100, 0b011))
        self.assertEqual(
            intersection_basis((0b001, 0b010), (0b011, 0b100), 3),
            (0b011,),
        )


if __name__ == "__main__":
    unittest.main()
