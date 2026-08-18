from __future__ import annotations

import unittest

from relay.gf2 import (
    apply_matrix,
    invert_matrix_from_columns,
    matrix_rows_from_columns,
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


if __name__ == "__main__":
    unittest.main()
