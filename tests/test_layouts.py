from __future__ import annotations

import unittest
from itertools import product

from relay import (
    MatrixSpec,
    ResourceMap,
    apply_flag_preserving_shears,
    canonical_layout_from_word,
    enumerate_flag_preserving_swizzles,
    layout_codegen_runs,
    layout_matrix_rows,
    low_address_flag,
    resource_color_destination_bits,
    tiled_row_major_layout,
)


class CanonicalLayoutWordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MatrixSpec("A", (8, 4), 4, ("i", "j"))

    def test_full_word_maps_logical_bits_from_low_to_high(self) -> None:
        layout = canonical_layout_from_word(self.matrix, "iijij")

        self.assertEqual(layout.name, "canonical_iijij")
        self.assertEqual(layout.word, (0, 0, 1, 0, 1))
        self.assertEqual(layout.tile_exponents, (3, 2))
        self.assertEqual(layout.tile_shape, (8, 4))
        self.assertEqual(layout.outer_order, (1, 0))
        self.assertEqual(layout.physical_bit_labels(self.matrix), ("i0", "i1", "j0", "i2", "j1"))
        self.assertEqual(layout.offset(self.matrix, (5, 3)), 0b11101)

    def test_shorter_word_defines_an_inner_tile(self) -> None:
        layout = canonical_layout_from_word(
            self.matrix,
            "ji",
            name="two_by_two_row_inner",
        )

        self.assertEqual(layout.name, "two_by_two_row_inner")
        self.assertEqual(layout.tile_exponents, (1, 1))
        self.assertEqual(layout.tile_shape, (2, 2))
        self.assertEqual(layout.outer_order, (1, 0))
        # (2, 3) is in outer tile (1, 1), whose row-major tile rank is 3.
        # Its inner coordinate (0, 1) has low-to-high word bits j0, i0 = 01.
        self.assertEqual(layout.offset(self.matrix, (2, 3)), 13)
        self.assertEqual(
            layout_matrix_rows(self.matrix, layout),
            (1 << 3, 1 << 0, 1 << 4, 1 << 1, 1 << 2),
        )
        self.assertEqual(layout_codegen_runs(self.matrix, layout), 4)

    def test_tiled_row_major_fixes_row_major_outer_tile_order(self) -> None:
        layout = tiled_row_major_layout(self.matrix, (1, 1))

        self.assertEqual(layout.word, (1, 0))
        self.assertEqual(layout.outer_order, (1, 0))
        self.assertEqual(
            layout_matrix_rows(self.matrix, layout),
            (1 << 3, 1 << 0, 1 << 4, 1 << 1, 1 << 2),
        )

    def test_invalid_words_are_rejected(self) -> None:
        invalid = (
            ("", "cannot be empty"),
            ("ix", "unknown mode 'x'"),
            ("iiii", "provides only 3 bits"),
            ("jjj", "provides only 2 bits"),
        )
        for word, message in invalid:
            with self.subTest(word=word):
                with self.assertRaisesRegex(ValueError, message):
                    canonical_layout_from_word(self.matrix, word)


class FlagFiberLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MatrixSpec("A", (8, 4), 4, ("i", "j"))
        self.layout = canonical_layout_from_word(self.matrix, "jijii")

    def test_upper_triangular_shear_preserves_every_flag_prefix(self) -> None:
        identity = apply_flag_preserving_shears(
            self.matrix, self.layout, ()
        )
        swizzled = apply_flag_preserving_shears(
            self.matrix, self.layout, ((1, 4),)
        )

        self.assertEqual(
            low_address_flag(self.matrix, identity),
            low_address_flag(self.matrix, swizzled),
        )
        self.assertEqual(
            swizzled.basis_columns[4],
            identity.basis_columns[4] ^ identity.basis_columns[1],
        )
        for coord in product(range(8), range(4)):
            original = self.layout.offset(self.matrix, coord)
            expected = original ^ (((original >> 4) & 1) << 1)
            self.assertEqual(swizzled.offset(self.matrix, coord), expected)

    def test_sparse_enumeration_targets_only_resource_color_rows(self) -> None:
        resource_map = ResourceMap(
            "test_color",
            8,
            (1 << 3,),
            "simd_window.t4.cohort.load",
        )
        destinations = resource_color_destination_bits(
            (resource_map,), self.matrix.element_bytes, self.matrix.total_bits
        )
        seeds = enumerate_flag_preserving_swizzles(
            self.matrix,
            self.layout,
            max_xors=1,
            destination_bits=destinations,
        )

        self.assertEqual(destinations, (1,))
        self.assertEqual(
            [seed.shears for seed in seeds],
            [(), ((1, 2),), ((1, 3),), ((1, 4),)],
        )
        self.assertEqual(len({seed.layout.a_rows for seed in seeds}), 4)

    def test_non_flag_preserving_shear_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "i < j"):
            apply_flag_preserving_shears(
                self.matrix, self.layout, ((3, 1),)
            )


if __name__ == "__main__":
    unittest.main()
