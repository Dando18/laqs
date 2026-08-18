from __future__ import annotations

import unittest

from relay import MatrixSpec, canonical_layout_from_word


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


if __name__ == "__main__":
    unittest.main()
