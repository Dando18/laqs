from __future__ import annotations

from pathlib import Path
import sys
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from relay import Hyperedge, MatrixSpec, ObjectiveComponent
from stage1_counter_candidates import canonical_counter_panel


class Stage1CounterCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MatrixSpec(
            "A", (1024, 1024), 4, ("row", "column")
        )
        edge = Hyperedge.make(
            ((row, 0) for row in range(64)),
            weight=16_384,
            source="gesummv.issue",
        )
        self.component = ObjectiveComponent(
            "issue", 128, {"A": (edge,)}
        )

    def test_fixed_64_square_panel_has_six_expected_levels(self) -> None:
        panel = canonical_counter_panel(
            self.matrix,
            self.component,
            ((64, 64),),
            group_by_tile=False,
        )

        expected = [32_768 * (1 << exponent) for exponent in range(6)]
        self.assertEqual(panel["enumerated_word_count"], 924)
        self.assertEqual(panel["unique_mapping_count"], 924)
        self.assertEqual(panel["representative_count"], 6)
        self.assertEqual(panel["quotient_levels"], expected)
        self.assertEqual(
            len({candidate["mapping_id"] for candidate in panel["candidates"]}),
            6,
        )
        self.assertEqual(
            [candidate["inner_word"] for candidate in panel["candidates"]],
            [
                "rrrrrrcccccc",
                "crrrrrrccccc",
                "ccrrrrrrcccc",
                "cccrrrrrrccc",
                "ccccrrrrrrcc",
                "cccccrrrrrrc",
            ],
        )
        for candidate in panel["candidates"]:
            low_row_bits = candidate["inner_mode_order"][:5].count("row")
            self.assertEqual(
                candidate["quotient_score"],
                16_384 * (1 << (6 - low_row_bits)),
            )

    def test_tile_panel_retains_one_representative_per_tile_level(self) -> None:
        panel = canonical_counter_panel(
            self.matrix,
            self.component,
            ((64, 1), (64, 2), (64, 4)),
            group_by_tile=True,
        )

        groups = [
            (tuple(candidate["inner_tile_shape"]), candidate["quotient_score"])
            for candidate in panel["candidates"]
        ]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(
            len({candidate["mapping_id"] for candidate in panel["candidates"]}),
            len(panel["candidates"]),
        )
        self.assertEqual(panel["enumerated_word_count"], 1 + 7 + 28)


if __name__ == "__main__":
    unittest.main()
