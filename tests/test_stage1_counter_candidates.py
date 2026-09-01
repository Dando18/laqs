from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from relay import Hyperedge, MatrixSpec, ObjectiveComponent
from stage1_counter_candidates import (
    canonical_counter_panel,
    random_linear_counter_panel,
)
from stage1_counter_sweep import _gesummv_theory


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

    def test_gesummv_theory_uses_execution_layout_width(self) -> None:
        def panel_record(*, h100: bool, transaction_bytes: int = 128):
            words = [
                "rrrrrrcccccc",
                "crrrrrrccccc",
                "ccrrrrrrcccc",
                "cccrrrrrrccc",
                "ccccrrrrrrcc",
                "cccccrrrrrrc",
            ]
            lane_bases = [[value] for value in (2, 4, 8, 16, 32)]
            register_bases = [[1]]
            dynamic_cohorts = 32_768
            if not h100:
                lane_bases = [[value] for value in (1, 2, 4, 8, 16, 32)]
                register_bases = []
                dynamic_cohorts = 16_384
            lane_bits = [value[0].bit_length() - 1 for value in lane_bases]
            low_address_dimension = (transaction_bytes // 4).bit_length() - 1
            candidates = []
            scores = set()
            for index, word in enumerate(words, 1):
                row_bit = 0
                column_bit = 10
                physical_rows = []
                for mode in word:
                    if mode == "r":
                        physical_rows.append(1 << row_bit)
                        row_bit += 1
                    else:
                        physical_rows.append(1 << column_bit)
                        column_bit += 1
                low_rows = physical_rows[:low_address_dimension]
                low_lane_bits = sum(
                    any(row & (1 << bit) for row in low_rows)
                    for bit in lane_bits
                )
                score = dynamic_cohorts * (
                    1 << (len(lane_bits) - low_lane_bits)
                )
                if score in scores:
                    continue
                scores.add(score)
                candidates.append(
                    {
                        "candidate_id": f"candidate_{index}",
                        "a_rows": physical_rows,
                        "quotient_score": score,
                    }
                )
            return {
                "operand_shape": [1024, 1024],
                "execution_layout": {
                    "bases": {
                        "lane": lane_bases,
                        "register": register_bases,
                    },
                    "input_sizes": {
                        "lane": 32 if h100 else 64,
                        "register": 2 if h100 else 1,
                    },
                    "output_shape": [64],
                },
                "panel": {
                    "enumerated_word_count": 924,
                    "unique_mapping_count": 924,
                    "quotient_levels": [
                        candidate["quotient_score"]
                        for candidate in candidates
                    ],
                    "candidates": candidates,
                },
            }

        args = SimpleNamespace(
            case="gesummv",
            tile_shape=(64, 64),
            transaction_bytes=128,
        )
        mi300a = _gesummv_theory(
            args, panel_record(h100=False), "fixed_tile_levels"
        )
        h100 = _gesummv_theory(
            args, panel_record(h100=True), "fixed_tile_levels"
        )

        self.assertEqual(mi300a["row_issue_dimension"], 6)
        self.assertEqual(mi300a["register_issue_count"], 1)
        self.assertEqual(mi300a["expected_quotient_scores"][0], 32_768)
        self.assertEqual(h100["row_issue_dimension"], 5)
        self.assertEqual(h100["register_issue_count"], 2)
        self.assertEqual(h100["expected_quotient_scores"][0], 65_536)
        self.assertEqual(h100["expected_quotient_scores"][-1], 1_048_576)

        args.transaction_bytes = 64
        mi300a_q64 = _gesummv_theory(
            args,
            panel_record(h100=False, transaction_bytes=64),
            "fixed_tile_levels",
        )
        args.transaction_bytes = 32
        h100_q32 = _gesummv_theory(
            args,
            panel_record(h100=True, transaction_bytes=32),
            "fixed_tile_levels",
        )

        self.assertEqual(
            mi300a_q64["expected_quotient_scores"],
            [65_536, 131_072, 262_144, 524_288, 1_048_576],
        )
        self.assertEqual(
            h100_q32["expected_quotient_scores"],
            [262_144, 524_288, 1_048_576],
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

    def test_random_linear_panel_is_reproducible_and_distinct(self) -> None:
        first = random_linear_counter_panel(
            self.matrix,
            self.component,
            ((64, 1), (64, 2), (64, 4)),
            samples=20,
            seed=17,
        )
        repeated = random_linear_counter_panel(
            self.matrix,
            self.component,
            ((64, 1), (64, 2), (64, 4)),
            samples=20,
            seed=17,
        )
        changed = random_linear_counter_panel(
            self.matrix,
            self.component,
            ((64, 1), (64, 2), (64, 4)),
            samples=20,
            seed=18,
        )

        first_rows = [candidate["a_rows"] for candidate in first["candidates"]]
        self.assertEqual(
            first_rows,
            [candidate["a_rows"] for candidate in repeated["candidates"]],
        )
        self.assertNotEqual(
            first_rows,
            [candidate["a_rows"] for candidate in changed["candidates"]],
        )
        self.assertEqual(first["unique_mapping_count"], 20)
        self.assertEqual(len({tuple(rows) for rows in first_rows}), 20)
        self.assertTrue(
            all(
                candidate["grammar"] == "linear_inner"
                for candidate in first["candidates"]
            )
        )
        self.assertEqual(
            sum(tile["sample_count"] for tile in first["tile_grammars"]),
            20,
        )

    def test_random_linear_panel_supports_one_dimensional_kernels(self) -> None:
        matrix = MatrixSpec("bias", (1024,), 4, ("feature",))
        edge = Hyperedge.make(
            ((feature,) for feature in range(64)),
            weight=1024,
            source="bias.issue",
        )
        component = ObjectiveComponent("issue", 64, {"bias": (edge,)})

        panel = random_linear_counter_panel(
            matrix,
            component,
            ((32,), (64,), (128,), (256,)),
            samples=100,
            seed=0,
        )

        self.assertEqual(panel["representative_count"], 100)
        self.assertEqual(
            len({candidate["mapping_id"] for candidate in panel["candidates"]}),
            100,
        )


if __name__ == "__main__":
    unittest.main()
