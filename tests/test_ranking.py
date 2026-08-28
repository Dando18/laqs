from __future__ import annotations

import unittest

from relay import spearman_rank_correlation, summarize_rank_quality


class RankCorrelationTests(unittest.TestCase):
    def test_spearman_correlation_handles_ties(self) -> None:
        self.assertAlmostEqual(
            spearman_rank_correlation((1, 1, 2, 3), (4, 3, 2, 1)),
            -0.9486832980505138,
        )

    def test_spearman_correlation_reports_constant_ranking(self) -> None:
        self.assertIsNone(spearman_rank_correlation((1, 1, 1), (1, 2, 3)))


class RankQualityTests(unittest.TestCase):
    @staticmethod
    def record(
        candidate_id: str,
        quotient_score: float,
        runtime_ms: float,
        flag_id: str,
        mapping_id: str,
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "quotient_score": quotient_score,
            "runtime_ms": runtime_ms,
            "flag_id": flag_id,
            "mapping_id": mapping_id,
        }

    def test_summary_reports_regret_and_distinct_same_flag_realizations(self) -> None:
        summary = summarize_rank_quality(
            (
                self.record("a", 1, 12, "f1", "m1"),
                self.record("b", 1, 10, "f1", "m2"),
                self.record("c", 2, 9, "f2", "m3"),
                self.record("d", 3, 11, "f3", "m4"),
            )
        )

        self.assertAlmostEqual(summary["regret"]["top_1"]["regret"], 1 / 3)
        self.assertEqual(summary["regret"]["top_3"]["regret"], 0.0)
        equal_score = summary["equal_quotient_score_runtime_spread"]
        self.assertEqual(equal_score[0]["candidate_ids"], ["a", "b"])
        self.assertAlmostEqual(equal_score[0]["relative_spread"], 0.2)
        same_flag = summary["same_flag_runtime_spread"]
        self.assertTrue(same_flag["available"])
        self.assertEqual(same_flag["groups"][0]["distinct_mapping_count"], 2)

    def test_identical_mappings_are_noise_controls_not_fiber_variants(self) -> None:
        summary = summarize_rank_quality(
            (
                self.record("a", 1, 10, "f1", "m1"),
                self.record("b", 1, 11, "f1", "m1"),
            )
        )

        self.assertFalse(summary["same_flag_runtime_spread"]["available"])
        duplicate = summary["duplicate_mapping_runtime_spread"]
        self.assertEqual(duplicate[0]["mapping_id"], "m1")
        self.assertAlmostEqual(duplicate[0]["relative_spread"], 0.1)


if __name__ == "__main__":
    unittest.main()
