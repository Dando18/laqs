from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from stage1_counter_analysis import (
    COUNTER_FIELDS,
    aggregate_profiles,
    compare_summaries,
    linear_fit,
    parse_counter_csv,
    rank_correlation,
)


def counter_row(index: int, kernel: str, scale: float):
    return {
        "Index": index,
        "KernelName": kernel,
        "BeginNs": index * 1000,
        "EndNs": index * 1000 + scale * 100,
        "TCP_TOTAL_CACHE_ACCESSES_sum": 100 * scale,
        "TCP_TCC_READ_REQ_sum": 80 * scale,
        "TCP_TCC_WRITE_REQ_sum": 20 * scale,
        "TCC_REQ_sum": 90 * scale,
        "TCC_HIT_sum": 60 * scale,
        "TCC_MISS_sum": 30 * scale,
        "FETCH_SIZE": 2 * scale,
        "WRITE_SIZE": scale,
    }


class Stage1CounterAnalysisTests(unittest.TestCase):
    def test_parser_uses_only_final_target_dispatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counters.csv"
            fieldnames = [
                "Index",
                "KernelName",
                "BeginNs",
                "EndNs",
                *COUNTER_FIELDS,
            ]
            rows = [
                counter_row(0, "unrelated_kernel", 100),
                counter_row(1, "gemv_kernel.kd", 99),
                counter_row(2, "gemv_kernel.kd", 1),
                counter_row(3, "gemv_kernel.kd", 2),
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = parse_counter_csv(
                path, kernel_name="gemv_kernel", profile_iterations=2
            )

        summary = result["steady_state"]
        self.assertEqual(result["target_dispatch_count"], 3)
        self.assertEqual(summary["l1_to_l2_read_requests"], 120)
        self.assertEqual(summary["l2_hit_rate_percent"], 200 / 3)
        self.assertEqual(summary["hbm_read_bytes"], 3072)
        self.assertEqual(summary["hbm_bandwidth_gbps"], 30.72)

    def test_aggregation_and_comparison_use_launch_medians(self) -> None:
        first = {
            "steady_state": {
                "dispatch_count": 2,
                **{
                    field: 10.0
                    for field in (
                        "l1_cache_line_accesses",
                        "l1_to_l2_read_requests",
                        "l1_to_l2_write_requests",
                        "l1_to_l2_total_requests",
                        "l2_tag_requests",
                        "l2_hits",
                        "l2_misses",
                        "hbm_read_bytes",
                        "hbm_write_bytes",
                        "hbm_total_bytes",
                        "duration_ns",
                        "hbm_bandwidth_gbps",
                        "l2_hit_rate_percent",
                    )
                },
            }
        }
        second = {
            "steady_state": {
                **first["steady_state"],
                "l1_to_l2_read_requests": 20.0,
            }
        }
        aggregate = aggregate_profiles((first, second))
        selected = {**aggregate["steady_state"]}
        selected["l1_to_l2_read_requests"] = 12.0
        comparison = compare_summaries(aggregate["steady_state"], selected)

        self.assertEqual(
            aggregate["steady_state"]["l1_to_l2_read_requests"], 15.0
        )
        self.assertAlmostEqual(
            comparison["l1_to_l2_read_requests"]["reduction"], 0.2
        )

    def test_rank_correlation_handles_ties(self) -> None:
        self.assertAlmostEqual(
            rank_correlation((1, 1, 2, 3), (4, 4, 2, 1)), -1.0
        )
        self.assertIsNone(rank_correlation((1, 1), (1, 2)))

    def test_linear_fit_has_free_intercept(self) -> None:
        fit = linear_fit((1, 2, 3), (5, 7, 9))

        self.assertAlmostEqual(fit["slope"], 2.0)
        self.assertAlmostEqual(fit["intercept"], 3.0)
        self.assertAlmostEqual(fit["r_squared"], 1.0)


if __name__ == "__main__":
    unittest.main()
