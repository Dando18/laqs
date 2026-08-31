from __future__ import annotations

from argparse import Namespace
import csv
from pathlib import Path
import tempfile
import unittest

from experiments.locality_counters import (
    COUNTER_FIELDS,
    compare_summaries,
    parse_counter_csv,
    solve_kernel,
)
from experiments.layout_ranking import KERNEL_SPECS


def counter_row(index: int, kernel: str, scale: float) -> dict[str, object]:
    values = {
        "TCP_TOTAL_CACHE_ACCESSES_sum": 100 * scale,
        "TCP_TCC_READ_REQ_sum": 80 * scale,
        "TCP_TCC_WRITE_REQ_sum": 20 * scale,
        "TCC_REQ_sum": 90 * scale,
        "TCC_HIT_sum": 60 * scale,
        "TCC_MISS_sum": 30 * scale,
        "FETCH_SIZE": 2 * scale,
        "WRITE_SIZE": scale,
    }
    return {
        "Index": index,
        "KernelName": kernel,
        "BeginNs": 1000 * index,
        "EndNs": 1000 * index + 100 * scale,
        **values,
    }


class LocalityCounterTests(unittest.TestCase):
    def test_exact_dp_selects_a_lower_fine_quotient_for_atax(self) -> None:
        args = Namespace(
            hardware_profile="mi300a",
            size=8,
            block_size=8,
            block_x=8,
            block_y=8,
        )

        result = solve_kernel(KERNEL_SPECS["atax"], args)

        self.assertTrue(result["solver"]["exact"])
        self.assertEqual(result["solver"]["grammar"], "canonical")
        self.assertEqual(result["baseline"]["words"], {"A": "jjjiii"})
        self.assertEqual(result["selected"]["words"], {"A": "jiiijj"})
        self.assertEqual(
            result["model_comparison"]["selected_to_baseline_ratio"],
            2 / 3,
        )

    def test_parser_groups_atax_dispatches_into_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "atax.csv"
            fieldnames = [
                "Index",
                "KernelName",
                "BeginNs",
                "EndNs",
                *COUNTER_FIELDS,
            ]
            rows = []
            for operation, scale in enumerate((9.0, 1.0, 2.0)):
                rows.extend(
                    (
                        counter_row(2 * operation, "atax_tmp_kernel", scale),
                        counter_row(2 * operation + 1, "atax_y_kernel", scale),
                    )
                )
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = parse_counter_csv(
                path,
                dispatch_names=("atax_tmp_kernel", "atax_y_kernel"),
                steady_operations=2,
            )

        self.assertEqual(result["target_dispatch_count"], 6)
        self.assertEqual(
            result["cold_first_operation"]["l1_to_l2_read_requests"],
            1440,
        )
        self.assertEqual(
            result["steady_state"]["l1_to_l2_read_requests"],
            240,
        )
        self.assertEqual(result["steady_state"]["l2_hit_rate_percent"], 200 / 3)
        self.assertEqual(result["steady_state"]["hbm_read_bytes"], 6144)

    def test_comparison_reports_reductions_and_zero_baselines(self) -> None:
        baseline = {
            "l1_cache_line_accesses": 200,
            "l1_to_l2_read_requests": 100,
            "l1_to_l2_write_requests": 20,
            "l1_to_l2_total_requests": 120,
            "l2_tag_requests": 110,
            "l2_hits": 80,
            "l2_misses": 30,
            "hbm_read_bytes": 0,
            "hbm_write_bytes": 10,
            "duration_ns": 50,
            "l2_hit_rate_percent": 70,
        }
        selected = {
            **baseline,
            "l1_to_l2_read_requests": 75,
            "l1_to_l2_total_requests": 95,
            "hbm_read_bytes": 0,
            "l2_hit_rate_percent": 72,
        }

        comparison = compare_summaries(baseline, selected)

        self.assertEqual(comparison["l1_to_l2_read_requests"]["reduction"], 0.25)
        self.assertTrue(comparison["l1_to_l2_read_requests"]["fewer"])
        self.assertIsNone(comparison["hbm_read_bytes"]["reduction"])
        self.assertEqual(
            comparison["l2_hit_rate_percent"]["percentage_point_change"], 2
        )


if __name__ == "__main__":
    unittest.main()
