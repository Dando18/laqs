from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from stage15_analysis import COUNTER_FIELDS, counter_comparison, parse_counter_csv


class CounterAnalysisTests(unittest.TestCase):
    def test_parser_uses_only_final_profiled_dispatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counters.csv"
            fieldnames = ["Index", "KernelName", *COUNTER_FIELDS, "BeginNs", "EndNs"]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for index, requests in enumerate((999, 120, 100)):
                    writer.writerow(
                        {
                            "Index": index,
                            "KernelName": "gemm_prepacked_b_kernel.kd",
                            "TCP_TCC_READ_REQ_sum": requests,
                            "TCC_REQ_sum": 200,
                            "TCC_HIT_sum": 150,
                            "TCC_MISS_sum": 50,
                            "FETCH_SIZE": 2,
                            "WRITE_SIZE": 1,
                            "MemUnitStalled": 4,
                            "TCP_TCP_TA_DATA_STALL_CYCLES_sum": 500,
                            "MfmaUtil": 6,
                            "TOTAL_16_OPS": 1_000_000,
                            "BeginNs": 100,
                            "EndNs": 1100,
                        }
                    )

            result = parse_counter_csv(path, profile_iterations=2)

        self.assertEqual(result["summary"]["profiled_dispatch_count"], 2)
        self.assertEqual(result["summary"]["l1_to_l2_read_requests"], 110)
        self.assertEqual(result["summary"]["l2_hit_rate_percent"], 75)
        self.assertEqual(result["summary"]["hbm_read_bytes"], 2048)
        self.assertEqual(result["summary"]["hbm_bandwidth_gbps"], 3.072)

    def test_comparison_reports_traffic_reduction(self) -> None:
        default = {
            "summary": {
                "l1_to_l2_read_requests": 200,
                "l2_tag_requests": 220,
                "l2_misses": 20,
                "hbm_read_bytes": 0,
                "duration_ns": 20,
            }
        }
        selected = {
            "summary": {
                "l1_to_l2_read_requests": 150,
                "l2_tag_requests": 160,
                "l2_misses": 10,
                "hbm_read_bytes": 0,
                "duration_ns": 16,
            }
        }

        comparison = counter_comparison(
            default,
            selected,
            selected_to_default_b_request_ratio=0.5,
        )

        self.assertEqual(comparison["l1_to_l2_read_request_reduction"], 0.25)
        self.assertEqual(comparison["l2_tag_request_reduction"], 3 / 11)
        self.assertEqual(comparison["l2_miss_reduction"], 0.5)
        self.assertIsNone(comparison["hbm_read_byte_reduction"])
        self.assertEqual(comparison["profiled_duration_speedup"], 1.25)
        decomposition = comparison["inferred_request_decomposition"]
        self.assertEqual(decomposition["l1_to_l2_reads"]["default_b_requests"], 100)
        self.assertEqual(decomposition["l1_to_l2_reads"]["selected_b_requests"], 50)


if __name__ == "__main__":
    unittest.main()
