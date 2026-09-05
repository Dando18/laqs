from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from stage1_nvidia_counter_analysis import (
    COUNTER_METRICS,
    DURATION_METRIC,
    FIRST_LEVEL_COUNTER,
    GLOBAL_LOAD_REQUESTS_COUNTER,
    HBM_READ_COUNTER,
    L1_TO_L2_READ_COUNTER,
    L2_READ_MISS_COUNTER,
    NATIVE_UNIT,
    SUMMARY_METRICS,
    TEX_SOURCE_L2_READ_REQUESTS_COUNTER,
    aggregate_profiles,
    parse_counter_csv,
)


def counter_row(index: int, kernel: str, scale: float) -> list[str]:
    values = {
        FIRST_LEVEL_COUNTER: 20 * scale,
        GLOBAL_LOAD_REQUESTS_COUNTER: 10 * scale,
        TEX_SOURCE_L2_READ_REQUESTS_COUNTER: 6 * scale,
        L1_TO_L2_READ_COUNTER: 8 * scale,
        L2_READ_MISS_COUNTER: 3 * scale,
        HBM_READ_COUNTER: 96 * scale,
        DURATION_METRIC: 2 * scale,
    }
    return [str(index), kernel, *(str(values[field]) for field in COUNTER_METRICS)]


class Stage1NvidiaCounterAnalysisTests(unittest.TestCase):
    def test_parser_uses_final_target_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counters.csv"
            fields = ["ID", "Kernel Name", *COUNTER_METRICS]
            rows = [
                ["unit", "", *("unit" for _field in COUNTER_METRICS)],
                counter_row(0, "unrelated_kernel", 99),
                counter_row(1, "bias_relu_kernel", 5),
                counter_row(2, "bias_relu_kernel", 1),
                counter_row(3, "bias_relu_kernel", 2),
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                stream.write("==WARNING== synthetic profiler warning\n")
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)

            result = parse_counter_csv(
                path,
                kernel_name="bias_relu_kernel",
                profile_iterations=2,
            )

        self.assertEqual(result["target_dispatch_count"], 3)
        self.assertEqual(result["steady_state"]["dispatch_count"], 2)
        self.assertEqual(
            result["steady_state"]["first_level_memory_accesses"], 30.0
        )
        self.assertEqual(result["steady_state"]["duration_ns"], 3.0)
        self.assertEqual(result["steady_state"]["l1_to_l2_read_traffic"], 12.0)
        self.assertEqual(
            result["steady_state"]["tex_source_l2_read_requests"], 9.0
        )
        self.assertEqual(result["steady_state"]["l1_miss_demand_to_l2"], 9.0)
        self.assertEqual(result["steady_state"]["l2_read_work"], 12.0)
        self.assertEqual(result["steady_state"]["l2_read_misses"], 4.5)
        self.assertEqual(result["steady_state"]["hbm_read_bytes"], 144.0)
        self.assertEqual(result["steady_state"]["global_load_requests"], 15.0)
        self.assertEqual(result["steady_state"]["sectors_per_request"], 2.0)
        self.assertEqual(result["steady_state"]["native_unit"], NATIVE_UNIT)
        self.assertEqual(
            result["steady_state"]["native_counters"][FIRST_LEVEL_COUNTER],
            30.0,
        )
        self.assertNotIn("l1_cache_line_accesses", result["steady_state"])
        self.assertEqual(
            [entry["index"] for entry in result["steady_dispatches"]],
            [2, 3],
        )

    def test_aggregation_uses_cross_launch_medians(self) -> None:
        profiles = [
            {
                "steady_state": {
                    "dispatch_count": 2,
                    **{field: 10.0 for field in SUMMARY_METRICS},
                    "native_counters": {
                        metric: 10.0 for metric in COUNTER_METRICS
                    },
                }
            },
            {
                "steady_state": {
                    "dispatch_count": 2,
                    **{field: 30.0 for field in SUMMARY_METRICS},
                    "native_counters": {
                        metric: 30.0 for metric in COUNTER_METRICS
                    },
                }
            },
        ]

        result = aggregate_profiles(profiles)["steady_state"]

        self.assertEqual(result["first_level_memory_accesses"], 20.0)
        self.assertEqual(result["sectors_per_request"], 20.0)
        self.assertEqual(result["duration_ns"], 20.0)
        self.assertEqual(result["native_unit"], NATIVE_UNIT)
        self.assertEqual(
            result["native_counters"][FIRST_LEVEL_COUNTER], 20.0
        )
        self.assertEqual(result["profile_launch_count"], 2)
        self.assertEqual(result["dispatches_per_launch"], [2, 2])

    def test_parser_rejects_missing_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counters.csv"
            path.write_text(
                '"ID","Kernel Name","gpu__time_duration.sum"\n'
                '"0","bias_relu_kernel","10"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no Nsight Compute"):
                parse_counter_csv(
                    path,
                    kernel_name="bias_relu_kernel",
                    profile_iterations=1,
                )


if __name__ == "__main__":
    unittest.main()
