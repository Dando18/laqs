from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from stage1_nvidia_counter_analysis import (
    DURATION_METRIC,
    aggregate_profiles,
    parse_counter_csv,
)


L1_METRIC = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"


class Stage1NvidiaCounterAnalysisTests(unittest.TestCase):
    def test_parser_uses_final_target_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counters.csv"
            fields = ["ID", "Kernel Name", L1_METRIC, DURATION_METRIC]
            rows = [
                ["unit", "", "sector", "nsecond"],
                ["0", "unrelated_kernel", "999", "999"],
                ["1", "bias_relu_kernel", "100", "10"],
                ["2", "bias_relu_kernel", "20", "2"],
                ["3", "bias_relu_kernel", "40", "4"],
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
                metric=L1_METRIC,
            )

        self.assertEqual(result["target_dispatch_count"], 3)
        self.assertEqual(result["steady_state"]["dispatch_count"], 2)
        self.assertEqual(
            result["steady_state"]["l1_cache_line_accesses"], 30.0
        )
        self.assertEqual(result["steady_state"]["duration_ns"], 3.0)
        self.assertEqual(
            [entry["index"] for entry in result["steady_dispatches"]],
            [2, 3],
        )

    def test_aggregation_uses_cross_launch_medians(self) -> None:
        profiles = [
            {
                "steady_state": {
                    "dispatch_count": 2,
                    "l1_cache_line_accesses": 10.0,
                    "duration_ns": 20.0,
                }
            },
            {
                "steady_state": {
                    "dispatch_count": 2,
                    "l1_cache_line_accesses": 30.0,
                    "duration_ns": 40.0,
                }
            },
        ]

        result = aggregate_profiles(profiles)["steady_state"]

        self.assertEqual(result["l1_cache_line_accesses"], 20.0)
        self.assertEqual(result["duration_ns"], 30.0)
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
                    metric=L1_METRIC,
                )


if __name__ == "__main__":
    unittest.main()
