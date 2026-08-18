from __future__ import annotations

import math
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.gemm_layout_ranking import (
    TimingResult,
    average_tie_ranks,
    parse_evaluator_output,
    run,
    spearman_correlation,
    traditional_layout_cases,
)


class RankHelpersTests(unittest.TestCase):
    def test_average_tie_ranks_are_one_based_and_ascending(self) -> None:
        self.assertEqual(
            average_tie_ranks([10.0, 20.0, 20.0, 5.0]),
            [2.0, 3.5, 3.5, 1.0],
        )

    def test_spearman_handles_perfect_and_tied_rankings(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [6.0, 5.0, 4.0]),
            -1.0,
        )
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
            3.0 / math.sqrt(10.0),
        )

    def test_spearman_is_undefined_for_a_constant_ranking(self) -> None:
        self.assertIsNone(
            spearman_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        )


class EvaluatorParserTests(unittest.TestCase):
    OUTPUT = """\
Generated /tmp/relay-gemm/generated_gemm.cu
Compiling: hipcc -O3 generated_gemm.cu -o generated_gemm
Running /tmp/relay-gemm/generated_gemm
Device: AMD Radeon Graphics
GEMM FP64: N=256, block=32x32, samples=3, iterations=5, warmup=3
Layouts: A=jjjjiiii B=jjjjiiii C=jjjjiiii (words are low -> high bits)
Correctness: PASS (5 points, max abs 1.000e-14, max rel 2.000e-15)

Results (kernel time only; packing and validation are excluded):
median_ms  mean_ms  min_ms  sd_ms  GFLOP/s
   1.2500   1.3000  1.2000  0.1000    26.84
Samples (ms): 1.2000 1.2500 1.4500
"""

    def test_parse_evaluator_output(self) -> None:
        result = parse_evaluator_output(self.OUTPUT)
        self.assertEqual(result.device, "AMD Radeon Graphics")
        self.assertEqual(result.median_ms, 1.25)
        self.assertEqual(result.mean_ms, 1.3)
        self.assertEqual(result.min_ms, 1.2)
        self.assertEqual(result.sd_ms, 0.1)
        self.assertEqual(result.gflops, 26.84)
        self.assertEqual(result.samples_ms, (1.2, 1.25, 1.45))

    def test_parse_requires_successful_correctness(self) -> None:
        with self.assertRaisesRegex(ValueError, "Correctness: PASS"):
            parse_evaluator_output(self.OUTPUT.replace("PASS", "FAIL"))


class TraditionalCasesTests(unittest.TestCase):
    def test_tile_cases_wider_than_n_are_omitted(self) -> None:
        self.assertEqual(
            [case.name for case in traditional_layout_cases(8)],
            [
                "row_major",
                "column_major",
                "tile8_row_major",
                "tile8_column_major",
            ],
        )

    def test_score_only_experiment_writes_ranked_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "ranking.json"
            terminal = StringIO()
            with redirect_stdout(terminal):
                status = run(
                    [
                        "--n",
                        "8",
                        "--score-only",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(status, 0)
            report = json.loads(output_path.read_text())
            self.assertEqual(
                report["experiment"], "gemm-traditional-layout-ranking"
            )
            self.assertEqual(len(report["results"]), 4)
            self.assertEqual(report["benchmark_run_order"], [])
            self.assertIsNone(report["spearman_correlation"])
            self.assertEqual(
                report["score_mode_spearman_correlations"],
                {
                    "peak-normalized-excess": None,
                    "weighted-normalized-excess": None,
                    "weighted-region-count": None,
                },
            )
            self.assertTrue(
                all(result["timing"] is None for result in report["results"])
            )
            self.assertTrue(
                all(
                    set(result["aggregate_scores"])
                    == {
                        "weighted-region-count",
                        "peak-normalized-excess",
                        "weighted-normalized-excess",
                    }
                    for result in report["results"]
                )
            )
            self.assertIn("score rank", terminal.getvalue())

    def test_one_runtime_pass_correlates_every_score_mode(self) -> None:
        medians = {
            "row_major": 4.0,
            "column_major": 3.0,
            "tile8_row_major": 1.0,
            "tile8_column_major": 2.0,
        }

        def benchmark(case, args):
            del args
            median = medians[case.name]
            timing = TimingResult(
                "mock GPU", median, median, median, 0.0, 1.0, (median,)
            )
            return timing, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "ranking.json"
            with patch(
                "experiments.gemm_layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                status = run(["--n", "8", "--output", str(output_path)])

            self.assertEqual(status, 0)
            report = json.loads(output_path.read_text())
            runtime_values = [
                result["timing"]["median_ms"] for result in report["results"]
            ]
            for mode, observed in report[
                "score_mode_spearman_correlations"
            ].items():
                score_values = [
                    result["aggregate_scores"][mode]
                    for result in report["results"]
                ]
                self.assertAlmostEqual(
                    observed,
                    spearman_correlation(score_values, runtime_values),
                )


if __name__ == "__main__":
    unittest.main()
