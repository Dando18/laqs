from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.layout_ranking import (
    KERNEL_SPECS,
    TimingResult,
    average_tie_ranks,
    evaluator_command,
    parse_arguments,
    parse_evaluator_output,
    run,
    runtime_rank_ranges,
    traditional_layout_cases,
    variation_aware_rank_metrics,
)


def timing(median: float, lower: float, upper: float) -> TimingResult:
    return TimingResult(
        "mock GPU",
        median,
        median,
        lower,
        (upper - lower) / 2.0,
        1.0,
        (lower, median, upper),
    )


class RankHelpersTests(unittest.TestCase):
    def test_average_tie_ranks_are_one_based_and_ascending(self) -> None:
        self.assertEqual(
            average_tie_ranks([10.0, 20.0, 20.0, 5.0]),
            [2.0, 3.5, 3.5, 1.0],
        )

    def test_runtime_rank_ranges_keep_overlapping_samples_unordered(self) -> None:
        timings = (
            timing(1.1, 1.0, 1.2),
            timing(1.2, 1.1, 1.3),
            timing(2.0, 1.9, 2.1),
        )
        self.assertEqual(runtime_rank_ranges(timings), [(1, 2), (1, 2), (3, 3)])

    def test_variation_aware_metric_accepts_a_raw_swap_within_ranges(self) -> None:
        timings = (
            timing(1.1, 1.0, 1.2),
            timing(1.2, 1.1, 1.3),
            timing(2.0, 1.9, 2.1),
        )
        metric = variation_aware_rank_metrics([2.0, 1.0, 3.0], timings)
        self.assertEqual(metric["accurate_layouts"], 3)
        self.assertEqual(metric["rank_accuracy"], 1.0)
        self.assertEqual(metric["mean_rank_error"], 0.0)

        contradicted = variation_aware_rank_metrics([3.0, 2.0, 1.0], timings)
        self.assertEqual(contradicted["accurate_layouts"], 1)
        self.assertAlmostEqual(contradicted["mean_rank_error"], 1.0)
        self.assertEqual(contradicted["max_rank_error"], 2.0)


class EvaluatorTests(unittest.TestCase):
    OUTPUT = """\
Device: AMD Radeon Graphics
GESUMMV FP64: N=256, block=128, samples=3, iterations=5, warmup=3
Layouts: A=jjjjiiii B=jjjjiiii (words are low -> high bits)
Correctness: PASS (5 points, max abs 1.000e-14, max rel 2.000e-15)

Results (kernel time only; packing and validation are excluded):
median_ms  mean_ms  min_ms  sd_ms  GFLOP/s
   1.2500   1.3000  1.2000  0.1000    26.84
Samples (ms): 1.2000 1.2500 1.4500
"""

    def test_parse_shared_evaluator_output(self) -> None:
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

    def test_commands_use_each_kernel_evaluator_contract(self) -> None:
        _, args = parse_arguments(["--score-only"])
        case = traditional_layout_cases(8)[0]
        gemm = evaluator_command(KERNEL_SPECS["gemm"], 8, case, args)
        gesummv = evaluator_command(KERNEL_SPECS["gesummv"], 8, case, args)

        self.assertEqual(gemm[2:5], [case.word, case.word, case.word])
        self.assertIn("--block-x", gemm)
        self.assertEqual(gesummv[2:4], [case.word, case.word])
        self.assertIn("--block-size", gesummv)


class CombinedExperimentTests(unittest.TestCase):
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

    def test_score_only_writes_all_kernel_size_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "ranking.json"
            markdown_path = Path(directory) / "tables.md"
            with redirect_stdout(StringIO()):
                status = run(
                    [
                        "--kernel",
                        "gemm",
                        "--kernel",
                        "gesummv",
                        "--size",
                        "8",
                        "--size",
                        "16",
                        "--score-only",
                        "--output",
                        str(json_path),
                        "--markdown",
                        str(markdown_path),
                    ]
                )

            self.assertEqual(status, 0)
            report = json.loads(json_path.read_text())
            self.assertEqual(
                report["experiment"], "multi-kernel-traditional-layout-ranking"
            )
            self.assertEqual(len(report["runs"]), 4)
            self.assertEqual(report["benchmark_run_order"], [])
            self.assertTrue(
                all(
                    result["timing"] is None
                    for group in report["runs"]
                    for result in group["results"]
                )
            )
            self.assertEqual(report["component_weight_overrides"], {})
            self.assertTrue(
                all(group["component_weights"] for group in report["runs"])
            )
            self.assertTrue(
                any(
                    objective["provenance"] == "hypothesis"
                    for group in report["runs"]
                    for objective in group["objectives"]
                )
            )
            markdown = markdown_path.read_text()
            self.assertIn("## GEMM — N=8", markdown)
            self.assertIn("## GEMM — N=16", markdown)
            self.assertIn("## GESUMMV — N=8", markdown)
            self.assertIn("## GESUMMV — N=16", markdown)
            self.assertIn("### Objective model", markdown)
            self.assertIn(
                "| Objective | Provenance | Region B | Tau | Meaning |",
                markdown,
            )
            self.assertIn("| Score rank | Layout |", markdown)

    def test_runtime_report_keeps_raw_ranks_and_uses_variation_for_metrics(self) -> None:
        medians = {
            "row_major": (4.0, 3.8, 4.2),
            "column_major": (3.0, 2.8, 3.2),
            "tile8_row_major": (1.1, 1.0, 1.2),
            "tile8_column_major": (1.2, 1.1, 1.3),
        }

        def benchmark(spec, n, case, args):
            del spec, n, args
            values = medians[case.name]
            result = timing(*values)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "ranking.json"
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                status = run(
                    [
                        "--kernel",
                        "gesummv",
                        "--size",
                        "8",
                        "--output",
                        str(json_path),
                    ]
                )

            self.assertEqual(status, 0)
            report = json.loads(json_path.read_text())
            group = report["runs"][0]
            records = group["results"]
            runtime_values = [record["timing"]["median_ms"] for record in records]
            self.assertEqual(
                [record["runtime_rank"] for record in records],
                average_tie_ranks(runtime_values),
            )
            timings = [
                TimingResult(
                    record["timing"]["device"],
                    record["timing"]["median_ms"],
                    record["timing"]["mean_ms"],
                    record["timing"]["min_ms"],
                    record["timing"]["sd_ms"],
                    record["timing"]["gflops"],
                    tuple(record["timing"]["samples_ms"]),
                )
                for record in records
            ]
            for mode, observed in group["variation_aware_rank_metrics"].items():
                scores = [record["aggregate_scores"][mode] for record in records]
                self.assertEqual(
                    observed,
                    variation_aware_rank_metrics(scores, timings),
                )

            markdown = json_path.with_suffix(".md").read_text()
            self.assertIn("| Score rank | Runtime rank |", markdown)
            self.assertIn("### Variation-aware metrics", markdown)
            self.assertIn("Observed range ms", markdown)

    def test_benchmark_checkpoint_resumes_without_rescoring(self) -> None:
        def benchmark(spec, n, case, args):
            del spec, n, args
            median = float(len(case.name))
            result = timing(median, median - 0.1, median + 0.1)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "ranking.json"
            base_arguments = [
                "--kernel",
                "gesummv",
                "--size",
                "8",
                "--output",
                str(json_path),
            ]
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                first_status = run(
                    [*base_arguments, "--max-benchmarks", "2"]
                )

            self.assertEqual(first_status, 0)
            checkpoint = json.loads(json_path.read_text())
            self.assertFalse(checkpoint["complete"])
            self.assertEqual(
                sum(
                    record["timing"] is not None
                    for record in checkpoint["runs"][0]["results"]
                ),
                2,
            )
            self.assertIn("metrics pending", json_path.with_suffix(".md").read_text())

            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), patch(
                "experiments.layout_ranking.score_group",
                side_effect=AssertionError("resume rebuilt scores"),
            ), redirect_stdout(StringIO()):
                second_status = run([*base_arguments, "--resume"])

            self.assertEqual(second_status, 0)
            completed = json.loads(json_path.read_text())
            self.assertTrue(completed["complete"])
            self.assertEqual(len(completed["benchmark_run_order"]), 4)
            self.assertTrue(
                all(
                    record["timing"] is not None
                    for record in completed["runs"][0]["results"]
                )
            )


if __name__ == "__main__":
    unittest.main()
