from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.layout_ranking import (
    KERNEL_SPECS,
    PARETO_OBJECTIVES,
    TimingResult,
    average_tie_ranks,
    evaluator_command,
    layout_cases,
    parse_arguments,
    parse_evaluator_output,
    run,
    runtime_rank_ranges,
    selected_layout_cases,
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
        case = layout_cases(8)[0]
        atax = evaluator_command(KERNEL_SPECS["atax"], 8, case, args)
        gemm = evaluator_command(KERNEL_SPECS["gemm"], 8, case, args)
        gesummv = evaluator_command(KERNEL_SPECS["gesummv"], 8, case, args)
        mvt = evaluator_command(KERNEL_SPECS["mvt"], 8, case, args)
        syrk = evaluator_command(KERNEL_SPECS["syrk"], 8, case, args)

        self.assertEqual(atax[2:3], [case.word])
        self.assertIn("--block-size", atax)
        self.assertEqual(gemm[2:5], [case.word, case.word, case.word])
        self.assertIn("--block-x", gemm)
        self.assertEqual(gesummv[2:4], [case.word, case.word])
        self.assertIn("--block-size", gesummv)
        self.assertEqual(mvt[2:3], [case.word])
        self.assertIn("--block-size", mvt)
        self.assertEqual(syrk[2:4], [case.word, case.word])
        self.assertIn("--block-x", syrk)


class CombinedExperimentTests(unittest.TestCase):
    def test_default_kernel_registry_contains_five_nontrivial_kernels(self) -> None:
        self.assertEqual(
            set(KERNEL_SPECS),
            {"atax", "gemm", "gesummv", "mvt", "syrk"},
        )

    def test_tile_cases_wider_than_n_are_omitted(self) -> None:
        cases = layout_cases(8)
        self.assertEqual([case.name for case in cases[:4]], [
            "row_major",
            "column_major",
            "tile8_row_major",
            "tile8_column_major",
        ])
        self.assertFalse(any(case.name.startswith("tile16") for case in cases))
        self.assertEqual(
            len(
                {
                    case.word
                    for case in cases
                    if len(case.word) == 6
                    and case.word.count("i") == case.word.count("j") == 3
                }
            ),
            20,
        )

    def test_rectangular_and_interleaved_cases_are_explicit(self) -> None:
        cases = {case.name: case.word for case in layout_cases(32)}

        self.assertEqual(cases["tile8x16_row_major"], "jjjjiii")
        self.assertEqual(cases["tile8x16_column_major"], "iiijjjj")
        self.assertEqual(cases["tile32x8_row_major"], "jjjiiiii")
        self.assertEqual(cases["tile32x8_column_major"], "iiiiijjj")
        self.assertEqual(cases["tile16_interleaved"], "jijijiji")
        self.assertEqual(cases["tile32_interleaved"], "jijijijiji")
        self.assertEqual(len(cases), 73)
        self.assertEqual(
            len(
                {
                    word
                    for word in cases.values()
                    if len(word) == 6 and word.count("i") == 3
                }
            ),
            20,
        )
        self.assertEqual(
            len(
                {
                    word
                    for word in cases.values()
                    if len(word) == 7 and word.count("i") == 3
                }
            ),
            35,
        )

    def test_named_layout_subset_preserves_requested_order(self) -> None:
        cases = selected_layout_cases(
            32,
            ["tile16_interleaved", "row_major", "tile16_interleaved"],
        )

        self.assertEqual(
            [case.name for case in cases],
            ["tile16_interleaved", "row_major"],
        )
        with self.assertRaisesRegex(ValueError, "N=8 does not provide"):
            selected_layout_cases(8, ["tile32_row_major"])

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
                report["experiment"], "multi-kernel-layout-ranking"
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
            self.assertIsNone(report["frontier_analysis"])
            self.assertFalse(
                json_path.with_name(json_path.stem + "_plots").exists()
            )
            self.assertTrue(
                all(group["component_weights"] for group in report["runs"])
            )
            for group in report["runs"]:
                interleaved = next(
                    (
                        result
                        for result in group["results"]
                        if result["name"] == "tile16_interleaved"
                    ),
                    None,
                )
                if group["matrix_size"] == 16:
                    self.assertIsNotNone(interleaved)
                    target_count = 3 if group["kernel"] == "gemm" else 2
                    self.assertEqual(
                        interleaved["score"]["codegen"]["runs"],
                        8 * target_count,
                    )
                    self.assertEqual(interleaved["score"]["codegen"]["xors"], 0)
                    self.assertEqual(
                        len(interleaved["score"]["codegen"]["arrays"]),
                        target_count,
                    )
            self.assertTrue(
                any(
                    objective["provenance"] == "hypothesis"
                    for group in report["runs"]
                    for objective in group["objectives"]
                )
            )
            for group in report["runs"]:
                frontier = group["pareto_frontier"]
                self.assertEqual(
                    [objective["name"] for objective in frontier["objectives"]],
                    list(PARETO_OBJECTIVES),
                )
                member_names = {
                    member["name"] for member in frontier["members"]
                }
                self.assertTrue(member_names)
                self.assertEqual(
                    member_names,
                    {
                        result["name"]
                        for result in group["results"]
                        if result["pareto_frontier_member"]
                    },
                )
                gated = group["fine_locality_gated_frontiers"]
                self.assertEqual(
                    [item["delta"] for item in gated],
                    [0.0, 0.01, 0.05, 0.1],
                )
                self.assertTrue(
                    all(item["members"] and item["eligible_count"] for item in gated)
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
            self.assertIn("### Score Pareto frontier", markdown)
            self.assertIn(
                "| Layout | Q fine | J peak | J area | Runs | XORs |",
                markdown,
            )
            self.assertIn("| Score rank | Layout |", markdown)
            self.assertIn("| Score | Runs | XORs |", markdown)

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
                        "--layout-case",
                        "row_major",
                        "--layout-case",
                        "column_major",
                        "--layout-case",
                        "tile8_row_major",
                        "--layout-case",
                        "tile8_column_major",
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

            frontier_analysis = report["frontier_analysis"]
            self.assertEqual(frontier_analysis["instance_count"], 1)
            self.assertIn("oracle_regret", frontier_analysis)
            self.assertIn("epsilon_optimal", frontier_analysis)
            self.assertIn("top_k", frontier_analysis)
            self.assertIn("frontier_runtime_analysis", group)
            self.assertEqual(
                set(frontier_analysis["plots"]),
                {
                    "epsilon_optimal_coverage",
                    "retained_fraction_vs_regret",
                    "purity_and_enrichment",
                    "top_k_regret",
                    "tau_weight_robustness",
                },
            )
            self.assertTrue(
                all(
                    Path(plot["path"]).stat().st_size > 0
                    for plot in frontier_analysis["plots"].values()
                )
            )

            markdown = json_path.with_suffix(".md").read_text()
            self.assertIn("| Score rank | Runtime rank |", markdown)
            self.assertIn("### Variation-aware metrics", markdown)
            self.assertIn("Observed range ms", markdown)
            self.assertIn("## Frontier candidate-generation scorecard", markdown)
            self.assertIn("### Retained fraction versus oracle regret", markdown)
            self.assertIn("### Epsilon-optimal coverage", markdown)
            self.assertIn("### Top-k scalar-score regret", markdown)
            self.assertIn("![Top-k scalar-score regret]", markdown)

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
                "--layout-case",
                "row_major",
                "--layout-case",
                "column_major",
                "--layout-case",
                "tile8_row_major",
                "--layout-case",
                "tile8_column_major",
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

    def test_completed_timings_can_be_reused_for_fresh_scores(self) -> None:
        def benchmark(spec, n, case, args):
            del spec, n, args
            median = float(len(case.name))
            result = timing(median, median - 0.1, median + 0.1)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            final_path = Path(directory) / "final.json"
            common = [
                "--kernel",
                "gesummv",
                "--size",
                "8",
            ]
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                self.assertEqual(
                    run([*common, "--output", str(baseline_path)]),
                    0,
                )

            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=AssertionError("reused report ran a benchmark"),
            ), redirect_stdout(StringIO()):
                self.assertEqual(
                    run(
                        [
                            *common,
                            "--reuse-timings",
                            str(baseline_path),
                            "--output",
                            str(final_path),
                        ]
                    ),
                    0,
                )

            report = json.loads(final_path.read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(
                report["timing_sources"], [str(baseline_path.resolve())]
            )
            self.assertEqual(
                report["runs"][0]["benchmark_run_order"],
                json.loads(baseline_path.read_text())["runs"][0][
                    "benchmark_run_order"
                ],
            )
            self.assertTrue(
                all(
                    record["timing"] is not None
                    for record in report["runs"][0]["results"]
                )
            )
            self.assertIn(
                "Runtime samples were reused",
                final_path.with_suffix(".md").read_text(),
            )

    def test_partial_timing_seed_prefills_only_matching_layouts(self) -> None:
        def benchmark(spec, n, case, args):
            del spec, n, args
            result = timing(float(len(case.name)), 0.9, 1.1)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        selected = [
            "row_major",
            "column_major",
            "tile8_row_major",
            "tile8_column_major",
        ]
        with tempfile.TemporaryDirectory() as directory:
            seed_path = Path(directory) / "seed.json"
            expanded_path = Path(directory) / "expanded.json"
            subset_arguments = [
                "--kernel",
                "gesummv",
                "--size",
                "8",
                *sum((["--layout-case", name] for name in selected), []),
            ]
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                self.assertEqual(
                    run([*subset_arguments, "--output", str(seed_path)]), 0
                )

            with redirect_stdout(StringIO()):
                self.assertEqual(
                    run(
                        [
                            "--kernel",
                            "gesummv",
                            "--size",
                            "8",
                            "--prepare-checkpoint",
                            "--seed-timings",
                            str(seed_path),
                            "--output",
                            str(expanded_path),
                        ]
                    ),
                    0,
                )

            expanded = json.loads(expanded_path.read_text())
            records = expanded["runs"][0]["results"]
            self.assertEqual(sum(record["timing"] is not None for record in records), 4)
            self.assertFalse(expanded["complete"])
            self.assertEqual(
                expanded["seed_timing_sources"], [str(seed_path.resolve())]
            )

    def test_timing_reuse_rejects_incompatible_benchmark_settings(self) -> None:
        def benchmark(spec, n, case, args):
            del spec, n, args
            result = timing(1.0, 0.9, 1.1)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            final_path = Path(directory) / "final.json"
            common = ["--kernel", "gesummv", "--size", "8"]
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                self.assertEqual(
                    run(
                        [
                            *common,
                            "--samples",
                            "3",
                            "--output",
                            str(baseline_path),
                        ]
                    ),
                    0,
                )

            with redirect_stdout(StringIO()), redirect_stderr(
                StringIO()
            ), self.assertRaises(SystemExit) as error:
                run(
                    [
                        *common,
                        "--samples",
                        "4",
                        "--reuse-timings",
                        str(baseline_path),
                        "--output",
                        str(final_path),
                    ]
                )
            self.assertEqual(error.exception.code, 2)

    def test_disjoint_timing_reports_can_be_combined(self) -> None:
        def benchmark(spec, n, case, args):
            del spec, args
            median = float(n + len(case.name))
            result = timing(median, median - 0.1, median + 0.1)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"n{n}.json" for n in (8, 16)]
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), redirect_stdout(StringIO()):
                for n, path in zip((8, 16), paths):
                    self.assertEqual(
                        run(
                            [
                                "--kernel",
                                "gesummv",
                                "--size",
                                str(n),
                                "--output",
                                str(path),
                            ]
                        ),
                        0,
                    )

            combined_path = Path(directory) / "combined.json"
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=AssertionError("combined report ran a benchmark"),
            ), redirect_stdout(StringIO()):
                self.assertEqual(
                    run(
                        [
                            "--kernel",
                            "gesummv",
                            "--size",
                            "8",
                            "--size",
                            "16",
                            "--reuse-timings",
                            str(paths[0]),
                            "--reuse-timings",
                            str(paths[1]),
                            "--output",
                            str(combined_path),
                        ]
                    ),
                    0,
                )

            combined = json.loads(combined_path.read_text())
            self.assertTrue(combined["complete"])
            self.assertEqual(len(combined["runs"]), 2)
            self.assertEqual(
                combined["timing_sources"],
                [str(path.resolve()) for path in paths],
            )
            self.assertEqual(
                len(combined["benchmark_run_order"]),
                sum(
                    len(json.loads(path.read_text())["benchmark_run_order"])
                    for path in paths
                ),
            )

    def test_prepare_checkpoint_can_be_resumed_without_rescoring(self) -> None:
        def benchmark(spec, n, case, args):
            del spec, n, args
            result = timing(1.0, 0.9, 1.1)
            return result, ["mock-evaluator"], "Correctness: PASS", ""

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "prepared.json"
            arguments = [
                "--kernel",
                "gesummv",
                "--size",
                "8",
                "--output",
                str(output_path),
            ]
            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=AssertionError("preparation ran a benchmark"),
            ), redirect_stdout(StringIO()):
                self.assertEqual(
                    run([*arguments, "--prepare-checkpoint"]),
                    0,
                )

            prepared = json.loads(output_path.read_text())
            self.assertFalse(prepared["complete"])
            self.assertTrue(
                all(
                    record["timing"] is None
                    for record in prepared["runs"][0]["results"]
                )
            )

            with patch(
                "experiments.layout_ranking.benchmark_case",
                side_effect=benchmark,
            ), patch(
                "experiments.layout_ranking.score_group",
                side_effect=AssertionError("resume rebuilt prepared scores"),
            ), redirect_stdout(StringIO()):
                self.assertEqual(run([*arguments, "--resume"]), 0)

            completed = json.loads(output_path.read_text())
            self.assertTrue(completed["complete"])


if __name__ == "__main__":
    unittest.main()
