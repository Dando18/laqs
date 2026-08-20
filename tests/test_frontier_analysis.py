from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.frontier_analysis import (
    analyze_frontier_group,
    analyze_frontier_report,
    analyze_score_equivalence,
    analyze_tau_weight_robustness,
    poisson_binomial_upper_tail,
    random_subset_coverage_probability,
    render_frontier_plots,
)


def group(
    name: str,
    frontier_names: set[str],
    runtimes: tuple[float, ...] = (1.0, 1.002, 1.1, 1.2),
) -> dict[str, object]:
    names = ("A", "B", "C", "D")
    score_order = {"B": 0.0, "A": 1.0, "C": 2.0, "D": 3.0}
    return {
        "kernel": name.lower(),
        "display_name": name,
        "matrix_size": 8,
        "fine_component": "issue.g64.stream.load.64B",
        "results": [
            {
                "name": layout_name,
                "selected_score": score_order[layout_name],
                "pareto_frontier_member": layout_name in frontier_names,
                "timing": {"median_ms": runtime},
                "score": {
                    "components": [
                        {
                            "name": "issue.g64.stream.load.64B",
                            "weight": 0.0,
                            "raw_region_count": 1.0 + (layout_name > "B"),
                            "normalized_excess": 0.0,
                            "excess_footprint": 0.0,
                        },
                        {
                            "name": "reuse",
                            "weight": 2.0,
                            "raw_region_count": 1.0,
                            "normalized_excess": score_order[layout_name],
                            "excess_footprint": score_order[layout_name],
                        },
                    ],
                    "aggregates": {
                        "peak_normalized_excess": score_order[layout_name],
                        "weighted_normalized_excess": (
                            2.0 * score_order[layout_name]
                        ),
                        "hardware_peak": score_order[layout_name],
                        "hardware_area": 2.0 * score_order[layout_name],
                    },
                    "codegen": {"runs": 1, "xors": 0},
                },
            }
            for layout_name, runtime in zip(names, runtimes)
        ],
    }


class FrontierMetricTests(unittest.TestCase):
    def test_random_subset_probability_is_exact(self) -> None:
        self.assertAlmostEqual(
            random_subset_coverage_probability(4, 2, 2),
            5.0 / 6.0,
        )
        self.assertAlmostEqual(
            poisson_binomial_upper_tail((0.5, 0.25), 1),
            0.625,
        )

    def test_group_regret_coverage_purity_and_top_k(self) -> None:
        analysis = analyze_frontier_group(
            group("Miss", {"B", "C"}),
            epsilons=(0.0, 0.005),
        )

        self.assertEqual(analysis["frontier_size"], 2)
        self.assertEqual(analysis["layout_count"], 4)
        self.assertEqual(analysis["retained_fraction"], 0.5)
        self.assertAlmostEqual(analysis["oracle_regret"], 0.002)
        self.assertEqual(analysis["optimal_layouts"], ["A"])
        self.assertEqual(analysis["best_frontier_layouts"], ["B"])

        exact, relaxed = analysis["epsilon_metrics"]
        self.assertFalse(exact["covered"])
        self.assertTrue(relaxed["covered"])
        self.assertEqual(relaxed["optimal_layout_count"], 2)
        self.assertEqual(relaxed["frontier_optimal_layout_count"], 1)
        self.assertEqual(relaxed["purity"], 0.5)
        self.assertEqual(relaxed["search_space_prevalence"], 0.5)
        self.assertEqual(relaxed["enrichment"], 1.0)
        self.assertAlmostEqual(
            relaxed["random_size_matched_coverage_probability"],
            5.0 / 6.0,
        )

        self.assertAlmostEqual(analysis["top_k"][0]["regret"], 0.002)
        self.assertEqual(analysis["top_k"][0]["best_layouts"], ["B"])
        self.assertEqual(analysis["top_k"][1]["regret"], 0.0)
        self.assertEqual(analysis["top_k"][1]["best_layouts"], ["A"])

    def test_report_aggregates_instances_and_renders_plots(self) -> None:
        analysis = analyze_frontier_report(
            (
                group("Miss", {"B", "C"}),
                group("Hit", {"A"}),
            ),
            epsilons=(0.0, 0.005),
        )

        self.assertEqual(analysis["instance_count"], 2)
        self.assertEqual(
            analysis["exact_winner_coverage"],
            {"covered_instances": 1, "coverage": 0.5},
        )
        baseline = analysis["random_exact_winner_baseline"]
        self.assertEqual(baseline["expected_covered_instances"], 0.75)
        self.assertAlmostEqual(
            baseline["probability_at_least_observed_hits"],
            0.625,
        )
        self.assertAlmostEqual(analysis["oracle_regret"]["maximum"], 0.002)
        self.assertEqual(analysis["top_k"][1]["regret"]["maximum"], 0.0)

        robustness = analyze_tau_weight_robustness(
            (
                group("Miss", {"B", "C"}),
                group("Hit", {"A"}),
            ),
            trials=4,
            seed=7,
        )
        analysis["tau_weight_robustness"] = robustness
        self.assertEqual(robustness["trials_per_instance"], 4)
        self.assertEqual(len(robustness["instances"]), 2)

        with tempfile.TemporaryDirectory() as directory:
            paths = render_frontier_plots(analysis, Path(directory))
            self.assertEqual(
                set(paths),
                {
                    "epsilon_optimal_coverage",
                    "retained_fraction_vs_regret",
                    "purity_and_enrichment",
                    "top_k_regret",
                    "tau_weight_robustness",
                },
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths.values()))

    def test_score_equivalence_records_non_singleton_runtime_spread(self) -> None:
        example = group("Equal", {"A"}, runtimes=(1.0, 1.2, 1.5, 2.0))
        example["results"][1]["score"] = example["results"][0]["score"]
        example["fine_locality_gated_frontiers"] = [
            {"delta": 0.0, "delta_percent": 0.0, "fine_limit": 1.0}
        ]
        analysis = analyze_score_equivalence(example)

        main = analysis["main_frontier_vector"]
        self.assertEqual(main["non_singleton_group_count"], 1)
        self.assertAlmostEqual(
            main["non_singleton_runtime_spread"]["maximum"], 0.2
        )

    def test_tau_ablation_is_reproducible(self) -> None:
        example = group("Weights", {"A", "B"})
        first = analyze_tau_weight_robustness((example,), trials=5, seed=11)
        second = analyze_tau_weight_robustness((example,), trials=5, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(first["factors"], [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])


if __name__ == "__main__":
    unittest.main()
