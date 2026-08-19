from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.frontier_analysis import (
    analyze_frontier_group,
    analyze_frontier_report,
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
        "results": [
            {
                "name": layout_name,
                "selected_score": score_order[layout_name],
                "pareto_frontier_member": layout_name in frontier_names,
                "timing": {"median_ms": runtime},
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

        with tempfile.TemporaryDirectory() as directory:
            paths = render_frontier_plots(analysis, Path(directory))
            self.assertEqual(
                set(paths),
                {
                    "epsilon_optimal_coverage",
                    "retained_fraction_vs_regret",
                    "purity_and_enrichment",
                    "top_k_regret",
                },
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
