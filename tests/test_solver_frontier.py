from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations, product
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from experiments.layout_ranking import KERNEL_SPECS
from experiments.solver_frontier import (
    _add_benchmark,
    evaluator_command,
    finalize_report,
    nonnegative_float,
    parse_arguments,
    render_plot,
)
from relay import (
    ExplicitRegions,
    Hyperedge,
    MatrixSpec,
    ObjectiveComponent,
    SimpleRelayProblem,
    canonical_layout_from_word,
    score_layouts,
    simple_solve,
)
from relay.simple_solver import _standard_words


def canonical_words(matrix: MatrixSpec) -> tuple[str, ...]:
    """Enumerate every rank-two canonical word independently of the solver."""

    first_bits, second_bits = matrix.mode_bits
    total_bits = first_bits + second_bits
    words = []
    for first_positions in combinations(range(total_bits), first_bits):
        positions = set(first_positions)
        words.append(
            "".join(
                matrix.mode_names[0]
                if position in positions
                else matrix.mode_names[1]
                for position in range(total_bits)
            )
        )
    return tuple(words)


def alternating_problem(grammar: str) -> SimpleRelayProblem:
    """Build a small two-target problem whose canonical-only word matters."""

    matrices = (
        MatrixSpec("A", (4, 8), 4, ("i", "j")),
        MatrixSpec("B", (4, 8), 4, ("i", "j")),
    )
    prefix_counts = ((0, 1), (1, 1), (1, 2), (2, 2))
    objectives = []
    for dimension, (i_bits, j_bits) in enumerate(prefix_counts, 1):
        edges = {
            matrix.name: (
                Hyperedge.make(
                    product(range(1 << i_bits), range(1 << j_bits))
                ),
            )
            for matrix in matrices
        }
        objectives.append(
            ExplicitRegions(
                "fine" if dimension == 1 else f"prefix.{dimension}",
                4 * (1 << dimension),
                edges,
            )
        )
    return SimpleRelayProblem(
        matrices=matrices,
        events=(),
        sequences=(),
        objectives=tuple(objectives),
        grammar=grammar,
        frontier_type="pareto",
        fine_component="fine",
        fine_tolerance=0.05,
        name=f"tiny_{grammar}",
    )


def exhaustive_frontier(
    problem: SimpleRelayProblem,
    components: tuple[ObjectiveComponent, ...],
) -> dict[
    tuple[tuple[str, str], ...], tuple[float, ...]
]:
    """Compute the final joint frontier without using either search routine."""

    matrices = {matrix.name: matrix for matrix in problem.matrices}
    target_names = tuple(
        matrix.name for matrix in problem.matrices if matrix.target
    )
    if problem.grammar == "standard":
        word_sets = [
            tuple(
                "".join(matrix.mode_names[mode] for mode in word)
                for word in _standard_words(matrix)
            )
            for matrix in problem.matrices
            if matrix.target
        ]
    else:
        word_sets = [
            canonical_words(matrix)
            for matrix in problem.matrices
            if matrix.target
        ]

    candidates = []
    for words in product(*word_sets):
        layouts = {
            name: canonical_layout_from_word(
                matrices[name], word, name=f"oracle_{name}_{word}"
            )
            for name, word in zip(target_names, words)
        }
        score = score_layouts(
            matrices,
            components,
            layouts,
            component_weights={
                component.name: problem.component_weights.get(
                    component.name, 1.0
                )
                for component in components
            },
        )
        cost = (
            score.component(problem.fine_component).raw_region_count,
            score.peak_normalized_excess,
            score.weighted_normalized_excess,
            float(score.codegen.runs),
            float(score.codegen.xors),
        )
        signature = tuple(zip(target_names, words))
        candidates.append((signature, cost))

    def dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        return all(a <= b for a, b in zip(left, right)) and any(
            a < b for a, b in zip(left, right)
        )

    if problem.frontier_type == "fine-gated":
        fine_minimum = min(cost[0] for _, cost in candidates)
        fine_limit = (1.0 + problem.fine_tolerance) * fine_minimum
        candidates = [
            item for item in candidates if item[1][0] <= fine_limit
        ]
        objective_slice = slice(1, None)
    else:
        objective_slice = slice(None)

    return {
        signature: cost
        for signature, cost in candidates
        if not any(
            other_signature != signature
            and dominates(other_cost[objective_slice], cost[objective_slice])
            for other_signature, other_cost in candidates
        )
    }


class GrammarFrontierTests(unittest.TestCase):
    def test_standard_words_are_the_nine_unique_cut_point_layouts(self) -> None:
        matrix = MatrixSpec("A", (4, 8), 4, ("i", "j"))
        standard = {
            "".join(matrix.mode_names[mode] for mode in word)
            for word in _standard_words(matrix)
        }
        canonical = set(canonical_words(matrix))

        self.assertEqual(len(standard), 9)
        self.assertEqual(len(canonical), 10)
        self.assertEqual(canonical - standard, {"jijij"})
        self.assertEqual(standard - canonical, set())

    def test_both_joint_frontiers_match_exhaustive_grammar_search(self) -> None:
        results = {}
        for grammar in ("standard", "canonical"):
            with self.subTest(grammar=grammar):
                problem = alternating_problem(grammar)
                result = simple_solve(problem)
                matrices = {
                    matrix.name: matrix for matrix in problem.matrices
                }
                observed = {
                    member.word_signature(matrices): member.cost.values
                    for member in result.frontier
                }

                self.assertTrue(result.exact)
                self.assertEqual(
                    observed,
                    exhaustive_frontier(problem, result.components),
                )
                expected_layout_count = 9 if grammar == "standard" else 10
                self.assertEqual(
                    [
                        search.grammar_layout_count
                        for search in result.array_searches
                    ],
                    [expected_layout_count, expected_layout_count],
                )
                if grammar == "canonical":
                    self.assertTrue(
                        all(
                            search.search_stats is not None
                            and search.search_stats.exact
                            and not search.search_stats.truncated
                            for search in result.array_searches
                        )
                    )
                results[grammar] = observed

        alternating = (("A", "jijij"), ("B", "jijij"))
        self.assertNotIn(alternating, results["standard"])
        self.assertIn(alternating, results["canonical"])

    def test_fine_gated_frontier_matches_exhaustive_search(self) -> None:
        for grammar in ("standard", "canonical"):
            with self.subTest(grammar=grammar):
                problem = replace(
                    alternating_problem(grammar),
                    frontier_type="fine-gated",
                    fine_tolerance=0.25,
                )
                result = simple_solve(problem)
                matrices = {
                    matrix.name: matrix for matrix in problem.matrices
                }
                observed = {
                    member.word_signature(matrices): member.cost.values
                    for member in result.frontier
                }

                self.assertEqual(
                    observed,
                    exhaustive_frontier(problem, result.components),
                )
                self.assertEqual(
                    result.frontier_objectives,
                    (
                        "peak-normalized-excess",
                        "weighted-normalized-excess",
                        "codegen-runs",
                        "codegen-xors",
                    ),
                )


class SolverFrontierExperimentTests(unittest.TestCase):
    def test_frontier_tolerance_must_be_finite(self) -> None:
        for value in ("nan", "inf", "-inf", "-0.1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    nonnegative_float(value)

    def test_normal_pareto_frontier_is_the_experiment_default(self) -> None:
        _, args = parse_arguments([])

        self.assertEqual(args.frontier_type, "pareto")

    def test_evaluator_command_preserves_distinct_operand_words(self) -> None:
        _, args = parse_arguments(
            ["--kernel", "gemm", "--size", "8", "--arch", "gfx942"]
        )
        words = {
            "B": "jjjiii",
            "C": "ijijij",
            "A": "iiijjj",
        }

        command = evaluator_command(KERNEL_SPECS["gemm"], words, args)

        self.assertEqual(
            command[:2],
            [sys.executable, str(KERNEL_SPECS["gemm"].evaluator)],
        )
        self.assertEqual(command[2:5], ["iiijjj", "jjjiii", "ijijij"])
        self.assertIn("--block-x", command)
        self.assertIn("--block-y", command)
        self.assertEqual(command[-2:], ["--arch", "gfx942"])

    def test_benchmark_records_are_deduplicated_by_ordered_words(self) -> None:
        spec = KERNEL_SPECS["gemm"]
        benchmarks: list[dict[str, object]] = []
        benchmark_ids: dict[tuple[str, tuple[str, ...]], str] = {}
        first_words = {"A": "iiijjj", "B": "jjjiii", "C": "ijijij"}

        first = _add_benchmark(
            benchmarks, benchmark_ids, spec, first_words
        )
        duplicate = _add_benchmark(
            benchmarks,
            benchmark_ids,
            spec,
            {"C": "ijijij", "A": "iiijjj", "B": "jjjiii"},
        )
        distinct = _add_benchmark(
            benchmarks,
            benchmark_ids,
            spec,
            {**first_words, "B": "jijiji"},
        )

        self.assertEqual(first, duplicate)
        self.assertNotEqual(first, distinct)
        self.assertEqual(len(benchmarks), 2)
        self.assertEqual(
            benchmarks[0]["words"],
            {"A": "iiijjj", "B": "jjjiii", "C": "ijijij"},
        )

    def test_finalize_selects_fastest_frontier_member_and_speedup(self) -> None:
        report = {
            "benchmarks": [
                {"id": "baseline", "timing": {"median_ms": 4.0}},
                {"id": "candidate-slow", "timing": {"median_ms": 3.0}},
                {"id": "candidate-fast", "timing": {"median_ms": 1.6}},
            ],
            "kernels": [
                {
                    "kernel": "gemm",
                    "display_name": "GEMM",
                    "baseline": {
                        "benchmark_id": "baseline",
                        "words": {"A": "jjjiii", "B": "jjjiii", "C": "jjjiii"},
                        "speedup": None,
                    },
                    "solvers": [
                        {
                            "algorithm": "G_C dynamic programming",
                            "best": None,
                            "frontier": [
                                {
                                    "id": "canonical-0000",
                                    "benchmark_id": "candidate-slow",
                                    "words": {
                                        "A": "iiijjj",
                                        "B": "iiijjj",
                                        "C": "iiijjj",
                                    },
                                    "cost": {"fine_region_count": 2.0},
                                },
                                {
                                    "id": "canonical-0001",
                                    "benchmark_id": "candidate-fast",
                                    "words": {
                                        "A": "ijijij",
                                        "B": "jijiji",
                                        "C": "ijijij",
                                    },
                                    "cost": {"fine_region_count": 3.0},
                                },
                            ],
                        }
                    ],
                }
            ],
            "plot_data": [],
        }

        finalize_report(report)

        kernel = report["kernels"][0]
        solver = kernel["solvers"][0]
        self.assertTrue(report["complete"])
        self.assertEqual(kernel["baseline"]["speedup"], 1.0)
        self.assertEqual(solver["best"]["frontier_member_id"], "canonical-0001")
        self.assertEqual(solver["best"]["median_ms"], 1.6)
        self.assertEqual(solver["best"]["speedup"], 2.5)
        self.assertEqual(
            [row["algorithm"] for row in report["plot_data"]],
            ["Baseline", "G_C dynamic programming"],
        )
        self.assertEqual(report["plot_data"][1]["speedup"], 2.5)

    def test_completed_report_renders_speedup_plot(self) -> None:
        report = {
            "complete": True,
            "configuration": {
                "kernels": ["gemm"],
                "grammars": ["canonical"],
                "matrix_size": 8,
            },
            "plot_data": [
                {
                    "display_name": "GEMM",
                    "algorithm": "Baseline",
                    "speedup": 1.0,
                },
                {
                    "display_name": "GEMM",
                    "algorithm": "G_C dynamic programming",
                    "speedup": 1.5,
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "speedup.png"

            render_plot(report, path)

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
