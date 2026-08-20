from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from experiments.layout_ranking import KERNEL_SPECS
from experiments.solver_frontier import (
    _add_benchmark,
    _cost_dict,
    evaluator_command,
    finalize_report,
    nonnegative_float,
    parse_arguments,
    prepare_report,
    render_plot,
    reuse_timings,
)
from relay import (
    AffineAccessLayout,
    ExplicitRegions,
    FrontierCost,
    Hyperedge,
    LinearInnerLayout,
    MatrixSpec,
    NonDistributiveAccessError,
    ObjectiveComponent,
    SimpleRelayProblem,
    canonical_layout_from_word,
    score_layouts,
    simple_solve,
)
from relay.simple_solver import _standard_words
from relay.gf2 import invert_matrix_from_columns


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
    if problem.hardware_profile is not None:
        weights = problem.hardware_profile.component_weights(components)
        peak_tolerances = problem.hardware_profile.peak_tolerances(components)
    else:
        weights = {
            component.name: problem.component_weights.get(component.name, 1.0)
            for component in components
        }
        peak_tolerances = dict(problem.peak_tolerances)
        if not peak_tolerances:
            peak_tolerances = {
                name: 1.0 for name, weight in weights.items() if weight > 0
            }
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
            component_weights=weights,
            peak_tolerances=peak_tolerances,
        )
        cost = (
            score.component(problem.fine_component).raw_region_count,
            score.hardware_peak,
            score.hardware_area,
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
    def test_outer_canonical_search_finds_mixed_inner_direction(self) -> None:
        matrix = MatrixSpec("A", (4, 4), 4, ("i", "j"))
        edges = {
            "A": (Hyperedge.make(((0, 0), (1, 1))),)
        }
        problem = SimpleRelayProblem(
            matrices=(matrix,),
            events=(),
            sequences=(),
            objectives=(
                ExplicitRegions("fine", 8, edges),
                ExplicitRegions("coarse", 16, edges),
            ),
            grammar="outer_canonical",
            fine_component="fine",
            outer_canonical_max_inner_bits=2,
        )

        result = simple_solve(problem)

        search = result.array_searches[0]
        self.assertTrue(result.exact)
        self.assertTrue(search.search_stats.exact)
        self.assertEqual(search.tile_hypotheses, 6)
        self.assertEqual(search.grammar_layout_count, 36)
        self.assertTrue(search.score_ties_collapsed)
        mixed = [
            member
            for member in result.frontier
            if isinstance(member.layouts["A"], LinearInnerLayout)
        ]
        self.assertTrue(mixed)
        self.assertEqual(mixed[0].cost.fine_region_count, 1.0)
        self.assertEqual(mixed[0].cost.codegen_xors, 1)
        self.assertEqual(
            mixed[0].word_signature({"A": matrix}),
            (("A", "linear:1,1:2,3:ij"),),
        )

    def test_outer_canonical_exact_width_is_explicitly_bounded(self) -> None:
        matrix = MatrixSpec("A", (2, 2), 4, ("i", "j"))
        problem = SimpleRelayProblem(
            matrices=(matrix,),
            events=(),
            sequences=(),
            objectives=(
                ExplicitRegions(
                    "fine",
                    8,
                    {"A": (Hyperedge.make(((0, 0), (0, 1))),)},
                ),
            ),
            grammar="outer_canonical",
            fine_component="fine",
            outer_canonical_max_inner_bits=5,
        )

        with self.assertRaisesRegex(ValueError, "between zero and four"):
            simple_solve(problem)

    def test_outer_canonical_preserves_the_complete_canonical_tie_family(self) -> None:
        canonical_problem = alternating_problem("canonical")
        outer_problem = replace(
            canonical_problem,
            grammar="outer_canonical",
            outer_canonical_max_inner_bits=0,
        )

        canonical = simple_solve(canonical_problem)
        outer = simple_solve(outer_problem)
        matrices = {
            matrix.name: matrix for matrix in canonical_problem.matrices
        }

        self.assertEqual(
            {
                member.word_signature(matrices): member.cost.values
                for member in outer.frontier
            },
            {
                member.word_signature(matrices): member.cost.values
                for member in canonical.frontier
            },
        )

    def test_affine_access_dp_finds_mixed_shared_direction(self) -> None:
        matrix = MatrixSpec("A", (4, 4), 4, ("i", "j"))

        def coordinate(bits: int) -> tuple[int, int]:
            return bits & 0b11, bits >> 2

        shared = 0b0101
        first_only = 0b0010
        second_only = 0b1000
        first_space = tuple(
            coordinate(bits)
            for bits in (0, shared, first_only, shared ^ first_only)
        )
        second_space = tuple(
            coordinate(bits)
            for bits in (0, shared, second_only, shared ^ second_only)
        )
        edges = {
            "A": (
                Hyperedge.make(first_space),
                Hyperedge.make(second_space),
            )
        }
        problem = SimpleRelayProblem(
            matrices=(matrix,),
            events=(),
            sequences=(),
            objectives=(
                ExplicitRegions("fine", 8, edges),
                ExplicitRegions("coarse", 16, edges),
            ),
            grammar="affine",
            fine_component="fine",
        )

        result = simple_solve(problem)

        search = result.array_searches[0]
        self.assertEqual(search.access_block_dimensions, (1, 1, 1))
        self.assertEqual(search.grammar_layout_count, 6)
        self.assertEqual(search.active_rank, 3)
        self.assertEqual(search.inactive_rank, 1)
        self.assertTrue(search.score_ties_collapsed)
        self.assertTrue(result.frontier)
        self.assertTrue(
            all(
                isinstance(member.layouts["A"], AffineAccessLayout)
                for member in result.frontier
            )
        )
        self.assertTrue(
            any(
                shared in member.layouts["A"].basis_columns
                for member in result.frontier
            )
        )
        self.assertTrue(
            all(member.cost.codegen_xors >= 1 for member in result.frontier)
        )

        representative = result.frontier[0].layouts["A"]
        active = representative.basis_columns[:3]
        inactive = representative.basis_columns[3:]
        exhaustive_costs = []
        for index, order in enumerate(permutations(active)):
            columns = (*order, *inactive)
            layout = AffineAccessLayout(
                f"oracle_{index}",
                "A",
                matrix.mode_bits,
                invert_matrix_from_columns(columns, matrix.total_bits),
                (1, 0),
                columns,
                (0, 1, 2),
                (1, 1, 1),
                1,
            )
            score = score_layouts(
                {"A": matrix},
                result.components,
                {"A": layout},
                peak_tolerances={
                    component.name: 1.0 for component in result.components
                },
            )
            exhaustive_costs.append(
                (
                    score.component("fine").raw_region_count,
                    score.hardware_peak,
                    score.hardware_area,
                    float(score.codegen.runs),
                    float(score.codegen.xors),
                )
            )

        def dominates(left, right) -> bool:
            return all(a <= b for a, b in zip(left, right)) and any(
                a < b for a, b in zip(left, right)
            )

        exhaustive_frontier_costs = {
            cost
            for cost in exhaustive_costs
            if not any(
                other != cost and dominates(other, cost)
                for other in exhaustive_costs
            )
        }
        self.assertEqual(
            {member.cost.values for member in result.frontier},
            exhaustive_frontier_costs,
        )

    def test_affine_access_dp_rejects_nondistributive_lattice(self) -> None:
        matrix = MatrixSpec("A", (2, 2), 4, ("i", "j"))
        edges = {
            "A": tuple(
                Hyperedge.make(((0, 0), point))
                for point in ((1, 0), (0, 1), (1, 1))
            )
        }
        problem = SimpleRelayProblem(
            matrices=(matrix,),
            events=(),
            sequences=(),
            objectives=(ExplicitRegions("fine", 8, edges),),
            grammar="affine",
            fine_component="fine",
        )

        with self.assertRaises(NonDistributiveAccessError):
            simple_solve(problem)

    def test_affine_access_dp_rejects_affine_hull_only_edge(self) -> None:
        matrix = MatrixSpec("A", (2, 2), 4, ("i", "j"))
        problem = SimpleRelayProblem(
            matrices=(matrix,),
            events=(),
            sequences=(),
            objectives=(
                ExplicitRegions(
                    "fine",
                    8,
                    {
                        "A": (
                            Hyperedge.make(((0, 0), (1, 0), (0, 1))),
                        )
                    },
                ),
            ),
            grammar="affine",
            fine_component="fine",
        )

        with self.assertRaisesRegex(ValueError, "is not an affine coset"):
            simple_solve(problem)

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
                        "hardware-peak",
                        "hardware-area",
                        "codegen-runs",
                        "codegen-xors",
                    ),
                )


class SolverFrontierExperimentTests(unittest.TestCase):
    def test_matching_prior_timing_can_be_reused(self) -> None:
        configuration = {
            "matrix_size": 8,
            "hardware_profile": "mi300a",
            "hardware_profile_id": "mi300a-gfx942-universal-v1-poc",
            "frontier_type": "pareto",
            "fine_component": "issue.g64.stream.load.64B",
            "fine_tolerance": None,
            "samples": 2,
            "iterations": 1,
            "warmup": 0,
            "device": 0,
            "block_size": 8,
            "block_x": 4,
            "block_y": 4,
            "compiler": "hipcc",
            "arch": None,
        }
        source = {
            "experiment": "solver-frontier-speedup",
            "configuration": configuration,
            "benchmarks": [
                {
                    "id": "old",
                    "kernel": "atax",
                    "words": {"A": "jjjiii"},
                    "timing": {"median_ms": 2.0},
                    "command": ["old-command"],
                    "stdout": "Correctness: PASS",
                    "stderr": "",
                }
            ],
        }
        report = {
            "configuration": dict(configuration),
            "benchmarks": [
                {
                    "id": "new",
                    "kernel": "atax",
                    "words": {"A": "jjjiii"},
                    "timing": None,
                    "command": None,
                    "stdout": None,
                    "stderr": None,
                    "timing_source": None,
                }
            ],
            "kernels": [],
            "plot_data": [],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            path.write_text(json.dumps(source))

            reused = reuse_timings(report, path)

        self.assertEqual(reused, 1)
        self.assertEqual(
            report["benchmarks"][0]["timing"], {"median_ms": 2.0}
        )
        self.assertEqual(
            report["benchmarks"][0]["timing_source"]["benchmark_id"],
            "old",
        )

    def test_frontier_tolerance_must_be_finite(self) -> None:
        for value in ("nan", "inf", "-inf", "-0.1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    nonnegative_float(value)

    def test_normal_pareto_frontier_is_the_experiment_default(self) -> None:
        _, args = parse_arguments([])

        self.assertEqual(args.frontier_type, "pareto")
        self.assertEqual(args.hardware_profile, "mi300a")
        self.assertIsNone(args.grammar)

    def test_cost_report_uses_hardware_aggregate_names(self) -> None:
        cost = FrontierCost(3.0, 0.25, 1.5, 2, 1)

        self.assertEqual(
            _cost_dict(cost),
            {
                "fine_region_count": 3.0,
                "hardware_peak": 0.25,
                "hardware_area": 1.5,
                "codegen_runs": 2,
                "codegen_xors": 1,
            },
        )

    def test_prepared_report_uses_one_hardware_profile_and_universal_scopes(
        self,
    ) -> None:
        _, args = parse_arguments(
            [
                "--kernel",
                "atax",
                "--grammar",
                "standard",
                "--size",
                "8",
                "--block-size",
                "8",
                "--prepare-only",
            ]
        )

        report = prepare_report(args, ("atax",), ("standard",))

        profile = report["hardware_profile"]
        kernel = report["kernels"][0]
        objective = kernel["objectives"][0]
        candidate = kernel["solvers"][0]["frontier"][0]
        self.assertEqual(report["configuration"]["hardware_profile"], "mi300a")
        self.assertEqual(
            report["configuration"]["hardware_profile_id"],
            profile["profile_id"],
        )
        self.assertEqual(
            report["configuration"]["fine_component"],
            profile["fine_component"],
        )
        self.assertEqual(
            set(kernel["component_weights"]),
            {item["name"] for item in kernel["objectives"]},
        )
        self.assertEqual(
            kernel["peak_tolerances"],
            {
                name: tolerance
                for name, tolerance in profile["kappa"].items()
                if name in kernel["component_weights"]
            },
        )
        self.assertEqual(objective["provenance"], "universal-v1")
        self.assertIsNotNone(objective["edge_family"])
        self.assertIsNotNone(objective["normalization_bytes"])
        self.assertEqual(
            set(candidate["cost"]),
            {
                "fine_region_count",
                "hardware_peak",
                "hardware_area",
                "codegen_runs",
                "codegen_xors",
            },
        )
        self.assertIn("hardware_peak", candidate["score"]["aggregates"])
        self.assertIn("hardware_area", candidate["score"]["aggregates"])

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
