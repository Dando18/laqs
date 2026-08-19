from __future__ import annotations

import unittest

from relay import (
    SCORE_MODES,
    Access,
    CodegenCost,
    ComponentScore,
    Hyperedge,
    LayoutScore,
    LinearInnerLayout,
    MatrixSpec,
    MemoryEvent,
    ObjectiveComponent,
    RelayProblem,
    SimultaneousRegions,
    column_major_layout,
    layout_codegen_cost,
    normalized_excess,
    pareto_frontier,
    quotient_region_count,
    row_major_layout,
    score_layouts,
    score_problem,
    score_to_dict,
    weighted_component_region_count,
)


class QuotientScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MatrixSpec("A", (4, 4), 4, ("i", "j"))
        self.row_major = row_major_layout(self.matrix)
        self.column_major = column_major_layout(self.matrix)

    def test_quotient_region_count_reflects_layout_locality(self) -> None:
        column = Hyperedge.make((i, 0) for i in range(4))

        self.assertEqual(
            quotient_region_count(self.matrix, self.row_major, column, 16),
            4,
        )
        self.assertEqual(
            quotient_region_count(self.matrix, self.column_major, column, 16),
            1,
        )

    def test_weighted_raw_count_packing_bound_and_normalized_excess(self) -> None:
        component = ObjectiveComponent(
            "wave.16B",
            16,
            {
                "A": (
                    Hyperedge.make(
                        ((0, 0), (1, 0), (2, 0), (3, 0)),
                        weight=2.0,
                    ),
                    Hyperedge.make(((0, 0), (0, 3)), weight=0.5),
                )
            },
        )

        raw = weighted_component_region_count(
            self.matrix,
            self.column_major,
            component,
        )
        bound = component.packing_bound(self.matrix)

        self.assertEqual(raw, 3.0)
        self.assertEqual(bound, 2.5)
        self.assertAlmostEqual(normalized_excess(raw, bound), 0.2)
        self.assertEqual(normalized_excess(0.0, 0.0), 0.0)


class AggregateScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrices = {
            name: MatrixSpec(name, (4, 4), 4, ("i", "j"))
            for name in ("A", "B")
        }
        self.layouts = {
            name: row_major_layout(matrix)
            for name, matrix in self.matrices.items()
        }
        self.fine = ObjectiveComponent(
            "fine",
            16,
            {
                "A": (Hyperedge.make((i, 0) for i in range(4)),),
                "B": (Hyperedge.make((0, j) for j in range(4)),),
            },
        )
        self.coarse = ObjectiveComponent(
            "coarse",
            32,
            {"A": (Hyperedge.make((i, 0) for i in range(4)),)},
        )
        self.score = score_layouts(
            self.matrices,
            (self.fine, self.coarse),
            self.layouts,
            component_weights={"fine": 2.0, "coarse": 0.0},
        )

    def test_multi_array_components_are_aggregated_before_normalization(self) -> None:
        fine = self.score.component("fine")

        self.assertEqual(fine.raw_region_count, 5.0)
        self.assertEqual(fine.packing_lower_bound, 2.0)
        self.assertEqual(fine.normalized_excess, 1.5)
        self.assertEqual(
            [array.array for array in fine.arrays],
            ["A", "B"],
        )
        self.assertEqual(
            [array.raw_region_count for array in fine.arrays],
            [4.0, 1.0],
        )
        self.assertEqual(
            [array.packing_lower_bound for array in fine.arrays],
            [1.0, 1.0],
        )

    def test_zero_weight_disables_a_component_from_all_aggregates(self) -> None:
        coarse = self.score.component("coarse")

        self.assertEqual(coarse.weight, 0.0)
        self.assertEqual(coarse.raw_region_count, 2.0)
        self.assertEqual(coarse.normalized_excess, 1.0)
        self.assertEqual(self.score.weighted_region_count, 10.0)
        self.assertEqual(self.score.peak_normalized_excess, 1.5)
        self.assertEqual(self.score.weighted_normalized_excess, 3.0)

    def test_layout_score_value_supports_every_public_mode(self) -> None:
        expected = {
            "weighted-region-count": 10.0,
            "peak-normalized-excess": 1.5,
            "weighted-normalized-excess": 3.0,
        }

        for mode, value in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(self.score.value(mode), value)

    def test_score_to_dict_preserves_aggregates_and_component_detail(self) -> None:
        data = score_to_dict(self.score)

        self.assertEqual(
            data["codegen"],
            {
                "runs": 4,
                "xors": 0,
                "arrays": [
                    {
                        "name": "A",
                        "grammar": "canonical",
                        "runs": 2,
                        "xors": 0,
                    },
                    {
                        "name": "B",
                        "grammar": "canonical",
                        "runs": 2,
                        "xors": 0,
                    },
                ],
            },
        )
        self.assertEqual(
            data["aggregates"],
            {
                "weighted_region_count": 10.0,
                "peak_normalized_excess": 1.5,
                "weighted_normalized_excess": 3.0,
            },
        )
        fine, coarse = data["components"]
        self.assertEqual(
            fine,
            {
                "name": "fine",
                "region_bytes": 16,
                "weight": 2.0,
                "raw_region_count": 5.0,
                "packing_lower_bound": 2.0,
                "normalized_excess": 1.5,
                "weighted_region_count": 10.0,
                "weighted_normalized_excess": 3.0,
                "arrays": [
                    {
                        "name": "A",
                        "raw_region_count": 4.0,
                        "packing_lower_bound": 1.0,
                        "normalized_excess": 3.0,
                    },
                    {
                        "name": "B",
                        "raw_region_count": 1.0,
                        "packing_lower_bound": 1.0,
                        "normalized_excess": 0.0,
                    },
                ],
            },
        )
        self.assertEqual(coarse["name"], "coarse")
        self.assertEqual(coarse["weight"], 0.0)
        self.assertEqual(coarse["weighted_region_count"], 0.0)
        self.assertEqual(coarse["weighted_normalized_excess"], 0.0)

    def test_codegen_cost_counts_runs_and_xors_without_mixing_them(self) -> None:
        matrix = MatrixSpec("L", (4, 4), 4, ("i", "j"))
        layout = LinearInnerLayout(
            "linear",
            "L",
            (1, 1),
            (0b11, 0b10),
            (1, 0),
        )

        cost = layout_codegen_cost({"L": matrix}, {"L": layout})

        self.assertEqual(cost.runs, 2)
        self.assertEqual(cost.xors, 1)
        self.assertEqual(cost.arrays[0].grammar, "linear_inner")

    def test_codegen_cost_excludes_non_target_context_arrays(self) -> None:
        target = MatrixSpec("A", (4, 4), 4, ("i", "j"))
        context = MatrixSpec("x", (4,), 4, ("i",), target=False)

        cost = layout_codegen_cost(
            {"A": target, "x": context},
            {"A": row_major_layout(target), "x": row_major_layout(context)},
        )

        self.assertEqual([array.array for array in cost.arrays], ["A"])
        self.assertEqual(cost.runs, 2)


class ProblemScoringTests(unittest.TestCase):
    def test_score_problem_builds_objectives_and_applies_component_weight(self) -> None:
        matrix = MatrixSpec("M", (4, 4), 4, ("i", "j"))
        event = MemoryEvent.make(
            "M.column",
            "M.load",
            [Access("M", (lane, 0), lane=lane) for lane in range(4)],
            weight=2.0,
        )
        problem = RelayProblem(
            matrices=(matrix,),
            events=(event,),
            sequences=(),
            objectives=(SimultaneousRegions("wave.16B", 16),),
        )

        score = score_problem(
            problem,
            {"M": row_major_layout(matrix)},
            component_weights={"wave.16B": 0.25},
        )

        component = score.component("wave.16B")
        self.assertEqual(component.raw_region_count, 8.0)
        self.assertEqual(component.packing_lower_bound, 2.0)
        self.assertEqual(component.normalized_excess, 3.0)
        self.assertEqual(score.weighted_region_count, 2.0)
        self.assertEqual(score.peak_normalized_excess, 3.0)
        self.assertEqual(score.weighted_normalized_excess, 0.75)


class ParetoFrontierTests(unittest.TestCase):
    @staticmethod
    def score(fine_q: float, peak: float, area: float) -> LayoutScore:
        fine = ComponentScore(
            name="fine",
            region_bytes=64,
            weight=1.0,
            raw_region_count=fine_q,
            packing_lower_bound=1.0,
            normalized_excess=max(fine_q - 1.0, 0.0),
            arrays=(),
        )
        return LayoutScore(
            components=(fine,),
            codegen=CodegenCost(()),
            weighted_region_count=fine_q,
            peak_normalized_excess=peak,
            weighted_normalized_excess=area,
        )

    def test_custom_objectives_return_exact_non_dominated_points(self) -> None:
        scores = {
            "fine": self.score(1.0, 4.0, 4.0),
            "balanced": self.score(2.0, 3.0, 2.0),
            "balanced_tie": self.score(2.0, 3.0, 2.0),
            "area": self.score(3.0, 2.0, 1.0),
            "dominated": self.score(2.0, 4.0, 3.0),
        }
        frontier = pareto_frontier(
            scores,
            objectives={
                "fine.raw-region-count": (
                    lambda score: score.component("fine").raw_region_count
                ),
                "peak-normalized-excess": (
                    lambda score: score.peak_normalized_excess
                ),
                "weighted-normalized-excess": (
                    lambda score: score.weighted_normalized_excess
                ),
            },
        )

        self.assertEqual(
            frontier.objectives,
            (
                "fine.raw-region-count",
                "peak-normalized-excess",
                "weighted-normalized-excess",
            ),
        )
        self.assertEqual(
            frontier.names,
            ("fine", "balanced", "balanced_tie", "area"),
        )
        self.assertEqual(frontier.points[1].values, (2.0, 3.0, 2.0))

    def test_default_objectives_include_aggregates_and_codegen_costs(self) -> None:
        frontier = pareto_frontier(
            {
                "best": self.score(1.0, 1.0, 1.0),
                "worse": self.score(2.0, 2.0, 2.0),
            }
        )

        self.assertEqual(
            frontier.objectives,
            (*SCORE_MODES, "codegen-runs", "codegen-xors"),
        )
        self.assertEqual(frontier.names, ("best",))

    def test_frontier_requires_an_objective(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one objective"):
            pareto_frontier({"only": self.score(1.0, 1.0, 1.0)}, objectives={})


if __name__ == "__main__":
    unittest.main()
