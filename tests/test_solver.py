from __future__ import annotations

import unittest

from relay import (
    Access,
    MatrixSpec,
    MemoryEvent,
    RelayProblem,
    ScorePolicy,
    SimultaneousRegions,
    SolverConfig,
    solve,
)


class SolverTests(unittest.TestCase):
    def test_inner_tile_search_keeps_outer_order_fixed(self) -> None:
        matrix = MatrixSpec("M", (8, 8), 4, ("i", "j"))
        event = MemoryEvent.make(
            "load",
            "load",
            [Access("M", (lane, 0), lane=lane) for lane in range(8)],
        )
        result = solve(
            RelayProblem(
                matrices=(matrix,),
                events=(event,),
                sequences=(),
                objectives=(SimultaneousRegions("fine", 16),),
                config=SolverConfig(
                    policy=ScorePolicy(
                        "lexicographic", ("fine", "runs", "xors")
                    ),
                    tile_shapes={"M": ((2, 2),)},
                    general_tile_shapes={"M": ()},
                    include_global_canonical=False,
                    enable_linear_inner=False,
                    include_column_major_control=False,
                    include_tiled_row_major_control=True,
                    primary_tolerance=0.0,
                    per_array_candidates=8,
                ),
            )
        )

        candidates = result.arrays["M"].candidates
        by_name = {candidate.layout.name: candidate for candidate in candidates}
        self.assertIn("row_major", by_name)
        self.assertIn("tiled_row_major", by_name)
        self.assertNotIn("column_major", by_name)
        self.assertEqual(
            by_name["tiled_row_major"].layout.tile_exponents,
            (1, 1),
        )
        self.assertEqual(
            by_name["tiled_row_major"].layout.outer_order,
            (1, 0),
        )

    def test_conflict_packing_gap(self) -> None:
        matrix = MatrixSpec("M", (2, 2), 4, ("i", "j"))
        point_sets = (
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
        )
        events = tuple(
            MemoryEvent.make(
                f"e{index}",
                f"s{index}",
                [Access("M", point, lane=lane) for lane, point in enumerate(points)],
            )
            for index, points in enumerate(point_sets)
        )
        problem = RelayProblem(
            matrices=(matrix,),
            events=events,
            sequences=(),
            objectives=(SimultaneousRegions("fine", 8),),
            config=SolverConfig(
                policy=ScorePolicy("lexicographic", ("fine", "runs", "xors")),
                tile_shapes={"M": ((2, 2),)},
                general_tile_shapes={"M": ((2, 2),)},
                general_exact_rank=2,
                per_array_candidates=8,
            ),
        )
        result = solve(problem)
        best = result.arrays["M"].candidates[0]
        self.assertEqual(best.scores["fine"], 6)
        self.assertEqual(best.packing_bounds["fine"], 4)
        self.assertTrue(any(candidate.grammar == "linear_inner" for candidate in result.arrays["M"].candidates))

    def test_two_target_arrays_form_joint_candidates(self) -> None:
        matrices = (
            MatrixSpec("A", (8, 8), 4, ("i", "j")),
            MatrixSpec("B", (8, 8), 4, ("i", "j")),
        )
        events = (
            MemoryEvent.make(
                "A.load",
                "A.load",
                [Access("A", (0, lane), lane=lane) for lane in range(8)],
            ),
            MemoryEvent.make(
                "B.load",
                "B.load",
                [Access("B", (lane, 0), lane=lane) for lane in range(8)],
            ),
        )
        problem = RelayProblem(
            matrices=matrices,
            events=events,
            sequences=(),
            objectives=(SimultaneousRegions("fine", 32),),
            config=SolverConfig(
                policy=ScorePolicy("lexicographic", ("fine", "runs")),
                tile_shapes={"A": ((8, 8),), "B": ((8, 8),)},
                general_tile_shapes={"A": (), "B": ()},
                per_array_candidates=4,
                joint_candidates=4,
            ),
        )
        result = solve(problem)
        self.assertIn("A", result.arrays)
        self.assertIn("B", result.arrays)
        self.assertGreaterEqual(len(result.joint_candidates), 1)
        top = result.joint_candidates[0]
        self.assertEqual(set(top.layouts), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
