from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from kernels.mvt import evaluate, problem
from relay import (
    MI300A_V1,
    UNIVERSAL_V1_BASIS,
    UniversalScopeObjectives,
    build_edge_families,
)
from relay.objectives import build_objectives


class MvtProblemTests(unittest.TestCase):
    def test_problem_models_both_matrix_directions(self) -> None:
        config = problem.build_config(problem_size=16, block_size=16)
        matrices = problem.get_matrices(config)
        events, sequences = problem.get_events_and_sequences(config)

        self.assertEqual(
            [(matrix.name, matrix.shape, matrix.target) for matrix in matrices],
            [
                ("A", (16, 16), True),
                ("y1", (16,), False),
                ("y2", (16,), False),
                ("x1", (16,), False),
                ("x2", (16,), False),
            ],
        )
        self.assertEqual(len(events), 4 * 16 + 4)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].event_ids, tuple(event.id for event in events))

        row_load = events[2]
        transpose_load = events[4]
        self.assertEqual(row_load.site, "A.row.load")
        self.assertEqual(
            [access.coord for access in row_load.accesses],
            [(i, 0) for i in range(16)],
        )
        self.assertEqual(transpose_load.site, "A.transpose.load")
        self.assertEqual(
            [access.coord for access in transpose_load.accesses],
            [(0, i) for i in range(16)],
        )
        self.assertEqual(
            [events[event_index].site for event_index in range(2, 6)],
            ["A.row.load", "y1.load", "A.transpose.load", "y2.load"],
        )

    def test_array_lane_windows_union_row_and_transpose_streams(self) -> None:
        config = problem.build_config(problem_size=8, block_size=8)
        matrices = {matrix.name: matrix for matrix in problem.get_matrices(config)}
        event_items, sequences = problem.get_events_and_sequences(config)
        events = {event.id: event for event in event_items}
        families = {
            family.name: family
            for family in build_edge_families(matrices, events, sequences)
        }

        stream_edges = families[
            "lane_window.t4.stream.load"
        ].edges_by_array["A"]
        array_edges = families["lane_window.t4.array.load"].edges_by_array["A"]

        self.assertNotEqual(
            {(edge.points, edge.weight) for edge in stream_edges},
            {(edge.points, edge.weight) for edge in array_edges},
        )
        self.assertTrue(
            any((0, 1) in edge.points and (1, 0) in edge.points for edge in array_edges)
        )

    def test_universal_scopes_preserve_both_matrix_directions(self) -> None:
        config = problem.build_config(problem_size=16, block_size=16)
        matrices = {matrix.name: matrix for matrix in problem.get_matrices(config)}
        event_items, sequences = problem.get_events_and_sequences(config)
        events = {event.id: event for event in event_items}
        components = build_objectives(
            (UniversalScopeObjectives(MI300A_V1.byte_scales),),
            matrices,
            events,
            sequences,
        )
        by_name = {component.name: component for component in components}
        fine = by_name[MI300A_V1.fine_component]

        self.assertEqual(fine.edge_family, "issue.g64.stream.load")
        self.assertEqual(
            {edge.points for edge in fine.edges_by_array["A"]},
            {
                tuple((i, 0) for i in range(16)),
                tuple((0, j) for j in range(16)),
            },
        )
        self.assertEqual(
            {edge.weight for edge in fine.edges_by_array["A"]}, {16.0}
        )

        schema = {scope.name for scope in UNIVERSAL_V1_BASIS.scope_keys()}
        families = {component.edge_family for component in components}
        self.assertLessEqual(families, schema)
        for family in families:
            materialized = [
                component
                for component in components
                if component.edge_family == family
            ]
            self.assertEqual(
                tuple(component.region_bytes for component in materialized),
                MI300A_V1.byte_scales,
            )
            self.assertTrue(
                all(
                    component.edges_by_array is materialized[0].edges_by_array
                    for component in materialized
                )
            )
        self.assertTrue(
            all(component.provenance == "universal-v1" for component in components)
        )
        self.assertFalse(hasattr(problem, "get_objectives"))
        self.assertFalse(hasattr(problem, "get_component_weights"))


class MvtEvaluatorTests(unittest.TestCase):
    def test_canonical_layout_and_generated_source(self) -> None:
        layout = evaluate.canonical_layout("jjii", "a_word")
        source = evaluate.generate_source(
            layout,
            n=16,
            samples=3,
            iterations=4,
            warmup=2,
            device=1,
            block_size=64,
        )

        self.assertEqual(layout.tile_rows, 4)
        self.assertEqual(layout.tile_columns, 4)
        self.assertIn("__global__ void mvt_kernel", source)
        self.assertIn("a[a_offset(i, j, n)]", source)
        self.assertIn("a[a_offset(j, i, n)]", source)
        self.assertIn("word(low -> high)=jjii", source)
        self.assertIn("constexpr uint32_t n = 16;", source)
        self.assertIn("constexpr uint32_t block_size = 64;", source)
        self.assertIn("Correctness: %s", source)
        self.assertIn("median_ms  mean_ms  min_ms  sd_ms  GFLOP/s", source)
        self.assertIn("Samples (ms):", source)

    def test_emit_only_cli_writes_the_generated_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            with redirect_stdout(output):
                status = evaluate.run(
                    [
                        "jjjiii",
                        "--n",
                        "64",
                        "--block-size",
                        "64",
                        "--build-dir",
                        directory,
                        "--emit-only",
                    ]
                )

            self.assertEqual(status, 0)
            source_path = Path(directory) / "generated_mvt.cu"
            self.assertTrue(source_path.exists())
            self.assertIn("Layout: A=%s", source_path.read_text())
            self.assertIn(str(source_path.resolve()), output.getvalue())

    def test_layout_word_must_tile_the_problem(self) -> None:
        errors = StringIO()
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as caught,
        ):
            evaluate.run(["iiii", "--n", "8", "--emit-only"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("must be divisible", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
