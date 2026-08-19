from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from kernels.mvt import evaluate, problem
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
        self.assertEqual(len(sequences), 2)
        self.assertEqual([len(sequence.event_ids) for sequence in sequences], [16, 16])

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
        self.assertTrue(
            all(
                event_id.startswith("A.row")
                for event_id in sequences[0].event_ids
            )
        )
        self.assertTrue(
            all(
                event_id.startswith("A.transpose")
                for event_id in sequences[1].event_ids
            )
        )

    def test_objectives_label_grounded_and_hypothesis_scopes(self) -> None:
        config = problem.build_config(problem_size=16, block_size=16)
        matrices = {matrix.name: matrix for matrix in problem.get_matrices(config)}
        event_items, sequences = problem.get_events_and_sequences(config)
        events = {event.id: event for event in event_items}
        components = build_objectives(
            problem.get_objectives(config), matrices, events, sequences
        )
        by_name = {component.name: component for component in components}

        self.assertEqual(len(components), 17)
        self.assertEqual(
            {
                component.name
                for component in components
                if component.provenance == "grounded"
            },
            {"wave_load.64B", "output_store.64B"},
        )
        self.assertEqual(len(by_name["wave_load.64B"].edges_by_array["A"]), 32)
        self.assertEqual(
            len(by_name["row_lane_stream.128B.window16"].edges_by_array["A"]),
            16,
        )
        self.assertEqual(
            len(by_name["row_lane_stream.512B.window16"].edges_by_array["A"]),
            16,
        )
        self.assertEqual(
            len(
                by_name["transpose_lane_stream.128B.window16"]
                .edges_by_array["A"]
            ),
            16,
        )
        cross_points = by_name["workgroup_step_cross.2048B"].edges_by_array[
            "A"
        ][0].points
        self.assertEqual(len(cross_points), 31)
        self.assertIn((0, 15), cross_points)
        self.assertIn((15, 0), cross_points)
        for region_bytes in (512, 1024, 4096, 8192):
            component = by_name[
                f"transpose_wave_neighborhood.{region_bytes}B"
            ]
            self.assertEqual(component.provenance, "hypothesis")
            self.assertEqual(len(component.edges_by_array["A"]), 16)
        self.assertEqual(set(problem.get_component_weights(config)), set(by_name))
        self.assertEqual(
            problem.get_component_weights(config)[
                "transpose_wave_neighborhood.4096B"
            ],
            0.0625,
        )
        self.assertEqual(
            problem.get_component_weights(config)[
                "row_lane_stream.512B.window16"
            ],
            0.0,
        )
        self.assertEqual(
            problem.get_component_weights(config)[
                "transpose_wave_neighborhood.512B"
            ],
            0.0,
        )
        self.assertEqual(
            problem.get_component_weights(config)[
                "A.wave_lane_group.lane64.512B"
            ],
            0.0,
        )
        self.assertEqual(
            problem.get_component_weights(config)["wave_neighborhood.512B"],
            0.0,
        )


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
