from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from kernels.gesummv import evaluate, problem
from relay.objectives import build_objectives


class GesummvProblemTests(unittest.TestCase):
    def test_problem_models_one_complete_workgroup_trace(self) -> None:
        config = problem.build_config(problem_size=8, block_size=8)
        matrices = problem.get_matrices(config)
        events, sequences = problem.get_events_and_sequences(config)

        self.assertEqual(
            [(matrix.name, matrix.shape, matrix.target) for matrix in matrices],
            [
                ("A", (8, 8), True),
                ("B", (8, 8), True),
                ("x", (8,), False),
                ("y", (8,), False),
            ],
        )
        self.assertEqual(len(events), 3 * 8 + 1)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].event_ids, tuple(event.id for event in events))

        first_a, first_b, first_x = events[:3]
        self.assertEqual(first_a.site, "A.load")
        self.assertEqual(
            [access.coord for access in first_a.accesses],
            [(i, 0) for i in range(8)],
        )
        self.assertEqual(first_b.site, "B.load")
        self.assertEqual(first_x.site, "x.load")
        self.assertEqual({access.coord for access in first_x.accesses}, {(0,)})
        self.assertEqual(events[-1].site, "y.store")

    def test_objectives_cover_matrix_loads_and_output_stores(self) -> None:
        config = problem.build_config(problem_size=8, block_size=8)
        matrices = {matrix.name: matrix for matrix in problem.get_matrices(config)}
        event_items, sequences = problem.get_events_and_sequences(config)
        events = {event.id: event for event in event_items}
        components = build_objectives(
            problem.get_objectives(config), matrices, events, sequences
        )
        by_name = {component.name: component for component in components}

        self.assertEqual(len(components), 10)
        self.assertEqual(len(by_name["wave_load.64B"].edges_by_array["A"]), 8)
        self.assertEqual(len(by_name["wave_load.64B"].edges_by_array["B"]), 8)
        self.assertNotIn("x", by_name["wave_load.64B"].edges_by_array)
        self.assertEqual(
            len(by_name["output_store.64B"].edges_by_array["y"]), 1
        )
        self.assertIn("lane_reuse.128B.window16", by_name)
        self.assertEqual(
            by_name["wave_lane_group.lane8.64B"]
            .edges_by_array["A"][0]
            .points,
            tuple((i, 0) for i in range(8)),
        )
        self.assertEqual(
            by_name["workgroup_step_panel.1024B"]
            .edges_by_array["A"][0]
            .points,
            tuple((i, 0) for i in range(8)),
        )
        self.assertEqual(
            {
                component.name
                for component in components
                if component.provenance == "grounded"
            },
            {"wave_load.64B", "output_store.64B"},
        )
        self.assertEqual(
            set(problem.get_component_weights(config)),
            set(by_name),
        )
        self.assertEqual(
            problem.get_component_weights(config)[
                "wave_lane_group.lane16.128B"
            ],
            0.125,
        )


class GesummvEvaluatorTests(unittest.TestCase):
    def test_canonical_layout_and_generated_source(self) -> None:
        layouts = (
            evaluate.canonical_layout("jjii", "a_word"),
            evaluate.canonical_layout("iijj", "b_word"),
        )
        source = evaluate.generate_source(
            layouts,
            n=16,
            samples=3,
            iterations=4,
            warmup=2,
            device=1,
            block_size=64,
        )

        self.assertEqual(layouts[0].tile_rows, 4)
        self.assertEqual(layouts[0].tile_columns, 4)
        self.assertIn("__global__ void gesummv_kernel", source)
        self.assertIn("word(low -> high)=jjii", source)
        self.assertIn("word(low -> high)=iijj", source)
        self.assertIn("constexpr uint32_t n = 16;", source)
        self.assertIn("constexpr uint32_t block_size = 64;", source)
        self.assertIn("Correctness: %s", source)
        self.assertIn("Samples (ms):", source)

    def test_emit_only_cli_writes_the_generated_driver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            with redirect_stdout(output):
                status = evaluate.run(
                    [
                        "jjjiii",
                        "iiijjj",
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
            source_path = Path(directory) / "generated_gesummv.cu"
            self.assertTrue(source_path.exists())
            self.assertIn(
                "Layouts: A=%s B=%s",
                source_path.read_text(),
            )
            self.assertIn(str(source_path.resolve()), output.getvalue())

    def test_layout_words_must_tile_the_problem(self) -> None:
        errors = StringIO()
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as caught,
        ):
            # A has a 16-row inner tile, which cannot tile N=8.
            evaluate.run(["iiii", "jjj", "--n", "8", "--emit-only"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("must be divisible", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
