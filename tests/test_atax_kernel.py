from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from kernels.atax import evaluate, problem
from relay import (
    MI300A_V1,
    UNIVERSAL_V1_BASIS,
    UniversalScopeObjectives,
)
from relay.objectives import build_objectives


class AtaxProblemTests(unittest.TestCase):
    def test_problem_models_both_complete_matrix_vector_stages(self) -> None:
        config = problem.build_config(problem_size=8, block_size=8)
        matrices = problem.get_matrices(config)
        events, sequences = problem.get_events_and_sequences(config)

        self.assertEqual(
            [(matrix.name, matrix.shape, matrix.target) for matrix in matrices],
            [
                ("A", (8, 8), True),
                ("x", (8,), False),
                ("tmp", (8,), False),
                ("y", (8,), False),
            ],
        )
        self.assertEqual(len(events), 2 * (2 * 8 + 1))
        self.assertEqual(
            [sequence.name for sequence in sequences],
            ["stage1.wave0", "stage2.wave0"],
        )

        first_stage_a, first_x = events[:2]
        self.assertEqual(first_stage_a.site, "A.stage1.load")
        self.assertEqual(
            [access.coord for access in first_stage_a.accesses],
            [(i, 0) for i in range(8)],
        )
        self.assertEqual({access.coord for access in first_x.accesses}, {(0,)})
        self.assertEqual(events[16].site, "tmp.store")

        first_second_stage_a, first_tmp = events[17:19]
        self.assertEqual(first_second_stage_a.site, "A.stage2.load")
        self.assertEqual(
            [access.coord for access in first_second_stage_a.accesses],
            [(0, j) for j in range(8)],
        )
        self.assertEqual({access.coord for access in first_tmp.accesses}, {(0,)})
        self.assertEqual(events[-1].site, "y.store")
        self.assertEqual(
            sequences[0].event_ids,
            tuple(event.id for event in events[:17]),
        )
        self.assertEqual(
            sequences[1].event_ids,
            tuple(event.id for event in events[17:]),
        )

    def test_universal_scopes_capture_both_a_access_orientations(self) -> None:
        config = problem.build_config(problem_size=8, block_size=8)
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
                tuple((i, 0) for i in range(8)),
                tuple((0, j) for j in range(8)),
            },
        )
        self.assertEqual(
            {edge.weight for edge in fine.edges_by_array["A"]}, {8.0}
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


class AtaxEvaluatorTests(unittest.TestCase):
    def test_linear_layout_descriptor_supports_a_canonical_outer_word(self) -> None:
        layout = evaluate.canonical_layout(
            "linear:1,1:2,3:ij", "a_word"
        )
        source = evaluate.generate_source(
            layout,
            n=4,
            samples=1,
            iterations=1,
            warmup=0,
            device=0,
            block_size=4,
        )

        self.assertEqual(layout.a_rows, (0x2, 0x3))
        self.assertEqual(layout.outer_word, "ij")
        self.assertIn("first) >> 1", source)
        self.assertIn("second) >> 1", source)

    def test_linear_layout_descriptor_emits_xor_address_code(self) -> None:
        layout = evaluate.canonical_layout(
            "linear:2,2:5,2,8,4", "a_word"
        )
        source = evaluate.generate_source(
            layout,
            n=4,
            samples=1,
            iterations=1,
            warmup=0,
            device=0,
            block_size=4,
        )

        self.assertEqual(layout.a_rows, (0x5, 0x2, 0x8, 0x4))
        self.assertIn("linear A rows(low -> high)", source)
        self.assertIn(
            "(static_cast<uint64_t>(first) >> 0) ^ "
            "(static_cast<uint64_t>(second) >> 0)",
            source,
        )

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
        self.assertIn("__global__ void atax_tmp_kernel", source)
        self.assertIn("__global__ void atax_y_kernel", source)
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
            source_path = Path(directory) / "generated_atax.cu"
            self.assertTrue(source_path.exists())
            self.assertIn("Layouts: A=%s", source_path.read_text())
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
