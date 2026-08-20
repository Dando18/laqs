from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from kernels.syrk import evaluate, problem
from relay import (
    MI300A_V1,
    UNIVERSAL_V1_BASIS,
    UniversalScopeObjectives,
)
from relay.objectives import build_objectives


class SyrkProblemTests(unittest.TestCase):
    def test_problem_models_both_a_streams_for_a_complete_workgroup(self) -> None:
        config = problem.build_config(
            problem_size=8,
            block_size=(8, 8, 1),
        )
        matrices = problem.get_matrices(config)
        events, sequences = problem.get_events_and_sequences(config)

        self.assertEqual(
            [(matrix.name, matrix.shape, matrix.target) for matrix in matrices],
            [("A", (8, 8), True), ("C", (8, 8), True)],
        )
        self.assertEqual(len(events), 2 * 8 + 2)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].event_ids, tuple(event.id for event in events))

        row_i, row_j = events[1:3]
        self.assertEqual(row_i.site, "A.row_i.load")
        self.assertEqual(row_j.site, "A.row_j.load")
        self.assertEqual(
            {access.coord for access in row_i.accesses},
            {(i, 0) for i in range(8)},
        )
        self.assertEqual(
            {access.coord for access in row_j.accesses},
            {(j, 0) for j in range(8)},
        )
        self.assertEqual(events[0].site, "C.load")
        self.assertEqual(events[-1].site, "C.store")
        self.assertEqual(events[0].meta("step"), "0")
        self.assertEqual(events[-1].meta("step"), "0")

    def test_universal_scopes_cover_loads_and_stores(self) -> None:
        config = problem.build_config(
            problem_size=8,
            block_size=(8, 8, 1),
        )
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
        self.assertEqual(set(fine.edges_by_array), {"A", "C"})
        self.assertEqual(
            fine.edges_by_array["A"][0].points,
            tuple((i, 0) for i in range(8)),
        )
        stores = by_name["issue.g64.stream.store.64B"]
        self.assertEqual(set(stores.edges_by_array), {"C"})

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


class SyrkEvaluatorTests(unittest.TestCase):
    def test_canonical_layout_and_generated_source(self) -> None:
        layouts = (
            evaluate.canonical_layout("jjii", "a_word"),
            evaluate.canonical_layout("iijj", "c_word"),
        )
        source = evaluate.generate_source(
            layouts,
            n=16,
            samples=3,
            iterations=4,
            warmup=2,
            device=1,
            block_x=8,
            block_y=16,
        )

        self.assertEqual(layouts[0].tile_rows, 4)
        self.assertEqual(layouts[0].tile_columns, 4)
        self.assertIn("__global__ void syrk_kernel", source)
        self.assertIn("a[a_offset(i, k, n)]", source)
        self.assertIn("a[a_offset(j, k, n)]", source)
        self.assertIn("word(low -> high)=jjii", source)
        self.assertIn("word(low -> high)=iijj", source)
        self.assertIn("constexpr uint32_t n = 16;", source)
        self.assertIn("constexpr uint32_t block_x = 8;", source)
        self.assertIn("constexpr uint32_t block_y = 16;", source)
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
                        "iiijjj",
                        "--n",
                        "64",
                        "--block-x",
                        "16",
                        "--block-y",
                        "16",
                        "--build-dir",
                        directory,
                        "--emit-only",
                    ]
                )

            self.assertEqual(status, 0)
            source_path = Path(directory) / "generated_syrk.cu"
            self.assertTrue(source_path.exists())
            self.assertIn("Layouts: A=%s C=%s", source_path.read_text())
            self.assertIn(str(source_path.resolve()), output.getvalue())

    def test_layout_words_must_tile_the_problem(self) -> None:
        errors = StringIO()
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as caught,
        ):
            evaluate.run(["iiii", "jjj", "--n", "8", "--emit-only"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("must be divisible", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
