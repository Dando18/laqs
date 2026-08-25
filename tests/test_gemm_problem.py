from __future__ import annotations

import unittest

from kernels.gemm import problem
from relay import (
    MI300A_V1,
    UNIVERSAL_V1_BASIS,
    UniversalScopeObjectives,
)
from relay.objectives import build_objectives


class GemmProblemTests(unittest.TestCase):
    def test_inner_sequences_and_universal_scopes_are_explicit(self) -> None:
        config = problem.build_config(
            problem_size=16,
            block_size=(8, 8, 1),
        )
        matrices = {matrix.name: matrix for matrix in problem.get_matrices(config)}
        event_items, sequences = problem.get_events_and_sequences(config)
        events = {event.id: event for event in event_items}

        self.assertEqual(len(event_items), 2 * 16 + 2)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(
            sequences[0].event_ids,
            tuple(event.id for event in event_items),
        )
        self.assertEqual(events[sequences[0].event_ids[0]].meta("step"), "0")
        self.assertEqual(events[sequences[0].event_ids[-1]].meta("step"), "0")

        components = build_objectives(
            (UniversalScopeObjectives(MI300A_V1.byte_scales),),
            matrices,
            events,
            sequences,
        )
        by_name = {component.name: component for component in components}
        fine = by_name[MI300A_V1.fine_component]

        self.assertEqual(fine.edge_family, "issue.g64.stream.load")
        self.assertEqual(set(fine.edges_by_array), {"A", "B", "C"})
        self.assertEqual(
            fine.edges_by_array["A"][0].points,
            tuple((i, 0) for i in range(8)),
        )
        self.assertEqual(
            fine.edges_by_array["B"][0].points,
            tuple((0, j) for j in range(8)),
        )
        self.assertIn("issue.g64.stream.store.64B", by_name)

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


if __name__ == "__main__":
    unittest.main()
