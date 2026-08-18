from __future__ import annotations

import unittest

from kernels.gemm import problem
from relay.objectives import build_objectives


class GemmProblemTests(unittest.TestCase):
    def test_inner_sequences_and_locality_objectives_are_explicit(self) -> None:
        config = problem.build_config(
            problem_size=16,
            block_size=(8, 8, 1),
        )
        matrices = {matrix.name: matrix for matrix in problem.get_matrices(config)}
        event_items, sequences = problem.get_events_and_sequences(config)
        events = {event.id: event for event in event_items}

        self.assertEqual(len(event_items), 2 * 16 + 2)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(sequences[0].event_ids), 2 * 16)
        self.assertTrue(
            all(
                events[event_id].meta("phase") == "inner"
                for event_id in sequences[0].event_ids
            )
        )

        components = build_objectives(
            problem.get_objectives(config), matrices, events, sequences
        )
        by_name = {component.name: component for component in components}
        self.assertEqual(len(components), 11)
        self.assertEqual(
            {
                component.name
                for component in components
                if component.provenance == "grounded"
            },
            {"wave_load.64B", "output_store.64B"},
        )
        self.assertEqual(
            len(by_name["workgroup_k_panel.256B"].edges_by_array["A"]),
            16,
        )
        self.assertEqual(
            len(by_name["wave_k_window.4096B"].edges_by_array["B"]),
            1,
        )
        self.assertEqual(
            set(problem.get_component_weights(config)),
            set(by_name),
        )
        self.assertEqual(
            problem.get_component_weights(config)["wave_load.64B"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
