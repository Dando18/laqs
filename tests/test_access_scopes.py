from __future__ import annotations

import unittest

from kernels.atax import problem as atax_problem
from kernels.gesummv import problem as gesummv_problem
from relay import (
    Access,
    EventSequence,
    Hyperedge,
    MatrixSpec,
    MemoryEvent,
    ResourceCohort,
)
from relay.access_scopes import (
    UNIVERSAL_V1_BASIS,
    _compress_edges,
    build_edge_families,
    build_resource_cohorts,
    compress_resource_cohorts,
    materialize_edge_families,
)


def kernel_families(module):
    config = module.build_config(problem_size=8, block_size=8)
    matrices = {matrix.name: matrix for matrix in module.get_matrices(config)}
    event_items, sequences = module.get_events_and_sequences(config)
    events = {event.id: event for event in event_items}
    return matrices, build_edge_families(matrices, events, sequences)


class UniversalAccessScopeTests(unittest.TestCase):
    def test_schema_is_global_and_separates_scope_from_byte_scale(self) -> None:
        schema = {scope.name for scope in UNIVERSAL_V1_BASIS.scope_keys()}

        self.assertIn("issue.g64.stream.load", schema)
        self.assertIn("lane_window.t16.array.store", schema)
        self.assertIn("simd_window.t4.stream.atomic", schema)
        self.assertIn("workgroup_step.array.load", schema)
        self.assertIn("workgroup_window.t16.stream.load", schema)
        self.assertIn("phase.workgroup.array.load", schema)
        self.assertTrue(all(not name.endswith("B") for name in schema))

    def test_atax_and_gesummv_use_the_same_family_types(self) -> None:
        _, atax = kernel_families(atax_problem)
        _, gesummv = kernel_families(gesummv_problem)

        self.assertEqual(
            {family.name for family in atax},
            {family.name for family in gesummv},
        )
        self.assertEqual(len(atax), 24)
        self.assertTrue(
            all(
                token not in family.name.lower()
                for family in atax
                for token in ("atax", "stage1", "stage2", "a.")
            )
        )

    def test_translated_events_are_compressed_with_dynamic_multiplicity(self) -> None:
        _, families = kernel_families(gesummv_problem)
        by_name = {family.name: family for family in families}

        issue = by_name["issue.g64.stream.load"]
        self.assertEqual(issue.normalization_bytes, 1024.0)
        self.assertEqual(len(issue.edges_by_array["A"]), 1)
        self.assertEqual(issue.edges_by_array["A"][0].weight, 8.0)
        self.assertEqual(len(issue.edges_by_array["A"][0].points), 8)

        lane_window = by_name["lane_window.t4.stream.load"]
        self.assertEqual(len(lane_window.edges_by_array["A"]), 1)
        self.assertEqual(lane_window.edges_by_array["A"][0].weight, 16.0)

    def test_translated_non_affine_edges_use_an_exact_canonical_key(self) -> None:
        matrix = MatrixSpec("A", (8, 8), 8, ("i", "j"))
        points = ((0, 0), (0, 1), (1, 0))
        translation = (3, 2)
        translated = tuple(
            (i ^ translation[0], j ^ translation[1]) for i, j in points
        )

        compressed = _compress_edges(
            matrix,
            (
                Hyperedge.make(points, weight=2, source="base"),
                Hyperedge.make(translated, weight=5, source="translated"),
            ),
        )

        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0].points, points)
        self.assertEqual(compressed[0].weight, 7.0)
        self.assertIn("+1 XOR translations", compressed[0].source)

    def test_large_non_affine_edges_remain_explicit(self) -> None:
        matrix = MatrixSpec("A", (512, 2), 8, ("i", "j"))
        points = tuple((i, 0) for i in range(257))
        translated = tuple((i ^ 1, 1) for i in range(257))

        compressed = _compress_edges(
            matrix,
            (
                Hyperedge.make(points, weight=2, source="base"),
                Hyperedge.make(translated, weight=5, source="translated"),
            ),
        )

        self.assertEqual(len(compressed), 2)
        self.assertEqual([edge.weight for edge in compressed], [2.0, 5.0])
        self.assertTrue(
            all("XOR translations" not in edge.source for edge in compressed)
        )

    def test_one_family_is_materialized_at_every_profile_scale(self) -> None:
        matrices, families = kernel_families(gesummv_problem)
        issue = next(
            family
            for family in families
            if family.name == "issue.g64.stream.load"
        )
        components = materialize_edge_families(
            (issue,), matrices, (64, 128, 512)
        )

        self.assertEqual(
            [component.name for component in components],
            [
                "issue.g64.stream.load.64B",
                "issue.g64.stream.load.128B",
                "issue.g64.stream.load.512B",
            ],
        )
        self.assertTrue(
            all(component.edge_family == issue.name for component in components)
        )
        self.assertTrue(
            all(
                component.edges_by_array is issue.edges_by_array
                for component in components
            )
        )

    def test_simd_windows_use_unfiltered_global_event_slots(self) -> None:
        matrices = {
            "A": MatrixSpec("A", (8, 8), 8, ("i", "j"), target=True),
            "x": MatrixSpec("x", (8,), 8, ("i",), target=False),
        }
        event_items = (
            MemoryEvent.make(
                "a0", "A.load", (Access("A", (0, 0), lane=0),), order=0
            ),
            MemoryEvent.make(
                "x0", "x.load", (Access("x", (0,), lane=0),), order=1
            ),
            MemoryEvent.make(
                "a1", "A.load", (Access("A", (0, 1), lane=0),), order=2
            ),
            MemoryEvent.make(
                "x1", "x.load", (Access("x", (1,), lane=0),), order=3
            ),
            MemoryEvent.make(
                "a2", "A.load", (Access("A", (0, 2), lane=0),), order=4
            ),
        )
        events = {event.id: event for event in event_items}
        sequences = (
            EventSequence.make("wave0", tuple(event.id for event in event_items)),
        )

        families = {
            family.name: family
            for family in build_edge_families(matrices, events, sequences)
        }
        edges = families["simd_window.t4.stream.load"].edges_by_array["A"]

        self.assertEqual(
            {edge.points for edge in edges},
            {((0, 0), (0, 1)), ((0, 2),)},
        )

    def test_resource_cohorts_retain_cross_allocation_accesses(self) -> None:
        matrices = {
            "A": MatrixSpec("A", (8,), 8, ("i",), target=True),
            "x": MatrixSpec("x", (8,), 8, ("i",), target=False),
        }
        event_items = (
            MemoryEvent.make(
                "a0", "A.load", (Access("A", (0,), lane=0),), order=0
            ),
            MemoryEvent.make(
                "x0", "x.load", (Access("x", (0,), lane=0),), order=1
            ),
            MemoryEvent.make(
                "a1", "A.load", (Access("A", (1,), lane=0),), order=2
            ),
        )
        events = {event.id: event for event in event_items}
        sequences = (EventSequence.make("wave0", tuple(events)),)

        cohorts = build_resource_cohorts(
            matrices,
            events,
            sequences,
            ("simd_window.t2.cohort.load",),
        )["simd_window.t2.cohort.load"]

        self.assertEqual(len(cohorts), 2)
        self.assertEqual(
            [access.array for access in cohorts[0].accesses],
            ["A", "x"],
        )
        self.assertEqual([access.array for access in cohorts[1].accesses], ["A"])

    def test_resource_cohort_xor_translations_are_compressed(self) -> None:
        matrices = {"A": MatrixSpec("A", (8,), 8, ("i",))}
        cohorts = (
            ResourceCohort(
                "simd_window.t2.cohort.load",
                (Access("A", (0,)), Access("A", (1,))),
                2.0,
                "first",
            ),
            ResourceCohort(
                "simd_window.t2.cohort.load",
                (Access("A", (6,)), Access("A", (7,))),
                3.0,
                "second",
            ),
        )

        compressed = compress_resource_cohorts(matrices, cohorts)

        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0].weight, 5.0)
        self.assertIn("+1 XOR translations", compressed[0].source)

    def test_trace_contract_rejects_incomplete_and_ambiguous_traces(self) -> None:
        matrices = {
            "A": MatrixSpec("A", (2, 2), 8, ("i", "j"), target=True)
        }
        event0 = MemoryEvent.make(
            "e0", "A.load", (Access("A", (0, 0), lane=0),), order=0
        )
        event1 = MemoryEvent.make(
            "e1", "A.load", (Access("A", (0, 1), lane=0),), order=1
        )
        events = {event.id: event for event in (event0, event1)}

        with self.assertRaisesRegex(ValueError, "contain every event at least once"):
            build_edge_families(
                matrices,
                events,
                (EventSequence.make("incomplete", ("e0",)),),
            )
        with self.assertRaisesRegex(ValueError, "appears more than once in sequence"):
            build_edge_families(
                matrices,
                events,
                (EventSequence.make("duplicate", ("e0", "e0", "e1")),),
            )
        weighted_event = MemoryEvent.make(
            "e1",
            "A.load",
            (Access("A", (0, 1), lane=0),),
            order=1,
            weight=2,
        )
        with self.assertRaisesRegex(
            ValueError, "one common MemoryEvent.weight within the trace class"
        ):
            build_edge_families(
                matrices,
                {"e0": event0, "e1": weighted_event},
                (EventSequence.make("mixed", ("e0", "e1")),),
            )

        with self.assertRaisesRegex(ValueError, "not in nondecreasing"):
            build_edge_families(
                matrices,
                events,
                (EventSequence.make("reversed", ("e1", "e0")),),
            )

    def test_trace_class_multiplicity_weights_every_universal_scope(self) -> None:
        matrices = {"A": MatrixSpec("A", (8,), 4, ("i",), target=True)}

        def event(event_id: str, coord: int, order: int, step: int) -> MemoryEvent:
            return MemoryEvent.make(
                event_id,
                "A.load",
                (Access("A", (coord,), lane=0),),
                order=order,
                weight=3,
                metadata={
                    "workgroup": "wg0",
                    "wave": "wave0",
                    "step": step,
                    "phase": "body",
                },
            )

        event_items = (
            event("shared", 0, 0, 0),
            event("interior.next", 1, 1, 1),
            event("boundary.next", 2, 1, 1),
        )
        events = {item.id: item for item in event_items}
        sequences = (
            EventSequence.make("interior", ("shared", "interior.next"), weight=2),
            EventSequence.make("boundary", ("shared", "boundary.next"), weight=5),
        )

        families = {
            family.name: family
            for family in build_edge_families(matrices, events, sequences)
        }

        self.assertTrue(
            all(family.normalization_bytes == 168.0 for family in families.values())
        )
        self.assertEqual(
            families["issue.g64.stream.load"].edges_by_array["A"][0].weight,
            42.0,
        )
        for family_name in (
            "lane_window.t4.stream.load",
            "simd_window.t4.array.load",
            "workgroup_window.t4.array.load",
            "phase.workgroup.array.load",
        ):
            self.assertEqual(
                sorted(
                    edge.weight
                    for edge in families[family_name].edges_by_array["A"]
                ),
                [6.0, 15.0],
            )
        self.assertEqual(
            families["workgroup_step.array.load"].edges_by_array["A"][0].weight,
            42.0,
        )

        cohorts = build_resource_cohorts(
            matrices,
            events,
            sequences,
            ("simd_window.t4.cohort.load",),
        )["simd_window.t4.cohort.load"]
        self.assertEqual([cohort.weight for cohort in cohorts], [6.0, 15.0])

    def test_trace_contract_accepts_one_common_event_multiplicity(self) -> None:
        matrices = {
            "A": MatrixSpec("A", (2, 2), 8, ("i", "j"), target=True)
        }
        event_items = tuple(
            MemoryEvent.make(
                f"e{index}",
                "A.load",
                (Access("A", (0, index), lane=0),),
                order=index,
                weight=3,
            )
            for index in range(2)
        )
        families = build_edge_families(
            matrices,
            {event.id: event for event in event_items},
            (EventSequence.make("wave0", ("e0", "e1")),),
        )

        self.assertTrue(
            all(family.normalization_bytes == 48.0 for family in families)
        )


if __name__ == "__main__":
    unittest.main()
