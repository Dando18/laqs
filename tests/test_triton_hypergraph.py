from __future__ import annotations

import unittest

from relay import (
    HardwareLocation,
    MatrixSpec,
    ObservedAccess,
    TritonLinearLayout,
    induce_memory_event,
    row_major_layout,
    validate_induced_hypergraph,
)


class TritonLinearLayoutTests(unittest.TestCase):
    def test_apply_matches_tritons_swizzled_header_example(self) -> None:
        layout = TritonLinearLayout.from_bases(
            (
                ("thread", ((1, 1), (2, 2))),
                ("warp", ((0, 1), (0, 2))),
            ),
            (("x", 4), ("y", 4)),
        )

        self.assertEqual(layout.apply({"thread": 0, "warp": 0}), (0, 0))
        self.assertEqual(layout.apply({"thread": 3, "warp": 0}), (3, 3))
        self.assertEqual(layout.apply({"thread": 0, "warp": 3}), (0, 3))
        self.assertEqual(layout.apply({"thread": 3, "warp": 3}), (3, 0))

    def test_cohort_enumeration_fixes_non_issue_dimensions(self) -> None:
        layout = one_wave_layout()

        locations = layout.locations(
            fixed={"register": 0, "warp": 0, "block": 0}
        )

        self.assertEqual(len(locations), 8)
        self.assertEqual(
            [location.value("lane") for location in locations],
            list(range(8)),
        )

    def test_blocked_layout_scales_lane_bases_after_register_bases(self) -> None:
        layout = TritonLinearLayout.from_blocked(
            (8,),
            size_per_thread=(2,),
            threads_per_warp=(4,),
            warps_per_cta=(1,),
            order=(0,),
        )

        self.assertEqual(
            layout.bases,
            (
                ("register", ((1,),)),
                ("lane", ((2,), (4,))),
                ("warp", ()),
                ("block", ()),
            ),
        )
        self.assertEqual(
            layout.apply({"register": 1, "lane": 3, "warp": 0, "block": 0}),
            (7,),
        )

    def test_blocked_layout_respects_tritons_minor_to_major_order(self) -> None:
        layout = TritonLinearLayout.from_blocked(
            (4, 8),
            size_per_thread=(1, 2),
            threads_per_warp=(4, 2),
            warps_per_cta=(1, 2),
            order=(1, 0),
        )

        self.assertEqual(
            layout.bases,
            (
                ("register", ((0, 1),)),
                ("lane", ((0, 2), (1, 0), (2, 0))),
                ("warp", ((0, 4),)),
                ("block", ()),
            ),
        )
        self.assertEqual(
            layout.apply({"register": 1, "lane": 5, "warp": 1, "block": 0}),
            (2, 7),
        )

    def test_incomplete_location_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*warp"):
            one_wave_layout().apply(
                {"register": 0, "lane": 0, "block": 0}
            )


class InducedHypergraphValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = MatrixSpec("input", (8,), 4, ("i",))
        self.execution_layout = one_wave_layout()
        self.locations = self.execution_layout.locations(
            fixed={"register": 0, "warp": 0, "block": 0}
        )
        self.induced = induce_memory_event(
            self.execution_layout,
            self.matrix,
            self.locations,
            id="tile.load.wave0",
            site="tile.load",
        )
        self.memory_layout = row_major_layout(self.matrix)

    def reference_trace(self) -> tuple[ObservedAccess, ...]:
        return tuple(
            ObservedAccess(location, (lane,), lane * 4)
            for lane, location in enumerate(self.locations)
        )

    def test_induced_event_preserves_hardware_lane_identity(self) -> None:
        self.assertEqual(
            [access.coord for access in self.induced.event.accesses],
            [(lane,) for lane in range(8)],
        )
        self.assertEqual(
            [access.lane for access in self.induced.event.accesses],
            list(range(8)),
        )
        self.assertEqual(
            self.induced.hyperedge.points,
            tuple((lane,) for lane in range(8)),
        )

    def test_reference_addresses_and_transaction_groups_validate(self) -> None:
        validation = validate_induced_hypergraph(
            self.induced,
            self.matrix,
            self.memory_layout,
            self.reference_trace(),
            transaction_bytes=16,
        )

        self.assertTrue(validation.valid)
        self.assertEqual(validation.expected_transaction_ids, (0, 1))
        self.assertEqual(validation.observed_transaction_ids, (0, 1))
        self.assertEqual(validation.expected_quotient_count, 2)
        self.assertEqual(validation.observed_transaction_count, 2)
        self.assertEqual(validation.expected_groups, validation.observed_groups)
        validation.require_valid()

    def test_bit_order_error_is_visible_even_when_quotient_count_matches(self) -> None:
        wrong_layout = TritonLinearLayout.from_bases(
            (
                ("register", ()),
                ("lane", ((2,), (1,), (4,))),
                ("warp", ()),
                ("block", ()),
            ),
            (("dim0", 8),),
        )
        wrong_event = induce_memory_event(
            wrong_layout,
            self.matrix,
            self.locations,
            id="tile.load.wave0",
            site="tile.load",
        )

        validation = validate_induced_hypergraph(
            wrong_event,
            self.matrix,
            self.memory_layout,
            self.reference_trace(),
            transaction_bytes=16,
        )

        self.assertFalse(validation.valid)
        self.assertEqual(validation.expected_quotient_count, 2)
        self.assertEqual(validation.observed_transaction_count, 2)
        self.assertIn(
            "logical-coordinate",
            {mismatch.kind for mismatch in validation.mismatches},
        )
        with self.assertRaisesRegex(ValueError, "logical-coordinate"):
            validation.require_valid()

    def test_wrong_address_grouping_is_reported(self) -> None:
        trace = list(self.reference_trace())
        last = trace[-1]
        trace[-1] = ObservedAccess(
            last.location,
            last.logical_coord,
            32,
        )

        validation = validate_induced_hypergraph(
            self.induced,
            self.matrix,
            self.memory_layout,
            trace,
            transaction_bytes=16,
        )

        self.assertFalse(validation.valid)
        self.assertEqual(validation.observed_transaction_ids, (0, 1, 2))
        self.assertEqual(
            {mismatch.kind for mismatch in validation.mismatches},
            {"byte-offset", "transaction-ids"},
        )

    def test_register_dimension_models_multiple_elements_per_lane(self) -> None:
        execution_layout = TritonLinearLayout.from_blocked(
            (8,),
            size_per_thread=(2,),
            threads_per_warp=(4,),
            warps_per_cta=(1,),
            order=(0,),
        )
        locations = execution_layout.locations(fixed={"warp": 0, "block": 0})
        induced = induce_memory_event(
            execution_layout,
            self.matrix,
            locations,
            id="tile.vector-load.wave0",
            site="tile.vector-load",
        )
        trace = tuple(
            ObservedAccess(
                location,
                execution_layout.apply(location),
                execution_layout.apply(location)[0] * 4,
            )
            for location in locations
        )

        validation = validate_induced_hypergraph(
            induced,
            self.matrix,
            self.memory_layout,
            trace,
            transaction_bytes=16,
        )

        self.assertTrue(validation.valid)
        self.assertEqual(
            [access.lane for access in induced.event.accesses],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )


def one_wave_layout() -> TritonLinearLayout:
    return TritonLinearLayout.from_blocked(
        (8,),
        size_per_thread=(1,),
        threads_per_warp=(8,),
        warps_per_cta=(1,),
        order=(0,),
    )


if __name__ == "__main__":
    unittest.main()
