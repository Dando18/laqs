from __future__ import annotations

import unittest

from relay import (
    Access,
    EventSequence,
    MatrixSpec,
    MemoryEvent,
    TemporalPersistenceBasis,
    build_transition_families,
    canonical_layout_from_word,
    score_temporal_persistence,
)


class TemporalPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = MatrixSpec("A", (8, 8), 8, ("i", "j"))
        self.b = MatrixSpec("B", (8, 8), 8, ("i", "j"))
        self.matrices = {"A": self.a, "B": self.b}

    def test_stream_turnover_distinguishes_persistent_regions(self) -> None:
        events = {
            f"A{j}": MemoryEvent.make(
                f"A{j}",
                "A.load",
                [Access("A", (i, j), lane=i) for i in range(8)],
                order=j,
            )
            for j in range(2)
        }
        sequences = (EventSequence.make("wave0", events),)
        basis = TemporalPersistenceBasis(
            deltas=(1,), families=("simd_stream", "lane_stream")
        )
        families = build_transition_families(
            self.matrices, events, sequences, basis=basis
        )

        self.assertEqual(
            {family.name for family in families},
            {"simd_stream.d1.load", "lane_stream.d1.load"},
        )
        row_major = canonical_layout_from_word(self.a, "jjjiii")
        column_major = canonical_layout_from_word(self.a, "iiijjj")
        row_score = score_temporal_persistence(
            self.matrices, {"A": row_major}, families, (16,)
        )
        column_score = score_temporal_persistence(
            self.matrices, {"A": column_major}, families, (16,)
        )

        self.assertEqual(row_score.hardware_persist, 0.0)
        self.assertEqual(column_score.hardware_persist, 9.0)

    def test_schedule_regions_are_tagged_by_allocation(self) -> None:
        events = {
            "A": MemoryEvent.make(
                "A", "A.load", [Access("A", (0, 0), lane=0)], order=0
            ),
            "B": MemoryEvent.make(
                "B", "B.load", [Access("B", (0, 0), lane=0)], order=1
            ),
        }
        sequences = (EventSequence.make("wave0", events),)
        families = build_transition_families(
            self.matrices,
            events,
            sequences,
            basis=TemporalPersistenceBasis(
                deltas=(1,), families=("simd_schedule",)
            ),
        )
        layouts = {
            name: canonical_layout_from_word(matrix, "jjjiii")
            for name, matrix in self.matrices.items()
        }
        score = score_temporal_persistence(
            self.matrices, layouts, families, (16,)
        )

        self.assertEqual(score.hardware_persist, 1.0)

    def test_transition_basis_rejects_invalid_deltas(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            TemporalPersistenceBasis(deltas=(0,))

    def test_shared_events_are_weighted_once_per_trace_class(self) -> None:
        events = {
            "shared": MemoryEvent.make(
                "shared",
                "A.load",
                [Access("A", (0, 0), lane=0)],
                order=0,
                weight=3,
            ),
            "interior": MemoryEvent.make(
                "interior",
                "A.load",
                [Access("A", (0, 1), lane=0)],
                order=1,
                weight=3,
            ),
            "boundary": MemoryEvent.make(
                "boundary",
                "A.load",
                [Access("A", (0, 2), lane=0)],
                order=1,
                weight=3,
            ),
        }
        sequences = (
            EventSequence.make("interior", ("shared", "interior"), weight=2),
            EventSequence.make("boundary", ("shared", "boundary"), weight=5),
        )
        families = build_transition_families(
            self.matrices,
            events,
            sequences,
            basis=TemporalPersistenceBasis(
                deltas=(1,), families=("simd_schedule",)
            ),
        )

        self.assertEqual(len(families), 1)
        self.assertEqual(families[0].transition_count, 2)
        self.assertEqual(families[0].transition_weight, 21.0)
        self.assertEqual(
            sorted(transition.weight for transition in families[0].transitions),
            [6.0, 15.0],
        )


if __name__ == "__main__":
    unittest.main()
