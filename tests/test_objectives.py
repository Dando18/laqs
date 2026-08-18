from __future__ import annotations

import unittest

from relay import (
    Access,
    EventFilter,
    EventSequence,
    GroupedRegions,
    LanePrefixRegions,
    MatrixSpec,
    MemoryEvent,
    PerLaneTemporalRegions,
    TemporalWindowRegions,
)
from relay.objectives import build_objectives


class ObjectiveTests(unittest.TestCase):
    def test_multiple_arrays_and_event_types(self) -> None:
        matrices = {
            "A": MatrixSpec("A", (8, 8), 4, ("i", "j")),
            "B": MatrixSpec("B", (8, 8), 4, ("i", "j")),
        }
        event0 = MemoryEvent.make(
            "e0",
            "load",
            [
                Access("A", (0, lane), lane=lane)
                for lane in range(4)
            ]
            + [Access("B", (lane, 0), lane=lane) for lane in range(4)],
            metadata={"workgroup": "wg0", "phase": "p0"},
        )
        event1 = MemoryEvent.make(
            "e1",
            "load",
            [Access("A", (1, lane), lane=lane) for lane in range(4)],
            metadata={"workgroup": "wg0", "phase": "p0"},
        )
        events = {event.id: event for event in (event0, event1)}
        sequences = (EventSequence.make("seq", ("e0", "e1")),)
        specs = (
            LanePrefixRegions("lane", ((2, 8), (4, 16))),
            TemporalWindowRegions("window", 32, window=2),
            GroupedRegions("phase", 64, group_by=("workgroup", "phase")),
        )
        components = build_objectives(specs, matrices, events, sequences)
        self.assertEqual(len(components), 4)
        self.assertEqual(len(components[0].edges_by_array["A"]), 4)
        self.assertEqual(len(components[0].edges_by_array["B"]), 2)
        self.assertEqual(len(next(c for c in components if c.name == "window").edges_by_array["A"]), 1)
        self.assertIn("B", next(c for c in components if c.name == "phase").edges_by_array)

    def test_per_lane_temporal_windows_filter_interleaved_events(self) -> None:
        matrices = {
            "A": MatrixSpec("A", (8, 8), 4, ("i", "j")),
            "B": MatrixSpec("B", (8, 8), 4, ("i", "j")),
            "x": MatrixSpec("x", (8,), 4, ("j",), target=False),
        }
        ordered: list[MemoryEvent] = []
        for j in range(4):
            ordered.extend(
                (
                    MemoryEvent.make(
                        f"A{j}",
                        "A.load",
                        [Access("A", (lane, j), lane=lane) for lane in range(2)],
                    ),
                    MemoryEvent.make(
                        f"B{j}",
                        "B.load",
                        [Access("B", (lane, j), lane=lane) for lane in range(2)],
                    ),
                    MemoryEvent.make(
                        f"x{j}",
                        "x.load",
                        [Access("x", (j,), lane=lane) for lane in range(2)],
                    ),
                )
            )
        events = {event.id: event for event in ordered}
        sequences = (
            EventSequence.make(
                "wave0", (event.id for event in ordered), weight=3.0
            ),
        )
        matrix_reads = EventFilter.make(arrays=("A", "B"), kinds=("read",))
        components = build_objectives(
            (
                PerLaneTemporalRegions(
                    "lane_window128",
                    region_bytes=128,
                    windows=(2, 4),
                    event_filter=matrix_reads,
                ),
            ),
            matrices,
            events,
            sequences,
        )

        self.assertEqual(
            [component.name for component in components],
            ["lane_window128.window2", "lane_window128.window4"],
        )
        window2 = components[0]
        self.assertEqual(len(window2.edges_by_array["A"]), 6)
        self.assertEqual(len(window2.edges_by_array["B"]), 6)
        self.assertEqual(window2.edges_by_array["A"][0].points, ((0, 0), (0, 1)))
        self.assertEqual(window2.edges_by_array["A"][0].weight, 3.0)
        self.assertEqual(
            window2.edges_by_array["A"][0].source,
            "wave0:A:lane0[0:2]",
        )

        window4 = components[1]
        self.assertEqual(len(window4.edges_by_array["A"]), 2)
        self.assertEqual(len(window4.edges_by_array["B"]), 2)
        self.assertEqual(
            window4.edges_by_array["A"][1].points,
            ((1, 0), (1, 1), (1, 2), (1, 3)),
        )

    def test_per_lane_temporal_windows_require_lane_ids(self) -> None:
        matrix = MatrixSpec("A", (2, 2), 4, ("i", "j"))
        event = MemoryEvent.make("e", "load", [Access("A", (0, 0))])
        spec = PerLaneTemporalRegions("lane_window", 16, (1,))
        with self.assertRaisesRegex(ValueError, "requires lane ids"):
            spec.build(
                {"A": matrix},
                {event.id: event},
                (EventSequence.make("seq", (event.id,)),),
            )


if __name__ == "__main__":
    unittest.main()
