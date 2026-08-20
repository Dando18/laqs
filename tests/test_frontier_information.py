from __future__ import annotations

import unittest

from experiments.frontier_information import (
    analyze_frontier_information,
    diagnostic_layout_signature,
)
from relay import MatrixSpec, canonical_layout_from_word
from relay.objectives import Hyperedge, ObjectiveComponent


def _record(
    name: str,
    runtime: float,
    fine: float,
    first: float,
    second: float,
    dense: tuple[float, float],
) -> dict[str, object]:
    return {
        "name": name,
        "timing": {
            "median_ms": runtime,
            "samples_ms": [runtime - 0.01, runtime, runtime + 0.01],
        },
        "score": {
            "components": [
                {
                    "name": "issue.g64.stream.load.64B",
                    "weight": 0.0,
                    "raw_region_count": fine,
                    "normalized_excess": 0.0,
                    "excess_footprint": 0.0,
                },
                {
                    "name": "first",
                    "weight": 1.0,
                    "raw_region_count": first + 1.0,
                    "normalized_excess": first,
                    "excess_footprint": first,
                },
                {
                    "name": "second",
                    "weight": 1.0,
                    "raw_region_count": second + 1.0,
                    "normalized_excess": second,
                    "excess_footprint": second,
                },
            ],
            "aggregates": {
                "peak_normalized_excess": max(first, second),
                "weighted_normalized_excess": first + second,
                "hardware_peak": max(first, second),
                "hardware_area": first + second,
            },
            "codegen": {"runs": 1, "xors": 0},
        },
        "diagnostic_signatures": {
            "stream_split": {
                "first::A.row": {
                    "normalized_excess": first,
                    "excess_footprint": first,
                },
                "first::A.transpose": {
                    "normalized_excess": second,
                    "excess_footprint": second,
                },
            },
            "dense_scales": {
                "dimensions": [0, 1],
                "values": {"edge.d0": dense[0], "edge.d1": dense[1]},
            },
        },
        "frontier_representations": {},
    }


class DiagnosticSignatureTests(unittest.TestCase):
    def test_dense_curves_and_directional_partitions_are_explicit(self) -> None:
        matrix = MatrixSpec("A", (8, 8), 8, ("i", "j"), target=True)
        layout = canonical_layout_from_word(matrix, "jjjiii")
        component = ObjectiveComponent(
            "loads.64B",
            64,
            {
                "A": (
                    Hyperedge.make(
                        ((0, 0), (0, 1), (0, 2), (0, 3)),
                        source="A.row.w0",
                    ),
                    Hyperedge.make(
                        ((0, 0), (1, 0), (2, 0), (3, 0)),
                        source="A.transpose.w0",
                    ),
                )
            },
        )

        signature = diagnostic_layout_signature(
            {"A": matrix}, (component,), {"A": layout}
        )

        dense = signature["dense_scales"]
        self.assertEqual(dense["dimensions"], list(range(7)))
        self.assertEqual(
            list(dense["values"].values()),
            [8.0, 6.0, 5.0, 5.0, 3.0, 2.0, 2.0],
        )
        split = signature["stream_split"]
        self.assertEqual(
            set(split),
            {"loads.64B::A.row", "loads.64B::A.transpose"},
        )
        self.assertEqual(
            split["loads.64B::A.row"]["normalized_excess"], 0.0
        )
        self.assertEqual(
            split["loads.64B::A.transpose"]["normalized_excess"], 3.0
        )


class InformationLadderTests(unittest.TestCase):
    def test_component_frontier_recovers_an_aggregate_dominance_miss(self) -> None:
        group = {
            "kernel": "example",
            "display_name": "Example",
            "matrix_size": 8,
            "fine_component": "issue.g64.stream.load.64B",
            "results": [
                _record("winner", 1.0, 2.0, 2.0, 0.0, (0.0, 2.0)),
                _record("aggregate-best", 1.2, 1.0, 0.0, 1.0, (2.0, 0.0)),
                _record("slow", 2.0, 3.0, 3.0, 3.0, (3.0, 3.0)),
            ],
        }

        analysis = analyze_frontier_information((group,))
        by_name = {
            representation["name"]: representation
            for representation in analysis["representations"]
        }

        aggregate = by_name["aggregate"]
        active = by_name["active-components"]
        self.assertGreater(aggregate["oracle_regret"]["maximum"], 0.0)
        self.assertEqual(active["oracle_regret"]["maximum"], 0.0)
        certificate = aggregate["instances"][0][
            "missed_winner_certificates"
        ][0]
        self.assertEqual(certificate["winner"], "winner")
        self.assertEqual(certificate["dominators"][0]["name"], "aggregate-best")
        self.assertEqual(
            aggregate["dominance_violations"][
                "confirmed_nonoverlap_violation_count"
            ],
            1,
        )
        self.assertEqual(
            set(certificate["dominators"][0]["component_excess_deltas"]),
            {"issue.g64.stream.load.64B", "first", "second"},
        )
        membership = group["results"][0]["frontier_representations"]
        self.assertEqual(membership["aggregate"]["pareto_depth"], 2)
        self.assertTrue(membership["active-components"]["member"])


if __name__ == "__main__":
    unittest.main()
