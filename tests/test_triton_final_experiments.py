from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from analyze import correlation_rows
from layout_panels import (
    _stable_id,
    _stratified_candidates,
    rescore_recorded_panel,
)


def component(name: str, amplification: float) -> dict[str, object]:
    return {
        "name": name,
        "raw_region_count": 10.0 * amplification,
        "packing_lower_bound": 10.0,
        "excess_footprint": amplification - 1.0,
    }


class FinalExperimentAnalysisTests(unittest.TestCase):
    def test_recorded_panel_rescore_preserves_mapping_and_adds_windows(self):
        from relay import EdgeFamily, Hyperedge, MatrixSpec, ScopeKey

        matrix = MatrixSpec("weight", (4, 4), 4, ("M", "K"), target=True)
        edge = Hyperedge.make(((0, 0), (0, 1)))
        families = (
            EdgeFamily(
                ScopeKey("issue", 32, "stream", "load"),
                {matrix.name: (edge,)},
                64.0,
            ),
            EdgeFamily(
                ScopeKey("simd_window", 64, "stream", "load"),
                {matrix.name: (edge,)},
                64.0,
            ),
        )
        analysis = SimpleNamespace(
            matrices=(matrix,),
            edge_families=families,
            selected_config={"laqs_scope_temporal_windows": [4, 16, 64]},
            manifest=SimpleNamespace(schema="test", version=1),
            grid=(1,),
            allocations=(SimpleNamespace(name="weight", eligible=True),),
            events=(),
            sequences=(),
            require_supported=lambda: None,
        )
        rows = [1 << bit for bit in range(matrix.total_bits)]
        mapping_id = _stable_id("mapping", rows)
        source = {
            "mode": "experiment1_gc_whole",
            "candidates": [
                {
                    "candidate_id": "saved-candidate",
                    "mapping_id": mapping_id,
                    "layout": "saved-layout",
                    "a_rows": rows,
                    "sample_index": 1,
                    "sampling_attempt": 1,
                    "sampling_origin": "row_major_anchor",
                }
            ],
        }

        result = rescore_recorded_panel(
            analysis,
            "weight",
            source,
            platform="matrix",
            source_path=Path("saved-report.json"),
        )

        self.assertEqual(result["candidates"][0]["mapping_id"], mapping_id)
        self.assertEqual(
            result["score_profile"]["component_model"]["scope_basis"][
                "temporal_windows"
            ],
            [4, 16, 64],
        )
        self.assertIn(
            "simd_window.t64.stream.load.256B",
            {
                item["name"]
                for item in result["candidates"][0]["score"]["components"]
            },
        )

    def test_issue_stratification_balances_issue_cells_and_keeps_anchors(self):
        candidates = []
        for index, amplification in enumerate((1.0, 8.0, 1.2, 1.7, 3.0, 5.0)):
            candidates.append(
                {
                    "candidate_id": f"candidate-{index}",
                    "sample_index": index + 1,
                    "sampling_origin": (
                        "row_major_anchor"
                        if index == 0
                        else "column_major_anchor"
                        if index == 1
                        else "random_pool"
                    ),
                    "score": {
                        "components": [
                            component("issue.g32.stream.load.32B", amplification),
                            component(
                                "simd_window.t16.stream.load.32B",
                                1.0 if index % 2 else 8.0,
                            ),
                        ]
                    },
                }
            )

        selected, metadata = _stratified_candidates(
            candidates, samples=5, mode="issue"
        )

        self.assertEqual(len(selected), 5)
        self.assertEqual(
            [item["candidate_id"] for item in selected[:2]],
            ["candidate-0", "candidate-1"],
        )
        self.assertEqual(metadata["active_feature_count"], 1)
        self.assertEqual(
            metadata["active_features"], ["issue.g32.stream.load.32B"]
        )
        selected_counts = metadata["selected_marginal_stratum_counts"][
            "issue.g32.stream.load.32B"
        ]
        self.assertEqual(set(selected_counts), {"0", "1", "2", "3", "4"})

    def test_correlations_promote_every_byte_scale_and_j_area(self):
        components = (
            "issue.g32.stream.load.32B",
            "issue.g32.stream.load.64B",
            "issue.g32.stream.load.128B",
            "issue.g32.stream.load.256B",
        )
        candidates = []
        for index, value in enumerate((1.0, 2.0, 3.0), 1):
            candidates.append(
                {
                    "candidate_id": f"candidate-{index}",
                    "mapping_id": f"mapping-{index}",
                    "j_area": value,
                    "peak_normalized_excess": value,
                    "score": {
                        "components": [component(name, value) for name in components]
                    },
                    "counters": {
                        "steady_state": {"l1_miss_demand_to_l2": value}
                    },
                }
            )
        report = {
            "final_experiment": 1,
            "case": "gemv",
            "panel": {
                "stratification": {"mode": "all"},
                "score_profile": {
                    "platform": "matrix",
                    "counter_components": {
                        "l1_miss_demand_to_l2": components[0]
                    },
                    "focus_counters": ["l1_miss_demand_to_l2"],
                },
            },
            "candidates": candidates,
        }

        rows = correlation_rows(report)
        primary = [
            row
            for row in rows
            if row["relationship"]
            in {"counter_component", "counter_scope_bytescale", "aggregate"}
        ]

        self.assertEqual(len(primary), 5)
        self.assertEqual(
            {
                row["predictor_byte_scale"]
                for row in primary
                if row["predictor"].startswith("Q:")
            },
            {32, 64, 128, 256},
        )
        self.assertTrue(all(row["spearman_rho"] == 1.0 for row in primary))


if __name__ == "__main__":
    unittest.main()
