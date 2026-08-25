from __future__ import annotations

import argparse
import unittest

from experiments.interleaved_confirmation import (
    Candidate,
    analyze_measurements,
    generate_gemm_source,
    generate_gesummv_source,
    select_candidates,
)


class InterleavedConfirmationTests(unittest.TestCase):
    def test_candidate_panel_is_oracle_prefix_union_frontier(self) -> None:
        def layout(word: str, time_ms: float) -> dict[str, object]:
            return {
                "word": word,
                "timing": {"median_ms": time_ms},
            }

        record = {
            "oracle": {
                "top_layouts": [
                    layout("jjii", 1.0),
                    layout("jiji", 1.1),
                    layout("iijj", 1.2),
                ]
            },
            "frontier": {
                "layouts": [layout("jiji", 1.1), layout("ijij", 1.3)]
            },
        }

        panel = select_candidates(record, 2)

        self.assertEqual([item.word for item in panel], ["jjii", "jiji", "ijij"])
        self.assertEqual(panel[1].roles, ("oracle_top_2", "analytical_frontier"))

    def test_generated_kernels_preserve_original_runtime_operations(self) -> None:
        args = argparse.Namespace(
            rounds=8,
            iterations=3,
            warmup=2,
            priming_cycles=1,
            seed=7,
            device=0,
        )
        panel = (Candidate("jjii", ("oracle_top_1",), 1.0),)

        gemm = generate_gemm_source(4, panel, args)
        gesummv = generate_gesummv_source(4, panel, args)

        self.assertIn("double alpha, double beta, uint32_t n", gemm)
        self.assertIn("beta * c[c_index]", gemm)
        self.assertIn("double alpha, double beta, uint32_t n", gesummv)
        self.assertIn("y[i] = alpha * sum_a + beta * sum_b", gesummv)
        self.assertIn("(position + shift) % case_count", gemm)
        self.assertIn("launch_case(id, 0)", gemm)
        self.assertIn("buffers[buffer_slot].c", gemm)

    def test_analysis_uses_same_round_paired_ratios(self) -> None:
        panel = (
            Candidate("jjii", ("oracle_top_1",), 1.0),
            Candidate("iijj", ("analytical_frontier",), 1.1),
        )
        measurements = []
        for round_index, scale in enumerate((1.0, 1.2, 0.8)):
            measurements.extend(
                (
                    {
                        "word": "jjii",
                        "round": round_index,
                        "time_ms": scale,
                    },
                    {
                        "word": "iijj",
                        "round": round_index,
                        "time_ms": 1.1 * scale,
                    },
                )
            )

        analysis = analyze_measurements(panel, measurements, seed=3)

        self.assertEqual(analysis["confirmed_oracle_word"], "jjii")
        paired = analysis["frontier_paired_vs_confirmed_oracle"]
        self.assertAlmostEqual(paired["mean_regret"], 0.1)
        self.assertEqual(paired["round_count"], 3)


if __name__ == "__main__":
    unittest.main()
