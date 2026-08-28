from __future__ import annotations

from pathlib import Path
import sys
import unittest

from relay import MI300A_V1, MatrixSpec, TritonLinearLayout


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from stage2_probe import (
    aggregate_probe_results,
    analyze_fiber_candidates,
    build_gemm_b_resource_groups,
)


class Stage2ResourceCohortTests(unittest.TestCase):
    def test_gemm_groups_preserve_dynamic_tile_and_program_weights(self) -> None:
        matrix = MatrixSpec("B", (64, 64), 2, ("k", "n"))
        execution = TritonLinearLayout.from_blocked(
            (32, 32),
            size_per_thread=(1, 4),
            threads_per_warp=(8, 8),
            warps_per_cta=(4, 1),
            order=(1, 0),
            output_dim_names=("k", "n"),
        )
        groups = build_gemm_b_resource_groups(
            matrix,
            execution,
            m=64,
            n=64,
            k=64,
            block_m=32,
            block_n=32,
            block_k=32,
            resource_map=MI300A_V1.resource_maps[0],
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].relative_bits), 256)
        self.assertEqual(len(groups[0].occurrences), 16)
        self.assertEqual(
            sum(occurrence.weight for occurrence in groups[0].occurrences),
            32,
        )


class Stage2ProbeAnalysisTests(unittest.TestCase):
    @staticmethod
    def candidate(
        candidate_id: str,
        runtime_ms: float,
        service: float,
        shears,
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "flag_id": "flag",
            "quotient_score": 8.0,
            "resource_service_score": service,
            "runtime_ms": runtime_ms,
            "shears": shears,
            "codegen_cost": {
                "runs": 2,
                "xors": len(shears),
                "swizzle_xors": len(shears),
            },
        }

    def test_analysis_applies_the_controlled_gate(self) -> None:
        analysis = analyze_fiber_candidates(
            [
                self.candidate("identity", 12.0, 2.0, []),
                self.candidate("best", 10.0, 1.0, [[1, 3]]),
                self.candidate("peer", 11.0, 1.0, [[1, 4]]),
            ]
        )

        self.assertTrue(analysis["quotient_invariant"])
        self.assertAlmostEqual(analysis["runtime_relative_spread"], 0.2)
        self.assertEqual(analysis["service_optimal_set_regret"], 0.0)
        self.assertTrue(analysis["gate"]["develop_stage_2"])

    def test_aggregation_uses_median_process_runtime(self) -> None:
        def result(identity_runtime, best_runtime, process):
            candidates = [
                self.candidate("identity", identity_runtime, 2.0, []),
                self.candidate("best", best_runtime, 1.0, [[1, 3]]),
            ]
            for candidate in candidates:
                candidate["timing"] = {
                    "median_ms": candidate["runtime_ms"],
                    "samples_ms": [candidate["runtime_ms"]],
                }
                candidate["compiled_codegen"] = {"n_regs": 8}
            return {
                "stage": 2,
                "experiment": "probe",
                "candidates": candidates,
                "analysis": {},
                "process": {"pid": process},
                "correct": True,
            }

        aggregate = aggregate_probe_results(
            [result(12.0, 10.0, 1), result(13.0, 9.0, 2), result(11.0, 8.0, 3)]
        )

        self.assertEqual(aggregate["process_launch_count"], 3)
        self.assertEqual(aggregate["candidates"][0]["runtime_ms"], 12.0)
        self.assertEqual(aggregate["candidates"][1]["runtime_ms"], 9.0)
        self.assertTrue(aggregate["analysis"]["gate"]["develop_stage_2"])


if __name__ == "__main__":
    unittest.main()
