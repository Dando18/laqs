from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "triton" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from tritonbench_cases import CASES, OPERATORS
from search_algorithms import _word_count, natural_tile_exponents
from relay import Access, MatrixSpec, MemoryEvent


class TritonBenchSearchExperimentTests(unittest.TestCase):
    def test_broad_portable_panel_has_declared_breadth(self):
        self.assertEqual(len(CASES), 29)
        self.assertEqual(len(OPERATORS), 15)
        self.assertEqual(len({case.case_id for case in CASES}), 29)
        self.assertEqual(
            {case.operator for case in CASES},
            {
                "vector_add", "vector_exp", "low_mem_dropout", "softmax",
                "sum", "layer_norm", "gemm", "bf16xint16_gemm", "int4_gemm",
                "fp8_gemm", "gather_gemv", "template_attention", "jagged_sum",
                "jagged_mean", "jagged_softmax",
            },
        )

    def test_natural_tile_is_power_of_two_bounding_footprint(self):
        matrix = MatrixSpec("x", (64, 128), 4, ("i", "j"), role="read")
        event = MemoryEvent.make(
            "load", "load",
            [Access("x", (i, j)) for i in range(3, 8) for j in range(9, 42)],
        )
        self.assertEqual(natural_tile_exponents(matrix, (event,)), (3, 6))
        self.assertEqual(_word_count((3, 6)), 84)

    def test_legacy_entry_point_remains_limited_to_experiments_one_to_three(self):
        source = (EXPERIMENTS / "run.py").read_text(encoding="utf-8")
        self.assertIn("choices=(1, 2, 3)", source)
        for experiment in (4, 5, 6):
            for platform in ("tuolumne", "matrix"):
                self.assertTrue(
                    (EXPERIMENTS / f"submit-experiment-{experiment}-{platform}.bash").is_file()
                )


if __name__ == "__main__":
    unittest.main()
