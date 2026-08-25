from __future__ import annotations

import unittest

from experiments.byte_scale_validation import parse_arguments, prepare_report
from kernels.atax import evaluate as atax_evaluate
from kernels.gesummv import evaluate as gesummv_evaluate
from relay.evaluator_dtype import EVALUATOR_DTYPES
from relay.evaluator_layout import canonical_layout


class EvaluatorDTypeTests(unittest.TestCase):
    def test_supported_types_have_distinct_physical_widths(self) -> None:
        self.assertEqual(
            {name: spec.element_bytes for name, spec in EVALUATOR_DTYPES.items()},
            {"fp64": 8, "fp32": 4, "fp16": 2},
        )

    def test_atax_source_uses_selected_storage_and_accumulator(self) -> None:
        layout = canonical_layout("jjjiii", "A")
        source = atax_evaluate.generate_source(
            layout,
            dtype="fp16",
            n=8,
            samples=1,
            iterations=1,
            warmup=0,
            device=0,
            block_size=8,
        )
        self.assertIn("using scalar_t = __half;", source)
        self.assertIn("using accum_t = float;", source)
        self.assertIn("ATAX FP16-storage/FP32-accumulation", source)

    def test_gesummv_source_preserves_fp64_default(self) -> None:
        layout = canonical_layout("jjjiii", "A")
        source = gesummv_evaluate.generate_source(
            (layout, layout),
            n=8,
            samples=1,
            iterations=1,
            warmup=0,
            device=0,
            block_size=8,
        )
        self.assertIn("using scalar_t = double;", source)
        self.assertIn("using accum_t = double;", source)
        self.assertIn("GESUMMV FP64", source)


class ByteScaleExperimentTests(unittest.TestCase):
    def test_prepare_keeps_edges_and_profile_fixed_across_types(self) -> None:
        _, args = parse_arguments(
            [
                "--size",
                "8",
                "--samples",
                "1",
                "--iterations",
                "1",
                "--warmup",
                "0",
                "--prepare-only",
            ]
        )
        report = prepare_report(
            args,
            ("atax", "gesummv"),
            ("fp64", "fp32", "fp16"),
            (8,),
        )
        self.assertEqual(len(report["groups"]), 6)
        for kernel in ("atax", "gesummv"):
            groups = [
                group for group in report["groups"] if group["kernel"] == kernel
            ]
            self.assertEqual(
                len({group["edge_geometry_sha256"] for group in groups}), 1
            )
            self.assertEqual(
                len({tuple(group["component_names"]) for group in groups}), 1
            )
            self.assertEqual(
                {group["element_bytes"] for group in groups}, {2, 4, 8}
            )
            ratios = {
                group["normalization_bytes"] / group["element_bytes"]
                for group in groups
            }
            self.assertEqual(len(ratios), 1)
            self.assertTrue(all(group["frontier"]["members"] for group in groups))
            self.assertTrue(
                all(
                    {"row_major", "column_major"}
                    <= {record["name"] for record in group["panel"]}
                    for group in groups
                )
            )


if __name__ == "__main__":
    unittest.main()
