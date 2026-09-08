"""GPU regressions for physical packing and the post-coalescing rewrite.

Run in the platform Triton environment inside a one-GPU debug allocation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires a GPU and the platform Triton environment")
class TritonLayoutRewriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "triton/experiments/run-search.py"
        spec = importlib.util.spec_from_file_location("rewrite_test_runner", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner._activate_triton_source("tuolumne" if torch.version.hip else "matrix")

    def _check(self, kind):
        from layout_runtime import (RuntimeLayout, freeze_launch, fresh_outputs,
                                    replace_inputs, rewrite_layouts)
        from layout_contract import compiler_statistics, realization_rejections
        from tritonbench_cases import _dropout, _vector_add
        from relay import MatrixSpec, layout_matrix_rows, row_major_layout

        spec = (_dropout(2**18, "test") if kind == "dropout" else
                _vector_add(80 if kind == "tail" else 2**18, "test"))
        launch = freeze_launch(spec, {})
        argument = 1 if kind == "dropout" else 0
        source = launch.values[argument]
        if kind == "tail":
            source = source.reshape(8, 10)
            launch.values[argument] = source
        shape = tuple(source.shape)
        envelope = tuple(1 << (n - 1).bit_length() for n in shape)
        matrix = MatrixSpec("input", envelope, source.element_size(),
                            tuple(f"d{i}" for i in range(len(envelope))))
        rows = list(layout_matrix_rows(matrix, row_major_layout(matrix)))
        left, right = (0, 1) if kind == "tail" else (1, 3)
        rows[left], rows[right] = rows[right], rows[left]
        layout = RuntimeLayout("input", argument, shape, tuple(source.stride()),
                               envelope, tuple(rows))
        baseline = fresh_outputs(launch, [2])
        selected = fresh_outputs(replace_inputs(launch, (layout,)), [2])
        ordinary_kernel = baseline.run()
        with rewrite_layouts((layout,)):
            selected_kernel = selected.run()
        torch.cuda.synchronize()
        torch.testing.assert_close(selected.values[2], baseline.values[2], rtol=0, atol=0)
        if kind == "vector":
            with tempfile.TemporaryDirectory(prefix="relay-rewrite-test-") as directory:
                ordinary = compiler_statistics(ordinary_kernel, directory, "ordinary")
                rewritten = compiler_statistics(selected_kernel, directory, "rewritten")
                self.assertTrue(realization_rejections(ordinary, rewritten))

    def test_scalarizing_layout_is_correct_but_rejected(self):
        self._check("vector")

    def test_boolean_pointer_bitcast_preserves_storage_width(self):
        self._check("dropout")

    def test_nonpower_of_two_packing_uses_correct_narrowed_offsets(self):
        self._check("tail")


if __name__ == "__main__":
    unittest.main()
