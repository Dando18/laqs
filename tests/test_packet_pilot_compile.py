"""Compile every packet pilot on both backends without reserving a GPU."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton/experiments'))
sys.path.insert(0, str(ROOT / 'triton/packet_layout'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'requires the platform Triton environment')
class PacketPilotCompileTests(unittest.TestCase):
    def test_all_pilots_keep_packet_contract_on_both_targets(self):
        import _bootstrap
        _bootstrap.activate('tuolumne' if torch.version.hip else 'matrix')
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts
        import packet_pilot_kernels as kernels

        ordinary = (32, 64, 128, 256, 512, 1, 2, 4, 8, 16)
        selected = (32, 64, 1, 2, 4, 8, 16, 128, 256, 512)
        constants = dict(A_ROWS=ordinary, B_ROWS=ordinary, W_ROWS=ordinary,
                         ROW_BITS=5, MODE_BITS=5, M=32, N=32, K=32, D=32,
                         BAG_SIZE=4, BLOCK=32, ALPHA=1.5, BETA=1.2)
        for backend, architecture, warp in [('cuda', 90, 32), ('hip', 'gfx942', 64)]:
            for case in ('softmax_bias', 'embedding_bag', 'gemv', 'mvt', 'gesummv', 'stencil5'):
                with self.subTest(backend=backend, case=case):
                    kernel = getattr(kernels, case + '_kernel')
                    signature = {name: ('constexpr' if name in constants else '*i32' if name == 'indices' else '*fp32')
                                 for name in kernel.arg_names}
                    source = ASTSource(kernel, signature,
                                       constexprs={k: v for k, v in constants.items() if k in signature},
                                       attrs={(i,): [('tt.divisibility', 16)] for i, name in enumerate(kernel.arg_names)
                                              if name not in constants})
                    argument = 1 if case == 'softmax_bias' else 0
                    layout = RuntimeLayout('target', argument, (32, 32), (32, 1), (32, 32), selected)
                    contracts = []
                    for inspect in (True, False):
                        with structured_layouts((layout,), inspect=inspect):
                            compiled = triton.compile(source, target=GPUTarget(backend, architecture, warp),
                                                      options={'num_warps': 4})
                        contracts.append(json.loads(compiled.metadata.laqs_packet_contracts))
                        self.assertTrue(compiled.asm['cubin' if backend == 'cuda' else 'hsaco'])
                        self.assertNotIn('tt.ptr_to_int', compiled.asm['ttgir'])
                    self.assertTrue(contracts[0])
                    self.assertEqual(contracts[0], contracts[1])


if __name__ == '__main__':
    unittest.main()
