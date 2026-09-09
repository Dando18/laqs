"""Numerical and compiler checks for structured packet addressing on a GPU."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton/packet_layout'))
import _bootstrap
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires the platform Triton environment and one GPU')
class PacketLayoutGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _bootstrap.activate('tuolumne' if torch.version.hip else 'matrix')

    def test_packet_vector_copy(self):
        import triton
        import triton.language as tl
        from layout_contract import RuntimeLayout
        from layout_runtime import pack_tensor
        from packet_runtime import structured_layouts, statistics, primitive_rejections

        @triton.jit
        def kernel(A, O, N: tl.constexpr):
            i = tl.arange(0, 32)
            j = tl.arange(0, 32)
            values = tl.load(A + i[:, None] * N + j[None, :])
            tl.store(O + i[:, None] * N + j[None, :], values)

        source = torch.arange(1024, device='cuda', dtype=torch.float32).reshape(32, 32)
        ordinary = torch.empty_like(source)
        output = torch.empty_like(source)
        rows = (32, 64, 1, 2, 4, 8, 16, 128, 256, 512)
        layout = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32), rows)
        packed = pack_tensor(source, layout)
        with structured_layouts((layout,), inspect=True):
            baseline = kernel[(1,)](source, ordinary, 32, num_warps=4)
        with structured_layouts((layout,)):
            selected = kernel[(1,)](packed, output, 32, num_warps=4)
        torch.testing.assert_close(output, source, rtol=0, atol=0)
        self.assertNotIn('tt.ptr_to_int', selected.asm['ttgir'])
        with tempfile.TemporaryDirectory() as path:
            before = statistics(baseline, path, 'baseline')
            after = statistics(selected, path, 'selected')
            self.assertEqual(before['packet_sites'], after['packet_sites'])
            self.assertEqual(before['memory_primitives'], after['memory_primitives'])

    def test_loop_carried_offsets_and_masks(self):
        import triton
        import triton.language as tl
        from layout_contract import RuntimeLayout
        from layout_runtime import pack_tensor
        from packet_runtime import structured_layouts

        @triton.jit
        def kernel(A, O, count: tl.constexpr):
            x = tl.arange(0, 64)
            pointer = A + x
            for k in range(0, 16):
                value = tl.load(pointer, x + k * 64 < count, 0)
                tl.store(O + x + k * 64, value, x + k * 64 < count)
                pointer += 64

        source = torch.arange(1024, device='cuda', dtype=torch.float32).reshape(32, 32)
        output = torch.full((1024,), -1., device='cuda')
        rows = (32, 64, 1, 2, 4, 8, 16, 128, 256, 512)
        layout = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32), rows)
        packed = pack_tensor(source, layout)
        with structured_layouts((layout,)):
            selected = kernel[(1,)](packed, output, 1003, num_warps=4)
        torch.testing.assert_close(output[:1003], source.flatten()[:1003], rtol=0, atol=0)
        self.assertTrue(bool((output[1003:] == -1).all()))
        self.assertNotIn('tt.ptr_to_int', selected.asm['ttgir'])

    def test_carrying_integer_addition_is_not_distributed(self):
        import triton
        import triton.language as tl
        from layout_contract import RuntimeLayout
        from layout_runtime import pack_tensor
        from packet_runtime import structured_layouts

        @triton.jit
        def kernel(A, O, delta):
            x = tl.arange(0, 512)
            z = x + delta
            tl.store(O + x, tl.load(A + z))

        source = torch.arange(1024, device='cuda', dtype=torch.float32).reshape(32, 32)
        output = torch.empty((512,), device='cuda')
        layout = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32),
                               (32, 64, 1, 2, 4, 8, 16, 128, 256, 512))
        packed = pack_tensor(source, layout)
        for delta in (3, 31, 127, 255):
            with structured_layouts((layout,)):
                kernel[(1,)](packed, output, delta, num_warps=4)
            torch.testing.assert_close(output, source.flatten()[delta:delta + 512], rtol=0, atol=0)

    def test_transformed_atomic_storage_is_rejected(self):
        import triton
        import triton.language as tl
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts

        @triton.jit
        def kernel(A):
            x = tl.arange(0, 32)
            tl.atomic_add(A + x, 1.)

        source = torch.zeros((32, 32), device='cuda')
        layout = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32),
                               (32, 64, 1, 2, 4, 8, 16, 128, 256, 512))
        with self.assertRaises(Exception), structured_layouts((layout,)):
            kernel[(1,)](source)
        self.assertEqual(source.count_nonzero().item(), 0)

    def test_conventional_row_column_storage(self):
        from layout_runtime import freeze_launch, replace_inputs
        from packet_cases import row_column, reference
        from packet_runtime import runtime_layouts
        from conventional import storage_candidates, SCHEDULES
        spec = row_column(128, 'test')
        ordinary = freeze_launch(spec, SCHEDULES[0])
        ordinary.run()
        reference('row_column', ordinary)
        expected = [ordinary.values[i].clone() for i in (3, 4)]
        for storage in storage_candidates(128, {'candidates': []})[1:]:
            for config in SCHEDULES:
                launch = replace_inputs(freeze_launch(spec, {**config, 'STORAGE': storage['storage']}), runtime_layouts(storage))
                launch.run()
                for observed, correct in zip((launch.values[3], launch.values[4]), expected):
                    torch.testing.assert_close(observed, correct, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(torch is not None and torch.version.hip, 'MI300A FP8 compiler regression')
    def test_fp8_packets_survive_gemm_pipeline_and_buffer_conversion(self):
        from tritonbench_cases import CASE_BY_ID
        from layout_runtime import freeze_launch, replace_inputs
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts, statistics, primitive_rejections
        config = {'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                  'GROUP_SIZE_M': 1, 'kpack': 1, 'matrix_instr_nonkdim': 0,
                  'num_ctas': 1, 'num_stages': 2, 'num_warps': 4, 'waves_per_eu': 8}
        native = freeze_launch(CASE_BY_ID['fp8_gemm--small'].factory(), config)
        layouts = tuple(RuntimeLayout(name, index, (1024, 1024), (1024, 1), (1024, 1024),
                       tuple(1 << bit for bit in (*range(10, 10 + v), *range(10), *range(10 + v, 20))))
                        for index, (name, v) in enumerate((('a_ptr', 3), ('b_ptr', 4))))
        packed = replace_inputs(native, layouts)
        with structured_layouts(layouts, inspect=True):
            baseline = native.run()
        expected = native.values[2].clone()
        with structured_layouts(layouts):
            selected = packed.run()
        torch.testing.assert_close(packed.values[2], expected)
        with tempfile.TemporaryDirectory() as directory:
            before = statistics(baseline, directory, 'native')
            after = statistics(selected, directory, 'structured')
        self.assertEqual(primitive_rejections(before, after), [])
        self.assertIn('buffer_load_dwordx2', after['memory_primitives'])
        self.assertNotIn('buffer_load_ubyte', after['memory_primitives'])
        self.assertEqual(after['n_spills'], 0)


@unittest.skipUnless(torch is not None, 'requires a platform Triton environment')
class PacketLayoutNvidiaCompileTests(unittest.TestCase):
    def test_h100_vector_packets_compile_without_a_gpu(self):
        _bootstrap.activate('tuolumne' if torch.version.hip else 'matrix')
        import triton
        import triton.language as tl
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from packet_runtime import structured_layouts
        from layout_contract import RuntimeLayout

        @triton.jit
        def copy(A, O, N: tl.constexpr):
            i = tl.arange(0, 32)
            j = tl.arange(0, 32)
            tl.store(O + i[:, None] * N + j[None, :], tl.load(A + i[:, None] * N + j[None, :]))

        source = ASTSource(copy, {'A': '*fp32', 'O': '*fp32', 'N': 'constexpr'}, constexprs={'N': 32},
                           attrs={(0,): [('tt.divisibility', 16)], (1,): [('tt.divisibility', 16)]})
        layout = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32),
                               (32, 64, 1, 2, 4, 8, 16, 128, 256, 512))
        for inspect in (True, False):
            with structured_layouts((layout,), inspect=inspect):
                kernel = triton.compile(source, target=GPUTarget('cuda', 90, 32), options={'num_warps': 4})
            self.assertIn('ld.global.v4.b32', kernel.asm['ptx'])
            self.assertTrue(kernel.asm['cubin'])
            self.assertNotIn('tt.ptr_to_int', kernel.asm['ttgir'])
