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

        from contextlib import nullcontext
        from layout_runtime import FrozenLaunch, fresh_outputs
        from packet_measure import graph_samples
        identity = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32),
                                 (32, 64, 128, 256, 512, 1, 2, 4, 8, 16))
        native = FrozenLaunch(kernel, (1,), [source, ordinary, 32], {'num_warps': 4})
        launches, contexts = {'ordinary': native}, {'ordinary': nullcontext}
        labels = ('identity_legacy', 'identity_repaired')
        for label, mode in zip(labels, ('legacy', 'repaired')):
            launch = fresh_outputs(native, [1])
            launches[label] = launch
            contexts[label] = lambda launch=launch, mode=mode: structured_layouts((identity,), launch=launch, mode=mode)
        captured = {}
        graph_samples(launches, contexts, [1], process_index=1, samples=2, iterations=2, warmup=1,
                      same_pointer_labels=labels, on_capture=lambda label, compiled: captured.update({label: compiled}))
        for label in labels:
            self.assertEqual(launches[label].values[0].data_ptr(), native.values[0].data_ptr())
            self.assertEqual(launches[label].values[1].data_ptr(), native.values[1].data_ptr())
            self.assertNotEqual(captured[label].metadata.laqs_packet_contracts, '[]')
        torch.testing.assert_close(native.values[1], source, rtol=0, atol=0)

        from packet_measure import allocation_samples
        native = FrozenLaunch(kernel, (1,), [source, torch.empty_like(source), 32], {'num_warps': 4})
        selected = FrozenLaunch(kernel, (1,), [packed, torch.empty_like(source), 32], {'num_warps': 4})
        checked = []
        def check(label, launch):
            torch.testing.assert_close(launch.values[1], source, rtol=0, atol=0)
            checked.append(label)
        measured = allocation_samples({'ordinary': native, 'selected': selected},
            {'ordinary': nullcontext, 'selected': lambda: structured_layouts((layout,), launch=selected)},
            [0], [1], process_index=1, samples=2, iterations=2, warmup=1, placements=2, on_ready=check)
        self.assertEqual(len(checked), 8)
        addresses = []
        for placement in measured['allocation']['placements']:
            self.assertEqual(len({row['0'] for row in placement['addresses'].values()}), 1)
            addresses.append(placement['addresses']['ordinary']['0'])
            self.assertEqual(placement['addresses']['ordinary'], placement['addresses']['same_pointer'])
        self.assertEqual(len(set(addresses)), 2)
        self.assertEqual(len(measured['timings']['selected']['samples_ms']), 4)

    def test_real_sum_affine_recurrence_preserves_values(self):
        from layout_runtime import freeze_launch, replace_inputs
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts
        from packet_search import split_templates
        from relay import MatrixSpec, layout_matrix_rows
        from tritonbench_cases import CASE_BY_ID
        config = dict(BLOCK_SIZE_NON_REDUCE_DIM=16, BLOCK_SIZE_REDUCE_DIM=16,
                      num_warps=2, num_stages=3)
        native = freeze_launch(CASE_BY_ID['sum--small'].factory(), config)
        native.run()
        expected = native.values[1].clone()
        matrix = MatrixSpec('input_ptr', (4096, 1024), 4, ('i', 'k'))
        templates = {c.name: c for c in split_templates(matrix, 1)}
        for name in ('split-a12-v2', 'split-a1-v4'):
            layout = RuntimeLayout('input_ptr', 0, (4096, 1024), (1024, 1), (4096, 1024),
                                   layout_matrix_rows(matrix, templates[name]))
            launch = replace_inputs(native, (layout,))
            with structured_layouts((layout,), launch=launch):
                kernel = launch.run()
            self.assertIn('laqs.physical_recurrences = 1', kernel.asm['ttgir'])
            torch.testing.assert_close(launch.values[1], expected)

    def test_affine_carry_boundary_and_nonlinear_offsets_preserve_values(self):
        import triton
        import triton.language as tl
        from layout_runtime import FrozenLaunch, pack_tensor
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts

        @triton.jit
        def kernel(A, O, count, NONLINEAR: tl.constexpr):
            x = tl.arange(0, 128)
            acc = tl.full((128,), 0, tl.float32)
            for k in range(count):
                delta = k * k * 16 if NONLINEAR else k * 16
                acc += tl.load(A + x + delta)
            tl.store(O + x, acc)

        source = torch.arange(1024, device='cuda', dtype=torch.float32).reshape(32, 32)
        layout = RuntimeLayout('A', 0, (32, 32), (32, 1), (32, 32),
                               (32, 64, 1, 2, 4, 8, 16, 128, 256, 512))
        packed = pack_tensor(source, layout)
        for nonlinear in (False, True):
            for count in (0, 1, 4):
                output = torch.empty(128, device='cuda')
                launch = FrozenLaunch(kernel, (1,), [packed, output, count, nonlinear], {'num_warps': 4})
                with structured_layouts((layout,), launch=launch):
                    compiled = launch.run()
                expected = torch.zeros_like(output)
                for k in range(count):
                    delta = (k * k if nonlinear else k) * 16
                    expected += source.flatten()[delta:delta + 128]
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                if count == 4:
                    self.assertIn('laqs.physical_recurrences = 0', compiled.asm['ttgir'])

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
        with structured_layouts(layouts, inspect=True, launch=native):
            baseline = native.run()
        expected = native.values[2].clone()
        with structured_layouts(layouts, launch=packed):
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
    @classmethod
    def setUpClass(cls):
        _bootstrap.activate('tuolumne' if torch.version.hip else 'matrix')

    def test_frozen_launch_rejects_changed_proof_assumptions(self):
        from layout_runtime import FrozenLaunch
        from packet_runtime import structured_layouts
        from types import SimpleNamespace
        jit = SimpleNamespace(arg_names=['K'], run=lambda *args, **kwargs: 'ran')
        launch = FrozenLaunch(jit, (1,), [128], {})
        with structured_layouts((), launch=launch):
            self.assertEqual(launch.run(), 'ran')
            launch.values[0] = 256
            with self.assertRaisesRegex(ValueError, 'scalar arguments'):
                launch.run()
            launch.values[0] = 128
            launch.grid = (2,)
            with self.assertRaisesRegex(ValueError, 'launch does not match'):
                launch.run()
        self.assertEqual(launch.run(), 'ran')

    def test_sum_induction_offsets_become_physical_recurrences(self):
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from types import SimpleNamespace
        from tritonbench.operators.sum.kernels import triton_sum_kernel_1D_result_sum_then_buffer
        from layout_runtime import unwrap_jit
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts
        from packet_search import split_templates
        from relay import MatrixSpec, layout_matrix_rows

        jit = unwrap_jit(triton_sum_kernel_1D_result_sum_then_buffer)
        bound = dict(input_ptr=None, output_ptr=None, M=4096, N=1024,
                     BLOCK_SIZE_NON_REDUCE_DIM=16, BLOCK_SIZE_REDUCE_DIM=16, dim=1)
        constants = {p.name: bound[p.name] for p in jit.params if p.is_constexpr}
        source = ASTSource(jit, {p.name: 'constexpr' if p.is_constexpr else '*fp32' if p.num < 2 else 'i32'
                                for p in jit.params}, constexprs=constants,
                           attrs={(p.num,): [('tt.divisibility', 16)] for p in jit.params if not p.is_constexpr})
        launch = SimpleNamespace(jit=jit, values=[bound[n] for n in jit.arg_names], grid=(256,))
        matrix = MatrixSpec('input_ptr', (4096, 1024), 4, ('i', 'k'))
        templates = {c.name: c for c in split_templates(matrix, 1)}
        for target in (GPUTarget('cuda', 90, 32), GPUTarget('hip', 'gfx942', 64)):
            for template in ('split-a12-v2', 'split-a1-v4'):
                layout = RuntimeLayout('input_ptr', 0, (4096, 1024), (1024, 1), (4096, 1024),
                                       layout_matrix_rows(matrix, templates[template]))
                for mode in ('legacy', 'repaired'):
                    with structured_layouts((layout,), launch=launch, mode=mode):
                        kernel = triton.compile(source, target=target, options={'num_warps': 2, 'num_stages': 3})
                    self.assertIn(f'laqs.physical_recurrences = {int(mode == "repaired")}', kernel.asm['ttgir'])
                    body = kernel.asm['ttgir'].split('scf.for', 1)[1].split('scf.yield', 1)[0]
                    self.assertEqual('arith.shrui' in body, mode == 'legacy')
                    self.assertEqual('arith.shli' in body, mode == 'legacy')

    def test_frozen_asymmetric_gemm_removes_hot_loop_reconstruction(self):
        import collections
        from contextlib import nullcontext
        import re
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from types import SimpleNamespace
        from tritonbench.operators.gemm.triton_matmul import matmul_kernel
        from layout_runtime import unwrap_jit
        from layout_contract import RuntimeLayout
        from packet_runtime import structured_layouts
        from packet_search import split_templates
        from relay import MatrixSpec, layout_matrix_rows, row_major_layout
        jit = unwrap_jit(matmul_kernel)
        bound = dict(a_ptr=None, b_ptr=None, c_ptr=None, M=512, N=4096, K=4096,
            stride_am=4096, stride_ak=1, stride_bk=4096, stride_bn=1, stride_cm=4096, stride_cn=1,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, ACTIVATION='', ENABLE_BUFFER_OPS_ASSUMES=True)
        constants = {p.name: bound[p.name] for p in jit.params if p.is_constexpr or bound[p.name] == 1}
        signature = {p.name: 'constexpr' if p.name in constants else '*fp16' if p.num < 3 else 'i32'
                     for p in jit.params}
        attrs = {(p.num,): [('tt.divisibility', 16)] for p in jit.params if p.name not in constants}
        source = ASTSource(jit, signature, constexprs=constants, attrs=attrs)
        launch = SimpleNamespace(jit=jit, values=[bound[n] for n in jit.arg_names], grid=(256,))
        matrix = MatrixSpec('a_ptr', (512, 4096), 2, ('i', 'k'))
        templates = {c.name: c for c in split_templates(matrix, 3)}
        ordinary = row_major_layout(matrix)
        variants = [('ordinary', ordinary, 'repaired'), ('current', templates['split-a9-v3'], 'legacy'),
                    ('repaired', templates['split-a9-v3'], 'repaired'),
                    ('smaller', templates['split-a1-v5'], 'repaired'),
                    ('identity_legacy', ordinary, 'legacy'), ('identity_repaired', ordinary, 'repaired')]
        counts = {}
        for label, template, mode in variants:
            layout = RuntimeLayout('a_ptr', 0, (512, 4096), (4096, 1), (512, 4096),
                                   layout_matrix_rows(matrix, template))
            scope = nullcontext() if label == 'ordinary' else structured_layouts((layout,), mode=mode, launch=launch)
            with scope:
                kernel = triton.compile(source, target=GPUTarget('cuda', 90, 32),
                    options={'num_warps': 4, 'num_stages': 4, 'num_ctas': 1})
            lines = kernel.asm['ptx'].splitlines()
            start = next(i for i, line in enumerate(lines) if 'Inner Loop Header' in line)
            target = lines[start].split(':')[0]
            end = next(i for i in range(start + 1, len(lines)) if 'bra' in lines[i] and target in lines[i])
            pattern = r'(?:@!?%\w+\s+)?([a-z]\w*(?:\.\w+)*)\s'
            counts[label] = collections.Counter(m[1] for line in lines[start:end + 1]
                                                if (m := re.match(pattern, line.strip())))
            self.assertEqual('asm "", "=r,0"' in kernel.asm['llir'], mode == 'legacy')
        self.assertGreater(counts['current']['and.b32'], counts['ordinary']['and.b32'])
        for label in ('repaired', 'smaller', 'identity_repaired'):
            for op in ('and.b32', 'shr.u32', 'mad.wide.s32', 'cp.async.cg.shared.global'):
                self.assertEqual(counts[label][op], counts['ordinary'][op], (label, op))
            self.assertEqual({k: v for k, v in counts[label].items() if k.startswith('wgmma')},
                             {k: v for k, v in counts['ordinary'].items() if k.startswith('wgmma')})
        self.assertEqual(counts['identity_repaired'], counts['ordinary'])

    def test_physical_recurrence_and_transparent_carrier(self):
        _bootstrap.activate('tuolumne' if torch.version.hip else 'matrix')
        import triton
        import triton.language as tl
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from types import SimpleNamespace
        from packet_runtime import structured_layouts
        from layout_contract import RuntimeLayout
        import re

        @triton.jit
        def loop_copy(A, O, K, N: tl.constexpr):
            i = tl.arange(0, 8)
            j = tl.arange(0, 16)
            p = A + i[:, None] * N + j[None, :]
            acc = tl.full((8, 16), 0, tl.float32)
            for k in range(tl.cdiv(K, 16)):
                acc += tl.load(p, j[None, :] + k * 16 < K, 0)
                p += 16
            tl.store(O + i[:, None] * 16 + j[None, :], acc)

        source = ASTSource(loop_copy, {'A': '*fp32', 'O': '*fp32', 'K': 'i32', 'N': 'constexpr'},
                           constexprs={'N': 128}, attrs={(0,): [('tt.divisibility', 16)],
                                                        (1,): [('tt.divisibility', 16)]})
        layout = RuntimeLayout('A', 0, (8, 128), (128, 1), (8, 128),
                               (8, 16, 1, 2, 4, 32, 64, 128, 256, 512))
        launch = SimpleNamespace(jit=loop_copy, values=[None, None, 128, 128], grid=(1,))
        for target in (GPUTarget('cuda', 90, 32), GPUTarget('hip', 'gfx942', 64)):
            for mode in ('legacy', 'transparent', 'repaired'):
                with structured_layouts((layout,), mode=mode, launch=launch):
                    kernel = triton.compile(source, target=target, options={'num_warps': 4})
                opaque = re.findall(r'\basm(?:\s+\w+)*\s+""\s*,\s*"=[rv],0"', kernel.asm['llir'])
                self.assertEqual(bool(opaque), mode == 'legacy')
                if mode == 'repaired':
                    self.assertIn('laqs.physical_recurrences = 1', kernel.asm['ttgir'])
                    body = kernel.asm['ttgir'].split('scf.for', 1)[1].split('scf.yield', 1)[0]
                    self.assertNotIn('arith.andi', body)
                    self.assertNotIn('arith.shrui', body)
        launch.values[2] = 256
        with self.assertRaises(Exception), structured_layouts((layout,), launch=launch):
            triton.compile(source, target=target, options={'num_warps': 4})

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
