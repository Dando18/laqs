from dataclasses import replace
from itertools import product
from pathlib import Path
import math
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton/packet_layout'))
import _bootstrap
from layout_contract import RuntimeLayout
from manual_layout import split_tile
from packet_search import split_templates
from relay import MatrixSpec, layout_matrix_rows


def mapping(shape, a, v, argument=0):
    matrix = MatrixSpec('A', shape, 4, ('i', 'j'))
    n, m = matrix.mode_bits
    bits = (*range(n, n + v), *range(a), *range(n + v, n + m), *range(a, n))
    return RuntimeLayout('A', argument, shape, (shape[1], 1), shape, tuple(1 << b for b in bits))


class ManualLayoutTests(unittest.TestCase):
    def test_every_small_split_map_matches_blocked_indexing(self):
        for shape in ((2, 4), (8, 16), (32, 8)):
            matrix = MatrixSpec('A', shape, 4, ('i', 'j'))
            for layout in split_templates(matrix, 0):
                runtime = RuntimeLayout('name-does-not-describe-the-tile', 0, shape, (shape[1], 1),
                                        shape, layout_matrix_rows(matrix, layout))
                ti, tj = split_tile(runtime)
                for i, j in product(range(shape[0]), range(shape[1])):
                    offset = (i // ti) * shape[1] * ti + (j // tj) * ti * tj + (i % ti) * tj + j % tj
                    self.assertEqual(offset, layout.offset(matrix, (i, j)))

    def test_non_split_or_non_dense_map_is_rejected(self):
        layout = mapping((8, 8), 1, 1)
        for invalid in (replace(layout, rows=(8, 1, 16, 2, 32, 4)),
                        replace(layout, rows=(8, 2, 1, 16, 32, 4)),
                        replace(layout, rows=(8, 1, 17, 2, 32, 4)),
                        replace(layout, strides=(1, 8)),
                        replace(layout, shape=(7, 8))):
            with self.assertRaises(ValueError):
                split_tile(invalid)

    def test_matched_summary_preserves_placement_pairing_and_process_unit(self):
        from packet_measure import matched_summary, sample_record
        records = []
        for values in (((1., 2.), (100., 100.)), ((4., 2.), (100., 100.))):
            placements = []
            for base, candidate in values:
                placements.append({'timings': {name: sample_record([value] * 3) for name, value in
                    [('ordinary', base), ('candidate', candidate), ('identity', base), ('same_pointer', base)]}})
            records.append({'allocation': {'placements': placements}})
        result = matched_summary(records, 'ordinary', 'candidate')
        self.assertAlmostEqual(result['speedup'], 1.)
        self.assertAlmostEqual(result['process_speedups'][0], math.sqrt(.5))
        self.assertAlmostEqual(result['process_speedups'][1], math.sqrt(2))
        self.assertEqual(result['placement_speedups'], [[.5, 1.], [2., 1.]])
        self.assertIsNotNone(result['speedup_ci95'])
        self.assertEqual(result['same_pointer_max_deviation'], 0.)
        self.assertIsNone(matched_summary(records[:1], 'ordinary', 'candidate')['speedup_ci95'])

    def test_manual_runner_is_one_resumable_stage_sequence(self):
        import debug_suite
        args = debug_suite.parser().parse_args(['--platform', 'tuolumne', '--manual', '--cases', 'row_column--small'])
        self.assertEqual(debug_suite.stages(args), ('capture', 'search', 'conventional'))
        self.assertEqual(debug_suite.command(args, args.cases[0], 'search')[0], str(ROOT / '.venv/bin/python'))

    def test_ir_text_differences_do_not_claim_ownership_changes(self):
        from manual_layout import execution_comparison
        auto = {'execution_encodings': ['#blocked1', '#blocked1'], 'packet_sites': [{'ownership': 'same'}]}
        explicit = {'execution_encodings': ['#blocked2'], 'inspected_packet_sites': [{'ownership': 'same'}]}
        result = execution_comparison(auto, explicit)
        self.assertEqual(result['changed_fields'], [])
        self.assertTrue(result['same_lowered_evidence'])
        self.assertTrue(result['raw_ir_encoding_text_changed'])
        missing = execution_comparison(auto, {})
        self.assertFalse(missing['ownership_comparable'])
        self.assertFalse(missing['same_lowered_evidence'])

    def test_manual_timing_resume_retains_completed_processes(self):
        import subprocess
        import tempfile
        from unittest.mock import patch
        import packet_workflow as workflow
        with tempfile.TemporaryDirectory() as directory:
            args = workflow.args_parser(['--stage', 'conventional', '--platform', 'tuolumne',
                '--case', 'row_column--small', '--root', directory, '--resume', '--processes', '2'])
            calls = []
            def execute(command, **kwargs):
                index = int(command[command.index('--process-index') + 1])
                calls.append(index)
                if calls == [0, 1]:
                    raise subprocess.CalledProcessError(1, command)
                workflow.write_json(Path(command[command.index('--worker-output') + 1]), {'index': index})
            with patch.object(workflow, 'verify'), \
                    patch.object(workflow, 'read', return_value={'selection_hash': 'frozen'}), \
                    patch.object(workflow.subprocess, 'run', side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError):
                    workflow.run_processes(args, 'conventional-fixed')
                workflow.run_processes(args, 'conventional-fixed')
                workflow.run_processes(args, 'conventional-fixed')
                self.assertEqual(calls, [0, 1, 1])
                args.samples += 1
                workflow.run_processes(args, 'conventional-fixed')
                self.assertEqual(calls, [0, 1, 1, 0, 1])


try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False


@unittest.skipUnless(GPU, 'requires a GPU allocation and platform Triton environment')
class ManualGPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _bootstrap.activate('tuolumne' if torch.version.hip else 'matrix')

    def test_packing_and_manual_kernels_match_complete_outputs(self):
        from conventional import pack_inputs
        from layout_runtime import freeze_launch, fresh_outputs, pack_tensor
        from manual_layout import explicit_spec, pack_split, unpack_split
        from packet_cases import row_column, reference
        from tritonbench_cases import _sum, _gemm
        cases = [
            (row_column(128, 'test'), {'BLOCK_M': 4, 'BLOCK_K': 64, 'num_warps': 4}, (3, 4), [(128, 128)]),
            (_sum(128, 64, 'test'), {'BLOCK_SIZE_NON_REDUCE_DIM': 16, 'BLOCK_SIZE_REDUCE_DIM': 16,
                                   'num_warps': 4}, (1,), [(128, 64)]),
            (_gemm(128, 64, 64, 'test'), {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 1,
                                         'num_warps': 4, 'num_stages': 2}, (2,), [(128, 64), (64, 64)]),
        ]
        for spec, config, outputs, shapes in cases:
            native = freeze_launch(spec, config)
            native.run()
            reference(spec.operator, native)
            for a, v in ((0, 0), (1, 2), (4, 5)):
                layouts = [mapping(shape, a, v, argument) for argument, shape in enumerate(shapes)]
                for layout in layouts:
                    tensor = native.values[layout.argument]
                    packed = pack_split(tensor, split_tile(layout))
                    self.assertTrue(torch.equal(packed, pack_tensor(tensor, layout)))
                    self.assertTrue(torch.equal(tensor, unpack_split(packed, layout.shape, split_tile(layout))))
                manual_spec, parameters = explicit_spec(spec, layouts)
                manual = fresh_outputs(freeze_launch(manual_spec, {**config, **parameters}), outputs)
                packed = pack_inputs(native, layouts)
                for layout in layouts:
                    manual.values[layout.argument] = packed.values[layout.argument]
                manual.run()
                checked = manual.clone()
                for layout in layouts:
                    checked.values[layout.argument] = native.values[layout.argument]
                reference(spec.operator, checked)
                for index in outputs:
                    torch.testing.assert_close(manual.values[index], native.values[index], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
