from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton/packet_layout'))
import _bootstrap
import address_experiment as experiment
import debug_suite
from experiment_support import digest
from relay import MatrixSpec, row_major_layout, layout_matrix_rows


class AddressExperimentTests(unittest.TestCase):
    def test_copy_probe_startup_hashes_real_sources_before_building(self):
        import copy_service_probe as probe
        from debug_suite import file_hash
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with patch.object(probe.shutil, 'which', side_effect=lambda tool: '/usr/bin/' + tool), \
                    patch.object(probe.subprocess, 'check_output', return_value='test version'), \
                    patch.object(probe.signal, 'signal'), \
                    patch.object(probe, 'run_stage', return_value=None) as build:
                probe.run(SimpleNamespace(root=directory, minutes=1))
            build.assert_called_once()
            config = json.loads((directory / 'config.json').read_text())
            self.assertEqual(config['sources']['stage1_nvidia_counter_analysis.py'],
                             file_hash(ROOT / 'triton/stage1_nvidia_counter_analysis.py'))
            self.assertFalse((directory / 'build.complete.json').exists())

    def test_practical_choice_is_frozen_in_tuning_without_replacing_analytical_choice(self):
        import packet_workflow
        for gain, expected in ((.98, 'ordinary'), (1.15, 'smaller')):
            with self.subTest(gain=gain), tempfile.TemporaryDirectory() as name:
                args = SimpleNamespace(address_reference=Path('/tmp/reference'), directory=Path(name),
                    selection='analytical', minimum_gain=.01, processes=3, samples=21, iterations=50, warmup=10)
                def summary(speedup):
                    return {'speedup': speedup, 'speedup_ci95': [speedup - .01, speedup + .01],
                            'identity_max_deviation': 0., 'same_pointer_max_deviation': 0.}
                timing = {label: summary(speedup) for label, speedup in
                          [('ordinary', 1.), ('repaired', .5), ('smaller', gain), ('identity_repaired', 2.)]}
                with patch.object(packet_workflow, 'verify'), \
                        patch.object(packet_workflow, 'read', return_value={'analytical_top1': 'repaired', 'selection_hash': 's'}), \
                        patch.object(packet_workflow, 'accepted_candidates', return_value=[{'id': label} for label in timing]), \
                        patch.object(packet_workflow, 'run_processes', return_value=timing) as runs:
                    packet_workflow.tune(args)
                runs.assert_called_once_with(args, 'tune')
                choice = json.loads((args.directory / 'choice.json').read_text())
                self.assertEqual(choice['selected'], 'repaired')
                self.assertEqual(choice['deployment']['selected'], expected)
                self.assertEqual(choice['choice_hash'], digest({k: v for k, v in choice.items() if k != 'choice_hash'}))

    def test_comparison_pairs_processes_instead_of_dividing_aggregate_medians(self):
        values = {'current': {'selected': {'process_medians_ms': [2., 8., 8.]}},
                  'repaired': {'selected': {'process_medians_ms': [1., 4., 4.]}},
                  'smaller': {'selected': {'process_medians_ms': [2., 2., 2.]}}}
        result = experiment.summarize({'evaluation': {'candidates': values}})
        self.assertAlmostEqual(result['same_layout_repair_speedup'], 2.)
        self.assertAlmostEqual(result['smaller_vs_repaired_speedup'], 2 ** (1 / 3))

    def test_copy_probe_geometry_and_checkpoint_integrity(self):
        from copy_service_probe import checkpoint_valid, sector_witness, verify_instructions
        from debug_suite import file_hash
        witness = sector_witness()
        self.assertEqual(witness['large'], {'4': 32, '8': 16, '32': 16})
        self.assertEqual(witness['ordinary'], witness['smaller'])
        sass = '\n'.join(f'Function : _Z10copy_probeILi{layout}ELb{async_copy}EEvPKtPj\n' +
                         ('LDGSTS.E.BYPASS.128\n' * 2 if async_copy else 'LDG.E.128\nSTS.128\n' * 2)
                         for layout in range(3) for async_copy in (0, 1))
        self.assertEqual(len(verify_instructions(sass)), 6)
        with self.assertRaisesRegex(ValueError, 'unexpected vector'):
            verify_instructions(sass.replace('LDGSTS.E.BYPASS.128', 'LDGSTS.E.64', 1))
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            output, marker = directory / 'result.json', directory / 'complete.json'
            output.write_text('{}')
            self.assertFalse(checkpoint_valid(marker, 'b', [output]))
            marker.write_text(json.dumps({'binding': 'b', 'outputs': {output.name: file_hash(output)}}))
            self.assertTrue(checkpoint_valid(marker, 'b', [output]))
            output.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                checkpoint_valid(marker, 'b', [output])

    def test_panel_preserves_old_map_and_forces_both_identity_paths(self):
        matrix = MatrixSpec('A', (8, 16), 4, ('i', 'j'))
        identity = list(layout_matrix_rows(matrix, row_major_layout(matrix)))
        ordinary = {'id': 'ordinary', 'runtime_layouts': [], 'score': {'hardware_area': 2.}}
        old_map = {'name': 'A', 'rows': list(reversed(identity))}
        selected = {'id': 'split', 'runtime_layouts': [old_map], 'score': {'hardware_area': 1.}}
        small = {**selected, 'runtime_layouts': [{'name': 'A', 'rows': identity[1:] + identity[:1]}]}
        old = {'candidates': [ordinary, selected], 'analytical_top1': 'split', 'graph_hash': 'g',
               'hardware_profile': {}, 'selection_hash': 'old'}
        new = {'candidates': [ordinary, small], 'analytical_top1': 'split'}
        with patch.object(experiment, 'reference', return_value=(Path('/tmp/reference'), {'capture_hash': 'c'}, old)), \
                patch.object(experiment, 'load_graph', return_value=SimpleNamespace(matrices=(matrix,))), \
                patch('packet_search.select_candidates', return_value=new):
            _, result = experiment.selection(None, {}, SimpleNamespace(to_dict=lambda: {}))
        panel = {c['id']: c for c in result['candidates']}
        self.assertEqual(tuple(panel), experiment.VARIANTS)
        self.assertEqual(panel['current']['runtime_layouts'], panel['repaired']['runtime_layouts'])
        self.assertEqual(panel['repaired']['runtime_layouts'], [old_map])
        self.assertNotEqual(panel['smaller']['runtime_layouts'], [old_map])
        for label, mode in [('identity_legacy', 'legacy'), ('identity_repaired', 'repaired')]:
            self.assertTrue(panel[label]['same_pointer'])
            self.assertEqual(panel[label]['address_mode'], mode)
            self.assertEqual(panel[label]['runtime_layouts'][0]['rows'], identity)
        self.assertEqual(selected['runtime_layouts'], [old_map])

    def test_reference_rejects_changed_capture_and_selection(self):
        with tempfile.TemporaryDirectory() as name:
            args = SimpleNamespace(address_reference=Path(name), platform='matrix', case='gemm--asymmetric', tau_name='expert')
            directory = Path(name) / args.platform / args.case / args.tau_name
            directory.mkdir(parents=True)
            args.directory = directory
            from experiment_support import save_graph
            capture_hash = save_graph(directory / 'capture.pkl.gz', {'grid': (1,), 'bound': {'__names__': {0: 'K'}, 0: 128}})
            captured = {'capture_hash': capture_hash, 'config': {}, 'inputs': [], 'output_arguments': [2],
                        'kernel_name': 'gemm', 'source_identity': {'sources': {}}}
            selected = {'capture_hash': capture_hash, 'candidates': [{'runtime_layouts': ['changed']}], 'elapsed_seconds': 3}
            selected['selection_hash'] = digest({k: v for k, v in selected.items() if k != 'elapsed_seconds'})
            (directory / 'capture.json').write_text(json.dumps(captured))
            (directory / 'search.json').write_text(json.dumps(selected))
            experiment.check_capture(args, captured)
            with self.assertRaisesRegex(ValueError, 'reference config changed'):
                experiment.check_capture(args, {**captured, 'config': {'BLOCK_K': 64}})
            selected['candidates'][0]['runtime_layouts'] = []
            (directory / 'search.json').write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError, 'selection hash mismatch'):
                experiment.reference(args)

    def test_counter_csv_requires_one_dispatch_and_all_absolute_metrics(self):
        from stage1_nvidia_counter_analysis import COUNTER_METRICS
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'counter.csv'
            header = ['ID', 'Kernel Name', *COUNTER_METRICS, experiment.WAVEFRONTS]
            row = ['0', 'matmul_kernel', *['1,024'] * (len(header) - 2)]
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow(row)
            self.assertEqual(experiment.parse_counters(path, 'matmul_kernel')[experiment.WAVEFRONTS], 1024)
            with path.open('a') as stream:
                csv.writer(stream).writerow(row)
            with self.assertRaisesRegex(ValueError, 'one profiled target'):
                experiment.parse_counters(path, 'matmul_kernel')

    def test_counter_retry_keeps_complete_variants_and_uses_application_warmup(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / 'evaluate').mkdir()
            (directory / 'evaluate/process-0.json').write_text('{}')
            args = SimpleNamespace(directory=directory, root=directory, platform='matrix',
                case='gemm--asymmetric', address_reference=directory, tau_name='expert',
                warmup=10, iterations=50, resume=True)
            records = {'search.json': {'selection_hash': 's'}, 'validated.json': {'validation_hash': 'v'},
                       'evaluation.json': {}}
            calls = []
            def execute(command, **kwargs):
                label = command[command.index('--profile-candidate') + 1]
                calls.append(label)
                self.assertIn('application', command)
                self.assertEqual(command[command.index('--profile-from-start') + 1], 'off')
                self.assertEqual(command[command.index('--cache-control') + 1], 'none')
                if calls == ['ordinary', 'repaired']:
                    raise RuntimeError('interrupted')
                Path(command[command.index('--log-file') + 1]).write_text('counters')
                Path(command[command.index('--worker-output') + 1]).write_text('{}')
            with patch('packet_workflow.verify', return_value={'kernel_name': 'matmul'}), \
                    patch('packet_workflow.read', side_effect=lambda _, n: records[n]), \
                    patch('packet_workflow.accepted_candidates', return_value=[{'id': 'ordinary'}, {'id': 'repaired'}]), \
                    patch.object(experiment.shutil, 'which', return_value='/tmp/ncu'), \
                    patch.object(experiment.subprocess, 'check_output', return_value='test ncu'), \
                    patch.object(experiment.subprocess, 'run', side_effect=execute), \
                    patch.object(experiment, 'parse_counters', return_value={'sectors': 32}):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    experiment.profile(args)
                experiment.profile(args)
                experiment.profile(args)
            self.assertEqual(calls, ['ordinary', 'repaired', 'repaired'])
            self.assertIn('incomplete', json.loads((directory / 'profile.json').read_text())['status'])

    def test_debug_comparison_retains_cpu_stage_and_profiles_last(self):
        args = debug_suite.parser().parse_args(['--platform', 'matrix', '--address-reference', '/tmp/old',
            '--profile-addresses', '--cases', 'gemm--asymmetric'])
        self.assertEqual(debug_suite.stages(args)[-2:], ('evaluate', 'profile'))
        self.assertIn('--address-reference', debug_suite.command(args, args.cases[0], 'search'))
        self.assertEqual(debug_suite.command(args, args.cases[0], 'search')[0], str(ROOT / '.venv/bin/python'))
        args.profile_addresses = False
        self.assertEqual(debug_suite.stages(args)[-1], 'evaluate')


if __name__ == '__main__':
    unittest.main()
