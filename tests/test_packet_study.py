from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'triton/packet_layout'))
import _bootstrap
from packet_compatibility import native_contracts, constrain_graph, protected_bits
from layout_study import panel, freeze_choice, equivalence_report
from relay import Access, MemoryEvent, MatrixSpec, HardwareProfile


class CompatibilityTests(unittest.TestCase):
    def fixture(self):
        return ({'body': [{'op': 'load', 'site_id': 'memory.0', 'element_bytes': 2,
                          'source': {'file': '/kernel.py', 'line': 7, 'column': 9}}]},
                '.target sm_90a\n.file 1 "/kernel.py"\n.loc 1 7 9\n'
                'cp.async.cg.shared.global [%r1], [%rd2], 0x10, %r3;')

    def test_guard_requires_native_path_and_h100(self):
        manifest, ptx = self.fixture()
        contracts = native_contracts(manifest, {'ptx': ptx}, 'NVIDIA H100 80GB HBM3')
        self.assertEqual(contracts[0]['protected_element_bits'], 4)
        for gpu, code in [('AMD Instinct MI300A', ptx), ('NVIDIA H200', ptx),
                          ('NVIDIA H100', ptx.replace('.cg.', '.ca.')),
                          ('NVIDIA H100', ptx.replace('0x10', '8'))]:
            self.assertEqual(native_contracts(manifest, {'ptx': code}, gpu), [])
        with self.assertRaisesRegex(ValueError, 'matched'):
            native_contracts({'body': []}, {'ptx': ptx}, 'NVIDIA H100')

    def test_guard_preserves_sector_membership_without_regrouping(self):
        from relay.triton_frontend import TritonLaunchAnalysis
        from relay import layout_matrix_rows, row_major_layout
        from packet_search import split_templates
        from layout_contract import preserves_vector_bits
        manifest, ptx = self.fixture()
        event = MemoryEvent.make('e', 'memory.0', [Access('A', (0, 0))],
                                 metadata={'vector_elements': '8'})
        graph = TritonLaunchAnalysis(True, events=(event,))
        guarded = constrain_graph(graph, native_contracts(manifest, {'ptx': ptx}, 'NVIDIA H100'))
        self.assertEqual(guarded.events[0].accesses, event.accesses)
        self.assertEqual(guarded.components, graph.components)
        self.assertEqual(protected_bits(guarded.events), {'A': 4})
        matrix = MatrixSpec('A', (32, 64), 2, ('i', 'j'))
        ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
        unsafe = next(l for l in split_templates(matrix, 3) if l.name == 'split-a5-v3')
        self.assertTrue(preserves_vector_bits(layout_matrix_rows(matrix, unsafe), ordinary, 3))
        self.assertFalse(preserves_vector_bits(layout_matrix_rows(matrix, unsafe), ordinary, 4))
        for layout in split_templates(matrix, 4):
            for start in range(0, 32*64, 16):
                offsets = [layout.offset(matrix, divmod(start+j, 64)) for j in range(16)]
                self.assertEqual(offsets, list(range(offsets[0], offsets[0]+16)))
                self.assertEqual(offsets[0] % 16, 0)


def fixture(shape=(32, 32), bits=2):
    from relay.objectives import EdgeFamily, Hyperedge, ScopeKey
    matrix = MatrixSpec('A', shape, 2, ('i', 'j'), role='read')
    component = EdgeFamily(ScopeKey('issue', 32, 'stream', 'load'),
                            {'A': (Hyperedge.make([(0, 0)]),)}, normalization_bytes=2).at_scale(32)
    event = MemoryEvent.make('e', 'load', [Access('A', (0, 0))],
                             metadata={'vector_elements': '1', 'protected_element_bits': str(bits)})
    allocation = SimpleNamespace(name='A', argument=0, eligible=True, path=(), true_shape=shape,
        envelope_shape=shape, strides=(shape[1], 1), dense_status='dense', element_bytes=2)
    graph = SimpleNamespace(matrices=(matrix,), allocations=(allocation,), events=(event,),
                            components=(component,), bound_arguments={'__names__': {0: 'A'}})
    profile = HardwareProfile(profile_id='test', device={}, byte_scales=(32,),
                              tau={component.name: 1.}, fine_component=component.name)
    return graph, profile


class StudyTests(unittest.TestCase):
    def test_worker_checkpoint_resumes_after_measurement_adds_diagnostics(self):
        import json
        import tempfile
        from unittest.mock import patch
        from experiment_support import digest
        from layout_study import worker, measure_batch, storages
        from packet_workflow import write_json
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(directory=Path(directory), samples=11, iterations=20,
                                   warmup=3, process_index=0)
            candidates = [{'id': 'ordinary', 'runtime_layouts': []}]
            search = {'candidates': candidates}
            search['selection_hash'] = digest(search)
            write_json(args.directory/'search.json', search)
            batch = {'phase': 'study-tune/batch-0', 'selection_hash': search['selection_hash'],
                     'choice_hash': None, 'storages': storages(candidates)}
            args.study_batch = args.directory/'study-tune/batch-0/batch.json'
            args.worker_output = args.study_batch.parent/'process-0.json'
            write_json(args.study_batch, batch)
            def measure(args):
                for storage in args.study_storages:
                    storage['tiles'] = {}
                write_json(args.worker_output, {'storages': args.study_storages, 'timings': {'ordinary': 1.}})
            with patch('conventional.worker', side_effect=measure):
                worker(args)
            saved = json.loads(args.worker_output.read_text())
            self.assertEqual(saved['batch_hash'], digest(batch))
            self.assertIn('tiles', saved['storages'][0])
            with patch('layout_study.subprocess.run') as launch:
                self.assertEqual(measure_batch(args, search, 'study-tune', 0, candidates, 0), saved)
                launch.assert_not_called()
                args.samples += 1
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    measure_batch(args, search, 'study-tune', 0, candidates, 0)
                args.samples -= 1
                saved['timings']['ordinary'] = 2.
                write_json(args.worker_output, saved)
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    measure_batch(args, search, 'study-tune', 0, candidates, 0)

    def test_count_reuse_and_restart_match_direct_scoring(self):
        import tempfile
        from unittest.mock import patch
        from relay import row_major_layout, score_layouts
        from relay.objectives import EdgeFamily, Hyperedge, ScopeKey
        from packet_search import split_templates
        from relay.prepared_scoring import PreparedRegionScorer
        graph, profile = fixture((8, 16), 1)
        component = EdgeFamily(ScopeKey('issue', 32, 'stream', 'load'),
            {'A': (Hyperedge.make([(0, 0), (3, 7), (7, 8)]),)}, normalization_bytes=6).at_scale(32)
        graph.components = (component,)
        matrix = graph.matrices[0]
        expected = {}
        for layout in split_templates(matrix, 1):
            from relay import layout_matrix_rows
            rows = tuple(layout_matrix_rows(matrix, layout))
            expected[rows] = score_layouts({'A': matrix}, (component,), {'A': layout}, hardware_profile=profile)
        with tempfile.TemporaryDirectory() as directory:
            first = panel(graph, profile, 'sum--small', checkpoint_dir=Path(directory))
            with patch.object(PreparedRegionScorer, 'populate', side_effect=AssertionError('must resume saved components')):
                second = panel(graph, profile, 'sum--small', checkpoint_dir=Path(directory))
            self.assertEqual(first, second)
        for candidate in first['candidates']:
            direct = expected[tuple(candidate['physical_rows'])]
            self.assertEqual(candidate['score']['hardware_area'], direct.hardware_area)
            self.assertEqual(candidate['score']['components'][0]['raw_region_count'], direct.components[0].raw_region_count)

    def test_frozen_winner_is_evaluated_on_new_data(self):
        import json
        import tempfile
        from unittest.mock import patch
        from experiment_support import digest
        from layout_study import run
        from packet_workflow import write_json
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(directory=Path(directory), processes=2, case='sum--small', platform='tuolumne')
            record = {'candidates': [{'id': name, 'runtime_layouts': []} for name in ('ordinary', 'top', 'fast')],
                      'analytical_top1': 'top', 'batch_size': 4, 'poor_speedup_threshold': .9}
            record['selection_hash'] = digest(record)
            write_json(args.directory / 'search.json', record)
            calls = []
            def measure(args, search, phase, batch, candidates, process, **kwargs):
                calls.append((phase, process))
                if phase == 'study-evaluate':
                    self.assertEqual(json.loads((args.directory/'study-choice.json').read_text())['empirical_winner'], 'fast')
                    self.assertTrue(kwargs['choice_hash'])
                times = {'ordinary': 1., 'identity': 1., 'same_pointer': 1.}
                for candidate in candidates:
                    value = (.8 if phase == 'study-tune' else 2.) if candidate['id'] == 'fast' else 1.
                    times[candidate['id']+'-fixed'] = value
                    if kwargs.get('manual'):
                        times[candidate['id']+'-explicit-fixed'] = value
                timings = {label: {'median_ms': value} for label, value in times.items()}
                packing = {name.removesuffix('-fixed'): {'median_ms': 0.} for name in times}
                return {'timings': timings, 'allocation': {'placements': [{'timings': timings}]},
                        'packing': packing, 'failures': {}, 'codegen': {label: {} for label in times}}
            with patch('layout_study.verify'), patch('layout_study.measure_batch', side_effect=measure):
                run(args)
            result = json.loads((args.directory/'study.json').read_text())
            self.assertEqual(calls, [('study-tune', 0), ('study-tune', 1), ('study-evaluate', 0), ('study-evaluate', 1)])
            self.assertEqual(result['H_ordinary_over_empirical']['speedup'], .5)
            self.assertEqual(result['S_top1_over_empirical']['speedup'], .5)
            self.assertEqual(result['choice']['empirical_winner'], 'fast')

    def test_equal_flags_and_scores_do_not_remove_physical_maps(self):
        graph, profile = fixture()
        result = panel(graph, profile, 'sum--small')
        candidates = result['candidates']
        self.assertEqual(len(candidates), 1+5*3)
        self.assertEqual(len({tuple(c['physical_rows']) for c in candidates}), len(candidates))
        self.assertLess(len({c['feature_class'] for c in candidates}), len(candidates))
        self.assertLess(len({c['flag_class'] for c in candidates}), len(candidates))
        self.assertEqual(result['analytical_top1'], 'ordinary')

    def test_h100_panel_is_exactly_nineteen_a_only_maps(self):
        graph, profile = fixture((512, 4096), 4)
        result = panel(graph, profile, 'gemm--asymmetric')
        self.assertEqual(len(result['candidates']), 19)
        self.assertEqual({tuple(c['tile']) for c in result['candidates'] if c['id'] != 'ordinary'},
                         {(2**a, j) for a in range(1, 10) for j in (16, 32)})
        for candidate in result['candidates']:
            self.assertTrue(all(layout['argument'] == 0 for layout in candidate['runtime_layouts']))
        graph, profile = fixture((512, 4096), 3)
        with self.assertRaisesRegex(ValueError, 'sector-protected'):
            panel(graph, profile, 'gemm--asymmetric')

    def test_selection_retains_rejections_and_freezes_manual_audits(self):
        record = {'selection_hash': 'panel', 'analytical_top1': 'top', 'poor_speedup_threshold': .9,
                  'candidates': [{'id': name} for name in ('ordinary', 'top', 'fast', 'slow', 'rejected')]}
        tuning = {'ordinary': {'speedup': 1}, 'top': {'speedup': .98}, 'fast': {'speedup': 1.2},
                  'slow': {'speedup': .8}, 'rejected': {'failed': ['compile rejection']}}
        choice = freeze_choice(record, tuning)
        self.assertEqual(choice['empirical_winner'], 'fast')
        self.assertEqual(choice['held_out'], ['ordinary', 'top', 'fast'])
        self.assertEqual(choice['manual_audits'], ['slow', 'rejected'])
        self.assertEqual(choice['tuning']['rejected'], tuning['rejected'])

    def test_feature_flag_and_scalar_equivalence_are_separate(self):
        graph, profile = fixture()
        record = panel(graph, profile, 'sum--small')
        tuning = {c['id']: {'speedup': 1+i/10} for i, c in enumerate(record['candidates'])}
        report = equivalence_report(record, tuning)
        self.assertEqual(set(report), {'scalar_class', 'feature_class', 'flag_class'})
        self.assertGreater(report['feature_class'][0]['training_time_spread'], 1)


if __name__ == '__main__':
    unittest.main()
