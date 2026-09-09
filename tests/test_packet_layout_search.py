from pathlib import Path
from itertools import product
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton/packet_layout'))
import _bootstrap
from packet_search import split_templates, chunk_templates, partial_flag, address_fields
from relay import MatrixSpec, CanonicalLayout, layout_matrix_rows
from layout_contract import preserves_vector_bits


class PacketSearchTests(unittest.TestCase):
    def test_address_template_cost_uses_ordinary_offset_coordinates(self):
        from relay import row_major_layout
        matrix = MatrixSpec('A', (8, 16), 4, ('i', 'j'))
        ordinary = row_major_layout(matrix)
        column = CanonicalLayout('column', 'A', (3, 4), (0,) * 3 + (1,) * 4, (1, 0))
        self.assertEqual(len(address_fields(matrix, ordinary)), 1)
        self.assertEqual(len(address_fields(matrix, column)), 2)
        for layout in split_templates(matrix, 1):
            fields = address_fields(matrix, layout)
            for i, j in product(range(8), range(16)):
                offset = i * 16 + j
                deposited = sum(((offset >> f.source_start) & ((1 << f.width) - 1)) << f.target_start for f in fields)
                self.assertEqual(deposited, layout.offset(matrix, (i, j)))

    def test_packet_witness_is_bijective_and_matches_split_formula(self):
        matrix = MatrixSpec('A', (32, 32), 4, ('i', 'j'))
        layout = next(c for c in split_templates(matrix, 2) if c.name == 'split-a5-v2')
        addresses = {layout.offset(matrix, (i, j)) for i, j in product(range(32), repeat=2)}
        self.assertEqual(addresses, set(range(1024)))
        for i, j in product(range(32), repeat=2):
            self.assertEqual(layout.offset(matrix, (i, j)), (j // 4) * 128 + i * 4 + j % 4)
        for group in range(8):
            points = [(i, group * 4 + s) for i in range(32) for s in range(4)]
            for scale, expected in [(32, 16), (128, 4)]:
                self.assertEqual(len({layout.offset(matrix, p) * 4 // scale for p in points}), expected)

    def test_partial_flags_preserve_partitions_without_fixing_complete_flag(self):
        matrix = MatrixSpec('A', (32, 32), 4, ('i', 'j'))
        a = CanonicalLayout('a', 'A', (5, 5), (1, 1, 0, 0, 0, 0, 0, 1, 1, 1), (1, 0))
        b = CanonicalLayout('b', 'A', (5, 5), (1, 1, 0, 0, 0, 1, 1, 1, 0, 0), (1, 0))
        ar, br = (layout_matrix_rows(matrix, layout) for layout in (a, b))
        self.assertEqual(partial_flag(ar, (2, 3, 5)), partial_flag(br, (2, 3, 5)))
        self.assertNotEqual(partial_flag(ar, range(11)), partial_flag(br, range(11)))
        points = list(product(range(32), repeat=2))
        for depth in (2, 3, 5):
            groups = []
            for layout in (a, b):
                partitions = {}
                for point in points:
                    partitions.setdefault(layout.offset(matrix, point) >> depth, set()).add(point)
                groups.append({frozenset(group) for group in partitions.values()})
            self.assertEqual(*groups)

    def test_every_template_preserves_packets_and_is_a_permutation(self):
        matrix = MatrixSpec('A', (8, 16), 4, ('i', 'j'))
        from relay import row_major_layout
        ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
        layouts = list(split_templates(matrix, 2)) + list(chunk_templates(matrix, 2, ((2, 3),)))
        for layout in layouts:
            rows = layout_matrix_rows(matrix, layout)
            self.assertTrue(preserves_vector_bits(rows, ordinary, 2))
            self.assertEqual(len({layout.offset(matrix, p) for p in product(range(8), range(16))}), 128)

    def test_register_and_instruction_growth_are_not_hard_vetoes(self):
        from packet_runtime import primitive_rejections
        baseline = dict(final_assembly='sass', memory_primitives={'LDG.E.128': 1},
                        matrix_primitives={}, communication={}, packet_sites=[{'site': 0}],
                        n_spills=0, shared_bytes=0, n_regs=20, final_instruction_count=50)
        candidate = {**baseline, 'n_regs': 24, 'final_instruction_count': 70}
        self.assertEqual(primitive_rejections(baseline, candidate), [])
        self.assertEqual(primitive_rejections(baseline, {**candidate, 'memory_primitives': {'LDG.E.128': 3}}), [])
        self.assertTrue(primitive_rejections(baseline, {**candidate, 'n_spills': 1}))
        self.assertTrue(primitive_rejections(baseline, {**candidate, 'memory_primitives': {'LDG.E': 4}}))

    def test_family_selection_matches_exhaustive_supported_scores(self):
        from types import SimpleNamespace
        from relay import Access, MemoryEvent, HardwareProfile, SimpleRelayProblem, score_layouts
        from relay.objectives import EdgeFamily, Hyperedge, ScopeKey
        from packet_search import select_candidates
        matrix = MatrixSpec('A', (8, 16), 4, ('i', 'j'), role='read')
        points = tuple(product(range(8), range(2)))
        component = EdgeFamily(ScopeKey('issue', 32, 'stream', 'load'),
            {'A': (Hyperedge.make(points),)}, normalization_bytes=64).at_scale(32)
        event = MemoryEvent.make('load', 'load', [Access('A', p) for p in points], metadata={'vector_elements': '2'})
        profile = HardwareProfile(profile_id='test', device={}, byte_scales=(32,),
                                  tau={component.name: 1.}, fine_component=component.name)
        problem = SimpleRelayProblem((matrix,), (event,), (), (), 'canonical')
        allocation = SimpleNamespace(name='A', argument=0, eligible=True, path=(),
            true_shape=(8, 16), envelope_shape=(8, 16), strides=(16, 1), dense_status='dense', element_bytes=4)
        analysis = SimpleNamespace(relay_problem=lambda **kwargs: problem, allocations=(allocation,),
            events=(event,), components=(component,), bound_arguments={'__names__': {0: 'A'}})
        result = select_candidates(analysis, profile)
        self.assertLessEqual(len(result['candidates']), 3)
        best = min(score_layouts({'A': matrix}, (component,), {'A': candidate}, hardware_profile=profile).hardware_area
                   for candidate in split_templates(matrix, 1))
        selected = next(c for c in result['candidates'] if 'split' in c['families'])
        self.assertAlmostEqual(selected['score']['hardware_area'], best)
        self.assertNotEqual(result['analytical_top1'], 'ordinary')
        self.assertLess(best, result['candidates'][0]['score']['hardware_area'])
        self.assertEqual(len({str(c['runtime_layouts']) for c in result['candidates']}), len(result['candidates']))


class PacketMeasuredSelectionTests(unittest.TestCase):
    def test_conventional_selection_is_frozen_before_independent_evaluation(self):
        import json
        import tempfile
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch
        from conventional import run as compare
        from packet_workflow import write_json, read
        from packet_measure import summarize
        from experiment_support import digest

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(directory=Path(directory), processes=1, case='row_column--small')
            search = {'candidates': []}
            search['selection_hash'] = digest(search)
            write_json(args.directory / 'search.json', search)
            phases = []

            def measure(args, phase):
                phases.append(phase)
                if phase == 'conventional-tune':
                    values = {'ordinary-s0': 1.2, 'ordinary-s1': 1., 'column-s0': .8, 'column-s1': 1.1}
                else:
                    frozen = read(args, 'conventional-choice.json')
                    self.assertEqual(frozen['selected_schedules'], {'ordinary': 'ordinary-s1', 'column': 'column-s0'})
                    values = {'ordinary-s1': 1., 'column-s0': 2.}
                groups = {label: label.split('-')[0] for label in values}
                values.update(ordinary=1., identity=1., same_pointer=1.)
                timings = {label: {'median_ms': value, 'mean_ms': value, 'min_ms': value, 'samples_ms': [value] * 3}
                           for label, value in values.items()}
                record = {'timings': timings, 'groups': groups, 'failures': {},
                          'packing': {'ordinary': {'median_ms': 0}, 'column': {'median_ms': 5}}}
                write_json(args.directory / phase / 'process-0.json', record)
                return summarize([record])

            # A legacy module named run must not intercept the new driver's API.
            with patch.dict(sys.modules, {'run': ModuleType('legacy_run')}), patch('packet_workflow.verify'), patch('packet_workflow.run_processes', side_effect=measure):
                compare(args)
            result = json.loads((args.directory / 'conventional.json').read_text())
            self.assertEqual(phases, ['conventional-tune', 'conventional-evaluate'])
            self.assertEqual(result['choice']['budget_per_storage'], 6)
            self.assertEqual(result['results']['column']['speedup'], .5)
            self.assertFalse(result['deployment_selection'])

    def test_changed_artifacts_cannot_reuse_selection_or_validation_hashes(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from packet_workflow import read, accepted_candidates
        from experiment_support import digest
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(directory=Path(directory))
            selection = {'candidates': [{'id': 'ordinary', 'runtime_layouts': []}]}
            selection['selection_hash'] = digest(selection)
            selection['elapsed_seconds'] = 1
            path = args.directory / 'search.json'
            path.write_text(json.dumps(selection))
            self.assertEqual(read(args, 'search.json'), selection)
            selection['candidates'][0]['id'] = 'changed'
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read(args, 'search.json')
            validation = {'complete': True, 'selection_hash': 'search', 'candidates': [{'id': 'ordinary', 'accepted': True}]}
            validation['validation_hash'] = digest(validation)
            path = args.directory / 'validated.json'
            path.write_text(json.dumps(validation))
            self.assertEqual(len(accepted_candidates(args, {'selection_hash': 'search'})), 1)
            validation['candidates'][0]['id'] = 'changed'
            path.write_text(json.dumps(validation))
            with self.assertRaisesRegex(ValueError, 'changed or stale'):
                accepted_candidates(args, {'selection_hash': 'search'})

    def test_downstream_stage_preserves_interrupted_capture_as_cause(self):
        import json
        import tempfile
        from unittest.mock import patch
        import packet_workflow as run
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tuolumne/sum--small/expert'
            path.mkdir(parents=True)
            (path / 'status.json').write_text(json.dumps({'stage': 'capture', 'status': 'running'}))
            argv = ['run.py', '--stage', 'search', '--root', directory, '--platform', 'tuolumne', '--case', 'sum--small']
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'prior capture failure'):
                run.main()
            self.assertEqual(json.loads((path / 'status.json').read_text())['stage'], 'capture')

    def test_conversion_cost_can_reverse_kernel_speedup(self):
        from packet_measure import conversion_result
        result = conversion_result(10, 8, 25)
        self.assertEqual(result['kernel_speedup'], 1.25)
        self.assertLess(result['one_use_including_conversion_speedup'], 1)
        self.assertEqual(result['break_even_reuses'], 13)
        self.assertIsNone(conversion_result(10, 11, 25)['break_even_reuses'])

    def test_conventional_storage_matches_explicit_address_functions(self):
        from conventional import storage_candidates, SCHEDULES
        from layout_contract import RuntimeLayout
        candidates = storage_candidates(32, {'candidates': []})
        self.assertEqual(len(SCHEDULES), 6)
        for candidate in candidates[1:]:
            rows = candidate['runtime_layouts'][0]['rows']
            for i, j in product(range(32), repeat=2):
                logical = i | (j << 5)
                offset = sum(((logical & row).bit_count() & 1) << k for k, row in enumerate(rows))
                expected = j * 32 + i if candidate['storage'] == 1 else (i // 16) * 512 + (j // 16) * 256 + (i % 16) * 16 + j % 16
                self.assertEqual(offset, expected)

    def test_measured_choice_requires_gain_and_clean_controls(self):
        from packet_measure import measured_choice
        ordinary = {'speedup': 1., 'speedup_ci95': [1., 1.],
                    'identity_max_deviation': 0., 'same_pointer_max_deviation': 0.}
        good = {**ordinary, 'speedup': 1.1, 'speedup_ci95': [1.06, 1.14]}
        self.assertEqual(measured_choice({'ordinary': ordinary, 'split': good}), 'split')
        self.assertEqual(measured_choice({'ordinary': ordinary, 'split': {**good, 'identity_max_deviation': .04}}), 'ordinary')
        self.assertEqual(measured_choice({'ordinary': ordinary, 'split': {**good, 'same_pointer_max_deviation': .04}}), 'ordinary')
        clear = {**good, 'speedup': 1.5, 'speedup_ci95': [1.4, 1.6], 'identity_max_deviation': .02}
        self.assertEqual(measured_choice({'ordinary': ordinary, 'split': clear}), 'split')
        self.assertEqual(measured_choice({'ordinary': ordinary, 'split': {**good, 'identity_max_deviation': 1.}}), 'ordinary')
        self.assertEqual(measured_choice({'ordinary': ordinary, 'split': {**good, 'speedup_ci95': [.99, 1.2]}}), 'ordinary')

    def test_source_isolation_from_active_experiments(self):
        from experiment_support import source_identity
        sources = source_identity()['sources']
        self.assertFalse(any(name.startswith('triton/packet_layout/') for name in sources))

    def test_scheduler_cpu_search_does_not_request_tuolumne_gpu(self):
        from types import SimpleNamespace
        from submit import commands
        args = SimpleNamespace(platform='tuolumne', queue=None, root=Path('/tmp/packet-jobs'),
                               tau_name='expert', selection='measured', cases=['sum--small'])
        jobs = list(commands(args))
        self.assertEqual(len(jobs), 5)
        self.assertNotIn('-g1', jobs[1][2])
        for i in [0, 2, 3, 4]:
            self.assertIn('-g1', jobs[i][2])
        args.platform = 'matrix'
        self.assertTrue(all('--gpus=1' in job[2] for job in commands(args)))
        args.cases = ['row_column--small']
        self.assertEqual(list(commands(args))[-1][1], 'conventional')
