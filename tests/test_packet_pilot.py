"""Packet pilot grammar and job-boundary checks (no GPU required)."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton'))
sys.path.insert(0, str(ROOT / 'triton/experiments'))
from layout_panels import _candidate_layouts
from relay import MatrixSpec, row_major_layout, layout_matrix_rows
from layout_contract import preserves_vector_bits

spec = importlib.util.spec_from_file_location('submit_pilots', ROOT / 'triton/experiments/submit-pilots.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class PacketPilotTests(unittest.TestCase):
    def test_packet_canonical_census_matches_filtered_full_grammar(self):
        matrix = MatrixSpec('A', (4, 8), 4, ('r', 'c'))
        ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
        full, _, _ = _candidate_layouts(matrix, [(4, 8)], experiment=1, samples=100, seed=0)
        for bits in range(4):
            with self.subTest(bits=bits):
                panel, _, size = _candidate_layouts(matrix, [(4, 8)], experiment=1,
                                                   samples=100, seed=0, packet_bits=bits)
                expected = {layout_matrix_rows(matrix, x) for x, _, _ in full
                            if preserves_vector_bits(layout_matrix_rows(matrix, x), ordinary, bits)}
                self.assertEqual(expected, {layout_matrix_rows(matrix, x) for x, _, _ in panel})
                self.assertEqual(size, len(expected))
                self.assertIn(ordinary, expected)

    def test_sample_is_reproducible_unique_and_packet_compatible(self):
        matrix = MatrixSpec('A', (64, 64), 4, ('r', 'c'))
        options = dict(experiment=1, samples=20, seed=7, packet_bits=2)
        first, _, _ = _candidate_layouts(matrix, [(64, 64)], **options)
        second, _, _ = _candidate_layouts(matrix, [(64, 64)], **options)
        self.assertEqual(first, second)
        rows = [layout_matrix_rows(matrix, x) for x, _, _ in first]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(set(rows)), 20)
        ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
        self.assertTrue(all(preserves_vector_bits(x, ordinary, 2) for x in rows))

    def test_gl_panel_cannot_accidentally_enable_packet_constraints(self):
        matrix = MatrixSpec('A', (4, 8), 4, ('r', 'c'))
        with self.assertRaises(ValueError):
            _candidate_layouts(matrix, [(4, 8)], experiment=3, samples=10, seed=0, packet_bits=1)

    def test_resume_rejects_changed_identity_and_legacy_reports(self):
        from packet_pilot import freeze_run
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with patch('packet_pilot.fingerprint', return_value={'source': 'first'}):
                first = freeze_run(None, directory)
                self.assertEqual(first, freeze_run(None, directory))
            with patch('packet_pilot.fingerprint', return_value={'source': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    freeze_run(None, directory)
            legacy = directory / 'legacy'
            legacy.mkdir()
            (legacy / 'report.json').write_text('{}')
            with patch('packet_pilot.fingerprint', return_value={'source': 'first'}):
                with self.assertRaisesRegex(ValueError, 'fresh results root'):
                    freeze_run(None, legacy)
            self.assertFalse((legacy / 'packet-identity.json').exists())

    def test_worker_routes_packet_profiles_only_when_enabled(self):
        from stage1_counter_sweep import _worker_command
        args = SimpleNamespace(case='gemv', transaction_bytes=32, candidates=8,
                               platform='matrix', results_dir=Path('/tmp/pilot/profiles'),
                               packet_layout=True)
        command = _worker_command(args, '--profile-rows', '1', '2')
        self.assertIn('--packet-layout', command)
        self.assertEqual(command[command.index('--profile-panel') + 1], '/tmp/pilot/profiles/panel.json')
        args.packet_layout = False
        command = _worker_command(args, '--profile-rows', '1', '2')
        self.assertNotIn('--packet-layout', command)
        self.assertNotIn('--profile-panel', command)

    def test_minimal_jobs_exclude_bias_relu(self):
        for platform in ('matrix', 'tuolumne'):
            args = SimpleNamespace(platform=platform, cases=submit.CASES, experiments=[1],
                                   results_root=Path('/tmp/pilot'), plots_root=Path('/tmp/plots'),
                                   queue='pbatch', wall_time=None, layouts=100)
            jobs = list(submit.commands(args))
            self.assertEqual(len(jobs), 6)
            for _, command in jobs:
                self.assertNotIn('bias_relu', command)
                self.assertEqual(command[command.index('--stratification') + 1], 'all')
                self.assertEqual(command[command.index('--profile-launches') + 1], '3')
            args.experiments = [1, 3]
            self.assertEqual(len(list(submit.commands(args))), 12)


if __name__ == '__main__':
    unittest.main()
