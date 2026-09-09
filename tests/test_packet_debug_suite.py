from pathlib import Path
from dataclasses import dataclass
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'triton/packet_layout'))
import debug_suite as suite
import packet_workflow as workflow


@dataclass
class GraphFixture:
    supported: bool
    compiled_kernel: object
    manifest: object
    bound_arguments: dict


class DebugSuiteTests(unittest.TestCase):
    def args(self, directory, *extra):
        return suite.parser().parse_args(['--platform', 'tuolumne', '--root', directory, *extra])

    def output(self, args, case, stage):
        directory = args.root / args.platform / case / args.tau_name
        directory.mkdir(parents=True, exist_ok=True)
        for name in suite.OUTPUTS[stage]:
            suite.write_json(directory / name, {'stage': stage})
        if stage == 'evaluate':
            suite.write_json(directory / 'report.json', {'choice': {'selected': 'split'},
                'evaluation': {'candidates': {'split': {'speedup': 1.2}}}})

    def test_defaults_cover_all_29_cases_and_separate_cpu_python(self):
        args = self.args('/tmp/test-suite')
        self.assertEqual(len(args.cases), 29)
        self.assertEqual(len({suite.CASE_BY_ID[c].operator for c in args.cases}), 15)
        self.assertEqual((args.cpu_workers, args.grammar, args.selection), (8, 'split', 'analytical'))
        command = suite.command(args, args.cases[0], 'search')
        self.assertEqual(command[0], str(ROOT / '.venv/bin/python'))
        self.assertIn('--resume', command)
        self.assertEqual(suite.command(args, args.cases[0], 'capture')[0], str(ROOT / 'triton/.venv/bin/python'))
        frozen = suite.common_arguments(args)
        args.cpu_workers, args.minutes = 16, 55
        self.assertEqual(suite.common_arguments(args), frozen)

    def test_manifest_allows_worker_changes_but_rejects_grammar_changes(self):
        with tempfile.TemporaryDirectory() as name, patch.object(suite, 'activate'), \
                patch.object(suite, 'runtime_identity', return_value={'gpu': 'test'}), \
                patch.object(suite, 'identity', return_value={'source': 'test'}), \
                patch.object(suite.os, 'sched_getaffinity', return_value=set(range(16))), \
                patch.object(suite, 'collect', return_value=0), patch.object(suite.signal, 'signal'):
            command = ['--platform', 'tuolumne', '--root', name]
            self.assertEqual(suite.main(command), 0)
            self.assertEqual(suite.main([*command, '--cpu-workers', '16', '--minutes', '55']), 0)
            with self.assertRaisesRegex(SystemExit, 'settings changed'):
                suite.main([*command, '--grammar', 'split-chunks'])

    def test_graph_is_reused_if_selection_is_interrupted(self):
        with tempfile.TemporaryDirectory() as name:
            args = workflow.args_parser(['--stage', 'search', '--platform', 'tuolumne',
                '--case', 'sum--small', '--root', name, '--resume', '--grammar', 'split'])
            args.directory.mkdir(parents=True)
            captured = {'manifest': {}, 'grid': (1,), 'bound': {}, 'selected_config': {}}
            stamp = {'capture_hash': workflow.save_graph(args.directory / 'capture.pkl.gz', captured),
                     'source_identity': {'source_hash': 'test'}}
            analysis = GraphFixture(True, None, {}, {'__names__': {0: 'A'}})
            profile = SimpleNamespace(to_dict=lambda: {'test': True})
            with patch.object(workflow, 'verify', return_value=stamp), \
                    patch('search_algorithms.load_tau_profile', return_value=profile), \
                    patch('relay.triton_frontend.analyze_compiled_manifest', return_value=analysis) as analyze, \
                    patch('packet_search.select_candidates', side_effect=[RuntimeError('interrupted'), {'candidates': []}]) as select:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    workflow.search(args)
                args.cpu_workers = 8
                workflow.search(args)
                analyze.assert_called_once()
                self.assertEqual(select.call_args.kwargs, {'families': ('split',)})
            self.assertEqual(workflow.read(args, 'search.json')['candidates'], [])

    def test_restart_retains_completed_stages_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as name:
            args = self.args(name, '--cases', 'vector_add--small', 'sum--small', 'fp8_gemm--small')
            directory = args.root / args.platform
            directory.mkdir()
            calls = []

            def first(command, log, budget):
                case, stage = command[command.index('--case') + 1], command[command.index('--stage') + 1]
                calls.append((case, stage))
                if case == 'vector_add--small' and stage == 'validate':
                    return 1
                if case == 'sum--small' and stage == 'search':
                    budget.interrupt(signal.SIGTERM, None)
                    return None
                self.output(args, case, stage)
                return 0

            with patch.object(suite, 'run_stage', side_effect=first):
                self.assertEqual(suite.collect(args, directory, 'frozen', suite.Budget(1)), 0)
            state = json.loads((directory / 'debug-state.json').read_text())['cases']
            self.assertEqual(state['vector_add--small']['status'], 'failed')
            self.assertEqual(state['sum--small']['status'], 'paused')
            self.assertEqual(state['fp8_gemm--small']['status'], 'pending')
            calls.clear()

            def complete(command, log, budget):
                case, stage = command[command.index('--case') + 1], command[command.index('--stage') + 1]
                calls.append((case, stage))
                self.output(args, case, stage)
                return 0

            with patch.object(suite, 'run_stage', side_effect=complete):
                self.assertEqual(suite.collect(args, directory, 'frozen', suite.Budget(1)), 1)
            self.assertEqual(calls[0], ('fp8_gemm--small', 'capture'))
            self.assertNotIn(('sum--small', 'capture'), calls)
            self.assertFalse(any(case == 'vector_add--small' for case, stage in calls))
            calls.clear()
            args.retry_failed = True
            with patch.object(suite, 'run_stage', side_effect=complete):
                self.assertEqual(suite.collect(args, directory, 'frozen', suite.Budget(1)), 0)
            self.assertEqual(calls, [('vector_add--small', stage) for stage in ('validate', 'tune', 'evaluate')])
            with patch.object(suite, 'run_stage', side_effect=AssertionError('reran a complete stage')):
                suite.collect(args, directory, 'frozen', suite.Budget(1))
            (directory / 'sum--small/expert/search.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                suite.collect(args, directory, 'frozen', suite.Budget(1))

    def test_budget_stops_stage_and_its_descendant(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            child = directory / 'child.py'
            child.write_text('import pathlib, time\ntime.sleep(2)\npathlib.Path(' + repr(str(directory / 'leaked')) + ').touch()\n')
            parent = directory / 'parent.py'
            parent.write_text('import subprocess, sys, time\nsubprocess.Popen([sys.executable, ' + repr(str(child)) + '])\ntime.sleep(60)\n')
            self.assertIsNone(suite.run_stage([sys.executable, str(parent)], directory / 'stage.log', suite.Budget(.005)))
            # A following short process lets an accidentally surviving child expose itself.
            subprocess.run([sys.executable, '-c', 'import time; time.sleep(2)'], check=True)
            self.assertFalse((directory / 'leaked').exists())

    def test_timing_resume_reuses_completed_process_but_not_changed_measurement(self):
        with tempfile.TemporaryDirectory() as name:
            args = workflow.args_parser(['--stage', 'evaluate', '--platform', 'tuolumne',
                '--case', 'sum--small', '--root', name, '--resume', '--processes', '2'])
            records = {'search.json': {'selection_hash': 's'},
                       'validated.json': {'validation_hash': 'v'}, 'choice.json': {'choice_hash': 'c'}}
            calls = []

            def execute(command, **kwargs):
                output = Path(command[command.index('--worker-output') + 1])
                index = int(command[command.index('--process-index') + 1])
                calls.append(index)
                if index == 1 and calls == [0, 1]:
                    raise subprocess.CalledProcessError(1, command)
                suite.write_json(output, {'index': index})

            with patch.object(workflow, 'verify'), patch.object(workflow, 'read', side_effect=lambda _, name: records[name]), \
                    patch.object(workflow.subprocess, 'run', side_effect=execute), \
                    patch('packet_measure.summarize', side_effect=lambda records: records):
                with self.assertRaises(subprocess.CalledProcessError):
                    workflow.run_processes(args, 'evaluate')
                self.assertEqual(workflow.run_processes(args, 'evaluate'), [{'index': 0}, {'index': 1}])
                self.assertEqual(calls, [0, 1, 1])
                args.samples += 1
                workflow.run_processes(args, 'evaluate')
                self.assertEqual(calls, [0, 1, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
