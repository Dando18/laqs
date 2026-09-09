from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from relay.access_scopes import build_edge_families
from relay.parallel import ordered_parallel_map
from relay.triton_frontend import (EvaluationLimits, UnsupportedTritonAnalysis, _trace_chunk,
                                  evaluate_manifest, parse_access_manifest)
from test_triton_automatic_frontend import vector_manifest, bound_arguments


def multiply(context, value):
    if value == -1:
        raise UnsupportedTritonAnalysis('test-category', 'worker rejected input', site='site0')
    return context['factor'] * value


def recorded_multiply(context, value):
    directory = Path(context['directory'])
    if value == 3 and (directory / 'interrupt').exists():
        raise RuntimeError('simulated interruption')
    with (directory / f'calls-{value}').open('a') as stream:
        stream.write('called\n')
    return 3 * value


def loop_manifest():
    payload = vector_manifest(4)
    payload['expressions'].extend([
        {'id': 7, 'op': 'constant', 'type': 'i32', 'attributes': {'value': 0}},
        {'id': 8, 'op': 'constant', 'type': 'i32', 'attributes': {'value': 2}},
        {'id': 9, 'op': 'constant', 'type': 'i32', 'attributes': {'value': 1}},
        {'id': 10, 'op': 'loop_iv', 'type': 'i32', 'attributes': {'name': 'k'}},
        {'id': 11, 'op': 'mul', 'type': 'i32', 'operands': [10, 1]},
        {'id': 12, 'op': 'add', 'type': 'tensor<4xi32>', 'operands': [4, 11]},
        {'id': 13, 'op': 'cmp', 'type': 'tensor<4xi1>', 'operands': [12, 5],
         'attributes': {'predicate': 'slt'}},
    ])
    payload['body'] = [{'kind': 'for', 'iv': 'k', 'lower': 7, 'upper': 8, 'step': 9,
                        'body': [dict(node, offset=12, mask=13) for node in payload['body']]}]
    return parse_access_manifest(payload)


class ParallelGraphTests(unittest.TestCase):
    def test_checkpoint_resume_reuses_completed_work_across_worker_counts(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / 'interrupt').touch()
            context = {'directory': name}
            checkpoint = directory / 'checkpoint'
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                list(ordered_parallel_map(recorded_multiply, context, range(7), 2,
                                          checkpoint_dir=checkpoint))
            (directory / 'interrupt').unlink()
            self.assertEqual(list(ordered_parallel_map(recorded_multiply, context, range(7), 3,
                                                       checkpoint_dir=checkpoint)), list(range(0, 21, 3)))
            for value in range(7):
                self.assertEqual((directory / f'calls-{value}').read_text(), 'called\n')
            with patch('relay.parallel.multiprocessing.get_context', side_effect=AssertionError('spawned cached work')):
                self.assertEqual(list(ordered_parallel_map(recorded_multiply, context, range(7), 8,
                                                           checkpoint_dir=checkpoint)), list(range(0, 21, 3)))
            with self.assertRaisesRegex(ValueError, 'context changed'):
                list(ordered_parallel_map(recorded_multiply, {'directory': name, 'changed': True},
                                           range(7), 1, checkpoint_dir=checkpoint))

    def test_checkpointed_trace_and_scopes_are_exact_and_resume_serially(self):
        with tempfile.TemporaryDirectory() as name:
            expected = evaluate_manifest(loop_manifest(), bound_arguments(31), (8,))
            observed = evaluate_manifest(loop_manifest(), bound_arguments(31), (8,),
                limits=EvaluationLimits(workers=2, checkpoint_dir=name))
            self.assertEqual(observed, expected)
            with patch('relay.triton_frontend._trace_chunk', wraps=_trace_chunk) as trace:
                # A wrapper changes function identity; use original metadata for cache binding.
                trace.__module__, trace.__qualname__ = 'relay.triton_frontend', '_trace_chunk'
                restored = evaluate_manifest(loop_manifest(), bound_arguments(31), (8,),
                    limits=EvaluationLimits(workers=1, checkpoint_dir=name))
                trace.assert_not_called()
            self.assertEqual(restored, expected)
            _, matrices, events, sequences = observed
            matrices = {m.name: m for m in matrices}
            events = {e.id: replace(e, weight=.1) for e in events}
            sequences = tuple(replace(s, weight=s.weight * .3) for s in sequences)
            baseline = build_edge_families(matrices, events, sequences)
            checkpoint = Path(name) / 'scopes'
            self.assertEqual(build_edge_families(matrices, events, sequences, workers=2,
                                                checkpoint_dir=checkpoint), baseline)
            self.assertEqual(build_edge_families(matrices, events, sequences, workers=1,
                                                checkpoint_dir=checkpoint), baseline)

    def test_checkpoints_survive_a_new_interpreter_hash_seed(self):
        script = '''
import sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_parallel_graph import loop_manifest, bound_arguments
from relay.triton_frontend import evaluate_manifest, EvaluationLimits
from relay.access_scopes import build_edge_families
directory, workers = sys.argv[1], int(sys.argv[2])
_, matrices, events, sequences = evaluate_manifest(loop_manifest(), bound_arguments(31), (8,),
    limits=EvaluationLimits(workers=workers, checkpoint_dir=directory))
build_edge_families({m.name: m for m in matrices}, {e.id: e for e in events}, sequences,
                   workers=workers, checkpoint_dir=Path(directory) / 'scopes')
'''
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-c', script, directory]
            subprocess.run([*command, '2'], env={**os.environ, 'PYTHONHASHSEED': '1'}, check=True)
            paths = list(Path(directory).rglob('*.pkl.gz'))
            self.assertTrue(paths)
            before = {path: path.stat().st_mtime_ns for path in paths}
            subprocess.run([*command, '1'], env={**os.environ, 'PYTHONHASHSEED': '2'}, check=True)
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in paths})

    def test_ordered_workers_accept_readonly_context_and_preserve_errors(self):
        context = MappingProxyType({'factor': 3})
        self.assertEqual(list(ordered_parallel_map(multiply, context, range(9), 2)),
                         list(range(0, 27, 3)))
        with self.assertRaises(UnsupportedTritonAnalysis) as caught:
            list(ordered_parallel_map(multiply, context, [1, -1, 2], 2))
        self.assertEqual((caught.exception.category, caught.exception.site), ('test-category', 'site0'))

    def test_parallel_loop_trace_is_identical_with_masks_and_resource_anchors(self):
        manifest = loop_manifest()
        for preserve in (False, True):
            expected = evaluate_manifest(manifest, bound_arguments(31), (8,),
                                         preserve_resource_anchors=preserve)
            observed = evaluate_manifest(manifest, bound_arguments(31), (8,),
                preserve_resource_anchors=preserve, limits=EvaluationLimits(workers=2))
            self.assertEqual(observed, expected)

    def test_parallel_scope_graph_preserves_order_sources_and_fractional_weights(self):
        _, matrices, events, sequences = evaluate_manifest(loop_manifest(), bound_arguments(31), (8,))
        matrices = {m.name: m for m in matrices}
        events = {e.id: replace(e, weight=.1) for e in events}
        sequences = tuple(replace(s, weight=s.weight * .3) for s in sequences)
        serial = build_edge_families(matrices, events, sequences)
        self.assertEqual(build_edge_families(matrices, events, sequences, workers=2), serial)

    def test_parallel_retained_event_budget_and_out_of_bounds_are_not_hidden(self):
        manifest = loop_manifest()
        for workers in (1, 2):
            with self.assertRaises(UnsupportedTritonAnalysis) as caught:
                evaluate_manifest(manifest, bound_arguments(31), (8,),
                    limits=EvaluationLimits(workers=workers, max_dynamic_events=3))
            self.assertEqual(caught.exception.category, 'enumeration_bound')
            payload = vector_manifest(4)
            for node in payload['body']:
                node['mask'] = None
            with self.assertRaises(UnsupportedTritonAnalysis) as caught:
                evaluate_manifest(parse_access_manifest(payload), bound_arguments(3), (8,),
                    preserve_resource_anchors=True, limits=EvaluationLimits(workers=workers))
            self.assertEqual(caught.exception.category, 'out_of_bounds')
