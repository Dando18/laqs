#!/usr/bin/env python3
"""Staged packet-layout workflow with a module name distinct from legacy runners."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import fcntl
import hashlib
import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from _bootstrap import ROOT, HERE, activate
from experiment_support import (compiler_source_identity, digest, host_arguments, load_graph,
                                save_graph, source_identity)
from packet_cases import CASES


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def identity():
    sources = dict(source_identity()['sources'])
    for suffix in ('*.py', '*.cpp', '*.h', 'CMakeLists.txt'):
        for path in HERE.rglob(suffix):
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'protocol': 'laqs.packet.v1', 'source_hash': digest(sources), 'sources': sources}


def args_parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['capture', 'search', 'validate', 'tune', 'evaluate',
                                         'diagnose', 'conventional', 'worker', 'report', 'all'], required=True)
    parser.add_argument('--platform', choices=['tuolumne', 'matrix'], required=True)
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--root', type=Path, default=ROOT / 'triton/experiments/results/packet-v1')
    parser.add_argument('--tau-name', choices=['expert', 'l1_to_l2', 'speedup'], default='expert')
    parser.add_argument('--tau-profile', type=Path, default=ROOT / 'triton/experiments/tau-profiles.json')
    parser.add_argument('--selection', choices=['analytical', 'measured'], default='measured')
    parser.add_argument('--grammar', choices=['split', 'split-chunks'], default='split-chunks')
    parser.add_argument('--resume', action='store_true', help='reuse exact CPU and timing checkpoints')
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--samples', type=int, default=21)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--minimum-gain', type=float, default=.01)
    parser.add_argument('--max-trace-contexts', type=int, default=1 << 24)
    parser.add_argument('--max-events', type=int, default=1 << 20)
    parser.add_argument('--cpu-workers', type=int, default=1,
                        help='CPU processes for tracing and graph construction; reserve this many cores')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--phase', choices=['tune', 'evaluate', 'diagnose', 'conventional-tune', 'conventional-evaluate'], default='evaluate')
    parser.add_argument('--process-index', type=int, default=0)
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--diagnostic-selection', type=Path,
                        help='v2 proposed-selection.json for a same-map generic/structured diagnostic')
    args = parser.parse_args(argv)
    if min(args.processes, args.samples, args.iterations, args.warmup) <= 0:
        parser.error('timing counts must be positive')
    if args.cpu_workers < 1:
        parser.error('CPU worker count must be positive')
    if not 0 < args.minimum_gain < 1:
        parser.error('minimum gain must lie strictly between zero and one')
    args.root = args.root.resolve()
    args.directory = args.root / args.platform / args.case / args.tau_name
    return args


def read(args, name):
    data = json.loads((args.directory / name).read_text())
    signed = {'search.json': ('selection_hash', {'elapsed_seconds'}),
              'choice.json': ('choice_hash', set()),
              'conventional-choice.json': ('choice_hash', set())}
    if name in signed:
        field, excluded = signed[name]
        actual = digest({key: value for key, value in data.items() if key not in excluded | {field}})
        if data.get(field) != actual:
            raise ValueError(f'{name} content hash mismatch')
    return data


def input_probe(tensor):
    import torch
    flat = tensor.contiguous().view(torch.uint8).flatten()
    count = min(1024, flat.numel())
    indices = torch.arange(count, dtype=torch.int64, device=flat.device) * (flat.numel() - 1) // max(1, count - 1)
    return hashlib.sha256(flat[indices].cpu().numpy().tobytes()).hexdigest()


def verify_inputs(launch, stamp):
    for record in stamp['inputs']:
        tensor = launch.values[record['argument']]
        if (list(tensor.shape) != record['shape'] or list(tensor.stride()) != record['strides']
                or str(tensor.dtype) != record['dtype'] or input_probe(tensor) != record['probe_sha256']):
            raise ValueError(f"input changed since capture: {record['name']}")


def runtime_identity():
    import torch
    import triton
    from packet_runtime import plugin_path
    from layout_runtime import _plugin_path
    from relay.triton_frontend import _default_plugin_path
    return {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'triton': triton.__version__,
            'runtime': torch.version.hip or torch.version.cuda,
            'compiler': compiler_source_identity(triton.__file__),
            'plugins': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (plugin_path(), _plugin_path(), _default_plugin_path())}}


def verify(args, gpu=False):
    stamp = read(args, 'capture.json')
    if stamp['source_identity'] != identity():
        raise ValueError('packet suite sources changed; capture in a fresh result root')
    if gpu and stamp['runtime_identity'] != runtime_identity():
        raise ValueError('GPU/compiler/plugins changed since packet capture')
    return stamp


def capture(args):
    import torch
    from relay import AnalysisOptions, analyze_launch
    from relay.triton_frontend import MANIFEST_METADATA_KEY, infer_allocations
    from layout_runtime import freeze_launch, unwrap_jit
    from packet_cases import reference
    path = args.directory / 'capture.json'
    if path.exists() and not args.rerun:
        verify(args, gpu=True)
        return
    started = perf_counter()
    spec = CASES[args.case].factory()
    pinned = None
    if args.diagnostic_selection:
        pinned = json.loads(args.diagnostic_selection.read_text())
        if pinned['platform'] != args.platform or f"{pinned['operator']}--{pinned['config']}" != args.case:
            raise ValueError('diagnostic capture case/device mismatch')
        launch = freeze_launch(spec, pinned['selected_config'])
        analysis = analyze_launch(launch.jit, launch.grid, *launch.values,
                                  _laqs_options=AnalysisOptions(evaluate=False), **launch.options)
    else:
        analysis = analyze_launch(spec.kernel, spec.grid, *spec.args,
                                  _laqs_options=AnalysisOptions(evaluate=False), **spec.kwargs)
    if not analysis.supported:
        raise ValueError(f'{analysis.unsupported.category}: {analysis.unsupported.message}')
    config = dict(pinned['selected_config'] if pinned else analysis.selected_config)
    for key in ['num_warps', 'num_stages', 'num_ctas', 'maxnreg', 'waves_per_eu', 'matrix_instr_nonkdim', 'kpack']:
        value = getattr(analysis.compiled_kernel.metadata, key, None)
        if value is not None:
            config[key] = value
    frozen = freeze_launch(spec, config)
    frozen.run()
    validation = reference(spec.operator, frozen)
    allocations = infer_allocations(analysis.manifest, analysis.bound_arguments)
    names = {str(name): int(index) for index, name in analysis.bound_arguments['__names__'].items()}
    outputs = sorted({a.argument if isinstance(a.argument, int) else names[a.argument]
                      for a in allocations if a.role != 'read' and not a.path})
    capture_hash = save_graph(args.directory / 'capture.pkl.gz', {
        'manifest': getattr(analysis.compiled_kernel.metadata, MANIFEST_METADATA_KEY),
        'bound': host_arguments(analysis.bound_arguments), 'grid': analysis.grid, 'selected_config': config})
    inputs = []
    for a in allocations:
        if a.role != 'read' or a.path:
            continue
        t = analysis.bound_arguments[a.argument]
        inputs.append({'name': a.name, 'shape': a.true_shape, 'strides': a.strides,
                       'argument': a.argument if isinstance(a.argument, int) else names[a.argument], 'dtype': str(t.dtype),
                       'logical_bytes': prod(a.true_shape) * a.element_bytes,
                       'envelope_bytes': prod(a.envelope_shape) * a.element_bytes,
                       'probe_sha256': input_probe(t)})
    write_json(path, {'schema': 'laqs.packet.capture.v1', 'source_identity': identity(),
                     'runtime_identity': runtime_identity(), 'capture_hash': capture_hash,
                     'config': config, 'output_arguments': outputs, 'inputs': inputs, 'seed': 0,
                     'kernel_name': unwrap_jit(spec.kernel).fn.__name__, 'validation': validation,
                     'capture_seconds': perf_counter() - started})


def search(args):
    from relay import AnalysisOptions, EvaluationLimits
    from relay.triton_frontend import analyze_compiled_manifest
    from packet_search import select_candidates
    from search_algorithms import load_tau_profile
    timings = {'cpu_workers': args.cpu_workers}

    def completed(name, seconds):
        timings[name] = seconds
        write_json(args.directory / 'search-timings.json', timings)
        print(f'CPU {name}: {seconds:.3f}s', flush=True)

    stamp = verify(args)
    phase_started = perf_counter()
    captured = load_graph(args.directory / 'capture.pkl.gz', stamp['capture_hash'])
    completed('capture_load', perf_counter() - phase_started)
    profile = load_tau_profile(args.platform, args.tau_profile, args.tau_name)
    started = perf_counter()
    print(f'CPU trace and graph construction started (up to {args.cpu_workers} workers)', flush=True)
    cache_key = digest({'capture': stamp['capture_hash'], 'source': stamp['source_identity'],
                        'profile': profile.to_dict(), 'contexts': args.max_trace_contexts,
                        'events': args.max_events})
    ready_path = args.directory / 'graph-ready.json'
    ready = json.loads(ready_path.read_text()) if args.resume and ready_path.exists() else None
    if ready is not None and ready['key'] == cache_key:
        analysis = load_graph(args.directory / 'graph.pkl.gz', ready['graph_hash'])
        graph_hash = ready['graph_hash']
        print('CPU graph restored from checkpoint', flush=True)
    else:
        checkpoint = args.directory / 'search-checkpoints' / cache_key if args.resume else None
        analysis = analyze_compiled_manifest(None, captured['manifest'], captured['grid'], captured['bound'],
            selected_config=captured['selected_config'], options=AnalysisOptions(hardware_profile=profile,
            require_native_baseline=True, limits=EvaluationLimits(max_trace_contexts=args.max_trace_contexts,
                max_dynamic_events=args.max_events, workers=args.cpu_workers,
                checkpoint_dir=str(checkpoint) if checkpoint else None)), on_phase=completed)
        if not analysis.supported:
            raise ValueError(f'{analysis.unsupported.category}: {analysis.unsupported.message}')
        analysis = replace(analysis, compiled_kernel=None, manifest=None,
                           bound_arguments={'__names__': analysis.bound_arguments['__names__']})
        phase_started = perf_counter()
        graph_hash = save_graph(args.directory / 'graph.pkl.gz', analysis)
        write_json(ready_path, {'key': cache_key, 'graph_hash': graph_hash})
        completed('graph_write', perf_counter() - phase_started)
    phase_started = perf_counter()
    print('CPU layout selection started', flush=True)
    result = select_candidates(analysis, profile,
                               families=('split',) if args.grammar == 'split' else ('split', 'chunks'))
    completed('selection', perf_counter() - phase_started)
    result['graph_hash'] = graph_hash
    result['capture_hash'] = stamp['capture_hash']
    result['source_identity'] = stamp['source_identity']
    result['hardware_profile'] = profile.to_dict()
    result['tau_sha256'] = hashlib.sha256(args.tau_profile.read_bytes()).hexdigest()
    result['elapsed_seconds'] = perf_counter() - started
    completed('total', result['elapsed_seconds'] + timings['capture_load'])
    result['selection_hash'] = digest({k: v for k, v in result.items() if k != 'elapsed_seconds'})
    write_json(args.directory / 'search.json', result)


def prepared(args, candidate, *, stamp=None, native=None):
    from layout_runtime import freeze_launch, fresh_outputs, replace_inputs
    from packet_runtime import runtime_layouts
    stamp = stamp or verify(args, gpu=True)
    launch = native or freeze_launch(CASES[args.case].factory(), stamp['config'])
    return fresh_outputs(replace_inputs(launch, runtime_layouts(candidate)), stamp['output_arguments'])


def context(candidate, *, inspect=False):
    from packet_runtime import runtime_layouts, structured_layouts
    from layout_runtime import rewrite_layouts
    layouts = runtime_layouts(candidate)
    if candidate.get('realization') == 'generic':
        return lambda: rewrite_layouts(layouts)
    if not layouts:
        return nullcontext
    return lambda: structured_layouts(layouts, inspect=inspect)


def outputs_correct(ordinary, selected, outputs):
    from reference_validation import comparison
    checks = [comparison(selected.values[index], ordinary.values[index]) for index in outputs]
    if not all(check['allclose'] for check in checks):
        raise ValueError(f'transformed numerical comparison failed: {checks}')
    return checks


def diagnostic_candidate(args, search_record):
    if args.diagnostic_selection:
        data = json.loads(args.diagnostic_selection.read_text())
        if data['platform'] != args.platform or f"{data['operator']}--{data['config']}" != args.case:
            raise ValueError('diagnostic selection case/device mismatch')
        if data['selected_config'] != read(args, 'capture.json')['config']:
            raise ValueError('same-layout diagnostic requires the captured frozen configuration')
        return {'id': 'structured', 'runtime_layouts': data['runtime_layouts'],
                'reference_selection_sha256': hashlib.sha256(args.diagnostic_selection.read_bytes()).hexdigest()}
    candidate = next(c for c in search_record['candidates'] if c['id'] == search_record['analytical_top1'])
    if not candidate['runtime_layouts']:
        raise ValueError('analytical top-1 is ordinary; supply a changed --diagnostic-selection')
    return {**candidate, 'id': 'structured'}


def selection_record(args, diagnostic=False):
    if diagnostic and args.diagnostic_selection:
        candidate = diagnostic_candidate(args, None)
        return {'selection_hash': digest({'candidate': candidate, 'capture': read(args, 'capture.json')['capture_hash']}),
                'candidates': [candidate], 'analytical_top1': 'structured'}
    return read(args, 'search.json')


def validate(args):
    import torch
    from packet_cases import reference
    from packet_runtime import structured_layouts, runtime_layouts, statistics, primitive_rejections
    from layout_runtime import freeze_launch, fresh_outputs
    from packet_measure import packing_cost
    stamp = verify(args, gpu=True)
    search_record = read(args, 'search.json')
    candidate_records = []
    source = freeze_launch(CASES[args.case].factory(), stamp['config'])
    verify_inputs(source, stamp)
    for candidate in search_record['candidates']:
        record = {**candidate, 'accepted': False}
        started = perf_counter()
        try:
            native = fresh_outputs(source, stamp['output_arguments'])
            selected = prepared(args, candidate, stamp=stamp, native=source)
            torch.cuda.synchronize()
            record['packing_and_allocation_seconds'] = perf_counter() - started
            record['packing'] = packing_cost(source, runtime_layouts(candidate))
            with structured_layouts(runtime_layouts(candidate), inspect=True):
                baseline_kernel = native.run()
            with context(candidate)():
                selected_kernel = selected.run()
            torch.cuda.synchronize()
            reference(CASES[args.case].operator, native)
            record['numerical_checks'] = outputs_correct(native, selected, stamp['output_arguments'])
            directory = args.directory / 'codegen' / candidate['id']
            baseline_stats = statistics(baseline_kernel, directory, 'baseline')
            selected_stats = statistics(selected_kernel, directory, 'structured')
            record['codegen'] = {'baseline': baseline_stats, 'structured': selected_stats}
            record['rejections'] = primitive_rejections(baseline_stats, selected_stats) if candidate['runtime_layouts'] else []
            record['accepted'] = not record['rejections']
        except Exception as error:
            record['rejections'] = [f'{type(error).__name__}: {error}']
        record['validation_seconds'] = perf_counter() - started
        candidate_records.append(record)
        write_json(args.directory / 'validated.json', {'selection_hash': search_record['selection_hash'],
                                                      'candidates': candidate_records, 'complete': False})
    if not candidate_records[0]['accepted']:
        raise ValueError(f"ordinary kernel validation failed: {candidate_records[0]['rejections']}")
    validation = {'selection_hash': search_record['selection_hash'], 'candidates': candidate_records, 'complete': True}
    validation['validation_hash'] = digest(validation)
    write_json(args.directory / 'validated.json', validation)


def accepted_candidates(args, search_record):
    validation = read(args, 'validated.json')
    if (not validation['complete'] or validation['selection_hash'] != search_record['selection_hash']
            or validation.get('validation_hash') != digest({k: v for k, v in validation.items() if k != 'validation_hash'})):
        raise ValueError('candidate validation incomplete, changed or stale')
    return [c for c in validation['candidates'] if c['accepted']]


def worker(args):
    if args.phase.startswith('conventional-'):
        from conventional import worker as conventional_worker
        return conventional_worker(args)
    import torch
    from packet_measure import graph_samples
    from packet_runtime import statistics
    from layout_runtime import freeze_launch
    from packet_cases import reference
    stamp = verify(args, gpu=True)
    search_record = selection_record(args, diagnostic=args.phase == 'diagnose')
    if args.phase == 'diagnose':
        selected = diagnostic_candidate(args, search_record)
        candidates = [{'id': 'ordinary', 'runtime_layouts': []}, selected,
                      {**selected, 'id': 'generic', 'realization': 'generic'}]
    else:
        candidates = accepted_candidates(args, search_record)
        if len(candidates) > 3:
            raise ValueError('measured deployment exceeds three candidates')
    launches, contexts, codegen, packing, failures = {}, {}, {}, {}, {}
    source = freeze_launch(CASES[args.case].factory(), stamp['config'])
    verify_inputs(source, stamp)
    source.run()
    reference(CASES[args.case].operator, source)
    for candidate in candidates:
        try:
            started = perf_counter()
            launch = prepared(args, candidate, stamp=stamp, native=source)
            torch.cuda.synchronize()
            packing[candidate['id']] = perf_counter() - started
            with context(candidate)():
                kernel = launch.run()
            outputs_correct(source, launch, stamp['output_arguments'])
        except Exception as error:
            if args.phase != 'diagnose' or candidate['id'] == 'ordinary':
                raise
            failures[candidate['id']] = f'{type(error).__name__}: {error}'
            continue
        launches[candidate['id']] = launch
        contexts[candidate['id']] = context(candidate)
        codegen[candidate['id']] = statistics(kernel, args.directory / args.phase / f'process-{args.process_index}-codegen', candidate['id'])
    timings = graph_samples(launches, contexts, stamp['output_arguments'], process_index=args.process_index,
                           samples=args.samples, iterations=args.iterations, warmup=args.warmup)
    write_json(args.worker_output, {'schema': 'laqs.packet.timing.v1', 'phase': args.phase,
                                   'selection_hash': search_record['selection_hash'],
                                   'process_index': args.process_index, 'timings': timings,
                                   'codegen': codegen, 'failures': failures,
                                   'packing_and_allocation_seconds': packing})


def run_processes(args, phase):
    from packet_measure import summarize
    if args.resume:
        verify(args, gpu=True)
    records = []
    for index in range(args.processes):
        output = args.directory / phase / f'process-{index}.json'
        checkpoint = output.with_suffix('.complete.json')
        binding = None
        if args.resume and phase in ('tune', 'evaluate'):
            binding = digest({'selection': read(args, 'search.json')['selection_hash'],
                'validation': read(args, 'validated.json')['validation_hash'],
                'choice': read(args, 'choice.json')['choice_hash'] if phase == 'evaluate' else None,
                'phase': phase, 'index': index,
                'measurement': [args.samples, args.iterations, args.warmup]})
            if checkpoint.exists() and output.exists():
                saved = json.loads(checkpoint.read_text())
                if saved == {'binding': binding, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}:
                    records.append(json.loads(output.read_text()))
                    continue
        command = [sys.executable, str(HERE / 'run.py'), '--stage', 'worker', '--phase', phase,
                   '--platform', args.platform, '--case', args.case, '--root', str(args.root),
                   '--tau-name', args.tau_name, '--samples', str(args.samples), '--iterations', str(args.iterations),
                   '--warmup', str(args.warmup), '--process-index', str(index), '--worker-output', str(output)]
        if args.diagnostic_selection:
            command += ['--diagnostic-selection', str(args.diagnostic_selection.resolve())]
        subprocess.run(command, cwd=ROOT, check=True)
        if binding is not None:
            write_json(checkpoint, {'binding': binding, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()})
        records.append(json.loads(output.read_text()))
    return summarize(records)


def tune(args):
    from packet_measure import measured_choice
    verify(args, gpu=True)
    search_record = read(args, 'search.json')
    accepted = {c['id'] for c in accepted_candidates(args, search_record)}
    tuning = None
    if args.selection == 'measured':
        tuning = run_processes(args, 'tune')
        chosen = measured_choice(tuning, minimum_gain=args.minimum_gain)
    else:
        chosen = search_record['analytical_top1'] if search_record['analytical_top1'] in accepted else 'ordinary'
    choice = {'schema': 'laqs.packet.choice.v1', 'method': args.selection, 'selected': chosen,
              'analytical_top1': search_record['analytical_top1'], 'tuning': tuning,
              'selection_hash': search_record['selection_hash'], 'minimum_gain': args.minimum_gain,
              'measurement': [args.processes, args.samples, args.iterations, args.warmup],
              'evaluation_policy': 'freeze choice before fresh independent evaluation processes'}
    choice['choice_hash'] = digest(choice)
    write_json(args.directory / 'choice.json', choice)


def evaluate(args):
    choice = read(args, 'choice.json')
    search_record = read(args, 'search.json')
    if choice['selection_hash'] != search_record['selection_hash']:
        raise ValueError('selection changed after tuning')
    evaluation = run_processes(args, 'evaluate')
    if read(args, 'choice.json') != choice:
        raise ValueError('choice changed during held-out evaluation')
    write_json(args.directory / 'evaluation.json', {'choice_hash': choice['choice_hash'], 'candidates': evaluation})
    report(args)


def diagnose(args):
    summary = run_processes(args, 'diagnose')
    write_json(args.directory / 'diagnosis.json', {'schema': 'laqs.packet.same-map.v1',
               'mapping': diagnostic_candidate(args, selection_record(args, diagnostic=True)),
               'timings': summary, 'failures': [read(args, f'diagnose/process-{i}.json')['failures'] for i in range(args.processes)],
               'deployment_selection': False})


def conventional(args):
    from conventional import run
    run(args)


def report(args):
    from packet_measure import conversion_result
    choice, evaluation, search_record = (read(args, name) for name in ('choice.json', 'evaluation.json', 'search.json'))
    if evaluation['choice_hash'] != choice['choice_hash']:
        raise ValueError('evaluation does not belong to frozen choice')
    selected = choice['selected']
    timing = evaluation['candidates'][selected]
    validation = read(args, 'validated.json')
    record = next(c for c in validation['candidates'] if c['id'] == selected)
    conversion = conversion_result(timing['baseline']['median_ms'], timing['selected']['median_ms'], record['packing']['median_ms'])
    payload = {'schema': 'laqs.packet.report.v1', 'case': args.case, 'platform': args.platform,
               'source_identity': identity(), 'choice': choice, 'evaluation': evaluation,
               'search': search_record, 'realization': validation,
               'conversion': conversion}
    write_json(args.directory / 'report.json', payload)
    lines = [f'# Packet-layout result: {args.case} / {args.platform}', '',
             f"Analytical top-1: `{choice['analytical_top1']}`. {choice['method'].capitalize()} selection: `{selected}`.",
             'Evaluation uses fresh processes after the choice is frozen. Packing is excluded from kernel timings.', '',
             '| Candidate | Primitive validation | LAQS J | Evaluation speedup |',
             '| --- | --- | ---: | ---: |']
    for candidate in validation['candidates']:
        timing = evaluation['candidates'].get(candidate['id'])
        speedup = f"{timing['speedup']:.5f}×" if timing else 'not evaluated'
        status = 'passed' if candidate['accepted'] else '; '.join(candidate['rejections'])
        lines.append(f"| {candidate['id']} | {status} | {candidate['score']['hardware_area']:.6g} | {speedup} |")
    lines += ['', f"Measured packing/allocation: {record['packing']['median_ms']:.6g} ms. One use including conversion: {conversion['one_use_including_conversion_speedup']:.5f}×.",
              f"Estimated reuse count to amortize packing: {conversion['break_even_reuses'] or 'no positive estimate'}.",
              'See report.json for raw codegen, confidence intervals, controls, and per-array search evidence.']
    (args.directory / 'analysis.md').write_text('\n'.join(lines) + '\n')


def main():
    args = args_parser()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.stage not in ['search', 'report']:
        activate(args.platform)
    if args.stage == 'worker':
        worker(args)
        return
    stages = ['capture', 'search', 'validate', 'tune', 'evaluate'] if args.stage == 'all' else [args.stage]
    if args.stage == 'all' and args.case.startswith('row_column--'):
        stages.append('conventional')
    with (args.directory / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for stage in stages:
            status_path = args.directory / 'status.json'
            if status_path.exists() and stage != 'capture' and not args.resume:
                previous = json.loads(status_path.read_text())
                if previous['status'] == 'running' and previous['stage'] != stage:
                    previous.update(status='failed', reason='stage did not complete; inspect its scheduler log for timeout or interruption')
                    write_json(status_path, previous)
                if previous['status'] == 'failed' and previous['stage'] != stage:
                    raise ValueError(f"blocked by prior {previous['stage']} failure: {previous['reason']}")
            write_json(args.directory / 'status.json', {'status': 'running', 'stage': stage})
            try:
                globals()[stage](args)
            except Exception as error:
                write_json(args.directory / 'status.json', {'status': 'failed', 'stage': stage,
                           'reason': f'{type(error).__name__}: {error}'})
                raise
            write_json(args.directory / 'status.json', {'status': 'complete', 'stage': stage})


if __name__ == '__main__':
    main()
