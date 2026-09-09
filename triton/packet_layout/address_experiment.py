"""One frozen-layout compiler ablation, using the ordinary packet workflow."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from experiment_support import digest, load_graph

VARIANTS = ('ordinary', 'current', 'repaired', 'smaller', 'identity_legacy', 'identity_repaired')
WAVEFRONTS = 'l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum'


def reference(args):
    directory = args.address_reference.resolve() / args.platform / args.case / args.tau_name
    capture = json.loads((directory / 'capture.json').read_text())
    search = json.loads((directory / 'search.json').read_text())
    if search['selection_hash'] != digest({k: v for k, v in search.items()
                                          if k not in ('selection_hash', 'elapsed_seconds')}):
        raise ValueError('reference selection hash mismatch')
    if capture['capture_hash'] != search['capture_hash']:
        raise ValueError('reference capture/selection mismatch')
    if not any(c['runtime_layouts'] for c in search['candidates']):
        raise ValueError('address comparison requires a changed reference layout')
    return directory, capture, search


def check_capture(args, stamp):
    directory, old, _ = reference(args)
    for field in ('config', 'inputs', 'output_arguments', 'kernel_name'):
        if stamp[field] != old[field]:
            raise ValueError(f'frozen reference {field} changed; comparison would not isolate address generation')
    from experiment_support import HostTensor
    def launch_signature(path, record):
        captured = load_graph(path / 'capture.pkl.gz', record['capture_hash'])
        bound = captured['bound']
        return (captured['grid'], bound['__names__'],
                {index: value for index, value in bound.items()
                 if isinstance(index, int) and not isinstance(value, HostTensor)})
    if launch_signature(args.directory, stamp) != launch_signature(directory, old):
        raise ValueError('frozen reference scalar arguments or grid changed')
    for key in ('triton/tritonbench:revision', 'triton/tritonbench:patch'):
        if stamp['source_identity']['sources'].get(key) != old['source_identity']['sources'].get(key):
            raise ValueError('TritonBench kernel sources changed since the reference')


def selection(args, stamp, profile):
    from packet_search import select_candidates
    directory, old_capture, old = reference(args)
    if old['hardware_profile'] != profile.to_dict():
        raise ValueError('address ablation must preserve the reference objective')
    graph = load_graph(directory / 'graph.pkl.gz', old['graph_hash'])
    result = select_candidates(graph, profile, families=('split',))
    selected = next(c for c in old['candidates'] if c['id'] == old['analytical_top1'])
    if not selected['runtime_layouts']:
        raise ValueError('reference analytical selection is ordinary')
    smaller = next(c for c in result['candidates'] if c['id'] == result['analytical_top1'])
    if abs(smaller['score']['hardware_area'] - selected['score']['hardware_area']) > 1e-10:
        raise ValueError('new representative does not attain the reference objective value')
    ordinary = next(c for c in old['candidates'] if c['id'] == 'ordinary')
    candidates = [deepcopy(ordinary)]
    for label, mode, candidate in [('current', 'legacy', selected), ('repaired', 'repaired', selected),
                                    ('smaller', 'repaired', smaller)]:
        candidates.append({**deepcopy(candidate), 'id': label, 'address_mode': mode})
    # Keep identity specs present so the rewrite actually runs. Their inputs
    # are not packed, and timing aliases their outputs to the ordinary outputs.
    from relay import layout_matrix_rows, row_major_layout
    matrices = {m.name: m for m in graph.matrices}
    identity = deepcopy(selected['runtime_layouts'])
    for layout in identity:
        matrix = matrices[layout['name']]
        layout['rows'] = list(layout_matrix_rows(matrix, row_major_layout(matrix)))
    for label, mode in [('identity_legacy', 'legacy'), ('identity_repaired', 'repaired')]:
        candidates.append({**deepcopy(ordinary), 'id': label, 'runtime_layouts': identity,
                           'address_mode': mode, 'same_pointer': True})
    result.update(candidates=candidates, analytical_top1='repaired', maximum_realized_candidates=6,
        address_reference={'capture_hash': old_capture['capture_hash'], 'selection_hash': old['selection_hash'],
                           'directory': str(directory), 'same_map_comparison': ['current', 'repaired'],
                           'representative_comparison': ['repaired', 'smaller']})
    return graph, result


def summarize(report):
    values = report['evaluation']['candidates']
    def ratio(a, b):
        if a not in values or b not in values: return None
        left, right = (values[label]['selected']['process_medians_ms'] for label in (a, b))
        if len(left) != len(right):
            raise ValueError('address comparisons require paired processes')
        return math.exp(sum(math.log(x / y) for x, y in zip(left, right)) / len(left))
    return {'status': 'complete' if set(VARIANTS) <= values.keys() else 'incomplete: inspect rejected variants',
            'same_layout_repair_speedup': ratio('current', 'repaired'),
            'smaller_vs_repaired_speedup': ratio('repaired', 'smaller'),
            'note': 'Geometric means of paired process ratios; inspect process and shared-input placement results.'}


def profile_worker(args):
    """Warm one candidate, then expose exactly one dispatch to cudaProfilerStart."""
    import torch
    from layout_runtime import freeze_launch
    from packet_runtime import statistics
    from packet_workflow import (CASES, accepted_candidates, context, outputs_correct, prepared,
                                 read, verify, verify_inputs, write_json)
    stamp = verify(args, gpu=True)
    candidates = accepted_candidates(args, read(args, 'search.json'))
    candidate = next(c for c in candidates if c['id'] == args.profile_candidate)
    source = freeze_launch(CASES[args.case].factory(), stamp['config'])
    verify_inputs(source, stamp)
    source.run()
    launch = prepared(args, candidate, stamp=stamp, native=source)
    with context(candidate, launch=launch)():
        kernel = launch.run()
        outputs_correct(source, launch, stamp['output_arguments'])
        if candidate.get('same_pointer'):
            for index in stamp['output_arguments']:
                launch.values[index] = source.values[index]
        for _ in range(args.warmup * args.iterations):
            kernel = launch.run()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        kernel = launch.run()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    codegen = statistics(kernel, args.directory / 'profile/codegen', candidate['id'])
    timed = read(args, 'evaluate/process-0.json')['codegen'][candidate['id']]
    if codegen['artifacts']['cubin']['sha256'] != timed['artifacts']['cubin']['sha256']:
        raise ValueError('profiled binary differs from the unprofiled graph-captured binary')
    write_json(args.worker_output, {'candidate': candidate['id'], 'codegen': codegen,
                                   'matches_timed_binary': True})


def parse_counters(path, kernel_name):
    from stage1_nvidia_counter_analysis import COUNTER_METRICS, _records, _number
    records = [r for r in _records(path) if kernel_name in r.get('Kernel Name', '')]
    if len(records) != 1:
        raise ValueError(f'expected one profiled target dispatch, got {len(records)}: {path}')
    counts = {name: _number(records[0], name, path) for name in (*COUNTER_METRICS, WAVEFRONTS)}
    if any(not math.isfinite(n) or n < 0 for n in counts.values()):
        raise ValueError(f'invalid absolute counter values: {path}')
    return counts


def profile(args):
    """A resumable, separate H100 counter pass over the frozen six-variant panel."""
    from _bootstrap import HERE, ROOT
    from packet_workflow import accepted_candidates, read, verify, write_json
    from stage1_nvidia_counter_analysis import COUNTER_METRICS, counter_definitions
    stamp = verify(args, gpu=True)
    search = read(args, 'search.json')
    candidates = accepted_candidates(args, search)
    executable = shutil.which('ncu')
    if not executable:
        raise FileNotFoundError('ncu is not on PATH; load Nsight Compute and retry this stage with --retry-failed')
    version = subprocess.check_output([executable, '--version'], text=True)
    metrics = (*COUNTER_METRICS, WAVEFRONTS)
    timed_path = args.directory / 'evaluate/process-0.json'
    binding = digest({'selection': search['selection_hash'],
        'validation': read(args, 'validated.json')['validation_hash'],
        'evaluation': read(args, 'evaluation.json'), 'timed_binary_record': timed_path.read_text(),
        'warmup_launches': args.warmup * args.iterations, 'ncu': version, 'metrics': metrics})
    directory = args.directory / 'profile'
    directory.mkdir(exist_ok=True)
    records = {}
    for candidate in candidates:
        label = candidate['id']
        csv_path, worker_path, checkpoint = (directory / f'{label}{suffix}'
                                             for suffix in ('.csv', '.json', '.complete.json'))
        def hashes():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv_path, worker_path)}
        if args.resume and checkpoint.exists() and csv_path.exists() and worker_path.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['binding'] == binding and saved['outputs'] == hashes():
                records[label] = saved['counters']
                continue
        command = [executable, '--csv', '--page', 'raw', '--print-units', 'base',
            '--metrics', ','.join(metrics), '--profile-from-start', 'off',
            '--replay-mode', 'application', '--cache-control', 'none', '--clock-control', 'none',
            '--target-processes', 'application-only', '--log-file', str(csv_path),
            sys.executable, str(HERE / 'run.py'), '--stage', 'worker', '--phase', 'profile',
            '--profile-candidate', label, '--platform', args.platform, '--case', args.case,
            '--root', str(args.root), '--tau-name', args.tau_name,
            '--address-reference', str(args.address_reference.resolve()),
            '--grammar', 'split', '--selection', 'analytical', '--warmup', str(args.warmup),
            '--iterations', str(args.iterations), '--worker-output', str(worker_path)]
        # A failed or interrupted ncu invocation is retried; complete variants are retained.
        csv_path.unlink(missing_ok=True)
        subprocess.run(command, check=True, cwd=ROOT)
        records[label] = parse_counters(csv_path, stamp['kernel_name'])
        write_json(checkpoint, {'binding': binding, 'outputs': hashes(), 'counters': records[label]})
    definitions = counter_definitions()
    definitions[WAVEFRONTS] = 'global-load wavefronts sent from the L1TEX tag stage to the data stage'
    write_json(args.directory / 'profile.json', {
        'schema': 'laqs.packet.address-counters.v1', 'binding': binding, 'ncu': version,
        'status': 'complete' if set(VARIANTS) <= records.keys() else 'incomplete: rejected variants',
        'candidates': records, 'definitions': {name: definitions[name] for name in metrics},
        'cache_regime': f'one dispatch after {args.warmup * args.iterations} same-candidate launches; '
                        'application replay repeats warmup each pass; no profiler cache flush or clock control',
        'interpretation': 'Whole-kernel absolute counters, not per-operand counts. Duration is profiled and '
                          'must not be used as speedup. The 128-byte issue footprint is a locality/service '
                          'proxy, not the L1TEX global-load request count or a cache-miss prediction.'})
