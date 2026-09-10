"""Unpruned, fixed-schedule empirical layout selection with held-out evaluation."""
from collections import defaultdict
import json
import subprocess
import sys

from _bootstrap import HERE
from experiment_support import digest, load_graph
from packet_workflow import read, verify, write_json


def panel(analysis, profile, case, *, checkpoint_dir=None):
    """Enumerate physical maps; scalar, feature and partial-flag ties survive."""
    from relay import CanonicalLayout, layout_matrix_rows, row_major_layout, score_layouts
    from relay.prepared_scoring import PreparedRegionScorer
    from layout_contract import RuntimeLayout, preserves_vector_bits
    from manual_layout import split_tile
    from packet_compatibility import protected_bits
    from packet_search import address_fields, partial_flag, split_templates, supported_allocation
    from search_algorithms import _score_dict
    if profile.resource_maps:
        raise ValueError('split study requires independent array component scores')
    matrices = {m.name: m for m in analysis.matrices}
    allocations = {a.name: a for a in analysis.allocations}
    baseline = {name: row_major_layout(m) for name, m in matrices.items()}
    names = {str(name): int(index) for index, name in analysis.bound_arguments['__names__'].items()}
    targets = [m for m in matrices.values() if m.target and
               (allocations[m.name].argument == 0 or names.get(allocations[m.name].argument) == 0)]
    if len(targets) != 1:
        raise ValueError('study requires exactly one eligible argument-0 matrix')
    matrix = targets[0]
    allocation = allocations[matrix.name]
    bits = protected_bits(analysis.events).get(matrix.name, 0)
    reason = supported_allocation(matrix, allocation, bits)
    if reason:
        raise ValueError(reason)
    if case == 'gemm--asymmetric':
        if matrix.shape != (512, 4096) or matrix.element_bytes != 2 or bits != 4:
            raise ValueError('GEMM panel requires H100 sector-protected FP16 A[512,4096]')
        n, m = matrix.mode_bits
        layouts = [baseline[matrix.name]] + [CanonicalLayout(f'split-a{a}-v{v}', matrix.name,
            matrix.mode_bits, (1,) * v + (0,) * a + (1,) * (m-v) + (0,) * (n-a), (1, 0))
            for a in range(1, 10) for v in (4, 5)]
        domain = {'TI': [2**a for a in range(1, 10)], 'TJ': [16, 32], 'ordinary': True}
    else:
        layouts = list(split_templates(matrix, bits))
        domain = {'TI': [2**a for a in range(matrix.mode_bits[0]+1)],
                  'TJ': [2**v for v in range(bits, matrix.mode_bits[1]+1)], 'ordinary': True}
    ordinary = layout_matrix_rows(matrix, baseline[matrix.name])
    depths = sorted({bits, *(min(matrix.total_bits, (s // matrix.element_bytes).bit_length()-1)
                             for s in profile.byte_scales)})
    prepared = PreparedRegionScorer(matrices, analysis.components)
    cache, candidates, seen, ranking, modeled_counts = {}, [], set(), {}, {}
    for layout in layouts:
        rows = layout_matrix_rows(matrix, layout)
        if rows in seen:
            continue
        seen.add(rows)
        if not preserves_vector_bits(rows, ordinary, bits):
            raise ValueError('panel map violates native compatibility contract')
        runtime = RuntimeLayout(matrix.name, 0, allocation.true_shape, allocation.strides,
                                allocation.envelope_shape, rows)
        tile = split_tile(runtime)
        label = 'ordinary' if rows == ordinary else f'tile-{tile[0]}x{tile[1]}'
        selected = {**baseline, matrix.name: layout}
        flag = partial_flag(rows, depths)
        flag_by_depth = dict(flag)
        checkpoint = checkpoint_dir / (digest(rows)+'.json') if checkpoint_dir else None
        saved = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else {'counts': {}}
        if 'hash' in saved and saved['hash'] != digest({k: v for k, v in saved.items() if k != 'hash'}):
            raise ValueError('analytical component checkpoint changed')
        for component in analysis.components:
            modeled_key = (component.name, flag_by_depth[component.dimension(matrix)])
            resumed = component.name in saved['counts']
            if resumed:
                for name, value in saved['counts'][component.name].items():
                    cache[name, selected[name].signature(), component.name] = tuple(value)
            else:
                # Reuse exact quotient counts, never remove a physical candidate.
                # This avoids repeating expensive edge enumeration for equal
                # partitions at this component's scale.
                if modeled_key in modeled_counts:
                    cache[matrix.name, layout.signature(), component.name] = modeled_counts[modeled_key]
                prepared.populate((component,), selected, cache)
            saved['counts'][component.name] = {name: cache[name, selected[name].signature(), component.name]
                for name, edges in component.edges_by_array.items() if edges}
            if matrix.name in saved['counts'][component.name]:
                modeled_counts[modeled_key] = tuple(saved['counts'][component.name][matrix.name])
            if checkpoint and not resumed:
                saved['hash'] = digest({k: v for k, v in saved.items() if k != 'hash'})
                write_json(checkpoint, saved)
        score = _score_dict(score_layouts(matrices, analysis.components, selected,
                            hardware_profile=profile, array_component_cache=cache))
        candidates.append({'id': label, 'runtime_layouts': [] if label == 'ordinary' else [runtime.to_dict()],
            'physical_rows': rows, 'tile': tile, 'score': score, 'partial_flag': flag,
            'scalar_class': digest(score['hardware_area']), 'feature_class': digest(score['components']),
            'flag_class': digest(flag), 'address_fields': [vars(f) for f in address_fields(matrix, layout)]})
        ranking[label] = (score['hardware_area'], rows != ordinary,
                         sum(a != b for a, b in zip(rows, ordinary)), len(address_fields(matrix, layout)), rows)
        if checkpoint:
            print(f'CPU panel {len(candidates)}/{len(layouts)}: {label}, J={score["hardware_area"]:.6g}', flush=True)
    return {'schema': 'laqs.layout.study.panel.v1', 'candidates': candidates,
            'analytical_top1': min(ranking, key=ranking.get), 'domain': domain,
            'varied_argument': 0, 'protected_element_bits': bits, 'flag_depths': depths,
            'deduplication': 'identical physical maps only; no score or flag pruning',
            'selection_scope': 'analytical top-1 and empirical winner within this declared panel',
            'batch_size': 4, 'poor_speedup_threshold': .9}


def search(args):
    from address_experiment import reference, check_capture
    from packet_compatibility import constrain_graph
    from search_algorithms import load_tau_profile
    stamp = verify(args)
    check_capture(args, stamp)
    directory, old_capture, old = reference(args)
    profile = load_tau_profile(args.platform, args.tau_profile, args.tau_name)
    if old['hardware_profile'] != profile.to_dict():
        raise ValueError('reference graph objective changed')
    def manifest_semantics(value):
        if isinstance(value, dict):
            return {k: manifest_semantics(v) for k, v in value.items() if k != 'source'}
        if isinstance(value, list):
            return [manifest_semantics(v) for v in value]
        return value
    manifests = []
    for path, capture in ((args.directory, stamp), (directory, old_capture)):
        payload = load_graph(path / 'capture.pkl.gz', capture['capture_hash'])['manifest']
        manifests.append(manifest_semantics(json.loads(payload) if isinstance(payload, (str, bytes)) else payload))
    if manifests[0] != manifests[1]:
        raise ValueError('native access manifest changed; construct a new reference graph before this study')
    if args.case == 'gemm--asymmetric' and (args.platform != 'matrix' or stamp['config'].get('BLOCK_K') != 32):
        raise ValueError('GEMM study requires Matrix and frozen BK=32')
    graph = load_graph(directory / 'graph.pkl.gz', old['graph_hash'])
    graph = constrain_graph(graph, stamp.get('native_layout_contracts', []))
    binding = digest({'capture': stamp['capture_hash'], 'graph': old['graph_hash'],
                      'source': stamp['source_identity'], 'profile': profile.to_dict()})
    result = panel(graph, profile, args.case, checkpoint_dir=args.directory / 'study-scores' / binding)
    result.update(source_identity=stamp['source_identity'], capture_hash=stamp['capture_hash'],
                  hardware_profile=profile.to_dict(), native_layout_contracts=stamp.get('native_layout_contracts', []),
                  reference_graph={'path': str(directory), 'hash': old['graph_hash']},
                  fixed_schedule=stamp['config'])
    result['selection_hash'] = digest(result)
    write_json(args.directory / 'search.json', result)
    print(f"Saved {len(result['candidates'])} distinct physical maps; top-1={result['analytical_top1']}", flush=True)


def storages(candidates, manual=False):
    result = []
    for candidate in candidates:
        automatic = {**candidate, 'storage': 0, 'automatic': bool(candidate['runtime_layouts'])}
        result.append(automatic)
        if manual:
            result.append({**candidate, 'id': candidate['id'] + '-explicit', 'explicit': True})
    return result


def worker(args):
    from conventional import worker as measure
    batch = json.loads(args.study_batch.read_text())
    search_record = read(args, 'search.json')
    if batch['selection_hash'] != search_record['selection_hash']:
        raise ValueError('study batch belongs to a different panel')
    batch_hash = digest(batch)
    args.study_storages = batch['storages']
    args.phase = batch['phase']
    measure(args)
    record = json.loads(args.worker_output.read_text())
    record['batch_hash'] = batch_hash
    record['measurement'] = {name: getattr(args, name) for name in ('samples', 'iterations', 'warmup', 'process_index')}
    record['record_hash'] = digest(record)
    write_json(args.worker_output, record)


def measure_batch(args, search_record, phase, index, candidates, process, manual=False, choice_hash=None):
    directory = args.directory / phase / f'batch-{index}'
    batch = {'phase': f'{phase}/batch-{index}', 'selection_hash': search_record['selection_hash'],
             'choice_hash': choice_hash, 'storages': storages(candidates, manual)}
    path = directory / 'batch.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(batch)):
        raise ValueError('study batch changed; use a fresh root')
    write_json(path, batch)
    output = directory / f'process-{process}.json'
    measurement = {'samples': args.samples, 'iterations': args.iterations, 'warmup': args.warmup,
                   'process_index': process}
    if output.exists():
        saved = json.loads(output.read_text())
        if 'record_hash' in saved:
            if (saved['record_hash'] != digest({k: v for k, v in saved.items() if k != 'record_hash'})
                    or saved['batch_hash'] != digest(batch) or saved['measurement'] != measurement):
                raise ValueError('study timing checkpoint changed; use a fresh root')
            return saved
    print(f'{phase} batch {index} process {process}: {[c["id"] for c in candidates]}', flush=True)
    command = [sys.executable, str(HERE / 'run.py'), '--stage', 'study_worker', '--platform', args.platform,
               '--case', args.case, '--root', str(args.root), '--tau-name', args.tau_name,
               '--study-batch', str(path), '--worker-output', str(output)]
    for name, value in measurement.items():
        command += ['--' + name.replace('_', '-'), str(value)]
    subprocess.run(command, check=True)
    return json.loads(output.read_text())


def tuning_summary(candidates, records):
    from packet_measure import matched_summary
    results = {}
    for candidate in candidates:
        name = candidate['id']
        rows = records[name]
        label = name + '-fixed'
        if any(label not in row['timings'] for row in rows):
            results[name] = {'failed': [row['failures'] for row in rows]}
        else:
            results[name] = matched_summary(rows, 'ordinary', label)
    return results


def freeze_choice(search_record, tuning):
    eligible = [c['id'] for c in search_record['candidates'] if 'failed' not in tuning[c['id']]]
    if 'ordinary' not in eligible:
        raise ValueError('ordinary tuning validation failed')
    winner = min(eligible, key=lambda name: (-tuning[name]['speedup'], name != 'ordinary', name))
    selected = list(dict.fromkeys(['ordinary', search_record['analytical_top1'], winner]))
    contrasts = equivalence_report(search_record, tuning)['feature_class']
    contrast = contrasts[0] if contrasts and contrasts[0]['training_time_spread'] > 1.05 else None
    if contrast:
        selected = list(dict.fromkeys([*selected, contrast['fastest'], contrast['slowest']]))
    audits = [c['id'] for c in search_record['candidates'] if c['id'] not in selected and
              ('failed' in tuning[c['id']] or tuning[c['id']]['speedup'] < search_record['poor_speedup_threshold'])]
    choice = {'selection_hash': search_record['selection_hash'], 'empirical_winner': winner,
              'analytical_top1': search_record['analytical_top1'], 'held_out': selected, 'manual_audits': audits,
              'feature_contrast': contrast,
              'tuning': tuning, 'method': 'maximum training geometric mean of ordinary/candidate paired ratios; ordinary included'}
    choice['choice_hash'] = digest(choice)
    return choice


def equivalence_report(search_record, tuning):
    result = {}
    for field in ('scalar_class', 'feature_class', 'flag_class'):
        groups = defaultdict(list)
        for candidate in search_record['candidates']:
            if field in candidate:
                groups[candidate[field]].append(candidate['id'])
        contrasts = []
        for key, names in groups.items():
            measured = [name for name in names if 'speedup' in tuning[name]]
            if len(measured) < 2:
                continue
            fastest = max(measured, key=lambda name: tuning[name]['speedup'])
            slowest = min(measured, key=lambda name: tuning[name]['speedup'])
            contrasts.append({'class': key, 'members': names, 'fastest': fastest, 'slowest': slowest,
                              'training_time_spread': tuning[fastest]['speedup'] / tuning[slowest]['speedup'],
                              'interpretation': 'training contrast across matched-control batches; requires independent confirmation'})
        result[field] = sorted(contrasts, key=lambda row: -row['training_time_spread'])
    return result


def run(args):
    from conventional import results_for, realization_pairs
    from packet_measure import matched_summary
    verify(args, gpu=True)
    search_record = read(args, 'search.json')
    candidates = search_record['candidates']
    by_name = {c['id']: c for c in candidates}
    size = search_record['batch_size']
    changed = [c for c in candidates if c['id'] != 'ordinary']
    batches = [[by_name['ordinary'], *changed[i:i+size]] for i in range(0, len(changed), size)]
    records = {c['id']: [] for c in candidates}
    for process in range(args.processes):
        order = list(range(len(batches)))
        shift = process % len(order)
        order = order[shift:] + order[:shift]
        if process % 2:
            order.reverse()
        for index in order:
            row = measure_batch(args, search_record, 'study-tune', index, batches[index], process)
            for candidate in batches[index]:
                # Ordinary is timed identically in every batch; one batch per
                # process supplies its selection control, avoiding pseudoreplication.
                if candidate['id'] != 'ordinary' or index == 0:
                    records[candidate['id']].append(row)
            write_json(args.directory / 'study-progress.json',
                       {'phase': 'tune', 'completed_candidate_processes': sum(map(len, records.values())),
                        'required_candidate_processes': len(candidates)*args.processes})
    tuning = tuning_summary(candidates, records)
    # Ordinary is the reference choice, not a noisy duplicate-kernel contestant.
    tuning['ordinary']['speedup'] = 1.
    choice = freeze_choice(search_record, tuning)
    choice_path = args.directory / 'study-choice.json'
    if choice_path.exists() and json.loads(choice_path.read_text()) != choice:
        raise ValueError('frozen empirical selection changed')
    write_json(choice_path, choice)
    selected = [by_name[name] for name in choice['held_out']]
    heldout = [measure_batch(args, search_record, 'study-evaluate', 0, selected, process, manual=True,
                             choice_hash=choice['choice_hash']) for process in range(args.processes)]
    labels = {storage['id']: storage['id']+'-fixed' for storage in storages(selected, manual=True)}
    values = results_for(heldout, labels, 'ordinary')
    def ratio(left, right):
        if any(left not in row['timings'] or right not in row['timings'] for row in heldout):
            return {'unavailable': 'at least one realization failed; inspect failures'}
        return matched_summary(heldout, left, right)
    winner = choice['empirical_winner']
    top = choice['analytical_top1']
    def auto_label(name):
        return 'ordinary' if name == 'ordinary' else name+'-fixed'
    report = {'schema': 'laqs.layout.study.v1', 'choice': choice, 'panel': search_record,
              'H_ordinary_over_empirical': ratio('ordinary', auto_label(winner)),
              'S_top1_over_empirical': ratio(auto_label(top), auto_label(winner)),
              'C_automatic_over_manual': realization_pairs(heldout, labels),
              'held_out': values, 'held_out_failures': [r['failures'] for r in heldout],
              'equivalence_contrasts': equivalence_report(search_record, tuning),
              'scope': 'empirical selection estimate in declared family; fixed schedule; prepacked; not a global oracle',
              'manual_audits': {}, 'complete': False}
    contrast = choice['feature_contrast']
    if contrast:
        report['held_out_equal_feature_contrast'] = {
            **contrast, 'slow_over_fast': ratio(auto_label(contrast['slowest']), auto_label(contrast['fastest'])),
            'interpretation': 'A confirmed difference with equal full component vectors cannot be resolved by reweighting those components; inspect realization evidence before attributing it to the graph.'}
    # Publish the main result before potentially lengthy rejected/poor-map audits.
    write_json(args.directory / 'study-partial.json', report)
    audits = choice['manual_audits']
    for offset in range(0, len(audits), size):
        names = audits[offset:offset+size]
        diagnostic = [by_name['ordinary'], *(by_name[name] for name in names)]
        rows = [measure_batch(args, search_record, 'study-audit', offset//size, diagnostic, process,
                              manual=True, choice_hash=choice['choice_hash']) for process in range(args.processes)]
        audit_labels = {s['id']: s['id']+'-fixed' for s in storages(diagnostic, manual=True)}
        report['manual_audits'].update({name: {'result': results_for(rows,
            {name+'-explicit': name+'-explicit-fixed'}, 'ordinary'),
            'realization': realization_pairs(rows, audit_labels).get(name+'-explicit'),
            'failures': [r['failures'] for r in rows]} for name in names})
        write_json(args.directory / 'study-partial.json', report)
    report['complete'] = True
    report['profile_contrast'] = {'candidates': choice['held_out'],
        'condition': 'profile only if held-out H confidence interval exceeds one and placement controls are clean',
        'metrics': ['L1 requests/sectors', 'L2 requests/sectors', 'DRAM bytes', 'registers/spills', 'shared memory'],
        'status': 'separate counter run required; counter duration is not performance evidence'}
    write_json(args.directory / 'study.json', report)
    lines = [f'# Fixed-schedule layout study: {args.case} / {args.platform}', '',
             f"Panel: {len(candidates)} distinct physical maps. Top-1: `{top}`. Empirical selection: `{winner}`.",
             'Tuning uses ordinary-anchored batches; selected maps and manual copies are evaluated together in fresh processes.',
             'Three shared placements per process; packing excluded. The empirical winner estimates this family only.', '',
             '| Held-out ratio | Estimate | 95% process interval |', '| --- | ---: | --- |']
    for name in ('H_ordinary_over_empirical', 'S_top1_over_empirical'):
        value = report[name]
        lines.append(f"| {name} | {value.get('speedup', 'unavailable')} | {value.get('speedup_ci95')} |")
    for name, value in report['C_automatic_over_manual'].items():
        ratio_value = value['automatic_over_explicit']
        lines.append(f"| C: {name} | {ratio_value['speedup']} | {ratio_value['speedup_ci95']} |")
    lines += ['', 'Full component vectors, separate equality classes, training contrasts, failures, manual audits,',
              'packing amortization and placement controls are saved in study.json and the per-batch records.',
              'Interpret ratios with their controls and intervals; a training winner alone is not evidence of headroom.']
    (args.directory / 'study.md').write_text('\n'.join(lines)+'\n')


def profile_worker(args):
    """Expose one validated automatic dispatch, matching its held-out binary."""
    from contextlib import nullcontext
    import torch
    from conventional import pack_inputs
    from layout_runtime import freeze_launch, fresh_outputs
    from packet_cases import CASES, reference
    from packet_runtime import runtime_layouts, structured_layouts, statistics
    from packet_workflow import verify_inputs
    stamp = verify(args, gpu=True)
    record = read(args, 'search.json')
    candidate = next(c for c in record['candidates'] if c['id'] == args.profile_candidate)
    source = freeze_launch(CASES[args.case].factory(), stamp['config'])
    verify_inputs(source, stamp)
    layouts = runtime_layouts(candidate)
    launch = fresh_outputs(pack_inputs(source, layouts), stamp['output_arguments'])
    context = structured_layouts(layouts, launch=launch) if layouts else nullcontext()
    with context:
        launch.run()
        checked = launch.clone()
        for item in stamp['inputs']:
            checked.values[item['argument']] = source.values[item['argument']]
        reference(CASES[args.case].operator, checked)
        for _ in range(args.warmup * args.iterations):
            launch.run()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        kernel = launch.run()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    code = statistics(kernel, args.directory / 'study-profile/codegen', candidate['id'])
    timed = read(args, 'study-evaluate/batch-0/process-0.json')['codegen'][candidate['id']+'-fixed']
    if code['artifacts']['cubin']['sha256'] != timed['artifacts']['cubin']['sha256']:
        raise ValueError('profiled binary differs from held-out timed binary')
    write_json(args.worker_output, {'codegen': code, 'matches_timed_binary': True})


def profile(args):
    """Profile only the ordinary/top-1/winner contrast after confirmed headroom."""
    import hashlib
    import shutil
    from address_experiment import WAVEFRONTS, parse_counters
    from stage1_nvidia_counter_analysis import COUNTER_METRICS, counter_definitions
    from packet_measure import measured_choice
    stamp = verify(args, gpu=True)
    if args.platform != 'matrix':
        raise ValueError('this counter stage uses Nsight Compute on Matrix')
    report = read(args, 'study.json')
    headroom = report['H_ordinary_over_empirical']
    output = args.directory / 'study-profile.json'
    if ('speedup' not in headroom or measured_choice({'ordinary': {'speedup': 1.,
            'speedup_ci95': None, 'identity_max_deviation': 0., 'same_pointer_max_deviation': 0.},
            'winner': headroom}) == 'ordinary'):
        write_json(output, {'status': 'skipped: no control-qualified held-out headroom',
                            'headroom': headroom})
        return
    executable = shutil.which('ncu')
    if executable is None:
        raise FileNotFoundError('load Nsight Compute before running the study counter script')
    metrics = (*COUNTER_METRICS, WAVEFRONTS)
    version = subprocess.check_output([executable, '--version'], text=True)
    binding = digest({'report': report, 'ncu': version, 'metrics': metrics,
                      'warmup': args.warmup, 'iterations': args.iterations})
    directory = args.directory / 'study-profile'
    directory.mkdir(exist_ok=True)
    choice = report['choice']
    names = list(dict.fromkeys(['ordinary', choice['analytical_top1'], choice['empirical_winner']]))
    records = {}
    for name in names:
        csv, worker_output, checkpoint = [directory / (name+suffix) for suffix in ('.csv', '.json', '.complete.json')]
        def hashes():
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (csv, worker_output)}
        if checkpoint.exists() and csv.exists() and worker_output.exists():
            saved = json.loads(checkpoint.read_text())
            if saved['binding'] == binding and saved['outputs'] == hashes():
                records[name] = saved['counters']
                continue
        csv.unlink(missing_ok=True)
        command = [executable, '--csv', '--page', 'raw', '--print-units', 'base', '--metrics', ','.join(metrics),
            '--profile-from-start', 'off', '--replay-mode', 'application', '--cache-control', 'none',
            '--clock-control', 'none', '--target-processes', 'application-only', '--log-file', str(csv),
            sys.executable, str(HERE / 'run.py'), '--stage', 'study_profile_worker', '--platform', args.platform,
            '--case', args.case, '--root', str(args.root), '--tau-name', args.tau_name,
            '--profile-candidate', name, '--worker-output', str(worker_output),
            '--warmup', str(args.warmup), '--iterations', str(args.iterations)]
        subprocess.run(command, check=True)
        records[name] = parse_counters(csv, stamp['kernel_name'])
        write_json(checkpoint, {'binding': binding, 'outputs': hashes(), 'counters': records[name]})
    definitions = counter_definitions()
    definitions[WAVEFRONTS] = 'L1TEX global-load output wavefronts'
    write_json(output, {'status': 'complete', 'binding': binding, 'ncu': version, 'candidates': records,
        'definitions': {metric: definitions[metric] for metric in metrics},
        'interpretation': 'Whole-kernel counters include ordinary B; not A-only traffic. Profiled duration is not a speedup. Registers, spills and shared bytes are in each matched binary record.',
        'cache_regime': f'{args.warmup * args.iterations} same-candidate launches; application replay; no cache flush or clock control'})
