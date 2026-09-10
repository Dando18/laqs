"""Frozen-map source/pass comparison with matched placements and equal tuning budgets."""
from contextlib import nullcontext
import json
import statistics

from packet_measure import allocation_samples, packing_cost, conversion_result, matched_summary

SCHEDULES = tuple({'BLOCK_M': m, 'BLOCK_K': k, 'num_warps': 4}
                  for m in (1, 4, 16) for k in (64, 128))


def laqs_candidates(search):
    result = []
    for candidate in search['candidates']:
        if candidate['runtime_layouts']:
            result += [{**candidate, 'storage': 0, 'automatic': True,
                        'realization': 'automatic structured compiler pass'},
                       {**candidate, 'id': candidate['id'] + '-explicit', 'storage': 2,
                        'source_candidate': candidate['id'], 'explicit': True,
                        'realization': 'explicit exact split address function'}]
    return result


def storage_candidates(size, search):
    from relay import MatrixSpec, CanonicalLayout, layout_matrix_rows
    from layout_contract import RuntimeLayout
    matrix = MatrixSpec('A', (size, size), 4, ('i', 'j'))
    n, m = matrix.mode_bits
    column = CanonicalLayout('column', 'A', (n, m), (0,) * n + (1,) * m, (1, 0))
    tile = CanonicalLayout('tile16', 'A', (n, m), (1,) * 4 + (0,) * 4 + (1,) * (m - 4) + (0,) * (n - 4), (1, 0))
    def conventional(label, storage, layout):
        mapping = RuntimeLayout('A', 0, (size, size), (size, 1), (size, size), layout_matrix_rows(matrix, layout))
        return {'id': label, 'runtime_layouts': [mapping.to_dict()], 'storage': storage,
                'realization': 'explicit conventional address function'}
    return [{'id': 'ordinary', 'runtime_layouts': [], 'storage': 0},
            conventional('column', 1, column), conventional('tile16', 2, tile),
            *laqs_candidates(search)]


def pack_inputs(ordinary, layouts):
    from manual_layout import pack_split, split_tile
    result = ordinary.clone()
    for layout in layouts:
        result.values[layout.argument] = pack_split(ordinary.values[layout.argument], split_tile(layout))
    return result


def worker(args):
    import torch
    from experiment_support import digest
    from layout_runtime import freeze_launch, fresh_outputs, pack_tensor
    from manual_layout import explicit_spec, split_tile, unpack_split
    from packet_cases import CASES, reference
    from packet_runtime import runtime_layouts, structured_layouts, statistics as code_stats, primitive_rejections
    from packet_workflow import read, verify, write_json, verify_inputs
    stamp = verify(args, gpu=True)
    search = read(args, 'search.json')
    spec = CASES[args.case].factory()
    ordinary = freeze_launch(spec, stamp['config'])
    verify_inputs(ordinary, stamp)
    ordinary.run()
    reference(spec.operator, ordinary)
    if hasattr(args, 'study_storages'):
        storages = args.study_storages
    elif spec.operator == 'row_column':
        storages = storage_candidates(ordinary.values[5], search)
    else:
        storages = [{'id': 'ordinary', 'runtime_layouts': [], 'storage': 0},
                    {'id': 'source-ordinary', 'runtime_layouts': [], 'explicit': True},
                    *laqs_candidates(search)]
    frozen = read(args, 'conventional-choice.json') if args.phase == 'conventional-evaluate' else None
    fixed_schedule = args.phase == 'conventional-fixed' or args.phase.startswith('study-')
    schedules = ({},) if fixed_schedule else SCHEDULES
    launches, contexts, groups, failures, packing, codegen, checks = {}, {}, {}, {}, {}, {}, {}
    packed_cache = {}
    directory = args.directory / args.phase / f'process-{args.process_index}-codegen'
    for storage in storages:
        name = storage['id']
        layouts = runtime_layouts(storage)
        try:
            storage['tiles'] = {str(layout.argument): split_tile(layout) for layout in layouts}
            key = digest(storage['runtime_layouts'])
            if key not in packed_cache:
                packed = pack_inputs(ordinary, layouts)
                for layout in layouts:
                    source = ordinary.values[layout.argument]
                    physical = packed.values[layout.argument]
                    if not torch.equal(physical, pack_tensor(source, layout)):
                        raise ValueError('manual packing disagrees with saved bit-matrix map')
                    if not torch.equal(source, unpack_split(physical, layout.shape, split_tile(layout))):
                        raise ValueError('manual packing round trip failed')
                cost = packing_cost(ordinary, layouts, pack=pack_inputs)
                cost['method'] = 'warm reshape/permute packing plus allocation; synchronous wall time'
                packed_cache[key] = packed, cost
            packed, packing[name] = packed_cache[key]
        except Exception as error:
            failures[name] = f'{type(error).__name__}: {error}'
            continue
        for index, schedule in enumerate(schedules):
            label = f'{name}-fixed' if fixed_schedule else f'{name}-s{index}'
            if frozen and frozen['selected_schedules'].get(name) != label:
                continue
            try:
                config = {**stamp['config'], **schedule}
                launch_spec = spec
                if storage.get('explicit'):
                    launch_spec, parameters = explicit_spec(spec, layouts)
                    config.update(parameters)
                elif spec.operator == 'row_column':
                    config['STORAGE'] = storage['storage']
                launch = fresh_outputs(freeze_launch(launch_spec, config), stamp['output_arguments'])
                for record in stamp['inputs']:
                    arg = record['argument']
                    launch.values[arg] = packed.values[arg]
                automatic = storage.get('automatic', False)
                ctx = (lambda layouts=layouts, launch=launch, storage=storage: structured_layouts(
                    layouts, mode=storage.get('address_mode', 'repaired'), launch=launch)) if automatic else nullcontext
                with ctx():
                    kernel = launch.run()
                checked = launch.clone()
                for record in stamp['inputs']:
                    arg = record['argument']
                    checked.values[arg] = ordinary.values[arg]
                checks[label] = reference(spec.operator, checked)
                stats = code_stats(kernel, directory, label)
                stats['launch_config'] = config
                # Inspection collects native site ownership without rewriting addresses.
                # A manual source kernel does not require the pass's admission
                # rules: an unavailable diagnostic must not reject correct code.
                try:
                    with structured_layouts(layouts, inspect=True, launch=launch):
                        inspected = launch.run()
                    inspection = code_stats(inspected, directory, label + '-inspect')
                    stats['inspected_packet_sites'] = inspection['packet_sites']
                except Exception as error:
                    if automatic:
                        raise
                    stats['inspection_error'] = f'{type(error).__name__}: {error}'
                if automatic:
                    native = freeze_launch(spec, config)
                    with structured_layouts(layouts, inspect=True, launch=native):
                        native_kernel = native.run()
                    reasons = primitive_rejections(code_stats(native_kernel, directory, label + '-native'), stats)
                    if reasons:
                        raise ValueError('; '.join(reasons))
                launches[label], contexts[label], groups[label], codegen[label] = launch, ctx, name, stats
            except Exception as error:
                failures[label] = f'{type(error).__name__}: {error}'
    launches['ordinary'], contexts['ordinary'] = ordinary, nullcontext
    logical_inputs = {v['argument']: ordinary.values[v['argument']] for v in stamp['inputs']}
    def validate_placement(label, launch):
        checked = launch.clone()
        for index, tensor in logical_inputs.items():
            checked.values[index] = tensor
        reference(spec.operator, checked)
    def capture_code(label, kernel):
        previous = codegen.get(label, {})
        codegen[label] = code_stats(kernel, directory, label)
        codegen[label]['inspected_packet_sites'] = previous.get('inspected_packet_sites', [])
        codegen[label]['launch_config'] = previous.get('launch_config', stamp['config'])
        if 'inspection_error' in previous:
            codegen[label]['inspection_error'] = previous['inspection_error']
    measured = allocation_samples(launches, contexts, list(logical_inputs), stamp['output_arguments'],
        process_index=args.process_index, samples=args.samples, iterations=args.iterations,
        warmup=args.warmup, on_capture=capture_code, on_ready=validate_placement)
    write_json(args.worker_output, {'phase': args.phase, 'selection_hash': search['selection_hash'],
        **measured, 'groups': groups, 'failures': failures, 'packing': packing, 'validation': checks,
        'codegen': codegen, 'schedules': schedules, 'storages': storages})


def results_for(rows, selected, baseline):
    results = {}
    for storage, label in selected.items():
        if any(label not in row['timings'] or baseline not in row['timings'] for row in rows):
            results[storage] = {'failed': 'frozen schedule failed held-out validation'}
            continue
        value = matched_summary(rows, baseline, label)
        value['packing_ms'] = statistics.median(row['packing'][storage]['median_ms'] for row in rows)
        # Conversion ratios also pair within placement; do not divide pooled medians.
        converted = []
        for row in rows:
            for placement in row['allocation']['placements']:
                times = placement['timings']
                converted.append(conversion_result(times[baseline]['median_ms'], times[label]['median_ms'],
                                                   row['packing'][storage]['median_ms']))
        import math
        value['conversion'] = {
            'one_use_including_conversion_speedup': math.exp(statistics.fmean(
                math.log(c['one_use_including_conversion_speedup']) for c in converted)),
            'placement_results': converted,
            'break_even_reuses': ([c['break_even_reuses'] for c in converted])}
        results[storage] = value
    return results


def realization_pairs(rows, selected):
    from manual_layout import execution_comparison
    result = {}
    for storage, explicit in selected.items():
        if not storage.endswith('-explicit'):
            continue
        automatic = selected.get(storage.removesuffix('-explicit'))
        if automatic and all(automatic in row['timings'] and explicit in row['timings'] for row in rows):
            result[storage] = {'automatic_over_explicit': matched_summary(rows, automatic, explicit),
                'lowered_comparisons': [execution_comparison(row['codegen'][automatic], row['codegen'][explicit])
                                        for row in rows]}
    return result


def run(args):
    from experiment_support import digest
    from packet_workflow import read, write_json, run_processes, verify
    if args.case.split('--')[0] not in ('row_column', 'sum', 'gemm'):
        raise ValueError('manual comparisons support row_column, sum and gemm')
    verify(args, gpu=True)
    search = read(args, 'search.json')
    run_processes(args, 'conventional-fixed')
    fixed = [read(args, f'conventional-fixed/process-{i}.json') for i in range(args.processes)]
    fixed_labels = {storage: label for row in fixed for label, storage in row['groups'].items()}
    fixed_results = results_for(fixed, fixed_labels, 'ordinary')
    report = {'schema': 'laqs.manual.v1', 'selection_hash': search['selection_hash'],
              'analytical_top1': search['analytical_top1'], 'fixed_results': fixed_results,
              'fixed_realization': realization_pairs(fixed, fixed_labels),
              'fixed_failures': [row['failures'] for row in fixed], 'deployment_selection': False}
    if args.case.startswith('row_column--'):
        run_processes(args, 'conventional-tune')
        rows = [read(args, f'conventional-tune/process-{i}.json') for i in range(args.processes)]
        labels = set.intersection(*(set(row['groups']) for row in rows))
        winners = {}
        for label in sorted(labels):
            storage = rows[0]['groups'][label]
            score = matched_summary(rows, 'ordinary', label)['speedup']
            if storage not in winners or score > winners[storage][0]:
                winners[storage] = score, label
        if 'ordinary' not in winners:
            raise ValueError('no ordinary schedule passed validation')
        choice = {'selected_schedules': {name: pair[1] for name, pair in winners.items()},
                  'selection_hash': search['selection_hash'], 'schedules': SCHEDULES,
                  'budget_per_storage': len(SCHEDULES), 'maps': search['candidates']}
        choice['choice_hash'] = digest(choice)
        write_json(args.directory / 'conventional-choice.json', choice)
        run_processes(args, 'conventional-evaluate')
        if read(args, 'conventional-choice.json') != json.loads(json.dumps(choice)):
            raise ValueError('schedule choice changed during evaluation')
        rows = [read(args, f'conventional-evaluate/process-{i}.json') for i in range(args.processes)]
        selected = choice['selected_schedules']
        report.update(choice=choice, results=results_for(rows, selected, selected['ordinary']),
                      tuned_realization=realization_pairs(rows, selected), failures=[r['failures'] for r in rows])
    write_json(args.directory / 'conventional.json', report)
    lines = ['# Manual realization of frozen LAQS maps', '',
             'Complete outputs and exact packing are validated. All variants rotate through three shared input placements.',
             'Ratios are paired within placement before combining independent processes. Packing and compilation are excluded.',
             f"Analytical top-1: `{search['analytical_top1']}`; maps remain frozen throughout.", '',
             '| Comparison | Storage | Speedup vs ordinary (95% interval) | One use including packing |',
             '| --- | --- | ---: | ---: |']
    for phase, values in [('fixed schedule', fixed_results), ('equal-budget tuning', report.get('results', {}))]:
        for storage, value in values.items():
            if 'failed' in value:
                lines.append(f'| {phase} | {storage} | failed | |')
                continue
            ci = value['speedup_ci95']
            interval = f' ({ci[0]:.4f}–{ci[1]:.4f})' if ci else ' (one process; no interval)'
            lines.append(f"| {phase} | {storage} | {value['speedup']:.4f}×{interval} | "
                         f"{value['conversion']['one_use_including_conversion_speedup']:.4f}× |")
    lines += ['', 'Row/column tuning offers the same six schedules to each storage and realization, then evaluates in fresh processes.',
              'Sum and GEMM are fixed-schedule diagnostic controls; source-ordinary checks the manual kernel copy.', '',
              '| Comparison | Map | Automatic / explicit runtime | Changed ownership/primitive evidence |',
              '| --- | --- | ---: | --- |']
    for phase in ('fixed', 'tuned'):
        for name, pair in report.get(phase + '_realization', {}).items():
            fields = sorted({field for row in pair['lowered_comparisons'] for field in row['changed_fields']})
            status = ', '.join(fields) or 'none recorded'
            if not all(row['ownership_comparable'] for row in pair['lowered_comparisons']):
                status += '; ownership unavailable'
            lines.append(f"| {phase} | {name} | {pair['automatic_over_explicit']['speedup']:.4f}× | {status} |")
    lines += ['', 'Changed ownership or memory primitives require a new graph/score before attributing the explicit result to the original model.',
              'Raw IR encoding text differences are retained separately; alias names and repeated snippets are not ownership changes.',
              'Optional inspection failures are retained in codegen records and do not exclude otherwise correct manual kernels.',
              'Tuned realization ratios may also compare different chosen schedules; use the fixed-schedule ratios to isolate realization.',
              'Failures, controls, conversion break-even counts, per-placement timings, and actual captured binaries are retained in the JSON/worker records.']
    failures = report['fixed_failures'] + report.get('failures', [])
    for failure in failures:
        for label, reason in failure.items():
            entry = f'- `{label}`: {reason}'
            if entry not in lines:
                lines.append(entry)
    (args.directory / 'conventional.md').write_text('\n'.join(lines) + '\n')
