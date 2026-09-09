"""Equal-budget schedule comparison for the conflicting-orientation operator.

This is a separate experiment: it never expands the three-candidate deployment
choice or relabels hand-written storage indexing as an automatic compiler result.
"""
from contextlib import nullcontext
import json
import statistics

from packet_measure import graph_samples, packing_cost, conversion_result

# The same declared six schedules are offered to every storage representation.
SCHEDULES = tuple({'BLOCK_M': m, 'BLOCK_K': k, 'num_warps': 4}
                  for m in (1, 4, 16) for k in (64, 128))


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
            *[{**c, 'storage': 0, 'realization': 'automatic structured compiler pass'}
              for c in search['candidates'] if c['runtime_layouts']]]


def worker(args):
    import torch
    from layout_runtime import freeze_launch, fresh_outputs, replace_inputs
    from packet_cases import CASES, reference
    from packet_runtime import runtime_layouts, structured_layouts, statistics as code_stats, primitive_rejections
    from packet_workflow import read, verify, write_json, verify_inputs
    stamp = verify(args, gpu=True)
    search = read(args, 'search.json')
    spec = CASES[args.case].factory()
    ordinary = freeze_launch(spec, stamp['config'])
    verify_inputs(ordinary, stamp)
    ordinary.run()
    reference('row_column', ordinary)
    storages = storage_candidates(ordinary.values[5], search)
    frozen = read(args, 'conventional-choice.json') if args.phase == 'conventional-evaluate' else None
    launches, contexts, groups, failures, packing, codegen = {}, {}, {}, {}, {}, {}
    for storage in storages:
        layouts = runtime_layouts(storage)
        packed = replace_inputs(ordinary, layouts)
        packing[storage['id']] = packing_cost(ordinary, layouts)
        automatic = storage.get('realization') == 'automatic structured compiler pass'
        for index, schedule in enumerate(SCHEDULES):
            label = f"{storage['id']}-s{index}"
            if frozen and frozen['selected_schedules'].get(storage['id']) != label:
                continue
            try:
                config = {**stamp['config'], **schedule, 'STORAGE': storage['storage']}
                launch = freeze_launch(spec, config)
                launch.values[0] = packed.values[0]
                launch = fresh_outputs(launch, stamp['output_arguments'])
                ctx = (lambda layouts=layouts: structured_layouts(layouts)) if automatic else nullcontext
                with ctx():
                    kernel = launch.run()
                checked = launch.clone()
                checked.values[0] = ordinary.values[0]
                reference('row_column', checked)
                directory = args.directory / args.phase / f'process-{args.process_index}-codegen'
                stats = code_stats(kernel, directory, label)
                if automatic:
                    native = freeze_launch(spec, config)
                    with structured_layouts(layouts, inspect=True):
                        native_kernel = native.run()
                    reasons = primitive_rejections(code_stats(native_kernel, directory, label + '-native'), stats)
                    if reasons:
                        raise ValueError('; '.join(reasons))
                launches[label], contexts[label], groups[label], codegen[label] = launch, ctx, storage['id'], stats
            except Exception as error:
                failures[label] = f'{type(error).__name__}: {error}'
    # Fixed native anchor is a control; each storage's independently tuned winner
    # is compared with the tuned ordinary winner in the final report.
    launches['ordinary'], contexts['ordinary'] = ordinary, nullcontext
    timings = graph_samples(launches, contexts, stamp['output_arguments'], process_index=args.process_index,
                           samples=args.samples, iterations=args.iterations, warmup=args.warmup)
    write_json(args.worker_output, {'phase': args.phase, 'selection_hash': search['selection_hash'],
               'timings': timings, 'groups': groups, 'failures': failures, 'packing': packing,
               'codegen': codegen, 'schedules': SCHEDULES})


def run(args):
    from experiment_support import digest, process_summary
    from packet_workflow import read, write_json, run_processes, verify
    if not args.case.startswith('row_column--'):
        raise ValueError('conventional comparisons are defined only for row_column')
    verify(args, gpu=True)
    tuning = run_processes(args, 'conventional-tune')
    rows = [read(args, f'conventional-tune/process-{i}.json') for i in range(args.processes)]
    labels = set.intersection(*(set(row['groups']) for row in rows))
    winners = {}
    for label in sorted(labels):
        storage = rows[0]['groups'][label]
        cost = statistics.median(row['timings'][label]['median_ms'] for row in rows)
        if storage not in winners or cost < winners[storage][0]:
            winners[storage] = cost, label
    if 'ordinary' not in winners:
        raise ValueError('no conventional ordinary schedule passed validation')
    choice = {'selected_schedules': {name: pair[1] for name, pair in winners.items()},
              'selection_hash': read(args, 'search.json')['selection_hash'], 'schedules': SCHEDULES,
              'budget_per_storage': len(SCHEDULES), 'tuning': tuning,
              'automatic_storage_layouts': 'frozen LAQS proposals; schedule tuning does not change maps'}
    choice['choice_hash'] = digest(choice)
    write_json(args.directory / 'conventional-choice.json', choice)
    run_processes(args, 'conventional-evaluate')
    if read(args, 'conventional-choice.json') != json.loads(json.dumps(choice)):
        raise ValueError('conventional schedule choice changed during evaluation')
    rows = [read(args, f'conventional-evaluate/process-{i}.json') for i in range(args.processes)]
    results = {}
    base = choice['selected_schedules']['ordinary']
    for storage, label in choice['selected_schedules'].items():
        if any(label not in row['timings'] or base not in row['timings'] for row in rows):
            results[storage] = {'failed': 'frozen schedule failed held-out validation'}
            continue
        paired = [{'timings': {'baseline': row['timings'][base], 'selected': row['timings'][label],
                              'identity': row['timings']['identity']}} for row in rows]
        value = process_summary(paired)
        # Identity is the fixed native anchor, not a clone of the retuned baseline.
        value.pop('identity_max_deviation', None)
        value['packing_ms'] = statistics.median(row['packing'][storage]['median_ms'] for row in rows)
        value['conversion'] = conversion_result(value['baseline']['median_ms'], value['selected']['median_ms'], value['packing_ms'])
        results[storage] = value
    write_json(args.directory / 'conventional.json', {'choice': choice, 'results': results,
               'failures': [row['failures'] for row in rows],
               'control_scope': 'identity/same-pointer controls compare fixed ordinary anchor; timings retain that anchor',
               'deployment_selection': False})
    lines = ['# Equal-budget storage and schedule comparison', '',
             'Every storage receives the same six schedules. Storage maps are frozen before tuning; fresh processes evaluate the chosen schedules.', '',
             '| Storage | Schedule | Held-out speedup vs tuned ordinary | One use including packing |',
             '| --- | --- | ---: | ---: |']
    for storage, value in results.items():
        speedup = f"{value['speedup']:.4f}×" if 'speedup' in value else 'failed'
        converted = f"{value['conversion']['one_use_including_conversion_speedup']:.4f}×" if 'conversion' in value else 'failed'
        lines.append(f"| {storage} | {choice['selected_schedules'][storage]} | {speedup} | {converted} |")
    lines += ['', 'Column and tile16 use explicit conventional indexing. LAQS candidates use the automatic packet compiler pass. See conventional.json and the worker records for exclusions, confidence intervals and controls.']
    (args.directory / 'conventional.md').write_text('\n'.join(lines) + '\n')
