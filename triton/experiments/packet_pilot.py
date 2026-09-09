"""Experiment 1: canonical packet-preserving layouts and native counter launches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / 'triton/packet_layout'
if str(PACKET) not in sys.path:
    sys.path.insert(0, str(PACKET))

PROTOCOL = 'laqs.pilot.packet.v1'


def fingerprint():
    """Bind resumable profiles to the actual source, compiler and plugins."""
    import torch
    import triton
    from experiment_support import compiler_source_identity, source_identity
    from packet_runtime import plugin_path
    from relay.triton_frontend import _default_plugin_path

    sources = dict(source_identity()['sources'])
    paths = [*ROOT.joinpath('triton').glob('stage1_*.py'),
             ROOT / 'triton/run-stage1-kernel-case.py',
             ROOT / 'triton/experiments/run.py',
             ROOT / 'triton/experiments/layout_panels.py',
             ROOT / 'triton/experiments/tau-profiles.json',
             *ROOT.joinpath('triton/experiments').glob('packet_pilot*.py'),
             *PACKET.glob('*.py'), *PACKET.glob('*.cpp'), *PACKET.glob('*.h')]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    # rocprof replaces the marketing name with 'AMD Radeon Graphics'.
    # Architectural/resource properties remain stable across profiler launches.
    device = {'architecture': getattr(properties, 'gcnArchName', None) or f'sm{properties.major}{properties.minor}',
              'multiprocessors': properties.multi_processor_count,
              'total_memory': properties.total_memory}
    return {'protocol': PROTOCOL, 'sources': sources,
            'compiler': compiler_source_identity(triton.__file__),
            'triton': triton.__version__, 'torch': torch.__version__,
            'runtime': torch.version.hip or torch.version.cuda,
            'device': device,
            'plugins': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (plugin_path(), _default_plugin_path())}}


def freeze_run(args, directory):
    from stage1_counter_sweep import write_json
    identity = fingerprint()
    # Panel settings and per-launch settings are also part of each checkpoint key.
    path = directory / 'packet-identity.json'
    if not path.exists() and ((directory / 'report.json').exists() or (directory / 'profiles/panel.json').exists()):
        raise ValueError('existing pilot results have no packet identity; use a fresh results root')
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('packet pilot source/compiler/device changed; use a fresh results root')
    write_json(path, identity)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def rank_packet_operand(matrix, logical_operand, default_source, *, args,
                        make_output, make_launch, validate, inner_tile_shapes,
                        automatic_analysis, automatic_target_name, execution_layout_spec):
    """Profile a frozen mapping, or prepare a panel, without legacy layout lowering."""
    import torch
    from relay import row_major_layout, layout_matrix_rows
    from layout_contract import RuntimeLayout, preserves_vector_bits
    from packet_runtime import structured_layouts, statistics, primitive_rejections
    from stage1_common import (pack_tensor, stable_id, execution_layout_from_compiled,
                               execution_layout_record)
    from layout_panels import stratified_experiment_panel
    from stage1_counter_sweep import write_json

    if matrix.rank != 2:
        raise ValueError('packet pilot requires a two-dimensional target')
    ordinary = tuple(layout_matrix_rows(matrix, row_major_layout(matrix)))
    # The pilot factories pass their varied operand first, except softmax+bias.
    argument = 1 if args.case == 'softmax_bias' else 0
    def runtime(rows):
        return RuntimeLayout(matrix.name, argument, matrix.shape,
                             (matrix.shape[1], 1), matrix.shape, tuple(rows))

    directory = args.json.parent / 'codegen'
    output = make_output()
    baseline_launch = make_launch(default_source, output, ordinary)
    with structured_layouts((runtime(ordinary),), inspect=True):
        baseline = baseline_launch()
    torch.cuda.synchronize()
    validate('packet ordinary', output)
    baseline_stats = statistics(baseline, directory, 'ordinary')
    sites = baseline_stats['packet_sites']
    if not sites:
        raise ValueError('packet inspection found no target load sites')
    packet_bits = max((int(site['vector_elements']) - 1).bit_length() for site in sites)
    default = {'layout': 'row_major', 'a_rows': list(ordinary),
               'mapping_id': stable_id('mapping', list(ordinary)), 'compiled_codegen': baseline_stats}
    def checked(rows, label):
        source = pack_tensor(logical_operand, rows).to('cuda')
        output = make_output()
        ordinary_launch = make_launch(source, output, ordinary)
        with structured_layouts((runtime(rows),)):
            compiled = ordinary_launch()
        torch.cuda.synchronize()
        validate(label, output)
        codegen = statistics(compiled, directory / label, 'candidate')
        reasons = primitive_rejections(baseline_stats, codegen)
        def execution(kernel):
            blocked, layout = execution_layout_from_compiled(kernel, *execution_layout_spec)
            return execution_layout_record(blocked, layout)
        reference_execution, candidate_execution = execution(baseline), execution(compiled)
        if reference_execution != candidate_execution:
            reasons.append('execution layout changed')
        memory_matches = baseline_stats['memory_primitives'] == codegen['memory_primitives']
        if not memory_matches:
            reasons.append('final memory primitive counts changed')
        validation = {
            'accepted': not reasons, 'rejections': reasons,
            'execution_layout_matches_reference': reference_execution == candidate_execution,
            'reference_execution_layout': reference_execution,
            'candidate_execution_layout': candidate_execution,
            'load_instruction_structure_matches_reference': memory_matches,
            'store_instruction_structure_matches_reference': memory_matches,
            'load_instruction_structure_nonempty': bool(codegen['memory_primitives']),
            'no_candidate_spills': codegen['n_spills'] <= baseline_stats['n_spills'],
            'packet_sites': codegen['packet_sites'],
        }
        return ordinary_launch, codegen, validation

    if args.profile_rows is None:
        if automatic_analysis is None or args.counter_panel != 'experiment1_gc_whole':
            raise ValueError('packet panel needs the ordinary automatic graph for Experiment 1')
        widths = [int(event.meta('vector_elements', '1')) for event in automatic_analysis.events
                  if any(a.array == automatic_target_name for a in event.accesses)]
        if not widths or max(widths) != 1 << packet_bits:
            raise ValueError('manifest issue packets disagree with packet-plugin contracts')
        panel = stratified_experiment_panel(
            matrix, automatic_analysis, automatic_target_name, inner_tile_shapes,
            experiment=1, samples=args.panel_samples, seed=args.panel_seed,
            platform=args.counter_platform, stratification=args.panel_stratification,
            pool_multiplier=args.panel_pool_multiplier, packet_bits=packet_bits)
        accepted, rejected = [], []
        for candidate in panel['candidates']:
            try:
                _, codegen, validation = checked(tuple(candidate['a_rows']), candidate['candidate_id'])
                record = {**candidate, 'packet_preflight': validation}
                if not validation['accepted']:
                    rejected.append(record)
                else:
                    accepted.append(record)
            except Exception as error:
                rejected.append({**candidate, 'packet_preflight': {
                    'accepted': False, 'rejections': [f'{type(error).__name__}: {error}']}})
            write_json(args.json.parent / 'packet-preflight.json', {
                'accepted': accepted, 'rejected': rejected,
                'complete': len(accepted) + len(rejected) == len(panel['candidates'])})
        if not accepted or not any(c['a_rows'] == list(ordinary) for c in accepted):
            raise ValueError('ordinary packet layout did not pass preflight')
        panel.update(candidates=accepted, rejected_candidates=rejected,
                     proposed_mapping_count=len(accepted) + len(rejected),
                     profiled_mapping_policy='pre-counter compiler validation; no replacement of rejected mappings')
        panel.update(mode=args.counter_panel, objective='J_area', protocol=PROTOCOL,
                     packet_sites=sites, ordinary_codegen=baseline_stats,
                     source_identity=fingerprint())
        # This hash binds profiling to the captured scores, mappings and native contract.
        panel['packet_panel_hash'] = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
        return {'default': default, 'counter_panel': panel, 'correct': True}

    if args.profile_panel is None:
        raise ValueError('packet profiling requires the frozen --profile-panel')
    panel = json.loads(args.profile_panel.read_text())['panel']
    expected_hash = hashlib.sha256(json.dumps(
        {k: v for k, v in panel.items() if k != 'packet_panel_hash'}, sort_keys=True).encode()).hexdigest()
    if panel.get('packet_panel_hash') != expected_hash:
        raise ValueError('packet profile panel content hash changed')
    observed_identity = fingerprint()
    if panel.get('source_identity') != observed_identity:
        write_json(args.json.parent / 'observed-identity.json', observed_identity)
        changed = [key for key in observed_identity if observed_identity[key] != panel['source_identity'].get(key)]
        raise ValueError(f'packet source/compiler identity changed: {changed}')
    if sites != panel['packet_sites']:
        raise ValueError('native packet contracts changed since panel capture')
    rows = tuple(args.profile_rows)
    matches = [c for c in panel['candidates'] if c['candidate_id'] == args.profile_candidate_id]
    if len(matches) != 1 or matches[0]['a_rows'] != list(rows):
        raise ValueError('profile mapping is not in the frozen packet panel')
    candidate = matches[0]
    if candidate['quotient_score'] != args.profile_quotient_score:
        raise ValueError('profile quotient differs from captured packet score')
    if not preserves_vector_bits(rows, ordinary, packet_bits):
        raise ValueError('candidate changes the protected packet bits')
    layouts = (runtime(rows),)
    try:
        ordinary_launch, codegen, validation = checked(rows, candidate['candidate_id'])
        write_json(args.json.parent / 'packet-validation.json', validation)
        if not validation['accepted']:
            raise ValueError(f"previously accepted packet candidate rejected: {validation['rejections']}")
    except Exception as error:
        write_json(args.json.parent / 'packet-rejection.json', {
            'candidate_id': candidate['candidate_id'], 'a_rows': list(rows),
            'error': f'{type(error).__name__}: {error}', 'accepted': False})
        raise
    # Reuse the compiled launch without repeated plugin hashing in the measured loop.
    with structured_layouts(layouts):
        for _ in range(args.profile_warmup):
            ordinary_launch()
        torch.cuda.synchronize()
        for _ in range(args.profile_iterations):
            ordinary_launch()
        torch.cuda.synchronize()
    target = {**candidate, 'cache_mode': 'warm', 'compiled_codegen': codegen,
              'structural_validation': validation, 'issue_quotient_score_verified': True,
              'score_verification': 'source-bound captured packet graph and mapping',
              'packet_panel_hash': panel['packet_panel_hash']}
    return {'default': default, 'profile_target': target, 'correct': True}
