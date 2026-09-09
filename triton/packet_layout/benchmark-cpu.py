#!/usr/bin/env python3
"""Benchmark current CPU code against a trusted frozen capture, without a GPU."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from _bootstrap import ROOT
from experiment_support import load_graph
from packet_search import select_candidates
from packet_workflow import identity
from relay import AnalysisOptions, EvaluationLimits
from relay.triton_frontend import analyze_compiled_manifest
from search_algorithms import load_tau_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True,
                        help='Existing trusted case/tau directory containing capture.json')
    parser.add_argument('--platform', choices=['matrix', 'tuolumne'], required=True)
    parser.add_argument('--tau-name', default='expert', choices=['expert', 'l1_to_l2', 'speedup'])
    parser.add_argument('--tau-profile', type=Path, default=ROOT / 'triton/experiments/tau-profiles.json')
    parser.add_argument('--selection-only', action='store_true', help='Reuse a completed graph to time just selection')
    parser.add_argument('--max-trace-contexts', type=int, default=1 << 24)
    parser.add_argument('--max-events', type=int, default=1 << 20)
    parser.add_argument('--cpu-workers', type=int, default=1)
    args = parser.parse_args()
    if args.cpu_workers < 1:
        parser.error('CPU worker count must be positive')
    stamp = json.loads((args.directory / 'capture.json').read_text())
    previous_path = args.directory / 'search.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    profile = load_tau_profile(args.platform, args.tau_profile, args.tau_name)
    timings = {}

    def completed(name, seconds):
        timings[name] = seconds
        print(f'CPU {name}: {seconds:.3f}s', flush=True)

    started = perf_counter()
    if args.selection_only:
        if previous is None or previous['capture_hash'] != stamp['capture_hash']:
            raise ValueError('selection-only requires a completed search for this capture')
        analysis = load_graph(args.directory / 'graph.pkl.gz', previous['graph_hash'])
        if analysis.hardware_profile.to_dict() != profile.to_dict():
            raise ValueError('saved graph profile differs from the requested profile')
        completed('graph_load', perf_counter() - started)
    else:
        captured = load_graph(args.directory / 'capture.pkl.gz', stamp['capture_hash'])
        completed('capture_load', perf_counter() - started)
        analysis = analyze_compiled_manifest(None, captured['manifest'], captured['grid'], captured['bound'],
            selected_config=captured['selected_config'], options=AnalysisOptions(
                hardware_profile=profile, require_native_baseline=True,
                limits=EvaluationLimits(max_trace_contexts=args.max_trace_contexts,
                                        max_dynamic_events=args.max_events, workers=args.cpu_workers)), on_phase=completed)
        analysis.require_supported()
    selected_at = perf_counter()
    selection = select_candidates(analysis, profile)
    completed('selection', perf_counter() - selected_at)
    completed('total', perf_counter() - started)
    normalized = json.loads(json.dumps(selection))
    matches = (None if previous is None else
               all(normalized[key] == previous.get(key) for key in normalized))
    print(json.dumps({'kind': 'cpu-benchmark', 'cpu_workers': args.cpu_workers, 'timings': timings,
                      'source_hash': identity()['source_hash'], 'capture_hash': stamp['capture_hash'],
                      'selection_matches_previous': matches, 'selection': selection}, indent=2))


if __name__ == '__main__':
    main()
