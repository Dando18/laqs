"""Resumable H100 source-to-shared copy diagnostic for the frozen GEMM maps."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess

from _bootstrap import HERE, ROOT
from address_experiment import parse_counters, WAVEFRONTS
from debug_suite import Budget, file_hash, run_stage
from experiment_support import digest
from packet_workflow import write_json
import stage1_nvidia_counter_analysis as counter_analysis
from stage1_nvidia_counter_analysis import COUNTER_METRICS

LAYOUTS = ('ordinary', 'large', 'smaller')
MODES = ('async', 'sync')


def sector_witness():
    def offset(layout, i, k):
        if layout == 'large':
            return 4096 * (k // 8) + 8 * i + k % 8
        if layout == 'smaller':
            return 8192 * (i // 2) + 64 * (k // 32) + 32 * (i % 2) + k % 32
        return 4096 * i + k
    return {layout: {str(group): sum(len({(2 * offset(layout, lane // 4, 8 * (lane % 4) + element)) // 32
                                         for lane in range(start, start + group) for element in range(8)})
                                     for start in range(0, 32, group)) for group in (4, 8, 32)} for layout in LAYOUTS}


def verify_instructions(sass):
    """Fail if the supposedly identical vector-copy primitives changed."""
    parts = re.split(r'Function\s*:\s*(\S+)', sass)
    found = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        match = re.search(r'copy_probeILi([012])ELb([01])E', name)
        if not match:
            continue
        layout, mode = LAYOUTS[int(match[1])], 'async' if match[2] == '1' else 'sync'
        counts = {op: len(re.findall(r'\b' + op + r'\.[A-Z.]*128\b', body)) for op in ('LDGSTS', 'LDG', 'STS')}
        expected = {'LDGSTS': 2, 'LDG': 0, 'STS': 0} if mode == 'async' else {'LDGSTS': 0, 'LDG': 2, 'STS': 2}
        if counts != expected:
            raise ValueError(f'{layout}/{mode}: unexpected vector primitives {counts}; expected {expected}')
        found[f'{layout}-{mode}'] = counts
    if len(found) != 6:
        raise ValueError(f'expected six copy specializations in SASS, found {len(found)}')
    return found


def checkpoint_valid(marker, binding, paths):
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved != {'binding': binding, 'outputs': {p.name: file_hash(p) for p in paths if p.exists()}}:
        raise ValueError(f'completed probe checkpoint changed: {marker}; use a fresh root')
    return True


def report(directory, records):
    payload = {'complete': len(records) == 12, 'records': records, 'sector_witness': sector_witness(),
               'predicted_warp_union_sectors': 256 * 128 * 4 * 2 * 16,
               'predicted_warp_requests': 256 * 128 * 4 * 2,
               'interpretation': 'A-only copy diagnostic; four-lane counts are a hypothesis, not a hardware rule. '
                   'The shared swizzle and source ownership match GEMM, but the matrix pipeline and B loads are omitted. '
                   'Profile durations are not speedups. Timings are single-process diagnostics, not benchmark claims.'}
    write_json(directory / 'report.json', payload)
    lines = ['# Matrix copy-service diagnostic', '', payload['interpretation'], '',
             f"Completed {len(records)}/12 stages. Run the same command to continue.", '',
             '| Layout | Copy | Median µs (unprofiled) | Load requests | Load sectors | Load wavefronts | L2 read requests |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for layout in LAYOUTS:
        for mode in MODES:
            key = f'{layout}-{mode}'
            timing, counters = records.get(key + '-timing'), records.get(key + '-profile', {}).get('counters', {})
            duration = f"{1000 * statistics.median(timing['samples_ms']):.3f}" if timing else 'pending'
            counts = [str(int(counters[name])) if name in counters else 'pending' for name in (
                'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
                'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', WAVEFRONTS,
                'lts__t_requests_srcunit_tex_op_read.sum')]
            lines.append(f'| {layout} | {mode} | {duration} | ' + ' | '.join(counts) + ' |')
    lines += ['', 'Warp-union prediction for every layout: 4,194,304 sectors and 262,144 requests.',
              'If only the large async variant inflates sectors, investigate the async source-to-shared service rule.',
              'If synchronous copies also inflate, investigate the more general source access/service geometry.',
              'If neither reproduces GEMM, the next distinction is its surrounding pipeline, B traffic, or cache behavior.',
              'None of these outcomes alone establishes a universal four-lane grouping.']
    (directory / 'analysis.md').write_text('\n'.join(lines) + '\n')


def run(args):
    directory = args.root.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.minutes)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, budget.interrupt)
    with (directory / 'probe.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        nvcc, ncu, cuobjdump = (shutil.which(name) for name in ('nvcc', 'ncu', 'cuobjdump'))
        if not all((nvcc, ncu, cuobjdump)):
            raise RuntimeError('load CUDA and Nsight Compute using run-copy-probe-matrix.bash')
        metrics = (*COUNTER_METRICS, WAVEFRONTS)
        config = {'sources': {p.name: file_hash(p) for p in
                  (HERE / 'copy_service_probe.cu', Path(__file__), HERE / 'address_experiment.py',
                   HERE / 'debug_suite.py', Path(counter_analysis.__file__))},
                  'nvcc': subprocess.check_output([nvcc, '--version'], text=True),
                  'ncu': subprocess.check_output([ncu, '--version'], text=True), 'metrics': list(metrics)}
        path = directory / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('probe source or tools changed; use a fresh root')
        write_json(path, config)
        binding = digest(config)
        executable, sass = directory / 'copy-probe', directory / 'codegen.sass'
        built = directory / 'build.complete.json'
        if not checkpoint_valid(built, binding, [executable, sass]):
            print(f'Building copy probe; log: {directory / "build.log"}', flush=True)
            code = run_stage([nvcc, '-std=c++17', '-arch=sm_90a', '-O3', '-lineinfo',
                              str(HERE / 'copy_service_probe.cu'), '-o', str(executable)], directory / 'build.log', budget)
            if code is None:
                report(directory, {})
                return
            if code:
                raise RuntimeError(f'probe build failed; see {directory / "build.log"}')
            sass.unlink(missing_ok=True)
            code = run_stage([cuobjdump, '--dump-sass', '--dump-resource-usage', str(executable)], sass, budget)
            if code is None:
                report(directory, {})
                return
            if code:
                raise RuntimeError(f'cuobjdump failed; see {sass}')
            write_json(directory / 'primitives.json', verify_instructions(sass.read_text()))
            write_json(built, {'binding': binding, 'outputs': {p.name: file_hash(p) for p in (executable, sass)}})
        binding = digest({'config': binding, 'binary': file_hash(executable)})
        records = {}
        for phase in ('timing', 'profile'):
            for layout in LAYOUTS:
                for mode in MODES:
                    key = f'{layout}-{mode}-{phase}'
                    output, marker = directory / f'{key}.json', directory / f'{key}.complete.json'
                    csv_path = directory / f'{key}.csv'
                    paths = [output, csv_path] if phase == 'profile' else [output]
                    if not checkpoint_valid(marker, binding, paths):
                        if budget.expired():
                            report(directory, records)
                            print('Paused; run the same command with the same root to continue.', flush=True)
                            return
                        command = [str(executable), layout, mode, phase, str(output)]
                        if phase == 'profile':
                            csv_path.unlink(missing_ok=True)
                            command = [ncu, '--csv', '--page', 'raw', '--print-units', 'base', '--metrics', ','.join(metrics),
                                '--profile-from-start', 'off', '--replay-mode', 'application', '--cache-control', 'none',
                                '--clock-control', 'none', '--log-file', str(csv_path), *command]
                        print(f'{key}; log: {directory / (key + ".log")}', flush=True)
                        code = run_stage(command, directory / f'{key}.log', budget)
                        if code is None:
                            report(directory, records)
                            print('Paused; run the same command with the same root to continue.', flush=True)
                            return
                        if code:
                            raise RuntimeError(f'{key} failed; inspect its log and rerun the same command to retry')
                        record = json.loads(output.read_text())
                        if not record['correct'] or (record['layout'], record['mode'], record['phase']) != (layout, mode, phase):
                            raise ValueError(f'invalid probe result: {output}')
                        if phase == 'profile':
                            record['counters'] = parse_counters(csv_path, 'copy_probe')
                            write_json(output, record)
                        write_json(marker, {'binding': binding, 'outputs': {p.name: file_hash(p) for p in paths}})
                    records[key] = json.loads(output.read_text())
                    report(directory, records)
        print(f'Complete: {directory / "analysis.md"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'triton/experiments/results/copy-service-debug/matrix')
    parser.add_argument('--minutes', type=float, default=25)
    args = parser.parse_args()
    if not 0 < args.minutes < float('inf'):
        parser.error('--minutes must be finite and positive')
    run(args)
