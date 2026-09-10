"""Collect the TritonBench packet suite across resumable single-GPU allocations."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from _bootstrap import HERE, ROOT, activate
from experiment_support import digest
from packet_workflow import identity, runtime_identity, write_json
from tritonbench_cases import CASE_BY_ID
from packet_cases import CASES

STAGES = ('capture', 'search', 'validate', 'tune', 'evaluate', 'profile')
OUTPUTS = {'capture': ('capture.json',), 'search': ('search.json', 'graph-ready.json'),
           'validate': ('validated.json',), 'tune': ('choice.json',),
           'evaluate': ('evaluation.json', 'report.json', 'analysis.md'), 'profile': ('profile.json',),
           'conventional': ('conventional.json', 'conventional.md')}


def stages(args):
    if args.manual:
        return ('capture', 'search', 'conventional')
    return STAGES if args.profile_addresses else STAGES[:-1]


def case_order(case):
    # Collect inexpensive controls first, then small cases before large GEMMs.
    operator, config = case.split('--', 1)
    return (operator not in ('vector_add', 'vector_exp', 'low_mem_dropout'),
            config != 'small', operator in ('gemm', 'bf16xint16_gemm', 'int4_gemm', 'fp8_gemm'),
            operator, config)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--platform', choices=['tuolumne', 'matrix'], required=True)
    result.add_argument('--root', type=Path, default=ROOT / 'triton/experiments/results/packet-debug-split')
    result.add_argument('--cases', nargs='+', choices=tuple(CASES), default=sorted(CASE_BY_ID, key=case_order))
    result.add_argument('--manual', action='store_true', help='compare exact source/pass realizations; row/column also receives equal-budget tuning')
    result.add_argument('--minutes', type=float, default=25, help='work budget; leave scheduler cleanup margin')
    result.add_argument('--cpu-workers', type=int, default=8)
    result.add_argument('--grammar', choices=['split', 'split-chunks'], default='split')
    result.add_argument('--selection', choices=['analytical', 'measured'], default='analytical')
    result.add_argument('--address-reference', type=Path,
                        help='reuse a prior packet root for the six-variant address comparison')
    result.add_argument('--profile-addresses', action='store_true',
                        help='append separate Nsight Compute counters (Matrix address comparison only)')
    result.add_argument('--tau-name', choices=['expert', 'l1_to_l2', 'speedup'], default='expert')
    result.add_argument('--tau-profile', type=Path, default=ROOT / 'triton/experiments/tau-profiles.json')
    for name, default in [('processes', 3), ('samples', 21), ('iterations', 50), ('warmup', 10),
                          ('max-trace-contexts', 1 << 24), ('max-events', 1 << 20)]:
        result.add_argument('--' + name, type=int, default=default)
    result.add_argument('--minimum-gain', type=float, default=.01)
    result.add_argument('--retry-failed', action='store_true')
    result.add_argument('--dry-run', action='store_true', help='print commands without requiring a GPU or writing results')
    return result


def common_arguments(args):
    result = ['--platform', args.platform, '--root', str(args.root), '--resume']
    if args.address_reference:
        result += ['--address-reference', str(args.address_reference.resolve())]
    if args.profile_addresses:
        result += ['--profile-addresses']
    for name in ('grammar', 'selection', 'tau-name', 'tau-profile', 'processes', 'samples',
                 'iterations', 'warmup', 'minimum-gain', 'max-trace-contexts', 'max-events'):
        result += ['--' + name, str(getattr(args, name.replace('-', '_')))]
    return result


def command(args, case, stage):
    environment = '.venv' if args.platform == 'tuolumne' else '.venv-matrix'
    python = ROOT / '.venv/bin/python' if stage == 'search' else ROOT / 'triton' / environment / 'bin/python'
    return [str(python), str(HERE / 'run.py'), '--stage', stage, '--case', case,
            '--cpu-workers', str(args.cpu_workers), *common_arguments(args)]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_binding(directory, stage, config_hash):
    sequence = ('capture', 'search', 'conventional') if stage == 'conventional' else STAGES
    index = sequence.index(stage)
    previous = directory / '.debug-stages' / f'{sequence[index - 1]}.json' if index else None
    return digest({'config': config_hash, 'stage': stage,
                   'previous': file_hash(previous) if previous else None})


def stage_complete(directory, stage, config_hash):
    marker = directory / '.debug-stages' / f'{stage}.json'
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    expected = {'binding': stage_binding(directory, stage, config_hash),
                'outputs': {name: file_hash(directory / name) for name in OUTPUTS[stage]}}
    if record != expected:
        raise ValueError(f'completed {stage} checkpoint changed: {directory}; use a fresh root')
    return True


def commit_stage(directory, stage, config_hash):
    write_json(directory / '.debug-stages' / f'{stage}.json',
               {'binding': stage_binding(directory, stage, config_hash),
                'outputs': {name: file_hash(directory / name) for name in OUTPUTS[stage]}})


class Budget:
    def __init__(self, minutes):
        self.deadline = time.monotonic() + minutes * 60
        self.signal = None

    def interrupt(self, signum, _frame):
        self.signal = signum

    def expired(self):
        return self.signal is not None or time.monotonic() >= self.deadline


def run_stage(invocation, log_path, budget):
    """Stop the entire stage process group, including CPU and timing workers."""
    with log_path.open('a') as log:
        log.write(f'\n{time.ctime()}: {shlex.join(invocation)}\n')
        log.flush()
        process = subprocess.Popen(invocation, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if budget.expired():
                    return None
                time.sleep(.25)
            return None if process.returncode and budget.expired() else process.returncode
        finally:
            # Also stop orphaned descendants after a failed stage leader exits.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def save_summary(args, state, directory):
    write_json(directory / 'debug-state.json', state)
    lines = ['# Resumable TritonBench debug suite', '',
             f'Platform: {args.platform}. Grammar: {args.grammar}. Selection: {args.selection}.',
             'Only complete cases have final held-out results; failed cases remain listed.', '',
             '| Case | Status | Stage | Selected speedup | Detail |',
             '| --- | --- | --- | ---: | --- |']
    for case in args.cases:
        record = state['cases'][case]
        speedup = ''
        if record['status'] == 'complete' and args.manual:
            report = json.loads((directory / case / args.tau_name / 'conventional.json').read_text())
            selected = report['analytical_top1'] + '-explicit'
            value = report['fixed_results'].get(selected, {})
            speedup = f"{value['speedup']:.4f}× ({selected}, fixed)" if 'speedup' in value else 'no explicit result'
        elif record['status'] == 'complete':
            report = json.loads((directory / case / args.tau_name / 'report.json').read_text())
            selected = report['choice']['selected']
            speedup = f"{report['evaluation']['candidates'][selected]['speedup']:.4f}× ({selected})"
        detail = record.get('reason', '')
        if record['status'] == 'complete':
            report_name = 'conventional.md' if args.manual else 'analysis.md'
            detail = f'[report]({case}/{args.tau_name}/{report_name})'
        lines.append(f"| {case} | {record['status']} | {record.get('stage', '')} | {speedup} | {detail} |")
    (directory / 'debug-summary.md').write_text('\n'.join(lines) + '\n')


def collect(args, directory, config_hash, budget):
    state_path = directory / 'debug-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'cases': {case: {'status': 'pending', 'attempts': 0} for case in args.cases}}
    for record in state['cases'].values():
        if record['status'] == 'running' or (args.retry_failed and record['status'] == 'failed'):
            record.update(status='paused', reason='resuming previous attempt')
    order = sorted(args.cases, key=lambda case: (state['cases'][case]['attempts'], args.cases.index(case)))
    for case in order:
        record = state['cases'][case]
        if budget.expired():
            break
        if record['status'] == 'failed':
            continue
        case_dir = directory / case / args.tau_name
        case_dir.mkdir(parents=True, exist_ok=True)
        record['attempts'] += 1
        for stage in stages(args):
            if stage_complete(case_dir, stage, config_hash):
                continue
            if budget.expired():
                record.update(status='paused', stage=stage, reason='allocation work budget ended')
                break
            record.update(status='running', stage=stage, reason='')
            save_summary(args, state, directory)
            log = case_dir / f'debug-{stage}.log'
            print(f'{case}: {stage} ({args.cpu_workers} CPU workers); log: {log}', flush=True)
            code = run_stage(command(args, case, stage), log, budget)
            if code is None:
                record.update(status='paused', reason='interrupted; completed checkpoints retained')
                break
            if code:
                record.update(status='failed', reason=f'exit {code}; [{stage} log]({case}/{args.tau_name}/{log.name})')
                print(f'{case}: {stage} failed (exit {code}); continuing with other cases', flush=True)
                break
            commit_stage(case_dir, stage, config_hash)
        else:
            record.update(status='complete', stage=stages(args)[-1], reason='')
            print(f'{case}: complete', flush=True)
        save_summary(args, state, directory)
    save_summary(args, state, directory)
    counts = {status: sum(r['status'] == status for r in state['cases'].values())
              for status in ('complete', 'failed', 'paused', 'pending')}
    print(f'{counts}\nSummary: {directory / "debug-summary.md"}', flush=True)
    print('Run the same command with the same root to continue; --retry-failed retries failures.', flush=True)
    return 1 if counts['failed'] and not (counts['paused'] or counts['pending']) else 0


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if args.manual and (args.profile_addresses or args.grammar != 'split'
                        or any(c.split('--')[0] not in ('row_column', 'sum', 'gemm') for c in args.cases)):
        arg_parser.error('--manual requires split grammar and row_column, sum or gemm cases; counters are separate')
    if args.profile_addresses and (args.platform != 'matrix' or not args.address_reference):
        arg_parser.error('--profile-addresses requires Matrix and --address-reference')
    if args.address_reference and (args.grammar != 'split' or args.selection != 'analytical'):
        arg_parser.error('the frozen address comparison requires split grammar and analytical selection')
    if min(args.minutes, args.cpu_workers, args.processes, args.samples, args.iterations,
           args.warmup, args.max_trace_contexts, args.max_events) <= 0:
        arg_parser.error('budgets, worker counts and measurement counts must be positive')
    if not 0 < args.minimum_gain < 1:
        arg_parser.error('minimum gain must lie strictly between zero and one')
    args.cases = list(dict.fromkeys(args.cases))
    args.root, args.tau_profile = args.root.resolve(), args.tau_profile.resolve()
    if args.dry_run:
        print(f'{len(args.cases)} cases; {len({CASES[c].operator for c in args.cases})} operators; '
              f'{args.minutes:g} minutes; {args.cpu_workers} CPU workers')
        for case in args.cases:
            for stage in stages(args):
                print(shlex.join(command(args, case, stage)))
        return 0
    if len(os.sched_getaffinity(0)) < args.cpu_workers:
        arg_parser.error('CPU affinity is smaller than --cpu-workers; reserve and bind enough cores')
    budget = Budget(args.minutes)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, budget.interrupt)
    directory = args.root / args.platform
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'debug-suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'another runner owns {directory}; use one allocation per platform/root')
        # Fail once on a missing GPU/toolchain, before marking any case failed.
        activate(args.platform)
        config = {'schema': 'laqs.packet.debug.v1', 'arguments': common_arguments(args),
                  'cases': args.cases, 'stages': list(stages(args)), 'source': identity(), 'tau_sha256': file_hash(args.tau_profile),
                  'runtime': runtime_identity()}
        if args.address_reference:
            from address_experiment import reference
            from types import SimpleNamespace
            config['references'] = {}
            for case in args.cases:
                old_dir, _, _ = reference(SimpleNamespace(**{**vars(args), 'case': case}))
                config['references'][case] = {name: file_hash(old_dir / name)
                                             for name in ('capture.json', 'search.json')}
        manifest = directory / 'debug-suite.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != config:
                raise SystemExit('sources, GPU/toolchain or experiment settings changed; use a fresh --root')
        else:
            if any((directory / case / args.tau_name).exists() for case in args.cases):
                raise SystemExit('result root contains cases without a debug manifest; use a fresh --root')
            write_json(manifest, config)
        return collect(args, directory, digest(config), budget)


if __name__ == '__main__':
    raise SystemExit(main())
