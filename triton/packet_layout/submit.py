#!/usr/bin/env python3
"""Submit the declared packet-layout diagnostic panel, preserving old suites."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from _bootstrap import ROOT, HERE
from packet_cases import CASES, DEFAULT_CASES


def commands(args):
    """Yield scheduler requests; the caller supplies dependencies from returned IDs."""
    queue = args.queue or ('pbatch')
    common = ['--platform', args.platform, '--root', str(args.root.resolve()),
              '--tau-name', args.tau_name, '--selection', args.selection]
    python = ROOT / ('triton/.venv/bin/python' if args.platform == 'tuolumne' else 'triton/.venv-matrix/bin/python')
    for case in args.cases:
        stages = ['capture', 'search', 'validate', 'tune', 'evaluate']
        if case.startswith('row_column--'):
            stages.append('conventional')
        for stage in stages:
            cpu = stage == 'search'
            name = f'packet-{stage}-{case}'
            logs = args.root.resolve() / 'logs' / args.platform
            duration = '4h' if cpu else '1h'
            command = ([str(ROOT / '.venv/bin/python'), str(HERE / 'run.py')] if cpu
                       else [str(HERE / f'job-{args.platform}.bash')])
            command += ['--stage', stage, '--case', case, *common]
            cores = args.cpu_workers if cpu else 1
            if cpu:
                command += ['--cpu-workers', str(cores)]
            if args.platform == 'tuolumne':
                scheduler = ['flux', 'submit', '-N1', '-n1', f'-c{cores}', '-q', queue, '-t', duration,
                             '--cwd', str(ROOT), '--job-name', name, '--output', str(logs / f'{name}.out'),
                             '--error', str(logs / f'{name}.err')]
                if not cpu:
                    scheduler += ['-g1']
            else:
                # Matrix accepted GPU allocations but rejected CPU-only requests.
                scheduler = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', f'--cpus-per-task={cores}', '--gpus=1',
                             f'--partition={queue}', f'--time={int(duration[:-1]) * 60}', f'--chdir={ROOT}',
                             f'--job-name={name}', f'--output={logs / (name + "-%j.out")}',
                             f'--error={logs / (name + "-%j.err")}']
                if cpu:
                    command = ['--wrap', shlex.join(command)]
            yield case, stage, scheduler, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['matrix', 'tuolumne'], required=True)
    parser.add_argument('--root', type=Path, default=ROOT / 'triton/experiments/results/packet-v1')
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(DEFAULT_CASES))
    parser.add_argument('--tau-name', choices=['expert', 'l1_to_l2', 'speedup'], default='expert')
    parser.add_argument('--selection', choices=['analytical', 'measured'], default='measured')
    parser.add_argument('--queue')
    parser.add_argument('--cpu-workers', type=int, default=4,
                        help='Cores requested and processes used for each CPU search job')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.cpu_workers < 1:
        parser.error('CPU worker count must be positive')
    if Path.cwd().resolve() != ROOT.resolve():
        parser.error('run from the RELAY repository root')
    jobs, previous = [], {}
    for case, stage, scheduler, command in commands(args):
        if case in previous:
            dependency = f'afterany:{previous[case]}'
            scheduler += ['--dependency', dependency] if args.platform == 'tuolumne' else [f'--dependency={dependency}']
        invocation = scheduler + command
        if args.dry_run:
            job = f'DRY{len(jobs)+1}'
            print(shlex.join(invocation))
        else:
            (args.root / 'logs' / args.platform).mkdir(parents=True, exist_ok=True)
            job = subprocess.check_output(invocation, text=True).strip().split(';', 1)[0]
            print(case, stage, job)
        previous[case] = job
        jobs.append({'case': case, 'stage': stage, 'job': job, 'command': invocation})
        if not args.dry_run:
            path = args.root / f'submission-{args.platform}.json'
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(jobs, indent=2) + '\n')
            temp.replace(path)


if __name__ == '__main__':
    main()
