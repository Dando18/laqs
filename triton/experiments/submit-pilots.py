#!/usr/bin/env python3
"""Submit six-kernel pilot panels; Experiment 1 uses packet codegen, 3 uses legacy."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CASES = ('softmax_bias', 'embedding_bag', 'gemv', 'mvt', 'gesummv', 'stencil5')


def commands(args):
    for experiment in args.experiments:
        for case in args.cases:
            name = f'pilot-v2-e{experiment}-{case}-{args.platform}'
            log = args.results_root / f'experiment-{experiment}' / args.platform / 'stratified-all/logs'
            worker = [str(ROOT / f'triton/experiments/run-{args.platform}-job.bash'),
                      '--experiment', str(experiment), '--platform', args.platform,
                      '--case', case, '--stratification', 'all', '--layouts', str(args.layouts),
                      '--seed', '0', '--pool-multiplier', '20', '--profile-launches', '3',
                      '--results-root', str(args.results_root), '--plots-root', str(args.plots_root)]
            if args.platform == 'tuolumne':
                command = ['flux', 'submit', '-N1', '-n1', '-g1', '-q', args.queue,
                           '-t', args.wall_time or '8h', '--cwd', str(ROOT), '--job-name', name,
                           '--output', str(log / (name + '-{{id}}.out'))] + worker
            else:
                command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--gpus=1',
                           f'--partition={args.queue}', f'--time={args.wall_time or "08:00:00"}',
                           f'--chdir={ROOT}', f'--job-name={name}', f'--output={log / (name + "-%j.out")}'] + worker
            yield log, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('tuolumne', 'matrix'), required=True)
    parser.add_argument('--experiments', type=int, choices=(1, 3), nargs='+', default=[1])
    parser.add_argument('--cases', choices=CASES, nargs='+', default=CASES)
    parser.add_argument('--layouts', type=int, default=100)
    parser.add_argument('--queue', default='pbatch')
    parser.add_argument('--wall-time')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'triton/experiments/results/pilot-v2')
    parser.add_argument('--plots-root', type=Path, default=ROOT / 'triton/experiments/plots/pilot-v2')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.layouts < 1 or len(set(args.experiments)) != len(args.experiments) or len(set(args.cases)) != len(args.cases):
        parser.error('use positive layouts and distinct experiments/cases')
    args.results_root = args.results_root.resolve()
    args.plots_root = args.plots_root.resolve()
    for log, command in commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            log.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
