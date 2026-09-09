#!/usr/bin/env python3
"""Build the Experiment 1 L1 figure and Experiment 2 J_area/L2 table.

Run from any directory with the repository's .venv/bin/python.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/relay-paper-matplotlib')
sys.path.insert(0, str(ROOT / 'triton'))
from stage1_counter_analysis import rank_correlation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np

spec = importlib.util.spec_from_file_location(
    'quotient_plot', ROOT / 'triton/plot-stage1-quotient-level-counters.py'
)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)

KERNELS = {
    'gemv': 'GEMV', 'gesummv': 'GESUMMV', 'mvt': 'MVT',
    'embedding_bag': 'Embedding bag', 'softmax_bias': 'Softmax + bias',
    'stencil5': '5-point stencil',
}
METRICS = {
    'matrix': 'tex_source_l2_read_requests',
    'tuolumne': 'l1_to_l2_read_requests',
}


def collect():
    """Extract complete per-layout observations, preserving source provenance."""
    observations, sources = [], []
    for platform in ('matrix',):
        scale, lanes = (32, 32) if platform == 'matrix' else (64, 64)
        component = f'issue.g{lanes}.stream.load.{scale}B'
        for kernel in KERNELS:
            path = ROOT / f'triton/experiments/results/experiment-1/{platform}/stratified-all/{kernel}/report.json'
            raw = path.read_bytes()
            report = json.loads(raw)
            assert report['complete'] and report['correct'], path
            assert report['panel']['stratification']['mode'] == 'all'
            assert report['final_experiment'] == 1
            sources.append({'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(raw).hexdigest(),
                            'target_operand': report['target_operand'],
                            'operand_shape': report['operand_shape']})
            for candidate in report['candidates']:
                assert candidate['complete']
                assert candidate['structural_validation']['all_accepted']
                assert candidate['completed_profile_launches'] == 3
                components = {c['name']: c for c in candidate['score']['components']}
                q = components[component]['raw_region_count']
                assert q == candidate['quotient_score']
                summary = candidate['counters']['steady_state']
                row = {'platform': platform, 'kernel': kernel,
                       'candidate_id': candidate['candidate_id'],
                       'component': component, 'quotient_score': q}
                if platform == 'matrix':
                    row['l1_sectors'] = summary['first_level_memory_accesses']
                    launches = summary['first_level_memory_accesses_by_launch']
                    row['l1_min'] = min(launches)
                    row['l1_max'] = max(launches)
                observations.append(row)
    return observations, sources


def plot(observations):
    style._configure_font(plt)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), layout='constrained')
    for axis, (kernel, label) in zip(axes.flat, KERNELS.items()):
        rows = [r for r in observations if r['platform'] == 'matrix' and r['kernel'] == kernel]
        x = np.asarray([r['quotient_score'] for r in rows]) / 1e6
        y = np.asarray([r['l1_sectors'] for r in rows]) / 1e6
        low = np.asarray([r['l1_min'] for r in rows]) / 1e6
        high = np.asarray([r['l1_max'] for r in rows]) / 1e6
        axis.errorbar(x, y, yerr=[y-low, high-y], fmt='o', markersize=4.5,
                      markerfacecolor='#0072B2', markeredgecolor='black',
                      markeredgewidth=.5, ecolor='#4d4d4d', capsize=2,
                      elinewidth=.8, alpha=.8, zorder=3)
        rho = rank_correlation(x.tolist(), y.tolist())
        rho_text = 'undefined' if rho is None else f'{rho:.3f}'
        axis.set_title(label, pad=5)
        axis.text(.04, .95, rf'$n={len(rows)}$; $\rho={rho_text}$',
                  transform=axis.transAxes, va='top', fontsize=10,
                  bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 1})
        axis.set_xlim(0, max(x)*1.15)
        axis.set_ylim(0, max(high)*1.25)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        for coord in (axis.xaxis, axis.yaxis):
            coord.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.grid(color='#dddddd', linewidth=.7, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
    figure.supxlabel(r'Issue quotient $Q_{32\mathrm{B}}$ (million regions)', fontsize=11)
    figure.supylabel('L1 global-load activity (million 32-byte sectors)', fontsize=11)
    figure.suptitle('H100: quotient score versus first-level memory activity', fontsize=12)
    figure.savefig(OUTPUT / 'h100-quotient-l1.pdf')
    figure.savefig(OUTPUT / 'h100-quotient-l1.png', dpi=160)
    plt.close(figure)


def collect_table():
    """Read Experiment 2's all-scope panels under the l1_to_l2 tau profile."""
    observations, sources = [], []
    for platform, metric in METRICS.items():
        for kernel in KERNELS:
            path = ROOT / f'triton/experiments/results/tau-profiles/l1_to_l2/experiment-2/{platform}/stratified-all/{kernel}/report.json'
            raw = path.read_bytes()
            report = json.loads(raw)
            assert report['complete'] and report['correct'], path
            assert report['final_experiment'] == 2
            assert report['panel']['stratification']['mode'] == 'all'
            sources.append({'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(raw).hexdigest(),
                            'active_tau': report['panel']['score_profile']['active_tau']})
            for candidate in report['candidates']:
                assert candidate['complete']
                summary = candidate['counters']['steady_state']
                assert summary['profile_launch_count'] == 3
                assert summary['dispatches_per_launch'] == [20, 20, 20]
                observations.append({'platform': platform, 'kernel': kernel,
                                     'candidate_id': candidate['candidate_id'],
                                     'j_area': candidate['j_area'],
                                     'counter': metric, 'read_requests': summary[metric]})
    return observations, sources


def table(observations):
    rows, lines = [], []
    for kernel, label in KERNELS.items():
        values = []
        for platform, metric in METRICS.items():
            panel = [r for r in observations if r['platform'] == platform and r['kernel'] == kernel]
            rho = rank_correlation([r['j_area'] for r in panel], [r['read_requests'] for r in panel])
            values.append('---' if rho is None else f'{rho:.3f}')
            rows.append({'kernel': kernel, 'platform': platform, 'n': len(panel),
                         'experiment': 2, 'stratification': 'all', 'tau': 'l1_to_l2',
                         'predictor': 'J_area', 'counter': metric, 'spearman_rho': rho})
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    medians = [statistics.median(r['spearman_rho'] for r in rows
                                if r['platform'] == platform and r['spearman_rho'] is not None)
               for platform in METRICS]
    lines.extend([r'\midrule', 'Median & ' + ' & '.join(f'{value:.3f}' for value in medians) + r' \\'])
    tex = r'''\begin{tabular}{@{}lrr@{}}
\toprule
Kernel & H100 & MI300A \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
'''
    (OUTPUT / 'l2-correlations.tex').write_text(tex)
    with (OUTPUT / 'l2-correlations.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    observations, sources = collect()
    plot(observations)
    table_observations, table_sources = collect_table()
    correlations = table(table_observations)
    (OUTPUT / 'source-data.json').write_text(json.dumps(
        {'figure': {'experiment': 1, 'stratification': 'all', 'sources': sources,
                    'observations': observations},
         'table': {'experiment': 2, 'stratification': 'all', 'tau': 'l1_to_l2',
                   'sources': table_sources, 'observations': table_observations,
                   'correlations': correlations}}, indent=2) + '\n')
    print(f'Generated figure and table from {len(sources) + len(table_sources)} complete reports in {OUTPUT}')



if __name__ == '__main__':
    main()
