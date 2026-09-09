#!/usr/bin/env python3
"""Export J_area/L2 Spearman correlations for all saved pilot tau panels."""
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'triton'))
from stage1_counter_analysis import rank_correlation

TAUS = ['expert', 'l1_to_l2', 'speedup']
STRATA = ['all', 'issue', 'temporal']
KERNELS = ['gemv', 'gesummv', 'mvt', 'embedding_bag', 'softmax_bias', 'stencil5']
DEVICES = {'matrix': 'H100', 'tuolumne': 'MI300A'}
# Include native L2 and boundary counters, plus collected derived summaries.
METRICS = {
    'matrix': {
        'tex_source_l2_read_requests': ('Read requests', 'lts__t_requests_srcunit_tex_op_read.sum'),
        'l2_read_work': ('Read sectors', 'lts__t_sectors_srcunit_tex_op_read.sum'),
        'l2_read_misses': ('Read-sector misses', 'lts__t_sectors_op_read_lookup_miss.sum'),
    },
    'tuolumne': {
        'l1_to_l2_read_requests': ('TCP→TCC reads', 'TCP_TCC_READ_REQ_sum'),
        'l1_to_l2_write_requests': ('TCP→TCC writes', 'TCP_TCC_WRITE_REQ_sum'),
        'l1_to_l2_total_requests': ('TCP→TCC total', 'derived: TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum'),
        'l2_tag_requests': ('TCC tag requests', 'TCC_REQ_sum'),
        'second_level_read_requests': ('TCC reads', 'TCC_READ_sum'),
        'l2_hits': ('TCC hits', 'TCC_HIT_sum'),
        'l2_misses': ('TCC misses', 'TCC_MISS_sum'),
        'l2_hit_rate_percent': ('TCC hit rate', 'derived: 100 × TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)'),
    },
}


def extract():
    rows, sources = [], []
    checks = 0
    for platform, tau, experiment, strat, kernel in itertools.product(
            DEVICES, TAUS, [1, 2, 3], STRATA, KERNELS):
        path = ROOT / f'triton/experiments/results/tau-profiles/{tau}/experiment-{experiment}/{platform}/stratified-{strat}/{kernel}/report.json'
        raw = path.read_bytes()
        report = json.loads(raw)
        assert report['complete'] and report['correct'], path
        candidates = report['candidates']
        assert all(c['complete'] for c in candidates), path
        assert all(c['counters']['steady_state']['profile_launch_count'] == 3 for c in candidates), path
        assert all(c['counters']['steady_state']['dispatches_per_launch'] == [20, 20, 20] for c in candidates), path
        assert len({c['mapping_id'] for c in candidates}) == len(candidates)
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'active_tau': report['panel']['score_profile']['active_tau']})
        saved = {r['counter']: r for r in csv.DictReader(path.with_name('spearman.csv').open())
                 if r['predictor'] == 'J_area'}
        for metric, (label, native) in METRICS[platform].items():
            pairs = [(float(c['j_area']), float(c['counters']['steady_state'][metric]))
                     for c in candidates if c['counters']['steady_state'].get(metric) is not None]
            assert len(pairs) == len(candidates), (path, metric)
            x, y = zip(*pairs)
            rho = rank_correlation(x, y)
            reason = ('fewer than two layouts' if len(x) < 2 else
                      'constant J_area' if len(set(x)) < 2 else
                      'constant counter' if len(set(y)) < 2 else '')
            if metric in saved:
                previous = saved[metric]['spearman_rho']
                assert (rho is None and not previous) or (rho is not None and abs(rho-float(previous)) < 1e-12)
                checks += 1
            rows.append(dict(device=DEVICES[platform], tau=tau, experiment=experiment,
                             stratification=strat, kernel=kernel, counter=metric,
                             label=label, native_counter=native, n=len(x), spearman_rho=rho,
                             undefined_reason=reason))
    return rows, sources, checks


def fmt(x):
    return '—' if x is None else f'{x:.3f}'


def main():
    rows, sources, checks = extract()
    medians, sections = [], []
    for platform, tau, experiment, strat in itertools.product(DEVICES, TAUS, [1, 2, 3], STRATA):
        device = DEVICES[platform]
        panel = [r for r in rows if (r['device'], r['tau'], r['experiment'], r['stratification']) == (device, tau, experiment, strat)]
        table_rows = []
        for metric, (label, _) in METRICS[platform].items():
            selected = {r['kernel']: r for r in panel if r['counter'] == metric}
            values = [r['spearman_rho'] for r in selected.values() if r['spearman_rho'] is not None]
            median = statistics.median(values) if values else None
            medians.append(dict(device=device, tau=tau, experiment=experiment,
                                stratification=strat, counter=metric, label=label,
                                median_spearman_rho=median, defined_kernels=len(values)))
            table_rows.append([label, *[fmt(selected[k]['spearman_rho']) for k in KERNELS], fmt(median), str(len(values))])
        sections.append(dict(device=device, tau=tau, experiment=experiment, stratification=strat,
                             rows=table_rows, counts={k: next(r['n'] for r in panel if r['kernel']==k) for k in KERNELS}))
    for filename, data in [('per-kernel.csv', rows), ('medians.csv', medians)]:
        with (OUT / filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (OUT / 'provenance.json').write_text(json.dumps({'sources': sources, 'existing_correlations_verified': checks}, indent=2)+'\n')
    intro = '''J_area versus all collected L2-related counters for Experiments 1–3, all three tau profiles, both devices, and all three stratifications. Each per-kernel rho is computed across distinct layout mappings using average ranks for ties. Counter observations are the saved median across three profiler-launch medians (20 steady-state dispatches per launch). The summary is the median of defined per-kernel rho values, not a mean, not a pooled correlation, and not a median across launches. Undefined values are excluded from the median; the final column gives its kernel count. Bias+ReLU is excluded as requested. Undefined values can result from constant scores or counters; the CSV records the reason. Profiles reuse the pilot measurements with different J_area weights; learned-profile results describe the pilot data used for tuning.'''
    aliases = '''Aliases are shown once: on H100, l1_miss_demand_to_l2 equals tex_source_l2_read_requests, and l1_to_l2_read_traffic equals l2_read_work. On MI300A, l1_miss_demand_to_l2 equals l1_to_l2_read_requests. MI300A total requests and hit rate are derived summaries, included alongside every native L2/boundary counter. L1-only and HBM counters are outside this report.'''
    headers = ['Counter', 'GEMV', 'GESUMMV', 'MVT', 'Embedding bag', 'Softmax+bias', 'Stencil5', 'Median', 'Defined kernels']
    md = ['# Pilot J_area/L2 Spearman correlations', intro, aliases, '## Counter definitions']
    definitions = []
    for platform, metrics in METRICS.items():
        for field, (label, native) in metrics.items():
            definitions.append(f'{DEVICES[platform]} — {label}: {native} ({field})')
    md += ['\n'.join(f'- {x}' for x in definitions)]
    html_sections = []
    for s in sections:
        title = f"{s['device']} · {s['tau']} · Experiment {s['experiment']} · {s['stratification']} stratification"
        counts = ', '.join(f'{k}: {v}' for k, v in s['counts'].items())
        md += ['## '+title, 'Layout counts: '+counts,
               '| '+' | '.join(headers)+' |', '|'+ '|'.join(['---']*len(headers))+'|',
               *['| '+' | '.join(row)+' |' for row in s['rows']]]
        cells = lambda tag, values: ''.join(f'<{tag}>{html.escape(v)}</{tag}>' for v in values)
        html_sections.append(f'''<section data-device="{s['device']}" data-tau="{s['tau']}" data-experiment="{s['experiment']}" data-stratification="{s['stratification']}"><h2>{html.escape(title)}</h2><p class="counts">Layout counts: {html.escape(counts)}</p><div class="scroll"><table><thead><tr>{cells('th', headers)}</tr></thead><tbody>{''.join('<tr>'+cells('td', row)+'</tr>' for row in s['rows'])}</tbody></table></div></section>''')
    (OUT / 'correlations.md').write_text('\n\n'.join(md)+'\n')
    filters = []
    for key, options in [('device', list(DEVICES.values())), ('tau', TAUS), ('experiment', ['1','2','3']), ('stratification', STRATA)]:
        filters.append(f'<label>{key.capitalize()} <select id="{key}"><option value="">All</option>'+''.join(f'<option>{x}</option>' for x in options)+'</select></label>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pilot J_area / L2 correlations</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;color:#18222c}h1{font-size:28px}h2{font-size:21px;margin-top:36px}.filters{position:sticky;top:0;background:#eef3f7;padding:16px;display:flex;gap:22px;flex-wrap:wrap;border:1px solid #c9d2da}select{font:inherit;padding:4px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #d4dce3;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#edf3f7}td:nth-last-child(2){font-weight:bold;background:#f2f6fa}.counts{font-size:14px;color:#46525e}section[hidden]{display:none}summary{cursor:pointer}li{overflow-wrap:anywhere}@media print{.filters{position:static}section{break-inside:avoid}}</style><h1>Pilot J<sub>area</sub> versus L2: Spearman correlations</h1>'''
    page += '<p>'+html.escape(intro)+'</p><p>'+html.escape(aliases)+'</p>'
    page += '<p><a href="per-kernel.csv">Per-kernel CSV</a> · <a href="medians.csv">Median CSV</a> · <a href="correlations.md">Complete Markdown tables</a></p>'
    page += '<details><summary>Native counter definitions</summary><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in definitions)+'</ul></details>'
    page += '<div class="filters">'+''.join(filters)+'</div>'+''.join(html_sections)
    page += '''<script>const keys=['device','tau','experiment','stratification'];function filter(){document.querySelectorAll('section').forEach(s=>{s.hidden=keys.some(k=>document.getElementById(k).value && document.getElementById(k).value!==s.dataset[k]);});}keys.forEach(k=>document.getElementById(k).addEventListener('change',filter));</script></html>'''
    (OUT / 'correlations.html').write_text(page)
    print(f'{len(sources)} reports; {len(rows)} per-kernel correlations; {len(medians)} medians; {checks} saved correlations verified')


if __name__ == '__main__':
    main()
