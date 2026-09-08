"""CPU footprint witnesses; run with .venv/bin/python from the repository root."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'triton/experiments')]
from relay import Access, HardwareProfile, MatrixSpec, MemoryEvent, SimpleRelayProblem, row_major_layout, layout_matrix_rows
from relay.objectives import EdgeFamily, Hyperedge, ScopeKey
from search_algorithms import _search

results = []
for name, shape, points, region, experiment, repeats in [
    ('contiguous row', (32, 32), [(0, j) for j in range(32)], 32, 4, 1),
    ('strided column', (32, 32), [(i, 0) for i in range(32)], 32, 4, 1),
    ('strided column repeated', (32, 32), [(i, 0) for i in range(32)], 32, 4, 256),
    ('diagonal canonical', (4, 2), [(0, 0), (1, 1)], 8, 4, 1),
    ('diagonal G_OC', (4, 2), [(0, 0), (1, 1)], 8, 6, 1),
]:
    matrix = MatrixSpec('x', shape, 4, ('i', 'j'), role='read')
    component = EdgeFamily(ScopeKey('issue', 32, 'stream', 'load'),
        {'x': (Hyperedge.make(points, weight=repeats),)},
        normalization_bytes=4 * len(points) * repeats).at_scale(region)
    profile = HardwareProfile(profile_id='witness', device={}, byte_scales=(region,),
        tau={component.name: 1.0}, fine_component=component.name)
    event = MemoryEvent.make('load', 'load', [Access('x', point) for point in points])
    problem = SimpleRelayProblem((matrix,), (event,), (), (), 'canonical')
    layouts, record = _search(problem, profile, experiment=experiment, components=(component,))
    baseline = row_major_layout(matrix)
    selected = layouts['x']
    direct = [len({layout.offset(matrix, point) * 4 // region for point in points}) * repeats
              for layout in [baseline, selected]]
    predicted = [record[key]['components'][0]['raw_region_count'] for key in ['baseline_score', 'score']]
    assert direct == predicted
    results.append({'case': name, 'repeats': repeats, 'raw_regions': direct,
                    'J': [record[key]['hardware_area'] for key in ['baseline_score', 'score']],
                    'rows': list(layout_matrix_rows(matrix, selected))})
print(json.dumps(results, indent=2))
