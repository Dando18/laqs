from itertools import product
import random
import unittest

from relay import MatrixSpec, row_major_layout, score_layouts
from relay.gf2 import span_vectors
from relay.layouts import CanonicalLayout, LinearInnerLayout
from relay.objectives import Hyperedge, ObjectiveComponent
from relay.prepared_scoring import PreparedRegionScorer


class PreparedScoringTests(unittest.TestCase):
    def test_exact_scores_for_affine_and_nonaffine_edges_and_linear_layouts(self):
        rng = random.Random(42)
        matrix = MatrixSpec('A', (8, 16), 4, ('i', 'j'))
        points = list(product(range(8), range(16)))
        edges = [Hyperedge.make([(value & 7, value >> 3)
                                 for value in (x ^ 37 for x in span_vectors((3, 12, 80)))], weight=.3),
                 Hyperedge.make([(2, 4)], weight=1.7),
                 Hyperedge.make([(0, 0), (0, 1), (1, 0), (2, 0)], weight=.1)]
        edges += [Hyperedge.make(rng.sample(points, size), weight=.7)
                  for size in (3, 7, 16, 35, 127, 128)]
        edges = tuple(edges)
        components = tuple(ObjectiveComponent(f'd{d}', 4 << d, {'A': edges})
                           for d in range(9))
        matrices = {'A': matrix}
        prepared = PreparedRegionScorer(matrices, components)
        cache = {}
        layouts = [row_major_layout(matrix),
                   LinearInnerLayout('xor', 'A', (3, 4), (1, 3, 4, 8, 16, 32, 64), (1, 0)),
                   CanonicalLayout('inner', 'A', (2, 2), (0, 1, 0, 1), (0, 1))]
        for _ in range(20):
            word = [0] * 3 + [1] * 4
            rng.shuffle(word)
            layouts.append(CanonicalLayout('random', 'A', (3, 4), tuple(word), (1, 0)))
        for layout in layouts:
            selected = {'A': layout}
            expected = score_layouts(matrices, components, selected)
            prepared.populate(components[:3], selected, cache)
            prepared.populate(components, selected, cache)
            actual = score_layouts(matrices, components, selected, array_component_cache=cache)
            self.assertEqual(actual, expected)
        self.assertEqual(len(prepared.edges), 1)

    def test_fixed_arrays_and_empty_components(self):
        matrices = {'A': MatrixSpec('A', (4, 8), 2, ('i', 'j')),
                    'B': MatrixSpec('B', (1,), 8, ('i',), target=False)}
        components = (ObjectiveComponent('both', 8, {
            'A': (Hyperedge.make([(0, 0), (1, 1), (2, 2)]),),
            'B': (Hyperedge.make([(0,)]),)}),
            ObjectiveComponent('empty', 16, {'A': ()}))
        layouts = {name: row_major_layout(m) for name, m in matrices.items()}
        cache = {}
        PreparedRegionScorer(matrices, components).populate(components, layouts, cache)
        self.assertEqual(score_layouts(matrices, components, layouts),
                         score_layouts(matrices, components, layouts, array_component_cache=cache))

    def test_lookup_chunks_combine_with_xor_and_keep_high_address_bits(self):
        matrix = MatrixSpec('A', (32, 32), 4, ('i', 'j'))
        rows = (257,) + tuple(1 << i for i in range(1, 10))
        layout = LinearInnerLayout('cross-byte-xor', 'A', (5, 5), rows, (1, 0))
        edges = (Hyperedge.make([(x & 31, x >> 5) for x in range(513)], weight=.3),
                 Hyperedge.make([(x & 31, x >> 5) for x in span_vectors((257, 512, 4))]))
        components = tuple(ObjectiveComponent(str(d), 4 << d, {'A': edges}) for d in (0, 1, 7, 8, 9, 10))
        cache = {}
        PreparedRegionScorer({'A': matrix}, components).populate(components, {'A': layout}, cache)
        self.assertEqual(score_layouts({'A': matrix}, components, {'A': layout}),
                         score_layouts({'A': matrix}, components, {'A': layout}, array_component_cache=cache))

    def test_invalid_coordinate_is_not_hidden_by_encoding(self):
        matrix = MatrixSpec('A', (4, 4), 4, ('i', 'j'))
        components = (ObjectiveComponent('bad', 16, {'A': (Hyperedge.make([(4, 0)]),)}),)
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            PreparedRegionScorer({'A': matrix}, components).populate(
                components, {'A': row_major_layout(matrix)}, {})


class BroadcastIndexTests(unittest.TestCase):
    def test_broadcast_cache_preserves_scalar_rank_and_values(self):
        from relay.triton_frontend import TensorValue, _broadcast, UnsupportedTritonAnalysis
        for source, target in [((), (2, 3)), ((1, 3), (2, 3)), ((2, 1), (2, 3)),
                               ((3,), (2, 4, 3)), ((1, 2, 1), (4, 2, 3)),
                               ((1, 2), (10000, 2))]:
            count = 1
            for n in source:
                count *= n
            for bias in (0, 100):
                tensor = TensorValue(source, tuple(range(bias, bias + count)))
                padded = (1,) * (len(target) - len(source)) + source
                expected = []
                for coord in product(*(range(n) for n in target)):
                    source_coord = tuple(0 if n == 1 else c for c, n in zip(coord, padded))
                    expected.append(tensor.at(source_coord[len(target) - len(source):]))
                self.assertEqual(_broadcast(tensor, target).values, tuple(expected))
        with self.assertRaises(UnsupportedTritonAnalysis):
            _broadcast(TensorValue((2,), (1, 2)), (3,))
