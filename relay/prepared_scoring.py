"""Reusable exact region counts for concrete binary-linear layouts."""
from .gf2 import rank, rref_basis
from .layouts import AffineAccessLayout, CanonicalLayout, LinearInnerLayout, layout_matrix_rows


class PreparedRegionScorer:
    """Encode each edge once, using a basis only for proved affine cosets.

    For E = a + S and physical address matrix A, the number of d-bit
    regions is 2**rank((A S)[d:]). Translation changes region identities,
    but not their count. Non-affine edges retain exact point enumeration.
    No weights, candidate layouts, or objective components are discarded.
    """

    def __init__(self, matrices, components):
        self.matrices = matrices
        self.components = tuple(components)
        self.edges = {}
        self.bounds = {}
        self.rows = {}
        self.tables = {}

    def _prepare(self, matrix, edges):
        key = (matrix.name, id(edges))
        if key not in self.edges:
            encoded = {}
            shifts = matrix.bit_offsets()
            prepared = []
            for edge in edges:
                points = []
                for point in edge.points:
                    value = encoded.get(point)
                    if value is None:
                        matrix.validate_coord(point)
                        value = sum(x << shift for x, shift in zip(point, shifts))
                        if len(encoded) >= 65536:
                            encoded.clear()
                        encoded[point] = value
                    points.append(value)
                unique = set(points)
                basis = None
                if unique and len(unique) & (len(unique) - 1) == 0:
                    anchor = points[0]
                    trial = rref_basis(value ^ anchor for value in unique)
                    if len(unique) == 1 << len(trial):
                        basis = trial
                prepared.append((edge.weight, tuple(points) if basis is None else (), basis))
            self.edges[key] = prepared
        return self.edges[key]

    def populate(self, components, layouts, cache):
        """Fill score_layouts' array/component cache without changing reduction order."""
        offsets_by_array = {}
        for component in components:
            for name, edges in component.edges_by_array.items():
                if not edges:
                    continue
                matrix, layout = self.matrices[name], layouts[name]
                if type(layout) not in (CanonicalLayout, LinearInnerLayout, AffineAccessLayout):
                    raise TypeError("prepared region scoring requires a known binary-linear layout")
                signature = layout.signature()
                key = (name, signature, component.name)
                if key in cache:
                    continue
                row_key = (name, signature)
                if row_key not in self.rows:
                    rows = layout_matrix_rows(matrix, layout)
                    self.rows[row_key] = rows
                    # Contributions from disjoint input bytes combine by XOR.
                    self.tables[row_key] = tuple(
                        (shift, (1 << width) - 1,
                         tuple(sum((((value << shift) & row).bit_count() & 1) << i
                                   for i, row in enumerate(rows)) for value in range(1 << width)))
                        for shift in range(0, len(rows), 8)
                        for width in (min(8, len(rows) - shift),))
                depth = component.dimension(matrix)
                tables = self.tables[row_key]
                offsets = offsets_by_array.setdefault(name, {})

                def offset(value):
                    result = offsets.get(value)
                    if result is None:
                        result = 0
                        for shift, mask, table in tables:
                            result ^= table[(value >> shift) & mask]
                        offsets[value] = result
                    return result

                raw = 0.0
                for weight, points, basis in self._prepare(matrix, edges):
                    if basis is not None:
                        count = 1 << rank(offset(value) >> depth for value in basis)
                    else:
                        count = len({offset(value) >> depth for value in points})
                    raw += weight * count
                bound_key = (name, component.name)
                if bound_key not in self.bounds:
                    self.bounds[bound_key] = component.packing_bound(matrix)
                cache[key] = raw, self.bounds[bound_key]
