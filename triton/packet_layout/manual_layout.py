"""Exact source realizations of frozen rectangular split maps."""
from dataclasses import replace


def split_tile(layout):
    """Recover tile extents from matrix rows, never from a candidate's name.

    Logical bits are i then j; physical bits must be j^v i^a j^(m-v)
    i^(n-a). Enumerating these small words also checks rank and bit order.
    """
    shape = layout.shape
    if (len(shape) != 2 or shape != layout.envelope_shape
            or layout.strides != (shape[1], 1)
            or any(n <= 0 or n & (n - 1) for n in shape)):
        raise ValueError('manual split storage requires dense power-of-two matrices')
    n, m = (s.bit_length() - 1 for s in shape)
    for a in range(n + 1):
        for v in range(m + 1):
            bits = (*range(n, n + v), *range(a), *range(n + v, n + m), *range(a, n))
            if layout.rows == tuple(1 << b for b in bits):
                return 1 << a, 1 << v
    raise ValueError(f'{layout.name}: saved matrix rows are not a rectangular split map')


def pack_split(source, tile):
    ti, tj = tile
    m, n = source.shape
    if not source.is_contiguous() or m % ti or n % tj:
        raise ValueError('split packing requires a dense matrix divisible by its tile')
    return source.reshape(m // ti, ti, n // tj, tj).permute(0, 2, 1, 3).contiguous().reshape(-1)


def unpack_split(packed, shape, tile):
    ti, tj = tile
    m, n = shape
    return packed.reshape(m // ti, n // tj, ti, tj).permute(0, 2, 1, 3).contiguous().reshape(m, n)


def explicit_spec(spec, layouts):
    """Specialize only pointers, preserving the original launch and arithmetic."""
    from packet_kernels import manual_sum_kernel, manual_gemm_kernel
    tiles = {layout.argument: split_tile(layout) for layout in layouts}
    if spec.operator == 'row_column':
        if set(tiles) - {0}:
            raise ValueError('row/column source template only supports matrix A')
        ti, tj = tiles.get(0, (1, 1))
        return spec, {'STORAGE': 2, 'TILE_I': ti, 'TILE_J': tj}
    if spec.operator == 'sum':
        if set(tiles) - {0} or spec.kwargs['dim'] != 1:
            raise ValueError('manual sum supports the dim=1 matrix input only')
        ti, tj = tiles.get(0, (1, 1))
        return replace(spec, kernel=manual_sum_kernel), {'TILE_I': ti, 'TILE_J': tj}
    if spec.operator == 'gemm':
        if set(tiles) - {0, 1}:
            raise ValueError('manual GEMM supports the two read-only matrices')
        ai, aj = tiles.get(0, (1, 1))
        bi, bj = tiles.get(1, (1, 1))
        return replace(spec, kernel=manual_gemm_kernel), {
            'A_TILE_I': ai, 'A_TILE_J': aj, 'B_TILE_I': bi, 'B_TILE_J': bj}
    raise ValueError(f'no explicit source template for {spec.operator}')


def execution_comparison(automatic, explicit):
    # The generic encoding diagnostic contains repeated IR snippets and local
    # alias names. Its textual inequality is not an ownership comparison.
    sites = explicit.get('inspected_packet_sites')
    comparable = bool(automatic.get('packet_sites') and sites)
    explicit = {**explicit, 'packet_sites': sites}
    fields = ('memory_primitives', 'matrix_primitives', 'communication')
    if comparable:
        fields += ('packet_sites',)
    changed = [field for field in fields if automatic.get(field) != explicit.get(field)]
    return {'changed_fields': changed,
            'ownership_comparable': comparable,
            'raw_ir_encoding_text_changed': automatic.get('execution_encodings') != explicit.get('execution_encodings'),
            'same_lowered_evidence': comparable and not changed,
            'interpretation': ('ownership inspection unavailable; inspect saved IR before '
                               'claiming equivalent execution' if not comparable else
                               'recorded ownership or primitives differ; the captured ordinary graph '
                               'does not score this explicit realization' if changed else
                               'no difference in recorded ownership/primitive evidence; '
                               'this is not a proof of identical scheduling')}
