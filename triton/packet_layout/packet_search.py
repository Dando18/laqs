"""LAQS selection over explicit split-coordinate and chunk address templates."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

from relay import CanonicalLayout, layout_matrix_rows, row_major_layout, score_layouts
from relay.gf2 import invert_matrix_rows, rref_basis
from layout_contract import RuntimeLayout, preserves_vector_bits
from search_algorithms import _score_dict, natural_tile_hypotheses


@dataclass(frozen=True)
class AddressField:
    source_start: int
    width: int
    target_start: int


def address_fields(matrix, layout):
    """Deposit fields of the ordinary element offset, matching the compiler plan."""
    rows = layout_matrix_rows(matrix, layout)
    if any(row.bit_count() != 1 for row in rows):
        raise ValueError("structured templates require a bit permutation")
    ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
    positions = {row: bit for bit, row in enumerate(ordinary)}
    rows = tuple(1 << positions[row] for row in rows)
    fields = []
    position = 0
    while position < len(rows):
        source = rows[position].bit_length() - 1
        width = 1
        while position + width < len(rows) and rows[position + width] == 1 << (source + width):
            width += 1
        fields.append(AddressField(source, width, position))
        position += width
    return tuple(fields)


def partial_flag(rows, depths):
    """Canonical bases for A^-1 U_d, including all scored/protected depths."""
    inverse = invert_matrix_rows(rows, len(rows))
    columns = tuple(sum(((row >> j) & 1) << i for i, row in enumerate(inverse))
                    for j in range(len(rows)))
    return tuple((depth, rref_basis(columns[:depth])) for depth in sorted(set(depths)))


def split_templates(matrix, packet_bits):
    if matrix.rank != 2 or packet_bits > matrix.mode_bits[1]:
        return
    n, m = matrix.mode_bits
    seen = set()
    for a, v in product(range(n + 1), range(packet_bits, m + 1)):
        word = (1,) * v + (0,) * a + (1,) * (m - v) + (0,) * (n - a)
        if word not in seen:
            seen.add(word)
            yield CanonicalLayout(f"split-a{a}-v{v}", matrix.name, matrix.mode_bits, word, (1, 0))


def chunk_templates(matrix, packet_bits, tiles, *, max_fields=6):
    """Enumerate an explicit finite chunk grammar, preserving per-mode order.

    Boundaries are native footprint extents, the packet boundary, and allocation
    extents. Additional alternation is not admitted just because it is canonical.
    """
    if matrix.rank != 2 or packet_bits > matrix.mode_bits[1]:
        return
    cuts = [{0, extent} for extent in matrix.mode_bits]
    cuts[1].add(packet_bits)
    for tile in tiles:
        for mode, extent in enumerate(tile):
            cuts[mode].add(extent)
    chunks = [tuple(b - a for a, b in zip(sorted(c)[:-1], sorted(c)[1:]) if b > a)
              for c in cuts]

    def visit(placed, word):
        if len(word) == matrix.total_bits:
            layout = CanonicalLayout("native-chunks", matrix.name, matrix.mode_bits, word, (1, 0))
            if len(address_fields(matrix, layout)) <= max_fields:
                yield layout
            return
        for mode in range(2):
            if placed[mode] == len(chunks[mode]):
                continue
            if len(word) < packet_bits and mode != 1:
                continue
            next_placed = list(placed)
            width = chunks[mode][placed[mode]]
            next_placed[mode] += 1
            yield from visit(tuple(next_placed), word + (mode,) * width)

    yield from visit((0, 0), ())


def supported_allocation(matrix, allocation, packet_bits):
    if not matrix.target or not allocation.eligible or allocation.path:
        return "not an eligible direct read-only allocation"
    if matrix.rank != 2:
        return "initial structured family requires two logical dimensions"
    if allocation.true_shape != allocation.envelope_shape:
        return "initial structured family requires power-of-two extents"
    expected = (allocation.true_shape[1], 1)
    if allocation.dense_status != "dense" or tuple(allocation.strides) != expected:
        return "initial structured family requires dense ordinary row-major inputs"
    if matrix.total_bits >= 31:
        return "structured address envelope does not fit signed 32-bit elements"
    if packet_bits > matrix.mode_bits[1]:
        return "native packet crosses the minor coordinate"
    return None


def select_candidates(analysis, profile):
    """Return at most three physical candidates, with analytical top-1 separate.

    Exact minimization is over each enumerated supported template family for the
    supplied graph. Compiler primitive validation remains a separate obligation.
    Partial flags deduplicate only equivalent supported refinements; an arbitrary
    unrepresentable flag is never chosen first.
    """
    if profile.resource_maps:
        raise ValueError("packet-family separability requires no resource maps")
    problem = analysis.relay_problem(hardware_profile=profile, grammar="canonical")
    matrices = {m.name: m for m in problem.matrices}
    allocations = {a.name: a for a in analysis.allocations}
    baseline = {name: row_major_layout(m) for name, m in matrices.items()}
    active = tuple(c for c in analysis.components if profile.tau.get(c.name, 0) > 0)
    if not active:
        raise ValueError("profile has no active graph components")
    cache = {}

    def score(layouts, full=False):
        return score_layouts(matrices, analysis.components if full else active, layouts,
                            hardware_profile=profile, array_component_cache=cache)

    baseline_score = score(baseline)
    packets = {}
    for event in analysis.events:
        width = int(event.meta("vector_elements", "1"))
        for access in event.accesses:
            packets[access.array] = max(packets.get(access.array, 0), (width - 1).bit_length())
    names = {str(name): int(index) for index, name in analysis.bound_arguments["__names__"].items()}
    records = []
    family_choices = {}
    for family in ("split", "chunks"):
        chosen = dict(baseline)
        for matrix in problem.matrices:
            allocation = allocations[matrix.name]
            packet_bits = packets.get(matrix.name, 0)
            reason = supported_allocation(matrix, allocation, packet_bits)
            if reason:
                records.append({"family": family, "matrix": matrix.name, "fixed_reason": reason})
                continue
            tiles = natural_tile_hypotheses(matrix, analysis.events)
            layouts = list(split_templates(matrix, packet_bits))
            if family == "chunks":
                layouts.extend(chunk_templates(matrix, packet_bits, tiles))
            ordinary = layout_matrix_rows(matrix, baseline[matrix.name])
            depths = {min(matrix.total_bits, (scale // matrix.element_bytes).bit_length() - 1)
                      for scale in profile.byte_scales}
            depths.add(packet_bits)
            representatives = {}
            for layout in layouts:
                rows = layout_matrix_rows(matrix, layout)
                if not preserves_vector_bits(rows, ordinary, packet_bits):
                    continue
                flag = partial_flag(rows, depths)
                key = (rows != ordinary, len(address_fields(matrix, layout)), rows)
                previous = representatives.get(flag)
                if previous is None or key < previous[0]:
                    representatives[flag] = key, layout
            best, best_key = baseline[matrix.name], (baseline_score.hardware_area, False, 0, ordinary)
            for _, candidate in representatives.values():
                result = score({**baseline, matrix.name: candidate})
                rows = layout_matrix_rows(matrix, candidate)
                key = (result.hardware_area, rows != ordinary, len(address_fields(matrix, candidate)), rows)
                if key < best_key:
                    best, best_key = candidate, key
            chosen[matrix.name] = best
            records.append({"family": family, "matrix": matrix.name, "packet_bits": packet_bits,
                            "enumerated_templates": len(layouts), "partial_flag_classes": len(representatives),
                            "flag_depths": sorted(depths), "native_tiles": tiles,
                            "selected_word": best.word, "address_fields": [vars(f) for f in address_fields(matrix, best)]})
        family_choices[family] = chosen

    candidates = []
    signatures = {}
    for label, layouts in [("ordinary", baseline), *family_choices.items()]:
        signature = tuple((name, layout_matrix_rows(m, layouts[name])) for name, m in matrices.items())
        if signature in signatures:
            candidates[signatures[signature]]["families"].append(label)
            continue
        runtime = []
        for name, matrix in matrices.items():
            rows = layout_matrix_rows(matrix, layouts[name])
            if rows == layout_matrix_rows(matrix, baseline[name]):
                continue
            a = allocations[name]
            argument = a.argument if isinstance(a.argument, int) else names[a.argument]
            runtime.append(RuntimeLayout(name, argument, a.true_shape, a.strides, a.envelope_shape, rows).to_dict())
        result = score(layouts, full=True)
        if result.hardware_area > baseline_score.hardware_area + 1e-10:
            raise AssertionError("structured family selection regressed LAQS objective")
        signatures[signature] = len(candidates)
        candidates.append({"id": label, "families": [label], "runtime_layouts": runtime,
                           "score": _score_dict(result)})
    best = min(candidates, key=lambda c: (c["score"]["hardware_area"], bool(c["runtime_layouts"]), c["id"]))
    return {"schema": "laqs.packet.search.v1", "candidates": candidates,
            "analytical_top1": best["id"], "array_searches": records,
            "search_scope": "exact enumeration of split and native-boundary chunk templates with supported partial-flag refinements",
            "maximum_realized_candidates": 3}
