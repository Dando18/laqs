"""Scalar analytical selection with an ordinary-layout fallback."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import product
from math import comb
from pathlib import Path
from typing import Any, Sequence

from relay import (CanonicalLayout, HardwareProfile, LinearInnerLayout, ScorePolicy,
                   layout_matrix_rows, row_major_layout, score_layouts)
from relay.search import search_canonical, SearchStats
from layout_contract import RuntimeLayout, preserves_vector_bits


def load_tau_profile(platform: str, path: Path, name: str | None = None):
    from tau_profiles import load_hardware_profile
    return load_hardware_profile(platform, path, name)


def _score_dict(score) -> dict[str, Any]:
    return {
        "hardware_area": float(score.hardware_area),
        "hardware_peak": float(score.hardware_peak),
        "weighted_region_count": float(score.weighted_region_count),
        "peak_normalized_excess": float(score.peak_normalized_excess),
        "weighted_normalized_excess": float(score.weighted_normalized_excess),
        "codegen_runs": int(score.codegen.runs),
        "codegen_xors": int(score.codegen.xors),
        "components": [
            {
                "name": component.name,
                "raw_region_count": float(component.raw_region_count),
                "packing_lower_bound": float(component.packing_lower_bound),
                "normalized_excess": float(component.normalized_excess),
                "excess_footprint": float(component.excess_footprint),
                "weight": float(component.weight),
            }
            for component in score.components
        ],
    }


def _word_count(exponents: Sequence[int]) -> int:
    total = sum(exponents)
    result = 1
    placed = 0
    for count in exponents:
        result *= comb(placed + count, count)
        placed += count
    assert placed == total
    return result



def natural_tile_hypotheses(matrix, events):
    """Enclose all owners of a dynamic parent operation in aligned dyadic tiles."""
    footprints = defaultdict(list)
    for event in events:
        key = (event.meta("workgroup", event.group), event.meta("phase", ""),
               event.meta("operation_instance", event.id), event.site)
        footprints[key].extend(a.coord for a in event.accesses if a.array == matrix.name)
    counts = defaultdict(int)
    for coords in footprints.values():
        if not coords:
            continue
        tile = tuple(min(matrix.mode_bits[d],
                         (max(x[d] for x in coords) ^ min(x[d] for x in coords)).bit_length())
                     for d in range(matrix.rank))
        counts[tile] += 1
    if not counts:
        return (tuple(0 for _ in matrix.shape),)
    ordered = sorted(counts, key=lambda tile: (sum(tile), tile), reverse=True)
    repeated = [tile for tile in ordered[1:] if counts[tile] > 1]
    return (ordered[0], *repeated[:1])


def natural_tile_exponents(matrix, events):
    return natural_tile_hypotheses(matrix, events)[0]


def problem_components(problem):
    from relay.objectives import build_objectives
    return tuple(build_objectives(problem.objectives,
                 {m.name: m for m in problem.matrices},
                 {e.id: e for e in problem.events}, problem.sequences))


def _weights(components, profile):
    return {c.name: profile.tau.get(c.name, 0) * c.region_bytes /
            (c.normalization_bytes or 1) for c in components if profile.tau.get(c.name, 0) > 0}


def _prefix(matrix, bits):
    return tuple(d for d in reversed(range(matrix.rank)) for _ in range(matrix.mode_bits[d]))[:bits]


def _canonical_candidate(matrix, components, tile, weights, bits):
    prefix = _prefix(matrix, bits)
    if any(prefix.count(d) > tile[d] for d in range(matrix.rank)):
        return None, None
    if sum(tile) == 0:
        return CanonicalLayout("empty_tile", matrix.name, tile, (),
                               tuple(reversed(range(matrix.rank)))), None
    stats = []
    seeds = search_canonical(
        matrix, tuple(c for c in components if c.name in weights), tile, (tuple(reversed(range(matrix.rank))),),
        ScorePolicy(kind="weighted", order=tuple(weights), weights=weights,
                    tie_order=("runs", "xors"), paths_per_state=1, frontier_limit=1),
        candidates_per_tile=1, stats_sink=stats, required_prefix=prefix,
    )
    if not seeds or not stats[0].exact:
        raise RuntimeError(f"{matrix.name}: scalar canonical DP failed")
    return seeds[0].layout, stats[0]


def _bounded_goc_candidates(matrix, components, weights, bits):
    """Exact scalar GL(p<=4) search; skip maps invisible at all active scales."""
    from relay.simple_solver import (_component_difference_patterns, _linear_inner_candidates,
        _outer_canonical_paths, _embed_inner_vector)
    active = [c for c in components if c.name in weights and c.edges_by_array.get(matrix.name)]
    if not active:
        return
    min_rank = min(c.dimension(matrix) for c in active)
    if min_rank >= 4:
        return
    patterns, dimensions = _component_difference_patterns(matrix, active, set(weights))
    ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
    for tile in product(*(range(min(width, 4) + 1) for width in matrix.mode_bits)):
        width = sum(tile)
        if not min_rank < width <= 4:
            continue
        global_bits = [_embed_inner_vector(matrix, tile, 1 << i) for i in range(width)]
        if any(row not in global_bits for row in ordinary[:bits]):
            continue
        required = tuple(1 << global_bits.index(row) for row in ordinary[:bits])
        stats = SearchStats("scalar_goc", tile)
        inner = _linear_inner_candidates(matrix, tile, patterns, dimensions, tuple(weights), stats,
                                         scalar_weights=weights, required_rows=required)
        outer = _outer_canonical_paths(matrix, tile, patterns, dimensions, tuple(weights), stats,
                                      scalar_weights=weights)
        for first in inner:
            for second in outer:
                yield LinearInnerLayout("scalar_goc", matrix.name, tile, first.a_rows,
                      tuple(reversed(range(matrix.rank))), first.columns, width, second.word)


def _sparse_candidates(matrix, anchors, components, weights, bits):
    """Declared extension: one swap or XOR above protected vector bits.

    This is an exact enumeration of neighbors of the two anchors, not exact
    minimization over the unbounded G_OC grammar.
    """
    active = [c for c in components if c.name in weights and c.edges_by_array.get(matrix.name)]
    if not active:
        return
    width = min(matrix.total_bits, max(c.dimension(matrix) for c in active) + 1)
    for anchor in anchors:
        rows = layout_matrix_rows(matrix, anchor)
        for i in range(bits, width):
            for j in range(i + 1, width):
                for operation in ("swap", "xor_up", "xor_down"):
                    changed = list(rows)
                    if operation == "swap":
                        changed[i], changed[j] = changed[j], changed[i]
                    elif operation == "xor_up":
                        changed[j] ^= changed[i]
                    else:
                        changed[i] ^= changed[j]
                    yield LinearInnerLayout(operation, matrix.name, matrix.mode_bits, tuple(changed),
                                            tuple(reversed(range(matrix.rank))))


def _search(problem, profile, *, experiment, vector_bits=None, minimum_gain=0.01, components=None):
    if profile.resource_maps:
        raise ValueError("scalar per-allocation search requires a separable hardware profile")
    components = tuple(components if components is not None else problem_components(problem))
    matrices = {m.name: m for m in problem.matrices}
    baseline = {name: row_major_layout(m) for name, m in matrices.items()}
    cache = {}
    native_family = profile.fine_component.rsplit(".", 1)[0]
    admission_components = tuple(c for c in components if profile.tau.get(c.name, 0) > 0
                                 or c.name.startswith(native_family + "."))
    def score(layouts, *, diagnostics=False):
        return score_layouts(matrices, components if diagnostics else admission_components, layouts, hardware_profile=profile,
                             array_component_cache=cache)
    baseline_score = score(baseline, diagnostics=True)
    weights = _weights(components, profile)
    chosen = dict(baseline)
    records = []
    native_family = profile.fine_component.rsplit(".", 1)[0]
    issue_names = {c.name for c in components if c.name.startswith(native_family + ".")}
    baseline_components = {c.name: c.raw_region_count for c in baseline_score.components}
    for matrix in problem.matrices:
        if not matrix.target:
            continue
        bits = (vector_bits or {}).get(matrix.name, 0)
        tiles = (matrix.mode_bits,) if experiment != 5 else natural_tile_hypotheses(matrix, problem.events)
        candidates = [baseline[matrix.name]]
        hypotheses = []
        for tile in tiles:
            candidate, stats = _canonical_candidate(matrix, components, tile, weights, bits)
            hypotheses.append({"tile_exponents": list(tile), "tile_shape": [1 << t for t in tile],
                "grammar_layout_count": _word_count(tile),
                "dp_states": None if stats is None else stats.states,
                "dp_transitions": None if stats is None else stats.transitions,
                "vector_prefix_admissible": candidate is not None})
            if candidate is not None:
                candidates.append(candidate)
        if experiment == 6:
            canonical = min(candidates, key=lambda x: score({**baseline, matrix.name: x}).hardware_area)
            candidates.extend(_bounded_goc_candidates(matrix, components, weights, bits))
            candidates.extend(_sparse_candidates(matrix, (baseline[matrix.name], canonical), components, weights, bits))
        ordinary_rows = layout_matrix_rows(matrix, baseline[matrix.name])
        best = baseline[matrix.name]
        best_score = baseline_score
        seen = set()
        rejected = defaultdict(int)
        for candidate in candidates:
            rows = layout_matrix_rows(matrix, candidate)
            if rows in seen:
                continue
            seen.add(rows)
            if not preserves_vector_bits(rows, ordinary_rows, bits):
                rejected["vector_access_structure"] += 1
                continue
            candidate_score = score({**baseline, matrix.name: candidate})
            if any(c.raw_region_count > baseline_components[c.name] + 1e-9
                   for c in candidate_score.components if c.name in issue_names):
                rejected["native_issue_inflation"] += 1
                continue
            area = candidate_score.hardware_area
            if area >= baseline_score.hardware_area - 1e-12:
                continue
            if baseline_score.hardware_area - area < minimum_gain * max(1.0, abs(baseline_score.hardware_area)):
                rejected["insufficient_score_gain"] += 1
                continue
            key = (area, candidate.runs, candidate.xor_count, rows)
            best_key = (best_score.hardware_area, best.runs, best.xor_count, layout_matrix_rows(matrix, best))
            if key < best_key:
                best, best_score = candidate, candidate_score
        chosen[matrix.name] = best
        retained_baseline = layout_matrix_rows(matrix, best) == ordinary_rows
        for hypothesis in hypotheses:
            hypothesis["selected"] = not retained_baseline and tuple(hypothesis["tile_exponents"]) == best.tile_exponents
        records.append({"matrix": matrix.name, "tile_hypotheses": hypotheses,
            "canonical_grammar_layout_count": sum(h["grammar_layout_count"] for h in hypotheses),
            "grammar_layout_count": None if experiment == 6 else sum(h["grammar_layout_count"] for h in hypotheses),
            "bounded_goc_inner_bit_limit": 4 if experiment == 6 else None,
            "candidate_mapping_count": len(seen), "rejections": dict(rejected),
            "protected_vector_bits": bits, "baseline_fallback": retained_baseline,
            "selected_tile_exponents": list(best.tile_exponents),
            "selected_tile_shape": list(best.tile_shape),
            "selected_word": [matrix.mode_names[m] for m in best.word] if isinstance(best, CanonicalLayout) else None})
    selected_score = score(chosen, diagnostics=True)
    if selected_score.hardware_area > baseline_score.hardware_area + 1e-12:
        raise AssertionError("deployed score exceeds ordinary score")
    return chosen, {"algorithm": {4: "scalar_whole_tensor_dp", 5: "scalar_natural_tile_dp",
                    6: "scalar_bounded_goc_plus_sparse_neighbors"}[experiment],
        "search_scope": "vector-preserving scalar optima with issue and baseline admission checks",
        "sparse_extension": "single swap/XOR of ordinary and canonical-optimum anchors" if experiment == 6 else None,
        "minimum_score_gain": minimum_gain, "array_searches": records,
        "baseline_score": _score_dict(baseline_score), "score": _score_dict(selected_score)}


def _canonical_search(problem, profile, *, whole):
    return _search(problem, profile, experiment=4 if whole else 5, minimum_gain=0)


def select_layouts(analysis, experiment, profile, *, minimum_gain=0.01):
    problem = analysis.relay_problem(hardware_profile=profile, grammar="canonical")
    names = {str(name): int(i) for i, name in analysis.bound_arguments.get("__names__", {}).items()}
    def direct(allocation):
        return None if allocation.path else (allocation.argument if isinstance(allocation.argument, int)
                                            else names.get(allocation.argument))
    eligible = {a.name for a in analysis.allocations
                if a.eligible and a.dense_status == "dense" and direct(a) is not None}
    matrices = tuple(replace(m, target=m.target and m.name in eligible) for m in problem.matrices)
    if not any(m.target for m in matrices):
        raise ValueError("no eligible direct dense operand")
    problem = replace(problem, matrices=matrices)
    vector_bits = {}
    for event in problem.events:
        width = int(event.meta("vector_elements", "1"))
        for access in event.accesses:
            vector_bits[access.array] = max(vector_bits.get(access.array, 0), (width - 1).bit_length())
    layouts, record = _search(problem, profile, experiment=experiment, vector_bits=vector_bits,
                             minimum_gain=minimum_gain, components=analysis.components or None)
    allocations = {a.name: a for a in analysis.allocations}
    runtime = []
    record["layouts"] = {}
    for matrix in matrices:
        layout = layouts[matrix.name]
        rows = layout_matrix_rows(matrix, layout)
        ordinary = rows == layout_matrix_rows(matrix, row_major_layout(matrix))
        record["layouts"][matrix.name] = {"target": matrix.target, "grammar": layout.grammar,
            "descriptor": layout.word_string(matrix) if isinstance(layout, CanonicalLayout) else layout.evaluator_descriptor(matrix),
            "tile_exponents": list(layout.tile_exponents), "rows": list(rows), "is_baseline": ordinary}
        if matrix.target and not ordinary:
            a = allocations[matrix.name]
            runtime.append(RuntimeLayout(a.name, int(direct(a)), a.true_shape, a.strides, a.envelope_shape, rows))
    record["ordinary_storage"] = {
        a.name: {"shape": list(a.true_shape), "strides": list(a.strides),
                 "envelope_shape": list(a.envelope_shape),
                 "base_alignment_mod_largest_scale": a.base_pointer % max(profile.byte_scales),
                 "interpretation": "baseline uses native strides; canonical rows model its region partition"}
        for a in analysis.allocations}
    record["fixed_operand_reasons"] = {
        a.name: "ordinary pitches/alignment do not have a proved equivalent region partition at all profile scales; retain native storage"
        for a in analysis.allocations if a.eligible and not next(m.target for m in matrices if m.name == a.name)}
    record["optimized_array_count"] = sum(m.target for m in matrices)
    record["transformed_array_count"] = len(runtime)
    return tuple(runtime), record
