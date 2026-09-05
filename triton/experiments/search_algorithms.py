"""Exact layout selection helpers for TritonBench Experiments 4--6."""

from __future__ import annotations

from dataclasses import replace
import json
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence

from relay import (
    CanonicalLayout,
    HardwareProfile,
    ScorePolicy,
    layout_matrix_rows,
    row_major_layout,
    score_layouts,
    simple_solve,
)
from relay.search import search_canonical

def load_tau_profile(platform: str, path: Path) -> HardwareProfile:
    document = json.loads(path.read_text(encoding="utf-8"))
    entry = document["platforms"][platform]
    tau = {str(name): float(value) for name, value in entry["active_tau"].items()}
    return HardwareProfile(
        profile_id=str(entry["profile_id"]),
        device={"platform": platform},
        byte_scales=tuple(map(int, document["byte_scales"])),
        fine_component=next(iter(tau)),
        tau=tau,
    )


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


def _is_baseline(layouts, matrices) -> bool:
    return all(
        layout_matrix_rows(matrices[name], layout)
        == layout_matrix_rows(matrices[name], row_major_layout(matrices[name]))
        for name, layout in layouts.items()
        if matrices[name].target
    )


def _member_key(member, matrices):
    return (
        float(member.score.hardware_area),
        0 if _is_baseline(member.layouts, matrices) else 1,
        int(member.score.codegen.runs),
        int(member.score.codegen.xors),
        member.word_signature(matrices),
    )


def _word_count(exponents: Sequence[int]) -> int:
    total = sum(exponents)
    result = 1
    placed = 0
    for count in exponents:
        result *= comb(placed + count, count)
        placed += count
    assert placed == total
    return result


def _row_major_tile(matrix, tile):
    word = tuple(
        mode
        for mode in reversed(range(matrix.rank))
        for _ in range(tile[mode])
    )
    return CanonicalLayout(
        "row_major_tile",
        matrix.name,
        tuple(tile),
        word,
        tuple(reversed(range(matrix.rank))),
    )


def natural_tile_exponents(matrix, events) -> tuple[int, ...]:
    """Infer the largest single-operation power-of-two access footprint."""

    return natural_tile_hypotheses(matrix, events)[0]


def natural_tile_hypotheses(matrix, events) -> tuple[tuple[int, ...], ...]:
    """Return the primary footprint and one distinct repeated footprint."""

    counts: dict[tuple[int, ...], int] = {}
    for event in events:
        coords = [access.coord for access in event.accesses if access.array == matrix.name]
        if not coords:
            continue
        extents = []
        for dimension in range(matrix.rank):
            span = max(coord[dimension] for coord in coords) - min(
                coord[dimension] for coord in coords
            ) + 1
            extent = min(matrix.shape[dimension], 1 << (span - 1).bit_length())
            extents.append(extent)
        tile = tuple(extent.bit_length() - 1 for extent in extents)
        counts[tile] = counts.get(tile, 0) + 1
    if not counts:
        return (tuple(0 for _ in matrix.shape),)
    ordered = sorted(counts, key=lambda tile: (sum(tile), tile), reverse=True)
    primary = ordered[0]
    repeated = [tile for tile in ordered[1:] if counts[tile] > 1]
    return (primary, *repeated[:1])


def _canonical_search(problem, profile, *, whole: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    matrices = {matrix.name: matrix for matrix in problem.matrices}
    components = problem_components(problem)
    baseline = {name: row_major_layout(matrix) for name, matrix in matrices.items()}
    chosen = dict(baseline)
    array_searches = []
    for matrix in problem.matrices:
        if not matrix.target:
            continue
        tiles = (
            (matrix.mode_bits,)
            if whole
            else natural_tile_hypotheses(matrix, problem.events)
        )
        candidates = []
        hypothesis_records = []
        for tile in tiles:
            count = _word_count(tile)
            tile_baseline = _row_major_tile(matrix, tile)
            if sum(tile) == 0:
                selected = tile_baseline
                stats = None
            else:
                active_weights = {
                    component.name: profile.tau.get(component.name, 0.0)
                    * component.region_bytes
                    for component in components
                    if profile.tau.get(component.name, 0.0) > 0
                }
                stats_sink = []
                seeds = search_canonical(
                    matrix,
                    components,
                    tuple(tile),
                    (tuple(reversed(range(matrix.rank))),),
                    ScorePolicy(
                        kind="weighted",
                        order=tuple(sorted(active_weights)),
                        weights=active_weights,
                        tie_order=("runs", "xors"),
                        paths_per_state=1,
                        frontier_limit=1,
                    ),
                    candidates_per_tile=1,
                    stats_sink=stats_sink,
                )
                if not seeds or not stats_sink[0].exact:
                    raise RuntimeError(f"{matrix.name}: natural-tile canonical DP failed")
                selected = seeds[0].layout
                stats = stats_sink[0]

            candidate_score = score_layouts(
                matrices, components, {**baseline, matrix.name: selected},
                hardware_profile=profile,
            )
            baseline_score = score_layouts(
                matrices, components, baseline, hardware_profile=profile,
            )
            if abs(candidate_score.hardware_area - baseline_score.hardware_area) <= 1e-12:
                selected = tile_baseline
                candidate_score = baseline_score
            rows = layout_matrix_rows(matrix, selected)
            baseline_rows = layout_matrix_rows(matrix, baseline[matrix.name])
            candidates.append((
                (
                    float(candidate_score.hardware_area),
                    0 if rows == baseline_rows else 1,
                    int(selected.runs), int(selected.xor_count), tuple(tile),
                ),
                selected,
            ))
            hypothesis_records.append({
                "tile_exponents": list(tile),
                "tile_shape": [1 << exponent for exponent in tile],
                "grammar_layout_count": count,
                "dp_states": None if stats is None else stats.states,
                "dp_transitions": None if stats is None else stats.transitions,
            })

        _, selected = min(candidates, key=lambda item: item[0])
        chosen[matrix.name] = selected
        selected_tile = tuple(selected.tile_exponents)
        for hypothesis in hypothesis_records:
            hypothesis["selected"] = tuple(hypothesis["tile_exponents"]) == selected_tile
        array_searches.append(
            {
                "matrix": matrix.name,
                "tile_hypotheses": hypothesis_records,
                "grammar_layout_count": sum(item["grammar_layout_count"] for item in hypothesis_records),
                "selected_tile_exponents": list(selected.tile_exponents),
                "selected_tile_shape": list(selected.tile_shape),
                "selected_word": [matrix.mode_names[mode] for mode in selected.word],
            }
        )
    score = score_layouts(
        matrices, components, chosen, hardware_profile=profile
    )
    return chosen, {
        "algorithm": (
            "exact_whole_tensor_count_grid_dp"
            if whole
            else "exact_canonical_natural_tile_count_grid_dp"
        ),
        "array_searches": array_searches,
        "score": _score_dict(score),
    }


def problem_components(problem):
    from relay.objectives import build_objectives

    return tuple(
        build_objectives(
            problem.objectives,
            {matrix.name: matrix for matrix in problem.matrices},
            {event.id: event for event in problem.events},
            problem.sequences,
        )
    )


def select_layouts(analysis, experiment: int, profile: HardwareProfile):
    """Return the scalar-J_area optimum and an auditable exact-search record."""

    from layout_runtime import RuntimeLayout

    problem = analysis.relay_problem(
        hardware_profile=profile,
        grammar="outer_canonical" if experiment == 6 else "canonical",
    )
    argument_names = {
        str(name): int(index)
        for index, name in analysis.bound_arguments.get("__names__", {}).items()
    }

    def direct_argument(allocation):
        if allocation.path:
            return None
        if isinstance(allocation.argument, int):
            return allocation.argument
        return argument_names.get(str(allocation.argument))

    eligible_names = {
        allocation.name
        for allocation in analysis.allocations
        if allocation.eligible
        and allocation.dense_status == "dense"
        and direct_argument(allocation) is not None
    }
    matrices = tuple(
        replace(matrix, target=matrix.name in eligible_names)
        for matrix in problem.matrices
    )
    if not any(matrix.target for matrix in matrices):
        raise RuntimeError("no read-only ordinary-dense direct pointer is realizable")
    problem = replace(problem, matrices=matrices)
    matrix_map = {matrix.name: matrix for matrix in matrices}

    if experiment in (4, 5):
        layouts, record = _canonical_search(
            problem, profile, whole=experiment == 4
        )
    else:
        result = simple_solve(problem)
        member = min(result.frontier, key=lambda item: _member_key(item, matrix_map))
        baseline_layouts = {
            matrix.name: row_major_layout(matrix) for matrix in matrices
        }
        baseline_score = score_layouts(
            matrix_map,
            result.components,
            baseline_layouts,
            hardware_profile=profile,
        )
        if abs(baseline_score.hardware_area - member.score.hardware_area) <= 1e-12:
            layouts = baseline_layouts
            selected_score = baseline_score
        else:
            layouts = dict(member.layouts)
            selected_score = member.score
        record = {
            "algorithm": "exact_bounded_goc_inner_enumeration_outer_dp",
            "elapsed_seconds": float(result.elapsed_seconds),
            "joint_raw_frontier_count": int(result.joint_raw_frontier_count),
            "retained_frontier_count": len(result.frontier),
            "array_searches": [
                {
                    "matrix": item.matrix,
                    "grammar_layout_count": item.grammar_layout_count,
                    "raw_frontier_count": item.raw_frontier_count,
                    "tile_hypotheses": item.tile_hypotheses,
                    "max_inner_bits": item.max_inner_bits,
                }
                for item in result.array_searches
            ],
            "score": _score_dict(selected_score),
        }

    layouts = {
        matrix.name: layouts.get(matrix.name, row_major_layout(matrix))
        for matrix in matrices
    }

    runtime = []
    layout_records = {}
    allocations = {allocation.name: allocation for allocation in analysis.allocations}
    for matrix in matrices:
        layout = layouts[matrix.name]
        rows = layout_matrix_rows(matrix, layout)
        baseline_rows = layout_matrix_rows(matrix, row_major_layout(matrix))
        layout_records[matrix.name] = {
            "target": matrix.target,
            "grammar": layout.grammar,
            "descriptor": (
                layout.evaluator_descriptor(matrix)
                if hasattr(layout, "evaluator_descriptor")
                else layout.word_string(matrix)
            ),
            "tile_exponents": list(layout.tile_exponents),
            "rows": list(rows),
            "is_baseline": rows == baseline_rows,
        }
        if not matrix.target or rows == baseline_rows:
            continue
        allocation = allocations[matrix.name]
        runtime.append(
            RuntimeLayout(
                matrix.name,
                int(direct_argument(allocation)),
                allocation.true_shape,
                allocation.strides,
                allocation.envelope_shape,
                rows,
            )
        )
    record["layouts"] = layout_records
    record["optimized_array_count"] = sum(matrix.target for matrix in matrices)
    record["transformed_array_count"] = len(runtime)
    return tuple(runtime), record
