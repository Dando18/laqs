"""Shared all-candidate benchmark for one prepacked persistent operand."""

from __future__ import annotations

from collections.abc import Callable
import statistics

import torch

from stage1_common import (
    benchmark_layouts,
    canonical_layout_metadata,
    compiled_codegen_statistics,
    layout_rows,
    pack_tensor,
    require_aligned,
    solve_layouts,
    stable_id,
    timing_summary,
)


def rank_persistent_operand(
    matrix,
    logical_operand: torch.Tensor,
    default_source: torch.Tensor,
    events,
    *,
    args,
    problem_name: str,
    make_output: Callable[[], object],
    make_launch: Callable[[torch.Tensor, object, tuple[int, ...]], Callable],
    validate: Callable[[str, object], None],
    inner_tile_shapes: tuple[tuple[int, ...], ...],
    temporal_edges=(),
    temporal_model: dict[str, object] | None = None,
    temporal_mode: str = "issue",
    benchmark: Callable | None = None,
) -> dict[str, object]:
    """Solve, compile, validate, and time every retained operand layout."""

    from relay import (
        ObjectiveComponent,
        low_address_flag,
        row_major_layout,
        summarize_rank_quality,
        weighted_component_region_count,
    )

    objective, problem, result = solve_layouts(
        (matrix,),
        events,
        args,
        problem_name,
        inner_tile_shapes={matrix.name: inner_tile_shapes},
        temporal_edges={matrix.name: tuple(temporal_edges)},
        temporal_mode=temporal_mode,
    )
    objective_names = (
        (objective,) if isinstance(objective, str) else tuple(objective)
    )
    components_by_name = {
        component.name: component for component in result.components
    }
    ranking_components = tuple(
        components_by_name[name] for name in objective_names
    )
    issue_component = ObjectiveComponent(
        "diagnostic.issue",
        args.transaction_bytes,
        {matrix.name: tuple(event.hyperedge for event in events)},
    )
    temporal_component = ObjectiveComponent(
        "diagnostic.temporal",
        args.transaction_bytes,
        {matrix.name: tuple(temporal_edges)},
    )

    def component_scores(layout):
        issue_score = weighted_component_region_count(
            matrix, layout, issue_component
        )
        temporal_score = weighted_component_region_count(
            matrix, layout, temporal_component
        )
        return {
            "issue": float(issue_score),
            "temporal": float(temporal_score),
        }

    def total_candidate_score(candidate):
        return float(
            sum(candidate.scores.get(name, 0.0) for name in objective_names)
        )

    array_result = result.arrays[matrix.name]
    retained = array_result.candidates
    resolved_inner_tile_shapes = tuple(
        tuple(1 << exponent for exponent in tile)
        for tile in array_result.tile_hypotheses
    )
    default_layout = row_major_layout(matrix)
    default_rows = layout_rows(default_layout, matrix)
    default_score = sum(
        weighted_component_region_count(matrix, default_layout, component)
        for component in ranking_components
    )
    sources_by_mapping = {
        stable_id("mapping", list(default_rows)): default_source,
    }
    launches = {}
    outputs = {}
    compiled = {}
    records = []
    score_levels = sorted(
        {total_candidate_score(candidate) for candidate in retained}
    )
    for solver_rank, candidate in enumerate(retained, start=1):
        layout = candidate.layout
        rows = layout_rows(layout, matrix)
        mapping_id = stable_id("mapping", list(rows))
        flag_id = stable_id("flag", low_address_flag(matrix, layout))
        candidate_id = stable_id(
            "candidate",
            {
                "layout": layout.name,
                "grammar": layout.grammar,
                "a_rows": rows,
            },
        )
        source = sources_by_mapping.get(mapping_id)
        if source is None:
            source = pack_tensor(logical_operand, rows).to("cuda")
            sources_by_mapping[mapping_id] = source
        require_aligned(candidate_id, source, args.transaction_bytes)
        output = make_output()
        launch = make_launch(source, output, rows)
        outputs[candidate_id] = output
        launches[candidate_id] = launch
        compiled[candidate_id] = launch()
        score = total_candidate_score(candidate)
        quotient_components = component_scores(layout)
        expected_score = quotient_components["issue"]
        if temporal_mode != "issue":
            expected_score += quotient_components["temporal"]
        if abs(score - expected_score) > 1e-9:
            raise ValueError(
                f"{candidate_id}: objective score {score} does not match "
                f"issue/temporal decomposition {expected_score}"
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "solver_rank": solver_rank,
                "quotient_rank": score_levels.index(score) + 1,
                "layout": layout.name,
                "grammar": layout.grammar,
                **canonical_layout_metadata(layout, matrix),
                "a_rows": list(rows),
                "mapping_id": mapping_id,
                "flag_id": flag_id,
                "quotient_score": score,
                "quotient_components": quotient_components,
                "packing_bound": float(
                    sum(
                        candidate.packing_bounds.get(name, 0.0)
                        for name in objective_names
                    )
                ),
                "runs": int(candidate.scores["runs"]),
                "xor_count": layout.xor_count,
                "exact": candidate.exact,
                "note": candidate.note,
            }
        )

    torch.cuda.synchronize()
    for record in records:
        candidate_id = str(record["candidate_id"])
        validate(candidate_id, outputs[candidate_id])
    timing_function = benchmark
    if timing_function is None:
        timing_function = lambda candidate_launches: benchmark_layouts(
            candidate_launches,
            samples=args.samples,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    timings = timing_function(launches)
    for record in records:
        candidate_id = str(record["candidate_id"])
        timing = timings[candidate_id]
        record["runtime_ms"] = float(timing["median_ms"])
        record["timing"] = timing
        record["compiled_codegen"] = compiled_codegen_statistics(
            compiled[candidate_id]
        )

    rank_quality = summarize_rank_quality(records)
    selected = records[0]
    default = next(record for record in records if record["layout"] == "row_major")
    temporal_record = {
        "mode": temporal_mode,
        "edge_family": "per_hardware_location_nonoverlapping",
        "representative_edge_count": len(temporal_edges),
        "active_in_objective": temporal_mode != "issue",
        **(temporal_model or {}),
    }
    return {
        "objective": (
            objective
            if isinstance(objective, str)
            else {
                "aggregation": "equal_weight_raw_sum",
                "components": list(objective_names),
                "tie_breakers": ["runs", "xors"],
            }
        ),
        "objective_components": list(objective_names),
        "temporal_model": temporal_record,
        "search_scope": {
            "grammar": "canonical_inner_tile",
            "tile_policy": "explicit_hypothesis_sweep_v1",
            "inner_tile_shapes": [
                list(shape) for shape in resolved_inner_tile_shapes
            ],
            "outer_layout": "row_major_tiles",
            "fixed_outer_order": list(reversed(matrix.mode_names)),
            "temporal_mode": temporal_mode,
        },
        "packing_lower_bound": sum(
            component.packing_bound(matrix)
            for component in ranking_components
        ),
        "default": default,
        "selected": selected,
        "candidates": records,
        "rank_quality": rank_quality,
        "default_quotient": float(default_score),
        "selected_quotient": float(selected["quotient_score"]),
        "predicted_transaction_reduction": 1.0
        - float(selected["quotient_score"]) / float(default_score),
        "default_runtime_ms": float(default["runtime_ms"]),
        "selected_runtime_ms": float(selected["runtime_ms"]),
        "measured_speedup": float(default["runtime_ms"])
        / float(selected["runtime_ms"]),
        "laqs_made_no_change": selected["mapping_id"] == default["mapping_id"],
        "solver": {
            "name": problem.name,
            "elapsed_seconds": result.elapsed_seconds,
            "candidate_count": array_result.all_candidate_count,
            "retained_candidates": len(retained),
        },
        "correct": True,
    }


def aggregate_persistent_rankings(
    rankings: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate candidate runtimes across fresh process launches."""

    from relay import summarize_rank_quality

    if not rankings:
        raise ValueError("persistent-operand aggregation requires results")
    candidate_ids = [
        str(candidate["candidate_id"]) for candidate in rankings[0]["candidates"]
    ]
    for ranking in rankings[1:]:
        observed = [
            str(candidate["candidate_id"]) for candidate in ranking["candidates"]
        ]
        if observed != candidate_ids:
            raise ValueError("retained candidates changed between processes")

    candidates = []
    for index, candidate_id in enumerate(candidate_ids):
        process_candidates = [ranking["candidates"][index] for ranking in rankings]
        process_medians = [
            float(candidate["timing"]["median_ms"])
            for candidate in process_candidates
        ]
        samples = [
            float(sample)
            for candidate in process_candidates
            for sample in candidate["timing"]["samples_ms"]
        ]
        codegen = [candidate["compiled_codegen"] for candidate in process_candidates]
        candidate = {
            key: value
            for key, value in process_candidates[0].items()
            if key not in {"compiled_codegen", "runtime_ms", "timing"}
        }
        runtime = statistics.median(process_medians)
        candidate["runtime_ms"] = runtime
        candidate["timing"] = {
            **timing_summary(samples),
            "aggregation_runtime_ms": runtime,
            "aggregation_method": "median_of_process_medians",
            "process_medians_ms": process_medians,
            "process_launch_count": len(process_medians),
        }
        candidate["compiled_codegen"] = codegen[0]
        candidate["compiled_codegen_consistent"] = all(
            item == codegen[0] for item in codegen[1:]
        )
        if not candidate["compiled_codegen_consistent"]:
            candidate["compiled_codegen_by_process"] = codegen
        candidates.append(candidate)

    selected = candidates[0]
    default = next(candidate for candidate in candidates if candidate["layout"] == "row_major")
    quality = summarize_rank_quality(candidates)
    aggregate = {
        key: value
        for key, value in rankings[0].items()
        if key
        not in {
            "candidates",
            "correct",
            "default",
            "default_runtime_ms",
            "laqs_made_no_change",
            "measured_speedup",
            "rank_quality",
            "selected",
            "selected_runtime_ms",
        }
    }
    aggregate.update(
        {
            "process_launch_count": len(rankings),
            "process_speedups": [
                float(ranking["default_runtime_ms"])
                / float(ranking["selected_runtime_ms"])
                for ranking in rankings
            ],
            "default": default,
            "selected": selected,
            "candidates": candidates,
            "rank_quality": quality,
            "default_runtime_ms": float(default["runtime_ms"]),
            "selected_runtime_ms": float(selected["runtime_ms"]),
            "measured_speedup": float(default["runtime_ms"])
            / float(selected["runtime_ms"]),
            "laqs_made_no_change": selected["mapping_id"]
            == default["mapping_id"],
            "correct": all(bool(ranking["correct"]) for ranking in rankings),
        }
    )
    return aggregate
