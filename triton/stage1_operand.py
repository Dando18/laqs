"""Shared all-candidate benchmark for one prepacked persistent operand."""

from __future__ import annotations

from collections.abc import Callable
import statistics

import torch

from stage1_common import (
    benchmark_layouts,
    canonical_layout_metadata,
    compiled_codegen_statistics,
    execution_layout_from_compiled,
    execution_layout_record,
    is_load_opcode,
    is_store_opcode,
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
    execution_layout=None,
    execution_layout_spec: tuple[tuple[int, ...], tuple[str, ...]] | None = None,
    automatic_analysis=None,
    automatic_target_name: str | None = None,
) -> dict[str, object]:
    """Solve, compile, validate, and time every retained operand layout."""

    from relay import (
        ObjectiveComponent,
        low_address_flag,
        row_major_layout,
        summarize_rank_quality,
        weighted_component_region_count,
    )

    service_name = getattr(args, "service_model", "none")
    fiber_objectives = ()
    objective_taus = None
    service_record: dict[str, object] = {
        "name": service_name,
        "active": False,
    }
    if service_name == "mi300a_v1":
        from stage1_service import hardware_service_model

        if execution_layout is None:
            raise ValueError("the MI300A service model needs an execution layout")
        if temporal_mode != "split":
            raise ValueError("the MI300A service model requires temporal split mode")
        if args.transaction_bytes != 128:
            raise ValueError(
                "the MI300A service model requires 128-byte cache windows"
            )
        lane_bits = tuple(getattr(args, "lane_cohort_bits", (2, 3)))
        service = hardware_service_model(
            execution_layout,
            events,
            name=f"{problem_name}.service",
            lane_subsets=(lane_bits,),
        )
        instruction_bytes = int(getattr(args, "instruction_bytes", 64))
        lane_cohort_bytes = int(getattr(args, "lane_cohort_bytes", 64))
        instruction = service.instruction.at_scale(instruction_bytes)
        lane_cohort = service.lane_cohorts[0].at_scale(lane_cohort_bytes)
        fiber_objectives = (instruction, lane_cohort)
        objective_taus = {
            f"{problem_name}.issue.128B": float(args.issue_tau),
            f"{problem_name}.temporal.128B": float(args.temporal_tau),
            instruction.name: float(args.instruction_tau),
            lane_cohort.name: float(args.lane_cohort_tau),
        }
        service_record = {
            "name": service_name,
            "active": True,
            "instruction_region_bytes": instruction_bytes,
            "lane_cohort_region_bytes": lane_cohort_bytes,
            "lane_cohort_bits": list(lane_bits),
            "cache_window_region_bytes": args.transaction_bytes,
            "tau_by_component": objective_taus,
            "tuning": {
                "issue_tau": float(args.issue_tau),
                "temporal_tau": float(args.temporal_tau),
                "instruction_tau": float(args.instruction_tau),
                "lane_cohort_tau": float(args.lane_cohort_tau),
            },
            "fibers": service.record(),
            "selection_tie_breaker": "prefer_default_layout",
        }
    elif service_name != "none":
        raise ValueError(f"unknown service model {service_name!r}")

    objective, problem, result = solve_layouts(
        (matrix,),
        events,
        args,
        problem_name,
        inner_tile_shapes={matrix.name: inner_tile_shapes},
        temporal_edges={matrix.name: tuple(temporal_edges)},
        temporal_mode=temporal_mode,
        execution_fiber_objectives=fiber_objectives,
        objective_taus=objective_taus,
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
    policy = problem.config.policy
    if policy.kind == "weighted":
        objective_weights = {
            name: float(policy.weights[name]) for name in objective_names
        }
    elif len(objective_names) == 1:
        objective_weights = {objective_names[0]: 1.0}
    else:
        raise ValueError("Stage 1 needs a scalar or weighted objective")
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
    counter_panel = None
    counter_panels = None
    panel_mode = getattr(args, "counter_panel", None)
    if panel_mode is not None:
        from stage1_counter_candidates import (
            canonical_counter_panel,
            random_linear_counter_panel,
        )

        if temporal_mode != "issue":
            raise ValueError("counter panels require issue-only scoring")
        if panel_mode == "fixed_tile_levels":
            tile_shape = getattr(args, "panel_tile_shape", None)
            if tile_shape is None:
                raise ValueError(
                    "fixed-tile counter panels require --panel-tile-shape"
                )
            panel_tile_shapes = (tuple(tile_shape),)
            group_by_tile = False
        elif panel_mode == "random_layouts":
            counter_panel = random_linear_counter_panel(
                matrix,
                issue_component,
                inner_tile_shapes,
                samples=args.panel_samples,
                seed=args.panel_seed,
            )
        elif panel_mode.startswith("experiment"):
            from pathlib import Path
            import sys

            experiment_source = str(Path(__file__).with_name("experiments"))
            if experiment_source not in sys.path:
                sys.path.insert(0, experiment_source)
            from layout_panels import stratified_experiment_panel

            if automatic_analysis is None or automatic_target_name is None:
                raise ValueError(
                    "final experiments require the automatic Triton graph"
                )
            experiments = (
                (1, 2, 3)
                if panel_mode == "experiments123"
                else (int(panel_mode[len("experiment")]),)
            )
            counter_panels = {}
            for experiment in experiments:
                experiment_mode = {
                    1: "experiment1_gc_whole",
                    2: "experiment2_gc_tiles",
                    3: "experiment3_goc",
                }[experiment]
                panel = stratified_experiment_panel(
                    matrix,
                    automatic_analysis,
                    automatic_target_name,
                    inner_tile_shapes,
                    experiment=experiment,
                    samples=args.panel_samples,
                    seed=args.panel_seed,
                    platform=args.counter_platform,
                    stratification=args.panel_stratification,
                    pool_multiplier=args.panel_pool_multiplier,
                )
                panel["mode"] = experiment_mode
                panel["objective"] = "J_area"
                counter_panels[str(experiment)] = panel
            if len(counter_panels) == 1:
                counter_panel = next(iter(counter_panels.values()))
        elif panel_mode == "recorded_experiment":
            from pathlib import Path
            import json
            import sys

            experiment_source = str(Path(__file__).with_name("experiments"))
            if experiment_source not in sys.path:
                sys.path.insert(0, experiment_source)
            from layout_panels import rescore_recorded_panel

            if automatic_analysis is None or automatic_target_name is None:
                raise ValueError(
                    "recorded experiment rescoring requires the automatic "
                    "Triton graph"
                )
            source_path = getattr(args, "panel_source", None)
            if source_path is None:
                raise ValueError("recorded experiment rescoring needs a source")
            source = json.loads(Path(source_path).read_text(encoding="utf-8"))
            source_panel = source.get("panel", source)
            counter_panel = rescore_recorded_panel(
                automatic_analysis,
                automatic_target_name,
                source_panel,
                platform=args.counter_platform,
                source_path=Path(source_path),
            )
        else:
            panel_tile_shapes = inner_tile_shapes
            group_by_tile = True
        if counter_panel is None and counter_panels is None:
            counter_panel = canonical_counter_panel(
                matrix,
                issue_component,
                panel_tile_shapes,
                group_by_tile=group_by_tile,
            )
        if counter_panel is not None:
            if panel_mode == "recorded_experiment":
                counter_panel["rescore"]["worker_mode"] = panel_mode
            else:
                counter_panel["mode"] = panel_mode
            counter_panel["objective"] = (
                "J_area"
                if panel_mode.startswith("experiment")
                or panel_mode == "recorded_experiment"
                else "issue_only"
            )

    def diagnostic_component_scores(layout):
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

    def objective_component_scores(layout):
        return {
            component.name: float(
                weighted_component_region_count(matrix, layout, component)
            )
            for component in ranking_components
        }

    def aggregate_score(scores):
        return float(
            sum(
                objective_weights[name] * float(scores.get(name, 0.0))
                for name in objective_names
            )
        )

    def total_candidate_score(candidate):
        return aggregate_score(candidate.scores)

    array_result = result.arrays[matrix.name]
    retained = array_result.candidates
    resolved_inner_tile_shapes = tuple(
        tuple(1 << exponent for exponent in tile)
        for tile in array_result.tile_hypotheses
    )
    default_layout = row_major_layout(matrix)
    default_rows = layout_rows(default_layout, matrix)
    default_component_scores = objective_component_scores(default_layout)
    default_score = aggregate_score(default_component_scores)
    default_packing_bound = sum(
        objective_weights[component.name] * component.packing_bound(matrix)
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
        quotient_components = diagnostic_component_scores(layout)
        detailed_scores = objective_component_scores(layout)
        expected_score = aggregate_score(detailed_scores)
        if abs(score - expected_score) > 1e-9:
            raise ValueError(
                f"{candidate_id}: objective score {score} does not match "
                f"the weighted component decomposition {expected_score}"
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
                "objective_component_scores": detailed_scores,
                "packing_bound": float(
                    sum(
                        objective_weights[name]
                        * candidate.packing_bounds.get(name, 0.0)
                        for name in objective_names
                    )
                ),
                "runs": int(candidate.scores["runs"]),
                "xor_count": layout.xor_count,
                "exact": candidate.exact,
                "note": candidate.note,
            }
        )

    if service_name != "none":
        records.sort(
            key=lambda record: (
                float(record["quotient_score"]),
                record["layout"] != "row_major",
                int(record["solver_rank"]),
            )
        )
        for selection_rank, record in enumerate(records, start=1):
            record["selection_rank"] = selection_rank

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
    profile_target = None
    profile_layout = getattr(args, "profile_layout", None)
    explicit_profile_rows = getattr(args, "profile_rows", None)
    if profile_layout is not None or explicit_profile_rows is not None:
        if explicit_profile_rows is None:
            target = default if profile_layout == "default" else selected
            target_launch = launches[str(target["candidate_id"])]
            target_compiled = compiled[str(target["candidate_id"])]
            target_role = profile_layout
        else:
            from relay import LinearInnerLayout
            from relay.gf2 import invert_matrix_rows

            rows = tuple(int(row) for row in explicit_profile_rows)
            if len(rows) != matrix.total_bits:
                raise ValueError(
                    "--profile-rows must provide one row per logical bit"
                )
            invert_matrix_rows(rows, matrix.total_bits)
            mapping_id = stable_id("mapping", list(rows))
            candidate_id = (
                getattr(args, "profile_candidate_id", None)
                or stable_id("counter_candidate", list(rows))
            )
            source = sources_by_mapping.get(mapping_id)
            if source is None:
                source = pack_tensor(logical_operand, rows).to("cuda")
                sources_by_mapping[mapping_id] = source
            require_aligned(candidate_id, source, args.transaction_bytes)
            output = make_output()
            target_launch = make_launch(source, output, rows)
            target_compiled = target_launch()
            torch.cuda.synchronize()
            validate(candidate_id, output)
            quotient_score = getattr(args, "profile_quotient_score", None)
            profile_layout_object = LinearInnerLayout(
                name=str(candidate_id),
                matrix_name=matrix.name,
                tile_exponents=matrix.mode_bits,
                a_rows=rows,
                outer_order=tuple(reversed(range(matrix.rank))),
            )
            profile_layout_object.validate(matrix)
            verified_quotient_score = float(
                weighted_component_region_count(
                    matrix, profile_layout_object, issue_component
                )
            )
            if quotient_score is not None and abs(
                float(quotient_score) - verified_quotient_score
            ) > 1e-9:
                raise ValueError(
                    f"{candidate_id}: supplied issue quotient "
                    f"{quotient_score} does not match verified score "
                    f"{verified_quotient_score}"
                )
            quotient_score = verified_quotient_score
            target = {
                "candidate_id": candidate_id,
                "mapping_id": mapping_id,
                "a_rows": list(rows),
                "quotient_score": quotient_score,
                "quotient_components": {
                    "issue": quotient_score,
                    "temporal": 0.0,
                },
                "issue_quotient_score_verified": True,
            }
            target_role = "candidate"

        if execution_layout_spec is None:
            raise ValueError(
                "profile targets require an execution-layout extraction spec"
            )
        reference_compiled = compiled[str(default["candidate_id"])]
        reference_codegen = compiled_codegen_statistics(reference_compiled)
        target_codegen = compiled_codegen_statistics(target_compiled)
        reference_blocked, reference_execution = execution_layout_from_compiled(
            reference_compiled, *execution_layout_spec
        )
        target_blocked, target_execution = execution_layout_from_compiled(
            target_compiled, *execution_layout_spec
        )
        reference_execution_record = execution_layout_record(
            reference_blocked, reference_execution
        )
        target_execution_record = execution_layout_record(
            target_blocked, target_execution
        )

        def opcode_signature(codegen, predicate):
            return {
                opcode: count
                for opcode, count in codegen["opcode_counts"].items()
                if predicate(opcode)
            }

        reference_loads = opcode_signature(reference_codegen, is_load_opcode)
        target_loads = opcode_signature(target_codegen, is_load_opcode)
        reference_stores = opcode_signature(reference_codegen, is_store_opcode)
        target_stores = opcode_signature(target_codegen, is_store_opcode)
        structural_validation = {
            "execution_layout_matches_reference": (
                target_execution_record == reference_execution_record
            ),
            "reference_execution_layout": reference_execution_record,
            "candidate_execution_layout": target_execution_record,
            "load_instruction_structure_matches_reference": (
                target_loads == reference_loads
            ),
            "load_instruction_structure_nonempty": bool(
                reference_loads and target_loads
            ),
            "reference_load_opcode_counts": reference_loads,
            "candidate_load_opcode_counts": target_loads,
            "store_instruction_structure_matches_reference": (
                target_stores == reference_stores
            ),
            "reference_store_opcode_counts": reference_stores,
            "candidate_store_opcode_counts": target_stores,
            "reference_spills": reference_codegen["n_spills"],
            "candidate_spills": target_codegen["n_spills"],
            "no_candidate_spills": target_codegen["n_spills"] == 0,
        }
        structural_validation["accepted"] = all(
            structural_validation[name]
            for name in (
                "execution_layout_matches_reference",
                "load_instruction_structure_nonempty",
                "load_instruction_structure_matches_reference",
                "store_instruction_structure_matches_reference",
                "no_candidate_spills",
            )
        )
        if not structural_validation["accepted"]:
            raise ValueError(
                f"{target['candidate_id']}: counter candidate changed the "
                "execution layout, load/store structure, or spill state"
            )

        cache_mode = args.profile_cache_mode

        def prepare():
            return None

        if cache_mode == "thrashed":
            from stage1_kernels import cache_thrash_kernel

            elements = (args.cache_thrash_bytes + 3) // 4
            cache_buffer = torch.empty(
                elements, dtype=torch.float32, device="cuda"
            )
            grid = ((elements + 255) // 256,)

            def prepare():
                cache_thrash_kernel[grid](
                    cache_buffer, N=elements, BLOCK=256
                )

        for _ in range(args.profile_warmup):
            prepare()
            target_launch()
        torch.cuda.synchronize()
        for _ in range(args.profile_iterations):
            prepare()
            target_launch()
        torch.cuda.synchronize()
        profile_target = {
            "role": target_role,
            "candidate_id": target["candidate_id"],
            "mapping_id": target["mapping_id"],
            "a_rows": target["a_rows"],
            "quotient_score": target["quotient_score"],
            "quotient_components": target["quotient_components"],
            "issue_quotient_score_verified": target.get(
                "issue_quotient_score_verified",
                explicit_profile_rows is None,
            ),
            "cache_mode": cache_mode,
            "cache_thrash_bytes": (
                args.cache_thrash_bytes if cache_mode == "thrashed" else 0
            ),
            "warmup_dispatches": args.profile_warmup,
            "profiled_dispatches": args.profile_iterations,
            "compiled_codegen": target_codegen,
            "structural_validation": structural_validation,
        }
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
                "aggregation": "weighted_raw_sum",
                "components": list(objective_names),
                "weights": objective_weights,
                "tie_breakers": (
                    ["default_layout", "runs", "xors"]
                    if service_name != "none"
                    else ["runs", "xors"]
                ),
            }
        ),
        "objective_components": list(objective_names),
        "hardware_service_model": service_record,
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
            "hardware_service_model": service_name,
        },
        "packing_lower_bound": float(default_packing_bound),
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
        "counter_panel": counter_panel,
        "counter_panels": counter_panels,
        "profile_target": profile_target,
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
    default = next(
        candidate
        for candidate in candidates
        if candidate["layout"] == "row_major"
    )
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
