#!/usr/bin/env python3
"""Rank every retained Stage-1 GEMM B layout or run one profiling target."""

from __future__ import annotations

import argparse
from itertools import permutations
import json
import os
from pathlib import Path
import sys

import torch
import triton

from stage1_common import (
    benchmark_layouts,
    candidate_layout_metadata,
    compiled_codegen_statistics,
    distribution_version,
    execution_layout_from_compiled,
    execution_layout_record,
    issue_events,
    layout_rows,
    pack_tensor,
    positive_integer,
    require_aligned,
    require_power_of_two_multiple,
    solve_layouts,
    stable_id,
    validate_output,
)
from stage1_kernels import TILE_SIZE, gemm_prepacked_b_kernel


HARDWARE_HIERARCHY_SCALES = {
    "register": (8, 16, 32),
    "lane": (64, 256),
    "warp": (256, 512),
    "cta": (1024, 2048),
}

HARDWARE_HIERARCHY_TAUS = {
    "common-sense": {
        "register.8B": 0.125,
        "register.16B": 0.0625,
        "register.32B": 0.03125,
        "lane.64B": 1.0,
        "issue": 0.5,
        "lane.256B": 0.25,
        "warp.256B": 0.25,
        "warp.512B": 0.125,
        "cta.1024B": 0.0625,
        "cta.2048B": 0.03125,
    },
    "mi300a": {
        "register.8B": 0.25,
        "register.16B": 0.25,
        "register.32B": 0.25,
        "lane.64B": 1.0,
        "issue": 1.0,
        "lane.256B": 0.5,
        "warp.256B": 0.125,
        "warp.512B": 1.0,
        "cta.1024B": 0.0625,
        "cta.2048B": 0.03125,
    },
}


def gemm_inputs(n: int):
    i = torch.arange(n, dtype=torch.int64)[:, None]
    j = torch.arange(n, dtype=torch.int64)[None, :]
    logical_a = ((((i * 17 + j * 13) % 31) - 15).to(torch.float32) / 31).half()
    logical_b = ((((i * 11 + j * 19) % 29) - 14).to(torch.float32) / 29).half()
    return logical_a, logical_b, logical_a.float() @ logical_b.float()


def make_launch(n, device_a, device_b, output, rows):
    grid = (n // TILE_SIZE, n // TILE_SIZE)

    def launch():
        return gemm_prepacked_b_kernel[grid](
            device_a,
            device_b,
            output,
            B_ROWS=rows,
            MODE_BITS=n.bit_length() - 1,
            N=n,
            BLOCK_M=TILE_SIZE,
            BLOCK_N=TILE_SIZE,
            BLOCK_K=TILE_SIZE,
            num_warps=4,
        )

    return launch


def run_ranking(args: argparse.Namespace) -> dict[str, object]:
    from relay import (
        MatrixSpec,
        low_address_flag,
        linear_layout_hardware_basis_layout,
        linear_layout_resource_fiber,
        row_major_layout,
        summarize_rank_quality,
        weighted_component_region_count,
    )

    n = args.gemm_size
    matrix = MatrixSpec("B", (n, n), 2, ("k", "n"))
    default_layout = row_major_layout(matrix)
    default_rows = layout_rows(default_layout, matrix)
    logical_a, logical_b, reference = gemm_inputs(n)
    device_a = logical_a.flatten().to("cuda")
    default_b = pack_tensor(logical_b, default_rows).to("cuda")
    require_aligned("GEMM B default", default_b, args.transaction_bytes)
    probe_output = torch.empty((n, n), dtype=torch.float32, device="cuda")
    probe_launch = make_launch(n, device_a, default_b, probe_output, default_rows)
    compiled_probe = probe_launch()
    torch.cuda.synchronize()
    validate_output("GEMM probe", probe_output, reference, rtol=5e-3, atol=1e-1)

    blocked, execution = execution_layout_from_compiled(
        compiled_probe, (TILE_SIZE, TILE_SIZE), ("k", "n")
    )
    grid = (n // TILE_SIZE, n // TILE_SIZE)
    occurrences = grid[0] * grid[1] * (n // TILE_SIZE)
    events = issue_events(
        execution,
        matrix,
        prefix="gemm.B",
        site="gemm.B.load",
        weight=occurrences,
        coordinate_map=lambda coord: coord,
    )
    register_fiber = linear_layout_resource_fiber(
        execution,
        events,
        varying_dimensions=("register",),
        name="gemm_prepacked_b.register",
    )
    register_bytes = matrix.element_bytes * execution.input_size("register")
    register_objective = register_fiber.at_scale(
        register_bytes,
        name=f"gemm_prepacked_b.register.{register_bytes}B",
    )
    register_component = register_objective.build(
        {matrix.name: matrix}, {}, ()
    )[0]
    hierarchy_fibers = {
        "register": register_fiber,
        "lane": linear_layout_resource_fiber(
            execution,
            events,
            varying_dimensions=("lane",),
            name="gemm_prepacked_b.lane",
        ),
        "warp": linear_layout_resource_fiber(
            execution,
            events,
            varying_dimensions=("register", "lane"),
            name="gemm_prepacked_b.warp",
        ),
        "cta": linear_layout_resource_fiber(
            execution,
            events,
            varying_dimensions=("register", "lane", "warp"),
            name="gemm_prepacked_b.cta",
        ),
    }
    hierarchy_objectives = tuple(
        hierarchy_fibers[level].at_scale(
            region_bytes,
            name=f"gemm_prepacked_b.{level}.{region_bytes}B",
        )
        for level in ("register", "lane", "warp", "cta")
        for region_bytes in HARDWARE_HIERARCHY_SCALES[level]
    )
    if args.hardware_hierarchy:
        fiber_objectives = hierarchy_objectives
    elif args.register_fibers:
        fiber_objectives = (register_objective,)
    else:
        fiber_objectives = ()

    issue_name = f"gemm_prepacked_b.issue.{args.transaction_bytes}B"
    objective_taus = None
    if args.hardware_hierarchy:
        relative_taus = HARDWARE_HIERARCHY_TAUS[
            args.hardware_hierarchy_profile
        ]
        objective_taus = {
            (
                issue_name
                if relative_name == "issue"
                else f"gemm_prepacked_b.{relative_name}"
            ): tau
            for relative_name, tau in relative_taus.items()
        }

    hardware_candidates = ()
    if args.hardware_hierarchy:
        active_dimensions = tuple(
            input_name
            for input_name, input_bases in execution.bases
            if input_bases
        )
        hardware_candidates = tuple(
            linear_layout_hardware_basis_layout(
                execution,
                matrix,
                input_dimension_order=dimension_order,
                name="hardware_basis_" + "_".join(dimension_order),
            )
            for dimension_order in permutations(active_dimensions)
        )
    objective, problem, result = solve_layouts(
        (matrix,),
        events,
        args,
        "gemm_prepacked_b",
        inner_tile_shapes={matrix.name: ((TILE_SIZE, TILE_SIZE),)},
        execution_fiber_objectives=fiber_objectives,
        normalize_objectives=bool(fiber_objectives),
        objective_taus=objective_taus,
        candidate_layouts={matrix.name: hardware_candidates},
    )
    component = result.components[0]
    objective_names = (
        (objective,) if isinstance(objective, str) else tuple(objective)
    )
    tau_by_name = {
        component_name: (
            objective_taus.get(component_name, 1.0)
            if objective_taus is not None
            else 1.0
        )
        for component_name in objective_names
    }
    normalized_objective_offset = (
        sum(tau_by_name.values()) if fiber_objectives else 0.0
    )
    retained = result.arrays[matrix.name].candidates
    default_score = weighted_component_region_count(
        matrix, default_layout, component
    )

    launches = {}
    compiled_candidates = {}
    outputs = {}
    records = []
    sources_by_mapping = {
        stable_id("mapping", list(default_rows)): default_b,
    }
    quotient_levels = sorted(
        {float(candidate.scores[component.name]) for candidate in retained}
    )
    objective_levels = sorted(
        {problem.config.policy.key(candidate.scores)[0] for candidate in retained}
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
            source = pack_tensor(logical_b, rows).to("cuda")
            sources_by_mapping[mapping_id] = source
        require_aligned(candidate_id, source, args.transaction_bytes)
        output = torch.empty_like(probe_output)
        launch = make_launch(n, device_a, source, output, rows)
        launches[candidate_id] = launch
        outputs[candidate_id] = output
        compiled_candidates[candidate_id] = launch()
        score = float(candidate.scores[component.name])
        objective_score = (
            problem.config.policy.key(candidate.scores)[0]
            - normalized_objective_offset
        )
        quotient_components = {}
        for component_name in objective_names:
            raw = float(candidate.scores[component_name])
            bound = float(candidate.packing_bounds[component_name])
            normalized_component_excess = raw / bound - 1.0
            quotient_components[component_name] = {
                "quotient_score": raw,
                "packing_bound": bound,
                "normalized_excess": normalized_component_excess,
                "tau": tau_by_name[component_name],
                "weighted_normalized_excess": (
                    tau_by_name[component_name]
                    * normalized_component_excess
                ),
            }
        register_score = weighted_component_region_count(
            matrix, layout, register_component
        )
        register_bound = register_component.packing_bound(matrix)
        register_excess = register_score / register_bound - 1.0
        issue_excess = score / candidate.packing_bounds[component.name] - 1.0
        register_aware_score = issue_excess + register_excess
        hierarchy_score = sum(
            float(values["weighted_normalized_excess"])
            for values in quotient_components.values()
        )
        if args.register_fibers and abs(
            objective_score - register_aware_score
        ) > 1e-9:
            raise ValueError(
                "normalized register-fiber objective decomposition changed"
            )
        if args.hardware_hierarchy and abs(
            objective_score - hierarchy_score
        ) > 1e-9:
            raise ValueError(
                "normalized hardware-hierarchy objective decomposition changed"
            )
        records.append(
            {
                "candidate_id": candidate_id,
                "solver_rank": solver_rank,
                "quotient_rank": quotient_levels.index(score) + 1,
                "objective_rank": objective_levels.index(
                    problem.config.policy.key(candidate.scores)[0]
                )
                + 1,
                "layout": layout.name,
                "grammar": layout.grammar,
                **candidate_layout_metadata(layout, matrix),
                "a_rows": list(rows),
                "mapping_id": mapping_id,
                "flag_id": flag_id,
                "quotient_score": score,
                "packing_bound": float(
                    candidate.packing_bounds[component.name]
                ),
                "objective_score": objective_score,
                "quotient_components": quotient_components,
                "register_fiber_score": float(register_score),
                "register_fiber_packing_bound": float(register_bound),
                "register_fiber_normalized_excess": float(register_excess),
                "register_aware_score": float(register_aware_score),
                "hardware_hierarchy_score": (
                    float(hierarchy_score)
                    if args.hardware_hierarchy
                    else None
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
        validate_output(
            candidate_id,
            outputs[candidate_id],
            reference,
            rtol=5e-3,
            atol=1e-1,
        )
    timings = benchmark_layouts(
        launches,
        samples=args.samples,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    for record in records:
        candidate_id = str(record["candidate_id"])
        timing = timings[candidate_id]
        record["runtime_ms"] = float(timing["median_ms"])
        record["timing"] = timing
        record["compiled_codegen"] = compiled_codegen_statistics(
            compiled_candidates[candidate_id]
        )

    rank_quality = summarize_rank_quality(records)
    objective_rank_quality = summarize_rank_quality(
        records, score_key="objective_score"
    )
    register_aware_rank_quality = summarize_rank_quality(
        records, score_key="register_aware_score"
    )
    hierarchy_rank_quality = (
        summarize_rank_quality(
            records, score_key="hardware_hierarchy_score"
        )
        if args.hardware_hierarchy
        else None
    )
    selected = records[0]
    default = next(record for record in records if record["layout"] == "row_major")
    hierarchy_metadata = {}
    for level, fiber in hierarchy_fibers.items():
        scales = (
            HARDWARE_HIERARCHY_SCALES[level]
            if args.hardware_hierarchy
            else (
                (register_bytes,)
                if args.register_fibers and level == "register"
                else ()
            )
        )
        if not scales:
            continue
        hierarchy_metadata[level] = {
            "varying_dimensions": list(fiber.varying_dimensions),
            "hardware_fiber_count": fiber.hardware_fiber_count,
            "omitted_singleton_count": fiber.omitted_singleton_count,
            "scales": [
                {
                    "region_bytes": region_bytes,
                    "objective": (
                        f"gemm_prepacked_b.{level}.{region_bytes}B"
                    ),
                    "tau": tau_by_name[
                        f"gemm_prepacked_b.{level}.{region_bytes}B"
                    ],
                }
                for region_bytes in scales
            ],
        }
    return {
        "stage": 1.5,
        "experiment": "triton_stage15_gemm_candidate_ranking",
        "matrix_shape": list(matrix.shape),
        "element_bytes": matrix.element_bytes,
        "transaction_bytes": args.transaction_bytes,
        "tile_shape": [TILE_SIZE, TILE_SIZE, TILE_SIZE],
        "num_warps": 4,
        "execution_layout": execution_layout_record(blocked, execution),
        "dynamic_occurrences_per_event": occurrences,
        "induced_event_count": len(events),
        "objective": objective,
        "execution_fibers": {
            "enabled": bool(fiber_objectives),
            "mode": (
                "hardware_hierarchy"
                if args.hardware_hierarchy
                else "register"
                if args.register_fibers
                else "issue_only"
            ),
            "aggregation": (
                "tau-weighted normalized excess"
                if args.hardware_hierarchy
                else "equal-weight normalized excess"
                if args.register_fibers
                else None
            ),
            "profile": (
                args.hardware_hierarchy_profile
                if args.hardware_hierarchy
                else None
            ),
            "issue_tau": tau_by_name[component.name],
            "levels": hierarchy_metadata,
        },
        "search_scope": {
            "grammar": (
                "canonical_inner_tile_plus_hardware_basis_permutations"
                if args.hardware_hierarchy
                else "canonical_inner_tile"
            ),
            "inner_tile_shape": [TILE_SIZE, TILE_SIZE],
            "outer_layout": "row_major_tiles",
            "fixed_outer_order": list(reversed(matrix.mode_names)),
            "hardware_basis_candidate_count": len(hardware_candidates),
            "hardware_basis_orders": [
                layout.name.removeprefix("hardware_basis_").split("_")
                for layout in hardware_candidates
            ],
        },
        "packing_lower_bound": component.packing_bound(matrix),
        "default_candidate_id": default["candidate_id"],
        "selected_candidate_id": selected["candidate_id"],
        "default": default,
        "selected": selected,
        "candidates": records,
        "rank_quality": rank_quality,
        "objective_rank_quality": objective_rank_quality,
        "register_aware_rank_quality": register_aware_rank_quality,
        "hardware_hierarchy_rank_quality": hierarchy_rank_quality,
        "predicted_transaction_reduction": 1.0
        - float(selected["quotient_score"]) / default_score,
        "measured_speedup": float(default["runtime_ms"])
        / float(selected["runtime_ms"]),
        "correct": True,
        "process": {
            "pid": os.getpid(),
            "torch_version": torch.__version__,
            "triton_version": distribution_version("triton"),
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "solver": {
            "elapsed_seconds": result.elapsed_seconds,
            "candidate_count": result.arrays[matrix.name].all_candidate_count,
            "retained_candidates": len(retained),
            "name": problem.name,
        },
    }


def run_profile_target(args: argparse.Namespace) -> dict[str, object]:
    rows = tuple(args.profile_rows)
    n = args.gemm_size
    expected_rows = 2 * (n.bit_length() - 1)
    if len(rows) != expected_rows:
        raise ValueError(
            f"profile layout has {len(rows)} rows; expected {expected_rows}"
        )
    logical_a, logical_b, reference = gemm_inputs(n)
    device_a = logical_a.flatten().to("cuda")
    device_b = pack_tensor(logical_b, rows).to("cuda")
    require_aligned("profile GEMM B", device_b, args.transaction_bytes)
    output = torch.empty((n, n), dtype=torch.float32, device="cuda")
    launch = make_launch(n, device_a, device_b, output, rows)
    compiled = launch()
    torch.cuda.synchronize()
    validate_output("profile GEMM", output, reference, rtol=5e-3, atol=1e-1)
    for _ in range(args.profile_warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.profile_iterations):
        launch()
    torch.cuda.synchronize()
    return {
        "stage": 1.5,
        "experiment": "triton_stage15_gemm_profile_target",
        "matrix_shape": [n, n],
        "a_rows": list(rows),
        "profile_warmup": args.profile_warmup,
        "profile_iterations": args.profile_iterations,
        "compiled_codegen": compiled_codegen_statistics(compiled),
        "correct": True,
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemm-size", type=positive_integer, default=512)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    fiber_mode = parser.add_mutually_exclusive_group()
    fiber_mode.add_argument(
        "--register-fibers",
        action="store_true",
        help=(
            "add per-lane register-ownership fibers at their exact byte "
            "footprint and rank by equal-weight normalized excess"
        ),
    )
    fiber_mode.add_argument(
        "--hardware-hierarchy",
        action="store_true",
        help=(
            "add register, lane, warp-fragment, and CTA-fragment fibers at "
            "multiple address scales and seed hardware-basis layouts"
        ),
    )
    parser.add_argument(
        "--hardware-hierarchy-profile",
        choices=tuple(HARDWARE_HIERARCHY_TAUS),
        default="mi300a",
        help="tau profile for --hardware-hierarchy",
    )
    parser.add_argument("--samples", type=positive_integer, default=21)
    parser.add_argument("--iterations", type=positive_integer, default=50)
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--profile-rows", type=int, nargs="+")
    parser.add_argument("--profile-iterations", type=positive_integer, default=20)
    parser.add_argument("--profile-warmup", type=positive_integer, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    args = parse_arguments()
    require_power_of_two_multiple("GEMM size", args.gemm_size, TILE_SIZE)
    if not torch.cuda.is_available():
        raise RuntimeError("the Stage 1.5 GEMM experiment requires a Flux GPU")
    result = (
        run_profile_target(args) if args.profile_rows is not None else run_ranking(args)
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
