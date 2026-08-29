#!/usr/bin/env python3
"""Run one controlled same-flag Stage-2 probe on cache-thrashed GEMM B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import triton

from stage1_common import (
    benchmark_layouts,
    benchmark_layouts_isolated,
    compiled_codegen_statistics,
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
)
from stage1_kernels import cache_thrash_kernel, gemm_prepacked_b_general_kernel
from stage2_probe import analyze_fiber_candidates, build_gemm_b_resource_groups


def logical_inputs(m: int, n: int, k: int):
    ai = torch.arange(m, dtype=torch.int64)[:, None]
    aj = torch.arange(k, dtype=torch.int64)[None, :]
    bi = torch.arange(k, dtype=torch.int64)[:, None]
    bj = torch.arange(n, dtype=torch.int64)[None, :]
    logical_a = (
        (((ai * 17 + aj * 13) % 31) - 15).to(torch.float32) / 31
    ).half()
    logical_b = (
        (((bi * 11 + bj * 19) % 29) - 14).to(torch.float32) / 29
    ).half()
    return logical_a, logical_b


def make_launch(args, device_a, source, output, rows):
    grid = (args.m // args.block_m, args.n // args.block_n)

    def launch():
        return gemm_prepacked_b_general_kernel[grid](
            device_a,
            source,
            output,
            B_ROWS=rows,
            B_FIRST_BITS=args.k.bit_length() - 1,
            M=args.m,
            N=args.n,
            K=args.k,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            BLOCK_K=args.block_k,
            TRANS_A=False,
            TRANS_B=False,
            num_warps=args.num_warps,
        )

    return launch


def validate_output(label: str, output: torch.Tensor, reference: torch.Tensor):
    if torch.allclose(output, reference, rtol=5e-3, atol=1e-1):
        return
    error = torch.max(torch.abs(output - reference)).item()
    raise ValueError(f"{label} produced incorrect output: max error {error}")


def placement_record(placement) -> dict[str, object]:
    return {
        "name": placement.name,
        "cohort_family": placement.cohort_family,
        "transaction_bytes": placement.transaction_bytes,
        "color_count": placement.color_count,
        "phase_policy": placement.phase_policy,
        "cohort_count": placement.cohort_count,
        "cohort_weight": placement.cohort_weight,
        "raw_pair_excess": placement.raw_pair_excess,
        "normalized_contention": placement.normalized_contention,
        "within_contention": placement.within_contention,
        "cross_contention": placement.cross_contention,
        "weighted_contention": placement.weighted_contention,
    }


def run_probe(args) -> dict[str, object]:
    from relay import (
        get_hardware_profile,
        layout_codegen_cost,
        low_address_flag,
        MatrixSpec,
        resource_color_destination_bits,
        score_resource_placement,
        weighted_component_region_count,
        enumerate_flag_preserving_swizzles,
        row_major_layout,
    )

    matrix = MatrixSpec("B", (args.k, args.n), 2, ("k", "n"))
    default_layout = row_major_layout(matrix)
    default_rows = layout_rows(default_layout, matrix)
    logical_a, logical_b = logical_inputs(args.m, args.n, args.k)
    device_a = logical_a.flatten().to("cuda")
    reference = device_a.reshape(args.m, args.k).float() @ logical_b.to(
        "cuda"
    ).float()
    default_b = pack_tensor(logical_b, default_rows).to("cuda")
    require_aligned("Stage-2 default B", default_b, args.transaction_bytes)
    output = torch.empty(
        (args.m, args.n), dtype=torch.float32, device="cuda"
    )
    probe_launch = make_launch(args, device_a, default_b, output, default_rows)
    compiled_probe = probe_launch()
    torch.cuda.synchronize()
    validate_output("Stage-2 default probe", output, reference)

    blocked, execution = execution_layout_from_compiled(
        compiled_probe,
        (args.block_k, args.block_n),
        ("k", "n"),
    )
    occurrences = (
        (args.m // args.block_m)
        * (args.n // args.block_n)
        * (args.k // args.block_k)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="stage2_probe.B",
        site="stage2_probe.B.load",
        weight=occurrences,
        coordinate_map=lambda coord: coord,
    )
    objective, _problem, solve_result = solve_layouts(
        (matrix,),
        events,
        args,
        "stage2_probe",
        inner_tile_shapes={
            matrix.name: (args.block_k, args.block_n)
        },
    )
    component = solve_result.components[0]
    selected_candidate = solve_result.arrays[matrix.name].candidates[0]
    selected_layout = selected_candidate.layout
    selected_rows = layout_rows(selected_layout, matrix)
    selected_quotient = weighted_component_region_count(
        matrix, selected_layout, component
    )
    selected_flag = low_address_flag(matrix, selected_layout)
    flag_id = stable_id("flag", selected_flag)

    profile = get_hardware_profile(args.hardware_profile)
    destination_bits = resource_color_destination_bits(
        profile.resource_maps,
        matrix.element_bytes,
        selected_layout.inner_bits,
    )
    seeds = enumerate_flag_preserving_swizzles(
        matrix,
        selected_layout,
        max_xors=args.fiber_max_xors,
        destination_bits=destination_bits,
    )
    if not args.minimum_realizations <= len(seeds) <= args.maximum_realizations:
        raise ValueError(
            f"fiber probe generated {len(seeds)} realizations; expected between "
            f"{args.minimum_realizations} and {args.maximum_realizations}"
        )
    if len(profile.resource_maps) != 1:
        raise ValueError("the controlled probe requires one resource map")
    resource_map = profile.resource_maps[0]
    resource_groups = build_gemm_b_resource_groups(
        matrix,
        execution,
        m=args.m,
        n=args.n,
        k=args.k,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        resource_map=resource_map,
    )

    launches = {}
    records = []
    compiled = {}
    for seed in seeds:
        layout = seed.layout
        rows = layout_rows(layout, matrix)
        candidate_id = stable_id(
            "fiber",
            {"a_rows": rows, "shears": seed.shears, "flag": flag_id},
        )
        observed_flag = low_address_flag(matrix, layout)
        if observed_flag != selected_flag:
            raise ValueError(f"{candidate_id} does not preserve the selected flag")
        quotient = weighted_component_region_count(matrix, layout, component)
        if quotient != selected_quotient:
            raise ValueError(
                f"{candidate_id} changed quotient {selected_quotient} to {quotient}"
            )
        placement = score_resource_placement(
            {matrix.name: matrix},
            {matrix.name: layout},
            {resource_map.cohort_family: resource_groups},
            profile.resource_maps,
        )[0]
        codegen = layout_codegen_cost(
            {matrix.name: matrix}, {matrix.name: layout}
        )
        source = pack_tensor(logical_b, rows).to("cuda")
        require_aligned(candidate_id, source, args.transaction_bytes)
        launch = make_launch(args, device_a, source, output, rows)
        launches[candidate_id] = launch
        compiled[candidate_id] = launch()
        torch.cuda.synchronize()
        validate_output(candidate_id, output, reference)
        records.append(
            {
                "candidate_id": candidate_id,
                "identity": not seed.shears,
                "shears": [list(shear) for shear in seed.shears],
                "layout": layout.name,
                "a_rows": list(rows),
                "flag_id": flag_id,
                "quotient_score": float(quotient),
                "quotient_invariant": True,
                "resource_service_score": placement.weighted_contention,
                "resource_service": placement_record(placement),
                "codegen_cost": {
                    "runs": codegen.runs,
                    "xors": codegen.xors,
                    "swizzle_xors": seed.swizzle_xors,
                },
            }
        )

    if args.cache_mode == "thrashed":
        elements = (args.cache_thrash_bytes + 3) // 4
        cache_buffer = torch.empty(elements, dtype=torch.float32, device="cuda")
        grid = (triton.cdiv(elements, 256),)

        def before_each():
            cache_thrash_kernel[grid](cache_buffer, N=elements, BLOCK=256)

        timings = benchmark_layouts_isolated(
            launches,
            before_each=before_each,
            samples=args.samples,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    else:
        timings = benchmark_layouts(
            launches,
            samples=args.samples,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    for record in records:
        candidate_id = str(record["candidate_id"])
        record["runtime_ms"] = float(timings[candidate_id]["median_ms"])
        record["timing"] = timings[candidate_id]
        record["compiled_codegen"] = compiled_codegen_statistics(
            compiled[candidate_id]
        )

    analysis = analyze_fiber_candidates(
        records,
        meaningful_spread=args.meaningful_spread,
        predictive_correlation=args.predictive_correlation,
        maximum_service_regret=args.maximum_service_regret,
    )
    return {
        "stage": 2,
        "experiment": "triton_stage2_controlled_flag_fiber_probe",
        "configuration": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "block_k": args.block_k,
            "num_warps": args.num_warps,
            "transaction_bytes": args.transaction_bytes,
            "cache_mode": args.cache_mode,
            "cache_thrash_bytes": (
                args.cache_thrash_bytes
                if args.cache_mode == "thrashed"
                else None
            ),
            "samples": args.samples,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "fiber_max_xors": args.fiber_max_xors,
            "destination_bits": list(destination_bits),
        },
        "stage1_selected": {
            "layout": selected_layout.name,
            "word": selected_layout.word_string(matrix),
            "a_rows": list(selected_rows),
            "flag_id": flag_id,
            "quotient_score": float(selected_quotient),
            "solver_objective": objective,
        },
        "execution_layout": execution_layout_record(blocked, execution),
        "resource_service_model": {
            "hardware_profile": args.hardware_profile,
            "profile_id": profile.profile_id,
            "resource_map": resource_map.to_dict(),
            "scope": "prepacked B only",
            "cohort_semantics": "four B issue cohorts per warp and B tile",
            "translation_group_count": len(resource_groups),
            "dynamic_cohort_count": sum(
                len(group.occurrences) for group in resource_groups
            ),
            "dynamic_occurrences_per_B_issue": occurrences,
        },
        "candidates": records,
        "analysis": analysis,
        "process": {
            "pid": os.getpid(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "correct": True,
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_integer, default=2048)
    parser.add_argument("--n", type=positive_integer, default=2048)
    parser.add_argument("--k", type=positive_integer, default=2048)
    parser.add_argument("--block-m", type=positive_integer, default=32)
    parser.add_argument("--block-n", type=positive_integer, default=32)
    parser.add_argument("--block-k", type=positive_integer, default=32)
    parser.add_argument("--num-warps", type=positive_integer, default=4)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--fiber-max-xors", type=positive_integer, default=1)
    parser.add_argument("--minimum-realizations", type=positive_integer, default=16)
    parser.add_argument("--maximum-realizations", type=positive_integer, default=64)
    parser.add_argument("--hardware-profile", default="mi300a")
    parser.add_argument("--cache-mode", choices=("warm", "thrashed"), default="thrashed")
    parser.add_argument(
        "--cache-thrash-bytes", type=positive_integer, default=256 << 20
    )
    parser.add_argument("--samples", type=positive_integer, default=9)
    parser.add_argument("--iterations", type=positive_integer, default=1)
    parser.add_argument("--warmup", type=positive_integer, default=1)
    parser.add_argument("--meaningful-spread", type=float, default=0.02)
    parser.add_argument("--predictive-correlation", type=float, default=0.5)
    parser.add_argument("--maximum-service-regret", type=float, default=0.02)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    args = parse_arguments()
    for label, extent, block in (
        ("M", args.m, args.block_m),
        ("N", args.n, args.block_n),
        ("K", args.k, args.block_k),
    ):
        require_power_of_two_multiple(label, extent, block)
    if args.block_m != args.block_k:
        raise ValueError("the current execution bridge requires BLOCK_M == BLOCK_K")
    if args.num_warps not in (1, 2, 4, 8):
        raise ValueError("num warps must be one of 1, 2, 4, or 8")
    for label in (
        "meaningful_spread",
        "predictive_correlation",
        "maximum_service_regret",
    ):
        value = getattr(args, label)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must lie in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("the Stage-2 probe requires a Flux GPU")
    result = run_probe(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
