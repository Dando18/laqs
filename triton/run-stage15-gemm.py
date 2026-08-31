#!/usr/bin/env python3
"""Rank every retained Stage-1 GEMM B layout or run one profiling target."""

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
    canonical_layout_metadata,
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
    objective, problem, result = solve_layouts(
        (matrix,),
        events,
        args,
        "gemm_prepacked_b",
        inner_tile_shapes={matrix.name: ((TILE_SIZE, TILE_SIZE),)},
    )
    component = result.components[0]
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
    score_levels = sorted(
        {float(candidate.scores[objective]) for candidate in retained}
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
        score = float(candidate.scores[objective])
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
                "packing_bound": float(candidate.packing_bounds[objective]),
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
    selected = records[0]
    default = next(record for record in records if record["layout"] == "row_major")
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
        "search_scope": {
            "grammar": "canonical_inner_tile",
            "inner_tile_shape": [TILE_SIZE, TILE_SIZE],
            "outer_layout": "row_major_tiles",
            "fixed_outer_order": list(reversed(matrix.mode_names)),
        },
        "packing_lower_bound": component.packing_bound(matrix),
        "default_candidate_id": default["candidate_id"],
        "selected_candidate_id": selected["candidate_id"],
        "default": default,
        "selected": selected,
        "candidates": records,
        "rank_quality": rank_quality,
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
