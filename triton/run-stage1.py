#!/usr/bin/env python3
"""Solve and benchmark Stage 1's execution-conditioned quotient layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
from triton.tools import LinearLayout as NativeLinearLayout

from stage1_common import (
    benchmark_layouts,
    canonical_layout_metadata,
    compiled_codegen_statistics,
    layout_rows,
    pack_tensor,
    positive_integer,
    stable_id,
    validate_output,
)


BLOCK_SIZE = 64
ELEMENT_BYTES = 4


@triton.jit
def linear_offset(
    first,
    second,
    A_ROWS: tl.constexpr,
    MODE_BITS: tl.constexpr,
):
    logical = first | (second << MODE_BITS)
    physical = logical ^ logical
    for physical_bit in tl.static_range(len(A_ROWS)):
        selected = logical & A_ROWS[physical_bit]
        selected ^= selected >> 16
        selected ^= selected >> 8
        selected ^= selected >> 4
        selected ^= selected >> 2
        selected ^= selected >> 1
        physical |= (selected & 1) << physical_bit
    return physical


@triton.jit
def symmetric_sum(
    source,
    output,
    A_ROWS: tl.constexpr,
    MODE_BITS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    first = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for base in tl.range(0, N, BLOCK):
        second = base + lanes
        row_offsets = linear_offset(first, second, A_ROWS, MODE_BITS)
        column_offsets = linear_offset(second, first, A_ROWS, MODE_BITS)
        accumulator += tl.load(source + row_offsets)
        accumulator += tl.load(source + column_offsets)
    tl.store(output + first, tl.sum(accumulator, axis=0))


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the Stage 1 experiment requires a Flux GPU allocation")
    if args.matrix_size < BLOCK_SIZE or args.matrix_size % BLOCK_SIZE:
        raise ValueError(
            f"matrix size must be a multiple of the {BLOCK_SIZE}-lane wave"
        )
    if args.matrix_size & (args.matrix_size - 1):
        raise ValueError("matrix size must be a power of two")

    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from relay import (
        MatrixSpec,
        ScorePolicy,
        SolverConfig,
        TritonLinearLayout,
        execution_conditioned_quotient_problem,
        extract_blocked_layout,
        induce_memory_event,
        low_address_flag,
        row_major_layout,
        solve,
        summarize_rank_quality,
        weighted_component_region_count,
    )

    matrix = MatrixSpec(
        "A",
        (args.matrix_size, args.matrix_size),
        ELEMENT_BYTES,
        ("i", "j"),
    )
    mode_bits = matrix.mode_bits[0]
    default_layout = row_major_layout(matrix)
    default_rows = layout_rows(default_layout, matrix)

    first = torch.arange(args.matrix_size, dtype=torch.int64)[:, None]
    second = torch.arange(args.matrix_size, dtype=torch.int64)[None, :]
    logical = (((first * 17 + second * 13) % 101) - 50).to(torch.float32)
    logical /= 101.0
    reference = logical.sum(dim=1) + logical.sum(dim=0)
    default_source = pack_tensor(logical, default_rows).to("cuda")
    default_output = torch.empty(args.matrix_size, dtype=torch.float32, device="cuda")
    if default_source.data_ptr() % args.transaction_bytes:
        raise ValueError("default allocation is not transaction-aligned")

    compiled = symmetric_sum[(args.matrix_size,)](
        default_source,
        default_output,
        A_ROWS=default_rows,
        MODE_BITS=mode_bits,
        N=args.matrix_size,
        BLOCK=BLOCK_SIZE,
        num_warps=1,
    )
    torch.cuda.synchronize()
    validate_output("default", default_output, reference)

    blocked = extract_blocked_layout(compiled.asm["ttgir"])
    extracted = TritonLinearLayout.from_blocked(
        (BLOCK_SIZE,),
        size_per_thread=blocked.size_per_thread,
        threads_per_warp=blocked.threads_per_warp,
        warps_per_cta=blocked.warps_per_cta,
        order=blocked.order,
    )
    native = NativeLinearLayout.from_bases(
        extracted.bases,
        [name for name, _ in extracted.out_dims],
        [size for _, size in extracted.out_dims],
    )
    execution_layout = TritonLinearLayout.from_triton(native)
    cohort = execution_layout.locations(
        fixed={"register": 0, "warp": 0, "block": 0}
    )
    event_weight = args.matrix_size * (args.matrix_size // BLOCK_SIZE)
    induced = (
        induce_memory_event(
            execution_layout,
            matrix,
            cohort,
            id="symmetric_sum.row",
            site="symmetric_sum.row.load",
            weight=event_weight,
            coordinate_map=lambda coord: (0, coord[0]),
        ),
        induce_memory_event(
            execution_layout,
            matrix,
            cohort,
            id="symmetric_sum.column",
            site="symmetric_sum.column.load",
            weight=event_weight,
            coordinate_map=lambda coord: (coord[0], 0),
        ),
    )

    objective_name = f"triton.issue.{args.transaction_bytes}B"
    problem = execution_conditioned_quotient_problem(
        (matrix,),
        induced,
        transaction_bytes=args.transaction_bytes,
        objective_name=objective_name,
        config=SolverConfig(
            policy=ScorePolicy(
                "lexicographic", (objective_name, "runs", "xors")
            ),
            tile_shapes={
                matrix.name: ((BLOCK_SIZE, BLOCK_SIZE),)
            },
            general_tile_shapes={matrix.name: ()},
            include_global_canonical=False,
            enable_linear_inner=False,
            include_column_major_control=False,
            include_tiled_row_major_control=True,
            canonical_candidates_per_tile=args.candidates,
            primary_tolerance=0.0,
            per_array_candidates=max(args.candidates, 4),
            joint_candidates=max(args.candidates, 4),
        ),
        name="triton_stage1_symmetric_sum",
    )
    result = solve(problem)
    component = result.components[0]
    retained = result.arrays[matrix.name].candidates
    best = retained[0]
    default_score = weighted_component_region_count(
        matrix, default_layout, component
    )
    best_score = float(best.scores[objective_name])

    launches = {}
    compiled_candidates = {}
    candidate_records = []
    sources_by_mapping = {
        stable_id("mapping", list(default_rows)): default_source,
    }
    outputs = {}
    score_levels = sorted(
        {float(candidate.scores[objective_name]) for candidate in retained}
    )
    for solver_rank, candidate in enumerate(retained, start=1):
        layout = candidate.layout
        rows = layout_rows(layout, matrix)
        mapping_id = stable_id("mapping", list(rows))
        flag = low_address_flag(matrix, layout)
        flag_id = stable_id("flag", flag)
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
            source = pack_tensor(logical, rows).to("cuda")
            sources_by_mapping[mapping_id] = source
        if source.data_ptr() % args.transaction_bytes:
            raise ValueError(f"{candidate_id} allocation is not transaction-aligned")
        output = torch.empty_like(default_output)
        outputs[candidate_id] = output

        def launch(source=source, output=output, rows=rows):
            return symmetric_sum[(args.matrix_size,)](
                source,
                output,
                A_ROWS=rows,
                MODE_BITS=mode_bits,
                N=args.matrix_size,
                BLOCK=BLOCK_SIZE,
                num_warps=1,
            )

        launches[candidate_id] = launch
        compiled_candidates[candidate_id] = launch()
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "solver_rank": solver_rank,
                "quotient_rank": score_levels.index(
                    float(candidate.scores[objective_name])
                )
                + 1,
                "layout": layout.name,
                "grammar": layout.grammar,
                **canonical_layout_metadata(layout, matrix),
                "a_rows": list(rows),
                "mapping_id": mapping_id,
                "flag_id": flag_id,
                "quotient_score": float(candidate.scores[objective_name]),
                "packing_bound": float(candidate.packing_bounds[objective_name]),
                "runs": int(candidate.scores["runs"]),
                "xor_count": layout.xor_count,
                "exact": candidate.exact,
                "note": candidate.note,
            }
        )

    torch.cuda.synchronize()
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        validate_output(candidate_id, outputs[candidate_id], reference)
    timings = benchmark_layouts(
        launches,
        samples=args.samples,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        timing = timings[candidate_id]
        record["runtime_ms"] = float(timing["median_ms"])
        record["timing"] = timing
        record["compiled_codegen"] = compiled_codegen_statistics(
            compiled_candidates[candidate_id]
        )

    rank_quality = summarize_rank_quality(candidate_records)
    best_record = candidate_records[0]
    default_record = next(
        record for record in candidate_records if record["layout"] == "row_major"
    )
    default_median = float(default_record["runtime_ms"])
    best_median = float(best_record["runtime_ms"])

    return {
        "stage": 1,
        "experiment": problem.name,
        "matrix_shape": list(matrix.shape),
        "element_bytes": matrix.element_bytes,
        "transaction_bytes": args.transaction_bytes,
        "compiled_blocked_layout": {
            name: list(values) for name, values in blocked.as_dict().items()
        },
        "issue_cohort_size": len(cohort),
        "representative_hyperedges": {
            item.event.id: [list(point) for point in item.hyperedge.points]
            for item in induced
        },
        "dynamic_issue_multiplicity_per_orientation": event_weight,
        "objective": objective_name,
        "search_scope": {
            "grammar": "canonical_inner_tile",
            "inner_tile_shape": [BLOCK_SIZE, BLOCK_SIZE],
            "outer_layout": "row_major_tiles",
            "fixed_outer_order": list(reversed(matrix.mode_names)),
        },
        "packing_lower_bound": component.packing_bound(matrix),
        "default": {
            "candidate_id": default_record["candidate_id"],
            "layout": default_record["layout"],
            "word": default_record["word"],
            "a_rows": default_record["a_rows"],
            "predicted_transactions": default_score,
            "timing": default_record["timing"],
            "compiled_codegen": default_record["compiled_codegen"],
        },
        "laqs": {
            "candidate_id": best_record["candidate_id"],
            "layout": best_record["layout"],
            "grammar": best_record["grammar"],
            "word": best_record["word"],
            "a_rows": best_record["a_rows"],
            "predicted_transactions": best_score,
            "packing_bound": best_record["packing_bound"],
            "runs": best_record["runs"],
            "xor_count": best_record["xor_count"],
            "exact": best_record["exact"],
            "timing": best_record["timing"],
            "compiled_codegen": best_record["compiled_codegen"],
        },
        "candidates": candidate_records,
        "rank_quality": rank_quality,
        "predicted_transaction_reduction": 1.0 - best_score / default_score,
        "measured_speedup": default_median / best_median,
        "correct": True,
        "process": {
            "pid": os.getpid(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "solver": {
            "elapsed_seconds": result.elapsed_seconds,
            "candidate_count": result.arrays[matrix.name].all_candidate_count,
            "retained_candidates": len(result.arrays[matrix.name].candidates),
        },
    }


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=positive_integer, default=1024)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--samples", type=positive_integer, default=7)
    parser.add_argument("--iterations", type=positive_integer, default=10)
    parser.add_argument("--warmup", type=positive_integer, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress JSON on stdout (the --json file is still written)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    result = run_experiment(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
