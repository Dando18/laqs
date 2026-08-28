#!/usr/bin/env python3
"""Solve and benchmark Stage 1's execution-conditioned quotient layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.language as tl
from triton.tools import LinearLayout as NativeLinearLayout


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


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def layout_rows(layout, matrix) -> tuple[int, ...]:
    from relay import CanonicalLayout, LinearInnerLayout

    if isinstance(layout, CanonicalLayout):
        rows = layout.matrix_rows()
    elif isinstance(layout, LinearInnerLayout):
        rows = layout.a_rows
    else:
        raise TypeError(f"unsupported Stage 1 layout {type(layout).__name__}")
    if layout.tile_exponents != matrix.mode_bits:
        raise ValueError("the Stage 1 benchmark requires a full-matrix layout")
    return tuple(rows)


def physical_offsets(
    matrix_size: int,
    mode_bits: int,
    rows: tuple[int, ...],
) -> torch.Tensor:
    first = torch.arange(matrix_size, dtype=torch.int64)[:, None]
    second = torch.arange(matrix_size, dtype=torch.int64)[None, :]
    logical = first | (second << mode_bits)
    physical = torch.zeros_like(logical)
    for physical_bit, row in enumerate(rows):
        selected = logical & row
        selected ^= selected >> 32
        selected ^= selected >> 16
        selected ^= selected >> 8
        selected ^= selected >> 4
        selected ^= selected >> 2
        selected ^= selected >> 1
        physical |= (selected & 1) << physical_bit
    return physical


def pack_matrix(
    logical: torch.Tensor,
    mode_bits: int,
    rows: tuple[int, ...],
) -> torch.Tensor:
    offsets = physical_offsets(logical.shape[0], mode_bits, rows)
    packed = torch.empty_like(logical).flatten()
    packed[offsets.flatten()] = logical.flatten()
    return packed


def validate_output(
    label: str, output: torch.Tensor, reference: torch.Tensor
) -> None:
    observed = output.cpu()
    if torch.allclose(observed, reference, rtol=1e-4, atol=1e-3):
        return
    error = torch.max(torch.abs(observed - reference)).item()
    raise ValueError(f"{label} layout produced incorrect output: max error {error}")


def timing_summary(samples_ms: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "stdev_ms": (
            statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        ),
        "samples_ms": samples_ms,
    }


def benchmark_layouts(
    launches,
    *,
    samples: int,
    iterations: int,
    warmup: int,
) -> dict[str, dict[str, object]]:
    labels = tuple(launches)
    for _ in range(warmup):
        for label in labels:
            launches[label]()
    torch.cuda.synchronize()

    timings = {label: [] for label in labels}
    for sample in range(samples):
        order = labels if sample % 2 == 0 else tuple(reversed(labels))
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                launches[label]()
            stop.record()
            stop.synchronize()
            timings[label].append(float(start.elapsed_time(stop)) / iterations)
    return {label: timing_summary(values) for label, values in timings.items()}


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
        row_major_layout,
        solve,
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
    default_source = pack_matrix(logical, mode_bits, default_rows).to("cuda")
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
            tile_shapes={matrix.name: (matrix.shape,)},
            general_tile_shapes={matrix.name: ()},
            include_global_canonical=False,
            enable_linear_inner=False,
            canonical_candidates_per_tile=args.candidates,
            primary_tolerance=0.0,
            per_array_candidates=max(args.candidates, 4),
            joint_candidates=max(args.candidates, 4),
        ),
        name="triton_stage1_symmetric_sum",
    )
    result = solve(problem)
    component = result.components[0]
    best = result.arrays[matrix.name].candidates[0]
    best_layout = best.layout
    best_rows = layout_rows(best_layout, matrix)
    default_score = weighted_component_region_count(
        matrix, default_layout, component
    )
    best_score = float(best.scores[objective_name])

    best_source = pack_matrix(logical, mode_bits, best_rows).to("cuda")
    best_output = torch.empty_like(default_output)
    if best_source.data_ptr() % args.transaction_bytes:
        raise ValueError("LAQS allocation is not transaction-aligned")

    def launch_default():
        symmetric_sum[(args.matrix_size,)](
            default_source,
            default_output,
            A_ROWS=default_rows,
            MODE_BITS=mode_bits,
            N=args.matrix_size,
            BLOCK=BLOCK_SIZE,
            num_warps=1,
        )

    def launch_laqs():
        symmetric_sum[(args.matrix_size,)](
            best_source,
            best_output,
            A_ROWS=best_rows,
            MODE_BITS=mode_bits,
            N=args.matrix_size,
            BLOCK=BLOCK_SIZE,
            num_warps=1,
        )

    launch_laqs()
    torch.cuda.synchronize()
    validate_output("LAQS", best_output, reference)
    timings = benchmark_layouts(
        {"default": launch_default, "laqs": launch_laqs},
        samples=args.samples,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    default_median = float(timings["default"]["median_ms"])
    best_median = float(timings["laqs"]["median_ms"])

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
        "packing_lower_bound": component.packing_bound(matrix),
        "default": {
            "layout": default_layout.name,
            "word": default_layout.word_string(matrix),
            "a_rows": list(default_rows),
            "predicted_transactions": default_score,
            "timing": timings["default"],
        },
        "laqs": {
            "layout": best_layout.name,
            "grammar": best_layout.grammar,
            "word": best_layout.word_string(matrix),
            "a_rows": list(best_rows),
            "predicted_transactions": best_score,
            "packing_bound": best.packing_bounds[objective_name],
            "runs": best_layout.runs,
            "xor_count": best_layout.xor_count,
            "exact": best.exact,
            "timing": timings["laqs"],
        },
        "predicted_transaction_reduction": 1.0 - best_score / default_score,
        "measured_speedup": default_median / best_median,
        "correct": True,
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
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    result = run_experiment(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
