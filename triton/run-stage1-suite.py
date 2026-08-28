#!/usr/bin/env python3
"""Run the targeted four-regime execution-conditioned Stage 1 suite."""

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
    compiled_codegen_statistics,
    layout_rows,
    pack_tensor,
    positive_integer,
    validate_output,
)


WAVE_SIZE = 64
TILE_SIZE = 32


@triton.jit
def linear_offset(first, second, A_ROWS: tl.constexpr, FIRST_BITS: tl.constexpr):
    logical = first | (second << FIRST_BITS)
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
def linear_offset_1d(index, A_ROWS: tl.constexpr):
    physical = index ^ index
    for physical_bit in tl.static_range(len(A_ROWS)):
        selected = index & A_ROWS[physical_bit]
        selected ^= selected >> 16
        selected ^= selected >> 8
        selected ^= selected >> 4
        selected ^= selected >> 2
        selected ^= selected >> 1
        physical |= (selected & 1) << physical_bit
    return physical


@triton.jit
def vector_add_kernel(
    x,
    y,
    output,
    X_ROWS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    physical_x = linear_offset_1d(offsets, X_ROWS)
    tl.store(output + offsets, tl.load(x + physical_x) + tl.load(y + offsets))


@triton.jit
def tiled_sum_kernel(
    source,
    output,
    A_ROWS: tl.constexpr,
    MODE_BITS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program = tl.program_id(0)
    tiles_n = N // BLOCK_N
    tile_m = program // tiles_n
    tile_n = program % tiles_n
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offsets = linear_offset(rows, columns, A_ROWS, MODE_BITS)
    values = tl.load(source + offsets)
    row_sums = tl.sum(values, axis=1)
    tl.store(output + program, tl.sum(row_sums, axis=0))


@triton.jit
def gesummv_kernel(
    a,
    b,
    x,
    output,
    A_ROWS: tl.constexpr,
    B_ROWS: tl.constexpr,
    MODE_BITS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sum_a = tl.zeros((BLOCK,), tl.float32)
    sum_b = tl.zeros((BLOCK,), tl.float32)
    for column in tl.range(0, N):
        x_value = tl.load(x + column)
        a_offsets = linear_offset(rows, column, A_ROWS, MODE_BITS)
        b_offsets = linear_offset(rows, column, B_ROWS, MODE_BITS)
        sum_a += tl.load(a + a_offsets) * x_value
        sum_b += tl.load(b + b_offsets) * x_value
    tl.store(output + rows, ALPHA * sum_a + BETA * sum_b)


@triton.jit
def gemm_prepacked_b_kernel(
    a,
    b,
    c,
    B_ROWS: tl.constexpr,
    MODE_BITS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    rows = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_base in tl.range(0, N, BLOCK_K):
        k = k_base + k_offsets
        a_tile = tl.load(a + rows[:, None] * N + k[None, :])
        b_offsets = linear_offset(
            k[:, None], columns[None, :], B_ROWS, MODE_BITS
        )
        b_tile = tl.load(b + b_offsets)
        accumulator = tl.dot(a_tile, b_tile, accumulator)
    c_offsets = rows[:, None] * N + columns[None, :]
    tl.store(c + c_offsets, accumulator)


def execution_layout_from_compiled(compiled, shape, names):
    from relay import TritonLinearLayout, extract_blocked_layout

    blocked = extract_blocked_layout(compiled.asm["ttgir"])
    extracted = TritonLinearLayout.from_blocked(
        shape,
        size_per_thread=blocked.size_per_thread,
        threads_per_warp=blocked.threads_per_warp,
        warps_per_cta=blocked.warps_per_cta,
        order=blocked.order,
        output_dim_names=names,
    )
    native = NativeLinearLayout.from_bases(
        extracted.bases,
        [name for name, _ in extracted.out_dims],
        [size for _, size in extracted.out_dims],
    )
    return blocked, TritonLinearLayout.from_triton(native)


def execution_layout_record(blocked, execution) -> dict[str, object]:
    return {
        "blocked": {name: list(values) for name, values in blocked.as_dict().items()},
        "input_sizes": {
            name: execution.input_size(name) for name in execution.input_dims
        },
        "output_shape": list(execution.output_shape),
        "bases": {
            name: [list(basis) for basis in bases]
            for name, bases in execution.bases
        },
    }


def issue_events(
    execution,
    matrix,
    *,
    prefix: str,
    site: str,
    weight: float,
    coordinate_map,
):
    from relay import induce_memory_event

    events = []
    register_count = execution.input_size("register")
    warp_count = execution.input_size("warp")
    for warp in range(warp_count):
        for register in range(register_count):
            locations = execution.locations(
                fixed={"register": register, "warp": warp, "block": 0}
            )
            events.append(
                induce_memory_event(
                    execution,
                    matrix,
                    locations,
                    id=f"{prefix}.w{warp}.r{register}",
                    site=site,
                    weight=weight,
                    coordinate_map=coordinate_map,
                )
            )
    return tuple(events)


def solve_layouts(matrices, events, args, name):
    from relay import (
        ScorePolicy,
        SolverConfig,
        execution_conditioned_quotient_problem,
        solve,
    )

    objective = f"{name}.issue.{args.transaction_bytes}B"
    targets = tuple(matrix for matrix in matrices if matrix.target)
    problem = execution_conditioned_quotient_problem(
        matrices,
        events,
        transaction_bytes=args.transaction_bytes,
        objective_name=objective,
        config=SolverConfig(
            policy=ScorePolicy("lexicographic", (objective, "runs", "xors")),
            tile_shapes={matrix.name: (matrix.shape,) for matrix in targets},
            general_tile_shapes={matrix.name: () for matrix in targets},
            include_global_canonical=False,
            enable_linear_inner=False,
            canonical_candidates_per_tile=args.candidates,
            primary_tolerance=0.0,
            per_array_candidates=max(args.candidates, 4),
            joint_candidates=max(args.candidates, 4),
        ),
        name=name,
    )
    return objective, problem, solve(problem)


def array_result_record(matrix, result, component, objective):
    from relay import row_major_layout, weighted_component_region_count

    default = row_major_layout(matrix)
    default_rows = layout_rows(default, matrix)
    default_score = weighted_component_region_count(matrix, default, component)
    candidates = []
    for rank, candidate in enumerate(result.arrays[matrix.name].candidates, start=1):
        layout = candidate.layout
        candidates.append(
            {
                "solver_rank": rank,
                "layout": layout.name,
                "word": layout.word_string(matrix),
                "a_rows": list(layout_rows(layout, matrix)),
                "quotient_score": float(candidate.scores[objective]),
                "packing_bound": float(candidate.packing_bounds[objective]),
                "runs": layout.runs,
                "xor_count": layout.xor_count,
                "exact": candidate.exact,
            }
        )
    best = candidates[0]
    return {
        "default": {
            "layout": default.name,
            "word": default.word_string(matrix),
            "a_rows": list(default_rows),
            "quotient_score": default_score,
        },
        "selected": best,
        "predicted_transaction_reduction": 1.0
        - float(best["quotient_score"]) / default_score,
        "retained_candidates": candidates,
        "solver": {
            "realized_candidate_count": result.arrays[matrix.name].all_candidate_count,
            "retained_candidate_count": len(candidates),
        },
    }


def measure_variants(launches, validators, args):
    compiled = {label: launch() for label, launch in launches.items()}
    torch.cuda.synchronize()
    for label, validate in validators.items():
        validate()
    timings = benchmark_layouts(
        launches,
        samples=args.samples,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    return {
        label: {
            "timing": timings[label],
            "compiled_codegen": compiled_codegen_statistics(compiled[label]),
        }
        for label in launches
    }


def require_aligned(label: str, tensor: torch.Tensor, transaction_bytes: int):
    if tensor.data_ptr() % transaction_bytes:
        raise ValueError(f"{label} allocation is not transaction-aligned")


def require_power_of_two_multiple(label: str, value: int, multiple: int):
    if value < multiple or value % multiple:
        raise ValueError(f"{label} must be a multiple of {multiple}")
    if value & (value - 1):
        raise ValueError(f"{label} must be a power of two")


def contiguous_control(args):
    from relay import MatrixSpec, row_major_layout

    n = args.vector_size
    block = 256
    matrix = MatrixSpec("X", (n,), 4, ("i",))
    default = row_major_layout(matrix)
    default_rows = layout_rows(default, matrix)
    logical_x = ((torch.arange(n) * 7 % 101) - 50).to(torch.float32) / 101
    logical_y = ((torch.arange(n) * 11 % 103) - 51).to(torch.float32) / 103
    reference = logical_x + logical_y
    source = pack_tensor(logical_x, default_rows).to("cuda")
    require_aligned("vector default", source, args.transaction_bytes)
    y = logical_y.to("cuda")
    probe_output = torch.empty(n, dtype=torch.float32, device="cuda")
    compiled = vector_add_kernel[(n // block,)](
        source,
        y,
        probe_output,
        X_ROWS=default_rows,
        N=n,
        BLOCK=block,
        num_warps=4,
    )
    torch.cuda.synchronize()
    validate_output("contiguous probe", probe_output, reference)
    blocked, execution = execution_layout_from_compiled(compiled, (block,), ("i",))
    events = issue_events(
        execution,
        matrix,
        prefix="vector_add.X",
        site="vector_add.X.load",
        weight=n // block,
        coordinate_map=lambda coord: coord,
    )
    objective, _, result = solve_layouts((matrix,), events, args, "vector_add")
    component = result.components[0]
    array = array_result_record(matrix, result, component, objective)
    selected_rows = tuple(array["selected"]["a_rows"])
    same_mapping = selected_rows == default_rows
    selected_source = (
        source
        if same_mapping
        else pack_tensor(logical_x, selected_rows).to("cuda")
    )
    require_aligned("vector LAQS", selected_source, args.transaction_bytes)
    default_output = torch.empty_like(probe_output)
    outputs = {
        "default": default_output,
        "laqs": default_output if same_mapping else torch.empty_like(probe_output),
    }

    def make_launch(label, data, rows):
        def launch():
            return vector_add_kernel[(n // block,)](
                data,
                y,
                outputs[label],
                X_ROWS=rows,
                N=n,
                BLOCK=block,
                num_warps=4,
            )

        return launch

    launches = {
        "default": make_launch("default", source, default_rows),
        "laqs": make_launch("laqs", selected_source, selected_rows),
    }
    validators = {
        label: (lambda label=label: validate_output(label, outputs[label], reference))
        for label in launches
    }
    variants = measure_variants(launches, validators, args)
    default_time = float(variants["default"]["timing"]["median_ms"])
    selected_time = float(variants["laqs"]["timing"]["median_ms"])
    return {
        "regime": "negative_control_contiguous_vector_add",
        "matrix_shape": [n],
        "execution_layout": execution_layout_record(blocked, execution),
        "induced_event_count": len(events),
        "array": array,
        "selected_matches_default": same_mapping,
        "variants": variants,
        "measured_speedup": default_time / selected_time,
        "correct": True,
    }


def distributed_tile(args):
    from relay import MatrixSpec, row_major_layout

    n = args.tile_matrix_size
    matrix = MatrixSpec("A", (n, n), 4, ("i", "j"))
    default = row_major_layout(matrix)
    default_rows = layout_rows(default, matrix)
    i = torch.arange(n, dtype=torch.int64)[:, None]
    j = torch.arange(n, dtype=torch.int64)[None, :]
    logical = (((i * 17 + j * 13) % 101) - 50).to(torch.float32) / 101
    tiles = n // TILE_SIZE
    reference = (
        logical.reshape(tiles, TILE_SIZE, tiles, TILE_SIZE)
        .permute(0, 2, 1, 3)
        .sum(dim=(2, 3))
        .flatten()
    )
    default_source = pack_tensor(logical, default_rows).to("cuda")
    require_aligned("distributed default", default_source, args.transaction_bytes)
    output_size = tiles * tiles
    probe_output = torch.empty(output_size, dtype=torch.float32, device="cuda")
    compiled = tiled_sum_kernel[(output_size,)](
        default_source,
        probe_output,
        A_ROWS=default_rows,
        MODE_BITS=matrix.mode_bits[0],
        N=n,
        BLOCK_M=TILE_SIZE,
        BLOCK_N=TILE_SIZE,
        num_warps=4,
    )
    torch.cuda.synchronize()
    validate_output("distributed probe", probe_output, reference, atol=1e-2)
    blocked, execution = execution_layout_from_compiled(
        compiled, (TILE_SIZE, TILE_SIZE), ("i", "j")
    )
    events = issue_events(
        execution,
        matrix,
        prefix="tiled_sum.A",
        site="tiled_sum.A.load",
        weight=output_size,
        coordinate_map=lambda coord: coord,
    )
    objective, _, result = solve_layouts((matrix,), events, args, "distributed_tile")
    component = result.components[0]
    array = array_result_record(matrix, result, component, objective)
    selected_rows = tuple(array["selected"]["a_rows"])
    same_mapping = selected_rows == default_rows
    selected_source = (
        default_source
        if same_mapping
        else pack_tensor(logical, selected_rows).to("cuda")
    )
    require_aligned("distributed LAQS", selected_source, args.transaction_bytes)
    default_output = torch.empty_like(probe_output)
    outputs = {
        "default": default_output,
        "laqs": default_output if same_mapping else torch.empty_like(probe_output),
    }

    def make_launch(label, data, rows):
        def launch():
            return tiled_sum_kernel[(output_size,)](
                data,
                outputs[label],
                A_ROWS=rows,
                MODE_BITS=matrix.mode_bits[0],
                N=n,
                BLOCK_M=TILE_SIZE,
                BLOCK_N=TILE_SIZE,
                num_warps=4,
            )

        return launch

    launches = {
        "default": make_launch("default", default_source, default_rows),
        "laqs": make_launch("laqs", selected_source, selected_rows),
    }
    validators = {
        label: (
            lambda label=label: validate_output(
                label, outputs[label], reference, atol=1e-2
            )
        )
        for label in launches
    }
    variants = measure_variants(launches, validators, args)
    default_time = float(variants["default"]["timing"]["median_ms"])
    selected_time = float(variants["laqs"]["timing"]["median_ms"])
    return {
        "regime": "nontrivial_distributed_2d_tile",
        "matrix_shape": [n, n],
        "tile_shape": [TILE_SIZE, TILE_SIZE],
        "num_warps": 4,
        "execution_layout": execution_layout_record(blocked, execution),
        "induced_event_count": len(events),
        "array": array,
        "selected_matches_default": same_mapping,
        "variants": variants,
        "measured_speedup": default_time / selected_time,
        "correct": True,
    }


def gesummv(args):
    from relay import MatrixSpec, row_major_layout

    n = args.gesummv_size
    block = WAVE_SIZE
    alpha = 1.25
    beta = -0.75
    matrices = (
        MatrixSpec("A", (n, n), 4, ("i", "j")),
        MatrixSpec("B", (n, n), 4, ("i", "j")),
    )
    matrix_by_name = {matrix.name: matrix for matrix in matrices}
    i = torch.arange(n, dtype=torch.int64)[:, None]
    j = torch.arange(n, dtype=torch.int64)[None, :]
    logical_a = (((i * 17 + j * 13) % 101) - 50).to(torch.float32) / 101
    logical_b = (((i * 11 + j * 19) % 103) - 51).to(torch.float32) / 103
    x = (((torch.arange(n) * 7 % 37) - 18).to(torch.float32) / 37)
    reference = alpha * (logical_a @ x) + beta * (logical_b @ x)
    defaults = {name: row_major_layout(matrix) for name, matrix in matrix_by_name.items()}
    default_rows = {
        name: layout_rows(defaults[name], matrix) for name, matrix in matrix_by_name.items()
    }
    default_sources = {
        "A": pack_tensor(logical_a, default_rows["A"]).to("cuda"),
        "B": pack_tensor(logical_b, default_rows["B"]).to("cuda"),
    }
    for name, source in default_sources.items():
        require_aligned(f"GESUMMV {name} default", source, args.transaction_bytes)
    device_x = x.to("cuda")
    probe_output = torch.empty(n, dtype=torch.float32, device="cuda")
    compiled = gesummv_kernel[(n // block,)](
        default_sources["A"],
        default_sources["B"],
        device_x,
        probe_output,
        A_ROWS=default_rows["A"],
        B_ROWS=default_rows["B"],
        MODE_BITS=matrices[0].mode_bits[0],
        N=n,
        BLOCK=block,
        ALPHA=alpha,
        BETA=beta,
        num_warps=1,
    )
    torch.cuda.synchronize()
    validate_output("GESUMMV probe", probe_output, reference, rtol=2e-4, atol=2e-2)
    blocked, execution = execution_layout_from_compiled(compiled, (block,), ("i",))
    occurrence_weight = (n // block) * n
    events = []
    for matrix in matrices:
        events.extend(
            issue_events(
                execution,
                matrix,
                prefix=f"gesummv.{matrix.name}",
                site=f"gesummv.{matrix.name}.load",
                weight=occurrence_weight,
                coordinate_map=lambda coord: (coord[0], 0),
            )
        )
    objective, _, result = solve_layouts(matrices, events, args, "gesummv")
    component = result.components[0]
    arrays = {
        matrix.name: array_result_record(matrix, result, component, objective)
        for matrix in matrices
    }
    selected_rows = {
        name: tuple(arrays[name]["selected"]["a_rows"]) for name in arrays
    }
    selected_sources = {
        "A": pack_tensor(logical_a, selected_rows["A"]).to("cuda"),
        "B": pack_tensor(logical_b, selected_rows["B"]).to("cuda"),
    }
    for name, source in selected_sources.items():
        require_aligned(f"GESUMMV {name} LAQS", source, args.transaction_bytes)
    configurations = {
        "default": ("default", "default"),
        "laqs_a": ("laqs", "default"),
        "laqs_b": ("default", "laqs"),
        "laqs_both": ("laqs", "laqs"),
    }
    outputs = {
        label: torch.empty_like(probe_output) for label in configurations
    }

    def make_launch(label, a_choice, b_choice):
        a_rows = selected_rows["A"] if a_choice == "laqs" else default_rows["A"]
        b_rows = selected_rows["B"] if b_choice == "laqs" else default_rows["B"]
        a_source = selected_sources["A"] if a_choice == "laqs" else default_sources["A"]
        b_source = selected_sources["B"] if b_choice == "laqs" else default_sources["B"]

        def launch():
            return gesummv_kernel[(n // block,)](
                a_source,
                b_source,
                device_x,
                outputs[label],
                A_ROWS=a_rows,
                B_ROWS=b_rows,
                MODE_BITS=matrices[0].mode_bits[0],
                N=n,
                BLOCK=block,
                ALPHA=alpha,
                BETA=beta,
                num_warps=1,
            )

        return launch

    launches = {
        label: make_launch(label, *choices)
        for label, choices in configurations.items()
    }
    validators = {
        label: (
            lambda label=label: validate_output(
                label, outputs[label], reference, rtol=2e-4, atol=2e-2
            )
        )
        for label in launches
    }
    variants = measure_variants(launches, validators, args)
    default_time = float(variants["default"]["timing"]["median_ms"])
    for label, choices in configurations.items():
        variants[label]["layouts"] = {"A": choices[0], "B": choices[1]}
        variants[label]["speedup_over_default"] = default_time / float(
            variants[label]["timing"]["median_ms"]
        )
    return {
        "regime": "gesummv_independent_a_b",
        "matrix_shape": [n, n],
        "execution_layout": execution_layout_record(blocked, execution),
        "dynamic_occurrences_per_array_event": occurrence_weight,
        "induced_event_count": len(events),
        "arrays": arrays,
        "variants": variants,
        "correct": True,
    }


def gemm_prepacked_b(args):
    from relay import MatrixSpec, row_major_layout

    n = args.gemm_size
    matrix = MatrixSpec("B", (n, n), 2, ("k", "n"))
    default = row_major_layout(matrix)
    default_rows = layout_rows(default, matrix)
    i = torch.arange(n, dtype=torch.int64)[:, None]
    j = torch.arange(n, dtype=torch.int64)[None, :]
    logical_a = ((((i * 17 + j * 13) % 31) - 15).to(torch.float32) / 31).half()
    logical_b = ((((i * 11 + j * 19) % 29) - 14).to(torch.float32) / 29).half()
    reference = logical_a.float() @ logical_b.float()
    device_a = logical_a.flatten().to("cuda")
    default_b = pack_tensor(logical_b, default_rows).to("cuda")
    require_aligned("GEMM B default", default_b, args.transaction_bytes)
    probe_output = torch.empty((n, n), dtype=torch.float32, device="cuda")
    grid = (n // TILE_SIZE, n // TILE_SIZE)
    compiled = gemm_prepacked_b_kernel[grid](
        device_a,
        default_b,
        probe_output,
        B_ROWS=default_rows,
        MODE_BITS=matrix.mode_bits[0],
        N=n,
        BLOCK_M=TILE_SIZE,
        BLOCK_N=TILE_SIZE,
        BLOCK_K=TILE_SIZE,
        num_warps=4,
    )
    torch.cuda.synchronize()
    validate_output("GEMM probe", probe_output, reference, rtol=5e-3, atol=1e-1)
    blocked, execution = execution_layout_from_compiled(
        compiled, (TILE_SIZE, TILE_SIZE), ("k", "n")
    )
    occurrences = grid[0] * grid[1] * (n // TILE_SIZE)
    events = issue_events(
        execution,
        matrix,
        prefix="gemm.B",
        site="gemm.B.load",
        weight=occurrences,
        coordinate_map=lambda coord: coord,
    )
    objective, _, result = solve_layouts((matrix,), events, args, "gemm_prepacked_b")
    component = result.components[0]
    array = array_result_record(matrix, result, component, objective)
    selected_rows = tuple(array["selected"]["a_rows"])
    selected_b = pack_tensor(logical_b, selected_rows).to("cuda")
    require_aligned("GEMM B LAQS", selected_b, args.transaction_bytes)
    outputs = {
        label: torch.empty_like(probe_output) for label in ("default", "laqs")
    }

    def make_launch(label, data, rows):
        def launch():
            return gemm_prepacked_b_kernel[grid](
                device_a,
                data,
                outputs[label],
                B_ROWS=rows,
                MODE_BITS=matrix.mode_bits[0],
                N=n,
                BLOCK_M=TILE_SIZE,
                BLOCK_N=TILE_SIZE,
                BLOCK_K=TILE_SIZE,
                num_warps=4,
            )

        return launch

    launches = {
        "default": make_launch("default", default_b, default_rows),
        "laqs": make_launch("laqs", selected_b, selected_rows),
    }
    validators = {
        label: (
            lambda label=label: validate_output(
                label, outputs[label], reference, rtol=5e-3, atol=1e-1
            )
        )
        for label in launches
    }
    variants = measure_variants(launches, validators, args)
    default_time = float(variants["default"]["timing"]["median_ms"])
    selected_time = float(variants["laqs"]["timing"]["median_ms"])
    return {
        "regime": "gemm_fixed_config_prepacked_b",
        "matrix_shape": [n, n],
        "tile_shape": [TILE_SIZE, TILE_SIZE, TILE_SIZE],
        "num_warps": 4,
        "execution_layout": execution_layout_record(blocked, execution),
        "dynamic_occurrences_per_event": occurrences,
        "induced_event_count": len(events),
        "array": array,
        "variants": variants,
        "measured_speedup": default_time / selected_time,
        "correct": True,
    }


def run_suite(args):
    require_power_of_two_multiple("vector size", args.vector_size, 256)
    require_power_of_two_multiple(
        "tile matrix size", args.tile_matrix_size, TILE_SIZE
    )
    require_power_of_two_multiple("GESUMMV size", args.gesummv_size, WAVE_SIZE)
    require_power_of_two_multiple("GEMM size", args.gemm_size, TILE_SIZE)
    if not torch.cuda.is_available():
        raise RuntimeError("the Stage 1 suite requires a Flux GPU allocation")
    regimes = {}
    runners = (
        ("contiguous_control", contiguous_control),
        ("distributed_tile", distributed_tile),
        ("gesummv", gesummv),
        ("gemm_prepacked_b", gemm_prepacked_b),
    )
    for name, runner in runners:
        print(f"Stage 1 suite: {name}", file=sys.stderr, flush=True)
        regimes[name] = runner(args)
        torch.cuda.empty_cache()
    return {
        "stage": 1,
        "experiment": "triton_target_stage1_suite",
        "transaction_bytes": args.transaction_bytes,
        "timing_configuration": {
            "samples": args.samples,
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "process": {
            "pid": os.getpid(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "regimes": regimes,
        "correct": all(bool(regime["correct"]) for regime in regimes.values()),
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-size", type=positive_integer, default=1 << 20)
    parser.add_argument("--tile-matrix-size", type=positive_integer, default=1024)
    parser.add_argument("--gesummv-size", type=positive_integer, default=1024)
    parser.add_argument("--gemm-size", type=positive_integer, default=512)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--samples", type=positive_integer, default=21)
    parser.add_argument("--iterations", type=positive_integer, default=50)
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    args = parse_arguments()
    result = run_suite(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
