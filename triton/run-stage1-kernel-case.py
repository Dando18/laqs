#!/usr/bin/env python3
"""Run one realistic persistent-operand Stage-1 kernel case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import triton

from stage1_common import (
    execution_layout_from_compiled,
    execution_layout_record,
    issue_events,
    layout_rows,
    pack_tensor,
    positive_integer,
    require_aligned,
)
from stage1_kernels import (
    bias_relu_kernel,
    embedding_bag_kernel,
    gemv_kernel,
    gesummv_kernel,
    mvt_kernel,
    softmax_bias_kernel,
    stencil5_kernel,
)
from stage1_operand import rank_persistent_operand
from run_stage1_kernel_cases import CASES


INNER_TILE_FACTORS = (1, 2, 4, 8, 16, 32, 64)


def patterned(shape, first: int, second: int, modulus: int) -> torch.Tensor:
    i = torch.arange(shape[0], dtype=torch.int64)[:, None]
    j = torch.arange(shape[1], dtype=torch.int64)[None, :]
    return (((i * first + j * second) % modulus) - modulus // 2).float() / modulus


def validate_tensor(
    label: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-3,
):
    if torch.allclose(output, reference, rtol=rtol, atol=atol):
        return
    error = torch.max(torch.abs(output - reference)).item()
    raise ValueError(f"{label} layout produced incorrect output: max error {error}")


def result_record(case, matrix, blocked, execution, events, ranking, **metadata):
    return {
        "stage": 1,
        "experiment": "triton_stage1_kernel_breadth_case",
        "kernel": case,
        "target_operand": matrix.name,
        "operand_shape": list(matrix.shape),
        "execution_layout": execution_layout_record(blocked, execution),
        "induced_event_count": len(events),
        "ranking": ranking,
        **metadata,
        "process": {
            "pid": os.getpid(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "correct": bool(ranking["correct"]),
    }


def bias_relu(args):
    from relay import MatrixSpec, row_major_layout

    rows, n, block = 1024, 1024, 256
    elements = rows * n
    matrix = MatrixSpec("bias", (n,), 4, ("feature",))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    source = (torch.arange(elements).float() % 97 - 48) / 97
    logical_bias = (torch.arange(n).float() % 31 - 15) / 31
    reference = torch.maximum(
        source.reshape(rows, n) + logical_bias[None, :],
        torch.tensor(0.0),
    ).to("cuda")
    device_source = source.to("cuda")
    default_bias = pack_tensor(logical_bias, default_rows).to("cuda")
    require_aligned("bias_relu bias", default_bias, args.transaction_bytes)
    probe = torch.empty(elements, dtype=torch.float32, device="cuda")

    def make_launch(data, output, layout):
        return lambda: bias_relu_kernel[(elements // block,)](
            device_source,
            data,
            output,
            B_ROWS=layout,
            N=n,
            ELEMENTS=elements,
            BLOCK=block,
            num_warps=4,
        )

    compiled = make_launch(default_bias, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("bias_relu probe", probe.reshape(rows, n), reference)
    blocked, execution = execution_layout_from_compiled(
        compiled, (block,), ("feature",)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="bias_relu.bias",
        site="bias_relu.bias.load",
        weight=elements // block,
        coordinate_map=lambda coord: coord,
    )
    ranking = rank_persistent_operand(
        matrix,
        logical_bias,
        default_bias,
        events,
        args=args,
        problem_name="bias_relu",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(
            label, output.reshape(rows, n), reference
        ),
        inner_tile_shapes=((32,), (64,), (128,), (block,)),
    )
    return result_record(
        "bias_relu", matrix, blocked, execution, events, ranking, rows=rows
    )


def softmax_bias(args):
    from relay import MatrixSpec, row_major_layout

    m, n = 1024, 256
    matrix = MatrixSpec("bias", (m, n), 4, ("row", "feature"))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    logical_source = patterned((m, n), 17, 13, 101)
    logical_bias = patterned((m, n), 11, 19, 103)
    reference = torch.softmax(logical_source + logical_bias, dim=1).to("cuda")
    device_source = logical_source.flatten().to("cuda")
    default_bias = pack_tensor(logical_bias, default_rows).to("cuda")
    probe = torch.empty((m, n), dtype=torch.float32, device="cuda")

    def make_launch(data, output, layout):
        return lambda: softmax_bias_kernel[(m,)](
            device_source,
            data,
            output,
            B_ROWS=layout,
            ROW_BITS=m.bit_length() - 1,
            N=n,
            num_warps=4,
        )

    compiled = make_launch(default_bias, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("softmax_bias probe", probe, reference, rtol=2e-4)
    blocked, execution = execution_layout_from_compiled(
        compiled, (n,), ("feature",)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="softmax_bias.bias",
        site="softmax_bias.bias.load",
        weight=m,
        coordinate_map=lambda coord: (0, coord[0]),
    )
    ranking = rank_persistent_operand(
        matrix,
        logical_bias,
        default_bias,
        events,
        args=args,
        problem_name="softmax_bias",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(
            label, output, reference, rtol=2e-4
        ),
        inner_tile_shapes=tuple(
            (rows_per_tile, n)
            for rows_per_tile in INNER_TILE_FACTORS
        ),
    )
    return result_record(
        "softmax_bias", matrix, blocked, execution, events, ranking
    )


def embedding_bag(args):
    from relay import MatrixSpec, row_major_layout

    rows, dimensions, bags, bag_size = 4096, 128, 4096, 4
    matrix = MatrixSpec("weight", (rows, dimensions), 2, ("row", "dimension"))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    logical_weight = patterned((rows, dimensions), 17, 13, 101).half()
    bag = torch.arange(bags, dtype=torch.int64)[:, None]
    slot = torch.arange(bag_size, dtype=torch.int64)[None, :]
    indices = (bag * 17 + slot * 101) % rows
    reference = logical_weight.float()[indices].sum(dim=1).to("cuda")
    device_indices = indices.flatten().to("cuda")
    default_weight = pack_tensor(logical_weight, default_rows).to("cuda")
    probe = torch.empty(
        (bags, dimensions), dtype=torch.float32, device="cuda"
    )

    def make_launch(data, output, layout):
        return lambda: embedding_bag_kernel[(bags,)](
            data,
            device_indices,
            output,
            W_ROWS=layout,
            ROW_BITS=rows.bit_length() - 1,
            D=dimensions,
            BAG_SIZE=bag_size,
            num_warps=2,
        )

    compiled = make_launch(default_weight, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("embedding_bag probe", probe, reference, atol=2e-2)
    blocked, execution = execution_layout_from_compiled(
        compiled, (dimensions,), ("dimension",)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="embedding_bag.weight",
        site="embedding_bag.weight.load",
        weight=bags * bag_size,
        coordinate_map=lambda coord: (0, coord[0]),
    )
    ranking = rank_persistent_operand(
        matrix,
        logical_weight,
        default_weight,
        events,
        args=args,
        problem_name="embedding_bag",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(
            label, output, reference, atol=2e-2
        ),
        inner_tile_shapes=tuple(
            (rows_per_tile, dimensions)
            for rows_per_tile in INNER_TILE_FACTORS
        ),
    )
    return result_record(
        "embedding_bag",
        matrix,
        blocked,
        execution,
        events,
        ranking,
        bags=bags,
        bag_size=bag_size,
    )


def gemv(args):
    from relay import MatrixSpec, row_major_layout

    m, k, block = 1024, 1024, 64
    matrix = MatrixSpec("weight", (m, k), 4, ("row", "column"))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    logical_weight = patterned((m, k), 17, 13, 101)
    vector = (torch.arange(k).float() % 37 - 18) / 37
    reference = (logical_weight @ vector).to("cuda")
    device_vector = vector.to("cuda")
    default_weight = pack_tensor(logical_weight, default_rows).to("cuda")
    probe = torch.empty(m, dtype=torch.float32, device="cuda")

    def make_launch(data, output, layout):
        return lambda: gemv_kernel[(m // block,)](
            data,
            device_vector,
            output,
            W_ROWS=layout,
            ROW_BITS=m.bit_length() - 1,
            M=m,
            K=k,
            BLOCK=block,
            num_warps=1,
        )

    compiled = make_launch(default_weight, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("gemv probe", probe, reference, atol=2e-2)
    blocked, execution = execution_layout_from_compiled(
        compiled, (block,), ("row",)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="gemv.weight",
        site="gemv.weight.load",
        weight=(m // block) * k,
        coordinate_map=lambda coord: (coord[0], 0),
    )
    ranking = rank_persistent_operand(
        matrix,
        logical_weight,
        default_weight,
        events,
        args=args,
        problem_name="gemv",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(
            label, output, reference, atol=2e-2
        ),
        inner_tile_shapes=tuple(
            (block, columns_per_tile)
            for columns_per_tile in INNER_TILE_FACTORS
        ),
    )
    return result_record("gemv", matrix, blocked, execution, events, ranking)


def mvt(args):
    from relay import MatrixSpec, row_major_layout

    n, block = 1024, 64
    matrix = MatrixSpec("matrix", (n, n), 4, ("row", "column"))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    logical_matrix = patterned((n, n), 17, 13, 101)
    x = (torch.arange(n).float() % 37 - 18) / 37
    y = (torch.arange(n).float() % 41 - 20) / 41
    reference = (logical_matrix @ x + logical_matrix.T @ y).to("cuda")
    device_x = x.to("cuda")
    device_y = y.to("cuda")
    default_matrix = pack_tensor(logical_matrix, default_rows).to("cuda")
    probe = torch.empty(n, dtype=torch.float32, device="cuda")

    def make_launch(data, output, layout):
        return lambda: mvt_kernel[(n // block,)](
            data,
            device_x,
            device_y,
            output,
            A_ROWS=layout,
            ROW_BITS=n.bit_length() - 1,
            N=n,
            BLOCK=block,
            num_warps=1,
        )

    compiled = make_launch(default_matrix, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("mvt probe", probe, reference, atol=4e-2)
    blocked, execution = execution_layout_from_compiled(
        compiled, (block,), ("row",)
    )
    occurrence = (n // block) * n
    events = (
        *issue_events(
            execution,
            matrix,
            prefix="mvt.row",
            site="mvt.row.load",
            weight=occurrence,
            coordinate_map=lambda coord: (coord[0], 0),
        ),
        *issue_events(
            execution,
            matrix,
            prefix="mvt.column",
            site="mvt.column.load",
            weight=occurrence,
            coordinate_map=lambda coord: (0, coord[0]),
        ),
    )
    ranking = rank_persistent_operand(
        matrix,
        logical_matrix,
        default_matrix,
        events,
        args=args,
        problem_name="mvt",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(
            label, output, reference, atol=4e-2
        ),
        inner_tile_shapes=tuple(
            dict.fromkeys(
                [(block, factor) for factor in INNER_TILE_FACTORS]
                + [(factor, block) for factor in INNER_TILE_FACTORS]
            )
        ),
    )
    return result_record("mvt", matrix, blocked, execution, events, ranking)


def gesummv(args):
    from relay import MatrixSpec, row_major_layout

    n, block = 1024, 64
    alpha, beta = 1.25, -0.75
    matrix = MatrixSpec("A", (n, n), 4, ("row", "column"))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    logical_a = patterned((n, n), 17, 13, 101)
    logical_b = patterned((n, n), 11, 19, 103)
    vector = (torch.arange(n).float() % 37 - 18) / 37
    reference = (alpha * (logical_a @ vector) + beta * (logical_b @ vector)).to(
        "cuda"
    )
    device_x = vector.to("cuda")
    device_b = logical_b.flatten().to("cuda")
    default_a = pack_tensor(logical_a, default_rows).to("cuda")
    probe = torch.empty(n, dtype=torch.float32, device="cuda")

    def make_launch(data, output, layout):
        return lambda: gesummv_kernel[(n // block,)](
            data,
            device_b,
            device_x,
            output,
            A_ROWS=layout,
            B_ROWS=default_rows,
            MODE_BITS=n.bit_length() - 1,
            N=n,
            BLOCK=block,
            ALPHA=alpha,
            BETA=beta,
            num_warps=1,
        )

    compiled = make_launch(default_a, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("gesummv probe", probe, reference, atol=4e-2)
    blocked, execution = execution_layout_from_compiled(
        compiled, (block,), ("row",)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="gesummv.A",
        site="gesummv.A.load",
        weight=(n // block) * n,
        coordinate_map=lambda coord: (coord[0], 0),
    )
    ranking = rank_persistent_operand(
        matrix,
        logical_a,
        default_a,
        events,
        args=args,
        problem_name="gesummv",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(
            label, output, reference, atol=4e-2
        ),
        inner_tile_shapes=tuple(
            (block, columns_per_tile)
            for columns_per_tile in INNER_TILE_FACTORS
        ),
    )
    return result_record(
        "gesummv", matrix, blocked, execution, events, ranking
    )


def stencil5(args):
    from relay import MatrixSpec, row_major_layout

    m, n, block = 512, 512, 64
    matrix = MatrixSpec("field", (m, n), 4, ("row", "column"))
    default_rows = layout_rows(row_major_layout(matrix), matrix)
    logical = patterned((m, n), 17, 13, 101)
    left = torch.cat((logical[:, :1], logical[:, :-1]), dim=1)
    right = torch.cat((logical[:, 1:], logical[:, -1:]), dim=1)
    up = torch.cat((logical[:1, :], logical[:-1, :]), dim=0)
    down = torch.cat((logical[1:, :], logical[-1:, :]), dim=0)
    reference = (logical + left + right + up + down).to("cuda")
    default_field = pack_tensor(logical, default_rows).to("cuda")
    probe = torch.empty((m, n), dtype=torch.float32, device="cuda")
    grid = (m * n // block,)

    def make_launch(data, output, layout):
        return lambda: stencil5_kernel[grid](
            data,
            output,
            A_ROWS=layout,
            ROW_BITS=m.bit_length() - 1,
            M=m,
            N=n,
            BLOCK=block,
            num_warps=1,
        )

    compiled = make_launch(default_field, probe, default_rows)()
    torch.cuda.synchronize()
    validate_tensor("stencil5 probe", probe, reference)
    blocked, execution = execution_layout_from_compiled(
        compiled, (block,), ("column",)
    )
    occurrence = m * n // block
    events = tuple(
        event
        for site in ("center", "up", "down")
        for event in issue_events(
            execution,
            matrix,
            prefix=f"stencil5.{site}",
            site=f"stencil5.{site}.load",
            weight=occurrence,
            coordinate_map=lambda coord: (0, coord[0]),
        )
    )
    for block_index in range(n // block):
        left_base = block_index * block - 1
        right_base = block_index * block + 1
        events += issue_events(
            execution,
            matrix,
            prefix=f"stencil5.left.block{block_index}",
            site="stencil5.left.load",
            weight=m,
            coordinate_map=lambda coord, base=left_base: (
                0,
                max(base + coord[0], 0),
            ),
        )
        events += issue_events(
            execution,
            matrix,
            prefix=f"stencil5.right.block{block_index}",
            site="stencil5.right.load",
            weight=m,
            coordinate_map=lambda coord, base=right_base: (
                0,
                min(base + coord[0], n - 1),
            ),
        )
    ranking = rank_persistent_operand(
        matrix,
        logical,
        default_field,
        events,
        args=args,
        problem_name="stencil5",
        make_output=lambda: torch.empty_like(probe),
        make_launch=make_launch,
        validate=lambda label, output: validate_tensor(label, output, reference),
        inner_tile_shapes=tuple(
            (rows_per_tile, block)
            for rows_per_tile in INNER_TILE_FACTORS
        ),
    )
    return result_record(
        "stencil5", matrix, blocked, execution, events, ranking
    )


RUNNERS = {
    "bias_relu": bias_relu,
    "softmax_bias": softmax_bias,
    "embedding_bag": embedding_bag,
    "gemv": gemv,
    "mvt": mvt,
    "gesummv": gesummv,
    "stencil5": stencil5,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--samples", type=positive_integer, default=9)
    parser.add_argument("--iterations", type=positive_integer, default=20)
    parser.add_argument("--warmup", type=positive_integer, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    args = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("the kernel breadth case requires a Flux GPU")
    result = RUNNERS[args.case](args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
