#!/usr/bin/env python3
"""Benchmark one generalized prepacked-B GEMM Stage-1 regime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import triton

from stage1_common import (
    benchmark_layouts_isolated,
    distribution_version,
    execution_layout_from_compiled,
    execution_layout_record,
    issue_events,
    layout_rows,
    pack_tensor,
    positive_integer,
    require_aligned,
    require_power_of_two_multiple,
)
from stage1_kernels import (
    cache_thrash_kernel,
    gemm_prepacked_b_general_kernel,
)
from stage1_operand import rank_persistent_operand


def logical_inputs(m: int, n: int, k: int, trans_a: bool, trans_b: bool):
    a_rows = k if trans_a else m
    a_columns = m if trans_a else k
    b_rows = n if trans_b else k
    b_columns = k if trans_b else n
    ai = torch.arange(a_rows, dtype=torch.int64)[:, None]
    aj = torch.arange(a_columns, dtype=torch.int64)[None, :]
    bi = torch.arange(b_rows, dtype=torch.int64)[:, None]
    bj = torch.arange(b_columns, dtype=torch.int64)[None, :]
    logical_a = (
        (((ai * 17 + aj * 13) % 31) - 15).to(torch.float32) / 31
    ).half()
    logical_b = (
        (((bi * 11 + bj * 19) % 29) - 14).to(torch.float32) / 29
    ).half()
    return logical_a, logical_b


def make_launch(args, device_a, source, output, rows):
    grid = (args.m // args.block_m, args.n // args.block_n)
    b_first = args.n if args.trans_b else args.k

    def launch():
        return gemm_prepacked_b_general_kernel[grid](
            device_a,
            source,
            output,
            B_ROWS=rows,
            B_FIRST_BITS=b_first.bit_length() - 1,
            M=args.m,
            N=args.n,
            K=args.k,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            BLOCK_K=args.block_k,
            TRANS_A=args.trans_a,
            TRANS_B=args.trans_b,
            num_warps=args.num_warps,
        )

    return launch


def validate_gemm(label: str, output: torch.Tensor, reference: torch.Tensor):
    if torch.allclose(output, reference, rtol=5e-3, atol=1e-1):
        return
    error = torch.max(torch.abs(output - reference)).item()
    raise ValueError(f"{label} layout produced incorrect output: max error {error}")


def run_case(args) -> dict[str, object]:
    from relay import MatrixSpec, row_major_layout

    storage_shape = (args.n, args.k) if args.trans_b else (args.k, args.n)
    storage_names = ("n", "k") if args.trans_b else ("k", "n")
    matrix = MatrixSpec("B", storage_shape, 2, storage_names)
    default_layout = row_major_layout(matrix)
    default_rows = layout_rows(default_layout, matrix)
    logical_a, logical_b = logical_inputs(
        args.m, args.n, args.k, args.trans_a, args.trans_b
    )
    device_a = logical_a.flatten().to("cuda")
    math_a = device_a.reshape(logical_a.shape)
    if args.trans_a:
        math_a = math_a.T
    math_b = logical_b.to("cuda")
    if args.trans_b:
        math_b = math_b.T
    reference = math_a.float() @ math_b.float()
    default_source = pack_tensor(logical_b, default_rows).to("cuda")
    require_aligned("GEMM B default", default_source, args.transaction_bytes)
    probe_output = torch.empty(
        (args.m, args.n), dtype=torch.float32, device="cuda"
    )
    probe_launch = make_launch(
        args, device_a, default_source, probe_output, default_rows
    )
    compiled_probe = probe_launch()
    torch.cuda.synchronize()
    validate_gemm("GEMM probe", probe_output, reference)

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
    coordinate_map = (
        (lambda coord: (coord[1], coord[0]))
        if args.trans_b
        else (lambda coord: coord)
    )
    events = issue_events(
        execution,
        matrix,
        prefix="gemm.B",
        site="gemm.B.load",
        weight=occurrences,
        coordinate_map=coordinate_map,
    )

    benchmark = None
    if args.cache_mode == "thrashed":
        elements = (args.cache_thrash_bytes + 3) // 4
        cache_buffer = torch.empty(elements, dtype=torch.float32, device="cuda")
        grid = (triton.cdiv(elements, 256),)

        def before_each():
            cache_thrash_kernel[grid](cache_buffer, N=elements, BLOCK=256)

        def benchmark(launches):
            return benchmark_layouts_isolated(
                launches,
                before_each=before_each,
                samples=args.samples,
                iterations=args.iterations,
                warmup=args.warmup,
            )

    ranking = rank_persistent_operand(
        matrix,
        logical_b,
        default_source,
        events,
        args=args,
        problem_name="gemm_breadth",
        make_output=lambda: torch.empty_like(probe_output),
        make_launch=lambda source, output, rows: make_launch(
            args, device_a, source, output, rows
        ),
        validate=lambda label, output: validate_gemm(label, output, reference),
        benchmark=benchmark,
        inner_tile_shapes=(
            (
                (args.block_n, args.block_k)
                if args.trans_b
                else (args.block_k, args.block_n)
            ),
        ),
    )
    return {
        "stage": 1,
        "experiment": "triton_stage1_gemm_breadth_case",
        "configuration": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "block_k": args.block_k,
            "num_warps": args.num_warps,
            "trans_a": args.trans_a,
            "trans_b": args.trans_b,
            "cache_mode": args.cache_mode,
            "cache_thrash_bytes": (
                args.cache_thrash_bytes
                if args.cache_mode == "thrashed"
                else None
            ),
        },
        "execution_layout": execution_layout_record(blocked, execution),
        "dynamic_occurrences_per_event": occurrences,
        "induced_event_count": len(events),
        "ranking": ranking,
        "process": {
            "pid": os.getpid(),
            "torch_version": torch.__version__,
            "triton_version": distribution_version("triton"),
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "correct": bool(ranking["correct"]),
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_integer, default=512)
    parser.add_argument("--n", type=positive_integer, default=512)
    parser.add_argument("--k", type=positive_integer, default=512)
    parser.add_argument("--block-m", type=positive_integer, default=32)
    parser.add_argument("--block-n", type=positive_integer, default=32)
    parser.add_argument("--block-k", type=positive_integer, default=32)
    parser.add_argument("--num-warps", type=positive_integer, default=4)
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--cache-mode", choices=("warm", "thrashed"), default="warm")
    parser.add_argument(
        "--cache-thrash-bytes", type=positive_integer, default=256 << 20
    )
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--samples", type=positive_integer, default=9)
    parser.add_argument("--iterations", type=positive_integer, default=10)
    parser.add_argument("--warmup", type=positive_integer, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    args = parse_arguments()
    for label, value, block in (
        ("M", args.m, args.block_m),
        ("N", args.n, args.block_n),
        ("K", args.k, args.block_k),
    ):
        require_power_of_two_multiple(label, value, block)
    for label, value in (
        ("BLOCK_M", args.block_m),
        ("BLOCK_N", args.block_n),
        ("BLOCK_K", args.block_k),
    ):
        require_power_of_two_multiple(label, value, 16)
    if args.block_m != args.block_k:
        raise ValueError(
            "the current GEMM execution-layout bridge requires BLOCK_M == "
            "BLOCK_K; extracting dot-operand encodings is future work"
        )
    if args.num_warps not in (1, 2, 4, 8):
        raise ValueError("num warps must be one of 1, 2, 4, or 8")
    if not torch.cuda.is_available():
        raise RuntimeError("the GEMM breadth case requires a Flux GPU")
    result = run_case(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
