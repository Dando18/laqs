"""Triton kernels shared by the execution-conditioned Stage 1 experiments."""

from __future__ import annotations

import triton
import triton.language as tl


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
