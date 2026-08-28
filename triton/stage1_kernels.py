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


@triton.jit
def gemm_prepacked_b_general_kernel(
    a,
    b,
    c,
    B_ROWS: tl.constexpr,
    B_FIRST_BITS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    rows = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_base in tl.range(0, K, BLOCK_K):
        k = k_base + k_offsets
        if TRANS_A:
            a_offsets = k[None, :] * M + rows[:, None]
        else:
            a_offsets = rows[:, None] * K + k[None, :]
        a_tile = tl.load(a + a_offsets)
        if TRANS_B:
            b_offsets = linear_offset(
                columns[None, :], k[:, None], B_ROWS, B_FIRST_BITS
            )
        else:
            b_offsets = linear_offset(
                k[:, None], columns[None, :], B_ROWS, B_FIRST_BITS
            )
        b_tile = tl.load(b + b_offsets)
        accumulator = tl.dot(a_tile, b_tile, accumulator)
    c_offsets = rows[:, None] * N + columns[None, :]
    tl.store(c + c_offsets, accumulator)


@triton.jit
def cache_thrash_kernel(buffer, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(buffer + offsets, mask=offsets < N, other=0.0)
    tl.store(buffer + offsets, values + 1.0, mask=offsets < N)


@triton.jit
def bias_relu_kernel(
    source,
    bias,
    output,
    B_ROWS: tl.constexpr,
    N: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    features = offsets & (N - 1)
    bias_offsets = linear_offset_1d(features, B_ROWS)
    values = tl.load(source + offsets) + tl.load(bias + bias_offsets)
    tl.store(output + offsets, tl.maximum(values, 0.0), mask=offsets < ELEMENTS)


@triton.jit
def softmax_bias_kernel(
    source,
    bias,
    output,
    B_ROWS: tl.constexpr,
    ROW_BITS: tl.constexpr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, N)
    bias_offsets = linear_offset(row, columns, B_ROWS, ROW_BITS)
    values = tl.load(source + row * N + columns)
    values += tl.load(bias + bias_offsets)
    numerator = tl.exp(values - tl.max(values, axis=0))
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(output + row * N + columns, result)


@triton.jit
def embedding_bag_kernel(
    weight,
    indices,
    output,
    W_ROWS: tl.constexpr,
    ROW_BITS: tl.constexpr,
    D: tl.constexpr,
    BAG_SIZE: tl.constexpr,
):
    bag = tl.program_id(0)
    dimensions = tl.arange(0, D)
    accumulator = tl.zeros((D,), tl.float32)
    for slot in tl.static_range(BAG_SIZE):
        row = tl.load(indices + bag * BAG_SIZE + slot)
        offsets = linear_offset(row, dimensions, W_ROWS, ROW_BITS)
        accumulator += tl.load(weight + offsets)
    tl.store(output + bag * D + dimensions, accumulator)


@triton.jit
def gemv_kernel(
    weight,
    vector,
    output,
    W_ROWS: tl.constexpr,
    ROW_BITS: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for column in tl.range(0, K):
        offsets = linear_offset(rows, column, W_ROWS, ROW_BITS)
        accumulator += tl.load(weight + offsets) * tl.load(vector + column)
    tl.store(output + rows, accumulator, mask=rows < M)


@triton.jit
def mvt_kernel(
    matrix,
    x,
    y,
    output,
    A_ROWS: tl.constexpr,
    ROW_BITS: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for column in tl.range(0, N):
        row_offsets = linear_offset(rows, column, A_ROWS, ROW_BITS)
        column_offsets = linear_offset(column, rows, A_ROWS, ROW_BITS)
        accumulator += tl.load(matrix + row_offsets) * tl.load(x + column)
        accumulator += tl.load(matrix + column_offsets) * tl.load(y + column)
    tl.store(output + rows, accumulator)


@triton.jit
def stencil5_kernel(
    source,
    output,
    A_ROWS: tl.constexpr,
    ROW_BITS: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)
    blocks_per_row = N // BLOCK
    row = program // blocks_per_row
    columns = (program % blocks_per_row) * BLOCK + tl.arange(0, BLOCK)
    left = tl.maximum(columns - 1, 0)
    right = tl.minimum(columns + 1, N - 1)
    up = tl.maximum(row - 1, 0)
    down = tl.minimum(row + 1, M - 1)
    center = tl.load(source + linear_offset(row, columns, A_ROWS, ROW_BITS))
    left_values = tl.load(source + linear_offset(row, left, A_ROWS, ROW_BITS))
    right_values = tl.load(source + linear_offset(row, right, A_ROWS, ROW_BITS))
    up_values = tl.load(source + linear_offset(up, columns, A_ROWS, ROW_BITS))
    down_values = tl.load(source + linear_offset(down, columns, A_ROWS, ROW_BITS))
    result = center + left_values + right_values + up_values + down_values
    tl.store(output + row * N + columns, result)
