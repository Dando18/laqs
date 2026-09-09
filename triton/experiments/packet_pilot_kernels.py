"""Ordinary-address pilot kernels for post-coalescing packet rewriting.

Legacy element-wise layout kernels remain in stage1_kernels.py.
"""

import triton
import triton.language as tl


@triton.jit
def ordinary_offset(first, second, rows: tl.constexpr, first_bits: tl.constexpr):
    return first * (1 << (len(rows) - first_bits)) + second



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
        a_offsets = ordinary_offset(rows, column, A_ROWS, MODE_BITS)
        b_offsets = ordinary_offset(rows, column, B_ROWS, MODE_BITS)
        sum_a += tl.load(a + a_offsets) * x_value
        sum_b += tl.load(b + b_offsets) * x_value
    tl.store(output + rows, ALPHA * sum_a + BETA * sum_b)


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
    bias_offsets = ordinary_offset(row, columns, B_ROWS, ROW_BITS)
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
        offsets = ordinary_offset(row, dimensions, W_ROWS, ROW_BITS)
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
        offsets = ordinary_offset(rows, column, W_ROWS, ROW_BITS)
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
        row_offsets = ordinary_offset(rows, column, A_ROWS, ROW_BITS)
        column_offsets = ordinary_offset(column, rows, A_ROWS, ROW_BITS)
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
    center = tl.load(source + ordinary_offset(row, columns, A_ROWS, ROW_BITS))
    left_values = tl.load(source + ordinary_offset(row, left, A_ROWS, ROW_BITS))
    right_values = tl.load(source + ordinary_offset(row, right, A_ROWS, ROW_BITS))
    up_values = tl.load(source + ordinary_offset(up, columns, A_ROWS, ROW_BITS))
    down_values = tl.load(source + ordinary_offset(down, columns, A_ROWS, ROW_BITS))
    result = center + left_values + right_values + up_values + down_values
    tl.store(output + row * N + columns, result)
