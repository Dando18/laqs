"""Fixed-schedule diagnostic kernels; no hidden timed choice of storage."""
import triton
import triton.language as tl


@triton.jit
def storage_offset(i, j, N: tl.constexpr, STORAGE: tl.constexpr,
                   TILE_I: tl.constexpr, TILE_J: tl.constexpr):
    if STORAGE == 1:
        return j * N + i
    elif STORAGE == 2:
        return (i // TILE_I) * N * TILE_I + (j // TILE_J) * TILE_I * TILE_J + (i % TILE_I) * TILE_J + j % TILE_J
    else:
        return i * N + j


@triton.jit
def row_column_kernel(A, X, Z, Y, W, N: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                      STORAGE: tl.constexpr = 0, TILE_I: tl.constexpr = 16,
                      TILE_J: tl.constexpr = 16):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_K)
    row_sum = tl.full((BLOCK_M, BLOCK_K), 0, tl.float32)
    col_sum = tl.full((BLOCK_M, BLOCK_K), 0, tl.float32)
    for start in range(0, N, BLOCK_K):
        k = start + columns
        horizontal = tl.load(A + storage_offset(rows[:, None], k[None, :], N, STORAGE, TILE_I, TILE_J))
        vertical = tl.load(A + storage_offset(k[None, :], rows[:, None], N, STORAGE, TILE_I, TILE_J))
        x = tl.load(X + k)
        z = tl.load(Z + k)
        row_sum += horizontal * x[None, :]
        col_sum += vertical * z[None, :]
    tl.store(Y + rows, tl.sum(row_sum, 1))
    tl.store(W + rows, tl.sum(col_sum, 1))
