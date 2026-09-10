"""Fixed-schedule diagnostic kernels; no hidden timed choice of storage."""
import triton
import triton.language as tl


@triton.jit
def storage_offset(i, j, N, STORAGE: tl.constexpr,
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


@triton.jit
def manual_sum_kernel(input_ptr, output_ptr, M, N,
                      BLOCK_SIZE_NON_REDUCE_DIM: tl.constexpr,
                      BLOCK_SIZE_REDUCE_DIM: tl.constexpr, dim: tl.constexpr,
                      TILE_I: tl.constexpr, TILE_J: tl.constexpr):
    # Same sum-then-buffer arithmetic as the TritonBench dim=1 kernel.
    tl.static_assert(dim == 1)
    rows = tl.program_id(0) * BLOCK_SIZE_NON_REDUCE_DIM + tl.arange(0, BLOCK_SIZE_NON_REDUCE_DIM)
    columns = tl.arange(0, BLOCK_SIZE_REDUCE_DIM)
    buffer = tl.zeros((1, BLOCK_SIZE_NON_REDUCE_DIM), tl.float32)
    ptrs = input_ptr + storage_offset(rows[:, None], columns[None, :], N, 2, TILE_I, TILE_J)
    for start in range(0, N, BLOCK_SIZE_REDUCE_DIM):
        mask = (rows[:, None] < M) & (start + columns[None, :] < N)
        values = tl.load(ptrs, mask=mask, other=mask)
        buffer += tl.sum(values, 1)
        if BLOCK_SIZE_REDUCE_DIM % TILE_J == 0:
            ptrs += BLOCK_SIZE_REDUCE_DIM * TILE_I
        else:
            ptrs = input_ptr + storage_offset(rows[:, None], (start + BLOCK_SIZE_REDUCE_DIM + columns)[None, :], N, 2, TILE_I, TILE_J)
    tl.store(output_ptr + rows, buffer.reshape((BLOCK_SIZE_NON_REDUCE_DIM,)), mask=rows < M)


@triton.jit
def manual_gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       GROUP_M: tl.constexpr, ACTIVATION: tl.constexpr,
                       ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr,
                       A_TILE_I: tl.constexpr, A_TILE_J: tl.constexpr,
                       B_TILE_I: tl.constexpr, B_TILE_J: tl.constexpr):
    # Source copy of the TritonBench grouped GEMM with explicit blocked pointers.
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(M >= 0)
        tl.assume(N >= 0)
        tl.assume(K >= 0)
        tl.assume(stride_am >= 0)
        tl.assume(stride_ak >= 0)
        tl.assume(stride_bn >= 0)
        tl.assume(stride_bk >= 0)
        tl.assume(stride_cm >= 0)
        tl.assume(stride_cn >= 0)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    columns = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + storage_offset(rows[:, None], ks[None, :], K, 2, A_TILE_I, A_TILE_J)
    b_ptrs = b_ptr + storage_offset(ks[:, None], columns[None, :], N, 2, B_TILE_I, B_TILE_J)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=ks[None, :] < K - k * BLOCK_K, other=0.)
        b = tl.load(b_ptrs, mask=ks[:, None] < K - k * BLOCK_K, other=0.)
        accumulator += tl.dot(a, b)
        if BLOCK_K % A_TILE_J == 0:
            a_ptrs += BLOCK_K * A_TILE_I
        else:
            a_ptrs = a_ptr + storage_offset(rows[:, None], ((k + 1) * BLOCK_K + ks)[None, :], K, 2, A_TILE_I, A_TILE_J)
        if BLOCK_K % B_TILE_I == 0:
            b_ptrs += BLOCK_K * N
        else:
            b_ptrs = b_ptr + storage_offset(((k + 1) * BLOCK_K + ks)[:, None], columns[None, :], N, 2, B_TILE_I, B_TILE_J)
    if ACTIVATION == 'leaky_relu':
        activated = accumulator + 1
        accumulator = tl.where(activated >= 0, activated, .01 * activated)
    c = accumulator.to(tl.float16)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(c_ptr + stride_cm * rows[:, None] + stride_cn * columns[None, :], c,
             mask=(rows[:, None] < M) & (columns[None, :] < N))
