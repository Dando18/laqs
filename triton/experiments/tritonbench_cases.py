"""Fixed broad-portable TritonBench launch panel for Experiments 4--6."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Mapping


@dataclass
class LaunchSpec:
    operator: str
    config: str
    description: str
    kernel: Any
    grid: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class CaseDefinition:
    operator: str
    config: str
    description: str
    factory: Callable[[], LaunchSpec]

    @property
    def case_id(self) -> str:
        return f"{self.operator}--{self.config}"


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _launch(operator, config, description, kernel, grid, args, **kwargs):
    return LaunchSpec(operator, config, description, kernel, grid, tuple(args), kwargs)


def _vector_add(size: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.vector_add.kernels import triton_add_kernel

    torch = _torch()
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand_like(x)
    output = torch.empty_like(x)
    block = 1024
    return _launch(
        "vector_add", config, f"{size} FP32 elements", triton_add_kernel,
        (triton.cdiv(size, block),), (x, y, output, size), BLOCK_SIZE=block,
    )


def _vector_exp(size: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.vector_exp.kernels import triton_exp_kernel

    torch = _torch()
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)
    block = 1024
    return _launch(
        "vector_exp", config, f"{size} FP32 elements", triton_exp_kernel,
        (triton.cdiv(size, block),), (x, output, size),
        BLOCK_SIZE=block, profile_mem=None,
    )


def _dropout(size: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.low_mem_dropout.kernels import _triton_dropout

    torch = _torch()
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    keep = torch.rand(size, device="cuda") > 0.5
    output = torch.empty_like(x)
    block = 1024
    return _launch(
        "low_mem_dropout", config, f"{size} FP32 elements", _triton_dropout,
        (triton.cdiv(size, block),), (x, keep, output, size, 0.5), BLOCK_SIZE=block,
    )


def _softmax(m: int, n: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.softmax.operator import Operator, _softmax_heuristic

    torch = _torch()
    x = torch.randn((m, n), device="cuda", dtype=torch.float16)
    output = torch.empty_like(x)
    block = triton.next_power_of_2(n)
    warps, stages = _softmax_heuristic(block, n)
    return _launch(
        "softmax", config, f"M={m}, N={n}, FP16", Operator.softmax_kernel,
        (m,), (output, x, x.stride(0), output.stride(0), n),
        BLOCK_SIZE=block, num_warps=warps, num_stages=stages,
    )


def _sum(m: int, n: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.sum.kernels import (
        triton_sum_kernel_1D_result_sum_then_buffer,
    )

    torch = _torch()
    x = torch.randn((m, n), device="cuda", dtype=torch.float32)
    output = torch.empty((m,), device="cuda", dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_NON_REDUCE_DIM"]),)
    return _launch(
        "sum", config, f"M={m}, N={n}, reduce dim 1, FP32",
        triton_sum_kernel_1D_result_sum_then_buffer, grid,
        (x, output, m, n), dim=1,
    )


def _layer_norm(m: int, n: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.layer_norm.tutorial import _layer_norm_fwd_fused

    torch = _torch()
    x = torch.randn((m, n), device="cuda", dtype=torch.float16)
    output = torch.empty_like(x)
    weight = torch.randn((n,), device="cuda", dtype=torch.float16)
    bias = torch.randn((n,), device="cuda", dtype=torch.float16)
    mean = torch.empty((m,), device="cuda", dtype=torch.float32)
    rstd = torch.empty_like(mean)
    block = triton.next_power_of_2(n)
    warps = min(max(block // 256, 1), 8)
    return _launch(
        "layer_norm", config, f"M={m}, N={n}, FP16", _layer_norm_fwd_fused,
        (m,), (x, output, weight, bias, mean, rstd, x.stride(0), n, 1e-5),
        BLOCK_SIZE=block, num_warps=warps,
    )


def _gemm(m: int, n: int, k: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.gemm.triton_matmul import matmul_kernel

    torch = _torch()
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return _launch(
        "gemm", config, f"M={m}, N={n}, K={k}, FP16", matmul_kernel, grid,
        (a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride()),
        ACTIVATION="", ENABLE_BUFFER_OPS_ASSUMES=True,
    )


def _bf16xint16(m: int, n: int, k: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.bf16xint16_gemm.kernel import bf16xint16_matmul_kernel

    torch = _torch()
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randint(-(2**15), 2**15 - 1, (k, n), device="cuda", dtype=torch.int16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"])
        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    return _launch(
        "bf16xint16_gemm", config, f"M={m}, N={n}, K={k}",
        bf16xint16_matmul_kernel, grid,
        (a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride()),
        TRANSPOSE=False,
    )


def _int4(batches: int, length: int, n: int, k: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.int4_gemm.kernel import matmul_kernel, pack_2xint4

    torch = _torch()
    m = batches * length
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    logical_b = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    packed_b = pack_2xint4(logical_b).contiguous()
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"])
        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    return _launch(
        "int4_gemm", config, f"B={batches}, L={length}, N={n}, K={k}",
        matmul_kernel, grid,
        (a, packed_b, c, m, n, k, *a.stride(), *packed_b.stride(), *c.stride()),
    )


def _fp8(m: int, n: int, k: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.fp8_gemm.tutorial import matmul_kernel

    torch = _torch()
    dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    a = torch.randn((m, k), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16).to(dtype)
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"])
        * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    return _launch(
        "fp8_gemm", config, f"M={m}, N={n}, K={k}, native FP8", matmul_kernel,
        grid, (a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride()),
        ACTIVATION="",
    )


def _gather_gemv(size: int, config: str) -> LaunchSpec:
    import triton
    from tritonbench.operators.gather_gemv.triton_gather_gemv import (
        triton_red_fused_mv_0,
    )

    torch = _torch()
    weight = torch.randint(-8, 8, (8, size, size), device="cuda", dtype=torch.int8)
    indices = torch.tensor((1, 6), device="cuda", dtype=torch.int64)
    vector = torch.randn((size,), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((2 * size,), device="cuda", dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(2 * size, meta["XBLOCK"]),)
    return _launch(
        "gather_gemv", config, f"S={size}", triton_red_fused_mv_0, grid,
        (indices, weight, vector, output, 2 * size, size),
    )


def _template_attention() -> LaunchSpec:
    import triton
    from tritonbench.operators.template_attention.triton_attention import (
        triton_tem_fused_no_exp2,
    )

    torch = _torch()
    shape = (16, 16, 4096, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    output = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(4096, meta["BLOCK_M"]), 16 * 16, 1)
    return _launch(
        "template_attention", "default", "B=16, H=16, S=4096, D=64",
        triton_tem_fused_no_exp2, grid, (q, k, v, output, 4096),
    )


def _jagged(operator: str, b: int, m: int, max_seqlen: int, sparsity: float,
            config: str) -> LaunchSpec:
    import triton

    torch = _torch()
    generator = random.Random(0)
    target = int(max_seqlen * (1.0 - sparsity))
    margin = int(max_seqlen * 0.3)
    lower = max(target - margin, 1)
    upper = min(target + margin, max_seqlen)
    lengths = torch.tensor(
        [generator.randint(lower, upper) for _ in range(b)], dtype=torch.int64
    )
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))).cuda()
    values = torch.randn((int(lengths.sum()), m), device="cuda", dtype=torch.float32)

    if operator == "jagged_sum":
        from tritonbench.operators.jagged_sum.kernels import (
            triton_jagged_sum_kernel_simple_fused_sum_then_buffer as kernel,
        )
        output = torch.empty((b, m), device="cuda", dtype=torch.float32)
    elif operator == "jagged_mean":
        from tritonbench.operators.jagged_mean.kernels import (
            triton_jagged_mean_kernel_simple_fused_sum_then_buffer as kernel,
        )
        output = torch.empty((b, m), device="cuda", dtype=torch.float32)
    elif operator == "jagged_softmax":
        from tritonbench.operators.jagged_softmax.kernels import (
            triton_jagged_softmax_kernel_simple_fused_buffer_then_sum as kernel,
        )
        output = torch.empty_like(values)
    else:
        raise ValueError(operator)
    grid = lambda meta: (b * triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
    return _launch(
        operator, config,
        f"B={b}, M={m}, max_seqlen={max_seqlen}, sparsity={sparsity}",
        kernel, grid, (values, offsets, output, m, max_seqlen),
    )


def _case(operator: str, config: str, description: str,
          factory: Callable[[], LaunchSpec]) -> CaseDefinition:
    return CaseDefinition(operator, config, description, factory)


CASES = (
    _case("vector_add", "small", "2^18", lambda: _vector_add(2**18, "small")),
    _case("vector_add", "large", "2^26", lambda: _vector_add(2**26, "large")),
    _case("vector_exp", "small", "2^18", lambda: _vector_exp(2**18, "small")),
    _case("vector_exp", "large", "2^26", lambda: _vector_exp(2**26, "large")),
    _case("low_mem_dropout", "small", "2^18", lambda: _dropout(2**18, "small")),
    _case("low_mem_dropout", "large", "2^26", lambda: _dropout(2**26, "large")),
    _case("softmax", "small", "4096x1024", lambda: _softmax(4096, 1024, "small")),
    _case("softmax", "large", "4096x16384", lambda: _softmax(4096, 16384, "large")),
    _case("sum", "small", "4096x1024", lambda: _sum(4096, 1024, "small")),
    _case("sum", "large", "1024x16384", lambda: _sum(1024, 16384, "large")),
    _case("layer_norm", "small", "4096x1024", lambda: _layer_norm(4096, 1024, "small")),
    _case("layer_norm", "large", "4096x16384", lambda: _layer_norm(4096, 16384, "large")),
    _case("gemm", "square", "2048^3", lambda: _gemm(2048, 2048, 2048, "square")),
    _case("gemm", "asymmetric", "512x4096x4096", lambda: _gemm(512, 4096, 4096, "asymmetric")),
    _case("bf16xint16_gemm", "projection", "4096x1280x8192", lambda: _bf16xint16(4096, 1280, 8192, "projection")),
    _case("bf16xint16_gemm", "output", "4096x8192x1024", lambda: _bf16xint16(4096, 8192, 1024, "output")),
    _case("int4_gemm", "decode", "1x1x1280x8192", lambda: _int4(1, 1, 1280, 8192, "decode")),
    _case("int4_gemm", "prefill", "1x4096x1280x8192", lambda: _int4(1, 4096, 1280, 8192, "prefill")),
    _case("fp8_gemm", "small", "1024^3", lambda: _fp8(1024, 1024, 1024, "small")),
    _case("fp8_gemm", "large", "4096^3", lambda: _fp8(4096, 4096, 4096, "large")),
    _case("gather_gemv", "small", "S=2048", lambda: _gather_gemv(2048, "small")),
    _case("gather_gemv", "large", "S=8192", lambda: _gather_gemv(8192, "large")),
    _case("template_attention", "default", "16x16x4096x64", _template_attention),
    _case("jagged_sum", "small", "128x256x256@0.5", lambda: _jagged("jagged_sum", 128, 256, 256, 0.5, "small")),
    _case("jagged_sum", "large", "512x128x1024@0.75", lambda: _jagged("jagged_sum", 512, 128, 1024, 0.75, "large")),
    _case("jagged_mean", "small", "128x256x256@0.5", lambda: _jagged("jagged_mean", 128, 256, 256, 0.5, "small")),
    _case("jagged_mean", "large", "512x128x1024@0.75", lambda: _jagged("jagged_mean", 512, 128, 1024, 0.75, "large")),
    _case("jagged_softmax", "small", "128x256x256@0.5", lambda: _jagged("jagged_softmax", 128, 256, 256, 0.5, "small")),
    _case("jagged_softmax", "large", "512x128x1024@0.75", lambda: _jagged("jagged_softmax", 512, 128, 1024, 0.75, "large")),
)

CASE_BY_ID: Mapping[str, CaseDefinition] = {case.case_id: case for case in CASES}
OPERATORS = tuple(dict.fromkeys(case.operator for case in CASES))


def selected_cases(operator: str | None = None, config: str | None = None):
    result = tuple(
        case for case in CASES
        if (operator is None or case.operator == operator)
        and (config is None or case.config == config)
    )
    if not result:
        raise ValueError(f"no TritonBench cases match operator={operator!r}, config={config!r}")
    return result
