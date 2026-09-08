"""Independent operator references and finite-output validation for the panel."""
from __future__ import annotations


def comparison(observed, expected, *, rtol=None, atol=None):
    import torch
    finite = bool(torch.isfinite(observed).all() and torch.isfinite(expected).all())
    if not observed.is_floating_point():
        close = torch.equal(observed, expected)
        return {"finite": finite, "allclose": close, "max_abs_error": 0 if close else None}
    epsilon = torch.finfo(observed.dtype).eps
    rtol = max(2e-5, 2 * epsilon) if rtol is None else rtol
    atol = max(2e-5, 2 * epsilon) if atol is None else atol
    error = (observed.float() - expected.float()).abs()
    return {"finite": finite,
            "allclose": finite and bool(torch.allclose(observed.float(), expected.float(), rtol=rtol, atol=atol)),
            "max_abs_error": float(error.max()), "rtol": rtol, "atol": atol}


def operator_reference(operator, launch):
    """Validate native Triton independently before accepting its transformed twin.

    Matrix products use FP32 arithmetic on the kernel's effective input values.
    Large attention is checked in fixed row chunks, retaining its prefix mask.
    All reference work is outside both timing and profiling.
    """
    import torch
    v = launch.values
    checks = []

    def check(index, expected, observed=None, *, reduction=False):
        observed = v[index] if observed is None else observed
        epsilon = torch.finfo(observed.dtype).eps
        scale = float(expected.float().square().mean().sqrt())
        # Absolute tolerance follows output scale for cancellation near zero.
        tolerance = max(3e-5, (2 * epsilon if reduction else epsilon) * scale)
        result = comparison(observed, expected, rtol=max(3e-5, 2 * epsilon), atol=tolerance)
        checks.append({"argument": index, **result})

    with torch.no_grad():
        if operator == "vector_add":
            check(2, v[0] + v[1])
        elif operator == "vector_exp":
            check(1, v[0].exp())
        elif operator == "low_mem_dropout":
            check(2, torch.where(v[1], v[0] / (1 - v[4]), 0))
        elif operator == "softmax":
            check(0, v[1].float().softmax(-1))
        elif operator == "sum":
            check(1, v[0].sum(-1), reduction=True)
        elif operator == "layer_norm":
            x = v[0].float()
            mean = x.mean(-1)
            variance = (x - mean[:, None]).square().mean(-1)
            rstd = torch.rsqrt(variance + v[8])
            check(1, (x - mean[:, None]) * rstd[:, None] * v[2].float() + v[3].float())
            check(4, mean)
            check(5, rstd)
        elif operator in {"gemm", "bf16xint16_gemm", "int4_gemm", "fp8_gemm"}:
            a, b = v[:2]
            if operator == "int4_gemm":
                packed = b.to(torch.int8)
                b = torch.stack(((packed << 4) >> 4, packed >> 4), dim=1).reshape(a.shape[1], -1)
            if operator == "bf16xint16_gemm":
                b = b.to(torch.bfloat16)
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                for start in range(0, a.shape[0], 128):
                    expected = a[start:start + 128].float() @ b.float()
                    check(2, expected, v[2][start:start + 128], reduction=True)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        elif operator == "gather_gemv":
            for i, index in enumerate(v[0].tolist()):
                expected = v[1][index].float() @ v[2].float()
                check(3, expected, v[3].reshape(2, -1)[i], reduction=True)
        elif operator.startswith("jagged_"):
            offsets = v[1].tolist()
            for i, (start, stop) in enumerate(zip(offsets, offsets[1:])):
                values = v[0][start:stop].float()
                if operator == "jagged_sum":
                    check(2, values.sum(0), v[2][i], reduction=True)
                elif operator == "jagged_mean":
                    check(2, values.mean(0), v[2][i], reduction=True)
                else:
                    check(2, values.softmax(0), v[2][start:stop])
        elif operator == "template_attention":
            # The native implementation permits every pair inside the first
            # 1025 tokens, then applies causal attention. No 1/sqrt(D) scale.
            q, k, value, output = v[:4]
            columns = torch.arange(k.shape[-2], device=k.device)
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                for batch in range(q.shape[0]):
                    for head in range(q.shape[1]):
                        for start in range(0, q.shape[-2], 128):
                            rows = torch.arange(start, min(start + 128, q.shape[-2]), device=q.device)
                            scores = q[batch, head, start:start + 128].float() @ k[batch, head].float().T
                            mask = ((rows[:, None] <= 1024) & (columns[None, :] <= 1024)) | (rows[:, None] >= columns[None, :])
                            probability = scores.masked_fill(~mask, -float("inf")).softmax(-1)
                            expected = probability @ value[batch, head].float()
                            check(3, expected, output[batch, head, start:start + 128], reduction=True)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        else:
            raise ValueError(f"no independent reference for {operator}")
    if not checks or not all(r["allclose"] for r in checks):
        raise ValueError(f"native operator reference failed: {[r for r in checks if not r['allclose']][:5]}")
    return {"correct": True, "reference": "independent PyTorch operator", "checks": checks}
