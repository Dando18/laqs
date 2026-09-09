"""Declared controls and a matrix used in two conflicting orientations."""
from tritonbench_cases import CASE_BY_ID, CaseDefinition, LaunchSpec


def row_column(size, config):
    import torch
    import triton
    from packet_kernels import row_column_kernel
    torch.manual_seed(0)
    a = torch.randn((size, size), device='cuda', dtype=torch.float32)
    x = torch.randn((size,), device='cuda')
    z = torch.randn_like(x)
    y, w = torch.empty_like(x), torch.empty_like(x)
    return LaunchSpec('row_column', config, f'Concurrent A@x and A.T@z, N={size}',
                      row_column_kernel, lambda meta: (triton.cdiv(size, meta['BLOCK_M']),), (a, x, z, y, w, size),
                      {'BLOCK_M': 4, 'BLOCK_K': 128, 'num_warps': 4})


CASES = dict(CASE_BY_ID)
for size, label in [(1024, 'small'), (4096, 'large')]:
    case = CaseDefinition('row_column', label, f'N={size}',
                          lambda size=size, label=label: row_column(size, label))
    CASES[case.case_id] = case
DEFAULT_CASES = ('fp8_gemm--small', 'gemm--asymmetric', 'sum--small',
                 'row_column--small', 'row_column--large', 'vector_add--large')


def reference(operator, launch):
    if operator != 'row_column':
        from reference_validation import operator_reference
        return operator_reference(operator, launch)
    import torch
    from reference_validation import comparison
    a, x, z, y, w = launch.values[:5]
    checks = []
    for observed, matrix, vector in [(y, a, x), (w, a.T, z)]:
        expected = matrix.double() @ vector.double()
        # Per-output absolute error budget accounts for reduction conditioning.
        bound = 8 * torch.finfo(torch.float32).eps * a.shape[0] ** 0.5 * (matrix.double().abs() @ vector.double().abs())
        error = (observed.double() - expected).abs()
        correct = bool(torch.isfinite(observed).all() and (error <= bound + 1e-6).all())
        checks.append({'correct': correct, 'max_abs_error': error.max().item()})
    if not all(c['correct'] for c in checks):
        raise ValueError(f'row/column FP64 reference failed: {checks}')
    return {'correct': True, 'reference': 'independent FP64 matrix-vector products', 'checks': checks}
