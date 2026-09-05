#!/usr/bin/env python3
"""One-GPU correctness smoke test for the LAQS layout-rewrite pass."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = (
    str(ROOT / "triton" / "triton-lang" / "python"),
    str(ROOT / "triton" / "experiments"),
    str(ROOT),
)

import torch
import triton
import triton.language as tl

from layout_runtime import RuntimeLayout, pack_tensor, rewrite_layouts
from relay import MatrixSpec, canonical_layout_from_word, layout_matrix_rows


@triton.jit
def copy_kernel(source, output, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    tl.store(output + offsets, tl.load(source + offsets, mask=mask), mask=mask)


def main():
    shape = (64, 32)
    logical = torch.arange(shape[0] * shape[1], device="cuda", dtype=torch.float32).reshape(shape)
    matrix = MatrixSpec("source", shape, 4, ("i", "j"), role="read")
    column_major = canonical_layout_from_word(matrix, "i" * 6 + "j" * 5)
    layout = RuntimeLayout(
        "source", 0, shape, tuple(logical.stride()), shape,
        layout_matrix_rows(matrix, column_major),
    )
    packed = pack_tensor(logical, layout)
    output = torch.empty_like(logical)
    block = 256
    with rewrite_layouts((layout,)):
        copy_kernel[(triton.cdiv(logical.numel(), block),)](
            packed, output, logical.numel(), BLOCK=block
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, logical)
    print("LAQS non-row-major layout rewrite: PASS")


if __name__ == "__main__":
    main()
