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


def check_layout(shape, rows, label):
    logical = torch.arange(
        shape[0] * shape[1], device="cuda", dtype=torch.float32
    ).reshape(shape)
    layout = RuntimeLayout(
        "source", 0, shape, tuple(logical.stride()),
        tuple(1 << (extent - 1).bit_length() for extent in shape), rows,
    )
    packed = pack_tensor(logical, layout)
    output = torch.empty_like(logical)
    block = 256
    with rewrite_layouts((layout,)):
        compiled = copy_kernel.warmup(
            packed,
            output,
            logical.numel(),
            BLOCK=block,
            grid=(triton.cdiv(logical.numel(), block),),
        )
        copy_kernel[(triton.cdiv(logical.numel(), block),)](
            packed, output, logical.numel(), BLOCK=block
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, logical)
    print(f"LAQS {label} layout rewrite: PASS")
    return compiled.asm["ttgir"].count("arith.xori")


def canonical_rows(shape):
    matrix = MatrixSpec("source", shape, 4, ("i", "j"), role="read")
    column_major = canonical_layout_from_word(
        matrix,
        "i" * matrix.mode_bits[0] + "j" * matrix.mode_bits[1],
    )
    return layout_matrix_rows(matrix, column_major)


def identity_rows(shape):
    width = sum((extent - 1).bit_length() for extent in shape)
    return tuple(1 << bit for bit in range(width))


def main():
    shape = (64, 32)
    rows = canonical_rows(shape)
    assert check_layout(shape, rows, "canonical") == 0

    mixed_rows = list(rows)
    mixed_rows[0] ^= mixed_rows[1]
    assert check_layout(shape, tuple(mixed_rows), "mixed linear-inner") == 1

    non_power_of_two = (63, 31)
    assert check_layout(
        non_power_of_two,
        identity_rows(non_power_of_two),
        "non-power-of-two canonical",
    ) == 0


if __name__ == "__main__":
    main()
