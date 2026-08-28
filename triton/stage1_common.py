"""Shared packing, timing, and codegen utilities for Triton Stage 1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import statistics

import torch


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def layout_rows(layout, matrix) -> tuple[int, ...]:
    from relay import CanonicalLayout, LinearInnerLayout

    if isinstance(layout, CanonicalLayout):
        rows = layout.matrix_rows()
    elif isinstance(layout, LinearInnerLayout):
        rows = layout.a_rows
    else:
        raise TypeError(f"unsupported Stage 1 layout {type(layout).__name__}")
    if layout.tile_exponents != matrix.mode_bits:
        raise ValueError("the Stage 1 benchmark requires a full-matrix layout")
    return tuple(rows)


def physical_offsets(
    shape: tuple[int, ...], rows: tuple[int, ...]
) -> torch.Tensor:
    mode_bits = tuple(extent.bit_length() - 1 for extent in shape)
    logical = torch.zeros(shape, dtype=torch.int64)
    shift = 0
    for dimension, (extent, bits) in enumerate(zip(shape, mode_bits)):
        axis_shape = [1] * len(shape)
        axis_shape[dimension] = extent
        values = torch.arange(extent, dtype=torch.int64).reshape(axis_shape)
        logical |= values << shift
        shift += bits
    physical = torch.zeros_like(logical)
    for physical_bit, row in enumerate(rows):
        selected = logical & row
        selected ^= selected >> 32
        selected ^= selected >> 16
        selected ^= selected >> 8
        selected ^= selected >> 4
        selected ^= selected >> 2
        selected ^= selected >> 1
        physical |= (selected & 1) << physical_bit
    return physical


def pack_tensor(logical: torch.Tensor, rows: tuple[int, ...]) -> torch.Tensor:
    offsets = physical_offsets(tuple(logical.shape), rows)
    packed = torch.empty_like(logical).flatten()
    packed[offsets.flatten()] = logical.flatten()
    return packed


def validate_output(
    label: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-3,
) -> None:
    observed = output.cpu()
    if torch.allclose(observed, reference, rtol=rtol, atol=atol):
        return
    error = torch.max(torch.abs(observed - reference)).item()
    raise ValueError(f"{label} layout produced incorrect output: max error {error}")


def timing_summary(samples_ms: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "stdev_ms": (
            statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        ),
        "samples_ms": samples_ms,
    }


def benchmark_layouts(
    launches,
    *,
    samples: int,
    iterations: int,
    warmup: int,
) -> dict[str, dict[str, object]]:
    labels = tuple(launches)
    for _ in range(warmup):
        for label in labels:
            launches[label]()
    torch.cuda.synchronize()

    timings = {label: [] for label in labels}
    for sample in range(samples):
        rotation = (sample // 2) % len(labels)
        base = labels if sample % 2 == 0 else tuple(reversed(labels))
        order = base[rotation:] + base[:rotation]
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                launches[label]()
            stop.record()
            stop.synchronize()
            timings[label].append(float(start.elapsed_time(stop)) / iterations)
    return {label: timing_summary(values) for label, values in timings.items()}


def stable_id(prefix: str, value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def assembly_opcode_counts(assembly: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in assembly.splitlines():
        line = raw_line.split("//", 1)[0].split(";", 1)[0].strip()
        if not line or line.startswith((".", "#")) or line.endswith(":"):
            continue
        opcode = line.split(None, 1)[0]
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", opcode):
            counts[opcode] += 1
    return counts


def compiled_codegen_statistics(compiled) -> dict[str, object]:
    assembly_key = next(
        (
            key
            for key in ("amdgcn", "sass", "ptx")
            if isinstance(compiled.asm.get(key), str)
        ),
        None,
    )
    assembly = "" if assembly_key is None else compiled.asm[assembly_key]
    opcodes = assembly_opcode_counts(assembly)
    loads = sum(
        count
        for opcode, count in opcodes.items()
        if "load" in opcode or opcode.startswith(("global_atomic", "flat_atomic"))
    )
    stores = sum(count for opcode, count in opcodes.items() if "store" in opcode)
    xors = sum(count for opcode, count in opcodes.items() if "xor" in opcode)
    branches = sum(
        count
        for opcode, count in opcodes.items()
        if "branch" in opcode or "cbranch" in opcode
    )
    binary_key = next(
        (
            key
            for key, value in compiled.asm.items()
            if isinstance(value, bytes)
        ),
        None,
    )
    ir_statistics = {
        key: {
            "bytes": len(value.encode()),
            "lines": len(value.splitlines()),
        }
        for key, value in compiled.asm.items()
        if isinstance(value, str) and key != assembly_key
    }
    return {
        "n_regs": int(compiled.n_regs),
        "n_spills": int(compiled.n_spills),
        "n_max_threads": int(compiled.n_max_threads),
        "shared_bytes": int(compiled.metadata.shared),
        "binary_format": binary_key,
        "binary_bytes": (
            len(compiled.asm[binary_key]) if binary_key is not None else None
        ),
        "assembly_format": assembly_key,
        "assembly_bytes": len(assembly.encode()),
        "assembly_instruction_count": sum(opcodes.values()),
        "load_instruction_count": loads,
        "store_instruction_count": stores,
        "xor_instruction_count": xors,
        "branch_instruction_count": branches,
        "opcode_counts": dict(sorted(opcodes.items())),
        "ir": ir_statistics,
    }
