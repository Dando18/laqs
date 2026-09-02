#!/usr/bin/env python3
"""Compile and analyze a small, representative Triton kernel corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
PINNED_TRITON_PYTHON = REPOSITORY / "triton" / "triton-lang" / "python"
PINNED_TRITON_COMMIT = "b3376d6459bfb14f2500c1c20b3948ad59649bf8"
TRITONBENCH_SOURCE = REPOSITORY / "triton" / "tritonbench"

# Import the editable Triton selected by this Python environment before adding
# the repository root, whose `triton/` directory would otherwise form a
# namespace package. On Matrix this must remain the CUDA-only pinned clone.
def remove_source_path(path: Path) -> None:
    resolved = path.resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.curdir).resolve() != resolved
    ]


remove_source_path(REPOSITORY)
remove_source_path(PINNED_TRITON_PYTHON)
import triton

for source_path in (str(TRITONBENCH_SOURCE), str(REPOSITORY)):
    while source_path in sys.path:
        sys.path.remove(source_path)
sys.path[:0] = (str(TRITONBENCH_SOURCE), str(REPOSITORY))

from relay import AnalysisOptions, analyze_launch, get_hardware_profile


def active_triton_checkout() -> tuple[Path, Path]:
    module_file = getattr(triton, "__file__", None)
    if module_file is None:
        raise RuntimeError("the selected Triton module is a namespace package")
    triton_source = Path(module_file).resolve()
    checkout = triton_source.parents[2]
    expected_package = (checkout / "python" / "triton").resolve()
    if not triton_source.is_relative_to(expected_package):
        raise RuntimeError(
            f"coverage requires an editable Triton checkout, got {triton_source}"
        )
    try:
        commit = subprocess.run(
            ("git", "-C", str(checkout), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot verify the Triton checkout containing {triton_source}"
        ) from error
    if commit != PINNED_TRITON_COMMIT:
        raise RuntimeError(
            f"coverage requires pinned Triton {PINNED_TRITON_COMMIT}, got "
            f"{commit} at {checkout}"
        )
    return triton_source, checkout


@dataclass(frozen=True)
class LaunchCase:
    kernel: Any
    grid: Any
    arguments: tuple[Any, ...]
    keywords: Mapping[str, Any]


def vector_add_case(torch: Any, dtype: Any, device: str) -> LaunchCase:
    import triton
    from tritonbench.operators.vector_add.kernels import triton_add_kernel

    size = 4097
    x = torch.rand(size, dtype=dtype, device=device)
    y = torch.rand_like(x)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    return LaunchCase(
        triton_add_kernel,
        grid,
        (x, y, output, size),
        {"BLOCK_SIZE": 256},
    )


def softmax_case(torch: Any, dtype: Any, device: str) -> LaunchCase:
    from tritonbench.operators.softmax.operator import Operator

    rows, columns = 8, 257
    source = torch.rand((rows, columns), dtype=dtype, device=device)
    output = torch.empty_like(source)
    return LaunchCase(
        Operator.softmax_kernel,
        (rows,),
        (output, source, source.stride(0), output.stride(0), columns),
        {"BLOCK_SIZE": 512, "num_warps": 4, "num_stages": 1},
    )


def tutorial_matmul_case(torch: Any, dtype: Any, device: str) -> LaunchCase:
    import triton
    from tritonbench.operators.gemm.triton_matmul import matmul_kernel

    m, n, k = 128, 128, 96
    a = torch.rand((m, k), dtype=dtype, device=device)
    b = torch.rand((k, n), dtype=dtype, device=device)
    c = torch.empty((m, n), dtype=dtype, device=device)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return LaunchCase(
        matmul_kernel,
        grid,
        (
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        ),
        {"ACTIVATION": "", "ENABLE_BUFFER_OPS_ASSUMES": True},
    )


def layer_norm_case(torch: Any, dtype: Any, device: str) -> LaunchCase:
    from tritonbench.operators.layer_norm.tutorial import _layer_norm_fwd_fused

    rows, columns = 8, 257
    x = torch.rand((rows, columns), dtype=dtype, device=device)
    y = torch.empty_like(x)
    weight = torch.rand(columns, dtype=dtype, device=device)
    bias = torch.rand(columns, dtype=dtype, device=device)
    mean = torch.empty(rows, dtype=torch.float32, device=device)
    rstd = torch.empty(rows, dtype=torch.float32, device=device)
    return LaunchCase(
        _layer_norm_fwd_fused,
        (rows,),
        (x, y, weight, bias, mean, rstd, x.stride(0), columns, 1e-5),
        {"BLOCK_SIZE": 512, "num_warps": 4},
    )


def _descriptor_copy_kernel(source, output):
    value = source.load([0, 0])
    output.store([0, 0], value)


_DESCRIPTOR_COPY = None


def descriptor_copy_case(torch: Any, dtype: Any, device: str) -> LaunchCase:
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    global _DESCRIPTOR_COPY
    if _DESCRIPTOR_COPY is None:
        _DESCRIPTOR_COPY = triton.jit(_descriptor_copy_kernel)

    source = torch.rand((16, 16), dtype=dtype, device=device)
    output = torch.empty_like(source)
    source_descriptor = TensorDescriptor.from_tensor(source, [16, 16])
    output_descriptor = TensorDescriptor.from_tensor(output, [16, 16])
    return LaunchCase(
        _DESCRIPTOR_COPY,
        (1,),
        (source_descriptor, output_descriptor),
        {},
    )


CASES: Mapping[str, Callable[[Any, Any, str], LaunchCase]] = {
    "vector_add": vector_add_case,
    "softmax": softmax_case,
    "tutorial_matmul": tutorial_matmul_case,
    "layer_norm": layer_norm_case,
    "descriptor_copy": descriptor_copy_case,
}


def allocation_record(allocation: Any) -> dict[str, Any]:
    return {
        "name": allocation.name,
        "shape": list(allocation.true_shape),
        "envelope_shape": list(allocation.envelope_shape),
        "strides": list(allocation.strides),
        "role": allocation.role,
        "alias_group": allocation.alias_group,
        "dense_status": allocation.dense_status,
        "eligible": allocation.eligible,
    }


def analyze_case(launch: LaunchCase, options: AnalysisOptions) -> dict[str, Any]:
    try:
        analysis = analyze_launch(
            launch.kernel,
            launch.grid,
            *launch.arguments,
            _laqs_options=options,
            **launch.keywords,
        )
    except Exception as error:
        return {
            "status": "error",
            "category": "launch_error",
            "exception": type(error).__name__,
            "message": str(error),
        }
    if not analysis.supported:
        reason = analysis.unsupported
        return {
            "status": "unsupported",
            "category": reason.category,
            "message": reason.message,
            "site": reason.site,
        }
    return {
        "status": "supported",
        "grid": list(analysis.grid),
        "selected_config": dict(analysis.selected_config),
        "allocations": [allocation_record(item) for item in analysis.allocations],
        "matrix_count": len(analysis.matrices),
        "memory_event_count": len(analysis.events),
        "trace_class_count": len(analysis.sequences),
        "trace_class_multiplicity": sum(
            sequence.multiplicity for sequence in analysis.sequences
        ),
        "edge_family_count": len(analysis.edge_families),
        "objective_component_count": len(analysis.components),
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--hardware-profile", choices=("none", "mi300a"), default="none")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--require-supported",
        action="store_true",
        help="exit unsuccessfully if any requested case is unsupported or errors",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    import torch

    triton_source, triton_checkout = active_triton_checkout()

    dtype = getattr(torch, args.dtype)
    profile = (
        None
        if args.hardware_profile == "none"
        else get_hardware_profile(args.hardware_profile)
    )
    options = AnalysisOptions(
        plugin_path=args.plugin_path,
        hardware_profile=profile,
    )
    results = {}
    for index, name in enumerate(args.cases, start=1):
        print(
            f"automatic frontend coverage: {index}/{len(args.cases)} {name}",
            file=sys.stderr,
            flush=True,
        )
        try:
            launch = CASES[name](torch, dtype, args.device)
        except Exception as error:
            results[name] = {
                "status": "error",
                "category": "case_setup",
                "exception": type(error).__name__,
                "message": str(error),
            }
            continue
        results[name] = analyze_case(launch, options)

    supported = sum(result["status"] == "supported" for result in results.values())
    unsupported = sum(result["status"] == "unsupported" for result in results.values())
    errors = sum(result["status"] == "error" for result in results.values())
    payload = {
        "schema": "laqs.triton.coverage",
        "version": 1,
        "environment": {
            "hostname": platform.node(),
            "device": args.device,
            "device_name": torch.cuda.get_device_name(args.device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "triton_source": str(triton_source),
            "triton_checkout": str(triton_checkout),
            "triton_commit": PINNED_TRITON_COMMIT,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "hardware_profile": args.hardware_profile,
        },
        "requested_cases": list(args.cases),
        "summary": {
            "supported": supported,
            "unsupported": unsupported,
            "errors": errors,
        },
        "cases": results,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized + "\n", encoding="utf-8")
    return int(args.require_supported and supported != len(args.cases))


if __name__ == "__main__":
    raise SystemExit(main())
