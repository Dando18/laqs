#!/usr/bin/env python3
"""Repeat the targeted Stage 1 suite in fresh processes and aggregate it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


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


def aggregate_timing(variants):
    process_medians = [float(item["timing"]["median_ms"]) for item in variants]
    samples = [
        float(sample)
        for item in variants
        for sample in item["timing"]["samples_ms"]
    ]
    return {
        "median_ms": statistics.median(process_medians),
        "aggregation_method": "median_of_process_medians",
        "process_medians_ms": process_medians,
        "process_launch_count": len(process_medians),
        "all_samples_median_ms": statistics.median(samples),
        "all_samples_mean_ms": statistics.fmean(samples),
        "all_samples_stdev_ms": statistics.pstdev(samples),
        "samples_ms": samples,
    }


def aggregate_regime(regimes):
    first = regimes[0]
    for regime in regimes[1:]:
        if regime["regime"] != first["regime"]:
            raise ValueError("suite regime order changed between processes")
    aggregate = {
        key: value
        for key, value in first.items()
        if key not in {"measured_speedup", "variants"}
    }
    labels = tuple(first["variants"])
    for regime in regimes[1:]:
        if tuple(regime["variants"]) != labels:
            raise ValueError("suite variants changed between processes")
    aggregate_variants = {}
    for label in labels:
        process_variants = [regime["variants"][label] for regime in regimes]
        codegen = [item["compiled_codegen"] for item in process_variants]
        variant = {
            key: value
            for key, value in process_variants[0].items()
            if key
            not in {
                "compiled_codegen",
                "speedup_over_default",
                "timing",
            }
        }
        variant["timing"] = aggregate_timing(process_variants)
        variant["compiled_codegen"] = codegen[0]
        variant["compiled_codegen_consistent"] = all(
            item == codegen[0] for item in codegen[1:]
        )
        if not variant["compiled_codegen_consistent"]:
            variant["compiled_codegen_by_process"] = codegen
        aggregate_variants[label] = variant
    default_time = float(aggregate_variants["default"]["timing"]["median_ms"])
    for label, variant in aggregate_variants.items():
        variant["speedup_over_default"] = default_time / float(
            variant["timing"]["median_ms"]
        )
    aggregate["variants"] = aggregate_variants
    if "laqs" in aggregate_variants:
        aggregate["measured_speedup"] = aggregate_variants["laqs"][
            "speedup_over_default"
        ]
    aggregate["correct"] = all(bool(regime["correct"]) for regime in regimes)
    return aggregate


def run_sweep(args):
    worker = Path(__file__).with_name("run-stage1-suite.py")
    launches = []
    with tempfile.TemporaryDirectory(prefix="relay-stage1-suite-") as temporary:
        temporary_path = Path(temporary)
        for process_launch in range(1, args.process_launches + 1):
            print(
                f"Stage 1 suite process {process_launch}/{args.process_launches}",
                file=sys.stderr,
                flush=True,
            )
            output = temporary_path / f"process-{process_launch}.json"
            command = [
                sys.executable,
                str(worker),
                "--vector-size",
                str(args.vector_size),
                "--tile-matrix-size",
                str(args.tile_matrix_size),
                "--gesummv-size",
                str(args.gesummv_size),
                "--gemm-size",
                str(args.gemm_size),
                "--transaction-bytes",
                str(args.transaction_bytes),
                "--candidates",
                str(args.candidates),
                "--samples",
                str(args.samples),
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--json",
                str(output),
                "--quiet",
            ]
            subprocess.run(command, check=True)
            result = json.loads(output.read_text(encoding="utf-8"))
            result["process_launch"] = process_launch
            launches.append(result)

    regime_names = tuple(launches[0]["regimes"])
    return {
        "stage": 1,
        "experiment": "triton_target_stage1_suite_sweep",
        "configuration": {
            "process_launches": args.process_launches,
            "vector_size": args.vector_size,
            "tile_matrix_size": args.tile_matrix_size,
            "gesummv_size": args.gesummv_size,
            "gemm_size": args.gemm_size,
            "transaction_bytes": args.transaction_bytes,
            "samples": args.samples,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "runtime_aggregation": "median_of_process_medians",
        },
        "launches": launches,
        "regimes": {
            name: aggregate_regime(
                [launch["regimes"][name] for launch in launches]
            )
            for name in regime_names
        },
        "correct": all(bool(launch["correct"]) for launch in launches),
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-launches", type=positive_integer, default=3)
    parser.add_argument("--vector-size", type=positive_integer, default=1 << 20)
    parser.add_argument("--tile-matrix-size", type=positive_integer, default=1024)
    parser.add_argument("--gesummv-size", type=positive_integer, default=1024)
    parser.add_argument("--gemm-size", type=positive_integer, default=512)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--samples", type=positive_integer, default=21)
    parser.add_argument("--iterations", type=positive_integer, default=50)
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    result = run_sweep(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
