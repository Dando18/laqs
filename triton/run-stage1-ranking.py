#!/usr/bin/env python3
"""Repeat Stage 1 candidate rankings across sizes and fresh processes."""

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


def aggregate_size(results: list[dict[str, object]]) -> dict[str, object]:
    from relay import summarize_rank_quality

    first = results[0]
    candidate_ids = [
        str(candidate["candidate_id"]) for candidate in first["candidates"]
    ]
    for result in results[1:]:
        observed = [
            str(candidate["candidate_id"]) for candidate in result["candidates"]
        ]
        if observed != candidate_ids:
            raise ValueError(
                "retained Stage-1 candidates changed between process launches"
            )

    aggregate_candidates = []
    for index, candidate_id in enumerate(candidate_ids):
        process_candidates = [result["candidates"][index] for result in results]
        process_medians = [
            float(candidate["timing"]["median_ms"])
            for candidate in process_candidates
        ]
        all_samples = [
            float(sample)
            for candidate in process_candidates
            for sample in candidate["timing"]["samples_ms"]
        ]
        codegen = [candidate["compiled_codegen"] for candidate in process_candidates]
        reference = process_candidates[0]
        aggregate = {
            key: value
            for key, value in reference.items()
            if key not in {"compiled_codegen", "runtime_ms", "timing"}
        }
        aggregate["runtime_ms"] = statistics.median(process_medians)
        aggregate["timing"] = {
            **timing_summary(all_samples),
            "aggregation_runtime_ms": statistics.median(process_medians),
            "aggregation_method": "median_of_process_medians",
            "process_medians_ms": process_medians,
            "process_launch_count": len(process_medians),
        }
        aggregate["compiled_codegen"] = codegen[0]
        aggregate["compiled_codegen_consistent"] = all(
            item == codegen[0] for item in codegen[1:]
        )
        if not aggregate["compiled_codegen_consistent"]:
            aggregate["compiled_codegen_by_process"] = codegen
        aggregate_candidates.append(aggregate)

    rank_quality = summarize_rank_quality(aggregate_candidates)
    default = next(
        candidate
        for candidate in aggregate_candidates
        if candidate["layout"] == "row_major"
    )
    laqs = aggregate_candidates[0]
    return {
        "matrix_shape": first["matrix_shape"],
        "process_launch_count": len(results),
        "process_ids": [result["process"]["pid"] for result in results],
        "candidates": aggregate_candidates,
        "rank_quality": rank_quality,
        "measured_speedup": float(default["runtime_ms"])
        / float(laqs["runtime_ms"]),
        "correct": all(bool(result["correct"]) for result in results),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    worker = Path(__file__).with_name("run-stage1.py")
    sys.path.insert(0, str(repository))
    launches = []
    grouped = {size: [] for size in args.matrix_sizes}

    with tempfile.TemporaryDirectory(prefix="relay-stage1-ranking-") as temporary:
        temporary_path = Path(temporary)
        for process_launch in range(1, args.process_launches + 1):
            for matrix_size in args.matrix_sizes:
                print(
                    f"Stage 1 ranking: process {process_launch}/"
                    f"{args.process_launches}, matrix {matrix_size}",
                    file=sys.stderr,
                    flush=True,
                )
                output = temporary_path / f"p{process_launch}-n{matrix_size}.json"
                command = [
                    sys.executable,
                    str(worker),
                    "--matrix-size",
                    str(matrix_size),
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
                grouped[matrix_size].append(result)

    return {
        "stage": 1,
        "experiment": "triton_stage1_candidate_ranking_sweep",
        "configuration": {
            "matrix_sizes": args.matrix_sizes,
            "process_launches": args.process_launches,
            "transaction_bytes": args.transaction_bytes,
            "requested_retained_candidates": args.candidates,
            "samples_per_candidate_per_process": args.samples,
            "iterations_per_timing_batch": args.iterations,
            "warmup_rounds": args.warmup,
            "runtime_aggregation": "median_of_process_medians",
        },
        "launches": launches,
        "by_matrix_size": {
            str(size): aggregate_size(grouped[size]) for size in args.matrix_sizes
        },
    }


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-sizes",
        type=positive_integer,
        nargs="+",
        default=[512, 1024, 2048],
    )
    parser.add_argument("--process-launches", type=positive_integer, default=3)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--samples", type=positive_integer, default=21)
    parser.add_argument("--iterations", type=positive_integer, default=50)
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress JSON on stdout (the --json file is still written)",
    )
    return parser.parse_args(argv)


def main() -> None:
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
