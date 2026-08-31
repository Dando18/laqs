#!/usr/bin/env python3
"""Compare LAQS quotient predictions with MI300A memory-request counters.

The exact canonical DP selects one layout for every test kernel.  The selected
layout and a full row-major control are correctness-checked, timed, and then
profiled in isolated processes.  Counter summaries are paired by kernel and
reported for both the first operation and the final steady-state operations.

Solving is CPU-only and can be checkpointed before entering a GPU allocation::

    .venv/bin/python experiments/locality_counters.py \
        --size 512 --prepare-only \
        --output results/locality_counters_mi300a.json

    flux run -n1 -g1 -t 5m -q pdebug \
        .venv/bin/python experiments/locality_counters.py \
        --size 512 --resume \
        --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
        --output results/locality_counters_mi300a.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from math import isfinite
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.layout_ranking import KERNEL_SPECS, KernelSpec, parse_evaluator_output
from relay import (
    HARDWARE_PROFILES,
    CanonicalLayout,
    ScorePolicy,
    UniversalScopeObjectives,
    get_hardware_profile,
    row_major_layout,
    score_layouts,
)
from relay.objectives import build_objectives
from relay.search import search_canonical


EXPERIMENT_NAME = "laqs-locality-hardware-counters"
COUNTER_FIELDS = (
    "TCP_TOTAL_CACHE_ACCESSES_sum",
    "TCP_TCC_READ_REQ_sum",
    "TCP_TCC_WRITE_REQ_sum",
    "TCC_REQ_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "FETCH_SIZE",
    "WRITE_SIZE",
)
SUMMARY_COUNT_FIELDS = (
    "l1_cache_line_accesses",
    "l1_to_l2_read_requests",
    "l1_to_l2_write_requests",
    "l1_to_l2_total_requests",
    "l2_tag_requests",
    "l2_hits",
    "l2_misses",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "duration_ns",
)
PRIMARY_HARDWARE_METRIC = "l1_to_l2_read_requests"
KERNEL_DISPATCHES = {
    "atax": ("atax_tmp_kernel", "atax_y_kernel"),
    "gemm": ("gemm_kernel",),
    "gesummv": ("gesummv_kernel",),
    "mvt": ("mvt_kernel",),
    "syrk": ("syrk_kernel",),
}


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        action="append",
        choices=tuple(KERNEL_SPECS),
        default=None,
        help="test kernel to include; repeat as needed (default: all five)",
    )
    parser.add_argument(
        "--size",
        type=positive_integer,
        default=512,
        metavar="N",
        help="square matrix size (default: %(default)s)",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(HARDWARE_PROFILES),
        default="mi300a",
        help="analytical hardware response (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=5,
        help="timing samples per generated binary (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_integer,
        default=4,
        help="operations per timing sample and steady counter population",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_integer,
        default=5,
        help="untimed operations before timing (default: %(default)s)",
    )
    parser.add_argument("--device", type=nonnegative_integer, default=0)
    parser.add_argument("--block-size", type=positive_integer, default=128)
    parser.add_argument("--block-x", type=positive_integer, default=32)
    parser.add_argument("--block-y", type=positive_integer, default=32)
    parser.add_argument("--compiler", default="hipcc")
    parser.add_argument("--arch", default=None)
    parser.add_argument(
        "--rocprof",
        type=Path,
        default=Path("/opt/rocm-7.0.2/bin/rocprof"),
        help="rocprof v1 executable (default: %(default)s)",
    )
    parser.add_argument(
        "--counter-config",
        type=Path,
        default=Path(__file__).with_name("rocprof-locality.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/locality_counters_mi300a.json"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="raw profiler CSV directory (default: OUTPUT stem plus _raw)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="run the DP and write a GPU-free checkpoint",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue missing measurements from an exact matching checkpoint",
    )
    parser.add_argument(
        "--max-layouts",
        type=positive_integer,
        default=None,
        metavar="COUNT",
        help="measure at most COUNT pending kernel/layout pairs",
    )
    return parser, parser.parse_args(argv)


def configuration(
    args: argparse.Namespace, kernel_names: Sequence[str]
) -> dict[str, object]:
    profile = get_hardware_profile(args.hardware_profile)
    return {
        "kernels": list(kernel_names),
        "matrix_size": args.size,
        "solver_grammar": "canonical",
        "solver_algorithm": "exact single-objective count-grid dynamic programming",
        "selection_rule": (
            "minimum Q_fine in G_C; codegen runs and then deterministic "
            "DP traversal break exact quotient ties"
        ),
        "hardware_profile": args.hardware_profile,
        "hardware_profile_id": profile.profile_id,
        "fine_component": profile.fine_component,
        "samples": args.samples,
        "iterations": args.iterations,
        "steady_state_operations": args.samples * args.iterations,
        "warmup": args.warmup,
        "device": args.device,
        "block_size": args.block_size,
        "block_x": args.block_x,
        "block_y": args.block_y,
        "compiler": args.compiler,
        "arch": args.arch,
        "rocprof": str(args.rocprof.expanduser().resolve()),
        "counter_config": str(args.counter_config.expanduser().resolve()),
        "counter_passes_per_layout": 3,
        "seed": args.seed,
    }


def _block(spec: KernelSpec, args: argparse.Namespace) -> int | tuple[int, int, int]:
    if spec.block_style == "2d":
        return (args.block_x, args.block_y, 1)
    return args.block_size


def _word(layout, matrix) -> str:
    if isinstance(layout, CanonicalLayout):
        return layout.word_string(matrix)
    return layout.evaluator_descriptor(matrix)


def _score_summary(score, fine_component: str) -> dict[str, object]:
    return {
        "quotient_score": score.component(fine_component).raw_region_count,
        "hardware_peak": score.hardware_peak,
        "hardware_area": score.hardware_area,
        "codegen_runs": score.codegen.runs,
        "codegen_xors": score.codegen.xors,
    }


def _reduction(baseline: float, selected: float) -> float | None:
    if baseline == 0.0:
        return None
    return 1.0 - selected / baseline


def solve_kernel(
    spec: KernelSpec, args: argparse.Namespace
) -> dict[str, object]:
    start = perf_counter()
    profile = get_hardware_profile(args.hardware_profile)
    block = _block(spec, args)
    problem_config = spec.problem.build_config(
        problem_size=args.size,
        block_size=block,
    )
    matrices_tuple = tuple(spec.problem.get_matrices(problem_config))
    matrices = {matrix.name: matrix for matrix in matrices_tuple}
    events, sequences = spec.problem.get_events_and_sequences(problem_config)
    components = tuple(
        build_objectives(
            (UniversalScopeObjectives(profile.byte_scales),),
            matrices,
            {event.id: event for event in events},
            sequences,
        )
    )
    fine_component = next(
        component
        for component in components
        if component.name == profile.fine_component
    )
    selected_layouts = {}
    array_searches = []
    for matrix in matrices_tuple:
        if not matrix.target:
            continue
        stats = []
        seeds = search_canonical(
            matrix,
            (fine_component,),
            matrix.mode_bits,
            (tuple(reversed(range(matrix.rank))),),
            ScorePolicy(
                kind="lexicographic",
                order=(profile.fine_component, "runs"),
                paths_per_state=1,
                frontier_limit=1,
            ),
            candidates_per_tile=1,
            stats_sink=stats,
        )
        if len(seeds) != 1 or not stats or not stats[0].exact:
            raise RuntimeError(
                f"canonical minimum-Q DP failed for {spec.name}/{matrix.name}"
            )
        selected_layouts[matrix.name] = seeds[0].layout
        array_searches.append(
            {
                "matrix": matrix.name,
                "search_stats": asdict(stats[0]),
                "selected_search_scores": dict(seeds[0].search_scores),
            }
        )
    baseline_layouts = {
        matrix.name: row_major_layout(matrix) for matrix in matrices_tuple
    }
    baseline_score = score_layouts(
        matrices,
        components,
        baseline_layouts,
        hardware_profile=profile,
    )
    selected_score = score_layouts(
        matrices,
        components,
        {**baseline_layouts, **selected_layouts},
        hardware_profile=profile,
    )
    selected_words = {
        name: _word(selected_layouts[name], matrices[name])
        for name in spec.evaluator_arrays
    }
    baseline_words = {
        name: _word(baseline_layouts[name], matrices[name])
        for name in spec.evaluator_arrays
    }
    quotient_components = []
    for component in components:
        baseline_component = baseline_score.component(component.name)
        selected_component = selected_score.component(component.name)
        quotient_components.append(
            {
                "name": component.name,
                "edge_family": component.edge_family,
                "region_bytes": component.region_bytes,
                "baseline_region_count": baseline_component.raw_region_count,
                "selected_region_count": selected_component.raw_region_count,
                "predicted_reduction": _reduction(
                    baseline_component.raw_region_count,
                    selected_component.raw_region_count,
                ),
            }
        )
    baseline_summary = _score_summary(baseline_score, profile.fine_component)
    selected_summary = _score_summary(selected_score, profile.fine_component)
    return {
        "kernel": spec.name,
        "display_name": spec.display_name,
        "matrix_size": args.size,
        "block": list(block) if isinstance(block, tuple) else block,
        "dispatches_per_operation": list(KERNEL_DISPATCHES[spec.name]),
        "solver": {
            "grammar": "canonical",
            "algorithm": "exact single-objective count-grid dynamic programming",
            "objective": profile.fine_component,
            "tie_breaker": "minimum codegen runs, then deterministic traversal",
            "exact": True,
            "elapsed_seconds": perf_counter() - start,
            "fine_minimum": selected_summary["quotient_score"],
            "array_searches": array_searches,
        },
        "fine_component": profile.fine_component,
        "baseline": {
            "role": "old row-major layout",
            "words": baseline_words,
            "model": baseline_summary,
            "measurement": None,
        },
        "selected": {
            "role": "new LAQS-selected layout",
            "words": selected_words,
            "model": selected_summary,
            "measurement": None,
        },
        "model_comparison": {
            "baseline_quotient_score": baseline_summary["quotient_score"],
            "selected_quotient_score": selected_summary["quotient_score"],
            "selected_to_baseline_ratio": (
                float(selected_summary["quotient_score"])
                / float(baseline_summary["quotient_score"])
            ),
            "predicted_reduction": _reduction(
                float(baseline_summary["quotient_score"]),
                float(selected_summary["quotient_score"]),
            ),
            "predicts_lower": (
                float(selected_summary["quotient_score"])
                < float(baseline_summary["quotient_score"])
            ),
        },
        "quotient_components": quotient_components,
        "hardware_comparison": None,
    }


def prepare_report(
    args: argparse.Namespace, kernel_names: Sequence[str]
) -> dict[str, object]:
    kernels = []
    for index, kernel_name in enumerate(kernel_names, 1):
        spec = KERNEL_SPECS[kernel_name]
        print(
            f"[{index}/{len(kernel_names)}] Solving {spec.display_name} "
            f"N={args.size} with canonical DP...",
            flush=True,
        )
        record = solve_kernel(spec, args)
        print(
            "  Q_fine: "
            f"{record['model_comparison']['baseline_quotient_score']:.0f} -> "
            f"{record['model_comparison']['selected_quotient_score']:.0f} "
            f"({100.0 * record['model_comparison']['predicted_reduction']:.1f}% lower)",
            flush=True,
        )
        kernels.append(record)
    return {
        "experiment": EXPERIMENT_NAME,
        "configuration": configuration(args, kernel_names),
        "counter_definitions": {
            "TCP_TOTAL_CACHE_ACCESSES_sum": (
                "L1/TCP cache-line tag accesses, including hits and misses"
            ),
            "TCP_TCC_READ_REQ_sum": "read requests from L1/TCP to L2/TCC",
            "TCP_TCC_WRITE_REQ_sum": "write requests from L1/TCP to L2/TCC",
            "TCC_REQ_sum": "requests processed at the L2/TCC tag blocks",
            "TCC_HIT_sum": "L2/TCC cache hits",
            "TCC_MISS_sum": "L2/TCC cache misses",
            "FETCH_SIZE": "KiB fetched from memory, including cache effects",
            "WRITE_SIZE": "KiB written to memory, including cache effects",
        },
        "measurement_scope": {
            "pairing": "same generated kernel, launch geometry, size, and data",
            "cold": (
                "first target operation after allocation and host-to-device setup; "
                "reported as a cold-start proxy"
            ),
            "steady_state": (
                "median of the final timed operations after correctness and warmup"
            ),
            "attribution": (
                "whole target operation; fixed vector and output traffic remains "
                "in both sides of each pair"
            ),
        },
        "complete": False,
        "measurement_order": [],
        "kernels": kernels,
        "summary": None,
    }


def write_report(report: Mapping[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def evaluator_command(
    spec: KernelSpec,
    words: Mapping[str, str],
    args: argparse.Namespace,
    build_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(spec.evaluator),
        *[words[name] for name in spec.evaluator_arrays],
        "--n",
        str(args.size),
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--device",
        str(args.device),
        "--build-dir",
        str(build_dir),
    ]
    if spec.block_style == "2d":
        command.extend(
            ("--block-x", str(args.block_x), "--block-y", str(args.block_y))
        )
    else:
        command.extend(("--block-size", str(args.block_size)))
    command.extend(("--compiler", args.compiler))
    if args.arch:
        command.extend(("--arch", args.arch))
    return command


def _profiler_environment(rocprof: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["BASH_FUNC_realpath%%"] = (
        f"() {{ echo {shlex.quote(str(rocprof))}; }}"
    )
    return environment


def _float(record: Mapping[str, str], field: str, path: Path) -> float:
    value = record.get(field, "")
    if not value:
        raise ValueError(f"{path} has an empty profiler field {field!r}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"{path} has invalid {field!r} value {value!r}"
        ) from error
    if not isfinite(parsed):
        raise ValueError(f"{path} has non-finite {field!r} value {value!r}")
    return parsed


def _operation(dispatches: Sequence[Mapping[str, str]], path: Path) -> dict[str, object]:
    counters = {
        field: sum(_float(dispatch, field, path) for dispatch in dispatches)
        for field in COUNTER_FIELDS
    }
    duration_ns = sum(
        _float(dispatch, "EndNs", path) - _float(dispatch, "BeginNs", path)
        for dispatch in dispatches
    )
    hits = counters["TCC_HIT_sum"]
    misses = counters["TCC_MISS_sum"]
    read_bytes = counters["FETCH_SIZE"] * 1024.0
    write_bytes = counters["WRITE_SIZE"] * 1024.0
    return {
        "dispatch_indices": [int(dispatch["Index"]) for dispatch in dispatches],
        "kernel_names": [dispatch["KernelName"] for dispatch in dispatches],
        "duration_ns": duration_ns,
        "l1_cache_line_accesses": counters["TCP_TOTAL_CACHE_ACCESSES_sum"],
        "l1_to_l2_read_requests": counters["TCP_TCC_READ_REQ_sum"],
        "l1_to_l2_write_requests": counters["TCP_TCC_WRITE_REQ_sum"],
        "l1_to_l2_total_requests": (
            counters["TCP_TCC_READ_REQ_sum"]
            + counters["TCP_TCC_WRITE_REQ_sum"]
        ),
        "l2_tag_requests": counters["TCC_REQ_sum"],
        "l2_hits": hits,
        "l2_misses": misses,
        "l2_hit_rate_percent": (
            100.0 * hits / (hits + misses) if hits + misses else None
        ),
        "hbm_read_bytes": read_bytes,
        "hbm_write_bytes": write_bytes,
        "counters": counters,
    }


def _group_operations(
    records: Sequence[Mapping[str, str]],
    dispatch_names: Sequence[str],
    path: Path,
) -> list[dict[str, object]]:
    width = len(dispatch_names)
    if len(records) % width:
        raise ValueError(
            f"{path} has {len(records)} target dispatches, not a multiple of {width}"
        )
    operations = []
    for start in range(0, len(records), width):
        group = records[start : start + width]
        observed = [record["KernelName"] for record in group]
        for expected, name in zip(dispatch_names, observed):
            if expected not in name:
                raise ValueError(
                    f"{path} dispatch order mismatch: expected {expected!r}, got {name!r}"
                )
        operations.append(_operation(group, path))
    return operations


def _median_summary(operations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not operations:
        raise ValueError("cannot summarize an empty operation population")
    summary = {
        field: statistics.median(float(operation[field]) for operation in operations)
        for field in SUMMARY_COUNT_FIELDS
    }
    hit_rates = [
        float(operation["l2_hit_rate_percent"])
        for operation in operations
        if operation["l2_hit_rate_percent"] is not None
    ]
    summary["l2_hit_rate_percent"] = (
        statistics.median(hit_rates) if hit_rates else None
    )
    summary["operation_count"] = len(operations)
    return summary


def parse_counter_csv(
    path: Path,
    *,
    dispatch_names: Sequence[str],
    steady_operations: int,
) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    required = {"Index", "KernelName", "BeginNs", "EndNs", *COUNTER_FIELDS}
    if not records:
        raise ValueError(f"{path} contains no profiler records")
    missing = sorted(required - set(records[0]))
    if missing:
        raise ValueError(f"{path} is missing profiler columns: {missing}")
    records = [
        record
        for record in records
        if any(name in record.get("KernelName", "") for name in dispatch_names)
    ]
    records.sort(key=lambda record: int(record["Index"]))
    needed = (steady_operations + 1) * len(dispatch_names)
    if len(records) < needed:
        raise ValueError(
            f"{path} contains {len(records)} target dispatches; expected at least {needed}"
        )
    cold = _group_operations(records[: len(dispatch_names)], dispatch_names, path)
    steady_records = records[-steady_operations * len(dispatch_names) :]
    steady = _group_operations(steady_records, dispatch_names, path)
    return {
        "target_dispatch_count": len(records),
        "cold_first_operation": _median_summary(cold),
        "steady_state": _median_summary(steady),
        "steady_operations": steady,
    }


def profile_binary(
    executable: Path,
    raw_csv: Path,
    dispatch_names: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.rocprof.expanduser().resolve()),
        "-i",
        str(args.counter_config.expanduser().resolve()),
        "-o",
        str(raw_csv),
        "--timestamp",
        "on",
        str(executable),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=_profiler_environment(args.rocprof.expanduser().resolve()),
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"profiler exited with {completed.returncode}: {detail}"
        )
    if not raw_csv.exists():
        raise RuntimeError(f"profiler did not create {raw_csv}")
    try:
        counters = parse_counter_csv(
            raw_csv,
            dispatch_names=dispatch_names,
            steady_operations=args.samples * args.iterations,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    counters["command"] = command
    counters["raw_csv"] = str(raw_csv)
    return counters


def measure_layout(
    spec: KernelSpec,
    label: str,
    words: Mapping[str, str],
    args: argparse.Namespace,
    build_root: Path,
    raw_dir: Path,
) -> dict[str, object]:
    build_dir = build_root / spec.name / label
    build_dir.mkdir(parents=True, exist_ok=True)
    command = evaluator_command(spec, words, args, build_dir)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{spec.name}/{label} evaluator exited with "
            f"{completed.returncode}: {detail}"
        )
    try:
        timing = parse_evaluator_output(completed.stdout)
    except ValueError as error:
        raise RuntimeError(f"{spec.name}/{label}: {error}") from error
    executable = build_dir / f"generated_{spec.name}"
    if not executable.exists():
        raise RuntimeError(f"evaluator did not create {executable}")
    counters = profile_binary(
        executable,
        raw_dir / f"{spec.name}_{label}.csv",
        KERNEL_DISPATCHES[spec.name],
        args,
    )
    return {
        "correctness": "PASS",
        "timing": asdict(timing),
        "counter_profile": counters,
        "evaluator_command": command,
        "evaluator_stdout": completed.stdout,
        "evaluator_stderr": completed.stderr,
    }


def compare_summaries(
    baseline: Mapping[str, object], selected: Mapping[str, object]
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for field in SUMMARY_COUNT_FIELDS:
        baseline_value = float(baseline[field])
        selected_value = float(selected[field])
        metrics[field] = {
            "baseline": baseline_value,
            "selected": selected_value,
            "selected_to_baseline_ratio": (
                selected_value / baseline_value if baseline_value else None
            ),
            "reduction": _reduction(baseline_value, selected_value),
            "fewer": selected_value < baseline_value,
        }
    baseline_hit_rate = baseline.get("l2_hit_rate_percent")
    selected_hit_rate = selected.get("l2_hit_rate_percent")
    metrics["l2_hit_rate_percent"] = {
        "baseline": baseline_hit_rate,
        "selected": selected_hit_rate,
        "percentage_point_change": (
            float(selected_hit_rate) - float(baseline_hit_rate)
            if baseline_hit_rate is not None and selected_hit_rate is not None
            else None
        ),
    }
    return metrics


def finalize_kernel(record: dict[str, object]) -> None:
    baseline = record["baseline"]
    selected = record["selected"]
    assert isinstance(baseline, dict) and isinstance(selected, dict)
    baseline_measurement = baseline.get("measurement")
    selected_measurement = selected.get("measurement")
    if not isinstance(baseline_measurement, dict) or not isinstance(
        selected_measurement, dict
    ):
        record["hardware_comparison"] = None
        return
    baseline_profile = baseline_measurement["counter_profile"]
    selected_profile = selected_measurement["counter_profile"]
    assert isinstance(baseline_profile, dict) and isinstance(selected_profile, dict)
    record["hardware_comparison"] = {
        "cold_first_operation": compare_summaries(
            baseline_profile["cold_first_operation"],
            selected_profile["cold_first_operation"],
        ),
        "steady_state": compare_summaries(
            baseline_profile["steady_state"],
            selected_profile["steady_state"],
        ),
        "primary_hypothesis": {
            "modeled_metric": record["fine_component"],
            "hardware_metric": PRIMARY_HARDWARE_METRIC,
            "predicted_reduction": record["model_comparison"][
                "predicted_reduction"
            ],
            "measured_steady_state_reduction": None,
            "measured_fewer_requests": None,
        },
    }
    primary = record["hardware_comparison"]["primary_hypothesis"]
    metric = record["hardware_comparison"]["steady_state"][
        PRIMARY_HARDWARE_METRIC
    ]
    primary["measured_steady_state_reduction"] = metric["reduction"]
    primary["measured_fewer_requests"] = metric["fewer"]


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            result[ordered[position]] = rank
        start = end
    return result


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    return statistics.correlation(left_ranks, right_ranks)


def finalize_report(report: dict[str, object]) -> None:
    kernels = report["kernels"]
    assert isinstance(kernels, list)
    for record in kernels:
        assert isinstance(record, dict)
        finalize_kernel(record)
    complete = bool(kernels) and all(
        record["hardware_comparison"] is not None for record in kernels
    )
    report["complete"] = complete
    completed = [
        record
        for record in kernels
        if record["hardware_comparison"] is not None
        and record["model_comparison"]["predicts_lower"]
    ]
    if not completed:
        report["summary"] = None
        return
    metric_summary = {}
    predicted = [
        float(record["model_comparison"]["predicted_reduction"])
        for record in completed
    ]
    for field in SUMMARY_COUNT_FIELDS:
        comparisons = [
            record["hardware_comparison"]["steady_state"][field]
            for record in completed
        ]
        reductions = [
            float(comparison["reduction"])
            for comparison in comparisons
            if comparison["reduction"] is not None
        ]
        metric_summary[field] = {
            "kernel_count": len(comparisons),
            "fewer_count": sum(bool(comparison["fewer"]) for comparison in comparisons),
            "same_count": sum(
                float(comparison["selected"]) == float(comparison["baseline"])
                for comparison in comparisons
            ),
            "more_count": sum(
                float(comparison["selected"]) > float(comparison["baseline"])
                for comparison in comparisons
            ),
            "median_reduction": statistics.median(reductions) if reductions else None,
            "spearman_with_predicted_reduction": (
                _rank_correlation(predicted, reductions)
                if len(reductions) == len(predicted)
                else None
            ),
        }
    primary = metric_summary[PRIMARY_HARDWARE_METRIC]
    report["summary"] = {
        "eligible_kernel_count": len(completed),
        "primary_hardware_metric": PRIMARY_HARDWARE_METRIC,
        "primary_fewer_count": primary["fewer_count"],
        "primary_fraction": primary["fewer_count"] / len(completed),
        "metrics": metric_summary,
    }


def print_summary(report: Mapping[str, object]) -> None:
    print("\nPaired locality results:")
    for record in report["kernels"]:
        comparison = record.get("hardware_comparison")
        if comparison is None:
            print(f"  {record['display_name']:8s} pending")
            continue
        model = record["model_comparison"]
        primary = comparison["steady_state"][PRIMARY_HARDWARE_METRIC]
        print(
            f"  {record['display_name']:8s} "
            f"Q {100.0 * model['predicted_reduction']:5.1f}% lower; "
            f"TCP->TCC reads {100.0 * primary['reduction']:5.1f}% lower"
        )
    summary = report.get("summary")
    if isinstance(summary, dict):
        print(
            "Primary result: fewer steady-state TCP->TCC reads in "
            f"{summary['primary_fewer_count']}/{summary['eligible_kernel_count']} "
            "kernel pairs."
        )


def run(argv: Sequence[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    kernel_names = tuple(dict.fromkeys(args.kernel or KERNEL_SPECS))
    if args.size < 2 or args.size & (args.size - 1):
        parser.error("--size must be a power of two greater than one")
    if args.block_x * args.block_y > 1024:
        parser.error("--block-x times --block-y must not exceed 1024")
    if args.block_size > 1024:
        parser.error("--block-size must not exceed 1024")
    if args.prepare_only and args.resume:
        parser.error("--prepare-only cannot be combined with --resume")
    if args.prepare_only and args.max_layouts is not None:
        parser.error("--prepare-only cannot be combined with --max-layouts")
    if not args.counter_config.expanduser().exists():
        parser.error(f"counter configuration does not exist: {args.counter_config}")

    output = args.output.expanduser().resolve()
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else output.with_suffix("").with_name(output.stem + "_raw")
    )
    expected_configuration = configuration(args, kernel_names)
    if args.resume:
        if not output.exists():
            parser.error(f"resume checkpoint does not exist: {output}")
        try:
            report = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if report.get("experiment") != EXPERIMENT_NAME:
            parser.error("resume checkpoint is from a different experiment")
        if report.get("configuration") != expected_configuration:
            parser.error("resume checkpoint configuration does not match")
        print(f"Resuming {output}", flush=True)
    else:
        report = prepare_report(args, kernel_names)
        write_report(report, output)

    if args.prepare_only:
        print_summary(report)
        print(f"Wrote solver checkpoint {output}")
        return 0

    pending = []
    for record in report["kernels"]:
        for label in ("baseline", "selected"):
            if record[label]["measurement"] is None:
                pending.append((record, label))
    random.Random(args.seed).shuffle(pending)
    total_pending = len(pending)
    if args.max_layouts is not None:
        pending = pending[: args.max_layouts]

    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="relay-locality-counters-") as temporary:
            build_root = Path(temporary)
            for index, (record, label) in enumerate(pending, 1):
                spec = KERNEL_SPECS[record["kernel"]]
                print(
                    f"[{index}/{len(pending)}; {total_pending} pending] "
                    f"Measuring {spec.display_name} {label}...",
                    flush=True,
                )
                record[label]["measurement"] = measure_layout(
                    spec,
                    label,
                    record[label]["words"],
                    args,
                    build_root,
                    raw_dir,
                )
                report["measurement_order"].append(f"{spec.name}:{label}")
                finalize_report(report)
                write_report(report, output)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    finalize_report(report)
    write_report(report, output)
    print_summary(report)
    print(f"Wrote {output}")
    print(f"Wrote raw profiler data to {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
