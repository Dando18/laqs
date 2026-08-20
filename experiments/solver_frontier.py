#!/usr/bin/env python3
"""Benchmark exact RELAY grammar frontiers on the five kernel evaluators.

For each kernel, this experiment solves the standard grammar ``G_S`` by
exhaustive enumeration and the canonical grammar ``G_C`` by dynamic
programming. Every distinct layout mapping on the resulting analytical
frontier is correctness-checked and timed. The fastest measured member is the
algorithm result, and its speedup is measured against the full row-major
baseline.

The default is the ordinary Pareto frontier over
``(Q_fine, J_peak, J_area, runs, xors)``. A fine-locality-gated frontier from
``notes/relay.tex`` remains available as an experiment setting. Exact
analytical ties remain distinct and are all benchmarked.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import isfinite
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.layout_ranking import (
    KERNEL_SPECS,
    KernelSpec,
    parse_evaluator_output,
)
from relay import (
    SimpleRelayProblem,
    row_major_layout,
    score_layouts,
    score_to_dict,
    simple_solve,
)


EXPERIMENT_NAME = "solver-frontier-speedup"
FINE_COMPONENT = "wave_load.64B"
GRAMMARS = {
    "standard": {
        "notation": "G_S",
        "algorithm": "G_S exhaustive",
        "solver": "exhaustive enumeration",
    },
    "canonical": {
        "notation": "G_C",
        "algorithm": "G_C dynamic programming",
        "solver": "count-grid dynamic programming",
    },
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


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
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
        help="kernel to include; repeat as needed (default: all five)",
    )
    parser.add_argument(
        "--grammar",
        action="append",
        choices=tuple(GRAMMARS),
        default=None,
        help="grammar solver to include; repeat as needed (default: both)",
    )
    parser.add_argument(
        "--size",
        type=positive_integer,
        default=256,
        metavar="N",
        help="square matrix size (default: %(default)s)",
    )
    parser.add_argument(
        "--frontier-type",
        choices=("pareto", "fine-gated"),
        default="pareto",
        help="analytical frontier formulation (default: %(default)s)",
    )
    parser.add_argument(
        "--fine-tolerance",
        type=nonnegative_float,
        default=0.05,
        metavar="EPSILON",
        help=(
            "relative Q_fine gate for --frontier-type=fine-gated "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=10,
        help="timing samples per layout (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_integer,
        default=5,
        help="kernel launches per sample (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_integer,
        default=3,
        help="untimed launches per layout (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=nonnegative_integer,
        default=0,
        help="HIP device ordinal (default: %(default)s)",
    )
    parser.add_argument(
        "--block-size",
        type=positive_integer,
        default=128,
        help="workgroup size for one-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--block-x",
        type=positive_integer,
        default=32,
        help="workgroup width for two-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--block-y",
        type=positive_integer,
        default=32,
        help="workgroup height for two-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--compiler",
        default="hipcc",
        help="HIP compiler passed to kernel evaluators (default: %(default)s)",
    )
    parser.add_argument("--arch", default=None, help="optional GPU architecture")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="benchmark-order shuffle seed (default: %(default)s)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="solve and write an untimed checkpoint without accessing a GPU",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume missing timings from a compatible checkpoint",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=positive_integer,
        default=None,
        metavar="COUNT",
        help="time at most COUNT pending layouts before checkpointing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/solver_frontier.json"),
        help="JSON checkpoint/report path (default: %(default)s)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/solver_frontier_speedup.png"),
        help="final speedup plot path (default: %(default)s)",
    )
    return parser, parser.parse_args(argv)


def _configuration(
    args: argparse.Namespace,
    kernel_names: Sequence[str],
    grammars: Sequence[str],
) -> dict[str, object]:
    return {
        "kernels": list(kernel_names),
        "grammars": list(grammars),
        "matrix_size": args.size,
        "frontier_type": args.frontier_type,
        "fine_component": FINE_COMPONENT,
        "fine_tolerance": (
            args.fine_tolerance
            if args.frontier_type == "fine-gated"
            else None
        ),
        "frontier": (
            "exact Pareto filtering over "
            "(Q_fine, J_peak, J_area, runs, xors); score ties retained"
            if args.frontier_type == "pareto"
            else (
                "Q_fine <= (1 + epsilon) Q_fine*, followed by exact Pareto "
                "filtering over (J_peak, J_area, runs, xors); score ties retained"
            )
        ),
        "samples": args.samples,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "device": args.device,
        "block_size": args.block_size,
        "block_x": args.block_x,
        "block_y": args.block_y,
        "compiler": args.compiler,
        "arch": args.arch,
        "seed": args.seed,
    }


def _problem_inputs(
    spec: KernelSpec, args: argparse.Namespace
) -> tuple[object, tuple, tuple, tuple, tuple, dict[str, float], object]:
    if spec.block_style == "2d":
        block = (args.block_x, args.block_y, 1)
    else:
        block = args.block_size
    config = spec.problem.build_config(
        problem_size=args.size,
        block_size=block,
    )
    matrices = tuple(spec.problem.get_matrices(config))
    events, sequences = spec.problem.get_events_and_sequences(config)
    objectives = tuple(spec.problem.get_objectives(config))
    weights = dict(spec.problem.get_component_weights(config))
    return (
        config,
        matrices,
        tuple(events),
        tuple(sequences),
        objectives,
        weights,
        block,
    )


def _words(
    layouts: Mapping[str, object], matrices: Mapping[str, object]
) -> dict[str, str]:
    return {
        name: layouts[name].word_string(matrices[name])
        for name in layouts
    }


def _cost_dict(cost) -> dict[str, object]:
    return {
        "fine_region_count": cost.fine_region_count,
        "peak_normalized_excess": cost.peak_normalized_excess,
        "weighted_normalized_excess": cost.weighted_normalized_excess,
        "codegen_runs": cost.codegen_runs,
        "codegen_xors": cost.codegen_xors,
    }


def _add_benchmark(
    benchmarks: list[dict[str, object]],
    benchmark_ids: dict[tuple[str, tuple[str, ...]], str],
    spec: KernelSpec,
    words: Mapping[str, str],
) -> str:
    ordered_words = tuple(words[name] for name in spec.evaluator_arrays)
    key = (spec.name, ordered_words)
    existing = benchmark_ids.get(key)
    if existing is not None:
        return existing
    benchmark_id = f"{spec.name}-{len(benchmarks):04d}"
    benchmark_ids[key] = benchmark_id
    benchmarks.append(
        {
            "id": benchmark_id,
            "kernel": spec.name,
            "words": {name: words[name] for name in spec.evaluator_arrays},
            "timing": None,
            "command": None,
            "stdout": None,
            "stderr": None,
        }
    )
    return benchmark_id


def prepare_report(
    args: argparse.Namespace,
    kernel_names: Sequence[str],
    grammars: Sequence[str],
) -> dict[str, object]:
    benchmarks: list[dict[str, object]] = []
    benchmark_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    kernel_records: list[dict[str, object]] = []

    for kernel_name in kernel_names:
        spec = KERNEL_SPECS[kernel_name]
        (
            _config,
            matrices_tuple,
            events,
            sequences,
            objectives,
            weights,
            block,
        ) = _problem_inputs(spec, args)
        matrices = {matrix.name: matrix for matrix in matrices_tuple}
        solver_results = []
        built_results = []
        for grammar in grammars:
            print(
                f"Solving {spec.display_name} N={args.size} "
                f"with {GRAMMARS[grammar]['algorithm']}...",
                flush=True,
            )
            result = simple_solve(
                SimpleRelayProblem(
                    matrices=matrices_tuple,
                    events=events,
                    sequences=sequences,
                    objectives=objectives,
                    grammar=grammar,
                    component_weights=weights,
                    frontier_type=args.frontier_type,
                    fine_component=FINE_COMPONENT,
                    fine_tolerance=args.fine_tolerance,
                    name=f"{spec.name}_{grammar}_{args.size}",
                )
            )
            built_results.append(result)
            frontier_records = []
            for index, member in enumerate(result.frontier):
                words = _words(member.layouts, matrices)
                benchmark_id = _add_benchmark(
                    benchmarks, benchmark_ids, spec, words
                )
                frontier_records.append(
                    {
                        "id": f"{grammar}-{index:04d}",
                        "words": words,
                        "cost": _cost_dict(member.cost),
                        "score": score_to_dict(member.score),
                        "benchmark_id": benchmark_id,
                    }
                )
            solver_results.append(
                {
                    "grammar": grammar,
                    **GRAMMARS[grammar],
                    "exact": result.exact,
                    "solver_elapsed_seconds": result.elapsed_seconds,
                    "array_searches": [
                        asdict(search) for search in result.array_searches
                    ],
                    "joint_raw_frontier_count": (
                        result.joint_raw_frontier_count
                    ),
                    "frontier_definition": {
                        "type": result.problem.frontier_type,
                        "fine_component": result.problem.fine_component,
                        "fine_tolerance": (
                            result.problem.fine_tolerance
                            if result.problem.frontier_type == "fine-gated"
                            else None
                        ),
                        "fine_minimum": result.fine_minimum,
                        "fine_limit": result.fine_limit,
                        "eligible_count": result.fine_eligible_count,
                        "pareto_objectives": list(
                            result.frontier_objectives
                        ),
                    },
                    "frontier_size": len(frontier_records),
                    "frontier": frontier_records,
                    "best": None,
                }
            )
            print(
                f"  retained {len(frontier_records)} joint layouts",
                flush=True,
            )

        components = built_results[0].components
        baseline_layouts = {
            matrix.name: row_major_layout(matrix)
            for matrix in matrices_tuple
        }
        baseline_score = score_layouts(
            matrices,
            components,
            baseline_layouts,
            component_weights=weights,
        )
        baseline_words = {
            name: baseline_layouts[name].word_string(matrices[name])
            for name in spec.evaluator_arrays
        }
        baseline_benchmark = _add_benchmark(
            benchmarks, benchmark_ids, spec, baseline_words
        )
        kernel_records.append(
            {
                "kernel": spec.name,
                "display_name": spec.display_name,
                "matrix_size": args.size,
                "block": list(block) if isinstance(block, tuple) else block,
                "component_weights": weights,
                "objectives": [
                    {
                        "name": component.name,
                        "region_bytes": component.region_bytes,
                        "provenance": component.provenance,
                        "description": component.description,
                        "weight": weights[component.name],
                    }
                    for component in components
                ],
                "baseline": {
                    "algorithm": "Baseline",
                    "layout": "row-major",
                    "words": baseline_words,
                    "score": score_to_dict(baseline_score),
                    "benchmark_id": baseline_benchmark,
                    "speedup": None,
                },
                "solvers": solver_results,
            }
        )

    report: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "configuration": _configuration(
            args, kernel_names, grammars
        ),
        "complete": False,
        "benchmark_run_order": [],
        "benchmarks": benchmarks,
        "kernels": kernel_records,
        "plot_data": [],
    }
    finalize_report(report)
    return report


def evaluator_command(
    spec: KernelSpec,
    words: Mapping[str, str],
    args: argparse.Namespace,
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
    ]
    if spec.block_style == "2d":
        command.extend(
            (
                "--block-x",
                str(args.block_x),
                "--block-y",
                str(args.block_y),
            )
        )
    else:
        command.extend(("--block-size", str(args.block_size)))
    command.extend(("--compiler", args.compiler))
    if args.arch:
        command.extend(("--arch", args.arch))
    return command


def benchmark(
    record: dict[str, object], args: argparse.Namespace
) -> None:
    spec = KERNEL_SPECS[str(record["kernel"])]
    words = record["words"]
    assert isinstance(words, dict)
    command = evaluator_command(spec, words, args)
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
            f"benchmark {record['id']} exited with {completed.returncode}: "
            f"{detail}"
        )
    try:
        timing = parse_evaluator_output(completed.stdout)
    except ValueError as error:
        raise RuntimeError(f"benchmark {record['id']}: {error}") from error
    record["timing"] = asdict(timing)
    record["command"] = command
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr


def finalize_report(report: dict[str, object]) -> None:
    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, list)
    by_id = {
        str(record["id"]): record
        for record in benchmarks
        if isinstance(record, dict)
    }
    complete = bool(benchmarks) and all(
        record["timing"] is not None for record in benchmarks
    )
    report["complete"] = complete
    plot_data = []

    kernels = report["kernels"]
    assert isinstance(kernels, list)
    for kernel in kernels:
        assert isinstance(kernel, dict)
        baseline = kernel["baseline"]
        assert isinstance(baseline, dict)
        baseline_record = by_id[str(baseline["benchmark_id"])]
        if baseline_record["timing"] is None:
            baseline["speedup"] = None
            for solver in kernel["solvers"]:
                solver["best"] = None
            continue
        baseline_timing = baseline_record["timing"]
        assert isinstance(baseline_timing, dict)
        baseline_ms = float(baseline_timing["median_ms"])
        baseline["speedup"] = 1.0
        if not complete:
            continue
        plot_data.append(
            {
                "kernel": kernel["kernel"],
                "display_name": kernel["display_name"],
                "algorithm": "Baseline",
                "speedup": 1.0,
                "median_ms": baseline_ms,
                "words": baseline["words"],
            }
        )
        for solver in kernel["solvers"]:
            frontier = solver["frontier"]
            best = min(
                frontier,
                key=lambda candidate: (
                    float(
                        by_id[str(candidate["benchmark_id"])]["timing"][
                            "median_ms"
                        ]
                    ),
                    tuple(sorted(candidate["words"].items())),
                ),
            )
            best_benchmark = by_id[str(best["benchmark_id"])]
            timing = best_benchmark["timing"]
            assert isinstance(timing, dict)
            median_ms = float(timing["median_ms"])
            speedup = baseline_ms / median_ms
            solver["best"] = {
                "frontier_member_id": best["id"],
                "benchmark_id": best["benchmark_id"],
                "words": best["words"],
                "cost": best["cost"],
                "median_ms": median_ms,
                "speedup": speedup,
            }
            plot_data.append(
                {
                    "kernel": kernel["kernel"],
                    "display_name": kernel["display_name"],
                    "algorithm": solver["algorithm"],
                    "speedup": speedup,
                    "median_ms": median_ms,
                    "words": best["words"],
                }
            )
    report["plot_data"] = plot_data if complete else []


def write_report(report: Mapping[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def render_plot(report: Mapping[str, object], output: Path) -> None:
    if report.get("complete") is not True:
        return
    cache = Path("/tmp/relay-matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = report["plot_data"]
    assert isinstance(rows, list)
    data = {
        name: [row[name] for row in rows]
        for name in ("display_name", "algorithm", "speedup")
    }
    configuration = report["configuration"]
    assert isinstance(configuration, dict)
    kernel_order = [
        KERNEL_SPECS[name].display_name
        for name in configuration["kernels"]
    ]
    algorithm_order = ["Baseline"] + [
        str(GRAMMARS[name]["algorithm"])
        for name in configuration["grammars"]
    ]
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    sns.barplot(
        data=data,
        x="display_name",
        y="speedup",
        hue="algorithm",
        order=kernel_order,
        hue_order=algorithm_order,
        errorbar=None,
        ax=axis,
    )
    axis.axhline(1.0, color="0.25", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Kernel")
    axis.set_ylabel("Speedup over row-major baseline (×)")
    axis.set_title(
        "Best measured layout from each RELAY solver frontier "
        f"(N={configuration['matrix_size']})"
    )
    axis.legend(title="Layout algorithm", frameon=True)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f×", padding=3, fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
    plt.close(figure)
    temporary.replace(output)


def print_summary(report: Mapping[str, object]) -> None:
    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, list)
    complete_count = sum(record["timing"] is not None for record in benchmarks)
    print(f"Benchmarks complete: {complete_count}/{len(benchmarks)}")
    if report.get("complete") is not True:
        return
    print("\nBest frontier layouts:")
    for row in report["plot_data"]:
        if row["algorithm"] == "Baseline":
            continue
        words = ", ".join(
            f"{name}={word}" for name, word in row["words"].items()
        )
        print(
            f"  {row['display_name']:8s} {row['algorithm']:25s} "
            f"{row['speedup']:.3f}x  {words}"
        )


def run(argv: Sequence[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    kernel_names = tuple(dict.fromkeys(args.kernel or KERNEL_SPECS))
    grammars = tuple(dict.fromkeys(args.grammar or GRAMMARS))
    if args.size < 2 or args.size & (args.size - 1):
        parser.error("--size must be a power of two greater than one")
    if args.block_x * args.block_y > 1024:
        parser.error("--block-x times --block-y must not exceed 1024")
    if args.block_size > 1024:
        parser.error("--block-size must not exceed 1024")
    if args.prepare_only and args.resume:
        parser.error("--prepare-only cannot be combined with --resume")
    if args.prepare_only and args.max_benchmarks is not None:
        parser.error("--prepare-only cannot be combined with --max-benchmarks")

    output = args.output.expanduser().resolve()
    plot = args.plot.expanduser().resolve()
    expected_configuration = _configuration(args, kernel_names, grammars)
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
        report = prepare_report(args, kernel_names, grammars)
        write_report(report, output)

    if not args.prepare_only:
        benchmarks = report["benchmarks"]
        assert isinstance(benchmarks, list)
        pending = [record for record in benchmarks if record["timing"] is None]
        random.Random(args.seed).shuffle(pending)
        total_pending = len(pending)
        if args.max_benchmarks is not None:
            pending = pending[: args.max_benchmarks]
        try:
            for index, record in enumerate(pending, 1):
                print(
                    f"[{index}/{len(pending)}; {total_pending} pending] "
                    f"Benchmarking {record['id']}...",
                    flush=True,
                )
                benchmark(record, args)
                run_order = report["benchmark_run_order"]
                assert isinstance(run_order, list)
                run_order.append(record["id"])
                finalize_report(report)
                write_report(report, output)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    finalize_report(report)
    write_report(report, output)
    render_plot(report, plot)
    print_summary(report)
    print(f"Wrote {output}")
    if report.get("complete") is True:
        print(f"Wrote {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
