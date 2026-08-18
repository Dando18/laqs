#!/usr/bin/env python3
r"""Compare RELAY layout scores with GEMM and GESUMMV runtimes.

The experiment applies the same traditional canonical layout to every target
matrix in a kernel and repeats the comparison for several square matrix sizes.
Canonical words list physical element-address bits from low to high.  Context
vectors in GESUMMV remain contiguous.

Runtime ranks use exact sample medians.  Variation-aware metrics do not change
those ranks: they only check whether a score rank lies within the plausible
runtime-rank interval implied by each layout's observed sample range.

Scoring runs without a GPU.  Runtime measurement must run in a GPU allocation::

    flux run -n1 -g1 -t 5m -q pdebug \
        .venv/bin/python experiments/layout_ranking.py \
        --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942

All scores, runtimes, and ranks are ascending costs; lower is better.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import ModuleType
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernels.gemm import problem as gemm_problem
from kernels.gesummv import problem as gesummv_problem
from relay import (
    SCORE_MODES,
    CanonicalLayout,
    MatrixSpec,
    ScoreMode,
    canonical_layout_from_word,
    row_major_layout,
    score_layouts,
    score_to_dict,
)
from relay.layouts import Layout
from relay.objectives import build_objectives


DEFAULT_SIZES = (256, 512, 1024)
VARIATION_METHOD = "observed-sample-range-rank-bounds"


@dataclass(frozen=True)
class KernelSpec:
    """Problem and evaluator information for one kernel."""

    name: str
    display_name: str
    problem: ModuleType
    evaluator: Path
    evaluator_arrays: tuple[str, ...]


KERNEL_SPECS = {
    "gemm": KernelSpec(
        "gemm",
        "GEMM",
        gemm_problem,
        REPOSITORY_ROOT / "kernels" / "gemm" / "evaluate.py",
        ("A", "B", "C"),
    ),
    "gesummv": KernelSpec(
        "gesummv",
        "GESUMMV",
        gesummv_problem,
        REPOSITORY_ROOT / "kernels" / "gesummv" / "evaluate.py",
        ("A", "B"),
    ),
}


@dataclass(frozen=True)
class LayoutCase:
    """One traditional canonical layout applied to every target matrix."""

    name: str
    word: str


@dataclass(frozen=True)
class TimingResult:
    """Kernel-only statistics and samples printed by a HIP evaluator."""

    device: str
    median_ms: float
    mean_ms: float
    min_ms: float
    sd_ms: float
    gflops: float
    samples_ms: tuple[float, ...]


def traditional_layout_cases(n: int) -> tuple[LayoutCase, ...]:
    """Return uniform global and square-tiled row/column-major controls."""

    if n < 2 or n & (n - 1):
        raise ValueError("matrix sizes must be powers of two greater than one")

    matrix_bits = n.bit_length() - 1
    cases = [
        LayoutCase("row_major", "j" * matrix_bits + "i" * matrix_bits),
        LayoutCase("column_major", "i" * matrix_bits + "j" * matrix_bits),
    ]
    for tile_size in (8, 16, 32):
        if tile_size > n:
            continue
        tile_bits = tile_size.bit_length() - 1
        cases.extend(
            (
                LayoutCase(
                    f"tile{tile_size}_row_major",
                    "j" * tile_bits + "i" * tile_bits,
                ),
                LayoutCase(
                    f"tile{tile_size}_column_major",
                    "i" * tile_bits + "j" * tile_bits,
                ),
            )
        )
    return tuple(cases)


def average_tie_ranks(values: Sequence[float]) -> list[float]:
    """Rank ascending values from one, assigning exact ties their mean rank."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def observed_timing_interval(timing: TimingResult) -> tuple[float, float]:
    """Return the inclusive minimum/maximum interval of raw timing samples."""

    return min(timing.samples_ms), max(timing.samples_ms)


def runtime_rank_ranges(
    timings: Sequence[TimingResult],
) -> list[tuple[int, int]]:
    """Find each layout's plausible rank bounds from observed sample ranges.

    Another layout is definitely faster only when its entire observed interval
    is below this layout's interval.  It is definitely slower under the
    symmetric condition.  Every overlapping interval remains a possible order.
    """

    intervals = [observed_timing_interval(timing) for timing in timings]
    ranges: list[tuple[int, int]] = []
    for index, (lower, upper) in enumerate(intervals):
        definitely_faster = sum(
            other_upper < lower
            for other_index, (_, other_upper) in enumerate(intervals)
            if other_index != index
        )
        definitely_slower = sum(
            other_lower > upper
            for other_index, (other_lower, _) in enumerate(intervals)
            if other_index != index
        )
        ranges.append(
            (1 + definitely_faster, len(intervals) - definitely_slower)
        )
    return ranges


def variation_aware_rank_metrics(
    score_values: Sequence[float], timings: Sequence[TimingResult]
) -> dict[str, object]:
    """Measure score-rank error against sample-range runtime-rank bounds."""

    if len(score_values) != len(timings):
        raise ValueError("score and timing inputs must have the same length")
    score_ranks = average_tie_ranks(score_values)
    rank_ranges = runtime_rank_ranges(timings)
    errors = [
        max(float(lower) - rank, 0.0, rank - float(upper))
        for rank, (lower, upper) in zip(score_ranks, rank_ranges)
    ]
    accurate = sum(error == 0.0 for error in errors)
    total = len(errors)
    return {
        "method": VARIATION_METHOD,
        "accurate_layouts": accurate,
        "total_layouts": total,
        "rank_accuracy": None if total == 0 else accurate / total,
        "mean_rank_error": None if total == 0 else statistics.fmean(errors),
        "max_rank_error": None if total == 0 else max(errors),
    }


def parse_evaluator_output(output: str) -> TimingResult:
    """Parse the correctness and timing block shared by both HIP evaluators."""

    lines = output.splitlines()
    if not any(line.startswith("Correctness: PASS") for line in lines):
        raise ValueError("evaluator did not report Correctness: PASS")

    header = "median_ms  mean_ms  min_ms  sd_ms  GFLOP/s"
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.strip() == header
        )
        result_fields = lines[header_index + 1].split()
    except (StopIteration, IndexError) as error:
        raise ValueError("evaluator output has no timing result row") from error
    if len(result_fields) != 5:
        raise ValueError("evaluator timing row must contain five numbers")
    try:
        median_ms, mean_ms, min_ms, sd_ms, gflops = map(float, result_fields)
    except ValueError as error:
        raise ValueError("evaluator timing row contains a non-number") from error

    sample_prefix = "Samples (ms):"
    sample_line = next(
        (line for line in lines if line.startswith(sample_prefix)), None
    )
    if sample_line is None:
        raise ValueError("evaluator output has no timing samples")
    try:
        samples_ms = tuple(
            float(value) for value in sample_line[len(sample_prefix) :].split()
        )
    except ValueError as error:
        raise ValueError("evaluator samples contain a non-number") from error
    if not samples_ms:
        raise ValueError("evaluator reported an empty sample set")

    device_prefix = "Device:"
    device = next(
        (
            line[len(device_prefix) :].strip()
            for line in lines
            if line.startswith(device_prefix)
        ),
        "unknown",
    )
    return TimingResult(
        device=device,
        median_ms=median_ms,
        mean_ms=mean_ms,
        min_ms=min_ms,
        sd_ms=sd_ms,
        gflops=gflops,
        samples_ms=samples_ms,
    )


def parse_component_weights(values: Sequence[str]) -> dict[str, float]:
    """Parse repeated ``OBJECTIVE=WEIGHT`` assignments."""

    weights: dict[str, float] = {}
    for text in values:
        name, separator, value = text.partition("=")
        if not separator or not name or not value:
            raise ValueError(
                f"--component-weight expects OBJECTIVE=WEIGHT, got {text!r}"
            )
        try:
            weights[name] = float(value)
        except ValueError as error:
            raise ValueError(
                f"component weight for {name!r} must be a number"
            ) from error
        if weights[name] < 0:
            raise ValueError("component weights must be nonnegative")
    return weights


def layouts_for_case(
    case: LayoutCase, matrices: Mapping[str, MatrixSpec]
) -> dict[str, Layout]:
    """Apply the case to targets and keep context arrays row-major."""

    layouts: dict[str, Layout] = {}
    for name, matrix in matrices.items():
        if matrix.target:
            layouts[name] = canonical_layout_from_word(
                matrix,
                case.word,
                name=f"{case.name}.{name}",
            )
        else:
            layouts[name] = row_major_layout(matrix)
    return layouts


def _problem_config(
    spec: KernelSpec, n: int, args: argparse.Namespace
) -> tuple[object, object]:
    if spec.name == "gemm":
        block = (args.gemm_block_x, args.gemm_block_y, 1)
        return spec.problem.build_config(problem_size=n, block_size=block), list(block)
    block = args.gesummv_block_size
    return spec.problem.build_config(problem_size=n, block_size=block), block


def evaluator_command(
    spec: KernelSpec,
    n: int,
    case: LayoutCase,
    args: argparse.Namespace,
) -> list[str]:
    """Build one kernel evaluator invocation."""

    command = [
        sys.executable,
        str(spec.evaluator),
        *[case.word for _ in spec.evaluator_arrays],
        "--n",
        str(n),
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--device",
        str(args.device),
    ]
    if spec.name == "gemm":
        command.extend(
            (
                "--block-x",
                str(args.gemm_block_x),
                "--block-y",
                str(args.gemm_block_y),
            )
        )
    else:
        command.extend(("--block-size", str(args.gesummv_block_size)))
    command.extend(("--compiler", args.compiler))
    if args.arch:
        command.extend(("--arch", args.arch))
    return command


def benchmark_case(
    spec: KernelSpec,
    n: int,
    case: LayoutCase,
    args: argparse.Namespace,
) -> tuple[TimingResult, list[str], str, str]:
    """Run and parse one layout benchmark, retaining complete process output."""

    command = evaluator_command(spec, n, case, args)
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
            f"{spec.name} N={n} benchmark {case.name!r} exited with "
            f"{completed.returncode}: {detail}"
        )
    try:
        timing = parse_evaluator_output(completed.stdout)
    except ValueError as error:
        raise RuntimeError(
            f"{spec.name} N={n} benchmark {case.name!r}: {error}"
        ) from error
    return timing, command, completed.stdout, completed.stderr


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
    argv: list[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kernel",
        action="append",
        choices=tuple(KERNEL_SPECS),
        default=None,
        help="kernel to include; repeat as needed (default: both)",
    )
    parser.add_argument(
        "--size",
        action="append",
        type=positive_integer,
        default=None,
        metavar="N",
        help="square matrix size; repeat as needed (default: 256, 512, 1024)",
    )
    parser.add_argument(
        "--samples",
        type=positive_integer,
        default=10,
        help="independent HIP timing samples per layout (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_integer,
        default=5,
        help="kernel launches per timing sample (default: %(default)s)",
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
        "--gemm-block-x",
        type=positive_integer,
        default=32,
        help="GEMM workgroup width / j threads (default: %(default)s)",
    )
    parser.add_argument(
        "--gemm-block-y",
        type=positive_integer,
        default=32,
        help="GEMM workgroup height / i threads (default: %(default)s)",
    )
    parser.add_argument(
        "--gesummv-block-size",
        type=positive_integer,
        default=128,
        help="GESUMMV one-dimensional workgroup size (default: %(default)s)",
    )
    parser.add_argument(
        "--compiler",
        default="hipcc",
        help="HIP compiler command passed to both evaluators (default: %(default)s)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="optional GPU architecture, for example gfx942",
    )
    parser.add_argument(
        "--score-mode",
        choices=SCORE_MODES,
        default="weighted-normalized-excess",
        help="scalar cost used for displayed score ranks (default: %(default)s)",
    )
    parser.add_argument(
        "--component-weight",
        action="append",
        default=[],
        metavar="OBJECTIVE=WEIGHT",
        help=(
            "override one problem-provided tau weight wherever that objective "
            "exists"
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="compute scores and reports without running HIP evaluators",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume missing benchmarks from an existing compatible JSON report",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=positive_integer,
        default=None,
        metavar="COUNT",
        help=(
            "run at most this many missing benchmarks, checkpoint, and exit; "
            "use with --resume to split work across allocations"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/layout_ranking.json"),
        metavar="JSON",
        help="complete JSON report (default: %(default)s)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        metavar="MARKDOWN",
        help="Markdown report path (default: JSON path with .md suffix)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used to randomize all benchmark jobs (default: %(default)s)",
    )
    return parser, parser.parse_args(argv)


def _layout_words(
    matrices: Mapping[str, MatrixSpec], layouts: Mapping[str, Layout]
) -> dict[str, str]:
    words: dict[str, str] = {}
    for name, layout in layouts.items():
        if not isinstance(layout, CanonicalLayout):
            raise ValueError(f"experiment layout {layout.name!r} is not canonical")
        words[name] = layout.word_string(matrices[name])
    return words


def score_group(
    spec: KernelSpec,
    n: int,
    args: argparse.Namespace,
    component_weight_overrides: Mapping[str, float],
) -> tuple[dict[str, object], tuple[LayoutCase, ...], set[str]]:
    """Build and score one kernel/size group."""

    cases = traditional_layout_cases(n)
    config, block = _problem_config(spec, n, args)
    matrices_tuple = tuple(spec.problem.get_matrices(config))
    event_items, sequences = spec.problem.get_events_and_sequences(config)
    objective_specs = tuple(spec.problem.get_objectives(config))
    matrices = {matrix.name: matrix for matrix in matrices_tuple}
    events = {event.id: event for event in event_items}
    print(f"Building {spec.display_name} N={n} objective components...", flush=True)
    components = tuple(
        build_objectives(objective_specs, matrices, events, sequences)
    )
    component_names = {component.name for component in components}
    default_weights = dict(spec.problem.get_component_weights(config))
    unknown_defaults = sorted(set(default_weights) - component_names)
    missing_defaults = sorted(component_names - set(default_weights))
    if unknown_defaults or missing_defaults:
        details = []
        if unknown_defaults:
            details.append("unknown: " + ", ".join(unknown_defaults))
        if missing_defaults:
            details.append("missing: " + ", ".join(missing_defaults))
        raise ValueError(
            f"{spec.name} component-weight table does not match its objectives "
            f"({'; '.join(details)})"
        )
    applicable_overrides = {
        name: weight
        for name, weight in component_weight_overrides.items()
        if name in component_names
    }
    applied_weights = {**default_weights, **applicable_overrides}

    mode: ScoreMode = args.score_mode
    records: list[dict[str, object]] = []
    print(f"Scoring {len(cases)} traditional layouts...", flush=True)
    for case in cases:
        layouts = layouts_for_case(case, matrices)
        score = score_layouts(
            matrices,
            components,
            layouts,
            component_weights=applied_weights,
        )
        records.append(
            {
                "name": case.name,
                "word": case.word,
                "layouts": _layout_words(matrices, layouts),
                "selected_score": score.value(mode),
                "aggregate_scores": {
                    aggregate_mode: score.value(aggregate_mode)
                    for aggregate_mode in SCORE_MODES
                },
                "score": score_to_dict(score),
                "score_rank": None,
                "runtime_rank": None,
                "rank_delta": None,
                "timing": None,
                "timing_variation": None,
                "runtime_rank_variation_range": None,
                "score_rank_within_runtime_variation": None,
                "variation_aware_rank_error": None,
                "benchmark_command": None,
                "benchmark_stdout": None,
                "benchmark_stderr": None,
            }
        )

    score_ranks = average_tie_ranks(
        [float(record["selected_score"]) for record in records]
    )
    for record, rank in zip(records, score_ranks):
        record["score_rank"] = rank

    group: dict[str, object] = {
        "kernel": spec.name,
        "display_name": spec.display_name,
        "matrix_size": n,
        "block": block,
        "score_mode": mode,
        "component_weights": applied_weights,
        "component_weight_overrides": applicable_overrides,
        "objectives": [
            {
                "name": component.name,
                "region_bytes": component.region_bytes,
                "provenance": component.provenance,
                "description": component.description,
                "weight": applied_weights[component.name],
            }
            for component in components
        ],
        "benchmark_run_order": [],
        "variation_aware_rank_metrics": {
            aggregate_mode: None for aggregate_mode in SCORE_MODES
        },
        "results": records,
    }
    return group, cases, component_names


def _timing_from_record(record: Mapping[str, object]) -> TimingResult:
    timing = record["timing"]
    assert isinstance(timing, dict)
    return TimingResult(
        device=str(timing["device"]),
        median_ms=float(timing["median_ms"]),
        mean_ms=float(timing["mean_ms"]),
        min_ms=float(timing["min_ms"]),
        sd_ms=float(timing["sd_ms"]),
        gflops=float(timing["gflops"]),
        samples_ms=tuple(float(value) for value in timing["samples_ms"]),
    )


def finalize_runtime_group(group: dict[str, object]) -> None:
    """Add exact ranks and variation-aware metrics after benchmark completion."""

    records = group["results"]
    assert isinstance(records, list)
    timings = [_timing_from_record(record) for record in records]
    runtime_ranks = average_tie_ranks([timing.median_ms for timing in timings])
    rank_ranges = runtime_rank_ranges(timings)

    selected_score_values = [
        float(record["selected_score"]) for record in records
    ]
    selected_score_ranks = average_tie_ranks(selected_score_values)
    for record, timing, runtime_rank, score_rank, rank_range in zip(
        records, timings, runtime_ranks, selected_score_ranks, rank_ranges
    ):
        lower_ms, upper_ms = observed_timing_interval(timing)
        lower_rank, upper_rank = rank_range
        rank_error = max(
            float(lower_rank) - score_rank,
            0.0,
            score_rank - float(upper_rank),
        )
        record["runtime_rank"] = runtime_rank
        record["rank_delta"] = score_rank - runtime_rank
        record["timing_variation"] = {
            "method": "observed-sample-range",
            "lower_ms": lower_ms,
            "upper_ms": upper_ms,
        }
        record["runtime_rank_variation_range"] = [lower_rank, upper_rank]
        record["score_rank_within_runtime_variation"] = rank_error == 0.0
        record["variation_aware_rank_error"] = rank_error

    metrics: dict[str, object] = {}
    for aggregate_mode in SCORE_MODES:
        score_values = []
        for record in records:
            aggregate_scores = record["aggregate_scores"]
            assert isinstance(aggregate_scores, dict)
            score_values.append(float(aggregate_scores[aggregate_mode]))
        metrics[aggregate_mode] = variation_aware_rank_metrics(
            score_values, timings
        )
    group["variation_aware_rank_metrics"] = metrics


def _format_rank(value: object) -> str:
    if value is None:
        return "—"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.1f}"


def _format_metric(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def markdown_report(report: Mapping[str, object]) -> str:
    """Render the combined human-readable report."""

    configuration = report["configuration"]
    assert isinstance(configuration, dict)
    groups = report["runs"]
    assert isinstance(groups, list)
    score_mode = str(report["score_mode"])
    score_only = bool(configuration["score_only"])

    lines = [
        "# RELAY layout score/runtime experiment",
        "",
        "All scores, runtimes, and ranks are ascending costs; lower is better. "
        f"The displayed score uses `{score_mode}`.",
        "",
        "Runtime rank is the raw rank of the exact sample median. Score rank is "
        "the raw rank of the exact modeled score. Timing variation does not "
        "change either rank or any table value.",
        "",
        "The variation-aware rank metric uses each layout's observed minimum-to-"
        "maximum sample interval. An overlapping competitor can appear on either "
        "side, producing a plausible runtime-rank range. A score rank is counted "
        "accurate when it lies inside that range. This is a conservative observed-"
        "sample check, not a confidence interval.",
        "",
        "## Summary",
        "",
    ]

    summary_rows: list[list[str]] = []
    for group in groups:
        assert isinstance(group, dict)
        records = group["results"]
        assert isinstance(records, list)
        group_complete = all(record["timing"] is not None for record in records)
        if score_only or not group_complete:
            accuracy = "—"
            mean_error = "—"
        else:
            metrics_by_mode = group["variation_aware_rank_metrics"]
            assert isinstance(metrics_by_mode, dict)
            metric = metrics_by_mode[score_mode]
            assert isinstance(metric, dict)
            accuracy = _format_metric(metric["rank_accuracy"])
            mean_error = _format_metric(metric["mean_rank_error"])
        summary_rows.append(
            [
                str(group["display_name"]),
                str(group["matrix_size"]),
                str(len(records)),
                accuracy,
                mean_error,
            ]
        )
    lines.extend(
        (
            _markdown_table(
                (
                    "Kernel",
                    "N",
                    "Layouts",
                    "Variation-aware rank accuracy",
                    "Mean rank error",
                ),
                summary_rows,
            ),
            "",
        )
    )

    for group in groups:
        assert isinstance(group, dict)
        records = group["results"]
        assert isinstance(records, list)
        lines.extend(
            (
                f"## {group['display_name']} — N={group['matrix_size']}",
                "",
                f"Workgroup: `{group['block']}`.",
                "",
            )
        )
        objectives = group["objectives"]
        assert isinstance(objectives, list)
        objective_rows = [
            [
                f"`{objective['name']}`",
                str(objective["provenance"]),
                str(objective["region_bytes"]),
                f"{float(objective['weight']):.6g}",
                str(objective["description"]),
            ]
            for objective in objectives
        ]
        lines.extend(
            (
                "### Objective model",
                "",
                "`grounded` scopes come from traced memory instructions. "
                "`hypothesis` scopes encode proposed reuse or cache-locality "
                "neighborhoods.",
                "",
                _markdown_table(
                    ("Objective", "Provenance", "Region B", "Tau", "Meaning"),
                    objective_rows,
                ),
                "",
                "### Layout ranks",
                "",
            )
        )
        ordered = sorted(
            records,
            key=lambda record: (
                float(record["score_rank"]),
                str(record["name"]),
            ),
        )
        group_complete = all(record["timing"] is not None for record in records)
        if score_only:
            headers = ("Score rank", "Layout", "Word (low→high)", "Score")
            rows = [
                [
                    _format_rank(record["score_rank"]),
                    f"`{record['name']}`",
                    f"`{record['word']}`",
                    f"{float(record['selected_score']):.6g}",
                ]
                for record in ordered
            ]
        else:
            headers = (
                "Score rank",
                "Runtime rank",
                "Layout",
                "Word (low→high)",
                "Score",
                "Median ms",
                "Mean ms",
                "SD ms",
                "Observed range ms",
                "GFLOP/s",
                "Rank delta",
            )
            rows = []
            for record in ordered:
                timing = record["timing"]
                variation = record["timing_variation"]
                if not isinstance(timing, dict):
                    rows.append(
                        [
                            _format_rank(record["score_rank"]),
                            "—",
                            f"`{record['name']}`",
                            f"`{record['word']}`",
                            f"{float(record['selected_score']):.6g}",
                            "pending",
                            "pending",
                            "pending",
                            "pending",
                            "pending",
                            "—",
                        ]
                    )
                    continue
                if isinstance(variation, dict):
                    interval = (
                        f"{float(variation['lower_ms']):.6f}–"
                        f"{float(variation['upper_ms']):.6f}"
                    )
                else:
                    interval = "pending"
                rows.append(
                    [
                        _format_rank(record["score_rank"]),
                        _format_rank(record["runtime_rank"]),
                        f"`{record['name']}`",
                        f"`{record['word']}`",
                        f"{float(record['selected_score']):.6g}",
                        f"{float(timing['median_ms']):.6f}",
                        f"{float(timing['mean_ms']):.6f}",
                        f"{float(timing['sd_ms']):.6f}",
                        interval,
                        f"{float(timing['gflops']):.2f}",
                        (
                            "—"
                            if record["rank_delta"] is None
                            else f"{float(record['rank_delta']):+.1f}"
                        ),
                    ]
                )
        lines.extend((_markdown_table(headers, rows), ""))

        if not score_only and group_complete:
            metrics_by_mode = group["variation_aware_rank_metrics"]
            assert isinstance(metrics_by_mode, dict)
            metric_rows: list[list[str]] = []
            for mode in SCORE_MODES:
                metric = metrics_by_mode[mode]
                assert isinstance(metric, dict)
                metric_rows.append(
                    [
                        f"`{mode}`" + (" (selected)" if mode == score_mode else ""),
                        f"{metric['accurate_layouts']}/{metric['total_layouts']}",
                        _format_metric(metric["rank_accuracy"]),
                        _format_metric(metric["mean_rank_error"]),
                        _format_metric(metric["max_rank_error"]),
                    ]
                )
            lines.extend(
                (
                    "### Variation-aware metrics",
                    "",
                    _markdown_table(
                        (
                            "Score mode",
                            "Ranks within variation",
                            "Accuracy",
                            "Mean rank error",
                            "Max rank error",
                        ),
                        metric_rows,
                    ),
                    "",
                )
            )
        elif not score_only:
            completed = sum(record["timing"] is not None for record in records)
            lines.extend(
                (
                    f"Variation-aware metrics pending ({completed}/{len(records)} "
                    "layout benchmarks complete).",
                    "",
                )
            )

    return "\n".join(lines).rstrip() + "\n"


def print_summary(report: Mapping[str, object]) -> None:
    """Print a compact summary; detailed tables live in the Markdown report."""

    configuration = report["configuration"]
    groups = report["runs"]
    assert isinstance(configuration, dict)
    assert isinstance(groups, list)
    score_mode = str(report["score_mode"])
    print("\nRELAY multi-kernel traditional-layout experiment")
    print(f"  score mode: {score_mode} (lower is better)")
    for group in groups:
        assert isinstance(group, dict)
        label = f"{group['display_name']} N={group['matrix_size']}"
        if configuration["score_only"]:
            print(f"  {label}: scoring complete")
            continue
        records = group["results"]
        assert isinstance(records, list)
        completed = sum(record["timing"] is not None for record in records)
        if completed != len(records):
            print(f"  {label}: benchmark checkpoint {completed}/{len(records)}")
            continue
        metrics_by_mode = group["variation_aware_rank_metrics"]
        assert isinstance(metrics_by_mode, dict)
        metric = metrics_by_mode[score_mode]
        assert isinstance(metric, dict)
        print(
            f"  {label}: variation-aware rank accuracy "
            f"{metric['accurate_layouts']}/{metric['total_layouts']} "
            f"({_format_metric(metric['rank_accuracy'])})"
        )


def _configuration(
    args: argparse.Namespace,
    kernel_names: Sequence[str],
    sizes: Sequence[int],
) -> dict[str, object]:
    return {
        "kernels": list(kernel_names),
        "matrix_sizes": list(sizes),
        "gemm_block": [args.gemm_block_x, args.gemm_block_y, 1],
        "gesummv_block_size": args.gesummv_block_size,
        "samples": args.samples,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "device": args.device,
        "compiler": args.compiler,
        "arch": args.arch,
        "score_only": args.score_only,
        "seed": args.seed,
    }


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def write_reports(
    report: Mapping[str, object], output_path: Path, markdown_path: Path
) -> None:
    """Atomically write a JSON checkpoint and its Markdown rendering."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        output_path, json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(markdown_path, markdown_report(report))


def _prepared_from_report(
    report: Mapping[str, object],
) -> list[tuple[KernelSpec, int, dict[str, object], tuple[LayoutCase, ...]]]:
    groups = report["runs"]
    if not isinstance(groups, list):
        raise ValueError("resume report has no run list")
    prepared = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("resume report contains an invalid run")
        kernel_name = str(group["kernel"])
        if kernel_name not in KERNEL_SPECS:
            raise ValueError(f"resume report names unknown kernel {kernel_name!r}")
        records = group["results"]
        if not isinstance(records, list):
            raise ValueError("resume report run has no result list")
        cases = tuple(
            LayoutCase(str(record["name"]), str(record["word"]))
            for record in records
        )
        prepared.append(
            (
                KERNEL_SPECS[kernel_name],
                int(group["matrix_size"]),
                group,
                cases,
            )
        )
    return prepared


def run(argv: list[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    try:
        component_weight_overrides = parse_component_weights(
            args.component_weight
        )
        kernel_names = tuple(dict.fromkeys(args.kernel or KERNEL_SPECS))
        sizes = tuple(dict.fromkeys(args.size or DEFAULT_SIZES))
        for n in sizes:
            traditional_layout_cases(n)
        if args.gemm_block_x * args.gemm_block_y > 1024:
            raise ValueError(
                "--gemm-block-x times --gemm-block-y must not exceed 1024"
            )
        if args.gesummv_block_size > 1024:
            raise ValueError("--gesummv-block-size must not exceed 1024")
        if args.resume and args.score_only:
            raise ValueError("--resume cannot be combined with --score-only")
        if args.max_benchmarks is not None and args.score_only:
            raise ValueError("--max-benchmarks cannot be combined with --score-only")
    except ValueError as error:
        parser.error(str(error))

    output_path = args.output.expanduser().resolve()
    markdown_path = (
        args.markdown.expanduser().resolve()
        if args.markdown is not None
        else output_path.with_suffix(".md")
    )
    expected_configuration = _configuration(args, kernel_names, sizes)

    if args.resume:
        if not output_path.exists():
            parser.error(f"--resume report does not exist: {output_path}")
        try:
            report = json.loads(output_path.read_text())
            if report.get("experiment") != "multi-kernel-traditional-layout-ranking":
                raise ValueError("resume report is from a different experiment")
            if report.get("configuration") != expected_configuration:
                raise ValueError(
                    "resume report configuration does not match the requested run"
                )
            if report.get("score_mode") != args.score_mode:
                raise ValueError("resume report uses a different --score-mode")
            if (
                report.get("component_weight_overrides")
                != component_weight_overrides
            ):
                raise ValueError(
                    "resume report uses different component-weight overrides"
                )
            prepared = _prepared_from_report(report)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        print(f"Resuming benchmark checkpoint {output_path}", flush=True)
    else:
        prepared: list[
            tuple[KernelSpec, int, dict[str, object], tuple[LayoutCase, ...]]
        ] = []
        known_components: set[str] = set()
        try:
            for kernel_name in kernel_names:
                spec = KERNEL_SPECS[kernel_name]
                for n in sizes:
                    group, cases, component_names = score_group(
                        spec, n, args, component_weight_overrides
                    )
                    prepared.append((spec, n, group, cases))
                    known_components.update(component_names)
        except (TypeError, ValueError) as error:
            parser.error(str(error))

        unknown_weights = sorted(
            set(component_weight_overrides) - known_components
        )
        if unknown_weights:
            parser.error(
                "weights were supplied for unknown objective components: "
                + ", ".join(unknown_weights)
            )

        report = {
            "experiment": "multi-kernel-traditional-layout-ranking",
            "configuration": expected_configuration,
            "score_mode": args.score_mode,
            "component_weight_overrides": component_weight_overrides,
            "variation_awareness": {
                "method": VARIATION_METHOD,
                "timing_interval": "inclusive minimum and maximum raw samples",
                "changes_raw_ranks": False,
            },
            "benchmark_run_order": [],
            "complete": bool(args.score_only),
            "runs": [group for _, _, group, _ in prepared],
        }
        write_reports(report, output_path, markdown_path)

    if not args.score_only:
        jobs: list[
            tuple[
                KernelSpec,
                int,
                LayoutCase,
                dict[str, object],
                dict[str, object],
            ]
        ] = []
        for spec, n, group, cases in prepared:
            records = group["results"]
            assert isinstance(records, list)
            records_by_name = {str(record["name"]): record for record in records}
            for case in cases:
                jobs.append((spec, n, case, group, records_by_name[case.name]))
        random.Random(args.seed).shuffle(jobs)
        jobs = [job for job in jobs if job[4]["timing"] is None]
        total_missing = len(jobs)
        if args.max_benchmarks is not None:
            jobs = jobs[: args.max_benchmarks]

        try:
            for index, (spec, n, case, group, record) in enumerate(jobs, 1):
                print(
                    f"[{index}/{len(jobs)}; {total_missing} missing] "
                    f"Benchmarking {spec.display_name} "
                    f"N={n} {case.name}...",
                    flush=True,
                )
                timing, command, stdout, stderr = benchmark_case(
                    spec, n, case, args
                )
                record["timing"] = asdict(timing)
                record["benchmark_command"] = command
                record["benchmark_stdout"] = stdout
                record["benchmark_stderr"] = stderr
                order_item = {
                    "kernel": spec.name,
                    "matrix_size": n,
                    "layout": case.name,
                }
                global_run_order = report["benchmark_run_order"]
                assert isinstance(global_run_order, list)
                global_run_order.append(order_item)
                group_order = group["benchmark_run_order"]
                assert isinstance(group_order, list)
                group_order.append(case.name)
                records = group["results"]
                assert isinstance(records, list)
                if all(item["timing"] is not None for item in records):
                    finalize_runtime_group(group)
                write_reports(report, output_path, markdown_path)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        all_complete = all(
            record["timing"] is not None
            for _, _, group, _ in prepared
            for record in group["results"]
        )
        report["complete"] = all_complete
        write_reports(report, output_path, markdown_path)

    print_summary(report)
    print(f"\nWrote {output_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
