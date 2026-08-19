#!/usr/bin/env python3
r"""Compare RELAY layout scores with five representative GPU kernels.

The experiment applies the same canonical layout to every target matrix in a
kernel and repeats the comparison for several square matrix sizes. The suite
contains global, square-tiled, rectangular-tiled, interleaved, and complete
8x8 and 8x16 canonical inner-word families.
Canonical words list physical element-address bits from low to high. Non-target
vector operands retain their fixed contiguous layouts.

Runtime ranks use exact sample medians.  Variation-aware metrics do not change
those ranks: they only check whether a score rank lies within the plausible
runtime-rank interval implied by each layout's observed sample range.
Completed reports also evaluate the score Pareto frontier through oracle
regret, epsilon-optimal coverage, retained fraction, purity, enrichment, and a
size-matched random baseline, plus top-k regret for the selected scalar score.
Fine-locality-gated frontiers, exact-score runtime spread, and randomized tau
weight robustness are reported alongside the primary frontier.

Scoring runs without a GPU.  Runtime measurement must run in a GPU allocation::

    flux run -n1 -g1 -t 5m -q pdebug \
        .venv/bin/python experiments/layout_ranking.py \
        --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942

All scores, runtimes, and ranks are ascending costs; lower is better.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import ModuleType
from typing import Literal, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernels.atax import problem as atax_problem
from kernels.gemm import problem as gemm_problem
from kernels.gesummv import problem as gesummv_problem
from kernels.mvt import problem as mvt_problem
from kernels.syrk import problem as syrk_problem
from experiments.frontier_analysis import (
    analyze_frontier_group,
    analyze_frontier_report,
    analyze_score_equivalence,
    analyze_tau_weight_robustness,
    render_frontier_plots,
)
from relay import (
    SCORE_MODES,
    CanonicalLayout,
    LayoutScore,
    MatrixSpec,
    ScoreMode,
    canonical_layout_from_word,
    pareto_frontier,
    row_major_layout,
    score_layouts,
    score_to_dict,
)
from relay.layouts import Layout
from relay.objectives import build_objectives


DEFAULT_SIZES = (256, 512, 1024)
VARIATION_METHOD = "observed-sample-range-rank-bounds"
PARETO_FINE_COMPONENT = "wave_load.64B"
PARETO_OBJECTIVES = (
    "wave_load.64B.raw-region-count",
    "peak-normalized-excess",
    "weighted-normalized-excess",
    "codegen-runs",
    "codegen-xors",
)
FINE_LOCALITY_GATED_DELTAS = (0.0, 0.01, 0.05, 0.10)
FINE_LOCALITY_GATED_OBJECTIVES = PARETO_OBJECTIVES[1:]
RECTANGULAR_TILE_SHAPES = (
    (8, 16),
    (16, 8),
    (8, 32),
    (32, 8),
    (16, 32),
    (32, 16),
)


@dataclass(frozen=True)
class KernelSpec:
    """Problem and evaluator information for one kernel."""

    name: str
    display_name: str
    problem: ModuleType
    evaluator: Path
    evaluator_arrays: tuple[str, ...]
    block_style: Literal["1d", "2d"]


KERNEL_SPECS = {
    "atax": KernelSpec(
        "atax",
        "ATAX",
        atax_problem,
        REPOSITORY_ROOT / "kernels" / "atax" / "evaluate.py",
        ("A",),
        "1d",
    ),
    "gemm": KernelSpec(
        "gemm",
        "GEMM",
        gemm_problem,
        REPOSITORY_ROOT / "kernels" / "gemm" / "evaluate.py",
        ("A", "B", "C"),
        "2d",
    ),
    "gesummv": KernelSpec(
        "gesummv",
        "GESUMMV",
        gesummv_problem,
        REPOSITORY_ROOT / "kernels" / "gesummv" / "evaluate.py",
        ("A", "B"),
        "1d",
    ),
    "mvt": KernelSpec(
        "mvt",
        "MVT",
        mvt_problem,
        REPOSITORY_ROOT / "kernels" / "mvt" / "evaluate.py",
        ("A",),
        "1d",
    ),
    "syrk": KernelSpec(
        "syrk",
        "SYRK",
        syrk_problem,
        REPOSITORY_ROOT / "kernels" / "syrk" / "evaluate.py",
        ("A", "C"),
        "2d",
    ),
}


@dataclass(frozen=True)
class LayoutCase:
    """One canonical layout applied to every target matrix."""

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


def layout_cases(n: int) -> tuple[LayoutCase, ...]:
    """Return global, tiled, rectangular, and interleaved controls.

    Tile names use ``i x j`` dimensions. For each rectangular shape, the
    row-major inner word places the ``j`` bits low and the column-major word
    places the ``i`` bits low. Outer tiles remain row-major.
    """

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
    for i_extent, j_extent in RECTANGULAR_TILE_SHAPES:
        if i_extent > n or j_extent > n:
            continue
        i_bits = i_extent.bit_length() - 1
        j_bits = j_extent.bit_length() - 1
        prefix = f"tile{i_extent}x{j_extent}"
        cases.extend(
            (
                LayoutCase(
                    f"{prefix}_row_major",
                    "j" * j_bits + "i" * i_bits,
                ),
                LayoutCase(
                    f"{prefix}_column_major",
                    "i" * i_bits + "j" * j_bits,
                ),
            )
        )
    for tile_size in (16, 32):
        if tile_size > n:
            continue
        tile_bits = tile_size.bit_length() - 1
        cases.append(
            LayoutCase(
                f"tile{tile_size}_interleaved",
                "ji" * tile_bits,
            )
        )
    existing_words = {case.word for case in cases}
    for i_bits, j_bits, label in (
        (3, 3, "tile8x8"),
        (3, 4, "tile8x16"),
    ):
        if 1 << i_bits > n or 1 << j_bits > n:
            continue
        for i_positions in combinations(range(i_bits + j_bits), i_bits):
            positions = set(i_positions)
            word = "".join(
                "i" if position in positions else "j"
                for position in range(i_bits + j_bits)
            )
            if word in existing_words:
                continue
            cases.append(LayoutCase(f"{label}_canonical_{word}", word))
            existing_words.add(word)
    return tuple(cases)


def selected_layout_cases(
    n: int, names: Sequence[str] | None = None
) -> tuple[LayoutCase, ...]:
    """Return all cases or an explicitly ordered named subset."""

    available = layout_cases(n)
    if not names:
        return available
    by_name = {case.name: case for case in available}
    unknown = [name for name in dict.fromkeys(names) if name not in by_name]
    if unknown:
        raise ValueError(
            f"N={n} does not provide layout cases: {', '.join(unknown)}"
        )
    return tuple(by_name[name] for name in dict.fromkeys(names))


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
    """Parse the correctness and timing block shared by the HIP evaluators."""

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


def notes_pareto_frontier(
    scores: Mapping[str, LayoutScore],
) -> dict[str, object]:
    """Build the notes-aligned locality/codegen cost frontier."""

    frontier = pareto_frontier(
        scores,
        objectives={
            PARETO_OBJECTIVES[0]: (
                lambda score: score.component(
                    PARETO_FINE_COMPONENT
                ).raw_region_count
            ),
            PARETO_OBJECTIVES[1]: (
                lambda score: score.peak_normalized_excess
            ),
            PARETO_OBJECTIVES[2]: (
                lambda score: score.weighted_normalized_excess
            ),
            PARETO_OBJECTIVES[3]: lambda score: float(score.codegen.runs),
            PARETO_OBJECTIVES[4]: lambda score: float(score.codegen.xors),
        },
    )
    return {
        "dominance": (
            "all objectives are minimized; a member has no competitor that "
            "is no greater in every objective and smaller in at least one"
        ),
        "objectives": [
            {
                "name": PARETO_OBJECTIVES[0],
                "definition": (
                    "Q for the grounded wave_load.64B objective component"
                ),
            },
            {
                "name": PARETO_OBJECTIVES[1],
                "definition": "J_peak over active objective components",
            },
            {
                "name": PARETO_OBJECTIVES[2],
                "definition": "J_area = sum(tau * normalized excess)",
            },
            {
                "name": PARETO_OBJECTIVES[3],
                "definition": "sum of address-expression runs over target arrays",
            },
            {
                "name": PARETO_OBJECTIVES[4],
                "definition": "sum of address-expression XORs over target arrays",
            },
        ],
        "members": [
            {
                "name": point.name,
                "values": dict(zip(frontier.objectives, point.values)),
            }
            for point in frontier.points
        ],
    }


def fine_locality_gated_frontiers(
    scores: Mapping[str, LayoutScore],
) -> list[dict[str, object]]:
    """Gate on near-minimal fine locality, then Pareto-filter other costs."""

    fine_values = {
        name: score.component(PARETO_FINE_COMPONENT).raw_region_count
        for name, score in scores.items()
    }
    minimum = min(fine_values.values())
    frontiers = []
    for delta in FINE_LOCALITY_GATED_DELTAS:
        limit = (1.0 + delta) * minimum
        eligible = {
            name: score
            for name, score in scores.items()
            if fine_values[name] <= limit
        }
        frontier = pareto_frontier(
            eligible,
            objectives={
                FINE_LOCALITY_GATED_OBJECTIVES[0]: (
                    lambda score: score.peak_normalized_excess
                ),
                FINE_LOCALITY_GATED_OBJECTIVES[1]: (
                    lambda score: score.weighted_normalized_excess
                ),
                FINE_LOCALITY_GATED_OBJECTIVES[2]: (
                    lambda score: float(score.codegen.runs)
                ),
                FINE_LOCALITY_GATED_OBJECTIVES[3]: (
                    lambda score: float(score.codegen.xors)
                ),
            },
        )
        frontiers.append(
            {
                "delta": delta,
                "delta_percent": 100.0 * delta,
                "fine_minimum": minimum,
                "fine_limit": limit,
                "eligible_count": len(eligible),
                "objectives": list(frontier.objectives),
                "members": [
                    {
                        "name": point.name,
                        "values": dict(zip(frontier.objectives, point.values)),
                    }
                    for point in frontier.points
                ],
                "runtime_analysis": None,
            }
        )
    return frontiers


def _problem_config(
    spec: KernelSpec, n: int, args: argparse.Namespace
) -> tuple[object, object]:
    if spec.block_style == "2d":
        block = (args.block_x, args.block_y, 1)
        return spec.problem.build_config(problem_size=n, block_size=block), list(block)
    block = args.block_size
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
        help="kernel to include; repeat as needed (default: all five)",
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
        "--layout-case",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "layout case to include; repeat for an ordered subset "
            "(default: all documented cases)"
        ),
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
        "--block-size",
        type=positive_integer,
        default=128,
        help="workgroup size for one-dimensional kernels (default: %(default)s)",
    )
    parser.add_argument(
        "--compiler",
        default="hipcc",
        help="HIP compiler command passed to every evaluator (default: %(default)s)",
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
        "--prepare-checkpoint",
        action="store_true",
        help=(
            "compute scores and write an incomplete benchmark checkpoint; "
            "resume it inside a GPU allocation"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume missing benchmarks from an existing compatible JSON report",
    )
    parser.add_argument(
        "--reuse-timings",
        type=Path,
        action="append",
        default=None,
        metavar="JSON",
        help=(
            "score the current objective model and attach matching timings "
            "from completed reports instead of running evaluators; repeat "
            "to combine disjoint reports"
        ),
    )
    parser.add_argument(
        "--seed-timings",
        type=Path,
        action="append",
        default=None,
        metavar="JSON",
        help=(
            "pre-fill matching completed layouts from an older report and "
            "benchmark only the remaining layouts; repeat for disjoint reports"
        ),
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
        "--plots-dir",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help=(
            "frontier-analysis plot directory (default: "
            "<output stem>_plots beside the JSON report)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used to randomize all benchmark jobs (default: %(default)s)",
    )
    parser.add_argument(
        "--tau-perturbation-trials",
        type=positive_integer,
        default=128,
        help=(
            "random tau-weight ablations per completed kernel/size "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tau-perturbation-seed",
        type=int,
        default=0,
        help="seed for tau-weight ablations (default: %(default)s)",
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

    cases = selected_layout_cases(n, args.layout_case)
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
    scores_by_name: dict[str, LayoutScore] = {}
    print(f"Scoring {len(cases)} layout controls...", flush=True)
    for case in cases:
        layouts = layouts_for_case(case, matrices)
        score = score_layouts(
            matrices,
            components,
            layouts,
            component_weights=applied_weights,
        )
        scores_by_name[case.name] = score
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
                "pareto_frontier_member": None,
                "fine_locality_gated_frontier_deltas": [],
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

    frontier = notes_pareto_frontier(scores_by_name)
    gated_frontiers = fine_locality_gated_frontiers(scores_by_name)
    frontier_members = {
        str(member["name"])
        for member in frontier["members"]
    }
    for record in records:
        record["pareto_frontier_member"] = record["name"] in frontier_members
        record["fine_locality_gated_frontier_deltas"] = [
            gated["delta"]
            for gated in gated_frontiers
            if record["name"]
            in {member["name"] for member in gated["members"]}
        ]

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
        "pareto_frontier": frontier,
        "fine_locality_gated_frontiers": gated_frontiers,
        "score_equivalence_analysis": None,
        "benchmark_run_order": [],
        "variation_aware_rank_metrics": {
            aggregate_mode: None for aggregate_mode in SCORE_MODES
        },
        "frontier_runtime_analysis": None,
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
    group["frontier_runtime_analysis"] = analyze_frontier_group(group)


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


def _frontier_scorecard_markdown(
    analysis: Mapping[str, object],
) -> list[str]:
    """Render aggregate frontier regret and candidate-generation metrics."""

    regrets = analysis["oracle_regret"]
    retained = analysis["retained_fraction"]
    frontier_sizes = analysis["frontier_size"]
    exact = analysis["exact_winner_coverage"]
    random_baseline = analysis["random_exact_winner_baseline"]
    assert isinstance(regrets, dict)
    assert isinstance(retained, dict)
    assert isinstance(frontier_sizes, dict)
    assert isinstance(exact, dict)
    assert isinstance(random_baseline, dict)

    lines = [
        "## Frontier candidate-generation scorecard",
        "",
        "The frontier is evaluated as a retained candidate set. Oracle regret "
        "is the best frontier median runtime divided by the best evaluated "
        "median runtime in the layout family, minus one. Runtime is "
        "not used to construct the "
        "frontier.",
        "",
        _markdown_table(
            ("Metric", "Mean", "Median", "Minimum", "Maximum"),
            (
                (
                    "Oracle regret",
                    f"{100.0 * float(regrets['mean']):.6f}%",
                    f"{100.0 * float(regrets['median']):.6f}%",
                    f"{100.0 * float(regrets['minimum']):.6f}%",
                    f"{100.0 * float(regrets['maximum']):.6f}%",
                ),
                (
                    "Retained fraction",
                    f"{100.0 * float(retained['mean']):.3f}%",
                    f"{100.0 * float(retained['median']):.3f}%",
                    f"{100.0 * float(retained['minimum']):.3f}%",
                    f"{100.0 * float(retained['maximum']):.3f}%",
                ),
                (
                    "Frontier size",
                    f"{float(frontier_sizes['mean']):.3f}",
                    f"{float(frontier_sizes['median']):.3f}",
                    f"{float(frontier_sizes['minimum']):.0f}",
                    f"{float(frontier_sizes['maximum']):.0f}",
                ),
            ),
        ),
        "",
        f"Exact-winner coverage is {exact['covered_instances']}/"
        f"{analysis['instance_count']} "
        f"({100.0 * float(exact['coverage']):.3f}%). A uniformly random subset "
        "with each frontier's size would cover "
        f"{float(random_baseline['expected_covered_instances']):.3f} instances "
        "in expectation; its Poisson-binomial probability of at least the "
        "observed number of exact hits is "
        f"{float(random_baseline['probability_at_least_observed_hits']):.6g}.",
        "",
        "### Retained fraction versus oracle regret",
        "",
    ]

    instances = analysis["instances"]
    assert isinstance(instances, list)
    instance_rows = []
    for instance in instances:
        assert isinstance(instance, dict)
        optimal_names = ", ".join(
            f"`{name}`" for name in instance["optimal_layouts"]
        )
        frontier_names = ", ".join(
            f"`{name}`" for name in instance["best_frontier_layouts"]
        )
        instance_rows.append(
            (
                str(instance["display_name"]),
                str(instance["matrix_size"]),
                f"{instance['frontier_size']}/{instance['layout_count']}",
                f"{100.0 * float(instance['retained_fraction']):.3f}%",
                optimal_names,
                f"{float(instance['optimal_runtime_ms']):.6f}",
                frontier_names,
                f"{float(instance['best_frontier_runtime_ms']):.6f}",
                f"{100.0 * float(instance['oracle_regret']):.6f}%",
            )
        )
    lines.extend(
        (
            _markdown_table(
                (
                    "Kernel",
                    "N",
                    "K/L",
                    "Retained",
                    "Measured optimum",
                    "Optimum ms",
                    "Best frontier",
                    "Frontier ms",
                    "Regret",
                ),
                instance_rows,
            ),
            "",
        )
    )

    plots = analysis.get("plots")
    if isinstance(plots, dict):
        retained_plot = plots.get("retained_fraction_vs_regret")
        if isinstance(retained_plot, dict):
            lines.extend(
                (
                    "![Retained fraction versus frontier regret]"
                    f"({retained_plot['markdown_path']})",
                    "",
                )
            )

    lines.extend(
        (
            "### Epsilon-optimal coverage, purity, and enrichment",
            "",
            "An epsilon-optimal layout has median runtime no greater than "
            "`(1 + epsilon)` times the measured optimum. Purity is the "
            "epsilon-optimal fraction of the frontier; enrichment divides "
            "that purity by the epsilon-optimal fraction of the full layout "
            "set.",
            "",
        )
    )
    epsilon_metrics = analysis["epsilon_optimal"]
    assert isinstance(epsilon_metrics, list)
    epsilon_rows = []
    for metric in epsilon_metrics:
        assert isinstance(metric, dict)
        purity = metric["purity"]
        enrichment = metric["enrichment"]
        assert isinstance(purity, dict)
        assert isinstance(enrichment, dict)
        epsilon_rows.append(
            (
                f"{float(metric['epsilon_percent']):.2f}%",
                f"{metric['covered_instances']}/{analysis['instance_count']}",
                f"{100.0 * float(metric['coverage']):.3f}%",
                f"{100.0 * float(metric['random_expected_coverage']):.3f}%",
                f"{100.0 * float(purity['mean']):.3f}%",
                f"{100.0 * float(purity['median']):.3f}%",
                f"{float(enrichment['mean']):.3f}x",
                f"{float(enrichment['median']):.3f}x",
            )
        )
    lines.extend(
        (
            _markdown_table(
                (
                    "Epsilon",
                    "Covered",
                    "Coverage",
                    "Random coverage",
                    "Mean purity",
                    "Median purity",
                    "Mean enrichment",
                    "Median enrichment",
                ),
                epsilon_rows,
            ),
            "",
        )
    )
    if isinstance(plots, dict):
        for name, alt_text in (
            ("epsilon_optimal_coverage", "Epsilon-optimal frontier coverage"),
            ("purity_and_enrichment", "Frontier purity and enrichment"),
        ):
            plot = plots.get(name)
            if isinstance(plot, dict):
                lines.extend(
                    (f"![{alt_text}]({plot['markdown_path']})", "")
                )

    lines.extend(
        (
            "### Top-k scalar-score regret",
            "",
            "For an exact candidate budget `k`, layouts are ordered by the "
            "selected scalar score and then by layout name to break exact "
            "ties deterministically. The reported regret uses the fastest "
            "measured layout among those `k` candidates.",
            "",
        )
    )
    top_k = analysis["top_k"]
    assert isinstance(top_k, list)
    maximum_k = int(top_k[-1]["k"])
    checkpoints = sorted({1, 2, 4, 8, 16, maximum_k})
    top_k_rows = []
    for k in checkpoints:
        if k > maximum_k:
            continue
        metric = top_k[k - 1]
        regret = metric["regret"]
        assert isinstance(regret, dict)
        top_k_rows.append(
            (
                str(k),
                f"{100.0 * float(regret['median']):.6f}%",
                f"{100.0 * float(regret['mean']):.6f}%",
                f"{100.0 * float(regret['maximum']):.6f}%",
            )
        )
    lines.extend(
        (
            _markdown_table(
                ("k", "Median regret", "Mean regret", "Maximum regret"),
                top_k_rows,
            ),
            "",
        )
    )
    if isinstance(plots, dict):
        top_k_plot = plots.get("top_k_regret")
        if isinstance(top_k_plot, dict):
            lines.extend(
                (
                    f"![Top-k scalar-score regret]"
                    f"({top_k_plot['markdown_path']})",
                    "",
                )
            )
    robustness = analysis.get("tau_weight_robustness")
    if isinstance(robustness, dict):
        lines.extend(
            (
                "### Tau-weight robustness",
                "",
                "Each trial independently multiplies every nonzero tau by one "
                "of `0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.5`, rebuilds the five-cost "
                "frontier, and evaluates its regret and retained fraction.",
                "",
            )
        )
        robustness_rows = []
        for instance in robustness["instances"]:
            regret = instance["oracle_regret"]
            retained = instance["retained_fraction"]
            robustness_rows.append(
                (
                    str(instance["display_name"]),
                    str(instance["matrix_size"]),
                    f"{100.0 * float(regret['median']):.6f}%",
                    f"{100.0 * float(regret['mean']):.6f}%",
                    f"{100.0 * float(regret['maximum']):.6f}%",
                    f"{100.0 * float(retained['median']):.3f}%",
                    f"{100.0 * float(retained['mean']):.3f}%",
                )
            )
        lines.extend(
            (
                _markdown_table(
                    (
                        "Kernel",
                        "N",
                        "Median regret",
                        "Mean regret",
                        "Max regret",
                        "Median retained",
                        "Mean retained",
                    ),
                    robustness_rows,
                ),
                "",
            )
        )
        if isinstance(plots, dict):
            plot = plots.get("tau_weight_robustness")
            if isinstance(plot, dict):
                lines.extend(
                    (
                        f"![Tau-weight robustness]({plot['markdown_path']})",
                        "",
                    )
                )
    return lines


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
        "Runs and XORs are separate address-code generation costs. They are "
        "included in the Pareto frontier but are not folded into the scalar "
        "locality score or score rank.",
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
    ]
    timing_sources = report.get("timing_sources")
    if isinstance(timing_sources, list):
        rendered_sources = ", ".join(f"`{source}`" for source in timing_sources)
        lines.extend(
            (
                "Runtime samples were reused from "
                f"{rendered_sources}; objective scores and all rank metrics "
                "were recomputed for this report.",
                "",
            )
        )
    seed_timing_sources = report.get("seed_timing_sources")
    if isinstance(seed_timing_sources, list):
        rendered_sources = ", ".join(
            f"`{source}`" for source in seed_timing_sources
        )
        lines.extend(
            (
                "Matching runtime samples were seeded from "
                f"{rendered_sources}; only newly added layouts were "
                "benchmarked in this run.",
                "",
            )
        )
    lines.extend(("## Summary", ""))

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
                str(len(group["pareto_frontier"]["members"])),
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
                    "Pareto layouts",
                    "Variation-aware rank accuracy",
                    "Mean rank error",
                ),
                summary_rows,
            ),
            "",
        )
    )

    frontier_analysis = report.get("frontier_analysis")
    if isinstance(frontier_analysis, dict):
        lines.extend(_frontier_scorecard_markdown(frontier_analysis))

    gated_analysis = report.get("fine_locality_gated_analysis")
    if isinstance(gated_analysis, list):
        gated_rows = []
        for item in gated_analysis:
            regret = item["oracle_regret"]
            retained = item["retained_fraction"]
            exact = item["exact_winner_coverage"]
            gated_rows.append(
                (
                    f"{float(item['delta_percent']):.0f}%",
                    f"{exact['covered_instances']}/{item['instance_count']}",
                    f"{100.0 * float(regret['median']):.6f}%",
                    f"{100.0 * float(regret['mean']):.6f}%",
                    f"{100.0 * float(regret['maximum']):.6f}%",
                    f"{100.0 * float(retained['mean']):.3f}%",
                )
            )
        lines.extend(
            (
                "## Fine-locality-gated frontier scorecard",
                "",
                "For each delta, candidates first satisfy `Q_fine <= "
                "(1 + delta) Q_fine*`; the eligible set is then Pareto-"
                "filtered over `(J_peak, J_area, runs, XORs)`.",
                "",
                _markdown_table(
                    (
                        "Delta",
                        "Exact winners",
                        "Median regret",
                        "Mean regret",
                        "Max regret",
                        "Mean retained",
                    ),
                    gated_rows,
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
                "### Score Pareto frontier",
                "",
            )
        )
        frontier = group["pareto_frontier"]
        assert isinstance(frontier, dict)
        frontier_members = frontier["members"]
        assert isinstance(frontier_members, list)
        frontier_rows = [
            [
                f"`{member['name']}`",
                f"{float(member['values'][PARETO_OBJECTIVES[0]]):.6g}",
                f"{float(member['values'][PARETO_OBJECTIVES[1]]):.6g}",
                f"{float(member['values'][PARETO_OBJECTIVES[2]]):.6g}",
                f"{float(member['values'][PARETO_OBJECTIVES[3]]):.6g}",
                f"{float(member['values'][PARETO_OBJECTIVES[4]]):.6g}",
            ]
            for member in frontier_members
        ]
        lines.extend(
            (
                "This is the exact non-dominated set over the notes-aligned "
                "locality vector plus separate codegen run and XOR costs. "
                "Runtime is not a Pareto objective.",
                "",
                _markdown_table(
                    (
                        "Layout",
                        "Q fine",
                        "J peak",
                        "J area",
                        "Runs",
                        "XORs",
                    ),
                    frontier_rows,
                ),
                "",
                "### Fine-locality-gated frontiers",
                "",
            )
        )
        gated_rows = []
        for gated in group["fine_locality_gated_frontiers"]:
            runtime_analysis = gated.get("runtime_analysis")
            regret = (
                "—"
                if not isinstance(runtime_analysis, dict)
                else f"{100.0 * float(runtime_analysis['oracle_regret']):.6f}%"
            )
            gated_rows.append(
                (
                    f"{float(gated['delta_percent']):.0f}%",
                    f"{float(gated['fine_limit']):.6g}",
                    str(gated["eligible_count"]),
                    str(len(gated["members"])),
                    ", ".join(
                        f"`{member['name']}`" for member in gated["members"]
                    ),
                    regret,
                )
            )
        lines.extend(
            (
                _markdown_table(
                    (
                        "Delta",
                        "Q fine limit",
                        "Eligible",
                        "Frontier size",
                        "Members",
                        "Regret",
                    ),
                    gated_rows,
                ),
                "",
            )
        )
        equivalence = group.get("score_equivalence_analysis")
        if isinstance(equivalence, dict):
            equivalence_rows = []
            vector_entries = [
                ("Main five-cost", equivalence["main_frontier_vector"]),
                *[
                    (
                        f"Gated delta={float(item['delta_percent']):.0f}%",
                        item,
                    )
                    for item in equivalence["fine_locality_gated_vectors"]
                ],
            ]
            for label, item in vector_entries:
                spread = item["non_singleton_runtime_spread"]
                equivalence_rows.append(
                    (
                        label,
                        str(item["group_count"]),
                        str(item["non_singleton_group_count"]),
                        str(item["layouts_in_non_singleton_groups"]),
                        "—"
                        if spread is None
                        else f"{100.0 * float(spread['median']):.6f}%",
                        "—"
                        if spread is None
                        else f"{100.0 * float(spread['mean']):.6f}%",
                        "—"
                        if spread is None
                        else f"{100.0 * float(spread['maximum']):.6f}%",
                    )
                )
            lines.extend(
                (
                    "### Runtime spread within score-equivalent groups",
                    "",
                    "Score equality is exact across every coordinate. Spread "
                    "is `max(median runtime) / min(median runtime) - 1`; "
                    "singleton groups are excluded from the summaries.",
                    "",
                    _markdown_table(
                        (
                            "Vector",
                            "Groups",
                            "Non-singletons",
                            "Layouts in non-singletons",
                            "Median spread",
                            "Mean spread",
                            "Max spread",
                        ),
                        equivalence_rows,
                    ),
                    "",
                )
            )
        lines.extend(("### Layout ranks", ""))
        ordered = sorted(
            records,
            key=lambda record: (
                float(record["score_rank"]),
                str(record["name"]),
            ),
        )
        group_complete = all(record["timing"] is not None for record in records)
        if score_only:
            headers = (
                "Score rank",
                "Layout",
                "Word (low→high)",
                "Score",
                "Runs",
                "XORs",
            )
            rows = [
                [
                    _format_rank(record["score_rank"]),
                    f"`{record['name']}`",
                    f"`{record['word']}`",
                    f"{float(record['selected_score']):.6g}",
                    str(record["score"]["codegen"]["runs"]),
                    str(record["score"]["codegen"]["xors"]),
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
                "Runs",
                "XORs",
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
                            str(record["score"]["codegen"]["runs"]),
                            str(record["score"]["codegen"]["xors"]),
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
                        str(record["score"]["codegen"]["runs"]),
                        str(record["score"]["codegen"]["xors"]),
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
    print("\nRELAY multi-kernel layout experiment")
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
    frontier_analysis = report.get("frontier_analysis")
    if isinstance(frontier_analysis, dict):
        regret = frontier_analysis["oracle_regret"]
        exact = frontier_analysis["exact_winner_coverage"]
        assert isinstance(regret, dict)
        assert isinstance(exact, dict)
        print(
            "  frontier oracle regret: "
            f"median={100.0 * float(regret['median']):.6f}%, "
            f"mean={100.0 * float(regret['mean']):.6f}%, "
            f"max={100.0 * float(regret['maximum']):.6f}%"
        )
        print(
            "  frontier exact-winner coverage: "
            f"{exact['covered_instances']}/{frontier_analysis['instance_count']}"
        )


def _configuration(
    args: argparse.Namespace,
    kernel_names: Sequence[str],
    sizes: Sequence[int],
) -> dict[str, object]:
    return {
        "kernels": list(kernel_names),
        "matrix_sizes": list(sizes),
        "layout_cases": list(args.layout_case) if args.layout_case else None,
        "two_dimensional_block": [args.block_x, args.block_y, 1],
        "one_dimensional_block_size": args.block_size,
        "samples": args.samples,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "device": args.device,
        "compiler": args.compiler,
        "arch": args.arch,
        "score_only": args.score_only,
        "reuse_timings": (
            [
                str(path.expanduser().resolve())
                for path in args.reuse_timings
            ]
            if args.reuse_timings
            else None
        ),
        "seed": args.seed,
        "tau_perturbation_trials": args.tau_perturbation_trials,
        "tau_perturbation_seed": args.tau_perturbation_seed,
    }


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def update_frontier_analysis(report: dict[str, object]) -> None:
    """Refresh per-run and aggregate frontier metrics from stored timings."""

    groups = report.get("runs")
    if not isinstance(groups, list):
        raise ValueError("experiment report has no run list")
    all_complete = bool(groups)
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("experiment report contains an invalid run")
        records = group.get("results")
        if not isinstance(records, list):
            raise ValueError("experiment run has no result list")
        complete = bool(records) and all(
            isinstance(record, dict) and record.get("timing") is not None
            for record in records
        )
        if complete:
            group["frontier_runtime_analysis"] = analyze_frontier_group(group)
            group["score_equivalence_analysis"] = analyze_score_equivalence(
                group
            )
            gated_frontiers = group.get("fine_locality_gated_frontiers")
            if not isinstance(gated_frontiers, list):
                raise ValueError("experiment run has no gated frontiers")
            records_by_name = {
                str(record["name"]): record for record in records
            }
            for gated in gated_frontiers:
                if not isinstance(gated, dict):
                    raise ValueError("experiment run has an invalid gated frontier")
                member_names = {
                    str(member["name"]) for member in gated["members"]
                }
                gated_group = dict(group)
                gated_group["results"] = [
                    {
                        **records_by_name[str(record["name"])],
                        "pareto_frontier_member": (
                            str(record["name"]) in member_names
                        ),
                    }
                    for record in records
                ]
                gated["runtime_analysis"] = analyze_frontier_group(gated_group)
        else:
            group["frontier_runtime_analysis"] = None
            group["score_equivalence_analysis"] = None
            all_complete = False

    if not all_complete:
        report["frontier_analysis"] = None
        report["fine_locality_gated_analysis"] = None
        return

    analysis = analyze_frontier_report(groups)
    configuration = report.get("configuration")
    assert isinstance(configuration, dict)
    robustness = analyze_tau_weight_robustness(
        groups,
        trials=int(configuration.get("tau_perturbation_trials", 128)),
        seed=int(configuration.get("tau_perturbation_seed", 0)),
    )
    analysis["tau_weight_robustness"] = robustness
    gated_analyses = []
    for index, delta in enumerate(FINE_LOCALITY_GATED_DELTAS):
        gated_groups = []
        for group in groups:
            gated = group["fine_locality_gated_frontiers"][index]
            member_names = {
                str(member["name"]) for member in gated["members"]
            }
            gated_group = dict(group)
            gated_group["results"] = [
                {
                    **record,
                    "pareto_frontier_member": str(record["name"]) in member_names,
                }
                for record in group["results"]
            ]
            gated_groups.append(gated_group)
        gated_analysis = analyze_frontier_report(gated_groups)
        gated_analysis["delta"] = delta
        gated_analysis["delta_percent"] = 100.0 * delta
        gated_analyses.append(gated_analysis)
    report["fine_locality_gated_analysis"] = gated_analyses
    report["frontier_analysis"] = analysis


def write_reports(
    report: dict[str, object],
    output_path: Path,
    markdown_path: Path,
    plots_directory: Path,
) -> None:
    """Write a checkpoint, Markdown scorecard, and completed-run plots."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    update_frontier_analysis(report)
    analysis = report["frontier_analysis"]
    if isinstance(analysis, dict) and report.get("complete") is True:
        paths = render_frontier_plots(analysis, plots_directory)
        analysis["plots"] = {
            name: {
                "path": str(path.resolve()),
                "markdown_path": Path(
                    os.path.relpath(path, markdown_path.parent)
                ).as_posix(),
            }
            for name, path in paths.items()
        }
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


def reuse_report_timings(
    report: dict[str, object],
    sources: Sequence[Mapping[str, object]],
    *,
    require_all: bool = True,
) -> None:
    """Attach matching timing records and recompute completed-run metrics."""

    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("target report has no experiment configuration")
    timing_fields = (
        "samples",
        "iterations",
        "warmup",
        "device",
        "compiler",
        "arch",
    )
    source_records: dict[tuple[str, int, str, str], Mapping[str, object]] = {}
    source_groups: dict[tuple[str, int], Mapping[str, object]] = {}
    source_orders: list[object] = []
    for source in sources:
        source_configuration = source.get("configuration")
        if not isinstance(source_configuration, dict):
            raise ValueError("timing source has no experiment configuration")
        mismatched_fields = [
            field
            for field in timing_fields
            if configuration.get(field) != source_configuration.get(field)
        ]
        if mismatched_fields:
            raise ValueError(
                "timing source uses different benchmark settings: "
                + ", ".join(mismatched_fields)
            )
        source_runs = source.get("runs")
        if not isinstance(source_runs, list):
            raise ValueError("timing source has no run list")
        for group in source_runs:
            if not isinstance(group, dict) or not isinstance(
                group.get("results"), list
            ):
                raise ValueError("timing source contains an invalid run")
            group_key = (str(group["kernel"]), int(group["matrix_size"]))
            if group_key in source_groups:
                raise ValueError(
                    "timing sources contain duplicate runs for "
                    f"{group_key[0]} N={group_key[1]}"
                )
            source_groups[group_key] = group
            for record in group["results"]:
                if not isinstance(record, dict):
                    raise ValueError("timing source contains an invalid result")
                key = (
                    str(group["kernel"]),
                    int(group["matrix_size"]),
                    str(record["name"]),
                    str(record["word"]),
                )
                if record.get("timing") is not None:
                    source_records[key] = record
        source_order = source.get("benchmark_run_order", [])
        if not isinstance(source_order, list):
            raise ValueError(
                "timing source contains an invalid benchmark run order"
            )
        source_orders.extend(source_order)

    runs = report["runs"]
    assert isinstance(runs, list)
    target_record_keys: set[tuple[str, int, str, str]] = set()
    for group in runs:
        assert isinstance(group, dict)
        group_key = (str(group["kernel"]), int(group["matrix_size"]))
        source_group = source_groups.get(group_key)
        if source_group is None:
            if require_all:
                raise ValueError(
                    "timing source has no matching run for "
                    f"{group_key[0]} N={group_key[1]}"
                )
            continue
        if group.get("block") != source_group.get("block"):
            raise ValueError(
                "timing source uses a different workgroup for "
                f"{group_key[0]} N={group_key[1]}"
            )
        records = group["results"]
        assert isinstance(records, list)
        target_names = {str(record["name"]) for record in records}
        for record in records:
            assert isinstance(record, dict)
            key = (
                str(group["kernel"]),
                int(group["matrix_size"]),
                str(record["name"]),
                str(record["word"]),
            )
            target_record_keys.add(key)
            source_record = source_records.get(key)
            if source_record is None:
                if require_all:
                    raise ValueError(
                        "timing source has no completed matching benchmark for "
                        f"{key[0]} N={key[1]} {key[2]}"
                    )
                continue
            for field in (
                "timing",
                "benchmark_command",
                "benchmark_stdout",
                "benchmark_stderr",
            ):
                record[field] = source_record.get(field)
        source_group_order = source_group.get("benchmark_run_order", [])
        if not isinstance(source_group_order, list):
            raise ValueError("timing source contains an invalid run order")
        group["benchmark_run_order"] = [
            str(name) for name in source_group_order if str(name) in target_names
        ]
        if all(record["timing"] is not None for record in records):
            finalize_runtime_group(group)

    filtered_order = []
    for item in source_orders:
        if not isinstance(item, dict):
            raise ValueError("timing source contains an invalid benchmark run order")
        key_prefix = (
            str(item.get("kernel")),
            int(item.get("matrix_size")),
            str(item.get("layout")),
        )
        if any(key[:3] == key_prefix for key in target_record_keys):
            filtered_order.append(dict(item))
    report["benchmark_run_order"] = filtered_order
    report["complete"] = all(
        record["timing"] is not None
        for group in runs
        for record in group["results"]
    )
    if require_all and report["complete"] is not True:
        raise ValueError("timing sources did not complete the target report")


def run(argv: list[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    try:
        component_weight_overrides = parse_component_weights(
            args.component_weight
        )
        kernel_names = tuple(dict.fromkeys(args.kernel or KERNEL_SPECS))
        sizes = tuple(dict.fromkeys(args.size or DEFAULT_SIZES))
        for n in sizes:
            selected_layout_cases(n, args.layout_case)
        if args.block_x * args.block_y > 1024:
            raise ValueError(
                "--block-x times --block-y must not exceed 1024"
            )
        if args.block_size > 1024:
            raise ValueError("--block-size must not exceed 1024")
        if args.resume and args.score_only:
            raise ValueError("--resume cannot be combined with --score-only")
        if args.prepare_checkpoint and args.score_only:
            raise ValueError(
                "--prepare-checkpoint cannot be combined with --score-only"
            )
        if args.prepare_checkpoint and args.resume:
            raise ValueError(
                "--prepare-checkpoint cannot be combined with --resume"
            )
        if args.prepare_checkpoint and args.reuse_timings is not None:
            raise ValueError(
                "--prepare-checkpoint cannot be combined with --reuse-timings"
            )
        if args.resume and args.seed_timings is not None:
            raise ValueError("--resume cannot be combined with --seed-timings")
        if args.reuse_timings is not None and args.seed_timings is not None:
            raise ValueError(
                "--reuse-timings cannot be combined with --seed-timings"
            )
        if args.score_only and args.seed_timings is not None:
            raise ValueError("--score-only cannot be combined with --seed-timings")
        if args.prepare_checkpoint and args.max_benchmarks is not None:
            raise ValueError(
                "--prepare-checkpoint cannot be combined with --max-benchmarks"
            )
        if args.resume and args.reuse_timings is not None:
            raise ValueError("--resume cannot be combined with --reuse-timings")
        if args.score_only and args.reuse_timings is not None:
            raise ValueError("--score-only cannot be combined with --reuse-timings")
        if args.max_benchmarks is not None and args.score_only:
            raise ValueError("--max-benchmarks cannot be combined with --score-only")
        if args.max_benchmarks is not None and args.reuse_timings is not None:
            raise ValueError(
                "--max-benchmarks cannot be combined with --reuse-timings"
            )
    except ValueError as error:
        parser.error(str(error))

    output_path = args.output.expanduser().resolve()
    markdown_path = (
        args.markdown.expanduser().resolve()
        if args.markdown is not None
        else output_path.with_suffix(".md")
    )
    plots_directory = (
        args.plots_dir.expanduser().resolve()
        if args.plots_dir is not None
        else output_path.with_name(output_path.stem + "_plots")
    )
    expected_configuration = _configuration(args, kernel_names, sizes)

    if args.resume:
        if not output_path.exists():
            parser.error(f"--resume report does not exist: {output_path}")
        try:
            report = json.loads(output_path.read_text())
            if report.get("experiment") != "multi-kernel-layout-ranking":
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
            "experiment": "multi-kernel-layout-ranking",
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
            "frontier_analysis": None,
            "fine_locality_gated_analysis": None,
            "runs": [group for _, _, group, _ in prepared],
        }
        if args.reuse_timings is not None:
            timing_paths = [path.expanduser().resolve() for path in args.reuse_timings]
            try:
                timing_sources = [
                    json.loads(path.read_text()) for path in timing_paths
                ]
                reuse_report_timings(report, timing_sources)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                parser.error(str(error))
            report["timing_sources"] = [str(path) for path in timing_paths]
        elif args.seed_timings is not None:
            timing_paths = [
                path.expanduser().resolve() for path in args.seed_timings
            ]
            try:
                timing_sources = [
                    json.loads(path.read_text()) for path in timing_paths
                ]
                reuse_report_timings(
                    report, timing_sources, require_all=False
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                parser.error(str(error))
            report["seed_timing_sources"] = [str(path) for path in timing_paths]
        write_reports(report, output_path, markdown_path, plots_directory)

    if (
        not args.score_only
        and not args.prepare_checkpoint
        and args.reuse_timings is None
    ):
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
                write_reports(
                    report,
                    output_path,
                    markdown_path,
                    plots_directory,
                )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        all_complete = all(
            record["timing"] is not None
            for _, _, group, _ in prepared
            for record in group["results"]
        )
        report["complete"] = all_complete
        write_reports(report, output_path, markdown_path, plots_directory)

    print_summary(report)
    print(f"\nWrote {output_path}")
    print(f"Wrote {markdown_path}")
    if isinstance(report.get("frontier_analysis"), dict):
        print(f"Wrote plots under {plots_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
