#!/usr/bin/env python3
"""Compare RELAY scores with measured runtimes for traditional GEMM layouts.

The experiment applies one conventional layout uniformly to A, B, and C.  A
canonical word lists physical element-address bits from least significant to
most significant.  For example, ``jjjjiiii`` is row-major within each 16x16
tile, while the tile grid remains row-major.

Scoring runs anywhere.  Runtime measurement needs a GPU allocation and a HIP
compiler.  On an MI300A system, a typical invocation is::

    module load rocm/7.0.2
    flux run -n1 -g1 -t 5m -q pdebug \
        .venv/bin/python experiments/gemm_layout_ranking.py --n 1024 \
        --arch gfx942 --output results/gemm-layout-ranking.json

All reported scores and runtimes are costs: lower values are better.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Mapping, Sequence

# Direct execution sets ``sys.path[0]`` to ``experiments/``.  Add the checkout
# root so the sibling ``kernels`` namespace and local ``relay`` package resolve
# without requiring an editable installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernels.gemm import problem as gemm_problem
from relay import (
    SCORE_MODES,
    MatrixSpec,
    ScoreMode,
    canonical_layout_from_word,
    score_layouts,
    score_to_dict,
)
from relay.layouts import Layout
from relay.objectives import build_objectives


EVALUATOR = REPOSITORY_ROOT / "kernels" / "gemm" / "evaluate.py"


@dataclass(frozen=True)
class LayoutCase:
    """One canonical word applied uniformly to GEMM's A, B, and C arrays."""

    name: str
    word: str


@dataclass(frozen=True)
class TimingResult:
    """Kernel-only statistics printed by ``kernels/gemm/evaluate.py``."""

    device: str
    median_ms: float
    mean_ms: float
    min_ms: float
    sd_ms: float
    gflops: float
    samples_ms: tuple[float, ...]


def traditional_layout_cases(n: int) -> tuple[LayoutCase, ...]:
    """Return the explicit uniform row, column, and square-tile controls."""

    if n < 2 or n & (n - 1):
        raise ValueError("--n must be a power of two greater than one")

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
    """Rank ascending values from one, assigning tied values their mean rank."""

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


def spearman_correlation(
    left_values: Sequence[float], right_values: Sequence[float]
) -> float | None:
    """Return Spearman's rho using average ranks, or ``None`` if undefined."""

    if len(left_values) != len(right_values):
        raise ValueError("Spearman inputs must have the same length")
    if not left_values:
        return None

    left_ranks = average_tie_ranks(left_values)
    right_ranks = average_tie_ranks(right_values)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    left_centered = [value - left_mean for value in left_ranks]
    right_centered = [value - right_mean for value in right_ranks]
    numerator = sum(
        left * right for left, right in zip(left_centered, right_centered)
    )
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return None if denominator == 0.0 else numerator / denominator


def parse_evaluator_output(output: str) -> TimingResult:
    """Parse the stable correctness and timing block emitted by the HIP runner."""

    lines = output.splitlines()
    if not any(line.startswith("Correctness: PASS") for line in lines):
        raise ValueError("GEMM evaluator did not report Correctness: PASS")

    header = "median_ms  mean_ms  min_ms  sd_ms  GFLOP/s"
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.strip() == header
        )
        result_fields = lines[header_index + 1].split()
    except (StopIteration, IndexError) as error:
        raise ValueError("GEMM evaluator output has no timing result row") from error
    if len(result_fields) != 5:
        raise ValueError("GEMM evaluator timing row must contain five numbers")
    try:
        median_ms, mean_ms, min_ms, sd_ms, gflops = map(float, result_fields)
    except ValueError as error:
        raise ValueError("GEMM evaluator timing row contains a non-number") from error

    sample_prefix = "Samples (ms):"
    sample_line = next(
        (line for line in lines if line.startswith(sample_prefix)), None
    )
    if sample_line is None:
        raise ValueError("GEMM evaluator output has no timing samples")
    try:
        samples_ms = tuple(
            float(value) for value in sample_line[len(sample_prefix) :].split()
        )
    except ValueError as error:
        raise ValueError("GEMM evaluator samples contain a non-number") from error
    if not samples_ms:
        raise ValueError("GEMM evaluator reported an empty sample set")

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
    """Parse repeated ``OBJECTIVE=WEIGHT`` command-line assignments."""

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
    return weights


def layouts_for_case(
    case: LayoutCase, matrices: Mapping[str, MatrixSpec]
) -> dict[str, Layout]:
    """Realize a uniform case as one array-specific layout per GEMM matrix."""

    return {
        name: canonical_layout_from_word(
            matrix,
            case.word,
            name=f"{case.name}.{name}",
        )
        for name, matrix in matrices.items()
    }


def evaluator_command(case: LayoutCase, args: argparse.Namespace) -> list[str]:
    """Construct the existing evaluator command for one uniform layout case."""

    command = [
        sys.executable,
        str(EVALUATOR),
        case.word,
        case.word,
        case.word,
        "--n",
        str(args.n),
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--device",
        str(args.device),
        "--block-x",
        str(args.block_x),
        "--block-y",
        str(args.block_y),
        "--compiler",
        args.compiler,
    ]
    if args.arch:
        command.extend(("--arch", args.arch))
    return command


def benchmark_case(
    case: LayoutCase, args: argparse.Namespace
) -> tuple[TimingResult, list[str], str, str]:
    """Run and parse one layout benchmark, retaining its complete process output."""

    command = evaluator_command(case, args)
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
            f"benchmark {case.name!r} exited with {completed.returncode}: {detail}"
        )
    try:
        timing = parse_evaluator_output(completed.stdout)
    except ValueError as error:
        raise RuntimeError(f"benchmark {case.name!r}: {error}") from error
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
        "--n",
        type=positive_integer,
        default=256,
        help="square, power-of-two matrix dimension (default: %(default)s)",
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
        help="workgroup width / j threads (default: %(default)s)",
    )
    parser.add_argument(
        "--block-y",
        type=positive_integer,
        default=32,
        help="workgroup height / i threads (default: %(default)s)",
    )
    parser.add_argument(
        "--compiler",
        default="hipcc",
        help="HIP compiler command (default: %(default)s)",
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
        help="scalar cost used for score ranking (default: %(default)s)",
    )
    parser.add_argument(
        "--component-weight",
        action="append",
        default=[],
        metavar="OBJECTIVE=WEIGHT",
        help=(
            "set one objective's weight; unspecified objectives use 1 and "
            "weight 0 excludes a component from aggregate scores"
        ),
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="compute scores without compiling or running HIP kernels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="JSON",
        help="write the complete experiment report to this JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used to randomize benchmark order (default: %(default)s)",
    )
    return parser, parser.parse_args(argv)


def _format_rank(value: float | None) -> str:
    if value is None:
        return "-"
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def print_results(
    records: Sequence[dict[str, object]],
    score_mode: ScoreMode,
    correlations: Mapping[str, float | None],
    *,
    score_only: bool,
) -> None:
    """Print one compact rank-comparison table."""

    print("\nGEMM traditional-layout score experiment")
    print(f"  score mode: {score_mode} (lower is better)")
    if score_only:
        headers = ("score rank", "layout", "word (low -> high)", "score")
    else:
        headers = (
            "score rank",
            "runtime rank",
            "layout",
            "word (low -> high)",
            "score",
            "median ms",
            "rank delta",
        )

    rows: list[tuple[str, ...]] = []
    ordered = sorted(
        records,
        key=lambda record: (float(record["score_rank"]), str(record["name"])),
    )
    for record in ordered:
        score_rank = _format_rank(float(record["score_rank"]))
        if score_only:
            rows.append(
                (
                    score_rank,
                    str(record["name"]),
                    str(record["word"]),
                    f"{float(record['selected_score']):.6g}",
                )
            )
            continue
        timing = record["timing"]
        assert isinstance(timing, dict)
        rows.append(
            (
                score_rank,
                _format_rank(float(record["runtime_rank"])),
                str(record["name"]),
                str(record["word"]),
                f"{float(record['selected_score']):.6g}",
                f"{float(timing['median_ms']):.6f}",
                f"{float(record['rank_delta']):+.1f}",
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  " + "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    if not score_only:
        print("\n  Spearman correlations by score mode")
        for mode in SCORE_MODES:
            correlation = correlations[mode]
            suffix = " (selected)" if mode == score_mode else ""
            if correlation is None:
                value = "undefined (one rank vector is constant)"
            else:
                value = f"{correlation:.6f}"
            print(f"    {mode}: {value}{suffix}")
        print("  rank delta = score rank - runtime rank")


def run(argv: list[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    try:
        component_weights = parse_component_weights(args.component_weight)
        cases = traditional_layout_cases(args.n)
        config = gemm_problem.build_config(
            problem_size=args.n,
            block_size=(args.block_x, args.block_y, 1),
        )
        matrices_tuple = tuple(gemm_problem.get_matrices(config))
        events_tuple, sequences_tuple = gemm_problem.get_events_and_sequences(config)
        objective_specs = tuple(gemm_problem.get_objectives(config))
        matrices = {matrix.name: matrix for matrix in matrices_tuple}
        events = {event.id: event for event in events_tuple}
        print("Building GEMM objective components once...", flush=True)
        components = tuple(
            build_objectives(
                objective_specs,
                matrices,
                events,
                sequences_tuple,
            )
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    mode: ScoreMode = args.score_mode
    records: list[dict[str, object]] = []
    print(f"Scoring {len(cases)} traditional layouts...", flush=True)
    try:
        for case in cases:
            layouts = layouts_for_case(case, matrices)
            score = score_layouts(
                matrices,
                components,
                layouts,
                component_weights=component_weights,
            )
            records.append(
                {
                    "name": case.name,
                    "word": case.word,
                    "layouts": {name: case.word for name in matrices},
                    "selected_score": score.value(mode),
                    "aggregate_scores": {
                        aggregate_mode: score.value(aggregate_mode)
                        for aggregate_mode in SCORE_MODES
                    },
                    "score": score_to_dict(score),
                    "timing": None,
                    "benchmark_command": None,
                    "benchmark_stdout": None,
                    "benchmark_stderr": None,
                }
            )
    except ValueError as error:
        parser.error(str(error))

    score_values = [float(record["selected_score"]) for record in records]
    score_ranks = average_tie_ranks(score_values)
    for record, rank in zip(records, score_ranks):
        record["score_rank"] = rank

    run_order: list[str] = []
    correlation: float | None = None
    correlations: dict[str, float | None] = {
        aggregate_mode: None for aggregate_mode in SCORE_MODES
    }
    if not args.score_only:
        randomized_cases = list(cases)
        random.Random(args.seed).shuffle(randomized_cases)
        run_order = [case.name for case in randomized_cases]
        record_by_name = {str(record["name"]): record for record in records}
        try:
            for index, case in enumerate(randomized_cases, 1):
                print(
                    f"[{index}/{len(randomized_cases)}] Benchmarking {case.name}...",
                    flush=True,
                )
                timing, command, stdout, stderr = benchmark_case(case, args)
                record = record_by_name[case.name]
                record["timing"] = asdict(timing)
                record["benchmark_command"] = command
                record["benchmark_stdout"] = stdout
                record["benchmark_stderr"] = stderr
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

        runtime_values: list[float] = []
        for record in records:
            timing = record["timing"]
            assert isinstance(timing, dict)
            runtime_values.append(float(timing["median_ms"]))
        runtime_ranks = average_tie_ranks(runtime_values)
        for record, runtime_rank in zip(records, runtime_ranks):
            score_rank = float(record["score_rank"])
            record["runtime_rank"] = runtime_rank
            record["rank_delta"] = score_rank - runtime_rank
        for aggregate_mode in SCORE_MODES:
            aggregate_values: list[float] = []
            for record in records:
                aggregate_scores = record["aggregate_scores"]
                assert isinstance(aggregate_scores, dict)
                aggregate_values.append(float(aggregate_scores[aggregate_mode]))
            correlations[aggregate_mode] = spearman_correlation(
                aggregate_values, runtime_values
            )
        correlation = correlations[mode]

    print_results(records, mode, correlations, score_only=args.score_only)

    report = {
        "experiment": "gemm-traditional-layout-ranking",
        "configuration": {
            "n": args.n,
            "block": [args.block_x, args.block_y, 1],
            "samples": args.samples,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "device": args.device,
            "compiler": args.compiler,
            "arch": args.arch,
            "score_only": args.score_only,
            "seed": args.seed,
        },
        "score_mode": mode,
        "component_weights": component_weights,
        "benchmark_run_order": run_order,
        "spearman_correlation": correlation,
        "score_mode_spearman_correlations": correlations,
        "results": records,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
