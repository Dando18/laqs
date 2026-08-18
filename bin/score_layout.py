#!/usr/bin/env python3
"""Score concrete layouts for a Python RELAY problem description.

A problem module must provide ``build_config(**options)``,
``get_matrices(config)``, ``get_events_and_sequences(config)``, and
``get_objectives(config)``.  Layouts are supplied as ``ARRAY=SPEC``.  A spec
is ``row-major``, ``column-major``, or a canonical word whose symbols list
physical element-address bits from least significant to most significant.

Examples:

    .venv/bin/python bin/score_layout.py kernels/gemm/problem.py \
        --layout A=row-major --layout B=row-major --layout C=row-major

    .venv/bin/python bin/score_layout.py kernels/gemm/problem.py \
        --layout A=jjjiii --layout B=jjjiii --layout C=jjjiii \
        --score-mode weighted-normalized-excess --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Mapping

from relay import (
    SCORE_MODES,
    CanonicalLayout,
    LayoutScore,
    RelayProblem,
    ScoreMode,
    canonical_layout_from_word,
    column_major_layout,
    row_major_layout,
    score_problem,
    score_to_dict,
)
from relay.layouts import Layout


def _assignment(text: str, option: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not name or not value:
        raise ValueError(f"{option} expects NAME=VALUE, got {text!r}")
    return name, value


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("relay_score_problem", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_problem(path: Path, options: Mapping[str, object]) -> RelayProblem:
    """Load the small Python problem protocol used by ``kernels/gemm``."""

    module = _load_module(path)
    config = module.build_config(**options)
    matrices = tuple(module.get_matrices(config))
    events, sequences = module.get_events_and_sequences(config)
    objectives = tuple(module.get_objectives(config))
    return RelayProblem(
        matrices=matrices,
        events=tuple(events),
        sequences=tuple(sequences),
        objectives=objectives,
        name=path.stem,
    )


def _parse_problem_options(values: list[str]) -> dict[str, object]:
    options: dict[str, object] = {}
    for text in values:
        name, value = _assignment(text, "--problem-option")
        try:
            options[name] = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"--problem-option value for {name!r} must be valid JSON"
            ) from error
    return options


def _parse_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for text in values:
        name, value = _assignment(text, "--component-weight")
        try:
            weights[name] = float(value)
        except ValueError as error:
            raise ValueError(
                f"--component-weight value for {name!r} must be a number"
            ) from error
    return weights


def _layout_from_spec(matrix, text: str) -> Layout:
    normalized = text.lower()
    if normalized == "row-major":
        return row_major_layout(matrix)
    if normalized == "column-major":
        return column_major_layout(matrix)
    word = normalized.removeprefix("word:")
    return canonical_layout_from_word(matrix, word, name=f"canonical_{word}")


def parse_layouts(problem: RelayProblem, values: list[str]) -> dict[str, Layout]:
    matrices = {matrix.name: matrix for matrix in problem.matrices}
    assignments: dict[str, str] = {}
    default: str | None = None
    for text in values:
        name, layout_spec = _assignment(text, "--layout")
        if name == "all":
            default = layout_spec
        elif name not in matrices:
            raise ValueError(f"--layout names unknown array {name!r}")
        else:
            assignments[name] = layout_spec

    layouts: dict[str, Layout] = {}
    for name, matrix in matrices.items():
        layout_spec = assignments.get(name, default)
        if layout_spec is not None:
            layouts[name] = _layout_from_spec(matrix, layout_spec)
    return layouts


def _layout_description(layout: Layout, matrix) -> str:
    if isinstance(layout, CanonicalLayout):
        return layout.word_string(matrix)
    return layout.name


def _print_score(
    problem: RelayProblem,
    layouts: Mapping[str, Layout],
    score: LayoutScore,
    mode: ScoreMode,
) -> None:
    matrices = {matrix.name: matrix for matrix in problem.matrices}
    print("RELAY layout score (all costs are minimized)")
    print("Layouts (canonical words are low -> high physical bits)")
    for name, layout in layouts.items():
        print(f"  {name}: {_layout_description(layout, matrices[name])}")

    print("\nObjective components")
    print(
        "  name                            region  weight  "
        "Q (edge-weighted)  lower bound  normalized excess"
    )
    for component in score.components:
        print(
            f"  {component.name:<30} {component.region_bytes:>6}B  "
            f"{component.weight:>6g}  {component.raw_region_count:>17g}  "
            f"{component.packing_lower_bound:>11g}  "
            f"{component.normalized_excess:>17g}"
        )

    print("\nAggregate scores")
    print(f"  weighted-region-count:       {score.weighted_region_count:g}")
    print(f"  peak-normalized-excess:      {score.peak_normalized_excess:g}")
    print(f"  weighted-normalized-excess:  {score.weighted_normalized_excess:g}")
    print(f"\nSelected score ({mode}): {score.value(mode):g}")


def parse_arguments(
    argv: list[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem_file", type=Path, help="Python RELAY problem module")
    parser.add_argument(
        "--layout",
        action="append",
        required=True,
        metavar="ARRAY=SPEC",
        help=(
            "layout for an array: row-major, column-major, or a low-to-high "
            "canonical word; ARRAY=all provides a default"
        ),
    )
    parser.add_argument(
        "--score-mode",
        choices=SCORE_MODES,
        default="weighted-normalized-excess",
        help="scalar aggregate selected for the final score (default: %(default)s)",
    )
    parser.add_argument(
        "--component-weight",
        action="append",
        default=[],
        metavar="OBJECTIVE=WEIGHT",
        help=(
            "set tau for one objective; unspecified objectives use 1 and zero "
            "excludes an objective from aggregates"
        ),
    )
    parser.add_argument(
        "--problem-option",
        action="append",
        default=[],
        metavar="NAME=JSON_VALUE",
        help="keyword passed to build_config, for example problem_size=512",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable report instead of the terminal table",
    )
    return parser, parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    try:
        options = _parse_problem_options(args.problem_option)
        weights = _parse_weights(args.component_weight)
        problem = load_problem(args.problem_file.resolve(), options)
        layouts = parse_layouts(problem, args.layout)
        score = score_problem(problem, layouts, component_weights=weights)
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    mode: ScoreMode = args.score_mode
    if args.json:
        data = score_to_dict(score)
        data.update(
            {
                "problem": problem.name,
                "layouts": {
                    name: _layout_description(
                        layout,
                        next(matrix for matrix in problem.matrices if matrix.name == name),
                    )
                    for name, layout in layouts.items()
                },
                "selected_score_mode": mode,
                "selected_score": score.value(mode),
            }
        )
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        _print_score(problem, layouts, score, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
