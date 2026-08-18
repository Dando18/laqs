from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .layouts import CanonicalLayout, LinearInnerLayout
from .model import MatrixSpec
from .solver import Candidate, RelayResult


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rows_text = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rows_text:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    body = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows_text]
    return "\n".join([line, rule, *body])


def _short_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value) >= 1000:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _tile_text(candidate: Candidate) -> str:
    return "x".join(str(1 << exponent) for exponent in candidate.layout.tile_exponents)


def _layout_text(candidate: Candidate, matrix: MatrixSpec) -> str:
    layout = candidate.layout
    if isinstance(layout, CanonicalLayout):
        return layout.word_string(matrix)
    labels = layout.physical_bit_labels(matrix)
    return ",".join(labels[:4]) + ("..." if len(labels) > 4 else "")


def print_report(
    result: RelayResult,
    *,
    max_candidates: int = 12,
    show_joint: int = 8,
    show_layouts: int = 2,
    objective_names: Sequence[str] | None = None,
) -> None:
    problem = result.problem
    matrix_by_name = {matrix.name: matrix for matrix in problem.matrices}
    objective_names = tuple(objective_names or [component.name for component in result.components])

    print(f"RELAY solve: {problem.name}")
    print(
        f"  matrices={len(problem.matrices)}  events={len(problem.events)}  "
        f"sequences={len(problem.sequences)}  objectives={len(result.components)}  "
        f"elapsed={result.elapsed_seconds:.3f}s"
    )
    print()

    print("Matrices")
    print(
        _table(
            ("name", "shape", "bytes", "target", "role", "bits"),
            (
                (
                    matrix.name,
                    "x".join(map(str, matrix.shape)),
                    matrix.element_bytes,
                    "yes" if matrix.target else "context",
                    matrix.role,
                    "+".join(map(str, matrix.mode_bits)),
                )
                for matrix in problem.matrices
            ),
        )
    )
    print()

    print("Objective components")
    objective_rows = []
    for component in result.components:
        edge_summary = ", ".join(
            f"{name}:{len(edges)}" for name, edges in component.edges_by_array.items()
        )
        objective_rows.append(
            (
                component.name,
                f"{component.region_bytes}B",
                component.provenance,
                "yes" if component.search else "report",
                edge_summary or "-",
            )
        )
    print(_table(("name", "region", "status", "search", "edges by array"), objective_rows))
    print()

    for name, array_result in result.arrays.items():
        matrix = array_result.matrix
        print(f"Array {name}")
        print(
            f"  tile hypotheses={len(array_result.tile_hypotheses)}  "
            f"realized candidates={array_result.all_candidate_count}  "
            f"retained={len(array_result.candidates)}  elapsed={array_result.elapsed_seconds:.3f}s"
        )
        if array_result.search_stats:
            search_rows = []
            for stats in array_result.search_stats:
                shape = "x".join(str(1 << exponent) for exponent in stats.tile_exponents)
                search_rows.append((
                    stats.grammar,
                    shape,
                    stats.active_rank if stats.active_rank is not None else "-",
                    stats.states,
                    stats.transitions,
                    "yes" if stats.exact else "no",
                    stats.note or ("alternatives capped" if stats.truncated else ""),
                ))
            print("  Search runs")
            print(_table(("grammar", "tile", "active", "states", "trans", "exact", "note"), search_rows))
        columns = ["#", "layout", "grammar", "tile", "word / low bits"]
        columns.extend(objective_names)
        columns.extend(("runs", "xor", "exact"))
        rows = []
        for rank, candidate in enumerate(array_result.candidates[:max_candidates], 1):
            row: list[object] = [
                rank,
                candidate.layout.name,
                candidate.grammar,
                _tile_text(candidate),
                _layout_text(candidate, matrix),
            ]
            row.extend(_short_number(float(candidate.scores.get(obj, 0.0))) for obj in objective_names)
            row.extend(
                (
                    candidate.layout.runs,
                    candidate.layout.xor_count,
                    "yes" if candidate.exact else "capped",
                )
            )
            rows.append(row)
        print(_table(columns, rows))

        bound_rows = []
        if array_result.candidates:
            best = array_result.candidates[0]
            for objective in objective_names:
                if objective not in best.packing_bounds:
                    continue
                score = float(best.scores.get(objective, 0.0))
                bound = float(best.packing_bounds[objective])
                ratio = score / bound if bound else 0.0
                bound_rows.append((objective, _short_number(bound), _short_number(score), f"{ratio:.3f}"))
        if bound_rows:
            print("\n  Best candidate versus packing bounds")
            print(_table(("objective", "bound", "score", "ratio"), bound_rows))

        for candidate in array_result.candidates[:show_layouts]:
            print()
            print_layout(candidate, matrix)
        print()

    if result.joint_candidates:
        print("Joint configurations")
        columns = ["#", *result.arrays.keys(), *objective_names, "runs", "xor"]
        rows = []
        for rank, joint in enumerate(result.joint_candidates[:show_joint], 1):
            row: list[object] = [rank]
            for matrix_name in result.arrays:
                row.append(joint.layouts[matrix_name].layout.name)
            row.extend(_short_number(float(joint.scores.get(obj, 0.0))) for obj in objective_names)
            row.extend(
                (
                    _short_number(float(joint.scores.get("runs", 0.0))),
                    _short_number(float(joint.scores.get("xors", 0.0))),
                )
            )
            rows.append(row)
        print(_table(columns, rows))


def print_layout(candidate: Candidate, matrix: MatrixSpec, *, max_side: int = 8) -> None:
    layout = candidate.layout
    print(f"  Layout: {layout.name}")
    print(f"    grammar={layout.grammar} tile={_tile_text(candidate)} outer_order=" + ",".join(matrix.mode_names[i] for i in layout.outer_order))
    if isinstance(layout, CanonicalLayout):
        print(f"    word (low -> high physical bits): {layout.word_string(matrix)}")
        print(f"    bits: {' | '.join(layout.physical_bit_labels(matrix))}")
    else:
        print(f"    active rank={layout.active_rank}  A_in rows:")
        for index, expression in enumerate(layout.physical_bit_labels(matrix)):
            print(f"      y{index} = {expression}")
    print(f"    codegen proxy: runs={layout.runs}, xors={layout.xor_count}")
    for item in layout.encode_plan(matrix):
        print(f"      {item}")

    if matrix.rank == 2:
        rows = min(1 << layout.tile_exponents[0], max_side)
        cols = min(1 << layout.tile_exponents[1], max_side)
        print(f"    inner offsets for logical [{rows}x{cols}] corner:")
        width = max(2, len(str((1 << sum(layout.tile_exponents)) - 1)))
        header = " " * (width + 2) + " ".join(f"j={j:>{width - 2}}" for j in range(cols))
        print("      " + header)
        for i in range(rows):
            values = [layout.offset(matrix, (i, j)) for j in range(cols)]
            print("      " + f"i={i:<2} " + " ".join(f"{value:>{width}}" for value in values))

    stats = candidate.search_stats
    print(
        f"    search: {stats.grammar}, states={stats.states}, transitions={stats.transitions}, "
        f"active_rank={stats.active_rank if stats.active_rank is not None else '-'}, "
        f"exact={'yes' if candidate.exact else 'no'}"
    )


def result_to_dict(result: RelayResult) -> dict[str, object]:
    def candidate_dict(candidate: Candidate) -> dict[str, object]:
        layout = candidate.layout
        data: dict[str, object] = {
            "name": layout.name,
            "grammar": layout.grammar,
            "tile_exponents": list(layout.tile_exponents),
            "tile_shape": [1 << exponent for exponent in layout.tile_exponents],
            "outer_order": list(layout.outer_order),
            "runs": layout.runs,
            "xor_count": layout.xor_count,
            "scores": dict(candidate.scores),
            "packing_bounds": dict(candidate.packing_bounds),
            "search_scores": dict(candidate.search_scores),
            "exact": candidate.exact,
            "note": candidate.note,
            "search_stats": {
                "grammar": candidate.search_stats.grammar,
                "states": candidate.search_stats.states,
                "transitions": candidate.search_stats.transitions,
                "paths_considered": candidate.search_stats.paths_considered,
                "paths_retained": candidate.search_stats.paths_retained,
                "active_rank": candidate.search_stats.active_rank,
                "truncated": candidate.search_stats.truncated,
            },
        }
        if isinstance(layout, CanonicalLayout):
            data["word"] = list(layout.word)
        elif isinstance(layout, LinearInnerLayout):
            data["a_rows"] = list(layout.a_rows)
            data["basis_columns"] = list(layout.basis_columns)
        return data

    return {
        "problem": result.problem.name,
        "elapsed_seconds": result.elapsed_seconds,
        "matrices": [
            {
                "name": matrix.name,
                "shape": list(matrix.shape),
                "element_bytes": matrix.element_bytes,
                "mode_names": list(matrix.mode_names),
                "target": matrix.target,
                "role": matrix.role,
            }
            for matrix in result.problem.matrices
        ],
        "objectives": [
            {
                "name": component.name,
                "region_bytes": component.region_bytes,
                "provenance": component.provenance,
                "search": component.search,
                "edge_counts": {name: len(edges) for name, edges in component.edges_by_array.items()},
            }
            for component in result.components
        ],
        "arrays": {
            name: {
                "all_candidate_count": array_result.all_candidate_count,
                "tile_hypotheses": [list(item) for item in array_result.tile_hypotheses],
                "search_stats": [
                    {
                        "grammar": stats.grammar,
                        "tile_exponents": list(stats.tile_exponents),
                        "states": stats.states,
                        "transitions": stats.transitions,
                        "paths_considered": stats.paths_considered,
                        "paths_retained": stats.paths_retained,
                        "active_rank": stats.active_rank,
                        "exact": stats.exact,
                        "truncated": stats.truncated,
                        "note": stats.note,
                    }
                    for stats in array_result.search_stats
                ],
                "elapsed_seconds": array_result.elapsed_seconds,
                "candidates": [candidate_dict(candidate) for candidate in array_result.candidates],
            }
            for name, array_result in result.arrays.items()
        },
        "joint_candidates": [
            {
                "layouts": {name: candidate.layout.name for name, candidate in joint.layouts.items()},
                "scores": dict(joint.scores),
            }
            for joint in result.joint_candidates
        ],
    }


def dump_json(result: RelayResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result_to_dict(result), indent=2, sort_keys=True) + "\n")
