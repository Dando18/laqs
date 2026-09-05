#!/usr/bin/env python3
"""Run final Experiment 12 phase timings for one TritonBench case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from appendix_common import (
    EXPERIMENT_ROOT,
    OPERATORS,
    activate_triton_source,
    exclusion_report,
    positive,
    replace_target_matrices,
    selected_case,
    timed_repetitions,
    write_json,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument("--operator", choices=OPERATORS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--search-experiment",
        type=int,
        choices=(4, 5, 6),
        default=4,
        help="search grammar whose solve time is measured",
    )
    parser.add_argument("--graph-repeats", type=positive, default=3)
    parser.add_argument("--score-repeats", type=positive, default=5)
    parser.add_argument("--solve-repeats", type=positive, default=3)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results",
    )
    parser.add_argument(
        "--tau-profile",
        type=Path,
        default=EXPERIMENT_ROOT / "tau-profiles.json",
    )
    return parser.parse_args(argv)


def _write_row(path: Path, report) -> None:
    row = {
        "experiment": 12,
        "platform": report["platform"],
        "operator": report["operator"],
        "config": report["config"],
        "search_experiment": report["search_experiment"],
        "status": report["status"],
    }
    if report["status"] == "complete":
        row.update(
            {
                "trace_capture_seconds": report["trace_capture_seconds"],
                "graph_construction_median_seconds": report[
                    "graph_construction"
                ]["median_seconds"],
                "graph_construction_min_seconds": report["graph_construction"][
                    "min_seconds"
                ],
                "quotient_score_median_seconds": report["quotient_score"][
                    "median_seconds"
                ],
                "quotient_score_min_seconds": report["quotient_score"][
                    "min_seconds"
                ],
                "solve_median_seconds": report["solve"]["median_seconds"],
                "solve_min_seconds": report["solve"]["min_seconds"],
                "matrix_count": report["trace"]["matrix_count"],
                "event_count": report["trace"]["event_count"],
                "sequence_count": report["trace"]["sequence_count"],
                "edge_family_count": report["trace"]["edge_family_count"],
                "component_count": report["trace"]["component_count"],
                "optimized_array_count": report["selection"][
                    "optimized_array_count"
                ],
                "transformed_array_count": report["selection"][
                    "transformed_array_count"
                ],
                "selected_j_area": report["selection"]["score"][
                    "hardware_area"
                ],
            }
        )
    else:
        row.update(
            {
                "exclusion_category": report["exclusion"]["category"],
                "exclusion_message": report["exclusion"]["message"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run(args) -> None:
    from relay import AnalysisOptions, EvaluationLimits, analyze_launch
    from relay.access_scopes import (
        build_edge_families,
        materialize_edge_families,
    )
    from relay.layouts import row_major_layout
    from relay.scoring import score_layouts
    from search_algorithms import load_tau_profile, select_layouts

    case = selected_case(args.operator, args.config)
    case_root = (
        args.results_root
        / "experiment-12"
        / args.platform
        / f"grammar-e{args.search_experiment}"
        / case.case_id
    ).resolve()
    report_path = case_root / "report.json"
    if report_path.is_file() and not args.rerun:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"complete", "excluded"}:
            print(f"Reusing {existing['status']} result: {report_path}")
            return

    profile = load_tau_profile(args.platform, args.tau_profile)
    spec = case.factory()
    trace_start = perf_counter()
    analysis = analyze_launch(
        spec.kernel,
        spec.grid,
        *spec.args,
        _laqs_options=AnalysisOptions(
            hardware_profile=profile,
            limits=EvaluationLimits(
                max_trace_contexts=1 << 16,
                max_dynamic_events=1 << 20,
            ),
        ),
        **spec.kwargs,
    )
    trace_capture_seconds = perf_counter() - trace_start
    if not analysis.supported:
        report = exclusion_report(
            experiment=12,
            platform=args.platform,
            case=case,
            category=analysis.unsupported.category,
            message=analysis.unsupported.message,
            search_experiment=args.search_experiment,
            trace_capture_seconds=trace_capture_seconds,
        )
        write_json(report_path, report)
        _write_row(case_root / "raw-data.csv", report)
        print(f"Excluded {case.case_id}: {analysis.unsupported.message}")
        return

    matrix_map = {matrix.name: matrix for matrix in analysis.matrices}
    event_map = {event.id: event for event in analysis.events}

    def construct_graph():
        families = build_edge_families(
            matrix_map,
            event_map,
            analysis.sequences,
        )
        components = materialize_edge_families(
            families,
            matrix_map,
            profile.byte_scales,
        )
        return families, components

    try:
        (families, components), graph_timing = timed_repetitions(
            construct_graph,
            args.graph_repeats,
        )
        if tuple(component.name for component in components) != tuple(
            component.name for component in analysis.components
        ):
            raise RuntimeError(
                "replayed graph components differ from initial launch analysis"
            )

        target_matrices = replace_target_matrices(analysis)
        target_map = {matrix.name: matrix for matrix in target_matrices}
        baseline_layouts = {
            matrix.name: row_major_layout(matrix) for matrix in target_matrices
        }

        def quotient_score():
            return score_layouts(
                target_map,
                components,
                baseline_layouts,
                hardware_profile=profile,
            )

        score, score_timing = timed_repetitions(
            quotient_score,
            args.score_repeats,
        )

        def solve():
            return select_layouts(
                analysis,
                args.search_experiment,
                profile,
            )

        (runtime_layouts, selection), solve_timing = timed_repetitions(
            solve,
            args.solve_repeats,
        )
    except Exception as error:
        report = exclusion_report(
            experiment=12,
            platform=args.platform,
            case=case,
            category="phase_timing",
            message=f"{type(error).__name__}: {error}",
            search_experiment=args.search_experiment,
            trace_capture_seconds=trace_capture_seconds,
        )
        write_json(report_path, report)
        _write_row(case_root / "raw-data.csv", report)
        print(f"Excluded {case.case_id}: {type(error).__name__}: {error}")
        return

    report = {
        "schema": "relay.triton.experiment_12.v1",
        "experiment": 12,
        "platform": args.platform,
        "operator": case.operator,
        "config": case.config,
        "description": case.description,
        "search_experiment": args.search_experiment,
        "status": "complete",
        "hardware_profile": profile.to_dict(),
        "selected_config": dict(analysis.selected_config),
        "trace_capture_seconds": trace_capture_seconds,
        "phase_definitions": {
            "trace_capture_seconds": (
                "ordinary compile/autotune/launch, manifest evaluation, and the "
                "frontend's initial graph construction; reported as setup only"
            ),
            "graph_construction": (
                "build universal-v1 scale-free edge families from the exact "
                "recorded trace and materialize the hardware byte-scale ladder"
            ),
            "quotient_score": (
                "compute every Q component and aggregate J_area once for the "
                "ordinary row-major layouts"
            ),
            "solve": (
                "end-to-end exact layout selection, including solver objective "
                "materialization and final selected-layout score"
            ),
        },
        "trace": {
            "matrix_count": len(target_matrices),
            "target_matrix_count": sum(matrix.target for matrix in target_matrices),
            "event_count": len(analysis.events),
            "sequence_count": len(analysis.sequences),
            "edge_family_count": len(families),
            "component_count": len(components),
            "hyperedge_count": sum(
                len(edges)
                for component in components
                for edges in component.edges_by_array.values()
            ),
        },
        "graph_construction": graph_timing,
        "quotient_score": {
            **score_timing,
            "layout": "ordinary_row_major",
            "j_area": float(score.hardware_area),
            "component_count": len(score.components),
        },
        "solve": solve_timing,
        "selection": selection,
        "runtime_layouts": [layout.to_dict() for layout in runtime_layouts],
        "artifacts": {"raw_data": str(case_root / "raw-data.csv")},
    }
    write_json(report_path, report)
    _write_row(case_root / "raw-data.csv", report)
    print(
        f"Completed Experiment 12, grammar E{args.search_experiment}, "
        f"{case.case_id}, {args.platform}"
    )
    print(
        "Median seconds: "
        f"graph={graph_timing['median_seconds']:.6f}, "
        f"score={score_timing['median_seconds']:.6f}, "
        f"solve={solve_timing['median_seconds']:.6f}"
    )
    print(f"Report: {report_path}")


def main() -> None:
    args = arguments()
    activate_triton_source(args.platform)
    run(args)


if __name__ == "__main__":
    main()
