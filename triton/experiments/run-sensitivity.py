#!/usr/bin/env python3
"""Run final Experiment 10 hardware-profile sensitivity for one case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from time import perf_counter

from appendix_common import (
    EXPERIMENT_ROOT,
    OPERATORS,
    activate_triton_source,
    exclusion_report,
    fraction,
    output_arguments,
    perturbed_profiles,
    positive,
    runtime_layout_key,
    selected_case,
    time_layout_pair,
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
        default=5,
        help="layout grammar to resolve under each perturbed profile",
    )
    parser.add_argument(
        "--perturbation-magnitudes",
        type=fraction,
        nargs="+",
        default=(0.10, 0.25, 0.50),
    )
    parser.add_argument("--trials-per-magnitude", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timing-warmup", type=positive, default=10)
    parser.add_argument("--timing-samples", type=positive, default=11)
    parser.add_argument("--timing-iterations", type=positive, default=50)
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


def _write_rows(path: Path, report) -> None:
    rows = []
    nominal = next(
        trial for trial in report.get("trials", ()) if trial["trial_id"] == "nominal"
    )
    nominal_speedup = float(nominal["speedup"])
    for trial in report.get("trials", ()):
        rows.append(
            {
                "experiment": 10,
                "platform": report["platform"],
                "operator": report["operator"],
                "config": report["config"],
                "search_experiment": report["search_experiment"],
                "trial_id": trial["trial_id"],
                "magnitude": trial["magnitude"],
                "trial_index": trial["trial_index"],
                "selection_id": trial["selection_id"],
                "agrees_with_nominal": trial["selection_id"]
                == nominal["selection_id"],
                "solve_seconds": trial["solve_seconds"],
                "baseline_median_ms": trial["timing"]["baseline"]["median_ms"],
                "selected_median_ms": trial["timing"]["selected"]["median_ms"],
                "speedup": trial["speedup"],
                "speedup_change_percent": 100.0
                * (float(trial["speedup"]) / nominal_speedup - 1.0),
                "tau": json.dumps(trial["tau"], sort_keys=True),
                "factors": json.dumps(trial["factors"], sort_keys=True),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(trials) -> list[dict[str, object]]:
    nominal = next(trial for trial in trials if trial["trial_id"] == "nominal")
    nominal_speedup = float(nominal["speedup"])
    result = []
    for magnitude in sorted({float(trial["magnitude"]) for trial in trials}):
        group = [
            trial for trial in trials if float(trial["magnitude"]) == magnitude
        ]
        speedups = [float(trial["speedup"]) for trial in group]
        result.append(
            {
                "magnitude": magnitude,
                "trial_count": len(group),
                "selection_agreement_fraction": sum(
                    trial["selection_id"] == nominal["selection_id"]
                    for trial in group
                )
                / len(group),
                "median_speedup": statistics.median(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "median_speedup_change_percent": statistics.median(
                    100.0 * (speedup / nominal_speedup - 1.0)
                    for speedup in speedups
                ),
            }
        )
    return result


def run(args) -> None:
    import torch
    from relay import AnalysisOptions, EvaluationLimits, analyze_launch
    from search_algorithms import load_tau_profile, select_layouts

    case = selected_case(args.operator, args.config)
    case_root = (
        args.results_root
        / "experiment-10"
        / args.platform
        / f"grammar-e{args.search_experiment}"
        / case.case_id
    ).resolve()
    report_path = case_root / "report.json"
    selection_path = case_root / "selection.json"
    if report_path.is_file() and not args.rerun:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"complete", "excluded"}:
            print(f"Reusing {existing['status']} result: {report_path}")
            return

    baseline_profile = load_tau_profile(args.platform, args.tau_profile)
    profile_trials = perturbed_profiles(
        baseline_profile,
        args.perturbation_magnitudes,
        args.trials_per_magnitude,
        args.seed,
    )
    spec = case.factory()
    trace_start = perf_counter()
    analysis = analyze_launch(
        spec.kernel,
        spec.grid,
        *spec.args,
        _laqs_options=AnalysisOptions(
            hardware_profile=baseline_profile,
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
            experiment=10,
            platform=args.platform,
            case=case,
            category=analysis.unsupported.category,
            message=analysis.unsupported.message,
            search_experiment=args.search_experiment,
            trace_capture_seconds=trace_capture_seconds,
        )
        write_json(report_path, report)
        print(f"Excluded {case.case_id}: {analysis.unsupported.message}")
        return

    selections = []
    try:
        for metadata, profile in profile_trials:
            solve_start = perf_counter()
            runtime_layouts, search = select_layouts(
                analysis,
                args.search_experiment,
                profile,
            )
            solve_seconds = perf_counter() - solve_start
            runtime_values = [layout.to_dict() for layout in runtime_layouts]
            selections.append(
                {
                    **metadata,
                    "profile_id": profile.profile_id,
                    "solve_seconds": solve_seconds,
                    "selection_id": runtime_layout_key(runtime_values),
                    "runtime_layouts": runtime_values,
                    "search": search,
                }
            )
    except Exception as error:
        report = exclusion_report(
            experiment=10,
            platform=args.platform,
            case=case,
            category="sensitivity_search",
            message=f"{type(error).__name__}: {error}",
            search_experiment=args.search_experiment,
            trace_capture_seconds=trace_capture_seconds,
        )
        write_json(report_path, report)
        print(f"Excluded {case.case_id}: {type(error).__name__}: {error}")
        return

    outputs = output_arguments(analysis)
    selection_document = {
        "schema": "relay.triton.experiment_10.selections.v1",
        "experiment": 10,
        "platform": args.platform,
        "operator": case.operator,
        "config": case.config,
        "search_experiment": args.search_experiment,
        "selected_config": dict(analysis.selected_config),
        "output_arguments": list(outputs),
        "trials": selections,
    }
    write_json(selection_path, selection_document)

    timing_by_selection = {}
    unique_selections = {}
    for selection in selections:
        unique_selections.setdefault(selection["selection_id"], selection)
    try:
        for order_offset, (selection_id, selection) in enumerate(
            unique_selections.items()
        ):
            timing_path = case_root / "timings" / f"{selection_id}.json"
            if timing_path.is_file() and not args.rerun:
                timing = json.loads(timing_path.read_text(encoding="utf-8"))
            else:
                timing = time_layout_pair(
                    spec,
                    analysis.selected_config,
                    selection["runtime_layouts"],
                    outputs,
                    warmup=args.timing_warmup,
                    samples=args.timing_samples,
                    iterations=args.timing_iterations,
                    order_offset=order_offset,
                )
                write_json(timing_path, timing)
            timing_by_selection[selection_id] = timing
            torch.cuda.empty_cache()
    except Exception as error:
        report = exclusion_report(
            experiment=10,
            platform=args.platform,
            case=case,
            category="sensitivity_realization",
            message=f"{type(error).__name__}: {error}",
            search_experiment=args.search_experiment,
            trace_capture_seconds=trace_capture_seconds,
        )
        report["artifacts"] = {"selection": str(selection_path)}
        write_json(report_path, report)
        print(f"Excluded {case.case_id}: {type(error).__name__}: {error}")
        return

    trials = []
    for selection in selections:
        timing = timing_by_selection[selection["selection_id"]]
        trials.append(
            {
                **selection,
                "timing": timing,
                "speedup": timing["speedup"],
            }
        )
    report = {
        "schema": "relay.triton.experiment_10.v1",
        "experiment": 10,
        "platform": args.platform,
        "operator": case.operator,
        "config": case.config,
        "description": case.description,
        "search_experiment": args.search_experiment,
        "status": "complete",
        "trace_capture_seconds": trace_capture_seconds,
        "hardware_profile": baseline_profile.to_dict(),
        "protocol": {
            "seed": args.seed,
            "perturbation_magnitudes": list(args.perturbation_magnitudes),
            "trials_per_magnitude": args.trials_per_magnitude,
            "perturbation_distribution": "independent_uniform_multipliers",
            "timing_warmup": args.timing_warmup,
            "timing_samples": args.timing_samples,
            "timing_iterations": args.timing_iterations,
        },
        "trace": {
            "matrix_count": len(analysis.matrices),
            "event_count": len(analysis.events),
            "sequence_count": len(analysis.sequences),
            "edge_family_count": len(analysis.edge_families),
            "component_count": len(analysis.components),
        },
        "unique_selection_count": len(unique_selections),
        "trials": trials,
        "summary_by_magnitude": _summary(trials),
        "artifacts": {
            "selection": str(selection_path),
            "raw_data": str(case_root / "raw-data.csv"),
            "timings": str(case_root / "timings"),
        },
    }
    write_json(report_path, report)
    _write_rows(case_root / "raw-data.csv", report)
    print(
        f"Completed Experiment 10, grammar E{args.search_experiment}, "
        f"{case.case_id}, {args.platform}"
    )
    print(
        f"Trials: {len(trials)}; unique selected mappings: "
        f"{len(unique_selections)}"
    )
    print(f"Report: {report_path}")


def main() -> None:
    args = arguments()
    activate_triton_source(args.platform)
    run(args)


if __name__ == "__main__":
    main()
