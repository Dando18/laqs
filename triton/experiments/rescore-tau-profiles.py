#!/usr/bin/env python3
"""Rescore recorded Experiments 1--3 under every named tau profile."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (str(TRITON_ROOT), str(REPOSITORY), str(EXPERIMENT_ROOT))

from analyze import analyze_report, analyze_suite
from run_stage1_kernel_cases import CASES
from stage1_counter_sweep import write_json
from tau_profiles import TAU_NAMES, load_tau_document, tau_profile_record
from tune_tau import _apply_profile


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("tuolumne", "matrix"),
        required=True,
    )
    parser.add_argument(
        "--tau-names", choices=TAU_NAMES, nargs="+", default=TAU_NAMES
    )
    parser.add_argument(
        "--experiments",
        choices=(1, 2, 3),
        type=int,
        nargs="+",
        default=(1, 2, 3),
    )
    parser.add_argument(
        "--stratifications",
        choices=("all", "issue", "temporal"),
        nargs="+",
        default=("all", "issue", "temporal"),
    )
    parser.add_argument("--cases", choices=CASES, nargs="+", default=CASES)
    parser.add_argument(
        "--counter-source", type=Path, default=EXPERIMENT_ROOT / "results"
    )
    parser.add_argument(
        "--results-root", type=Path, default=EXPERIMENT_ROOT / "results"
    )
    parser.add_argument(
        "--plots-root", type=Path, default=EXPERIMENT_ROOT / "plots"
    )
    parser.add_argument(
        "--tau-profile",
        type=Path,
        default=EXPERIMENT_ROOT / "tau-profiles.json",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _compact(report, source: Path):
    """Keep analysis inputs while referring large profiler records to source."""

    result = deepcopy(report)
    for candidate in result["candidates"]:
        counters = candidate.get("counters")
        if counters is not None:
            candidate["counters"] = {"steady_state": counters["steady_state"]}
        for field in (
            "compiled_codegen_by_launch",
            "profile_checkpoints",
            "structural_validation",
        ):
            candidate.pop(field, None)
    scored = {
        candidate["mapping_id"]: candidate for candidate in result["candidates"]
    }
    result["panel"]["candidates"] = [
        {
            key: scored[candidate["mapping_id"]][key]
            for key in (
                "candidate_id",
                "mapping_id",
                "j_area",
                "j_area_rank",
                "peak_normalized_excess",
            )
        }
        for candidate in result["panel"]["candidates"]
    ]
    result["counter_source"] = {
        "report": str(source.resolve()),
        "reuse_key": "mapping_id",
        "profilers_relaunched": False,
        "large_profiler_records_retained_at_source": True,
    }
    result["configuration"]["counter_source"] = str(source.resolve())
    result["missing_profiles"] = []
    return result


def _output_current(output: Path, plot: Path, source: Path, profile) -> bool:
    if (
        not output.is_file()
        or not output.with_name("analysis.json").is_file()
        or not plot.is_file()
    ):
        return False
    report = json.loads(output.read_text(encoding="utf-8"))
    scored = report.get("panel", {}).get("score_profile", {})
    return (
        scored.get("profile_id") == profile["profile_id"]
        and scored.get("active_tau") == profile["active_tau"]
        and Path(report.get("counter_source", {}).get("report", "")).resolve()
        == source.resolve()
    )


def main() -> None:
    args = arguments()
    document = load_tau_document(args.tau_profile)
    for tau_name in args.tau_names:
        _, profile = tau_profile_record(document, args.platform, tau_name)
        destination = args.results_root / "tau-profiles" / tau_name
        plot_root = args.plots_root / "tau-profiles" / tau_name
        for experiment in args.experiments:
            for stratification in args.stratifications:
                for case in args.cases:
                    source = (
                        args.counter_source
                        / f"experiment-{experiment}"
                        / args.platform
                        / f"stratified-{stratification}"
                        / case
                        / "report.json"
                    )
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    output = (
                        destination
                        / f"experiment-{experiment}"
                        / args.platform
                        / f"stratified-{stratification}"
                        / case
                        / "report.json"
                    )
                    plot = (
                        plot_root
                        / f"experiment-{experiment}"
                        / args.platform
                        / f"stratified-{stratification}"
                        / f"{case}.pdf"
                    )
                    if args.skip_existing and _output_current(
                        output, plot, source, profile
                    ):
                        continue
                    report = json.loads(source.read_text(encoding="utf-8"))
                    _apply_profile(report, profile)
                    report = _compact(report, source)
                    write_json(output, report)
                    counter_csv = source.with_name("counter-data.csv")
                    if counter_csv.is_file():
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(
                            counter_csv, output.with_name("counter-data.csv")
                        )
                    analyze_report(output, plot)
                analyze_suite(
                    destination,
                    plot_root,
                    experiment=experiment,
                    platform=args.platform,
                    stratification=stratification,
                    regenerate_reports=False,
                )
        print(f"Rescored {args.platform} with {tau_name}: {destination}")


if __name__ == "__main__":
    main()
