#!/usr/bin/env python3
"""Run one pilot-kernel case for final counter experiments 1--3."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (str(TRITON_ROOT), str(REPOSITORY))

from analyze import analyze_report
from run_stage1_kernel_cases import CASES
from stage1_common import positive_integer
from stage1_counter_sweep import run_sweep as run_amd_sweep
from stage1_nvidia_counter_sweep import run_sweep as run_nvidia_sweep


PANEL_MODES = {
    1: "experiment1_gc_whole",
    2: "experiment2_gc_tiles",
    3: "experiment3_goc",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--layouts", type=positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-launches", type=positive_integer, default=3)
    parser.add_argument("--profile-warmup", type=positive_integer, default=5)
    parser.add_argument("--profile-iterations", type=positive_integer, default=20)
    parser.add_argument("--max-profiles", type=positive_integer)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--results-root", type=Path, default=EXPERIMENT_ROOT / "results"
    )
    parser.add_argument("--plots-root", type=Path, default=EXPERIMENT_ROOT / "plots")
    parser.add_argument(
        "--rocprof",
        type=Path,
        default=Path("/opt/rocm-7.0.2/bin/rocprof"),
    )
    parser.add_argument(
        "--ncu", type=Path, default=Path(shutil.which("ncu") or "ncu")
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    args.transaction_bytes = 64 if args.platform == "tuolumne" else 32
    case_root = (
        args.results_root
        / f"experiment-{args.experiment}"
        / args.platform
        / args.case
    ).resolve()
    args.results_dir = case_root / "profiles"
    args.json = case_root / "report.json"
    args.csv = case_root / "counter-data.csv"
    plot = (
        args.plots_root
        / f"experiment-{args.experiment}"
        / args.platform
        / f"{args.case}.pdf"
    ).resolve()

    if not args.analyze_only:
        common = {
            "panel_mode": PANEL_MODES[args.experiment],
            "experiment": f"triton_final_experiment_{args.experiment}",
            "panel_experiment": (
                f"triton_final_experiment_{args.experiment}_panel_{args.platform}"
            ),
            "profile_experiment": (
                f"triton_final_experiment_{args.experiment}_profile_{args.platform}"
            ),
        }
        if args.platform == "tuolumne":
            report = run_amd_sweep(args, **common)
        else:
            report = run_nvidia_sweep(args, **common)
        report["final_experiment"] = args.experiment
        report["configuration"]["score_profile"] = report["panel"][
            "score_profile"
        ]["profile_id"]
        from stage1_counter_sweep import write_json

        write_json(args.json, report)

    result = analyze_report(args.json, plot)
    if not args.quiet:
        print(
            f"Experiment {args.experiment}, {args.case}, {args.platform}: "
            f"{result['profiled_mapping_count']} mappings"
        )
        print(f"Raw data: {result['artifacts']['raw_data']}")
        print(f"Spearman: {result['artifacts']['spearman']}")
        print(f"Plot: {result['artifacts']['plot']}")


if __name__ == "__main__":
    main()
