#!/usr/bin/env python3
"""Profile one canonical representative at every issue-quotient level."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_stage1_kernel_cases import CASES
from stage1_common import positive_integer
from stage1_counter_sweep import print_summary, run_sweep


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="gesummv")
    parser.add_argument(
        "--tile-shape", type=positive_integer, nargs="+", default=(64, 64)
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="checkpoint directory (default is case-specific)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="aggregate JSON path (default is case-specific)",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-profiles", type=positive_integer)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=64)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--profile-launches", type=positive_integer, default=3)
    parser.add_argument("--profile-warmup", type=positive_integer, default=5)
    parser.add_argument(
        "--profile-iterations", type=positive_integer, default=20
    )
    parser.add_argument(
        "--rocprof",
        type=Path,
        default=Path("/opt/rocm-7.0.2/bin/rocprof"),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    result_root = Path(__file__).resolve().parent / "results"
    scale_tag = f"-q{args.transaction_bytes}b"
    if args.results_dir is None:
        args.results_dir = (
            result_root
            / f"stage1-quotient-level-counter-profiles{scale_tag}-mi300a"
            / args.case
        )
    if args.json is None:
        args.json = (
            result_root
            / (
                f"stage1-{args.case}-quotient-level-counters"
                f"{scale_tag}-mi300a.json"
            )
        )
    report = run_sweep(
        args,
        panel_mode="fixed_tile_levels",
        experiment="triton_stage1_quotient_level_counters",
        panel_experiment="triton_stage1_quotient_level_panel",
        profile_experiment="triton_stage1_quotient_level_counter_profile",
    )
    if not args.quiet:
        print_summary(report)
        print(f"Wrote {args.json}")
        print(f"Wrote {args.csv or args.json.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
