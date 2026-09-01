#!/usr/bin/env python3
"""Profile randomly sampled tile layouts against Stage-1 memory counters."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_stage1_kernel_cases import CASES
from stage1_common import positive_integer
from stage1_counter_sweep import print_summary, run_sweep


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", "--kernel", dest="case", choices=CASES, default="gesummv"
    )
    parser.add_argument(
        "--transaction-bytes",
        "--byte-level",
        dest="transaction_bytes",
        type=positive_integer,
        default=64,
    )
    parser.add_argument(
        "--layouts",
        type=positive_integer,
        default=100,
        help="number of distinct random layouts to profile",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random layout sampling seed"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="checkpoint directory (default includes kernel and byte level)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="aggregate JSON path (default includes kernel and byte level)",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-profiles", type=positive_integer)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--profile-launches", type=positive_integer, default=1)
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
            / f"stage1-random-layout-counter-profiles{scale_tag}-mi300a"
            / args.case
        )
    if args.json is None:
        args.json = (
            result_root
            / f"stage1-{args.case}-random-layout-counters{scale_tag}-mi300a.json"
        )
    report = run_sweep(
        args,
        panel_mode="random_layouts",
        experiment="triton_stage1_random_layout_counters",
        panel_experiment="triton_stage1_random_layout_counter_panel",
        profile_experiment="triton_stage1_random_layout_counter_profile",
    )
    if not args.quiet:
        print_summary(report)
        rho = report["statistics"]["tie_aware_spearman"]
        rho_text = "n/a" if rho is None else f"{rho:.3f}"
        print(f"Spearman rho: {rho_text}")
        print(f"Wrote {args.json}")
        print(f"Wrote {args.csv or args.json.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
