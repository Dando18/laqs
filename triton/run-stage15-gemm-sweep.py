#!/usr/bin/env python3
"""Run fresh-process GEMM rankings and optional isolated hardware counters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from stage15_analysis import (
    aggregate_rankings,
    counter_comparison,
    parse_counter_csv,
)
from stage1_common import positive_integer


def run_worker(worker: Path, args, output: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(worker),
        "--gemm-size",
        str(args.gemm_size),
        "--transaction-bytes",
        str(args.transaction_bytes),
        "--candidates",
        str(args.candidates),
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--json",
        str(output),
        "--quiet",
    ]
    if args.register_fibers:
        command.append("--register-fibers")
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def profiler_environment(rocprof: Path) -> dict[str, str]:
    environment = os.environ.copy()
    function_name = "BASH_FUNC_realpath%%"
    environment[function_name] = f"() {{ echo {shlex.quote(str(rocprof))}; }}"
    return environment


def profile_layout(
    worker: Path,
    args,
    *,
    label: str,
    rows: list[int],
    output: Path,
) -> dict[str, object]:
    command = [
        str(args.rocprof),
        "-i",
        str(args.profile_config),
        "-o",
        str(output),
        "--timestamp",
        "on",
        sys.executable,
        str(worker),
        "--gemm-size",
        str(args.gemm_size),
        "--transaction-bytes",
        str(args.transaction_bytes),
        "--profile-rows",
        *(str(row) for row in rows),
        "--profile-warmup",
        str(args.profile_warmup),
        "--profile-iterations",
        str(args.profile_iterations),
        "--quiet",
    ]
    print(f"Stage 1.5 counters: {label}", file=sys.stderr, flush=True)
    subprocess.run(
        command,
        check=True,
        env=profiler_environment(args.rocprof),
    )
    return parse_counter_csv(output, profile_iterations=args.profile_iterations)


def run_sweep(args) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    worker = Path(__file__).with_name("run-stage15-gemm.py")
    with tempfile.TemporaryDirectory(prefix="relay-stage15-gemm-") as temporary:
        temporary_path = Path(temporary)
        if args.ranking_json is None:
            launches = []
            for process_launch in range(1, args.process_launches + 1):
                print(
                    f"Stage 1.5 ranking: process {process_launch}/"
                    f"{args.process_launches}",
                    file=sys.stderr,
                    flush=True,
                )
                result = run_worker(
                    worker,
                    args,
                    temporary_path / f"ranking-{process_launch}.json",
                )
                result["process_launch"] = process_launch
                launches.append(result)
            aggregate = aggregate_rankings(launches)
            configuration = {
                "gemm_size": args.gemm_size,
                "process_launches": args.process_launches,
                "transaction_bytes": args.transaction_bytes,
                "requested_retained_candidates": args.candidates,
                "samples_per_candidate_per_process": args.samples,
                "iterations_per_timing_batch": args.iterations,
                "warmup_rounds": args.warmup,
                "runtime_aggregation": "median_of_process_medians",
                "register_fibers": args.register_fibers,
            }
        else:
            previous = json.loads(args.ranking_json.read_text(encoding="utf-8"))
            if previous.get("experiment") != "triton_stage15_gemm_ranking_sweep":
                raise ValueError("--ranking-json is not a Stage 1.5 GEMM sweep")
            launches = previous["launches"]
            aggregate = previous["ranking"]
            configuration = previous["configuration"]
            if int(configuration["gemm_size"]) != args.gemm_size:
                raise ValueError("--gemm-size does not match --ranking-json")
            if int(configuration["transaction_bytes"]) != args.transaction_bytes:
                raise ValueError(
                    "--transaction-bytes does not match --ranking-json"
                )
            if bool(configuration.get("register_fibers", False)) != (
                args.register_fibers
            ):
                raise ValueError(
                    "--register-fibers does not match --ranking-json"
                )
        hardware_counters = None
        if args.profile or args.profile_all:
            counter_dir = temporary_path / "counters"
            counter_dir.mkdir()
            targets = (
                aggregate["candidates"]
                if args.profile_all
                else (aggregate["default"], aggregate["selected"])
            )
            profiles = {}
            for index, candidate in enumerate(targets, start=1):
                candidate_id = str(candidate["candidate_id"])
                if candidate_id in profiles:
                    continue
                profiles[candidate_id] = profile_layout(
                    worker,
                    args,
                    label=(
                        f"candidate {index}/{len(targets)} {candidate_id}"
                        if args.profile_all
                        else (
                            "default"
                            if candidate_id
                            == aggregate["default"]["candidate_id"]
                            else "selected"
                        )
                    ),
                    rows=candidate["a_rows"],
                    output=counter_dir / f"{candidate_id}.csv",
                )
            default = profiles[str(aggregate["default"]["candidate_id"])]
            selected = profiles[str(aggregate["selected"]["candidate_id"])]
            candidate_profiles = None
            correlations = None
            if args.profile_all:
                from relay import spearman_rank_correlation

                candidate_profiles = []
                for candidate in aggregate["candidates"]:
                    profile = profiles[str(candidate["candidate_id"])]
                    candidate_profiles.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "mapping_id": candidate["mapping_id"],
                            "word": candidate["word"],
                            "quotient_score": candidate["quotient_score"],
                            "register_fiber_normalized_excess": candidate[
                                "register_fiber_normalized_excess"
                            ],
                            "register_aware_score": candidate[
                                "register_aware_score"
                            ],
                            "profile": profile,
                        }
                    )
                score_fields = (
                    "quotient_score",
                    "register_fiber_normalized_excess",
                    "register_aware_score",
                )
                counter_fields = (
                    "duration_ns",
                    "l1_to_l2_read_requests",
                    "l2_tag_requests",
                    "l2_misses",
                    "hbm_read_bytes",
                )
                correlations = {
                    score: {
                        counter: spearman_rank_correlation(
                            [
                                float(candidate[score])
                                for candidate in candidate_profiles
                            ],
                            [
                                float(candidate["profile"]["summary"][counter])
                                for candidate in candidate_profiles
                            ],
                        )
                        for counter in counter_fields
                    }
                    for score in score_fields
                }
            hardware_counters = {
                "scope": (
                    "whole-kernel counters; A and C are fixed, so the layout "
                    "comparison changes only persistent B"
                ),
                "cache_state": (
                    "profiled dispatches follow warmups; HBM bytes therefore "
                    "measure the steady cache-resident regime"
                ),
                "profiler": {
                    "tool": "rocprof v1",
                    "path": str(args.rocprof),
                    "configuration": str(args.profile_config),
                    "passes_per_layout": 4,
                    "profile_warmup": args.profile_warmup,
                    "profile_iterations": args.profile_iterations,
                },
                "default": default,
                "selected": selected,
                "candidate_profiles": candidate_profiles,
                "tie_aware_spearman": correlations,
                "comparison": counter_comparison(
                    default,
                    selected,
                    selected_to_default_b_request_ratio=(
                        float(aggregate["selected"]["quotient_score"])
                        / float(aggregate["default"]["quotient_score"])
                    ),
                ),
            }

    return {
        "stage": 1.5,
        "experiment": "triton_stage15_gemm_ranking_sweep",
        "configuration": configuration,
        "launches": launches,
        "ranking": aggregate,
        "hardware_counters": hardware_counters,
        "correct": bool(aggregate["correct"]),
    }


def parse_arguments(argv=None):
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemm-size", type=positive_integer, default=512)
    parser.add_argument("--process-launches", type=positive_integer, default=3)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--register-fibers", action="store_true")
    parser.add_argument("--samples", type=positive_integer, default=21)
    parser.add_argument("--iterations", type=positive_integer, default=50)
    parser.add_argument("--warmup", type=positive_integer, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="collect counters for every retained ranking candidate",
    )
    parser.add_argument(
        "--ranking-json",
        type=Path,
        help="reuse a completed ranking sweep and run only optional profiling",
    )
    parser.add_argument("--profile-iterations", type=positive_integer, default=20)
    parser.add_argument("--profile-warmup", type=positive_integer, default=5)
    parser.add_argument(
        "--rocprof",
        type=Path,
        default=Path("/opt/rocm-7.0.2/bin/rocprof"),
    )
    parser.add_argument(
        "--profile-config",
        type=Path,
        default=directory / "rocprof-stage15.txt",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    result = run_sweep(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
