#!/usr/bin/env python3
"""Profile row-major and LAQS layouts for the seven Triton test kernels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

from run_stage1_kernel_cases import CASES
from stage1_common import positive_integer
from stage1_counter_analysis import (
    COUNTER_DEFINITIONS,
    COUNT_METRICS,
    aggregate_profiles,
    compare_summaries,
    parse_counter_csv,
    rank_correlation,
)


EXPERIMENT = "triton_stage1_locality_counters"
PROFILE_EXPERIMENT = "triton_stage1_locality_counter_profile"
KERNEL_NAMES = {
    "bias_relu": "bias_relu_kernel",
    "softmax_bias": "softmax_bias_kernel",
    "embedding_bag": "embedding_bag_kernel",
    "gemv": "gemv_kernel",
    "mvt": "mvt_kernel",
    "gesummv": "gesummv_kernel",
    "stencil5": "stencil5_kernel",
}
COUNTER_PASSES = (
    "pmc : TCP_TOTAL_CACHE_ACCESSES_sum TCP_TCC_READ_REQ_sum "
    "TCP_TCC_WRITE_REQ_sum",
    "pmc : TCC_REQ_sum TCC_HIT_sum TCC_MISS_sum",
    "pmc : FETCH_SIZE WRITE_SIZE",
)


def _reduction(default: float, selected: float) -> float | None:
    if default == 0.0:
        return None
    return 1.0 - selected / default


def _profile_configuration(args, case, cache_mode, role, launch):
    return {
        "case": case,
        "kernel_name": KERNEL_NAMES[case],
        "cache_mode": cache_mode,
        "layout_role": role,
        "profile_launch": launch,
        "temporal_mode": args.temporal_mode,
        "transaction_bytes": args.transaction_bytes,
        "requested_retained_candidates": args.candidates,
        "profile_warmup": args.profile_warmup,
        "profile_iterations": args.profile_iterations,
        "cache_thrash_bytes": (
            args.cache_thrash_bytes if cache_mode == "thrashed" else 0
        ),
        "counter_passes": list(COUNTER_PASSES),
        "rocprof": str(args.rocprof.resolve()),
    }


def _current_profile(path: Path, configuration: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("experiment") == PROFILE_EXPERIMENT
        and record.get("configuration") == configuration
        and record.get("correct") is True
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _profiler_environment(rocprof: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["BASH_FUNC_realpath%%"] = (
        f"() {{ echo {shlex.quote(str(rocprof))}; }}"
    )
    return environment


def _write_counter_config(path: Path, kernel_name: str) -> None:
    payload = "\n".join(
        (
            "# MI300A locality counter passes; rocprof reruns the target.",
            *COUNTER_PASSES,
            f"kernel: {kernel_name}",
            "",
        )
    )
    path.write_text(payload, encoding="utf-8")


def profile_layout(
    args,
    *,
    case: str,
    cache_mode: str,
    role: str,
    launch: int,
    checkpoint: Path,
) -> dict[str, object]:
    profile_dir = checkpoint.parent
    profile_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = profile_dir / "counters.csv"
    worker_json = profile_dir / "worker.json"
    counter_config = profile_dir / "counters.txt"
    _write_counter_config(counter_config, KERNEL_NAMES[case])
    if checkpoint.exists():
        checkpoint.unlink()
    if raw_csv.exists():
        raw_csv.unlink()
    command = [
        str(args.rocprof.resolve()),
        "-i",
        str(counter_config.resolve()),
        "-o",
        str(raw_csv.resolve()),
        "--timestamp",
        "on",
        sys.executable,
        str(Path(__file__).with_name("run-stage1-kernel-case.py")),
        "--case",
        case,
        "--transaction-bytes",
        str(args.transaction_bytes),
        "--candidates",
        str(args.candidates),
        "--samples",
        "1",
        "--iterations",
        "1",
        "--warmup",
        "1",
        "--temporal-mode",
        args.temporal_mode,
        "--profile-layout",
        role,
        "--profile-cache-mode",
        cache_mode,
        "--profile-warmup",
        str(args.profile_warmup),
        "--profile-iterations",
        str(args.profile_iterations),
        "--cache-thrash-bytes",
        str(args.cache_thrash_bytes),
        "--json",
        str(worker_json.resolve()),
        "--quiet",
    ]
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=_profiler_environment(args.rocprof.resolve()),
    )
    worker = json.loads(worker_json.read_text(encoding="utf-8"))
    target = worker["ranking"]["profile_target"]
    if target["role"] != role or target["cache_mode"] != cache_mode:
        raise ValueError("profile worker reported a different target")
    counters = parse_counter_csv(
        raw_csv,
        kernel_name=KERNEL_NAMES[case],
        profile_iterations=args.profile_iterations,
    )
    result = {
        "experiment": PROFILE_EXPERIMENT,
        "configuration": _profile_configuration(
            args, case, cache_mode, role, launch
        ),
        "layout": target,
        "ranking": worker["ranking"],
        "target_operand": worker["target_operand"],
        "operand_shape": worker["operand_shape"],
        "execution_layout": worker["execution_layout"],
        "process": worker["process"],
        "counter_profile": counters,
        "artifacts": {
            "raw_csv": str(raw_csv),
            "worker_json": str(worker_json),
            "counter_config": str(counter_config),
        },
        "command": command,
        "correct": bool(worker["correct"]),
    }
    _write_json(checkpoint, result)
    return result


def _quotient_comparison(ranking: dict[str, object]) -> dict[str, object]:
    default = ranking["default"]
    selected = ranking["selected"]

    def comparison(field: str, component: str | None = None):
        if component is None:
            default_value = float(default[field])
            selected_value = float(selected[field])
        else:
            default_value = float(default[field][component])
            selected_value = float(selected[field][component])
        return {
            "default": default_value,
            "selected": selected_value,
            "selected_to_default_ratio": (
                selected_value / default_value if default_value else None
            ),
            "reduction": _reduction(default_value, selected_value),
            "lower": selected_value < default_value,
        }

    return {
        "objective": comparison("quotient_score"),
        "issue": comparison("quotient_components", "issue"),
        "temporal": comparison("quotient_components", "temporal"),
    }


def _aggregate_case_cache(
    profiles: dict[str, list[dict[str, object]]]
) -> dict[str, object]:
    all_profiles = profiles["default"] + profiles["selected"]
    reference_ranking = all_profiles[0]["ranking"]
    reference_signature = (
        reference_ranking["default"]["mapping_id"],
        reference_ranking["selected"]["mapping_id"],
        reference_ranking["default"]["quotient_score"],
        reference_ranking["selected"]["quotient_score"],
    )
    for profile in all_profiles[1:]:
        ranking = profile["ranking"]
        signature = (
            ranking["default"]["mapping_id"],
            ranking["selected"]["mapping_id"],
            ranking["default"]["quotient_score"],
            ranking["selected"]["quotient_score"],
        )
        if signature != reference_signature:
            raise ValueError(
                "solver layouts changed between independent counter profiles"
            )
    default = aggregate_profiles(
        [profile["counter_profile"] for profile in profiles["default"]]
    )
    selected = aggregate_profiles(
        [profile["counter_profile"] for profile in profiles["selected"]]
    )
    hardware = compare_summaries(
        default["steady_state"], selected["steady_state"]
    )
    return {
        "target_operand": all_profiles[0]["target_operand"],
        "operand_shape": all_profiles[0]["operand_shape"],
        "laqs_made_no_change": bool(reference_ranking["laqs_made_no_change"]),
        "default_layout": {
            key: reference_ranking["default"][key]
            for key in (
                "candidate_id",
                "mapping_id",
                "layout",
                "word",
                "a_rows",
                "inner_tile_shape",
                "runs",
                "xor_count",
                "compiled_codegen",
            )
        },
        "selected_layout": {
            key: reference_ranking["selected"][key]
            for key in (
                "candidate_id",
                "mapping_id",
                "layout",
                "word",
                "a_rows",
                "inner_tile_shape",
                "runs",
                "xor_count",
                "compiled_codegen",
            )
        },
        "quotient": _quotient_comparison(reference_ranking),
        "default": default,
        "selected": selected,
        "hardware_comparison": hardware,
        "profiled_duration_speedup": (
            float(default["steady_state"]["duration_ns"])
            / float(selected["steady_state"]["duration_ns"])
        ),
        "correct": all(bool(profile["correct"]) for profile in all_profiles),
    }


def _summarize_pairs(pairs: list[tuple[str, str, dict[str, object]]]):
    by_cache_mode = {}
    for cache_mode in ("warm", "thrashed"):
        mode_pairs = [pair for pair in pairs if pair[1] == cache_mode]
        if not mode_pairs:
            continue
        changed = [
            pair
            for pair in mode_pairs
            if pair[2]["quotient"]["issue"]["lower"]
        ]
        controls = [
            pair for pair in mode_pairs if pair[2]["laqs_made_no_change"]
        ]
        metric_summaries = {}
        for metric in COUNT_METRICS:
            usable = [
                pair
                for pair in changed
                if pair[2]["hardware_comparison"][metric]["reduction"]
                is not None
            ]
            reductions = [
                float(pair[2]["hardware_comparison"][metric]["reduction"])
                for pair in usable
            ]
            predicted = [
                float(pair[2]["quotient"]["issue"]["reduction"])
                for pair in usable
            ]
            metric_summaries[metric] = {
                "eligible_pair_count": len(usable),
                "fewer_count": sum(value > 0.0 for value in reductions),
                "same_count": sum(value == 0.0 for value in reductions),
                "more_count": sum(value < 0.0 for value in reductions),
                "median_reduction": (
                    statistics.median(reductions) if reductions else None
                ),
                "spearman_with_issue_quotient_reduction": rank_correlation(
                    predicted, reductions
                ),
            }
        control_noise = {}
        for metric in COUNT_METRICS:
            reductions = [
                abs(
                    float(
                        pair[2]["hardware_comparison"][metric]["reduction"]
                    )
                )
                for pair in controls
                if pair[2]["hardware_comparison"][metric]["reduction"]
                is not None
            ]
            control_noise[metric] = {
                "pair_count": len(reductions),
                "median_absolute_reduction": (
                    statistics.median(reductions) if reductions else None
                ),
            }
        by_cache_mode[cache_mode] = {
            "completed_pair_count": len(mode_pairs),
            "lower_issue_quotient_pair_count": len(changed),
            "no_change_control_count": len(controls),
            "metrics": metric_summaries,
            "no_change_control_noise": control_noise,
        }
    return {
        "primary_predictor": "issue quotient reduction at transaction_bytes",
        "primary_hardware_metric": "l1_cache_line_accesses",
        "downstream_hardware_metrics": [
            "l1_to_l2_read_requests",
            "l2_tag_requests",
            "l2_misses",
            "hbm_read_bytes",
        ],
        "by_cache_mode": by_cache_mode,
    }


def collect_results(args) -> dict[str, object]:
    completed = {}
    missing = []
    pairs = []
    for case in CASES:
        case_record = {"cache_modes": {}}
        for cache_mode in ("warm", "thrashed"):
            profiles = {"default": [], "selected": []}
            absent = []
            for launch in range(1, args.profile_launches + 1):
                for role in ("default", "selected"):
                    checkpoint = (
                        args.results_dir
                        / f"temporal-{args.temporal_mode}"
                        / case
                        / cache_mode
                        / role
                        / f"launch-{launch}"
                        / "profile.json"
                    )
                    configuration = _profile_configuration(
                        args, case, cache_mode, role, launch
                    )
                    if not _current_profile(checkpoint, configuration):
                        absent.append(str(checkpoint))
                        continue
                    profiles[role].append(
                        json.loads(checkpoint.read_text(encoding="utf-8"))
                    )
            if absent:
                missing.extend(absent)
                continue
            aggregate = _aggregate_case_cache(profiles)
            case_record["cache_modes"][cache_mode] = aggregate
            pairs.append((case, cache_mode, aggregate))
        if case_record["cache_modes"]:
            completed[case] = case_record
    return {
        "experiment": EXPERIMENT,
        "stage": 1,
        "configuration": {
            "case_order": list(CASES),
            "cache_modes": ["warm", "thrashed"],
            "temporal_mode": args.temporal_mode,
            "transaction_bytes": args.transaction_bytes,
            "requested_retained_candidates": args.candidates,
            "profile_launches": args.profile_launches,
            "profile_warmup": args.profile_warmup,
            "profile_iterations": args.profile_iterations,
            "cache_thrash_bytes": args.cache_thrash_bytes,
            "counter_passes_per_profile": len(COUNTER_PASSES),
            "rocprof": str(args.rocprof.resolve()),
        },
        "counter_definitions": COUNTER_DEFINITIONS,
        "measurement_scope": {
            "quotient": (
                "the issue component models transaction_bytes regions for "
                "the persistent target operand only"
            ),
            "hardware": (
                "whole-kernel counters; all non-target inputs and output "
                "traffic are fixed between each default/selected pair"
            ),
            "warm": "target dispatches follow target-layout warmups",
            "thrashed": (
                "a cache-thrash dispatch precedes every target dispatch and "
                "is excluded by the rocprof kernel filter"
            ),
            "aggregation": (
                "median dispatch within each profiler launch, then median "
                "across independent profiler launches"
            ),
        },
        "completed_cases": list(completed),
        "missing_profiles": missing,
        "cases": completed,
        "summary": _summarize_pairs(pairs),
        "complete": not missing,
        "correct": all(
            pair[2]["correct"] for pair in pairs
        ),
    }


def write_csv(report: dict[str, object], path: Path) -> None:
    metrics = (*COUNT_METRICS, "duration_ns", "hbm_bandwidth_gbps")
    fieldnames = [
        "case",
        "cache_mode",
        "laqs_made_no_change",
        "objective_default",
        "objective_selected",
        "objective_reduction",
        "issue_default",
        "issue_selected",
        "issue_reduction",
        "temporal_default",
        "temporal_selected",
        "temporal_reduction",
    ]
    for metric in metrics:
        fieldnames.extend(
            (f"{metric}_default", f"{metric}_selected", f"{metric}_reduction")
        )
    fieldnames.extend(
        (
            "l2_hit_rate_percent_default",
            "l2_hit_rate_percent_selected",
            "l2_hit_rate_percentage_point_change",
        )
    )
    rows = []
    for case, case_record in report["cases"].items():
        for cache_mode, pair in case_record["cache_modes"].items():
            row = {
                "case": case,
                "cache_mode": cache_mode,
                "laqs_made_no_change": pair["laqs_made_no_change"],
            }
            for component in ("objective", "issue", "temporal"):
                comparison = pair["quotient"][component]
                for field in ("default", "selected", "reduction"):
                    row[f"{component}_{field}"] = comparison[field]
            for metric in metrics:
                comparison = pair["hardware_comparison"][metric]
                for field in ("default", "selected", "reduction"):
                    row[f"{metric}_{field}"] = comparison[field]
            hit_rate = pair["hardware_comparison"]["l2_hit_rate_percent"]
            row["l2_hit_rate_percent_default"] = hit_rate["default"]
            row["l2_hit_rate_percent_selected"] = hit_rate["selected"]
            row["l2_hit_rate_percentage_point_change"] = hit_rate[
                "percentage_point_change"
            ]
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(report: dict[str, object]) -> None:
    print(
        "case             cache      issue Q       TCP lines      "
        "TCP->TCC reads    L2 misses"
    )
    for case, case_record in report["cases"].items():
        for cache_mode, pair in case_record["cache_modes"].items():
            predicted = pair["quotient"]["issue"]["reduction"]
            tcp_lines = pair["hardware_comparison"][
                "l1_cache_line_accesses"
            ]["reduction"]
            tcp_to_tcc = pair["hardware_comparison"][
                "l1_to_l2_read_requests"
            ]["reduction"]
            l2_misses = pair["hardware_comparison"]["l2_misses"][
                "reduction"
            ]
            predicted_text = "n/a" if predicted is None else f"{100*predicted:+7.2f}%"

            def percent(value):
                return "n/a" if value is None else f"{100*value:+7.2f}%"

            print(
                f"{case:16s} {cache_mode:9s} {predicted_text:>9s} "
                f"{percent(tcp_lines):>15s} {percent(tcp_to_tcc):>17s} "
                f"{percent(l2_misses):>12s}"
            )
    if report["missing_profiles"]:
        print(f"Pending profiles: {len(report['missing_profiles'])}")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["all"])
    parser.add_argument(
        "--cache-modes",
        nargs="+",
        choices=("warm", "thrashed"),
        default=["warm", "thrashed"],
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("triton/results/stage1-locality-counter-profiles"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("triton/results/stage1-locality-counters.json"),
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-profiles", type=positive_integer)
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--profile-launches", type=positive_integer, default=1)
    parser.add_argument("--profile-warmup", type=positive_integer, default=5)
    parser.add_argument("--profile-iterations", type=positive_integer, default=20)
    parser.add_argument(
        "--cache-thrash-bytes", type=positive_integer, default=256 << 20
    )
    parser.add_argument(
        "--temporal-mode",
        choices=("issue", "union", "split"),
        default="issue",
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
    requested = list(CASES) if args.cases == ["all"] else args.cases
    unknown = sorted(set(requested) - set(CASES))
    if unknown:
        raise ValueError(f"unknown Stage-1 kernel cases: {unknown}")
    args.results_dir = args.results_dir.resolve()
    args.json = args.json.resolve()
    args.rocprof = args.rocprof.resolve()
    tasks = []
    for case in requested:
        for cache_mode in args.cache_modes:
            for launch in range(1, args.profile_launches + 1):
                roles = (
                    ("default", "selected")
                    if launch % 2
                    else ("selected", "default")
                )
                for role in roles:
                    checkpoint = (
                        args.results_dir
                        / f"temporal-{args.temporal_mode}"
                        / case
                        / cache_mode
                        / role
                        / f"launch-{launch}"
                        / "profile.json"
                    )
                    configuration = _profile_configuration(
                        args, case, cache_mode, role, launch
                    )
                    if (
                        _current_profile(checkpoint, configuration)
                        and not args.rerun
                    ):
                        continue
                    tasks.append(
                        (case, cache_mode, role, launch, checkpoint)
                    )
    if args.max_profiles is not None:
        tasks = tasks[: args.max_profiles]
    for index, (case, cache_mode, role, launch, checkpoint) in enumerate(
        tasks, 1
    ):
        print(
            f"[{index}/{len(tasks)}] {case} {cache_mode} {role} "
            f"launch {launch}/{args.profile_launches}",
            file=sys.stderr,
            flush=True,
        )
        profile_layout(
            args,
            case=case,
            cache_mode=cache_mode,
            role=role,
            launch=launch,
            checkpoint=checkpoint,
        )
        report = collect_results(args)
        _write_json(args.json, report)
        write_csv(report, args.csv or args.json.with_suffix(".csv"))
    report = collect_results(args)
    _write_json(args.json, report)
    csv_path = args.csv.resolve() if args.csv else args.json.with_suffix(".csv")
    write_csv(report, csv_path)
    if not args.quiet:
        _print_summary(report)
        print(f"Wrote {args.json}")
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
