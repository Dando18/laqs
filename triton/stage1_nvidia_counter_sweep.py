"""Resumable Nsight Compute profiling for Stage-1 counter panels."""

from __future__ import annotations

import csv
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence
from urllib.parse import unquote, urlparse

from stage1_counter_analysis import linear_fit, rank_correlation
from stage1_counter_sweep import (
    KERNEL_NAMES,
    _current_record,
    _gesummv_theory,
    _panel_configuration,
    _profile_checkpoint,
    _worker_command,
    prepare_panel,
    write_json,
)
from stage1_nvidia_counter_analysis import (
    COUNTER_METRICS,
    FIRST_LEVEL_COUNTER,
    NATIVE_UNIT,
    SUMMARY_METRICS,
    aggregate_profiles,
    counter_definitions,
    parse_counter_csv,
)


def _matrix_triton_python() -> Path:
    direct_url = metadata.distribution("triton").read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError("the Matrix Triton install has no direct_url.json")
    source_url = json.loads(direct_url).get("url", "")
    parsed = urlparse(source_url)
    if parsed.scheme != "file":
        raise RuntimeError("the Matrix Triton install is not a local checkout")
    source = Path(unquote(parsed.path)).resolve()
    python_root = source / "python"
    if not (python_root / "triton" / "__init__.py").is_file():
        raise RuntimeError(f"invalid Matrix Triton checkout: {source}")
    shared_source = Path(__file__).with_name("triton-lang").resolve()
    if source == shared_source:
        raise RuntimeError("Matrix profiling must not use Tuolumne's Triton source")
    return python_root


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    entries = [
        str(_matrix_triton_python()),
        str(Path(__file__).resolve().parents[1]),
    ]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _profile_configuration(
    args,
    *,
    candidate: dict[str, object],
    launch: int,
) -> dict[str, object]:
    return {
        "profile_schema": 6,
        "profiler_backend": "nvidia_ncu",
        "case": args.case,
        "kernel_name": KERNEL_NAMES[args.case],
        "candidate_id": candidate["candidate_id"],
        "mapping_id": candidate["mapping_id"],
        "a_rows": candidate["a_rows"],
        "quotient_score": candidate["quotient_score"],
        "profile_launch": launch,
        "cache_mode": "warm",
        "temporal_mode": "issue",
        "transaction_bytes": args.transaction_bytes,
        "requested_retained_candidates": args.candidates,
        "stratification": getattr(args, "stratification", None),
        "gemv_k": getattr(args, "gemv_k", None),
        "profile_warmup": args.profile_warmup,
        "profile_iterations": args.profile_iterations,
        "native_counter": FIRST_LEVEL_COUNTER,
        "native_unit": NATIVE_UNIT,
        "counter_native_scale_bytes": 32,
        "quotient_scale_matches_native_counter": args.transaction_bytes == 32,
        "counter_metrics": list(COUNTER_METRICS),
        "ncu": str(args.ncu.resolve()),
        "ncu_cache_control": "none",
    }


def profile_candidate(
    args,
    *,
    candidate: dict[str, object],
    launch: int,
    checkpoint: Path,
    experiment: str,
) -> dict[str, object]:
    configuration = _profile_configuration(
        args, candidate=candidate, launch=launch
    )
    current = _current_record(
        checkpoint,
        experiment=experiment,
        configuration=configuration,
    )
    if current is not None and not args.rerun:
        return current

    profile_dir = checkpoint.parent
    profile_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = profile_dir / "counters.csv"
    worker_json = profile_dir / "worker.json"
    for stale in (checkpoint, raw_csv, worker_json):
        if stale.exists():
            stale.unlink()
    worker_command = _worker_command(
        args,
        "--profile-rows",
        *(str(value) for value in candidate["a_rows"]),
        "--profile-candidate-id",
        str(candidate["candidate_id"]),
        "--profile-quotient-score",
        str(candidate["quotient_score"]),
        "--profile-cache-mode",
        "warm",
        "--profile-warmup",
        str(args.profile_warmup),
        "--profile-iterations",
        str(args.profile_iterations),
        "--json",
        str(worker_json.resolve()),
        "--quiet",
    )
    command = [
        str(args.ncu.resolve()),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--metrics",
        ",".join(COUNTER_METRICS),
        "--kernel-name-base",
        "function",
        "--kernel-name",
        f"regex:{KERNEL_NAMES[args.case]}",
        "--cache-control",
        "none",
        "--target-processes",
        "application-only",
        "--log-file",
        str(raw_csv.resolve()),
        *worker_command,
    ]
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=_worker_environment(),
    )
    worker = json.loads(worker_json.read_text(encoding="utf-8"))
    target = worker["ranking"]["profile_target"]
    if (
        target["candidate_id"] != candidate["candidate_id"]
        or target["mapping_id"] != candidate["mapping_id"]
        or target["a_rows"] != candidate["a_rows"]
        or target["cache_mode"] != "warm"
        or target["issue_quotient_score_verified"] is not True
    ):
        raise ValueError("profile worker reported a different candidate")
    validation = target["structural_validation"]
    if not validation["accepted"]:
        raise ValueError("profile worker rejected candidate structure")
    counters = parse_counter_csv(
        raw_csv,
        kernel_name=KERNEL_NAMES[args.case],
        profile_iterations=args.profile_iterations,
    )
    record = {
        "experiment": experiment,
        "configuration": configuration,
        "candidate": candidate,
        "layout": target,
        "target_operand": worker["target_operand"],
        "operand_shape": worker["operand_shape"],
        "execution_layout": worker["execution_layout"],
        "process": worker["process"],
        "counter_profile": counters,
        "artifacts": {
            "raw_csv": str(raw_csv),
            "worker_json": str(worker_json),
        },
        "command": command,
        "correct": bool(worker["correct"] and validation["accepted"]),
    }
    write_json(checkpoint, record)
    return record


def profile_tasks(
    args,
    candidates: Sequence[dict[str, object]],
    *,
    profile_experiment: str,
) -> list[tuple[dict[str, object], int, Path]]:
    tasks = []
    for launch in range(1, args.profile_launches + 1):
        shift = (launch - 1) % len(candidates)
        rotated = [*candidates[shift:], *candidates[:shift]]
        for candidate in rotated:
            checkpoint = _profile_checkpoint(
                args.results_dir, candidate, launch
            )
            configuration = _profile_configuration(
                args, candidate=candidate, launch=launch
            )
            current = _current_record(
                checkpoint,
                experiment=profile_experiment,
                configuration=configuration,
            )
            if current is None or args.rerun:
                tasks.append((candidate, launch, checkpoint))
    if args.max_profiles is not None:
        tasks = tasks[: args.max_profiles]
    return tasks


def collect_report(
    args,
    *,
    panel_record: dict[str, object],
    panel_mode: str,
    experiment: str,
    profile_experiment: str,
) -> dict[str, object]:
    completed = []
    missing = []
    candidates = panel_record["panel"]["candidates"]
    for candidate in candidates:
        profiles = []
        profile_paths = []
        for launch in range(1, args.profile_launches + 1):
            checkpoint = _profile_checkpoint(
                args.results_dir, candidate, launch
            )
            configuration = _profile_configuration(
                args, candidate=candidate, launch=launch
            )
            profile = _current_record(
                checkpoint,
                experiment=profile_experiment,
                configuration=configuration,
            )
            if profile is None:
                missing.append(str(checkpoint))
            else:
                profiles.append(profile)
                profile_paths.append(str(checkpoint))
        record = {
            **candidate,
            "completed_profile_launches": len(profiles),
            "requested_profile_launches": args.profile_launches,
            "complete": len(profiles) == args.profile_launches,
        }
        if profiles:
            aggregate = aggregate_profiles(
                [profile["counter_profile"] for profile in profiles]
            )
            validations = [
                profile["layout"]["structural_validation"]
                for profile in profiles
            ]
            record.update(
                {
                    "counters": aggregate,
                    "structural_validation": {
                        "all_accepted": all(
                            validation["accepted"]
                            for validation in validations
                        ),
                        "execution_layout_matches_reference": all(
                            validation["execution_layout_matches_reference"]
                            for validation in validations
                        ),
                        "load_instruction_structure_matches_reference": all(
                            validation[
                                "load_instruction_structure_matches_reference"
                            ]
                            for validation in validations
                        ),
                        "load_instruction_structure_nonempty": all(
                            validation.get(
                                "load_instruction_structure_nonempty", False
                            )
                            for validation in validations
                        ),
                        "store_instruction_structure_matches_reference": all(
                            validation.get(
                                "store_instruction_structure_matches_reference",
                                True,
                            )
                            for validation in validations
                        ),
                        "no_spills": all(
                            validation["no_candidate_spills"]
                            for validation in validations
                        ),
                        "by_launch": validations,
                    },
                    "compiled_codegen_by_launch": [
                        profile["layout"]["compiled_codegen"]
                        for profile in profiles
                    ],
                    "profile_checkpoints": profile_paths,
                }
            )
        completed.append(record)

    fit_candidates = [
        candidate for candidate in completed if candidate["complete"]
    ]
    x = [float(candidate["quotient_score"]) for candidate in fit_candidates]
    y = [
        float(
            candidate["counters"]["steady_state"][
                "first_level_memory_accesses"
            ]
        )
        for candidate in fit_candidates
    ]
    statistics = {
        "observation_count": len(x),
        "primary_hardware_metric": "first_level_memory_accesses",
        "primary_hardware_counter": FIRST_LEVEL_COUNTER,
        "native_counter": FIRST_LEVEL_COUNTER,
        "native_unit": NATIVE_UNIT,
        "tie_aware_spearman": rank_correlation(x, y),
        "free_intercept_linear_fit": (
            linear_fit(x, y) if len(set(x)) > 1 else None
        ),
    }
    return {
        "experiment": experiment,
        "stage": 1,
        "case": args.case,
        "kernel_name": KERNEL_NAMES[args.case],
        "target_operand": panel_record["target_operand"],
        "operand_shape": panel_record["operand_shape"],
        "configuration": {
            **_panel_configuration(args, panel_mode),
            "profiler_backend": "nvidia_ncu",
            "quotient_notation": f"Q_{args.transaction_bytes}B",
            "profile_launches": args.profile_launches,
            "profile_warmup": args.profile_warmup,
            "profile_iterations": args.profile_iterations,
            "cache_mode": "warm",
            "candidate_order": "cyclic rotation by profiler launch",
            "profiler_invocations_per_profile": 1,
            "native_counter": FIRST_LEVEL_COUNTER,
            "native_unit": NATIVE_UNIT,
            "counter_native_scale_bytes": 32,
            "quotient_scale_matches_native_counter": (
                args.transaction_bytes == 32
            ),
            "quotient_scale_selection_rule": (
                "match the quotient byte scale to the measured counter's "
                "documented native accounting unit"
            ),
            "counter_metrics": list(COUNTER_METRICS),
            "ncu": str(args.ncu.resolve()),
            "ncu_cache_control": "none",
        },
        "measurement_scope": {
            "quotient": (
                f"Q_{args.transaction_bytes}B issue-only target-operand "
                "transaction quotient"
            ),
            "hardware": (
                "whole-kernel NVIDIA first-level, L2, and DRAM read-work "
                "counters"
            ),
            "aggregation": (
                "median of the final target launches within a profiler "
                "process, then median across independent launches"
            ),
            "process_isolation": "one ncu invocation per mapping and launch",
        },
        "native_counter": FIRST_LEVEL_COUNTER,
        "native_unit": NATIVE_UNIT,
        "counter_definitions": counter_definitions(),
        "panel": panel_record["panel"],
        "theory_validation": _gesummv_theory(args, panel_record, panel_mode),
        "candidates": completed,
        "statistics": statistics,
        "missing_profiles": missing,
        "complete": not missing,
        "correct": all(
            candidate.get("structural_validation", {}).get(
                "all_accepted", False
            )
            for candidate in completed
            if candidate["complete"]
        ),
    }


def write_csv(report: dict[str, object], path: Path) -> None:
    fields = [
        "case",
        "stratification",
        "candidate_id",
        "mapping_id",
        "sample_index",
        "candidate_pool_index",
        "sampling_origin",
        "grammar",
        "inner_tile_shape",
        "inner_word",
        "quotient_score",
        "native_counter",
        "native_unit",
        "address_expression_runs",
        "xor_count",
        "completed_profile_launches",
        "complete",
        *COUNTER_METRICS,
    ]
    for metric in SUMMARY_METRICS:
        fields.extend((metric, f"{metric}_min", f"{metric}_max"))
    rows = []
    for candidate in report["candidates"]:
        row = {
            field: candidate.get(field)
            for field in fields
            if field not in SUMMARY_METRICS
        }
        row["case"] = report["case"]
        row["stratification"] = report["panel"].get("stratification", {}).get(
            "mode", "none"
        )
        row["native_counter"] = report["native_counter"]
        row["native_unit"] = report["native_unit"]
        if "counters" in candidate:
            summary = candidate["counters"]["steady_state"]
            for native_metric in COUNTER_METRICS:
                row[native_metric] = summary["native_counters"][native_metric]
            for metric in SUMMARY_METRICS:
                row[metric] = summary[metric]
                values = summary[f"{metric}_by_launch"]
                row[f"{metric}_min"] = min(values) if values else None
                row[f"{metric}_max"] = max(values) if values else None
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    args,
    *,
    panel_mode: str,
    experiment: str,
    panel_experiment: str,
    profile_experiment: str,
) -> dict[str, object]:
    args.results_dir = args.results_dir.resolve()
    args.json = args.json.resolve()
    args.ncu = args.ncu.resolve()
    if not args.ncu.is_file():
        raise FileNotFoundError(f"Nsight Compute CLI not found: {args.ncu}")
    environment = _worker_environment()
    panel_checkpoint = args.results_dir / "panel.json"
    panel_record = prepare_panel(
        args,
        panel_mode=panel_mode,
        checkpoint=panel_checkpoint,
        experiment=panel_experiment,
        worker_environment=environment,
    )
    candidates = panel_record["panel"]["candidates"]
    tasks = profile_tasks(
        args, candidates, profile_experiment=profile_experiment
    )
    for index, (candidate, launch, checkpoint) in enumerate(tasks, 1):
        print(
            f"[{index}/{len(tasks)}] {args.case} "
            f"{candidate['candidate_id']} Q={candidate['quotient_score']:g} "
            f"launch {launch}/{args.profile_launches}",
            file=sys.stderr,
            flush=True,
        )
        profile_candidate(
            args,
            candidate=candidate,
            launch=launch,
            checkpoint=checkpoint,
            experiment=profile_experiment,
        )
        report = collect_report(
            args,
            panel_record=panel_record,
            panel_mode=panel_mode,
            experiment=experiment,
            profile_experiment=profile_experiment,
        )
        write_json(args.json, report)
        write_csv(report, args.csv or args.json.with_suffix(".csv"))
    report = collect_report(
        args,
        panel_record=panel_record,
        panel_mode=panel_mode,
        experiment=experiment,
        profile_experiment=profile_experiment,
    )
    write_json(args.json, report)
    write_csv(report, args.csv or args.json.with_suffix(".csv"))
    return report


def print_summary(report: dict[str, object]) -> None:
    print("tile        word          quotient  first-level read units  profiles")
    for candidate in report["candidates"]:
        tile = "x".join(str(value) for value in candidate["inner_tile_shape"])
        sectors = "pending"
        if "counters" in candidate:
            value = candidate["counters"]["steady_state"][
                "first_level_memory_accesses"
            ]
            sectors = f"{value:,.1f}"
        print(
            f"{tile:11s} {candidate['inner_word']:13s} "
            f"{candidate['quotient_score']:12,.0f} {sectors:>23s} "
            f"{candidate['completed_profile_launches']}/"
            f"{candidate['requested_profile_launches']}"
        )
    print(
        f"Complete: {report['complete']}; pending profiles: "
        f"{len(report['missing_profiles'])}"
    )
