"""Resumable profiling machinery for Stage-1 quotient counter panels."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from stage1_counter_analysis import (
    COUNTER_DEFINITIONS,
    SUMMARY_METRICS,
    aggregate_profiles,
    linear_fit,
    parse_counter_csv,
    rank_correlation,
)


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
    "pmc : TCP_TOTAL_READ_sum TCC_READ_sum",
    "pmc : FETCH_SIZE WRITE_SIZE",
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _current_record(
    path: Path,
    *,
    experiment: str,
    configuration: dict[str, object],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("experiment") != experiment
        or record.get("configuration") != configuration
        or record.get("correct") is not True
    ):
        return None
    return record


def _profiler_environment(rocprof: Path) -> dict[str, str]:
    environment = _worker_environment()
    environment["BASH_FUNC_realpath%%"] = (
        f"() {{ echo {shlex.quote(str(rocprof))}; }}"
    )
    return environment


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = Path(__file__).with_name("triton-lang") / "python"
    existing = environment.get("PYTHONPATH")
    entries = [str(source_root.resolve())]
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _write_counter_config(path: Path, kernel_name: str) -> None:
    path.write_text(
        "\n".join(
            (
                "# MI300A locality counter passes; rocprof reruns the target.",
                *COUNTER_PASSES,
                f"kernel: {kernel_name}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _worker_command(args, *extra: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run-stage1-kernel-case.py")),
        "--case",
        args.case,
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
        "issue",
    ]
    if hasattr(args, "platform"):
        command.extend(("--counter-platform", str(args.platform)))
    if getattr(args, "gemv_k", None) is not None:
        command.extend(("--gemv-k", str(args.gemv_k)))
    command.extend(extra)
    return command


def _panel_configuration(args, panel_mode: str) -> dict[str, object]:
    configuration = {
        "candidate_panel_schema": (
            2 if panel_mode.startswith("experiment") else 1
        ),
        "case": args.case,
        "panel_mode": panel_mode,
        "panel_tile_shape": (
            list(args.tile_shape) if panel_mode == "fixed_tile_levels" else None
        ),
        "temporal_mode": "issue",
        "transaction_bytes": args.transaction_bytes,
        "requested_retained_candidates": args.candidates,
    }
    if panel_mode == "random_layouts" or panel_mode.startswith("experiment"):
        configuration.update(
            {
                "random_layout_samples": args.layouts,
                "random_seed": args.seed,
            }
        )
    if panel_mode.startswith("experiment"):
        configuration.update(
            {
                "candidate_panel_schema": 3,
                "counter_score_profile_schema": 2,
                "platform": args.platform,
                "stratification": getattr(args, "stratification", "all"),
                "stratification_pool_multiplier": getattr(
                    args, "pool_multiplier", 20
                ),
            }
        )
    if getattr(args, "gemv_k", None) is not None:
        configuration["gemv_k"] = args.gemv_k
    return configuration


def prepare_panel(
    args,
    *,
    panel_mode: str,
    checkpoint: Path,
    experiment: str,
    worker_environment: dict[str, str] | None = None,
) -> dict[str, object]:
    configuration = _panel_configuration(args, panel_mode)
    current = _current_record(
        checkpoint,
        experiment=experiment,
        configuration=configuration,
    )
    if current is not None and not args.rerun:
        return current

    worker_json = checkpoint.parent / "worker.json"
    extra = ["--counter-panel", panel_mode]
    if panel_mode == "fixed_tile_levels":
        extra.extend(
            ["--panel-tile-shape", *(str(value) for value in args.tile_shape)]
        )
    elif panel_mode == "random_layouts" or panel_mode.startswith("experiment"):
        extra.extend(
            (
                "--panel-samples",
                str(args.layouts),
                "--panel-seed",
                str(args.seed),
            )
        )
        if panel_mode.startswith("experiment"):
            extra.extend(
                (
                    "--panel-stratification",
                    str(getattr(args, "stratification", "all")),
                    "--panel-pool-multiplier",
                    str(getattr(args, "pool_multiplier", 20)),
                )
            )
    extra.extend(("--json", str(worker_json.resolve()), "--quiet"))
    command = _worker_command(args, *extra)
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=(
            _worker_environment()
            if worker_environment is None
            else worker_environment
        ),
    )
    worker = json.loads(worker_json.read_text(encoding="utf-8"))
    panel = worker["ranking"]["counter_panel"]
    if panel is None or panel["mode"] != panel_mode:
        raise ValueError("counter-panel worker did not produce the requested panel")
    if not panel["candidates"]:
        raise ValueError("counter-panel worker produced no candidates")
    record = {
        "experiment": experiment,
        "configuration": configuration,
        "panel": panel,
        "kernel": worker["kernel"],
        "kernel_name": KERNEL_NAMES[args.case],
        "target_operand": worker["target_operand"],
        "operand_shape": worker["operand_shape"],
        "execution_layout": worker["execution_layout"],
        "reference_layout": worker["ranking"]["default"],
        "process": worker["process"],
        "artifacts": {"worker_json": str(worker_json)},
        "command": command,
        "correct": bool(worker["correct"]),
    }
    write_json(checkpoint, record)
    return record


def _profile_configuration(
    args,
    *,
    candidate: dict[str, object],
    launch: int,
) -> dict[str, object]:
    return {
        "profile_schema": 3,
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
        "counter_passes": list(COUNTER_PASSES),
        "rocprof": str(args.rocprof.resolve()),
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
    counter_config = profile_dir / "counters.txt"
    _write_counter_config(counter_config, KERNEL_NAMES[args.case])
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
        str(args.rocprof.resolve()),
        "-i",
        str(counter_config.resolve()),
        "-o",
        str(raw_csv.resolve()),
        "--timestamp",
        "on",
        *worker_command,
    ]
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=_profiler_environment(args.rocprof.resolve()),
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
            "counter_config": str(counter_config),
        },
        "command": command,
        "correct": bool(worker["correct"] and validation["accepted"]),
    }
    write_json(checkpoint, record)
    return record


def _profile_checkpoint(
    results_dir: Path, candidate: dict[str, object], launch: int
) -> Path:
    return (
        results_dir
        / str(candidate["candidate_id"])
        / f"launch-{launch}"
        / "profile.json"
    )


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


def _gesummv_theory(
    args, panel_record: dict[str, object], panel_mode: str
) -> dict[str, object] | None:
    if (
        args.case != "gesummv"
        or panel_mode != "fixed_tile_levels"
        or tuple(args.tile_shape) != (64, 64)
    ):
        return None
    panel = panel_record["panel"]
    shape = panel_record["operand_shape"]
    if panel["enumerated_word_count"] != 924:
        raise ValueError("GESUMMV 64x64 grammar did not enumerate 924 words")
    if panel["unique_mapping_count"] != 924:
        raise ValueError("GESUMMV 64x64 grammar did not yield 924 mappings")

    execution = panel_record["execution_layout"]
    output_shape = execution["output_shape"]
    if len(output_shape) != 1 or int(output_shape[0]) != 64:
        raise ValueError("GESUMMV theory requires a 64-element output layout")
    lane_bits = []
    for basis in execution["bases"]["lane"]:
        if len(basis) != 1:
            raise ValueError("GESUMMV lane basis must be one-dimensional")
        value = int(basis[0])
        if value <= 0 or value & (value - 1):
            raise ValueError("GESUMMV lane basis must contain powers of two")
        lane_bits.append(value.bit_length() - 1)
    lane_width = int(execution["input_sizes"]["lane"])
    register_width = int(execution["input_sizes"]["register"])
    if lane_width != 1 << len(lane_bits):
        raise ValueError("GESUMMV lane basis disagrees with the lane width")
    if lane_width * register_width != 64:
        raise ValueError("GESUMMV lane/register layout does not cover 64 outputs")
    if int(shape[0]) % 64:
        raise ValueError("GESUMMV row count must be divisible by 64")

    transaction_elements, remainder = divmod(args.transaction_bytes, 4)
    if (
        remainder
        or transaction_elements <= 0
        or transaction_elements & (transaction_elements - 1)
    ):
        raise ValueError("GESUMMV transaction scale must be power-of-two FP32")
    low_address_dimension = transaction_elements.bit_length() - 1
    dynamic_cohorts = (
        (int(shape[0]) // 64) * int(shape[1]) * register_width
    )
    candidates = []
    predicted_scores = []
    for candidate in panel["candidates"]:
        physical_rows = [int(value) for value in candidate["a_rows"]]
        low_rows = physical_rows[:low_address_dimension]
        low_lane_bits = sum(
            any(row & (1 << bit) for row in low_rows)
            for bit in lane_bits
        )
        cardinality = 1 << (len(lane_bits) - low_lane_bits)
        predicted = dynamic_cohorts * cardinality
        if float(predicted) != float(candidate["quotient_score"]):
            raise ValueError(
                "GESUMMV candidate quotient disagrees with its execution layout"
            )
        predicted_scores.append(predicted)
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "low_physical_row_bits": low_lane_bits,
                "issue_quotient_cardinality": cardinality,
                "dynamic_issue_cohorts": dynamic_cohorts,
                "predicted_quotient_score": predicted,
            }
        )
    expected_scores = sorted(set(predicted_scores))
    if [float(value) for value in panel["quotient_levels"]] != [
        float(value) for value in expected_scores
    ]:
        raise ValueError("GESUMMV quotient levels disagree with the model")
    return {
        "low_address_dimension": low_address_dimension,
        "row_issue_dimension": len(lane_bits),
        "lane_row_bits": lane_bits,
        "register_issue_count": register_width,
        "dynamic_issue_cohorts": dynamic_cohorts,
        "expected_quotient_scores": expected_scores,
        "candidates": candidates,
        "verified": True,
    }


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
                            validation[
                                "execution_layout_matches_reference"
                            ]
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

    fit_candidates = [candidate for candidate in completed if candidate["complete"]]
    x = [float(candidate["quotient_score"]) for candidate in fit_candidates]
    y = [
        float(candidate["counters"]["steady_state"]["l1_cache_line_accesses"])
        for candidate in fit_candidates
    ]
    statistics = {
        "observation_count": len(x),
        "primary_hardware_metric": "l1_cache_line_accesses",
        "primary_hardware_counter": "TCP_TOTAL_CACHE_ACCESSES_sum",
        "native_counter": "TCP_TOTAL_CACHE_ACCESSES_sum",
        "native_unit": "64-byte vL1D access",
        "tie_aware_spearman": rank_correlation(x, y),
        "free_intercept_linear_fit": linear_fit(x, y) if len(set(x)) > 1 else None,
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
            "profiler_backend": "amd_rocprof",
            "quotient_notation": f"Q_{args.transaction_bytes}B",
            "profile_launches": args.profile_launches,
            "profile_warmup": args.profile_warmup,
            "profile_iterations": args.profile_iterations,
            "cache_mode": "warm",
            "candidate_order": "cyclic rotation by profiler launch",
            "counter_passes_per_profile": len(COUNTER_PASSES),
            "native_counter": "TCP_TOTAL_CACHE_ACCESSES_sum",
            "native_unit": "64-byte vL1D access",
            "counter_native_scale_bytes": 64,
            "quotient_scale_matches_native_counter": (
                args.transaction_bytes == 64
            ),
            "quotient_scale_selection_rule": (
                "match the quotient byte scale to the measured counter's "
                "documented native accounting unit"
            ),
            "rocprof": str(args.rocprof.resolve()),
        },
        "measurement_scope": {
            "quotient": (
                f"Q_{args.transaction_bytes}B issue-only target-operand "
                "transaction quotient"
            ),
            "hardware": "whole-kernel counters with all other operands fixed",
            "aggregation": (
                "median of the final target dispatches within a profiler "
                "launch, then median across independent launches"
            ),
            "process_isolation": "one rocprof invocation per mapping and launch",
        },
        "native_counter": "TCP_TOTAL_CACHE_ACCESSES_sum",
        "native_unit": "64-byte vL1D access",
        "counter_definitions": COUNTER_DEFINITIONS,
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
        "inner_a_rows",
        "a_rows",
        "quotient_score",
        "address_expression_runs",
        "xor_count",
        "completed_profile_launches",
        "complete",
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
        if "counters" in candidate:
            summary = candidate["counters"]["steady_state"]
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
    args.rocprof = args.rocprof.resolve()
    panel_checkpoint = args.results_dir / "panel.json"
    panel_record = prepare_panel(
        args,
        panel_mode=panel_mode,
        checkpoint=panel_checkpoint,
        experiment=panel_experiment,
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
    print("tile        word          quotient       TCP cache lines  profiles")
    for candidate in report["candidates"]:
        tile = "x".join(str(value) for value in candidate["inner_tile_shape"])
        tcp = "pending"
        if "counters" in candidate:
            value = candidate["counters"]["steady_state"][
                "l1_cache_line_accesses"
            ]
            tcp = f"{value:,.1f}"
        print(
            f"{tile:11s} {candidate['inner_word']:13s} "
            f"{candidate['quotient_score']:12,.0f} {tcp:>17s} "
            f"{candidate['completed_profile_launches']}/"
            f"{candidate['requested_profile_launches']}"
        )
    print(
        f"Complete: {report['complete']}; pending profiles: "
        f"{len(report['missing_profiles'])}"
    )
