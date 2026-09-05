#!/usr/bin/env python3
"""Rebuild automatic graphs and rescore saved experiment counters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (str(TRITON_ROOT), str(REPOSITORY), str(EXPERIMENT_ROOT))

from analyze import analyze_report
from layout_panels import TAU_PROFILES
from run_stage1_kernel_cases import CASES
from stage1_common import positive_integer
from stage1_counter_analysis import linear_fit, rank_correlation
from stage1_counter_sweep import KERNEL_NAMES, _worker_command, write_json


MEASUREMENT_FIELDS = (
    "completed_profile_launches",
    "requested_profile_launches",
    "complete",
    "counters",
    "structural_validation",
    "compiled_codegen_by_launch",
    "profile_checkpoints",
)


def _counter_report(
    root: Path,
    experiment: int,
    platform: str,
    stratification: str,
    case: str,
) -> Path:
    if root.is_file():
        return root
    return (
        root
        / f"experiment-{experiment}"
        / platform
        / f"stratified-{stratification}"
        / case
        / "report.json"
    )


def _expected_profile(platform: str) -> str:
    if not TAU_PROFILES.is_file():
        return f"automatic-bootstrap-{platform}-v1"
    document = json.loads(TAU_PROFILES.read_text(encoding="utf-8"))
    return str(document["platforms"][platform]["profile_id"])


def _outputs_current(args) -> bool:
    expected_profile = _expected_profile(args.platform)
    for experiment in (1, 2, 3):
        path = (
            args.results_root
            / f"experiment-{experiment}"
            / args.platform
            / f"stratified-{args.stratification}"
            / args.case
            / "report.json"
        )
        plot = (
            args.plots_root
            / f"experiment-{experiment}"
            / args.platform
            / f"stratified-{args.stratification}"
            / f"{args.case}.pdf"
        )
        if (
            not path.is_file()
            or not path.with_name("counter-data.csv").is_file()
            or not path.with_name("analysis.json").is_file()
            or not plot.is_file()
        ):
            return False
        report = json.loads(path.read_text(encoding="utf-8"))
        source = _counter_report(
            args.counter_source,
            experiment,
            args.platform,
            args.stratification,
            args.case,
        ).resolve()
        graph = report.get("panel", {}).get("score_profile", {}).get(
            "component_model", {}
        )
        if (
            graph.get("construction")
            != "automatic_post_coalescing_manifest_universal_v1"
            or report["panel"].get("requested_sample_count") != args.layouts
            or report["panel"].get("random_seed") != args.seed
            or report["panel"].get("candidate_pool_multiplier")
            != args.pool_multiplier
            or report["panel"].get("stratification", {}).get("mode")
            != args.stratification
            or report["panel"]["score_profile"].get("profile_id")
            != expected_profile
            or Path(report.get("counter_source", {}).get("report", "")).resolve()
            != source
        ):
            return False
        if any(
            not Path(checkpoint).is_file()
            for candidate in report["candidates"]
            for checkpoint in candidate.get("profile_checkpoints", ())
        ):
            return False
    return True


def _worker_environment(platform: str) -> dict[str, str]:
    if platform == "tuolumne":
        from stage1_counter_sweep import _worker_environment
    else:
        from stage1_nvidia_counter_sweep import _worker_environment

    return _worker_environment()


def _write_counter_csv(platform: str, report, path: Path) -> None:
    if platform == "tuolumne":
        from stage1_counter_sweep import write_csv
    else:
        from stage1_nvidia_counter_sweep import write_csv

    write_csv(report, path)


def _automatic_panels(args) -> tuple[dict[str, object], Path, list[str]]:
    cache = (
        args.results_root
        / "automatic-graphs"
        / args.platform
        / f"stratified-{args.stratification}"
        / args.case
    )
    worker_json = cache / "worker.json"
    command = _worker_command(
        args,
        "--counter-panel",
        "experiments123",
        "--panel-samples",
        str(args.layouts),
        "--panel-seed",
        str(args.seed),
        "--panel-stratification",
        args.stratification,
        "--panel-pool-multiplier",
        str(args.pool_multiplier),
        "--automatic-temporal-windows",
        *(str(value) for value in args.temporal_windows),
        "--json",
        str(worker_json.resolve()),
        "--quiet",
    )
    worker_json.parent.mkdir(parents=True, exist_ok=True)
    environment = _worker_environment(args.platform)
    source_root = Path(environment["PYTHONPATH"].split(os.pathsep)[0])
    plugin = source_root / "triton" / "plugins" / "libLAQSTritonAccessManifest.so"
    configured_plugin = environment.get("LAQS_TRITON_PLUGIN_PATH")
    if configured_plugin is None and not plugin.is_file():
        raise FileNotFoundError(
            f"automatic manifest plugin is missing from {plugin}; rerun the "
            f"{args.platform} Triton setup before rescoring"
        )
    subprocess.run(
        command,
        check=True,
        cwd=REPOSITORY,
        env=environment,
    )
    worker = json.loads(worker_json.read_text(encoding="utf-8"))
    panels = worker["ranking"].get("counter_panels")
    if set(panels or {}) != {"1", "2", "3"}:
        raise ValueError("automatic worker did not produce all three panels")
    return worker, worker_json, command


def _automatic_recorded_panel(
    args, source: Path, experiment: int
) -> tuple[dict[str, object], dict[str, object], Path, list[str]]:
    cache = (
        args.results_root
        / "automatic-graphs"
        / args.platform
        / f"stratified-{args.stratification}"
        / args.case
        / f"experiment-{experiment}"
    )
    worker_json = cache / "worker.json"
    command = _worker_command(
        args,
        "--counter-panel",
        "recorded_experiment",
        "--panel-source",
        str(source.resolve()),
        "--automatic-temporal-windows",
        *(str(value) for value in args.temporal_windows),
        "--json",
        str(worker_json.resolve()),
        "--quiet",
    )
    worker_json.parent.mkdir(parents=True, exist_ok=True)
    environment = _worker_environment(args.platform)
    source_root = Path(environment["PYTHONPATH"].split(os.pathsep)[0])
    plugin = source_root / "triton" / "plugins" / "libLAQSTritonAccessManifest.so"
    configured_plugin = environment.get("LAQS_TRITON_PLUGIN_PATH")
    if configured_plugin is None and not plugin.is_file():
        raise FileNotFoundError(
            f"automatic manifest plugin is missing from {plugin}; rerun the "
            f"{args.platform} Triton setup before rescoring"
        )
    subprocess.run(command, check=True, cwd=REPOSITORY, env=environment)
    worker = json.loads(worker_json.read_text(encoding="utf-8"))
    panel = worker["ranking"].get("counter_panel")
    if (
        panel is None
        or panel.get("rescore", {}).get("worker_mode")
        != "recorded_experiment"
    ):
        raise ValueError("automatic worker did not rescore the recorded panel")
    return worker, panel, worker_json, command


def _statistics(report: dict[str, object], platform: str) -> dict[str, object]:
    candidates = [
        candidate for candidate in report["candidates"] if candidate["complete"]
    ]
    if platform == "tuolumne":
        metric = "l1_cache_line_accesses"
        native = "TCP_TOTAL_CACHE_ACCESSES_sum"
        unit = "64-byte vL1D access"
    else:
        metric = "first_level_memory_accesses"
        native = report["native_counter"]
        unit = report["native_unit"]
    x = [float(candidate["quotient_score"]) for candidate in candidates]
    y = [
        float(candidate["counters"]["steady_state"][metric])
        for candidate in candidates
    ]
    return {
        "observation_count": len(x),
        "primary_hardware_metric": metric,
        "primary_hardware_counter": native,
        "native_counter": native,
        "native_unit": unit,
        "tie_aware_spearman": rank_correlation(x, y),
        "free_intercept_linear_fit": (
            linear_fit(x, y) if len(set(x)) > 1 else None
        ),
    }


def _merge_report(
    old: dict[str, object],
    panel: dict[str, object],
    *,
    experiment: int,
    platform: str,
    counter_path: Path,
    worker: dict[str, object],
    worker_json: Path,
    command: list[str],
) -> dict[str, object]:
    def archived_profile_path(value: str) -> str:
        path = Path(value)
        try:
            profile_index = path.parts.index("profiles")
        except ValueError:
            return str(path)
        return str(
            (counter_path.parent / Path(*path.parts[profile_index:])).resolve()
        )

    old_by_mapping = {
        candidate["mapping_id"]: candidate for candidate in old["candidates"]
    }
    new_ids = {candidate["mapping_id"] for candidate in panel["candidates"]}
    if new_ids != set(old_by_mapping):
        missing = sorted(new_ids - set(old_by_mapping))
        extra = sorted(set(old_by_mapping) - new_ids)
        raise ValueError(
            f"saved counter panel differs from regenerated experiment {experiment}: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    candidates = []
    for scored in panel["candidates"]:
        measured = old_by_mapping[scored["mapping_id"]]
        merged = dict(scored)
        merged.update(
            {
                field: measured[field]
                for field in MEASUREMENT_FIELDS
                if field in measured
            }
        )
        if "profile_checkpoints" in merged:
            merged["profile_checkpoints"] = [
                archived_profile_path(path)
                for path in merged["profile_checkpoints"]
            ]
        candidates.append(merged)

    report = dict(old)
    component_model = panel["score_profile"]["component_model"]
    temporal_windows = component_model.get("scope_basis", {}).get(
        "temporal_windows", [4, 16]
    )
    report.update(
        {
            "target_operand": worker["target_operand"],
            "operand_shape": worker["operand_shape"],
            "execution_layout": worker["execution_layout"],
            "panel": panel,
            "candidates": candidates,
            "counter_source": {
                "report": str(counter_path.resolve()),
                "reuse_key": "mapping_id",
                "profilers_relaunched": False,
                "automatic_graph_worker": str(worker_json.resolve()),
                "automatic_graph_command": command,
            },
            "measurement_scope": {
                **old["measurement_scope"],
                "quotient": (
                    "automatic post-coalescing manifest graph; complete "
                    "universal-v1 scope vector for the varied operand, with "
                    f"temporal windows {temporal_windows}"
                ),
                "counter_reuse": (
                    "saved counter aggregates joined to regenerated scores by "
                    "the full-rank mapping_id"
                ),
            },
            "complete": all(candidate["complete"] for candidate in candidates),
            "correct": all(
                candidate.get("structural_validation", {}).get(
                    "all_accepted", False
                )
                for candidate in candidates
                if candidate["complete"]
            ),
        }
    )
    report["configuration"] = {
        **old["configuration"],
        "candidate_panel_schema": 3,
        "graph_construction": component_model["construction"],
        "automatic_temporal_windows": temporal_windows,
        "score_profile": panel["score_profile"]["profile_id"],
        "counter_source": str(counter_path.resolve()),
    }
    report["missing_profiles"] = [
        archived_profile_path(path)
        for path in old.get("missing_profiles", ())
    ]
    report["statistics"] = _statistics(report, platform)
    report["final_experiment"] = experiment
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--experiment", type=int, choices=(1, 2, 3))
    parser.add_argument("--counter-source", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path, default=EXPERIMENT_ROOT / "results"
    )
    parser.add_argument("--plots-root", type=Path, default=EXPERIMENT_ROOT / "plots")
    parser.add_argument("--layouts", type=positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratification", choices=("all", "issue", "temporal"), default="all"
    )
    parser.add_argument("--pool-multiplier", type=positive_integer, default=20)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument(
        "--gemv-k",
        type=positive_integer,
        help="override GEMV's reduction dimension for a sweep rescore",
    )
    parser.add_argument(
        "--preserve-panel",
        action="store_true",
        help="rescore exactly the mappings in the counter-source report",
    )
    parser.add_argument(
        "--temporal-windows",
        type=positive_integer,
        nargs="+",
        default=(4, 16),
        help="automatic temporal-window ladder (sorted powers of two)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    args.temporal_windows = tuple(args.temporal_windows)
    if (
        tuple(sorted(set(args.temporal_windows))) != args.temporal_windows
        or any(value & (value - 1) for value in args.temporal_windows)
    ):
        raise ValueError("--temporal-windows must be sorted unique powers of two")
    if args.preserve_panel and args.experiment is None:
        raise ValueError("--preserve-panel requires --experiment")
    if args.gemv_k is not None:
        if args.case != "gemv":
            raise ValueError("--gemv-k is only valid with --case gemv")
        if args.gemv_k & (args.gemv_k - 1):
            raise ValueError("--gemv-k must be a power of two")
    args.transaction_bytes = 64 if args.platform == "tuolumne" else 32
    if not args.preserve_panel and args.skip_existing and _outputs_current(args):
        print(f"already current: {args.platform}/{args.case}")
        return
    if args.preserve_panel:
        experiments = (args.experiment,)
        shared_worker = None
    else:
        experiments = (1, 2, 3)
        shared_worker = _automatic_panels(args)
    for experiment in experiments:
        source = _counter_report(
            args.counter_source,
            experiment,
            args.platform,
            args.stratification,
            args.case,
        )
        old = json.loads(source.read_text(encoding="utf-8"))
        if args.preserve_panel:
            worker, panel, worker_json, command = _automatic_recorded_panel(
                args, source, experiment
            )
        else:
            assert shared_worker is not None
            worker, worker_json, command = shared_worker
            panel = worker["ranking"]["counter_panels"][str(experiment)]
        report = _merge_report(
            old,
            panel,
            experiment=experiment,
            platform=args.platform,
            counter_path=source,
            worker=worker,
            worker_json=worker_json,
            command=command,
        )
        case_root = (
            args.results_root
            / f"experiment-{experiment}"
            / args.platform
            / f"stratified-{args.stratification}"
            / args.case
        )
        panel_record = {
            "experiment": f"triton_final_experiment_{experiment}_panel_{args.platform}",
            "configuration": report["configuration"],
            "panel": panel,
            "kernel": worker["kernel"],
            "kernel_name": KERNEL_NAMES[args.case],
            "target_operand": worker["target_operand"],
            "operand_shape": worker["operand_shape"],
            "execution_layout": worker["execution_layout"],
            "reference_layout": worker["ranking"]["default"],
            "process": worker["process"],
            "artifacts": {"worker_json": str(worker_json.resolve())},
            "command": command,
            "correct": bool(worker["correct"]),
        }
        write_json(case_root / "profiles" / "panel.json", panel_record)
        report_path = case_root / "report.json"
        write_json(report_path, report)
        _write_counter_csv(
            args.platform, report, case_root / "counter-data.csv"
        )
        plot_path = (
            args.plots_root
            / f"experiment-{experiment}"
            / args.platform
            / f"stratified-{args.stratification}"
            / f"{args.case}.pdf"
        )
        analyze_report(report_path, plot_path)
        print(f"wrote experiment {experiment}: {report_path}")


if __name__ == "__main__":
    main()
