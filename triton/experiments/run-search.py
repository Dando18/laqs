#!/usr/bin/env python3
"""Run one TritonBench configuration for final Experiments 4--6."""

from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (
    str(EXPERIMENT_ROOT),
    str(TRITON_ROOT / "tritonbench"),
    str(TRITON_ROOT),
    str(REPOSITORY),
)

from tritonbench_cases import OPERATORS, selected_cases


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(4, 5, 6), required=True)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument("--operator", choices=OPERATORS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timing-processes", type=positive, default=3)
    parser.add_argument("--timing-warmup", type=positive, default=10)
    parser.add_argument("--timing-samples", type=positive, default=21)
    parser.add_argument("--timing-iterations", type=positive, default=50)
    parser.add_argument("--profile-launches", type=positive, default=3)
    parser.add_argument("--profile-warmup", type=positive, default=5)
    parser.add_argument("--profile-iterations", type=positive, default=20)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--worker", choices=("timing", "profile"))
    parser.add_argument("--layout", choices=("baseline", "selected"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--timing-process-index", type=int, default=0)
    parser.add_argument("--results-root", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--plots-root", type=Path, default=EXPERIMENT_ROOT / "plots")
    parser.add_argument("--tau-profile", type=Path, default=EXPERIMENT_ROOT / "tau-profiles.json")
    parser.add_argument("--rocprof", type=Path, default=Path(shutil.which("rocprof") or "/opt/rocm-7.0.2/bin/rocprof"))
    parser.add_argument("--ncu", type=Path, default=Path(shutil.which("ncu") or "ncu"))
    return parser.parse_args(argv)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _case(args):
    matches = selected_cases(args.operator, args.config)
    if len(matches) != 1:
        raise ValueError(f"expected one case, found {len(matches)}")
    return matches[0]


def _activate_triton_source(platform: str) -> None:
    """Put the platform-specific editable Triton checkout ahead of this repo."""

    configured = os.environ.get("RELAY_TRITON_PYTHON_ROOT")
    if configured:
        source = Path(configured)
    elif platform == "tuolumne":
        source = TRITON_ROOT / "triton-lang" / "python"
    else:
        from urllib.parse import unquote, urlparse

        direct = metadata.distribution("triton").read_text("direct_url.json")
        if direct is None:
            raise RuntimeError("Matrix Triton install has no editable-source metadata")
        checkout = Path(unquote(urlparse(json.loads(direct)["url"]).path))
        source = checkout / "python"
    if not (source / "triton" / "__init__.py").is_file():
        raise RuntimeError(f"invalid platform Triton Python source: {source}")
    sys.path.insert(0, str(source.resolve()))


def _selection(args) -> dict[str, Any]:
    if args.selection is None:
        raise ValueError("internal workers require --selection")
    return json.loads(args.selection.read_text(encoding="utf-8"))


def _runtime_layouts(selection):
    from layout_runtime import RuntimeLayout

    return tuple(RuntimeLayout.from_dict(value) for value in selection["runtime_layouts"])


def _output_arguments(selection) -> tuple[int, ...]:
    return tuple(map(int, selection["output_arguments"]))


def _prepared_launches(args, selection):
    from layout_runtime import freeze_launch, fresh_outputs, replace_inputs

    spec = _case(args).factory()
    frozen = freeze_launch(spec, selection["selected_config"])
    outputs = _output_arguments(selection)
    baseline = fresh_outputs(frozen, outputs)
    selected = fresh_outputs(replace_inputs(frozen, _runtime_layouts(selection)), outputs)
    return baseline, selected


def _run(launch, layout: str, layouts) -> None:
    from layout_runtime import rewrite_layouts

    if layout == "selected":
        with rewrite_layouts(layouts):
            launch.run()
    else:
        launch.run()


def _validate_outputs(baseline, selected, output_arguments) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    records = []
    for argument in output_arguments:
        expected = baseline.values[argument]
        observed = selected.values[argument]
        close = torch.allclose(observed, expected, rtol=1e-2, atol=5e-2, equal_nan=True)
        error = float((observed.float() - expected.float()).abs().max().item())
        records.append({"argument": argument, "allclose": bool(close), "max_abs_error": error})
    if not records or not all(record["allclose"] for record in records):
        raise RuntimeError(f"transformed layout failed numerical validation: {records}")
    return {"correct": True, "outputs": records}


def timing_worker(args) -> None:
    import torch

    selection = _selection(args)
    layouts = _runtime_layouts(selection)
    outputs = _output_arguments(selection)
    baseline, selected = _prepared_launches(args, selection)
    _run(baseline, "baseline", layouts)
    _run(selected, "selected", layouts)
    validation = _validate_outputs(baseline, selected, outputs)

    for _ in range(args.timing_warmup):
        _run(baseline, "baseline", layouts)
        _run(selected, "selected", layouts)
    torch.cuda.synchronize()

    samples = {"baseline": [], "selected": []}
    launches = {"baseline": baseline, "selected": selected}
    labels = ("baseline", "selected")
    for sample in range(args.timing_samples):
        order = labels if (sample + args.timing_process_index) % 2 == 0 else tuple(reversed(labels))
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if label == "selected":
                from layout_runtime import rewrite_layouts
                with rewrite_layouts(layouts):
                    for _ in range(args.timing_iterations):
                        launches[label].run()
            else:
                for _ in range(args.timing_iterations):
                    launches[label].run()
            end.record()
            end.synchronize()
            samples[label].append(float(start.elapsed_time(end)) / args.timing_iterations)
    result = {
        "schema": "relay.tritonbench.timing.v1",
        "process_index": args.timing_process_index,
        "configuration": {
            "warmup": args.timing_warmup,
            "samples": args.timing_samples,
            "iterations": args.timing_iterations,
        },
        "validation": validation,
        "timings": {
            label: {
                "median_ms": statistics.median(values),
                "mean_ms": statistics.fmean(values),
                "min_ms": min(values),
                "samples_ms": values,
            }
            for label, values in samples.items()
        },
    }
    if args.worker_output is None:
        raise ValueError("timing worker requires --worker-output")
    write_json(args.worker_output, result)


def profile_worker(args) -> None:
    import torch

    if args.layout is None:
        raise ValueError("profile worker requires --layout")
    selection = _selection(args)
    layouts = _runtime_layouts(selection)
    baseline, selected = _prepared_launches(args, selection)
    launch = baseline if args.layout == "baseline" else selected
    for _ in range(args.profile_warmup):
        _run(launch, args.layout, layouts)
    torch.cuda.synchronize()
    if args.layout == "selected":
        from layout_runtime import rewrite_layouts
        with rewrite_layouts(layouts):
            for _ in range(args.profile_iterations):
                launch.run()
    else:
        for _ in range(args.profile_iterations):
            launch.run()
    torch.cuda.synchronize()
    if args.worker_output is not None:
        write_json(args.worker_output, {"correct": True, "layout": args.layout})


def _worker_command(args, selection: Path, *, worker: str, layout: str | None = None,
                    output: Path | None = None, process_index: int = 0) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--experiment", str(args.experiment), "--platform", args.platform,
        "--operator", args.operator, "--config", args.config,
        "--worker", worker, "--selection", str(selection.resolve()),
        "--timing-warmup", str(args.timing_warmup),
        "--timing-samples", str(args.timing_samples),
        "--timing-iterations", str(args.timing_iterations),
        "--profile-warmup", str(args.profile_warmup),
        "--profile-iterations", str(args.profile_iterations),
        "--timing-process-index", str(process_index),
    ]
    if layout is not None:
        command.extend(("--layout", layout))
    if output is not None:
        command.extend(("--worker-output", str(output.resolve())))
    return command


def _run_timing_processes(args, case_root: Path, selection_path: Path):
    records = []
    timing_dir = case_root / "timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.timing_processes):
        output = timing_dir / f"process-{index}.json"
        if args.rerun or not output.exists():
            subprocess.run(
                _worker_command(args, selection_path, worker="timing", output=output, process_index=index),
                check=True, cwd=REPOSITORY,
            )
        records.append(json.loads(output.read_text(encoding="utf-8")))
    return records


def _write_amd_counter_config(path: Path, kernel_name: str):
    from stage1_counter_sweep import COUNTER_PASSES

    path.write_text("\n".join(("# RELAY TritonBench evaluation counters", *COUNTER_PASSES,
                                f"kernel: {kernel_name}", "")), encoding="utf-8")


def _profile_once(args, case_root: Path, selection_path: Path, kernel_name: str,
                  layout: str, launch: int):
    directory = case_root / "profiles" / layout / f"launch-{launch}"
    checkpoint = directory / "profile.json"
    if checkpoint.exists() and not args.rerun:
        return json.loads(checkpoint.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    worker_output = directory / "worker.json"
    raw = directory / "counters.csv"
    worker = _worker_command(args, selection_path, worker="profile", layout=layout,
                             output=worker_output)
    if args.platform == "tuolumne":
        from stage1_counter_analysis import parse_counter_csv
        from stage1_counter_sweep import _profiler_environment

        config = directory / "counters.txt"
        _write_amd_counter_config(config, kernel_name)
        command = [str(args.rocprof), "-i", str(config), "-o", str(raw),
                   "--timestamp", "on", *worker]
        subprocess.run(
            command,
            check=True,
            cwd=REPOSITORY,
            env=_profiler_environment(args.rocprof.resolve()),
        )
        counters = parse_counter_csv(raw, kernel_name=kernel_name,
                                     profile_iterations=args.profile_iterations)
    else:
        from stage1_nvidia_counter_analysis import COUNTER_METRICS, parse_counter_csv

        command = [
            str(args.ncu), "--csv", "--page", "raw", "--print-units", "base",
            "--metrics", ",".join(COUNTER_METRICS), "--kernel-name-base", "function",
            "--kernel-name", f"regex:{kernel_name}", "--cache-control", "none",
            "--target-processes", "application-only", "--log-file", str(raw), *worker,
        ]
        subprocess.run(command, check=True, cwd=REPOSITORY)
        counters = parse_counter_csv(raw, kernel_name=kernel_name,
                                     profile_iterations=args.profile_iterations)
    record = {
        "schema": "relay.tritonbench.profile.v1", "layout": layout,
        "launch": launch, "kernel_name": kernel_name, "counters": counters,
        "command": command, "artifacts": {"raw_csv": str(raw), "worker": str(worker_output)},
    }
    write_json(checkpoint, record)
    return record


def _run_profiles(args, case_root: Path, selection_path: Path, kernel_name: str):
    profiles = {"baseline": [], "selected": []}
    for launch in range(1, args.profile_launches + 1):
        order = ("baseline", "selected") if launch % 2 else ("selected", "baseline")
        for layout in order:
            profiles[layout].append(
                _profile_once(args, case_root, selection_path, kernel_name, layout, launch)
            )
    if args.platform == "tuolumne":
        from stage1_counter_analysis import aggregate_profiles
    else:
        from stage1_nvidia_counter_analysis import aggregate_profiles
    return {
        layout: aggregate_profiles([record["counters"] for record in records])
        for layout, records in profiles.items()
    }


def _timing_summary(records):
    result = {}
    for layout in ("baseline", "selected"):
        values = [
            value
            for record in records
            for value in record["timings"][layout]["samples_ms"]
        ]
        result[layout] = {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
            "samples_ms": values,
        }
    result["speedup"] = result["baseline"]["median_ms"] / result["selected"]["median_ms"]
    result["processes"] = records
    return result


def _counter_reductions(counters):
    if counters is None:
        return {}
    baseline = counters["baseline"]["steady_state"]
    selected = counters["selected"]["steady_state"]
    reductions = {}
    for name, value in baseline.items():
        if name.endswith("_by_launch") or name in {"native_counters", "native_counter", "native_unit", "profile_launch_count", "dispatches_per_launch"}:
            continue
        if isinstance(value, (int, float)) and value and isinstance(selected.get(name), (int, float)):
            reductions[name] = 100.0 * (float(value) - float(selected[name])) / float(value)
    return reductions


def _write_raw_csv(path: Path, report):
    row = {
        "experiment": report["experiment"], "platform": report["platform"],
        "operator": report["operator"], "config": report["config"],
        "status": report["status"],
    }
    if report["status"] == "complete":
        row.update({
            "baseline_median_ms": report["timing"]["baseline"]["median_ms"],
            "selected_median_ms": report["timing"]["selected"]["median_ms"],
            "speedup": report["timing"]["speedup"],
            "search_seconds": report["search"].get("elapsed_seconds"),
            "optimized_array_count": report["search"]["optimized_array_count"],
            "transformed_array_count": report["search"]["transformed_array_count"],
            "selected_j_area": report["search"]["score"]["hardware_area"],
        })
        row.update({
            f"selected_Q:{component['name']}": component["raw_region_count"]
            for component in report["search"]["score"]["components"]
        })
        if report["counters"] is not None:
            for layout in ("baseline", "selected"):
                summary = report["counters"][layout]["steady_state"]
                row.update({
                    f"{layout}_counter:{name}": value
                    for name, value in summary.items()
                    if isinstance(value, (int, float))
                })
        row.update({f"reduction_percent:{key}": value for key, value in report["counter_reductions_percent"].items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _plot(path: Path, report):
    import matplotlib.pyplot as plt

    reductions = report["counter_reductions_percent"]
    columns = 2 if reductions else 1
    figure, axes = plt.subplots(1, columns, figsize=(6.7 * columns, 5.0), squeeze=False)
    timing = report["timing"]
    axis = axes[0][0]
    axis.bar(("Ordinary Triton", "Selected layout"),
             (timing["baseline"]["median_ms"], timing["selected"]["median_ms"]),
             color=("#0072B2", "#E69F00"), edgecolor="black", hatch=("", "//"))
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Median kernel runtime (ms)")
    axis.set_title(f"Runtime (speedup {timing['speedup']:.3f}×)")
    if reductions:
        preferred = [name for name in (
            "first_level_memory_accesses", "l1_miss_demand_to_l2",
            "l1_to_l2_read_traffic", "l2_read_misses", "hbm_read_bytes",
        ) if name in reductions]
        values = [reductions[name] for name in preferred]
        axis = axes[0][1]
        axis.barh(range(len(preferred)), values, color="#009E73", edgecolor="black", hatch="..")
        axis.set_yticks(range(len(preferred)), [name.replace("_", " ") for name in preferred])
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Reduction from ordinary Triton (%)")
        axis.set_title("Memory-counter change")
    figure.suptitle(f"Experiment {report['experiment']}: {report['operator']} / {report['config']}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def orchestrate(args) -> None:
    import torch
    from relay import AnalysisOptions, EvaluationLimits, analyze_launch
    from layout_runtime import freeze_launch, fresh_outputs, replace_inputs, unwrap_jit
    from search_algorithms import load_tau_profile, select_layouts

    case = _case(args)
    case_root = (args.results_root / f"experiment-{args.experiment}" / args.platform / case.case_id).resolve()
    report_path = case_root / "report.json"
    selection_path = case_root / "selection.json"
    plot_path = (args.plots_root / f"experiment-{args.experiment}" / args.platform / f"{case.case_id}.pdf").resolve()
    if report_path.exists() and not args.rerun:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"complete", "excluded"}:
            print(f"Reusing {existing['status']} result: {report_path}")
            return

    profile = load_tau_profile(args.platform, args.tau_profile)
    spec = case.factory()
    analysis_start = perf_counter()
    analysis = analyze_launch(
        spec.kernel, spec.grid, *spec.args,
        _laqs_options=AnalysisOptions(
            hardware_profile=profile,
            limits=EvaluationLimits(
                max_trace_contexts=1 << 16,
                max_dynamic_events=1 << 20,
            ),
        ),
        **spec.kwargs,
    )
    analysis_seconds = perf_counter() - analysis_start
    if not analysis.supported:
        report = {
            "schema": "relay.tritonbench.search.v1", "experiment": args.experiment,
            "platform": args.platform, "operator": case.operator, "config": case.config,
            "description": case.description, "status": "excluded",
            "exclusion": {"category": analysis.unsupported.category,
                          "message": analysis.unsupported.message,
                          "site": analysis.unsupported.site},
            "analysis_seconds": analysis_seconds,
        }
        write_json(report_path, report)
        _write_raw_csv(case_root / "raw-data.csv", report)
        print(f"Excluded {case.case_id}: {analysis.unsupported.category}: {analysis.unsupported.message}")
        return

    try:
        search_start = perf_counter()
        runtime_layouts, search = select_layouts(analysis, args.experiment, profile)
        search.setdefault("elapsed_seconds", perf_counter() - search_start)
        argument_names = {
            str(name): int(index)
            for index, name in analysis.bound_arguments.get("__names__", {}).items()
        }
        outputs = sorted({
            (
                int(allocation.argument)
                if isinstance(allocation.argument, int)
                else argument_names[str(allocation.argument)]
            )
            for allocation in analysis.allocations
            if allocation.role != "read"
            and not allocation.path
            and (
                isinstance(allocation.argument, int)
                or str(allocation.argument) in argument_names
            )
        })
        frozen = freeze_launch(spec, analysis.selected_config)
        baseline = fresh_outputs(frozen, outputs)
        selected = fresh_outputs(replace_inputs(frozen, runtime_layouts), outputs)
        _run(baseline, "baseline", runtime_layouts)
        _run(selected, "selected", runtime_layouts)
        validation = _validate_outputs(baseline, selected, outputs)
    except Exception as error:
        report = {
            "schema": "relay.tritonbench.search.v1",
            "experiment": args.experiment,
            "platform": args.platform,
            "operator": case.operator,
            "config": case.config,
            "description": case.description,
            "status": "excluded",
            "exclusion": {
                "category": "search_or_realization",
                "message": f"{type(error).__name__}: {error}",
                "site": None,
            },
            "analysis_seconds": analysis_seconds,
        }
        write_json(report_path, report)
        _write_raw_csv(case_root / "raw-data.csv", report)
        print(f"Excluded {case.case_id}: {type(error).__name__}: {error}")
        return
    selection = {
        "schema": "relay.tritonbench.selection.v1", "experiment": args.experiment,
        "platform": args.platform, "operator": case.operator, "config": case.config,
        "selected_config": dict(analysis.selected_config),
        "runtime_layouts": [layout.to_dict() for layout in runtime_layouts],
        "output_arguments": outputs, "kernel_name": unwrap_jit(spec.kernel).fn.__name__,
        "search": search, "validation": validation,
    }
    write_json(selection_path, selection)

    del selected, baseline, frozen, analysis, spec
    torch.cuda.empty_cache()
    timing_records = _run_timing_processes(args, case_root, selection_path)
    counters = None if args.no_profile else _run_profiles(
        args, case_root, selection_path, selection["kernel_name"]
    )
    report = {
        "schema": "relay.tritonbench.search.v1", "experiment": args.experiment,
        "platform": args.platform, "operator": case.operator, "config": case.config,
        "description": case.description, "status": "complete",
        "analysis_seconds": analysis_seconds, "hardware_profile": profile.to_dict(),
        "frozen_triton_config": selection["selected_config"], "search": search,
        "validation": validation, "timing": _timing_summary(timing_records),
        "counters": counters, "counter_reductions_percent": _counter_reductions(counters),
        "artifacts": {"selection": str(selection_path), "raw_data": str(case_root / "raw-data.csv"),
                      "plot": str(plot_path)},
    }
    write_json(report_path, report)
    _write_raw_csv(case_root / "raw-data.csv", report)
    _plot(plot_path, report)
    print(f"Completed Experiment {args.experiment}, {case.case_id}, {args.platform}")
    print(f"Speedup: {report['timing']['speedup']:.3f}x")
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")


def main() -> None:
    args = parse_arguments()
    _activate_triton_source(args.platform)
    if args.worker == "timing":
        timing_worker(args)
    elif args.worker == "profile":
        profile_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
