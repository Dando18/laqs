#!/usr/bin/env python3
"""Run one TritonBench configuration for final Experiments 4--6."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from tau_profiles import TAU_NAMES


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
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--worker", choices=("timing", "profile"))
    parser.add_argument("--layout", choices=("baseline", "selected"))
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--timing-process-index", type=int, default=0)
    parser.add_argument("--results-root", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--plots-root", type=Path, default=EXPERIMENT_ROOT / "plots")
    parser.add_argument("--tau-profile", type=Path, default=EXPERIMENT_ROOT / "tau-profiles.json")
    parser.add_argument(
        "--tau-name",
        choices=TAU_NAMES,
        default="expert",
        help="named device tau profile used by J_area layout selection",
    )
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
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    _verify_runtime(selection)
    return selection


def _verify_runtime(selection):
    import hashlib
    import torch
    import triton
    from relay.triton_frontend import _default_plugin_path
    from layout_runtime import _plugin_path
    from experiment_support import compiler_source_identity
    expected = selection["capture_identity"]
    if compiler_source_identity(triton.__file__) != expected["actual_triton_source"]:
        raise ValueError("active Triton checkout changed after capture")
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (_default_plugin_path(), _plugin_path())}
    if actual != expected["plugins"]:
        raise ValueError("plugin binaries changed after capture")
    if (torch.__version__ != expected["torch"] or triton.__version__ != expected["triton"]
            or torch.cuda.get_device_name() != expected["gpu"]):
        raise ValueError("GPU or compiler runtime changed after capture")


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


def _run(launch, layout: str, layouts):
    from layout_runtime import rewrite_layouts

    if layout == "selected":
        with rewrite_layouts(layouts):
            return launch.run()
    else:
        return launch.run()


def _validate_outputs(baseline, selected, output_arguments) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    records = []
    for argument in output_arguments:
        expected = baseline.values[argument]
        observed = selected.values[argument]
        from reference_validation import comparison
        records.append({"argument": argument, **comparison(observed, expected)})
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

    from layout_runtime import rewrite_layouts
    from layout_runtime import fresh_outputs
    identity = fresh_outputs(baseline, outputs)
    launches = {"baseline": baseline, "selected": selected, "identity": identity}
    graphs = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for label, launch in launches.items():
            context = rewrite_layouts(layouts) if label == "selected" else nullcontext()
            with context:
                launch.run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.timing_iterations):
                        launch.run()
                graphs[label] = graph
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for _ in range(args.timing_warmup):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {label: [] for label in graphs}
    labels = tuple(graphs)
    for sample in range(args.timing_samples):
        rotation = (sample + args.timing_process_index) % len(labels)
        order = labels[rotation:] + labels[:rotation]
        if (sample + args.timing_process_index) % 2:
            order = tuple(reversed(order))
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[label].replay()
            end.record()
            end.synchronize()
            samples[label].append(float(start.elapsed_time(end)) / args.timing_iterations)
    result = {
        "schema": "relay.tritonbench.timing.v2",
        "selection_hash": selection["selection_hash"],
        "method": "CUDA/HIP graph replay; warm cache; identity control",
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
        selection_hash = json.loads(selection_path.read_text())["selection_hash"]
        reusable = (output.exists() and json.loads(output.read_text()).get("selection_hash") == selection_hash)
        if args.rerun or not reusable:
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
    selection_hash = json.loads(selection_path.read_text())["selection_hash"]
    if checkpoint.exists() and not args.rerun:
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous.get("selection_hash") == selection_hash:
            return previous
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
        "schema": "relay.tritonbench.profile.v2", "layout": layout,
        "selection_hash": selection_hash,
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
    from experiment_support import process_summary
    return process_summary(records)


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
        "tau_name": report.get("tau_name"),
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
    """Measure a selection prepared by the shared GPU-capture/CPU-search stages."""
    import torch
    from experiment_support import digest, source_identity, proposal_key
    from layout_contract import compiler_statistics, realization_rejections
    from reference_validation import operator_reference

    if args.selection is None:
        raise ValueError("prepare shared captures first with submit-experiments-4-6-<platform>.bash")
    case = _case(args)
    case_root = (args.results_root / f"experiment-{args.experiment}" / args.platform / case.case_id).resolve()
    case_root.mkdir(parents=True, exist_ok=True)
    report_path = case_root / "report.json"
    selection_path = case_root / "selection.json"
    proposed = json.loads(args.selection.read_text())
    _verify_runtime(proposed)
    if proposed["source_identity"]["source_hash"] != source_identity()["source_hash"]:
        raise ValueError("sources changed after capture; submit a new suite")
    for key, expected in (("experiment", args.experiment), ("tau_name", args.tau_name),
                          ("platform", args.platform), ("operator", case.operator), ("config", case.config)):
        if proposed[key] != expected:
            raise ValueError(f"prepared selection mismatch: {key}")
    fingerprint = digest({"proposal_hash": proposal_key(proposed),
        "timing": [args.timing_processes, args.timing_warmup, args.timing_samples, args.timing_iterations],
        "profiling": [args.no_profile, args.profile_launches, args.profile_warmup, args.profile_iterations]})
    if report_path.exists() and not args.rerun:
        old = json.loads(report_path.read_text())
        if old.get("run_hash") == fingerprint and (old.get("status") == "complete" or (args.validate_only and old.get("status") == "validated")):
            print(f"Reusing {report_path}")
            return
    report = {"schema": "relay.tritonbench.search.v2", "experiment": args.experiment,
        "platform": args.platform, "operator": case.operator, "config": case.config,
        "description": case.description, "tau_name": args.tau_name, "status": "running",
        "run_hash": fingerprint, "source_identity": proposed["source_identity"],
        "capture_identity": proposed["capture_identity"], "hardware_profile": proposed["hardware_profile"],
        "capture_seconds": proposed["capture_seconds"],
        "analysis_seconds": proposed["analysis_seconds"], "stage": "realization"}
    write_json(report_path, report)
    setup = {}
    selection = dict(proposed)
    search = selection["search"]
    try:
        start = perf_counter()
        baseline, selected = _prepared_launches(args, selection)
        torch.cuda.synchronize()
        setup["packing_and_allocation_seconds"] = perf_counter() - start
        layouts = _runtime_layouts(selection)
        start = perf_counter()
        baseline_kernel = _run(baseline, "baseline", layouts)
        compile_error = None
        try:
            selected_kernel = _run(selected, "selected", layouts)
        except Exception as error:
            if not layouts:
                raise
            compile_error = f"candidate compilation failed: {type(error).__name__}: {error}"
            selected = baseline.clone()
            selected_kernel = baseline_kernel
        torch.cuda.synchronize()
        setup["compilation_and_first_launch_seconds"] = perf_counter() - start
        codegen = {"baseline": compiler_statistics(baseline_kernel, case_root / "codegen", "baseline"),
                   "proposed": compiler_statistics(selected_kernel, case_root / "codegen", "proposed")}
        reasons = ([compile_error] if compile_error else realization_rejections(codegen["baseline"], codegen["proposed"])) if layouts else []
        start = perf_counter()
        reference = operator_reference(case.operator, baseline)
        try:
            validation = _validate_outputs(baseline, selected, _output_arguments(selection))
        except RuntimeError as error:
            if not layouts:
                raise
            reasons.append(str(error))
        if reasons:
            selection["proposed_runtime_layouts"] = selection["runtime_layouts"]
            selection["runtime_layouts"] = []
            search["proposed_score"] = search["score"]
            search["score"] = search["baseline_score"]
            search["proposed_layouts"] = search["layouts"]
            search["layouts"] = selection["baseline_layouts"]
            search["proposed_transformed_array_count"] = search["transformed_array_count"]
            search["transformed_array_count"] = 0
            selected = baseline.clone()
            _run(selected, "baseline", ())
            validation = _validate_outputs(baseline, selected, _output_arguments(selection))
        validation["operator_reference"] = reference
        setup["validation_seconds"] = perf_counter() - start
        selection["realization"] = {"accepted": not reasons, "rejections": reasons, "codegen": codegen}
        selection["validation"] = validation
        selection["selection_hash"] = digest({"run_hash": fingerprint, "layouts": selection["runtime_layouts"],
            "codegen": {label: {"statistics": {k: v for k, v in c.items() if k != "artifacts"},
                                  "artifact_hashes": {kind: a["sha256"] for kind, a in c["artifacts"].items()}}
                        for label, c in codegen.items()}})
        write_json(selection_path, selection)
        if args.validate_only:
            report.update(status="validated", stage="validated", search=search,
                validation=validation, realization=selection["realization"], setup=setup,
                frozen_triton_config=selection["selected_config"])
            write_json(report_path, report)
            print(f"Preflight passed E{args.experiment} {case.case_id} {args.tau_name}")
            return
        del selected, baseline, baseline_kernel, selected_kernel
        torch.cuda.empty_cache()
        report["stage"] = "timing"
        write_json(report_path, report)
        timing = _timing_summary(_run_timing_processes(args, case_root, selection_path))
        report["stage"] = "profiling"
        write_json(report_path, report)
        counters = None if args.no_profile else _run_profiles(args, case_root, selection_path, selection["kernel_name"])
        plot_path = (args.plots_root / f"experiment-{args.experiment}" / args.platform / f"{case.case_id}.pdf").resolve()
        saved_ms = timing["baseline"]["median_ms"] - timing["selected"]["median_ms"]
        report.update({"status": "complete", "stage": "complete", "search": search,
            "frozen_triton_config": selection["selected_config"], "validation": validation,
            "realization": selection["realization"], "setup": setup,
            "packing_break_even_reuses_upper_bound": (
                int(setup["packing_and_allocation_seconds"] * 1000 / saved_ms) + 1
                if saved_ms > 0 and selection["runtime_layouts"] else None),
            "timing": timing, "counters": counters,
            "credible_speedup": bool(selection["runtime_layouts"] and timing["speedup_ci95"]
                and timing["speedup_ci95"][0] > 1.01 and timing["identity_max_deviation"] <= .01),
            "counter_reductions_percent": _counter_reductions(counters),
            "artifacts": {"selection": str(selection_path), "plot": str(plot_path)}})
        write_json(report_path, report)
        _write_raw_csv(case_root / "raw-data.csv", report)
        _plot(plot_path, report)
        print(f"Completed E{args.experiment} {case.case_id} {args.tau_name}: {timing['speedup']:.3f}x")
    except Exception as error:
        report.update({"status": "failed", "exclusion": {"category": report["stage"],
                       "message": f"{type(error).__name__}: {error}"}})
        write_json(report_path, report)
        _write_raw_csv(case_root / "raw-data.csv", report)
        raise


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
