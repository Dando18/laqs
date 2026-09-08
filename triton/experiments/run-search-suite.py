#!/usr/bin/env python3
"""Shared capture, CPU graph/search, and GPU measurement for Experiments 4--6."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
from dataclasses import replace
import hashlib
import importlib.util
import json
from math import prod
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(HERE.parent), str(HERE.parent / "tritonbench")]
from experiment_support import (compiler_source_identity, digest, host_arguments, load_graph,
                                save_graph, source_identity)
from tritonbench_cases import CASE_BY_ID
from tau_profiles import TAU_NAMES


def runner_module():
    spec = importlib.util.spec_from_file_location("search_runner", HERE / "run-search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = runner_module()
write_json = RUNNER.write_json


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("capture", "search", "validate", "measure"), required=True)
    p.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    p.add_argument("--case", choices=tuple(CASE_BY_ID), required=True)
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--plots-root", type=Path)
    p.add_argument("--tau-profile", type=Path, default=HERE / "tau-profiles.json")
    p.add_argument("--tau-names", nargs="+", choices=TAU_NAMES, default=list(TAU_NAMES))
    p.add_argument("--experiments", nargs="+", type=int, choices=(4, 5, 6), default=[4, 5, 6])
    p.add_argument("--minimum-score-gain", type=float, default=.01)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--max-retained-events", type=int, default=1 << 20)
    p.add_argument("--max-trace-contexts", type=int, default=1 << 24)
    args, remaining = p.parse_known_args(argv)
    args.measure_arguments = remaining
    args.suite_root = args.suite_root.resolve()
    return args


def shared_root(args):
    return args.suite_root / "shared" / args.platform / args.case


def cell_root(args, experiment, tau):
    return args.suite_root / "tau-profiles" / tau / f"experiment-{experiment}" / args.platform / args.case


def base_report(args, experiment, tau):
    case = CASE_BY_ID[args.case]
    return {"schema": "relay.tritonbench.search.v2", "experiment": experiment,
            "platform": args.platform, "operator": case.operator, "config": case.config,
            "description": case.description, "tau_name": tau}


def stage_status(args, status, stage, error=None):
    for experiment in args.experiments:
        for tau in args.tau_names:
            path = cell_root(args, experiment, tau) / "report.json"
            if status == "running" and path.exists() and not args.rerun:
                if json.loads(path.read_text()).get("status") == "complete":
                    continue
            value = {**base_report(args, experiment, tau), "status": status, "stage": stage}
            if error:
                value["exclusion"] = error
            write_json(cell_root(args, experiment, tau) / "report.json", value)


def capture(args):
    RUNNER._activate_triton_source(args.platform)
    import torch
    import triton
    from relay import AnalysisOptions, analyze_launch
    from relay.triton_frontend import MANIFEST_METADATA_KEY, _default_plugin_path, infer_allocations
    from layout_runtime import _plugin_path, freeze_launch, unwrap_jit
    from reference_validation import operator_reference

    root = shared_root(args)
    identity = source_identity()
    stamp_path = root / "capture.json"
    if stamp_path.exists() and not args.rerun:
        stamp = json.loads(stamp_path.read_text())
        if stamp.get("source_identity") == identity and stamp.get("status") == "complete":
            try:
                RUNNER._verify_runtime(stamp)
            except ValueError:
                pass
            else:
                print(f"Reusing shared capture {root}")
                return
    stage_status(args, "running", "capture")
    spec = CASE_BY_ID[args.case].factory()
    started = perf_counter()
    analysis = analyze_launch(spec.kernel, spec.grid, *spec.args,
        _laqs_options=AnalysisOptions(evaluate=False), **spec.kwargs)
    capture_seconds = perf_counter() - started
    if not analysis.supported:
        raise ValueError(f"{analysis.unsupported.category}: {analysis.unsupported.message}")
    metadata = analysis.compiled_kernel.metadata
    selected = dict(analysis.selected_config)
    for key in ("num_warps", "num_stages", "num_ctas", "maxnreg", "waves_per_eu", "matrix_instr_nonkdim", "kpack"):
        value = getattr(metadata, key, None)
        if value is not None:
            selected[key] = value
    frozen = freeze_launch(spec, selected)
    frozen.run()
    reference_start = perf_counter()
    reference = operator_reference(spec.operator, frozen)
    reference_seconds = perf_counter() - reference_start
    allocations = infer_allocations(analysis.manifest, analysis.bound_arguments)
    index_names = analysis.bound_arguments["__names__"]
    indices = {name: index for index, name in index_names.items()}
    outputs = sorted({a.argument if isinstance(a.argument, int) else indices[a.argument]
                      for a in allocations if a.role != "read" and not a.path})
    payload = getattr(metadata, MANIFEST_METADATA_KEY)
    bound = host_arguments(analysis.bound_arguments)
    data_hash = save_graph(root / "capture.pkl.gz", {"payload": payload, "bound": bound,
                           "grid": analysis.grid, "selected_config": selected})
    inputs = []
    for a in allocations:
        if a.role != "read":
            continue
        t = analysis.bound_arguments[a.argument]
        if not isinstance(t, torch.Tensor):
            continue
        # A bounded content probe complements seed, shape and runtime identity.
        flat = t.contiguous().view(torch.uint8).flatten()
        count = min(1024, flat.numel())
        probes = torch.arange(count, device=flat.device, dtype=torch.int64)
        probes = probes * (flat.numel() - 1) // max(1, count - 1)
        probe_bytes = flat[probes].cpu().numpy().tobytes()
        inputs.append({"name": a.name, "shape": list(a.true_shape), "envelope_shape": list(a.envelope_shape),
            "strides": list(a.strides), "dtype": a.dtype, "element_bytes": a.element_bytes,
            "allocation_expansion": prod(a.envelope_shape) / prod(a.true_shape),
            "sampled_values_sha256": hashlib.sha256(probe_bytes).hexdigest()})
    plugins = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (_default_plugin_path(), _plugin_path())}
    capture_identity = {"capture_hash": data_hash, "source_hash": identity["source_hash"],
        "plugins": plugins, "torch": torch.__version__, "triton": triton.__version__,
        "actual_triton_source": compiler_source_identity(triton.__file__),
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "runtime": torch.version.hip or torch.version.cuda, "seed": 0,
        "selected_config": selected, "grid": list(analysis.grid), "inputs": inputs}
    stamp = {"status": "complete", "source_identity": identity, "capture_identity": capture_identity,
        "capture_seconds": capture_seconds, "reference_seconds": reference_seconds,
        "reference": reference, "selected_config": selected, "output_arguments": outputs,
        "kernel_name": unwrap_jit(spec.kernel).fn.__name__}
    write_json(stamp_path, stamp)
    print(f"Captured and reference-validated {args.case}: {capture_seconds:.2f}s")


def search(args):
    from relay import AnalysisOptions, EvaluationLimits, layout_matrix_rows, row_major_layout
    from relay.triton_frontend import analyze_compiled_manifest
    from search_algorithms import load_tau_profile, select_layouts

    root = shared_root(args)
    stamp = json.loads((root / "capture.json").read_text())
    if stamp["source_identity"] != source_identity():
        raise ValueError("sources changed after shared capture; create a new suite")
    stage_status(args, "running", "graph")
    captured = load_graph(root / "capture.pkl.gz", stamp["capture_identity"]["capture_hash"])
    profile = load_tau_profile(args.platform, args.tau_profile, "expert")
    graph_key = digest({"capture": stamp["capture_identity"], "byte_scales": profile.byte_scales,
                       "limits": [args.max_trace_contexts, args.max_retained_events]})
    graph_stamp_path = root / "graph.json"
    if graph_stamp_path.exists() and not args.rerun:
        graph_stamp = json.loads(graph_stamp_path.read_text())
    else:
        graph_stamp = {}
    if graph_stamp.get("graph_key") == graph_key:
        analysis = load_graph(root / "graph.pkl.gz", graph_stamp["graph_hash"])
        analysis_seconds = graph_stamp["analysis_seconds"]
    else:
        started = perf_counter()
        analysis = analyze_compiled_manifest(None, captured["payload"], captured["grid"], captured["bound"],
            selected_config=captured["selected_config"], options=AnalysisOptions(hardware_profile=profile, require_native_baseline=True,
            limits=EvaluationLimits(max_trace_contexts=args.max_trace_contexts,
                                    max_dynamic_events=args.max_retained_events)))
        analysis_seconds = perf_counter() - started
        if not analysis.supported:
            stage_status(args, "excluded", "graph", {"category": analysis.unsupported.category,
                "message": analysis.unsupported.message, "site": analysis.unsupported.site})
            write_json(root / "graph.json", {"status": "excluded", "graph_key": graph_key})
            return
        analysis = replace(analysis, compiled_kernel=None, manifest=None,
                           bound_arguments={"__names__": analysis.bound_arguments["__names__"]})
        graph_hash = save_graph(root / "graph.pkl.gz", analysis)
        graph_stamp = {"status": "complete", "graph_key": graph_key, "graph_hash": graph_hash,
            "analysis_seconds": analysis_seconds, "events": len(analysis.events),
            "trace_classes": len(analysis.sequences)}
        write_json(graph_stamp_path, graph_stamp)
    baseline_layouts = {m.name: {"target": m.target, "grammar": "canonical", "descriptor": "ordinary row-major",
        "tile_exponents": list(m.mode_bits), "rows": list(layout_matrix_rows(m, row_major_layout(m))),
        "is_baseline": True} for m in analysis.matrices}
    for experiment in args.experiments:
        for tau in args.tau_names:
            report = {**base_report(args, experiment, tau), "status": "running", "stage": "search"}
            directory = cell_root(args, experiment, tau)
            previous_path = directory / "report.json"
            preserve_complete = (previous_path.exists() and not args.rerun
                                 and json.loads(previous_path.read_text()).get("status") == "complete")
            if not preserve_complete:
                write_json(previous_path, report)
            try:
                profile = load_tau_profile(args.platform, args.tau_profile, tau)
                started = perf_counter()
                runtime, record = select_layouts(analysis, experiment, profile,
                                                 minimum_gain=args.minimum_score_gain)
                record["elapsed_seconds"] = perf_counter() - started
                selection = {**base_report(args, experiment, tau), "schema": "relay.tritonbench.selection.v2",
                    "source_identity": stamp["source_identity"], "capture_identity": {**stamp["capture_identity"],
                        "graph_hash": graph_stamp["graph_hash"]}, "hardware_profile": profile.to_dict(),
                    "tau_profile_hash": hashlib.sha256(args.tau_profile.read_bytes()).hexdigest(),
                    "selected_config": stamp["selected_config"], "output_arguments": stamp["output_arguments"],
                    "kernel_name": stamp["kernel_name"], "analysis_seconds": analysis_seconds,
                    "capture_seconds": stamp["capture_seconds"], "baseline_layouts": baseline_layouts,
                    "runtime_layouts": [r.to_dict() for r in runtime], "search": record}
                write_json(directory / "proposed-selection.json", selection)
                report.update(status="prepared", search=record)
                if not preserve_complete:
                    write_json(directory / "report.json", report)
                print(f"Prepared E{experiment} {tau} {args.case}: {len(runtime)} transformed operands")
            except Exception as error:
                report.update(status="failed", exclusion={"category": "search", "message": f"{type(error).__name__}: {error}"})
                write_json(directory / "report.json", report)


def measure(args):
    RUNNER._activate_triton_source(args.platform)
    case = CASE_BY_ID[args.case]
    failures = []
    for experiment in args.experiments:
        for tau in args.tau_names:
            directory = cell_root(args, experiment, tau)
            report_path = directory / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if report.get("status") in {"excluded", "failed"}:
                continue
            if args.stage == "measure" and report.get("status") not in {"validated", "complete"}:
                write_json(report_path, {**base_report(args, experiment, tau), **report,
                    "status": "incomplete", "stage": "preflight",
                    "exclusion": {"category": "preflight_incomplete",
                        "message": "Preflight did not finish; inspect the recorded scheduler job/log before retrying"}})
                continue
            selection_path = directory / "proposed-selection.json"
            if not selection_path.exists():
                raise ValueError(f"search did not prepare {selection_path}")
            argv = ["--experiment", str(experiment), "--platform", args.platform,
                "--operator", case.operator, "--config", case.config, "--tau-name", tau,
                "--selection", str(selection_path), "--results-root", str(args.suite_root / "tau-profiles" / tau),
                "--plots-root", str((args.plots_root or args.suite_root / "plots") / "tau-profiles" / tau),
                "--tau-profile", str(args.tau_profile), *args.measure_arguments]
            if args.rerun:
                argv.append("--rerun")
            if args.stage == "validate":
                argv.append("--validate-only")
            try:
                RUNNER.orchestrate(RUNNER.parse_arguments(argv))
            except Exception as error:
                failures.append(f"E{experiment}/{tau}: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))


def main():
    args = arguments()
    if not 0 <= args.minimum_score_gain <= 1:
        raise SystemExit("minimum score gain must lie in [0,1]")
    root = shared_root(args)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "prepare.lock").open("a") if args.stage in {"capture", "search"} else nullcontext()
    try:
        with lock as stream:
            if stream is not None:
                fcntl.flock(stream, fcntl.LOCK_EX)
            {"capture": capture, "search": search, "validate": measure, "measure": measure}[args.stage](args)
    except Exception as error:
        if args.stage not in {"validate", "measure"}:
            stage_status(args, "failed", args.stage,
                         {"category": args.stage, "message": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
