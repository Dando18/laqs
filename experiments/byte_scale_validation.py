#!/usr/bin/env python3
"""Validate one hardware profile across FP16, FP32, and FP64 storage.

The experiment scores the complete 73-layout canonical corpus at each element
width, forms the ordinary memory-score frontier with one unchanged hardware
profile, and benchmarks a compact panel: the union of those dtype-specific
frontiers plus row- and column-major controls. Runtime regret is therefore
reported against the measured panel, not against every canonical layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.layout_ranking import (
    KERNEL_SPECS,
    layout_cases,
    layouts_for_case,
    notes_pareto_frontier,
    parse_evaluator_output,
)
from relay import UniversalScopeObjectives, get_hardware_profile, score_layouts
from relay.evaluator_dtype import EVALUATOR_DTYPES, get_evaluator_dtype
from relay.objectives import ObjectiveComponent, build_objectives


SUPPORTED_KERNELS = ("atax", "gesummv")
DEFAULT_DTYPES = ("fp64", "fp32", "fp16")


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        action="append",
        choices=SUPPORTED_KERNELS,
        default=None,
        help="kernel to test; repeat as needed (default: ATAX and GESUMMV)",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=DEFAULT_DTYPES,
        default=None,
        help="storage type to test; repeat as needed (default: all three)",
    )
    parser.add_argument(
        "--size",
        action="append",
        type=positive_integer,
        default=None,
        help="matrix size; repeat as needed (default: 256)",
    )
    parser.add_argument("--samples", type=positive_integer, default=5)
    parser.add_argument("--iterations", type=positive_integer, default=3)
    parser.add_argument("--warmup", type=nonnegative_integer, default=2)
    parser.add_argument("--device", type=nonnegative_integer, default=0)
    parser.add_argument("--block-size", type=positive_integer, default=128)
    parser.add_argument(
        "--compiler", default="/opt/rocm-7.0.2/bin/hipcc"
    )
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--hardware-profile", default="mi300a")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="score and create an untimed checkpoint without a GPU",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume pending panel benchmarks from --output",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=positive_integer,
        default=None,
        help="run at most this many pending evaluator jobs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/byte_scale_validation_mi300a.json"),
    )
    return parser, parser.parse_args(argv)


def _configuration(
    args: argparse.Namespace,
    kernels: Sequence[str],
    dtypes: Sequence[str],
    sizes: Sequence[int],
) -> dict[str, object]:
    profile = get_hardware_profile(args.hardware_profile)
    return {
        "kernels": list(kernels),
        "dtypes": list(dtypes),
        "sizes": list(sizes),
        "samples": args.samples,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "device": args.device,
        "block_size": args.block_size,
        "compiler": args.compiler,
        "arch": args.arch,
        "hardware_profile": args.hardware_profile,
        "hardware_profile_definition": profile.to_dict(),
        "seed": args.seed,
    }


def _edge_geometry_hash(components: Sequence[ObjectiveComponent]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for component in components:
        family = component.edge_family
        if family is None or family in seen:
            continue
        seen.add(family)
        digest.update(family.encode())
        for array, edges in sorted(component.edges_by_array.items()):
            digest.update(array.encode())
            for edge in edges:
                digest.update(repr((edge.weight, edge.points)).encode())
    return digest.hexdigest()


def _score_group(
    kernel: str,
    dtype: str,
    n: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    spec = KERNEL_SPECS[kernel]
    dtype_spec = get_evaluator_dtype(dtype)
    config = spec.problem.build_config(
        problem_size=n,
        block_size=args.block_size,
        element_bytes=dtype_spec.element_bytes,
    )
    matrices_tuple = tuple(spec.problem.get_matrices(config))
    event_items, sequences = spec.problem.get_events_and_sequences(config)
    matrices = {matrix.name: matrix for matrix in matrices_tuple}
    events = {event.id: event for event in event_items}
    profile = get_hardware_profile(args.hardware_profile)
    components = tuple(
        build_objectives(
            (UniversalScopeObjectives(profile.byte_scales),),
            matrices,
            events,
            sequences,
        )
    )
    scores = {}
    cases = layout_cases(n)
    for case in cases:
        scores[case.name] = score_layouts(
            matrices,
            components,
            layouts_for_case(case, matrices),
            hardware_profile=profile,
        )
    frontier = notes_pareto_frontier(scores, profile.fine_component)
    case_by_name = {case.name: case for case in cases}
    members = []
    for member in frontier["members"]:
        name = str(member["name"])
        members.append(
            {
                "name": name,
                "word": case_by_name[name].word,
                "values": member["values"],
            }
        )
    normalization_values = {
        float(component.normalization_bytes)
        for component in components
        if component.normalization_bytes is not None
    }
    if len(normalization_values) != 1:
        raise RuntimeError("universal components do not share one B_K")
    return {
        "kernel": kernel,
        "display_name": spec.display_name,
        "dtype": dtype,
        "dtype_label": dtype_spec.label,
        "element_bytes": dtype_spec.element_bytes,
        "matrix_size": n,
        "layout_count": len(cases),
        "frontier": {
            "definition": [item["name"] for item in frontier["objectives"]],
            "candidate_count": len(members),
            "members": members,
        },
        "edge_geometry_sha256": _edge_geometry_hash(components),
        "normalization_bytes": normalization_values.pop(),
        "component_names": [component.name for component in components],
        "active_tau": {
            name: value
            for name, value in profile.tau.items()
            if name in {component.name for component in components}
        },
        "active_kappa": {
            name: value
            for name, value in profile.kappa.items()
            if name in {component.name for component in components}
        },
        "panel": [],
        "runtime_analysis": None,
    }


def prepare_report(
    args: argparse.Namespace,
    kernels: Sequence[str],
    dtypes: Sequence[str],
    sizes: Sequence[int],
) -> dict[str, object]:
    groups = [
        _score_group(kernel, dtype, n, args)
        for kernel in kernels
        for n in sizes
        for dtype in dtypes
    ]
    all_cases = {
        n: {case.name: case for case in layout_cases(n)} for n in sizes
    }
    for kernel in kernels:
        for n in sizes:
            related = [
                group
                for group in groups
                if group["kernel"] == kernel and group["matrix_size"] == n
            ]
            panel_names = {"row_major", "column_major"}
            for group in related:
                panel_names.update(
                    str(member["name"])
                    for member in group["frontier"]["members"]
                )
            ordered_names = [
                case.name for case in layout_cases(n) if case.name in panel_names
            ]
            for group in related:
                frontier_names = {
                    str(member["name"])
                    for member in group["frontier"]["members"]
                }
                group["panel"] = [
                    {
                        "name": name,
                        "word": all_cases[n][name].word,
                        "frontier_member": name in frontier_names,
                        "timing": None,
                        "command": None,
                        "stdout": None,
                        "stderr": None,
                    }
                    for name in ordered_names
                ]
    order = [
        {
            "kernel": group["kernel"],
            "dtype": group["dtype"],
            "matrix_size": group["matrix_size"],
            "layout": record["name"],
        }
        for group in groups
        for record in group["panel"]
    ]
    random.Random(args.seed).shuffle(order)
    profile = get_hardware_profile(args.hardware_profile)
    report = {
        "experiment": "byte-scale-validation",
        "configuration": _configuration(args, kernels, dtypes, sizes),
        "hardware_profile": profile.to_dict(),
        "scope": (
            "complete 73-layout analytical frontiers; measured panel is the "
            "cross-dtype frontier union plus row/column-major controls"
        ),
        "benchmark_run_order": order,
        "groups": groups,
        "complete": False,
        "aggregate": None,
    }
    _validate_invariants(report)
    return report


def _validate_invariants(report: Mapping[str, object]) -> None:
    groups = report["groups"]
    assert isinstance(groups, list)
    buckets: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for group in groups:
        assert isinstance(group, dict)
        buckets.setdefault(
            (str(group["kernel"]), int(group["matrix_size"])), []
        ).append(group)
    for key, related in buckets.items():
        geometry = {str(group["edge_geometry_sha256"]) for group in related}
        if len(geometry) != 1:
            raise ValueError(f"edge geometry changes with dtype for {key}")
        names = {tuple(group["component_names"]) for group in related}
        if len(names) != 1:
            raise ValueError(f"component schema changes with dtype for {key}")
        tau = {json.dumps(group["active_tau"], sort_keys=True) for group in related}
        kappa = {
            json.dumps(group["active_kappa"], sort_keys=True) for group in related
        }
        if len(tau) != 1 or len(kappa) != 1:
            raise ValueError(f"hardware response changes with dtype for {key}")
        exposures = {
            int(group["element_bytes"]): float(group["normalization_bytes"])
            for group in related
        }
        reference = next(iter(related))
        reference_ratio = (
            float(reference["normalization_bytes"])
            / int(reference["element_bytes"])
        )
        if any(
            abs(value / width - reference_ratio) > 1.0e-9
            for width, value in exposures.items()
        ):
            raise ValueError(f"B_K does not scale with element bytes for {key}")


def _find_record(
    report: Mapping[str, object], job: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    groups = report["groups"]
    assert isinstance(groups, list)
    group = next(
        item
        for item in groups
        if item["kernel"] == job["kernel"]
        and item["dtype"] == job["dtype"]
        and item["matrix_size"] == job["matrix_size"]
    )
    record = next(item for item in group["panel"] if item["name"] == job["layout"])
    return group, record


def _evaluator_command(
    group: Mapping[str, object],
    record: Mapping[str, object],
    args: argparse.Namespace,
) -> list[str]:
    spec = KERNEL_SPECS[str(group["kernel"])]
    command = [
        sys.executable,
        str(spec.evaluator),
        *[str(record["word"]) for _ in spec.evaluator_arrays],
        "--dtype",
        str(group["dtype"]),
        "--n",
        str(group["matrix_size"]),
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--device",
        str(args.device),
        "--block-size",
        str(args.block_size),
        "--compiler",
        args.compiler,
    ]
    if args.arch:
        command.extend(("--arch", args.arch))
    return command


def _timing_dict(output: str) -> dict[str, object]:
    timing = parse_evaluator_output(output)
    return {
        "device": timing.device,
        "median_ms": timing.median_ms,
        "mean_ms": timing.mean_ms,
        "min_ms": timing.min_ms,
        "sd_ms": timing.sd_ms,
        "gflops": timing.gflops,
        "samples_ms": list(timing.samples_ms),
    }


def update_analysis(report: dict[str, object]) -> None:
    groups = report["groups"]
    assert isinstance(groups, list)
    complete = True
    regrets = []
    for group in groups:
        records = group["panel"]
        if any(record["timing"] is None for record in records):
            group["runtime_analysis"] = None
            complete = False
            continue
        best = min(records, key=lambda item: item["timing"]["median_ms"])
        frontier_records = [item for item in records if item["frontier_member"]]
        best_frontier = min(
            frontier_records, key=lambda item: item["timing"]["median_ms"]
        )
        row_major = next(item for item in records if item["name"] == "row_major")
        regret = (
            best_frontier["timing"]["median_ms"]
            / best["timing"]["median_ms"]
            - 1.0
        )
        regrets.append(regret)
        group["runtime_analysis"] = {
            "reference": "best median runtime in the measured cross-dtype panel",
            "panel_size": len(records),
            "frontier_size": len(frontier_records),
            "panel_winner": best["name"],
            "panel_winner_ms": best["timing"]["median_ms"],
            "best_frontier_layout": best_frontier["name"],
            "best_frontier_ms": best_frontier["timing"]["median_ms"],
            "panel_oracle_regret": regret,
            "row_major_ms": row_major["timing"]["median_ms"],
            "speedup_over_row_major": (
                row_major["timing"]["median_ms"]
                / best_frontier["timing"]["median_ms"]
            ),
        }
    report["complete"] = complete
    report["aggregate"] = (
        {
            "instance_count": len(regrets),
            "exact_panel_winner_coverage": sum(regret == 0.0 for regret in regrets),
            "within_one_percent": sum(regret <= 0.01 for regret in regrets),
            "mean_panel_oracle_regret": sum(regrets) / len(regrets),
            "maximum_panel_oracle_regret": max(regrets),
        }
        if complete and regrets
        else None
    )


def markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        "# MI300A byte-scale validation",
        "",
        "One unchanged edge construction and hardware profile are evaluated at",
        "2-, 4-, and 8-byte elements. FP16 uses FP16 storage and FP32 accumulation.",
        "Runtime regret is against the measured cross-dtype candidate panel, not",
        "the complete 73-layout corpus.",
        "",
        (
            "| Kernel | N | Type | Analytical frontier | Panel | Regret | "
            "Speedup vs row-major |"
        ),
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    groups = report["groups"]
    assert isinstance(groups, list)
    for group in groups:
        analysis = group["runtime_analysis"]
        if analysis is None:
            regret = "pending"
            speedup = "pending"
        else:
            regret = f"{100.0 * analysis['panel_oracle_regret']:.3f}%"
            speedup = f"{analysis['speedup_over_row_major']:.3f}x"
        lines.append(
            f"| {group['display_name']} | {group['matrix_size']} | "
            f"{group['dtype_label']} | {group['frontier']['candidate_count']}/73 | "
            f"{len(group['panel'])} | {regret} | {speedup} |"
        )
    lines.extend(("", "## Analytical invariants", ""))
    lines.append(
        "For each kernel/size pair, edge geometry, component names, tau, and kappa "
        "are identical across data types. The useful-byte denominator scales "
        "exactly with element width."
    )
    aggregate = report.get("aggregate")
    if isinstance(aggregate, dict):
        lines.extend(
            (
                "",
                "## Aggregate measured-panel result",
                "",
                f"- Exact panel winner: {aggregate['exact_panel_winner_coverage']}/"
                f"{aggregate['instance_count']}.",
                f"- Within 1%: {aggregate['within_one_percent']}/"
                f"{aggregate['instance_count']}.",
                f"- Mean regret: {100.0 * aggregate['mean_panel_oracle_regret']:.3f}%.",
                (
                    "- Maximum regret: "
                    f"{100.0 * aggregate['maximum_panel_oracle_regret']:.3f}%."
                ),
            )
        )
    return "\n".join(lines) + "\n"


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    markdown_path = path.with_suffix(".md")
    markdown_temporary = markdown_path.with_name(markdown_path.name + ".tmp")
    markdown_temporary.write_text(markdown_report(report))
    markdown_temporary.replace(markdown_path)


def run(argv: Sequence[str] | None = None) -> int:
    parser, args = parse_arguments(argv)
    kernels = tuple(dict.fromkeys(args.kernel or SUPPORTED_KERNELS))
    dtypes = tuple(dict.fromkeys(args.dtype or DEFAULT_DTYPES))
    sizes = tuple(dict.fromkeys(args.size or (256,)))
    output = args.output.expanduser().resolve()
    expected_configuration = _configuration(args, kernels, dtypes, sizes)
    if args.resume:
        if not output.exists():
            parser.error(f"resume report does not exist: {output}")
        report = json.loads(output.read_text())
        if report.get("configuration") != expected_configuration:
            parser.error("resume report configuration does not match")
    else:
        report = prepare_report(args, kernels, dtypes, sizes)
        update_analysis(report)
        _write_report(output, report)
    if args.prepare_only:
        print(f"Prepared {output}")
        return 0

    jobs = report["benchmark_run_order"]
    pending = []
    for job in jobs:
        _, record = _find_record(report, job)
        if record["timing"] is None:
            pending.append(job)
    limit = len(pending) if args.max_benchmarks is None else args.max_benchmarks
    for job in pending[:limit]:
        group, record = _find_record(report, job)
        command = _evaluator_command(group, record, args)
        print(
            f"Benchmarking {group['display_name']} N={group['matrix_size']} "
            f"{group['dtype']} {record['name']}",
            flush=True,
        )
        completed = subprocess.run(command, text=True, capture_output=True)
        record["command"] = command
        record["stdout"] = completed.stdout
        record["stderr"] = completed.stderr
        if completed.returncode != 0:
            _write_report(output, report)
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode
        record["timing"] = _timing_dict(completed.stdout)
        update_analysis(report)
        _write_report(output, report)
    update_analysis(report)
    _write_report(output, report)
    remaining = sum(
        record["timing"] is None
        for group in report["groups"]
        for record in group["panel"]
    )
    if remaining:
        print(f"Checkpointed {output}; {remaining} benchmarks remain")
    else:
        print(f"Completed {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
