#!/usr/bin/env python3
"""Evaluate temporal quotient persistence on a completed scoring corpus.

This is a score-only experiment: it reconstructs ``J_persist`` for every
layout in an exhaustive plan, joins the existing timing checkpoint, and
evaluates Pareto frontiers over combinations of Q_fine, J_peak, J_area,
frozen J_place, and persistence variants. It never invokes a GPU evaluator.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.layout_ranking import KERNEL_SPECS
from relay import (
    TemporalPersistenceBasis,
    build_transition_families,
    canonical_layout_from_word,
    get_hardware_profile,
    score_temporal_persistence,
)


DEFAULT_PLAN = Path("results/gesummv1024_oracle_feature_audit_mi300a.plan.json")
DEFAULT_RAW = Path("results/gesummv1024_oracle_feature_audit_mi300a.raw.jsonl")
DEFAULT_SUMMARY = Path("results/gesummv1024_oracle_feature_audit_mi300a.jsonl")
DEFAULT_OUTPUT = Path("results/gesummv1024_persistence_frontier_mi300a.json")
DEFAULT_MARKDOWN = Path("results/gesummv1024_persistence_frontier_mi300a.md")
BASE_OBJECTIVES = ("Q_fine", "J_peak", "J_area", "J_place")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--delta",
        type=int,
        action="append",
        default=None,
        help="transition distance; repeat as needed (default: 1, 4, 16)",
    )
    parser.add_argument(
        "--family",
        choices=("simd_stream", "lane_stream", "simd_schedule"),
        action="append",
        default=None,
        help="transition family; repeat as needed (default: all)",
    )
    return parser.parse_args(argv)


def _load_timings(path: Path) -> dict[tuple[str, int, str], dict[str, object]]:
    records = {}
    with path.open() as stream:
        metadata = json.loads(stream.readline())
        if metadata.get("record_type") != "metadata":
            raise ValueError(f"{path}: expected raw timing metadata")
        for line_number, line in enumerate(stream, 2):
            record = json.loads(line)
            if record.get("record_type") != "timing":
                raise ValueError(f"{path}:{line_number}: expected timing record")
            key = (
                str(record["kernel"]),
                int(record["matrix_size"]),
                str(record["word"]),
            )
            records[key] = record["timing"]
    return records


def _primitive_objectives(
    features: Mapping[str, object], profile: object
) -> dict[str, float]:
    locality = features["locality"]
    assert isinstance(locality, dict)
    area = sum(
        float(weight)
        * float(locality.get(name, {}).get("excess_footprint", 0.0))
        for name, weight in profile.tau.items()
    )
    peak = max(
        (
            float(locality.get(name, {}).get("normalized_excess", 0.0))
            / float(tolerance)
            for name, tolerance in profile.kappa.items()
        ),
        default=0.0,
    )
    placements = features["placement"]
    assert isinstance(placements, list)
    place = sum(
        float(placement["weight"]) * float(placement["robust_contention"])
        for placement in placements
    )
    return {
        "Q_fine": float(features["q_fine"]),
        "J_peak": peak,
        "J_area": area,
        "J_place": place,
    }


def _pareto_words(
    values: Mapping[str, Mapping[str, float]], objectives: Sequence[str]
) -> list[str]:
    def dominates(left: str, right: str) -> bool:
        no_worse = True
        strict = False
        for objective in objectives:
            left_value = values[left][objective]
            right_value = values[right][objective]
            if left_value > right_value:
                no_worse = False
                break
            strict |= left_value < right_value
        return no_worse and strict

    frontier: list[str] = []
    for word in values:
        if any(dominates(other, word) for other in frontier):
            continue
        frontier = [
            other for other in frontier if not dominates(word, other)
        ]
        frontier.append(word)
    return sorted(frontier)


def _nonempty_subsets(values: Sequence[str]) -> Iterable[tuple[str, ...]]:
    for size in range(1, len(values) + 1):
        yield from combinations(values, size)


def _frontier_record(
    objective_values: Mapping[str, Mapping[str, float]],
    runtimes: Mapping[str, float],
    objectives: Sequence[str],
) -> dict[str, object]:
    words = _pareto_words(objective_values, objectives)
    best_time = min(runtimes[word] for word in words)
    oracle_time = min(runtimes.values())
    return {
        "objectives": list(objectives),
        "size": len(words),
        "best_time_ms": best_time,
        "regret": best_time / oracle_time - 1.0,
        "best_words": sorted(
            word for word in words if runtimes[word] == best_time
        ),
        "oracle_retained": any(
            runtimes[word] == oracle_time for word in words
        ),
        "words": words,
    }


def _diagnostic_roles(
    group: Mapping[str, object],
    summary: Mapping[str, object] | None,
    runtimes: Mapping[str, float],
) -> dict[str, str]:
    oracle_word = min(runtimes, key=lambda word: (runtimes[word], word))
    roles = {"oracle": oracle_word}
    reranker = group["placement_reranker"]
    assert isinstance(reranker, dict)
    ranked = reranker["ranked"]
    assert isinstance(ranked, list)
    roles["current_selection"] = str(ranked[0]["word"])
    if summary is not None:
        audit = summary.get("oracle_feature_audit")
        if isinstance(audit, dict):
            for record in audit.get("oracles", []):
                if record.get("oracle_word") != oracle_word:
                    continue
                dominators = record.get("dominating_words", [])
                if dominators:
                    roles["near_oracle_dominator"] = str(dominators[0])
                break
    return roles


def _group_report(
    group: Mapping[str, object],
    configuration: Mapping[str, object],
    timings: Mapping[tuple[str, int, str], Mapping[str, object]],
    summary: Mapping[str, object] | None,
    basis: TemporalPersistenceBasis,
) -> dict[str, object]:
    kernel = str(group["kernel"])
    n = int(group["matrix_size"])
    spec = KERNEL_SPECS[kernel]
    if spec.block_style == "2d":
        block_size: object = (
            int(configuration["block_x"]),
            int(configuration["block_y"]),
            1,
        )
    else:
        block_size = int(configuration["block_size"])
    config = spec.problem.build_config(problem_size=n, block_size=block_size)
    matrices_tuple = tuple(spec.problem.get_matrices(config))
    matrices = {matrix.name: matrix for matrix in matrices_tuple}
    event_items, sequences = spec.problem.get_events_and_sequences(config)
    families = build_transition_families(
        matrices,
        {event.id: event for event in event_items},
        sequences,
        basis=basis,
    )

    feature_vectors = group.get("primitive_features")
    if not isinstance(feature_vectors, dict) or not feature_vectors:
        raise ValueError(
            "persistence experiment requires a plan built with "
            "--dump-oracle-components or --check-oracle-feature-dominance"
        )
    words = sorted(str(word) for word in feature_vectors)
    profile = get_hardware_profile(str(configuration["hardware_profile"]))
    target_matrices = tuple(matrix for matrix in matrices_tuple if matrix.target)
    objective_values: dict[str, dict[str, float]] = {}
    component_values: dict[str, dict[str, float]] = {}
    persistence_variants: set[str] = set()

    for word in words:
        layouts = {
            matrix.name: canonical_layout_from_word(
                matrix, word, name=f"persist_{word}.{matrix.name}"
            )
            for matrix in target_matrices
        }
        persistence = score_temporal_persistence(
            matrices,
            layouts,
            families,
            profile.byte_scales,
        )
        values = _primitive_objectives(feature_vectors[word], profile)
        components = {
            component.name: {
                "weighted_new_demand": component.weighted_new_demand,
                "mean_turnover": component.normalized_turnover,
            }
            for component in persistence.components
        }
        variants: dict[str, float] = {
            "J_persist": persistence.hardware_persist,
            "J_persist.mean_cells": sum(
                component.normalized_turnover
                for component in persistence.components
            ),
        }
        for family_name in basis.families:
            variants[f"J_persist.{family_name}"] = sum(
                component.weighted_new_demand
                for component in persistence.components
                if component.transition_family == family_name
            )
            variants[f"J_persist.mean.{family_name}"] = sum(
                component.normalized_turnover
                for component in persistence.components
                if component.transition_family == family_name
            )
        for delta in basis.deltas:
            variants[f"J_persist.d{delta}"] = sum(
                component.weighted_new_demand
                for component in persistence.components
                if component.delta == delta
            )
            variants[f"J_persist.mean.d{delta}"] = sum(
                component.normalized_turnover
                for component in persistence.components
                if component.delta == delta
            )
        for scale in profile.byte_scales:
            variants[f"J_persist.{scale}B"] = sum(
                component.weighted_new_demand
                for component in persistence.components
                if component.region_bytes == scale
            )
        for family_name in basis.families:
            for delta in basis.deltas:
                variants[f"J_persist.{family_name}.d{delta}"] = sum(
                    component.weighted_new_demand
                    for component in persistence.components
                    if component.transition_family == family_name
                    and component.delta == delta
                )
            for scale in profile.byte_scales:
                variants[f"J_persist.{family_name}.{scale}B"] = sum(
                    component.weighted_new_demand
                    for component in persistence.components
                    if component.transition_family == family_name
                    and component.region_bytes == scale
                )
        for delta in basis.deltas:
            for scale in profile.byte_scales:
                variants[f"J_persist.d{delta}.{scale}B"] = sum(
                    component.weighted_new_demand
                    for component in persistence.components
                    if component.delta == delta
                    and component.region_bytes == scale
                )
        for component in persistence.components:
            variants[
                f"J_persist.{component.transition_family}.d{component.delta}."
                f"{component.region_bytes}B"
            ] = component.weighted_new_demand
        values.update(variants)
        objective_values[word] = values
        component_values[word] = components
        persistence_variants.update(variants)

    runtimes = {}
    for word in words:
        timing = timings.get((kernel, n, word))
        if timing is None:
            raise ValueError(f"missing timing for {kernel} N={n} word {word}")
        runtimes[word] = float(timing["median_ms"])

    frontiers: dict[tuple[str, ...], dict[str, object]] = {}
    for objectives in _nonempty_subsets(BASE_OBJECTIVES):
        frontiers[objectives] = _frontier_record(
            objective_values, runtimes, objectives
        )
    for persistence_name in sorted(persistence_variants):
        for base_subset_size in range(len(BASE_OBJECTIVES) + 1):
            for base_subset in combinations(BASE_OBJECTIVES, base_subset_size):
                objectives = (*base_subset, persistence_name)
                frontiers[objectives] = _frontier_record(
                    objective_values, runtimes, objectives
                )

    frontier_records = sorted(
        frontiers.values(),
        key=lambda record: (
            float(record["regret"]),
            int(record["size"]),
            len(record["objectives"]),
            record["objectives"],
        ),
    )
    roles = _diagnostic_roles(group, summary, runtimes)
    oracle_time = min(runtimes.values())
    diagnostics = {}
    for role, word in roles.items():
        diagnostics[role] = {
            "word": word,
            "median_ms": runtimes[word],
            "regret": runtimes[word] / oracle_time - 1.0,
            "objectives": objective_values[word],
            "persistence_components": component_values[word],
        }

    baseline_key = ("Q_fine", "J_peak", "J_area")
    persist_key = (*baseline_key, "J_persist")
    place_key = (*baseline_key, "J_place")
    all_key = (*baseline_key, "J_place", "J_persist")
    highlighted = {
        "current_locality": frontiers[baseline_key],
        "locality_plus_persist": frontiers[persist_key],
        "locality_plus_place": frontiers[place_key],
        "all_five": frontiers[all_key],
    }
    near_targets = [
        record
        for record in frontier_records
        if float(record["regret"]) < 0.01 and int(record["size"]) < 10
    ]
    ideal_targets = [
        record
        for record in frontier_records
        if float(record["regret"]) < 0.01 and int(record["size"]) < 5
    ]
    below_one_percent = [
        record
        for record in frontier_records
        if float(record["regret"]) < 0.01
    ]
    below_ten_samples = [
        record
        for record in frontier_records
        if int(record["size"]) < 10
    ]
    below_five_samples = [
        record
        for record in frontier_records
        if int(record["size"]) < 5
    ]
    return {
        "kernel": kernel,
        "matrix_size": n,
        "layout_count": len(words),
        "oracle_time_ms": oracle_time,
        "transition_basis": {
            "deltas": list(basis.deltas),
            "families": list(basis.families),
            "region_bytes": list(profile.byte_scales),
            "allocation_identity": "tagged; quotient ids never alias across arrays",
            "component_aggregation": (
                "J_persist = sum_{family,delta,scale} sum_t w_t nu with "
                "rho=1; weighted cell means and sparse rho ablations are "
                "reported as separate experimental variants"
            ),
            "relations": [
                {
                    "name": family.name,
                    "transition_count": family.transition_count,
                    "compressed_count": len(family.transitions),
                    "transition_weight": family.transition_weight,
                }
                for family in families
            ],
        },
        "diagnostics": diagnostics,
        "highlighted_frontiers": highlighted,
        "target_frontiers": near_targets,
        "ideal_target_frontiers": ideal_targets,
        "boundary_results": {
            "smallest_below_one_percent": min(
                below_one_percent,
                key=lambda record: (
                    int(record["size"]), float(record["regret"])
                ),
            ),
            "best_below_ten_samples": min(
                below_ten_samples,
                key=lambda record: (
                    float(record["regret"]), int(record["size"])
                ),
            ),
            "best_below_five_samples": min(
                below_five_samples,
                key=lambda record: (
                    float(record["regret"]), int(record["size"])
                ),
            ),
        },
        "all_frontiers": frontier_records,
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Temporal quotient persistence frontier experiment",
        "",
        "This score-only experiment reuses the exhaustive measured G_S corpus; "
        "no new timings were collected. J_place is the frozen corrected robust "
        "statistic from the input plan.",
        "",
    ]
    aggregate = report.get("aggregate")
    if isinstance(aggregate, dict):
        lines.extend(
            (
                "## Cross-kernel objective tradeoff",
                "",
                "A shared combination uses the same objective coordinates for "
                "every kernel and size. Worst regret and maximum frontier size "
                "are taken over all reported instances.",
                "",
                "| Boundary | Objectives | Max samples | Mean samples | "
                "Worst regret |",
                "|---|---|---:|---:|---:|",
            )
        )
        for name, frontier in aggregate["boundary_results"].items():
            lines.append(
                f"| `{name}` | `{', '.join(frontier['objectives'])}` | "
                f"{frontier['maximum_size']} | "
                f"{float(frontier['mean_size']):.2f} | "
                f"{100 * float(frontier['worst_regret']):.3f}% |"
            )
        lines.extend(
            (
                "",
                "### Best shared combination per kernel",
                "",
                "| Kernel | Boundary | Objectives | Max samples | Worst regret |",
                "|---|---|---|---:|---:|",
            )
        )
        for kernel, kernel_summary in aggregate["kernels"].items():
            for boundary_name in (
                "smallest_below_one_percent",
                "best_below_ten_samples",
                "best_below_five_samples",
            ):
                frontier = kernel_summary["boundary_results"][boundary_name]
                lines.append(
                    f"| `{kernel}` | `{boundary_name}` | "
                    f"`{', '.join(frontier['objectives'])}` | "
                    f"{frontier['maximum_size']} | "
                    f"{100 * float(frontier['worst_regret']):.3f}% |"
                )
        lines.append("")
    for group in report["groups"]:
        lines.extend(
            (
                f"## {str(group['kernel']).upper()} N={group['matrix_size']}",
                "",
                f"Oracle median: {float(group['oracle_time_ms']):.6f} ms over "
                f"{group['layout_count']} layouts.",
                "",
                "### Main frontier comparison",
                "",
                "| Frontier | Samples | Best regret | Oracle retained |",
                "|---|---:|---:|:---:|",
            )
        )
        for name, frontier in group["highlighted_frontiers"].items():
            lines.append(
                f"| `{name}` | {frontier['size']} | "
                f"{100 * float(frontier['regret']):.3f}% | "
                f"{'yes' if frontier['oracle_retained'] else 'no'} |"
            )
        lines.extend(
            (
                "",
                "### Diagnostic layouts",
                "",
                "| Role | Word | Runtime regret | J_persist |",
                "|---|---|---:|---:|",
            )
        )
        for role, diagnostic in group["diagnostics"].items():
            lines.append(
                f"| `{role}` | `{diagnostic['word']}` | "
                f"{100 * float(diagnostic['regret']):.3f}% | "
                f"{float(diagnostic['objectives']['J_persist']):.6f} |"
            )
        lines.extend(
            (
                "",
                "### Combinations meeting <1% regret and <10 samples",
                "",
                "| Objectives | Samples | Best regret |",
                "|---|---:|---:|",
            )
        )
        targets = group["target_frontiers"]
        if targets:
            for frontier in targets:
                lines.append(
                    f"| `{', '.join(frontier['objectives'])}` | "
                    f"{frontier['size']} | "
                    f"{100 * float(frontier['regret']):.3f}% |"
                )
        else:
            lines.append("| _None_ | — | — |")
        lines.extend(
            (
                "",
                "### Target boundary",
                "",
                "The objective variants below are post-hoc ablations on this "
                "one measured instance; they are diagnostics, not calibrated "
                "transferable weights.",
                "",
                "| Boundary | Objectives | Samples | Best regret |",
                "|---|---|---:|---:|",
            )
        )
        for name, frontier in group["boundary_results"].items():
            lines.append(
                f"| `{name}` | `{', '.join(frontier['objectives'])}` | "
                f"{frontier['size']} | "
                f"{100 * float(frontier['regret']):.3f}% |"
            )
        lines.extend(("", "### Best compact combinations", ""))
        lines.extend(
            (
                "| Objectives | Samples | Best regret |",
                "|---|---:|---:|",
            )
        )
        compact = sorted(
            (
                frontier
                for frontier in group["all_frontiers"]
                if int(frontier["size"]) < 10
            ),
            key=lambda frontier: (
                float(frontier["regret"]), int(frontier["size"])
            ),
        )[:10]
        for frontier in compact:
            lines.append(
                f"| `{', '.join(frontier['objectives'])}` | "
                f"{frontier['size']} | "
                f"{100 * float(frontier['regret']):.3f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def _aggregate_combination(
    objectives: tuple[str, ...],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sizes = [int(record["size"]) for record in records]
    regrets = [float(record["regret"]) for record in records]
    return {
        "objectives": list(objectives),
        "instance_count": len(records),
        "maximum_size": max(sizes),
        "mean_size": sum(sizes) / len(sizes),
        "total_size": sum(sizes),
        "worst_regret": max(regrets),
        "mean_regret": sum(regrets) / len(regrets),
        "below_one_percent_count": sum(regret < 0.01 for regret in regrets),
        "instances": [
            {
                "kernel": record["kernel"],
                "matrix_size": record["matrix_size"],
                "size": record["size"],
                "regret": record["regret"],
            }
            for record in records
        ],
    }


def _aggregate_boundaries(
    combinations: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    below_one = [
        record for record in combinations if float(record["worst_regret"]) < 0.01
    ]
    below_ten = [
        record for record in combinations if int(record["maximum_size"]) < 10
    ]
    below_five = [
        record for record in combinations if int(record["maximum_size"]) < 5
    ]
    return {
        "smallest_below_one_percent": min(
            below_one,
            key=lambda record: (
                int(record["maximum_size"]),
                float(record["mean_size"]),
                float(record["worst_regret"]),
            ),
        ),
        "best_below_ten_samples": min(
            below_ten,
            key=lambda record: (
                float(record["worst_regret"]),
                int(record["maximum_size"]),
                float(record["mean_size"]),
            ),
        ),
        "best_below_five_samples": min(
            below_five,
            key=lambda record: (
                float(record["worst_regret"]),
                int(record["maximum_size"]),
                float(record["mean_size"]),
            ),
        ),
    }


def _aggregate_report(
    groups: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records_by_objectives: dict[
        tuple[str, ...], list[dict[str, object]]
    ] = {}
    for group in groups:
        for frontier in group["all_frontiers"]:
            objectives = tuple(str(value) for value in frontier["objectives"])
            records_by_objectives.setdefault(objectives, []).append(
                {
                    "kernel": group["kernel"],
                    "matrix_size": group["matrix_size"],
                    "size": frontier["size"],
                    "regret": frontier["regret"],
                }
            )
    expected_count = len(groups)
    combinations = [
        _aggregate_combination(objectives, records)
        for objectives, records in records_by_objectives.items()
        if len(records) == expected_count
    ]
    combinations.sort(
        key=lambda record: (
            float(record["worst_regret"]),
            int(record["maximum_size"]),
            float(record["mean_size"]),
            record["objectives"],
        )
    )

    kernel_reports = {}
    for kernel in sorted({str(group["kernel"]) for group in groups}):
        kernel_groups = [
            group for group in groups if str(group["kernel"]) == kernel
        ]
        kernel_records: dict[tuple[str, ...], list[dict[str, object]]] = {}
        for group in kernel_groups:
            for frontier in group["all_frontiers"]:
                objectives = tuple(
                    str(value) for value in frontier["objectives"]
                )
                kernel_records.setdefault(objectives, []).append(
                    {
                        "kernel": kernel,
                        "matrix_size": group["matrix_size"],
                        "size": frontier["size"],
                        "regret": frontier["regret"],
                    }
                )
        kernel_combinations = [
            _aggregate_combination(objectives, records)
            for objectives, records in kernel_records.items()
            if len(records) == len(kernel_groups)
        ]
        kernel_reports[kernel] = {
            "instance_count": len(kernel_groups),
            "boundary_results": _aggregate_boundaries(kernel_combinations),
        }

    return {
        "instance_count": expected_count,
        "kernel_count": len(kernel_reports),
        "shared_combination_count": len(combinations),
        "target_combinations": [
            record
            for record in combinations
            if float(record["worst_regret"]) < 0.01
            and int(record["maximum_size"]) < 10
        ],
        "boundary_results": _aggregate_boundaries(combinations),
        "kernels": kernel_reports,
        "all_combinations": combinations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    deltas = tuple(sorted(set(args.delta or (1, 4, 16))))
    if any(delta <= 0 for delta in deltas):
        raise ValueError("transition deltas must be positive")
    families = tuple(
        dict.fromkeys(
            args.family or ("simd_stream", "lane_stream", "simd_schedule")
        )
    )
    basis = TemporalPersistenceBasis(deltas=deltas, families=families)
    plan = json.loads(args.plan.read_text())
    configuration = plan["configuration"]
    timings = _load_timings(args.raw)
    summaries = {}
    if args.summary.exists():
        summaries = {
            (str(record["kernel"]), int(record["matrix_size"])): record
            for record in (
                json.loads(line)
                for line in args.summary.read_text().splitlines()
                if line.strip()
            )
        }
    groups = [
        _group_report(
            group,
            configuration,
            timings,
            summaries.get((str(group["kernel"]), int(group["matrix_size"]))),
            basis,
        )
        for group in plan["groups"]
    ]
    report = {
        "experiment": "temporal-quotient-persistence-frontier",
        "schema_version": 1,
        "source_plan": str(args.plan),
        "source_raw": str(args.raw),
        "timing_policy": "reuse only; no benchmarks executed",
        "groups": groups,
        "aggregate": _aggregate_report(groups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report) + "\n")
    for group in groups:
        current = group["highlighted_frontiers"]["current_locality"]
        persist = group["highlighted_frontiers"]["locality_plus_persist"]
        print(
            f"{group['kernel']} N={group['matrix_size']}: locality "
            f"{current['size']} layouts/{100 * current['regret']:.3f}% regret; "
            f"+J_persist {persist['size']} layouts/"
            f"{100 * persist['regret']:.3f}% regret; "
            f"target combinations={len(group['target_frontiers'])}"
        )
    print(args.output)
    print(args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
