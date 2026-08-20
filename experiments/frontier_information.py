"""Information-preserving frontier diagnostics for layout experiments.

The primary RELAY frontier intentionally uses a compact five-cost vector.
This module tests how much candidate information is lost by that compression:
it compares aggregate, componentwise, stream-partitioned, and dense-scale
quotient signatures against the same stored runtime observations.
"""

from __future__ import annotations

from math import ceil
import statistics
from typing import Mapping, Sequence

from relay.layouts import Layout
from relay.model import MatrixSpec
from relay.objectives import Hyperedge, ObjectiveComponent
from relay.scoring import excess_footprint, normalized_excess


REPRESENTATIONS = (
    "aggregate",
    "active-components",
    "all-components",
    "stream-split",
    "dense-scales",
)

REPRESENTATION_LABELS = {
    "aggregate": "F_agg",
    "active-components": "F_active",
    "all-components": "F_all",
    "stream-split": "F_split",
    "dense-scales": "F_dense-d",
}


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("a metric summary requires at least one value")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _source_semantic(source: str) -> str | None:
    lowered = source.lower()
    for token in ("stage1", "stage2", "transpose", "row_i", "row_j"):
        if token in lowered:
            return token
    if ".row." in lowered or ".row[" in lowered:
        return "row"
    return None


def _edge_curve(
    matrix: MatrixSpec,
    layout: Layout,
    edge: Hyperedge,
    dimensions: int,
    offset_cache: dict[tuple[int, ...], int],
    curve_cache: dict[tuple[str, tuple[tuple[int, ...], ...]], tuple[int, ...]],
) -> tuple[int, ...]:
    cache_key = (matrix.name, edge.points)
    cached = curve_cache.get(cache_key)
    if cached is not None:
        return cached
    offsets = []
    for point in edge.points:
        offset = offset_cache.get(point)
        if offset is None:
            offset = layout.offset(matrix, point)
            offset_cache[point] = offset
        offsets.append(offset)
    offsets.sort()
    if not offsets:
        curve = (0,) * (dimensions + 1)
        curve_cache[cache_key] = curve
        return curve

    highest_differences = [0] * dimensions
    for left, right in zip(offsets, offsets[1:]):
        highest = (left ^ right).bit_length() - 1
        if highest >= 0:
            highest_differences[min(highest, dimensions - 1)] += 1

    curve = [1] * (dimensions + 1)
    boundaries = 0
    for dimension in range(dimensions - 1, -1, -1):
        boundaries += highest_differences[dimension]
        curve[dimension] += boundaries
    result = tuple(curve)
    curve_cache[cache_key] = result
    return result


def diagnostic_layout_signature(
    matrices: Mapping[str, MatrixSpec],
    components: Sequence[ObjectiveComponent],
    layouts: Mapping[str, Layout],
    *,
    offset_cache_by_array: Mapping[
        str, dict[tuple[int, ...], int]
    ] | None = None,
) -> dict[str, object]:
    """Return stream partitions and complete quotient curves for one layout.

    Dense curves evaluate every existing target-array edge family at every
    feasible element-region dimension from zero through the largest target
    address width. Stream partitions preserve the original joint component
    and expose array, stage, or directional subsets where edge provenance
    makes that split available.
    """

    targets = {name for name, matrix in matrices.items() if matrix.target}
    if not targets:
        raise ValueError("diagnostic signatures require a target matrix")
    dimensions = max(
        sum(matrices[name].mode_bits) for name in targets
    )
    offset_caches = (
        {name: offset_cache_by_array[name] for name in targets}
        if offset_cache_by_array is not None
        else {name: {} for name in targets}
    )
    curve_cache: dict[
        tuple[str, tuple[tuple[int, ...], ...]], tuple[int, ...]
    ] = {}
    dense_values: dict[str, float] = {}
    dense_families_seen: set[str] = set()
    split_scores: dict[str, dict[str, object]] = {}

    for component in components:
        curves = [0.0] * (dimensions + 1)
        partitions: dict[
            str, list[tuple[str, Hyperedge]]
        ] = {}
        for array, edges in component.edges_by_array.items():
            if array not in targets:
                continue
            matrix = matrices[array]
            layout = layouts[array]
            for edge in edges:
                curve = _edge_curve(
                    matrix,
                    layout,
                    edge,
                    dimensions,
                    offset_caches[array],
                    curve_cache,
                )
                for dimension, count in enumerate(curve):
                    curves[dimension] += edge.weight * count
                semantic = _source_semantic(edge.source)
                partition = array if semantic is None else f"{array}.{semantic}"
                partitions.setdefault(partition, []).append((array, edge))

        family_name = component.edge_family or component.name
        if any(curves) and family_name not in dense_families_seen:
            dense_families_seen.add(family_name)
            for dimension, value in enumerate(curves):
                dense_values[f"{family_name}.d{dimension}"] = value

        if len(partitions) <= 1:
            continue
        for partition, members in sorted(partitions.items()):
            raw = 0.0
            bound = 0.0
            for array, edge in members:
                matrix = matrices[array]
                capacity = component.region_bytes // matrix.element_bytes
                dimension = capacity.bit_length() - 1
                curve = _edge_curve(
                    matrix,
                    layouts[array],
                    edge,
                    dimensions,
                    offset_caches[array],
                    curve_cache,
                )
                raw += edge.weight * curve[min(dimension, dimensions)]
                bound += edge.weight * ceil(len(edge.points) / capacity)
            name = f"{component.name}::{partition}"
            split_scores[name] = {
                "component": component.name,
                "partition": partition,
                "region_bytes": component.region_bytes,
                "raw_region_count": raw,
                "packing_lower_bound": bound,
                "normalized_excess": normalized_excess(raw, bound),
                "excess_footprint": excess_footprint(
                    raw,
                    bound,
                    component.region_bytes,
                    component.normalization_bytes,
                ),
            }

    return {
        "dense_scales": {
            "dimensions": list(range(dimensions + 1)),
            "values": dense_values,
        },
        "stream_split": split_scores,
    }


def _runtime(record: Mapping[str, object]) -> float:
    timing = record["timing"]
    assert isinstance(timing, dict)
    return float(timing["median_ms"])


def _timing_range(record: Mapping[str, object]) -> tuple[float, float]:
    timing = record["timing"]
    assert isinstance(timing, dict)
    samples = [float(value) for value in timing["samples_ms"]]
    return min(samples), max(samples)


def _component_map(
    record: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    score = record["score"]
    assert isinstance(score, dict)
    components = score["components"]
    assert isinstance(components, list)
    return {str(component["name"]): component for component in components}


def representation_vectors(
    group: Mapping[str, object], representation: str
) -> tuple[tuple[str, ...], dict[str, tuple[float, ...]]]:
    """Build one named frontier vector for every result in a group."""

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown frontier representation {representation!r}")
    records = group["results"]
    assert isinstance(records, list) and records
    fine_component = str(group["fine_component"])
    first_components = _component_map(records[0])
    component_names = tuple(first_components)
    active_names = tuple(
        name
        for name, component in first_components.items()
        if float(component["weight"]) > 0.0
        or component.get("peak_tolerance") is not None
    )

    if representation == "aggregate":
        objectives = (
            f"{fine_component}.raw-region-count",
            "hardware-peak",
            "hardware-area",
            "codegen-runs",
            "codegen-xors",
        )
    elif representation == "active-components":
        objectives = (
            f"{fine_component}.raw-region-count",
            *(f"{name}.excess-footprint" for name in active_names),
            "codegen-runs",
            "codegen-xors",
        )
    elif representation == "all-components":
        objectives = (
            f"{fine_component}.raw-region-count",
            *(f"{name}.excess-footprint" for name in component_names),
            "codegen-runs",
            "codegen-xors",
        )
    else:
        diagnostics = records[0].get("diagnostic_signatures")
        if not isinstance(diagnostics, dict):
            raise ValueError(
                f"{representation} requires diagnostic layout signatures"
            )
        if representation == "stream-split":
            split = diagnostics["stream_split"]
            assert isinstance(split, dict)
            objectives = (
                f"{fine_component}.raw-region-count",
                *(f"{name}.excess-footprint" for name in component_names),
                *(f"{name}.excess-footprint" for name in sorted(split)),
                "codegen-runs",
                "codegen-xors",
            )
        else:
            dense = diagnostics["dense_scales"]
            assert isinstance(dense, dict)
            values = dense["values"]
            assert isinstance(values, dict)
            objectives = (
                *tuple(sorted(values)),
                "codegen-runs",
                "codegen-xors",
            )

    vectors = {}
    for record in records:
        components = _component_map(record)
        score = record["score"]
        assert isinstance(score, dict)
        aggregates = score["aggregates"]
        codegen = score["codegen"]
        assert isinstance(aggregates, dict) and isinstance(codegen, dict)
        fine = float(components[fine_component]["raw_region_count"])
        tail = (float(codegen["runs"]), float(codegen["xors"]))
        if representation == "aggregate":
            vector = (
                fine,
                float(aggregates["hardware_peak"]),
                float(aggregates["hardware_area"]),
                *tail,
            )
        elif representation == "active-components":
            vector = (
                fine,
                *(float(components[name]["excess_footprint"]) for name in active_names),
                *tail,
            )
        elif representation == "all-components":
            vector = (
                fine,
                *(
                    float(components[name]["excess_footprint"])
                    for name in component_names
                ),
                *tail,
            )
        elif representation == "stream-split":
            diagnostics = record["diagnostic_signatures"]
            split = diagnostics["stream_split"]
            split_names = tuple(
                name.removesuffix(".excess-footprint")
                for name in objectives[1 + len(component_names) : -2]
            )
            vector = (
                fine,
                *(
                    float(components[name]["excess_footprint"])
                    for name in component_names
                ),
                *(
                    float(split[name]["excess_footprint"])
                    for name in split_names
                ),
                *tail,
            )
        else:
            diagnostics = record["diagnostic_signatures"]
            dense = diagnostics["dense_scales"]["values"]
            vector = (
                *(float(dense[name]) for name in objectives[:-2]),
                *tail,
            )
        vectors[str(record["name"])] = tuple(vector)
    return tuple(objectives), vectors


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _pareto_layers(
    vectors: Mapping[str, tuple[float, ...]],
) -> tuple[dict[str, int], list[list[str]]]:
    remaining = set(vectors)
    layer_by_name: dict[str, int] = {}
    layers = []
    depth = 1
    while remaining:
        frontier = sorted(
            name
            for name in remaining
            if not any(
                other != name
                and _dominates(vectors[other], vectors[name])
                for other in remaining
            )
        )
        for name in frontier:
            layer_by_name[name] = depth
        layers.append(frontier)
        remaining.difference_update(frontier)
        depth += 1
    return layer_by_name, layers


def _equivalence_analysis(
    records_by_name: Mapping[str, Mapping[str, object]],
    vectors: Mapping[str, tuple[float, ...]],
) -> dict[str, object]:
    by_vector: dict[tuple[float, ...], list[str]] = {}
    for name, vector in vectors.items():
        by_vector.setdefault(vector, []).append(name)
    groups = []
    spreads = []
    for vector, names in sorted(by_vector.items()):
        runtimes = [_runtime(records_by_name[name]) for name in names]
        spread = max(runtimes) / min(runtimes) - 1.0
        if len(names) > 1:
            spreads.append(spread)
        groups.append(
            {
                "score_vector": list(vector),
                "layouts": sorted(names),
                "layout_count": len(names),
                "minimum_runtime_ms": min(runtimes),
                "maximum_runtime_ms": max(runtimes),
                "runtime_spread": spread,
            }
        )
    return {
        "group_count": len(groups),
        "non_singleton_group_count": sum(
            int(group["layout_count"]) > 1 for group in groups
        ),
        "layouts_in_non_singleton_groups": sum(
            int(group["layout_count"])
            for group in groups
            if int(group["layout_count"]) > 1
        ),
        "non_singleton_runtime_spread": _summary(spreads) if spreads else None,
        "groups": groups,
    }


def _dominance_analysis(
    records_by_name: Mapping[str, Mapping[str, object]],
    vectors: Mapping[str, tuple[float, ...]],
) -> dict[str, object]:
    pair_count = 0
    median_violations = []
    confirmed_violations = []
    for dominator, left in vectors.items():
        for dominated, right in vectors.items():
            if dominator == dominated or not _dominates(left, right):
                continue
            pair_count += 1
            dominator_runtime = _runtime(records_by_name[dominator])
            dominated_runtime = _runtime(records_by_name[dominated])
            if dominated_runtime >= dominator_runtime:
                continue
            violation = {
                "dominator": dominator,
                "dominated_faster_layout": dominated,
                "dominator_runtime_ms": dominator_runtime,
                "dominated_runtime_ms": dominated_runtime,
                "runtime_penalty": dominator_runtime / dominated_runtime - 1.0,
            }
            median_violations.append(violation)
            dominator_lower, _ = _timing_range(records_by_name[dominator])
            _, dominated_upper = _timing_range(records_by_name[dominated])
            if dominated_upper < dominator_lower:
                confirmed_violations.append(violation)
    key = lambda item: float(item["runtime_penalty"])
    return {
        "dominance_pair_count": pair_count,
        "median_runtime_violation_count": len(median_violations),
        "confirmed_nonoverlap_violation_count": len(confirmed_violations),
        "median_runtime_violation_fraction": (
            0.0 if pair_count == 0 else len(median_violations) / pair_count
        ),
        "confirmed_nonoverlap_violation_fraction": (
            0.0 if pair_count == 0 else len(confirmed_violations) / pair_count
        ),
        "worst_median_runtime_violations": sorted(
            median_violations, key=key, reverse=True
        )[:20],
        "worst_confirmed_nonoverlap_violations": sorted(
            confirmed_violations, key=key, reverse=True
        )[:20],
    }


def _winner_certificates(
    records_by_name: Mapping[str, Mapping[str, object]],
    objectives: Sequence[str],
    vectors: Mapping[str, tuple[float, ...]],
    frontier_names: set[str],
) -> list[dict[str, object]]:
    optimum = min(_runtime(record) for record in records_by_name.values())
    winners = [
        name
        for name, record in records_by_name.items()
        if _runtime(record) == optimum and name not in frontier_names
    ]
    certificates = []
    for winner in winners:
        winner_components = _component_map(records_by_name[winner])
        dominators = []
        for name, vector in vectors.items():
            if not _dominates(vector, vectors[winner]):
                continue
            components = _component_map(records_by_name[name])
            dominators.append(
                {
                    "name": name,
                    "runtime_ms": _runtime(records_by_name[name]),
                    "runtime_penalty": (
                        _runtime(records_by_name[name]) / optimum - 1.0
                    ),
                    "representation_deltas": {
                        objective: left - right
                        for objective, left, right in zip(
                            objectives, vector, vectors[winner]
                        )
                    },
                    "component_excess_deltas": {
                        component_name: {
                            "weight": float(component["weight"]),
                            "winner": float(
                                winner_components[component_name][
                                    "normalized_excess"
                                ]
                            ),
                            "dominator": float(component["normalized_excess"]),
                            "delta": float(component["normalized_excess"])
                            - float(
                                winner_components[component_name][
                                    "normalized_excess"
                                ]
                            ),
                        }
                        for component_name, component in components.items()
                    },
                }
            )
        certificates.append(
            {
                "winner": winner,
                "winner_runtime_ms": optimum,
                "winner_timing_range_ms": list(
                    _timing_range(records_by_name[winner])
                ),
                "dominator_count": len(dominators),
                "dominators": sorted(
                    dominators,
                    key=lambda item: (float(item["runtime_ms"]), str(item["name"])),
                ),
            }
        )
    return certificates


def analyze_frontier_information(
    groups: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Construct and evaluate the complete frontier-information ladder."""

    representations = []
    for representation in REPRESENTATIONS:
        instances = []
        all_equivalence_spreads = []
        all_regrets = []
        all_retained = []
        total_dominance_pairs = 0
        total_median_violations = 0
        total_confirmed_violations = 0
        for group in groups:
            records = group["results"]
            assert isinstance(records, list)
            records_by_name = {
                str(record["name"]): record for record in records
            }
            objectives, vectors = representation_vectors(group, representation)
            layer_by_name, layers = _pareto_layers(vectors)
            frontier_names = set(layers[0])
            optimum = min(_runtime(record) for record in records)
            frontier_optimum = min(
                _runtime(records_by_name[name]) for name in frontier_names
            )
            regret = frontier_optimum / optimum - 1.0
            retained = len(frontier_names) / len(records)
            one_percent = any(
                _runtime(records_by_name[name]) <= 1.01 * optimum
                for name in frontier_names
            )
            equivalence = _equivalence_analysis(records_by_name, vectors)
            spread = equivalence["non_singleton_runtime_spread"]
            if isinstance(spread, dict):
                all_equivalence_spreads.extend(
                    float(item["runtime_spread"])
                    for item in equivalence["groups"]
                    if int(item["layout_count"]) > 1
                )
            dominance = _dominance_analysis(records_by_name, vectors)
            total_dominance_pairs += int(dominance["dominance_pair_count"])
            total_median_violations += int(
                dominance["median_runtime_violation_count"]
            )
            total_confirmed_violations += int(
                dominance["confirmed_nonoverlap_violation_count"]
            )

            cumulative = []
            cumulative_names: set[str] = set()
            for depth, layer in enumerate(layers, 1):
                cumulative_names.update(layer)
                best = min(
                    _runtime(records_by_name[name]) for name in cumulative_names
                )
                cumulative.append(
                    {
                        "depth": depth,
                        "candidate_count": len(cumulative_names),
                        "retained_fraction": len(cumulative_names) / len(records),
                        "oracle_regret": best / optimum - 1.0,
                        "one_percent_covered": best <= 1.01 * optimum,
                    }
                )

            for record in records:
                memberships = record.setdefault("frontier_representations", {})
                assert isinstance(memberships, dict)
                name = str(record["name"])
                memberships[representation] = {
                    "member": name in frontier_names,
                    "pareto_depth": layer_by_name[name],
                }

            certificates = _winner_certificates(
                records_by_name,
                objectives,
                vectors,
                frontier_names,
            )
            all_regrets.append(regret)
            all_retained.append(retained)
            instances.append(
                {
                    "kernel": str(group["kernel"]),
                    "display_name": str(group["display_name"]),
                    "matrix_size": int(group["matrix_size"]),
                    "objective_count": len(objectives),
                    "objectives": list(objectives),
                    "frontier_size": len(frontier_names),
                    "frontier_members": sorted(frontier_names),
                    "retained_fraction": retained,
                    "optimal_runtime_ms": optimum,
                    "best_frontier_runtime_ms": frontier_optimum,
                    "oracle_regret": regret,
                    "one_percent_covered": one_percent,
                    "equivalence": equivalence,
                    "dominance_violations": dominance,
                    "missed_winner_certificates": certificates,
                    "pareto_layer_count": len(layers),
                    "pareto_layers": [
                        {"depth": depth, "members": layer}
                        for depth, layer in enumerate(layers, 1)
                    ],
                    "cumulative_pareto_depth": cumulative,
                }
            )

        maximum_depth = max(
            int(instance["pareto_layer_count"]) for instance in instances
        )
        depth_summary = []
        for depth in range(1, maximum_depth + 1):
            entries = [
                instance["cumulative_pareto_depth"][
                    min(depth, int(instance["pareto_layer_count"])) - 1
                ]
                for instance in instances
            ]
            depth_summary.append(
                {
                    "depth": depth,
                    "oracle_regret": _summary(
                        [float(entry["oracle_regret"]) for entry in entries]
                    ),
                    "retained_fraction": _summary(
                        [float(entry["retained_fraction"]) for entry in entries]
                    ),
                    "one_percent_covered_instances": sum(
                        bool(entry["one_percent_covered"]) for entry in entries
                    ),
                }
            )

        representations.append(
            {
                "name": representation,
                "label": REPRESENTATION_LABELS[representation],
                "oracle_regret": _summary(all_regrets),
                "retained_fraction": _summary(all_retained),
                "exact_winner_covered_instances": sum(
                    float(instance["oracle_regret"]) == 0.0
                    for instance in instances
                ),
                "one_percent_covered_instances": sum(
                    bool(instance["one_percent_covered"])
                    for instance in instances
                ),
                "equivalence": {
                    "non_singleton_runtime_spread": (
                        _summary(all_equivalence_spreads)
                        if all_equivalence_spreads
                        else None
                    )
                },
                "dominance_violations": {
                    "dominance_pair_count": total_dominance_pairs,
                    "median_runtime_violation_count": total_median_violations,
                    "confirmed_nonoverlap_violation_count": (
                        total_confirmed_violations
                    ),
                    "median_runtime_violation_fraction": (
                        0.0
                        if total_dominance_pairs == 0
                        else total_median_violations / total_dominance_pairs
                    ),
                    "confirmed_nonoverlap_violation_fraction": (
                        0.0
                        if total_dominance_pairs == 0
                        else total_confirmed_violations / total_dominance_pairs
                    ),
                },
                "cumulative_pareto_depth": depth_summary,
                "instances": instances,
            }
        )

    return {
        "runtime_statistic": "median_ms",
        "instance_count": len(groups),
        "representations": representations,
        "definitions": {
            "aggregate": (
                "Q_fine, J_peak, J_area, codegen runs, and codegen XORs"
            ),
            "active-components": (
                "Q_fine, every exposure-weighted component feature with tau "
                "or kappa support, "
                "codegen runs, and codegen XORs"
            ),
            "all-components": (
                "Q_fine, every universal exposure-weighted component including "
                "zero-weight diagnostics, codegen runs, and codegen XORs"
            ),
            "stream-split": (
                "the all-component vector plus source-derived array, stage, "
                "row, and transpose component partitions"
            ),
            "dense-scales": (
                "Q_s(V_d) for every existing target edge family and every "
                "feasible element-region dimension d, plus codegen costs"
            ),
            "confirmed_dominance_violation": (
                "a dominated layout's maximum observed sample is below its "
                "analytical dominator's minimum observed sample"
            ),
        },
    }
