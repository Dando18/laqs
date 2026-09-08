#!/usr/bin/env python3
"""Fit the named device tau profiles from the pilot-kernel corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (str(TRITON_ROOT), str(REPOSITORY), str(EXPERIMENT_ROOT))

from analyze import spearman
from layout_panels import BYTE_SCALES, TAU_PROFILES
from stage1_counter_sweep import write_json
from tau_profiles import TAU_NAMES


TUNING_COUNTERS = {
    "tuolumne": ("l1_to_l2_read_requests",),
    "matrix": ("tex_source_l2_read_requests",),
}

TUNING_COUNTER_ALIASES = {
    "tuolumne": {
        "l1_miss_demand_to_l2": "l1_to_l2_read_requests",
    },
    "matrix": {
        "l2_read_work": "l1_to_l2_read_traffic",
        "l1_miss_demand_to_l2": "tex_source_l2_read_requests",
    },
}

EXPERT_TAU = {
    "matrix": {
        "issue.g32.stream.load.32B": 0.50,
        "issue.g32.stream.load.128B": 0.20,
        "simd_window.t16.stream.load.128B": 0.20,
        "workgroup_step.stream.load.128B": 0.10,
    },
    "tuolumne": {
        "issue.g64.stream.load.64B": 0.50,
        "issue.g64.stream.load.128B": 0.20,
        "simd_window.t16.stream.load.128B": 0.20,
        "workgroup_step.stream.load.128B": 0.10,
    },
}

FINE_COMPONENT = {
    "matrix": "issue.g32.stream.load.32B",
    "tuolumne": "issue.g64.stream.load.64B",
}

ANALYSIS_COUNTERS = {
    "tuolumne": (
        "l1_cache_line_accesses",
        "first_level_read_events",
        "l1_to_l2_read_requests",
        "l1_miss_demand_to_l2",
        "l1_to_l2_total_requests",
        "l2_tag_requests",
        "second_level_read_requests",
        "l2_hits",
        "l2_misses",
        "hbm_read_bytes",
    ),
    "matrix": (
        "first_level_memory_accesses",
        "global_load_requests",
        "tex_source_l2_read_requests",
        "l1_miss_demand_to_l2",
        "l1_to_l2_read_traffic",
        "l2_read_work",
        "l2_read_misses",
        "hbm_read_bytes",
        "sectors_per_request",
    ),
}


def _ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return result


def _reports(results_root: Path, platform: str):
    for experiment in (1, 2, 3):
        root = results_root / f"experiment-{experiment}" / platform
        yield from sorted(root.glob("stratified-*/*/report.json"))


def _observations(results_root: Path, platform: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for path in _reports(results_root, platform):
        report = json.loads(path.read_text(encoding="utf-8"))
        for candidate in report["candidates"]:
            if not candidate.get("complete") or "counters" not in candidate:
                continue
            grouped[(report["case"], candidate["mapping_id"])].append(candidate)

    observations = []
    for (case, mapping_id), candidates in grouped.items():
        component_sets = [
            {
                component["name"]: component
                for component in candidate["score"]["components"]
            }
            for candidate in candidates
        ]
        names = set(component_sets[0])
        if any(set(components) != names for components in component_sets[1:]):
            raise ValueError(f"component schema changed for {case}/{mapping_id}")
        features = {
            name: statistics.median(
                float(components[name]["excess_footprint"])
                for components in component_sets
            )
            for name in names
        }
        raw = {
            name: statistics.median(
                float(components[name]["raw_region_count"])
                for components in component_sets
            )
            for name in names
        }
        counters = {}
        for counter in ANALYSIS_COUNTERS[platform]:
            values = [
                candidate["counters"]["steady_state"].get(counter)
                for candidate in candidates
            ]
            values = [float(value) for value in values if value is not None]
            if values:
                counters[counter] = statistics.median(values)
        observations.append(
            {
                "case": case,
                "mapping_id": mapping_id,
                "features": features,
                "raw": raw,
                "counters": counters,
            }
        )
    return observations


def _case_groups(observations):
    groups = defaultdict(list)
    for observation in observations:
        groups[observation["case"]].append(observation)
    return groups


def _normalized_targets(observations, counters):
    targets = {}
    by_case = _case_groups(observations)
    for case, items in by_case.items():
        per_counter = {}
        for counter in counters:
            available = [item for item in items if counter in item["counters"]]
            values = [item["counters"][counter] for item in available]
            if len(values) < 2 or len(set(values)) < 2:
                continue
            ranks = (_ranks(values) - 1.0) / max(len(values) - 1, 1)
            per_counter[counter] = dict(
                zip((item["mapping_id"] for item in available), ranks)
            )
        for item in items:
            values = [
                ranks[item["mapping_id"]]
                for ranks in per_counter.values()
                if item["mapping_id"] in ranks
            ]
            if values:
                targets[(case, item["mapping_id"])] = float(np.mean(values))
    return targets


def _macro_spearman(observations, getter, target_getter) -> float | None:
    values = []
    for items in _case_groups(observations).values():
        pairs = [
            (getter(item), target_getter(item))
            for item in items
            if getter(item) is not None and target_getter(item) is not None
        ]
        if len(pairs) < 2:
            continue
        rho = spearman(
            [float(pair[0]) for pair in pairs],
            [float(pair[1]) for pair in pairs],
        )
        if rho is not None:
            values.append(rho)
    return float(statistics.mean(values)) if values else None


def _fit_nonnegative(observations, features, targets):
    usable = [
        item
        for item in observations
        if (item["case"], item["mapping_id"]) in targets
    ]
    x = np.asarray(
        [[item["features"][name] for name in features] for item in usable],
        dtype=float,
    )
    y = np.asarray(
        [targets[(item["case"], item["mapping_id"])] for item in usable],
        dtype=float,
    )
    for case, items in _case_groups(usable).items():
        indices = [index for index, item in enumerate(usable) if item["case"] == case]
        x[indices] -= np.mean(x[indices], axis=0)
        y[indices] -= np.mean(y[indices])
        scale = 1.0 / max(len(indices), 1) ** 0.5
        x[indices] *= scale
        y[indices] *= scale

    scales = np.sqrt(np.sum(x * x, axis=0))
    scales[scales == 0] = 1.0
    design = x / scales
    weights = np.zeros(len(features), dtype=float)
    residual = y.copy()
    ridge = 1e-4
    for _ in range(20_000):
        maximum_change = 0.0
        for index in range(len(features)):
            column = design[:, index]
            restored = residual + column * weights[index]
            updated = max(
                0.0,
                float(np.dot(column, restored))
                / (float(np.dot(column, column)) + ridge),
            )
            residual = restored - column * updated
            maximum_change = max(maximum_change, abs(updated - weights[index]))
            weights[index] = updated
        if maximum_change < 1e-10:
            break
    weights /= scales
    threshold = float(np.max(weights)) * 1e-3 if np.any(weights) else 0.0
    weights[weights < threshold] = 0.0
    if not np.any(weights):
        weights[0] = 1.0
    weights /= np.sum(weights)
    return {name: float(value) for name, value in zip(features, weights) if value}


def _tau_macro_spearman(observations, tau, targets) -> float:
    correlation = _macro_spearman(
        observations,
        lambda item: sum(
            tau.get(name, 0.0) * value
            for name, value in item["features"].items()
        ),
        lambda item: targets.get((item["case"], item["mapping_id"])),
    )
    return -2.0 if correlation is None else correlation


def _refine_rank_weights(observations, features, targets, initial):
    """Deterministically refine nonnegative tau for the reported rank metric."""

    names = tuple(features)
    steps = np.linspace(0.0, 1.0, 41)
    groups = []
    for case, items in _case_groups(observations).items():
        usable = [
            item
            for item in items
            if (case, item["mapping_id"]) in targets
        ]
        if len(usable) < 2:
            continue
        groups.append(
            (
                np.asarray(
                    [
                        [item["features"][name] for name in names]
                        for item in usable
                    ],
                    dtype=float,
                ),
                [
                    targets[(case, item["mapping_id"])]
                    for item in usable
                ],
            )
        )

    def normalized(values):
        total = float(sum(values))
        if total <= 0.0:
            raise ValueError("tau candidate must have positive weight")
        return tuple(float(value) / total for value in values)

    def record(values):
        weights = normalized(values)
        correlations = [
            spearman((matrix @ weights).tolist(), target)
            for matrix, target in groups
        ]
        informative = [
            correlation
            for correlation in correlations
            if correlation is not None
        ]
        score = (
            float(statistics.mean(informative)) if informative else -2.0
        )
        return score, weights

    candidates = [
        record(tuple(float(initial.get(name, 0.0)) for name in names))
    ]
    for index in range(len(names)):
        values = [0.0] * len(names)
        values[index] = 1.0
        candidates.append(record(values))
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            for first_weight in steps[1:-1]:
                values = [0.0] * len(names)
                values[first] = float(first_weight)
                values[second] = float(1.0 - first_weight)
                candidates.append(record(values))

    best_score, best = max(
        candidates,
        key=lambda item: (
            item[0],
            -sum(value > 1e-12 for value in item[1]),
        ),
    )
    refinement_iterations = 0
    for _ in range(8):
        candidates = [(best_score, best)]
        for index in range(len(names)):
            unit = tuple(
                1.0 if current == index else 0.0
                for current in range(len(names))
            )
            for retained_weight in steps[1:-1]:
                candidates.append(
                    record(
                        tuple(
                            float(retained_weight) * current
                            + float(1.0 - retained_weight) * added
                            for current, added in zip(best, unit)
                        )
                    )
                )
        candidate_score, candidate = max(
            candidates,
            key=lambda item: (
                item[0],
                -sum(value > 1e-12 for value in item[1]),
            ),
        )
        if candidate_score <= best_score + 1e-12:
            break
        best_score, best = candidate_score, candidate
        refinement_iterations += 1

    return (
        {
            name: value
            for name, value in zip(names, best)
            if value > 1e-12
        },
        best_score,
        refinement_iterations,
    )


def _fit_l1_to_l2_profile(
    results_root: Path, platform: str
) -> dict[str, object]:
    observations = _observations(results_root, platform)
    if not observations:
        raise FileNotFoundError(f"no scored counter observations for {platform}")
    common = set(observations[0]["features"])
    for observation in observations[1:]:
        common &= set(observation["features"])
    if not common:
        raise ValueError(f"pilot kernels share no automatic components on {platform}")

    counters = TUNING_COUNTERS[platform]
    targets = _normalized_targets(observations, counters)
    target_rho = {}
    for name in sorted(common):
        rho = _macro_spearman(
            observations,
            lambda item, component=name: item["features"][component],
            lambda item: targets.get((item["case"], item["mapping_id"])),
        )
        if rho is not None:
            target_rho[name] = rho
    ranked = sorted(target_rho, key=lambda name: (-target_rho[name], name))
    positive = [name for name in ranked if target_rho[name] > 0]
    selected = tuple((positive or ranked)[:16])
    regression_tau = _fit_nonnegative(observations, selected, targets)
    regression_rho = _tau_macro_spearman(
        observations, regression_tau, targets
    )
    tau, aggregate_rho, refinement_iterations = _refine_rank_weights(
        observations,
        selected,
        targets,
        regression_tau,
    )

    counter_components = {}
    counter_correlations = {}
    for counter in ANALYSIS_COUNTERS[platform]:
        correlations = {}
        for name in common:
            rho = _macro_spearman(
                observations,
                lambda item, component=name: item["raw"][component],
                lambda item, metric=counter: item["counters"].get(metric),
            )
            if rho is not None:
                correlations[name] = rho
        best = min(
            correlations or target_rho,
            key=lambda name: (-(correlations or target_rho)[name], name),
        )
        counter_components[counter] = best
        counter_correlations[counter] = correlations.get(best)

    return {
        "profile_id": f"pilot-{platform}-l1-to-l2-tau-v2",
        "name": "l1_to_l2",
        "fine_component": FINE_COMPONENT[platform],
        "active_tau": tau,
        "counter_components": counter_components,
        "fit": {
            "method": (
                "nonnegative_ridge_coordinate_descent_then_"
                "deterministic_spearman_refinement"
            ),
            "target": (
                "within-kernel rank of the native L1-miss demand request "
                "counter"
            ),
            "tuning_counters": list(counters),
            "excluded_tuning_counter_aliases": TUNING_COUNTER_ALIASES[platform],
            "pilot_kernels": sorted({item["case"] for item in observations}),
            "deduplicated_mapping_count": len(observations),
            "candidate_feature_count": len(common),
            "screened_feature_count": len(selected),
            "selected_feature_count": len(tau),
            "regression_macro_spearman": regression_rho,
            "rank_refinement": {
                "pair_grid_intervals": 40,
                "greedy_iteration_limit": 8,
                "completed_greedy_iterations": refinement_iterations,
                "selection_metric": "training_macro_spearman",
            },
            "training_macro_spearman": aggregate_rho,
            "counter_component_macro_spearman": counter_correlations,
        },
    }


def _speedup_groups(results_root: Path, platform: str):
    """Build deduplicated measured-layout groups for the speedup objective."""

    grouped = defaultdict(list)
    training_sources = set()
    baseline_ids = {}
    baseline_measurements = defaultdict(list)
    for path in _reports(results_root, platform):
        report = json.loads(path.read_text(encoding="utf-8"))
        experiment = int(report["final_experiment"])
        case = str(report["case"])
        if experiment == 1:
            anchors = [
                candidate
                for candidate in report["candidates"]
                if candidate.get("sampling_origin") == "row_major_anchor"
            ]
            if anchors:
                observed = str(anchors[0]["mapping_id"])
                previous = baseline_ids.setdefault(case, observed)
                if previous != observed:
                    raise ValueError(f"row-major mapping changed for {case}")
        for candidate in report["candidates"]:
            timing = candidate.get("unprofiled_timing", {})
            duration = timing.get("median_ns")
            if candidate.get("complete") and (
                timing.get("method") != "gpu_graph_replay"
                or candidate.get("realization_protocol") != "post_coalescing_rewrite_v2"
                or not timing.get("source_hash")
            ):
                raise ValueError(
                    f"{platform}/{case}: speedup fitting requires new unprofiled graph timings "
                    "and the experiment's post-coalescing realization; profiler duration is not a timing oracle"
                )
            if not candidate.get("complete") or duration is None:
                continue
            if not math.isfinite(float(duration)) or float(duration) <= 0:
                raise ValueError(f"invalid unprofiled duration for {platform}/{case}")
            training_sources.add(timing["source_hash"])
            if len(training_sources) != 1:
                raise ValueError("pilot timing/feature sources changed within the fit")
            components = {
                component["name"]: float(component["excess_footprint"])
                for component in candidate["score"]["components"]
            }
            grouped[(experiment, case, str(candidate["mapping_id"]))].append(
                {
                    "features": components,
                    "duration_ns": float(duration),
                    "address_expression_runs": int(
                        candidate.get("address_expression_runs", 1 << 30)
                    ),
                    "xor_count": int(candidate.get("xor_count", 1 << 30)),
                }
            )

    for (experiment, case, mapping_id), records in grouped.items():
        if experiment == 1 and mapping_id == baseline_ids.get(case):
            baseline_measurements[case].extend(
                record["duration_ns"] for record in records
            )

    records_by_group = defaultdict(list)
    common = None
    for (experiment, case, mapping_id), records in grouped.items():
        names = set(records[0]["features"])
        if any(set(record["features"]) != names for record in records[1:]):
            raise ValueError(
                f"component schema changed for E{experiment}/{case}/{mapping_id}"
            )
        features = {
            name: statistics.median(
                record["features"][name] for record in records
            )
            for name in names
        }
        common = names if common is None else common & names
        records_by_group[(experiment, case)].append(
            {
                "mapping_id": mapping_id,
                "features": features,
                "duration_ns": statistics.median(
                    record["duration_ns"] for record in records
                ),
                "address_expression_runs": min(
                    record["address_expression_runs"] for record in records
                ),
                "xor_count": min(record["xor_count"] for record in records),
            }
        )

    if not common:
        raise ValueError(f"no common speedup components for {platform}")
    names = tuple(sorted(common))
    result = []
    for (experiment, case), records in sorted(records_by_group.items()):
        if len(records) < 2:
            continue
        baseline_id = baseline_ids.get(case)
        local_baseline = next(
            (
                record["duration_ns"]
                for record in records
                if record["mapping_id"] == baseline_id
            ),
            None,
        )
        if local_baseline is None:
            values = baseline_measurements.get(case, ())
            if not values:
                raise ValueError(f"no row-major timing for {platform}/{case}")
            local_baseline = statistics.median(values)
        result.append(
            {
                "experiment": experiment,
                "case": case,
                "baseline_mapping_id": baseline_id,
                "source_hash": next(iter(training_sources)),
                "baseline_duration_ns": float(local_baseline),
                "mapping_ids": tuple(record["mapping_id"] for record in records),
                "features": np.asarray(
                    [
                        [record["features"][name] for name in names]
                        for record in records
                    ],
                    dtype=float,
                ),
                "duration_ns": np.asarray(
                    [record["duration_ns"] for record in records], dtype=float
                ),
                "complexity": tuple(
                    (
                        record["address_expression_runs"],
                        record["xor_count"],
                        record["mapping_id"],
                    )
                    for record in records
                ),
            }
        )
    return names, result


def _speedup_objective(names, groups, tau):
    indices = [names.index(name) for name in tau]
    weights = np.asarray([tau[name] for name in tau], dtype=float)
    speedups = []
    selections = []
    for group in groups:
        scores = group["features"][:, indices] @ weights
        minimum = float(np.min(scores))
        tied = np.flatnonzero(
            np.isclose(scores, minimum, rtol=1e-12, atol=1e-12)
        )
        selected = min(
            map(int, tied),
            key=lambda index: (
                group["mapping_ids"][index]
                != group["baseline_mapping_id"],
                *group["complexity"][index],
            ),
        )
        speedup = float(
            group["baseline_duration_ns"] / group["duration_ns"][selected]
        )
        speedups.append(speedup)
        selections.append(
            {
                "experiment": group["experiment"],
                "case": group["case"],
                "mapping_id": group["mapping_ids"][selected],
                "measured_speedup": speedup,
            }
        )
    geometric_mean = math.exp(
        statistics.mean(math.log(value) for value in speedups)
    )
    return {
        "geometric_mean_speedup": geometric_mean,
        "regression_count": sum(value < 1.0 - 1e-12 for value in speedups),
        "median_speedup": statistics.median(speedups),
        "minimum_speedup": min(speedups),
        "selections": selections,
    }


def _speedup_key(record, active_count):
    return (
        record["geometric_mean_speedup"],
        -record["regression_count"],
        record["median_speedup"],
        record["minimum_speedup"],
        -active_count,
    )


def _normalized_tau(tau):
    positive = {
        str(name): float(value)
        for name, value in tau.items()
        if float(value) > 1e-12
    }
    total = sum(positive.values())
    if not total:
        raise ValueError("tau candidate has no positive weights")
    return {name: value / total for name, value in positive.items()}


def _fit_speedup_profile(
    results_root: Path,
    platform: str,
    counter_components,
    seeds,
) -> dict[str, object]:
    names, groups = _speedup_groups(results_root, platform)

    def evaluate(tau):
        normalized = _normalized_tau(tau)
        return _speedup_objective(names, groups, normalized), normalized

    one_hot = []
    for name in names:
        objective, tau = evaluate({name: 1.0})
        one_hot.append((objective, tau))
    one_hot.sort(
        key=lambda item: _speedup_key(item[0], len(item[1])), reverse=True
    )
    screened = tuple(next(iter(tau)) for _, tau in one_hot[:16])
    candidates = list(one_hot[:16])
    for seed in seeds:
        if set(seed) <= set(names):
            candidates.append(evaluate(seed))

    steps = np.linspace(0.0, 1.0, 41)
    for first_index, first in enumerate(screened):
        for second in screened[first_index + 1 :]:
            for first_weight in steps[1:-1]:
                candidates.append(
                    evaluate(
                        {
                            first: float(first_weight),
                            second: float(1.0 - first_weight),
                        }
                    )
                )
    best = max(
        candidates,
        key=lambda item: _speedup_key(item[0], len(item[1])),
    )
    completed_iterations = 0
    for _ in range(8):
        refinements = [best]
        for name in screened:
            for retained_weight in steps[1:-1]:
                candidate = {
                    component: float(retained_weight) * weight
                    for component, weight in best[1].items()
                }
                candidate[name] = candidate.get(name, 0.0) + float(
                    1.0 - retained_weight
                )
                refinements.append(evaluate(candidate))
        refined = max(
            refinements,
            key=lambda item: _speedup_key(item[0], len(item[1])),
        )
        if (
            refined[0]["geometric_mean_speedup"]
            <= best[0]["geometric_mean_speedup"] + 1e-12
        ):
            break
        best = refined
        completed_iterations += 1

    objective, tau = best
    return {
        "profile_id": f"pilot-{platform}-speedup-tau-v3",
        "name": "speedup",
        "fine_component": FINE_COMPONENT[platform],
        "active_tau": tau,
        "counter_components": dict(counter_components),
        "fit": {
            "measurement_protocol": "post_coalescing_rewrite_v2",
            "timing_method": "gpu_graph_replay",
            "source_hash": groups[0]["source_hash"],
            "method": (
                "one_hot_screen_then_pair_grid_and_greedy_mixture_search"
            ),
            "target": (
                "geometric mean measured speedup of the lowest-J_area layout "
                "in each deduplicated pilot kernel/grammar panel"
            ),
            "selection_rule": (
                "minimize J_area, prefer ordinary row-major on exact ties, "
                "then address-expression complexity"
            ),
            "evaluation_leakage_policy": (
                "pilot kernels only; TritonBench and real kernels excluded"
            ),
            "pilot_kernels": sorted({group["case"] for group in groups}),
            "pilot_kernel_grammar_groups": len(groups),
            "candidate_feature_count": len(names),
            "screened_feature_count": len(screened),
            "selected_feature_count": len(tau),
            "pair_grid_intervals": 40,
            "greedy_iteration_limit": 8,
            "completed_greedy_iterations": completed_iterations,
            "training_geometric_mean_speedup": objective[
                "geometric_mean_speedup"
            ],
            "training_median_speedup": objective["median_speedup"],
            "training_minimum_speedup": objective["minimum_speedup"],
            "training_regression_count": objective["regression_count"],
            "training_selections": objective["selections"],
            "scope": (
                "empirical support of Experiments 1--3; the exact search in "
                "Experiments 4--6 minimizes the same J_area but can reach "
                "unprofiled layouts"
            ),
        },
    }


def _expert_profile(platform: str, counter_components) -> dict[str, object]:
    tau = _normalized_tau(EXPERT_TAU[platform])
    return {
        "profile_id": f"expert-{platform}-hierarchy-tau-v2",
        "name": "expert",
        "fine_component": FINE_COMPONENT[platform],
        "active_tau": tau,
        "counter_components": dict(counter_components),
        "fit": {
            "method": "hardware_informed_hand_authored",
            "target": (
                "hierarchy-faithful coalescing and short-range reuse heuristic"
            ),
            "uses_measured_counters_or_runtime": False,
            "rationale": (
                "H100 combines 32-thread/32-byte sector issue locality with "
                "128-byte line-scale issue and short temporal/workgroup reuse"
                if platform == "matrix"
                else "MI300A combines 64-lane/64-byte vector-L1 issue locality "
                "with 128-byte line-scale issue and short temporal/workgroup reuse"
            ),
        },
    }


def _apply_profile(report: dict[str, object], profile: dict[str, object]) -> None:
    tau = profile["active_tau"]
    for candidate in report["candidates"]:
        components = candidate["score"]["components"]
        active = []
        for component in components:
            weight = float(tau.get(component["name"], 0.0))
            component["weight"] = weight
            component["peak_tolerance"] = 1.0 if weight else None
            component["peak_excess_ratio"] = (
                float(component["normalized_excess"]) if weight else None
            )
            component["weighted_region_count"] = weight * float(
                component["raw_region_count"]
            )
            component["weighted_normalized_excess"] = weight * float(
                component["normalized_excess"]
            )
            component["weighted_excess_footprint"] = weight * float(
                component["excess_footprint"]
            )
            if weight:
                active.append(component)
        aggregates = candidate["score"]["aggregates"]
        aggregates["weighted_region_count"] = sum(
            component["weighted_region_count"] for component in active
        )
        aggregates["peak_normalized_excess"] = max(
            (component["normalized_excess"] for component in active), default=0.0
        )
        aggregates["weighted_normalized_excess"] = sum(
            component["weighted_normalized_excess"] for component in active
        )
        aggregates["hardware_peak"] = aggregates["peak_normalized_excess"]
        aggregates["hardware_area"] = sum(
            component["weighted_excess_footprint"] for component in active
        )
        candidate["j_area"] = float(aggregates["hardware_area"])
        candidate["peak_normalized_excess"] = float(
            aggregates["peak_normalized_excess"]
        )

    panel = report["panel"]
    levels = sorted({candidate["j_area"] for candidate in report["candidates"]})
    by_mapping = {
        candidate["mapping_id"]: candidate for candidate in report["candidates"]
    }
    for candidate in panel["candidates"]:
        scored = by_mapping[candidate["mapping_id"]]
        candidate["j_area"] = scored["j_area"]
        candidate["peak_normalized_excess"] = scored["peak_normalized_excess"]
        candidate["score"] = scored["score"]
        candidate["j_area_rank"] = levels.index(candidate["j_area"]) + 1
        scored["j_area_rank"] = candidate["j_area_rank"]
    panel["j_area_levels"] = levels
    panel["score_profile"].update(
        {
            "profile_id": profile["profile_id"],
            "tau_name": profile["name"],
            "fine_component": profile["fine_component"],
            "active_tau": tau,
            "counter_components": profile["counter_components"],
            "tuning": profile["fit"],
        }
    )
    report["configuration"]["score_profile"] = profile["profile_id"]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=EXPERIMENT_ROOT / "results"
    )
    parser.add_argument("--output", type=Path, default=TAU_PROFILES)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    platforms = {}
    for platform in ("tuolumne", "matrix"):
        expert = _expert_profile(platform, {})
        l1_to_l2 = _fit_l1_to_l2_profile(args.results_root, platform)
        expert["counter_components"] = dict(l1_to_l2["counter_components"])
        speedup = _fit_speedup_profile(
            args.results_root,
            platform,
            l1_to_l2["counter_components"],
            (expert["active_tau"], l1_to_l2["active_tau"]),
        )
        profiles = {
            "expert": expert,
            "l1_to_l2": l1_to_l2,
            "speedup": speedup,
        }
        if tuple(profiles) != TAU_NAMES:
            raise AssertionError("tau profile order changed")
        platforms[platform] = {
            "default_profile": "expert",
            "profiles": profiles,
        }
    document = {
        "schema": "relay.triton.tau_profiles",
        "version": 2,
        "graph_construction": "automatic_post_coalescing_manifest_universal_v1",
        "byte_scales": list(BYTE_SCALES),
        "profile_order": list(TAU_NAMES),
        "platforms": platforms,
    }
    write_json(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
