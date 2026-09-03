#!/usr/bin/env python3
"""Fit device tau profiles from pilot-kernel counters and rescore reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRITON_ROOT = EXPERIMENT_ROOT.parent
REPOSITORY = TRITON_ROOT.parent
sys.path[:0] = (str(TRITON_ROOT), str(REPOSITORY), str(EXPERIMENT_ROOT))

from analyze import analyze_report, analyze_suite, spearman
from layout_panels import BYTE_SCALES, TAU_PROFILES
from stage1_counter_sweep import write_json


TUNING_COUNTERS = {
    "tuolumne": (
        "l1_cache_line_accesses",
        "l1_to_l2_read_requests",
        "second_level_read_requests",
    ),
    "matrix": (
        "first_level_memory_accesses",
        "l1_to_l2_read_traffic",
    ),
}

TUNING_COUNTER_ALIASES = {
    "tuolumne": {},
    "matrix": {
        "l2_read_work": "l1_to_l2_read_traffic",
    },
}

ANALYSIS_COUNTERS = {
    "tuolumne": (
        "l1_cache_line_accesses",
        "first_level_read_events",
        "l1_to_l2_read_requests",
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
        yield from sorted(root.glob("*/report.json"))


def _observations(results_root: Path, platform: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for path in _reports(results_root, platform):
        report = json.loads(path.read_text(encoding="utf-8"))
        for candidate in report["candidates"]:
            if "counters" not in candidate:
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


def _fit_platform(results_root: Path, platform: str) -> dict[str, object]:
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
        "profile_id": f"pilot-automatic-{platform}-tau-v1",
        "active_tau": tau,
        "counter_components": counter_components,
        "fit": {
            "method": (
                "nonnegative_ridge_coordinate_descent_then_"
                "deterministic_spearman_refinement"
            ),
            "target": "mean within-kernel rank of informative L1/L2 counters",
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
            "active_tau": tau,
            "counter_components": profile["counter_components"],
            "tuning": profile["fit"],
        }
    )
    report["configuration"]["score_profile"] = profile["profile_id"]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=EXPERIMENT_ROOT / "results")
    parser.add_argument("--plots-root", type=Path, default=EXPERIMENT_ROOT / "plots")
    parser.add_argument("--output", type=Path, default=TAU_PROFILES)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    profiles = {
        platform: _fit_platform(args.results_root, platform)
        for platform in ("tuolumne", "matrix")
    }
    document = {
        "schema": "relay.triton.pilot_tau_profile",
        "version": 1,
        "graph_construction": "automatic_post_coalescing_manifest_universal_v1",
        "byte_scales": list(BYTE_SCALES),
        "platforms": profiles,
    }
    write_json(args.output, document)

    for platform, profile in profiles.items():
        for path in _reports(args.results_root, platform):
            report = json.loads(path.read_text(encoding="utf-8"))
            _apply_profile(report, profile)
            write_json(path, report)
            panel_path = path.parent / "profiles" / "panel.json"
            if panel_path.is_file():
                panel_record = json.loads(
                    panel_path.read_text(encoding="utf-8")
                )
                panel_record["configuration"] = report["configuration"]
                panel_record["panel"] = report["panel"]
                write_json(panel_path, panel_record)
            experiment = int(report["final_experiment"])
            plot = (
                args.plots_root
                / f"experiment-{experiment}"
                / platform
                / f"{report['case']}.pdf"
            )
            analyze_report(path, plot)
        for experiment in (1, 2, 3):
            analyze_suite(
                args.results_root,
                args.plots_root,
                experiment=experiment,
                platform=platform,
                regenerate_reports=False,
            )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
