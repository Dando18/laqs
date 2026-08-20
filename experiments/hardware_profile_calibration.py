"""Deterministically calibrate a sparse global tau from layout measurements.

The utility consumes a completed universal-scope layout-ranking report.  It
uses no numerical packages: feature columns are normalized and deduplicated,
then a seeded random/coordinate search optimizes the resulting five-cost
candidate frontiers.  The output is a recommendation for review, never an
in-place update of a checked-in hardware profile.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys


_REL_TOL = 1.0e-10
_ABS_TOL = 1.0e-12
_WEIGHT_REL_FLOOR = 1.0e-3


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 0
    iterations: int = 2000
    min_candidates: int = 5
    max_candidates: int = 7
    max_regret: float = 0.01
    search_support: int = 12

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("calibration iterations must be nonnegative")
        if self.min_candidates <= 0:
            raise ValueError("minimum candidate count must be positive")
        if self.max_candidates < self.min_candidates:
            raise ValueError(
                "maximum candidate count must be at least the minimum"
            )
        if not math.isfinite(self.max_regret) or self.max_regret < 0:
            raise ValueError("maximum regret must be finite and nonnegative")
        if self.search_support <= 0:
            raise ValueError("search support must be positive")


@dataclass(frozen=True)
class _RawLayout:
    name: str
    runtime_ms: float
    fine_region_count: float
    hardware_peak: float
    codegen_runs: float
    codegen_xors: float
    features: Mapping[str, float]

    @property
    def fixed_costs(self) -> tuple[float, float, float, float]:
        return (
            self.fine_region_count,
            self.hardware_peak,
            self.codegen_runs,
            self.codegen_xors,
        )


@dataclass(frozen=True)
class _RawGroup:
    kernel: str
    display_name: str
    matrix_size: int
    block: object
    fine_component: str
    layouts: tuple[_RawLayout, ...]

    @property
    def identity(self) -> tuple[str, int, str]:
        return (
            self.kernel,
            self.matrix_size,
            json.dumps(self.block, sort_keys=True, separators=(",", ":")),
        )


@dataclass(frozen=True)
class _RawCorpus:
    groups: tuple[_RawGroup, ...]
    profile: Mapping[str, object]
    source_tau: Mapping[str, float]


@dataclass(frozen=True)
class FeatureColumn:
    """One normalized feature and all globally proportional source cells."""

    name: str
    scale: float
    aliases: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class _Layout:
    name: str
    runtime_ms: float
    fixed_costs: tuple[float, float, float, float]
    features: tuple[float, ...]


@dataclass(frozen=True)
class _Group:
    kernel: str
    display_name: str
    matrix_size: int
    block: object
    fine_component: str
    layouts: tuple[_Layout, ...]
    possible_dominators: tuple[tuple[tuple[int, bool], ...], ...]


@dataclass(frozen=True)
class PreparedCorpus:
    groups: tuple[_Group, ...]
    columns: tuple[FeatureColumn, ...]
    source_weights: tuple[float, ...] | None
    input_feature_count: int
    informative_feature_count: int
    dropped_groupwise_constant: tuple[str, ...]
    profile: Mapping[str, object]


@dataclass(frozen=True)
class InstanceMetrics:
    kernel: str
    display_name: str
    matrix_size: int
    block: object
    layout_count: int
    candidate_count: int
    frontier_names: tuple[str, ...]
    optimal_runtime_ms: float
    best_frontier_runtime_ms: float
    best_frontier_names: tuple[str, ...]
    oracle_regret: float


@dataclass(frozen=True)
class Evaluation:
    instances: tuple[InstanceMetrics, ...]


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _number(
    value: object,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{path} must be positive")
    if nonnegative and result < -_ABS_TOL:
        raise ValueError(f"{path} must be nonnegative")
    return max(0.0, result) if nonnegative else result


def _profile_for_group(
    group: Mapping[str, object], path: str
) -> Mapping[str, object]:
    if group.get("hardware_profile") is None:
        raise ValueError(
            f"{path} has no universal hardware profile; rescore legacy timings "
            "with experiments/layout_ranking.py --reuse-timings"
        )
    profile = _mapping(group.get("hardware_profile"), f"{path}.hardware_profile")
    if not isinstance(profile.get("profile_id"), str):
        raise ValueError(f"{path}.hardware_profile.profile_id must be a string")
    if not isinstance(profile.get("fine_component"), str):
        raise ValueError(f"{path}.hardware_profile.fine_component must be a string")
    scales = _sequence(
        profile.get("byte_scales"), f"{path}.hardware_profile.byte_scales"
    )
    if not scales or any(
        isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0
        for scale in scales
    ):
        raise ValueError(
            f"{path}.hardware_profile.byte_scales must be positive integers"
        )
    return profile


def _objective_schema(
    group: Mapping[str, object], path: str
) -> dict[str, Mapping[str, object]]:
    objectives = _sequence(group.get("objectives"), f"{path}.objectives")
    if not objectives:
        raise ValueError(f"{path} has no universal objectives")
    result: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(objectives):
        objective = _mapping(item, f"{path}.objectives[{index}]")
        name = objective.get("name")
        family = objective.get("edge_family")
        region = objective.get("region_bytes")
        if not isinstance(name, str) or not isinstance(family, str):
            raise ValueError(
                f"{path}.objectives[{index}] is not a universal objective"
            )
        if objective.get("provenance") != "universal-v1":
            raise ValueError(
                f"{path}.objectives[{index}] is not from universal-v1; "
                "rescore legacy timings with experiments/layout_ranking.py "
                "--reuse-timings"
            )
        if isinstance(region, bool) or not isinstance(region, int) or region <= 0:
            raise ValueError(f"{path}.objectives[{index}].region_bytes is invalid")
        if name != f"{family}.{region}B":
            raise ValueError(
                f"{path}.objectives[{index}] has an incoherent scope-scale name"
            )
        _number(
            objective.get("normalization_bytes"),
            f"{path}.objectives[{index}].normalization_bytes",
            positive=True,
        )
        if name in result:
            raise ValueError(f"{path} has duplicate objective {name!r}")
        result[name] = objective
    normalizations = [
        float(objective["normalization_bytes"])
        for objective in result.values()
    ]
    if any(
        not _close(value, normalizations[0]) for value in normalizations[1:]
    ):
        raise ValueError(
            f"{path} objectives do not share one kernel exposure denominator"
        )
    return result


def _component_feature(
    component: Mapping[str, object],
    objective: Mapping[str, object],
    path: str,
) -> float:
    denominator = _number(
        objective.get("normalization_bytes"),
        f"{path}.objective.normalization_bytes",
        positive=True,
    )
    region_bytes = _number(
        objective.get("region_bytes"), f"{path}.objective.region_bytes", positive=True
    )
    component_region = component.get("region_bytes")
    if component_region is not None and component_region != objective["region_bytes"]:
        raise ValueError(f"{path}.region_bytes disagrees with its objective")
    component_normalization = component.get("normalization_bytes")
    if component_normalization is not None and not _close(
        _number(
            component_normalization,
            f"{path}.normalization_bytes",
            positive=True,
        ),
        denominator,
    ):
        raise ValueError(f"{path}.normalization_bytes disagrees with its objective")
    raw = _number(
        component.get("raw_region_count"),
        f"{path}.raw_region_count",
        nonnegative=True,
    )
    bound = _number(
        component.get("packing_lower_bound"),
        f"{path}.packing_lower_bound",
        nonnegative=True,
    )
    derived = _number(
        region_bytes * (raw - bound) / denominator,
        f"{path}.derived_excess_footprint",
        nonnegative=True,
    )
    if component.get("excess_footprint") is None:
        return derived
    stored = _number(
        component["excess_footprint"],
        f"{path}.excess_footprint",
        nonnegative=True,
    )
    if not _close(stored, derived):
        raise ValueError(f"{path}.excess_footprint disagrees with Q/LB metadata")
    return stored


def _hardware_peak(
    score: Mapping[str, object],
    components: Mapping[str, Mapping[str, object]],
    tolerances: Mapping[str, object],
    path: str,
) -> float:
    aggregates = _mapping(score.get("aggregates"), f"{path}.aggregates")
    if aggregates.get("hardware_peak") is not None:
        return _number(
            aggregates["hardware_peak"],
            f"{path}.aggregates.hardware_peak",
            nonnegative=True,
        )
    ratios = []
    for name, tolerance_value in tolerances.items():
        component = components.get(name)
        if component is None:
            continue
        tolerance = _number(
            tolerance_value,
            f"{path}.peak_tolerances[{name!r}]",
            positive=True,
        )
        excess = _number(
            component.get("normalized_excess"),
            f"{path}.components[{name!r}].normalized_excess",
            nonnegative=True,
        )
        ratios.append(excess / tolerance)
    if not ratios:
        raise ValueError(f"{path} has no hardware_peak or active peak tolerances")
    return max(ratios)


def _parse_group(
    value: object, index: int
) -> tuple[_RawGroup, Mapping[str, object], Mapping[str, float]]:
    path = f"runs[{index}]"
    group = _mapping(value, path)
    profile = _profile_for_group(group, path)
    objectives = _objective_schema(group, path)
    byte_scales = set(profile["byte_scales"])
    if any(
        objective["region_bytes"] not in byte_scales
        for objective in objectives.values()
    ):
        raise ValueError(
            f"{path} contains an objective outside the profile byte ladder"
        )
    kernel = group.get("kernel")
    display_name = group.get("display_name", kernel)
    matrix_size = group.get("matrix_size")
    if not isinstance(kernel, str) or not kernel:
        raise ValueError(f"{path}.kernel must be a nonempty string")
    if not isinstance(display_name, str):
        raise ValueError(f"{path}.display_name must be a string")
    if (
        isinstance(matrix_size, bool)
        or not isinstance(matrix_size, int)
        or matrix_size <= 0
    ):
        raise ValueError(f"{path}.matrix_size must be a positive integer")
    fine_component = group.get("fine_component", profile["fine_component"])
    if fine_component != profile["fine_component"]:
        raise ValueError(f"{path} disagrees with its hardware-profile fine component")
    if fine_component not in objectives:
        raise ValueError(f"{path} has no fine objective {fine_component!r}")

    tolerances = _mapping(
        group.get("peak_tolerances", profile.get("kappa", {})),
        f"{path}.peak_tolerances",
    )
    records = _sequence(group.get("results"), f"{path}.results")
    if not records:
        raise ValueError(f"{path} has no layout results")

    layouts: list[_RawLayout] = []
    expected_components: set[str] | None = None
    names: set[str] = set()
    for record_index, item in enumerate(records):
        record_path = f"{path}.results[{record_index}]"
        record = _mapping(item, record_path)
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{record_path}.name must be a nonempty string")
        if name in names:
            raise ValueError(f"{path} has duplicate layout name {name!r}")
        names.add(name)
        timing = _mapping(record.get("timing"), f"{record_path}.timing")
        runtime = _number(
            timing.get("median_ms"),
            f"{record_path}.timing.median_ms",
            positive=True,
        )
        score = _mapping(record.get("score"), f"{record_path}.score")
        component_items = _sequence(
            score.get("components"), f"{record_path}.score.components"
        )
        components: dict[str, Mapping[str, object]] = {}
        features: dict[str, float] = {}
        for component_index, component_item in enumerate(component_items):
            component_path = (
                f"{record_path}.score.components[{component_index}]"
            )
            component = _mapping(component_item, component_path)
            component_name = component.get("name")
            if not isinstance(component_name, str) or component_name not in objectives:
                raise ValueError(
                    f"{component_path} does not name a declared universal objective"
                )
            if component_name in components:
                raise ValueError(
                    f"{record_path} has duplicate component {component_name!r}"
                )
            components[component_name] = component
            features[component_name] = _component_feature(
                component, objectives[component_name], component_path
            )
        component_names = set(components)
        if expected_components is None:
            expected_components = component_names
        elif component_names != expected_components:
            raise ValueError(
                f"{path} has inconsistent component sets across layouts"
            )
        if component_names != set(objectives):
            raise ValueError(
                f"{record_path} does not contain every declared objective component"
            )
        fine = _number(
            components[str(fine_component)].get("raw_region_count"),
            f"{record_path}.fine_region_count",
            nonnegative=True,
        )
        peak = _hardware_peak(score, components, tolerances, record_path)
        codegen = _mapping(score.get("codegen"), f"{record_path}.score.codegen")
        runs = _number(
            codegen.get("runs"),
            f"{record_path}.score.codegen.runs",
            nonnegative=True,
        )
        xors = _number(
            codegen.get("xors"),
            f"{record_path}.score.codegen.xors",
            nonnegative=True,
        )
        layouts.append(
            _RawLayout(name, runtime, fine, peak, runs, xors, features)
        )

    source_tau_values = _mapping(profile.get("tau", {}), f"{path}.profile.tau")
    source_tau = {
        name: _number(value, f"{path}.profile.tau[{name!r}]", nonnegative=True)
        for name, value in source_tau_values.items()
    }
    return (
        _RawGroup(
            kernel=kernel,
            display_name=display_name,
            matrix_size=matrix_size,
            block=group.get("block"),
            fine_component=str(fine_component),
            layouts=tuple(sorted(layouts, key=lambda layout: layout.name)),
        ),
        profile,
        source_tau,
    )


def _parse_report(
    report: Mapping[str, object],
    *,
    kernels: Sequence[str] = (),
    sizes: Sequence[int] = (),
) -> _RawCorpus:
    if report.get("experiment") != "multi-kernel-layout-ranking":
        raise ValueError("input is not a multi-kernel layout-ranking report")
    if report.get("complete") is not True:
        raise ValueError("calibration requires a completed layout-ranking report")
    runs = _sequence(report.get("runs"), "runs")
    kernel_filter = frozenset(kernels)
    size_filter = frozenset(sizes)
    parsed = [_parse_group(value, index) for index, value in enumerate(runs)]
    selected = [
        item
        for item in parsed
        if (not kernel_filter or item[0].kernel in kernel_filter)
        and (not size_filter or item[0].matrix_size in size_filter)
    ]
    if not selected:
        raise ValueError("kernel/size filters selected no calibration groups")

    unknown_kernels = kernel_filter - {item[0].kernel for item in parsed}
    unknown_sizes = size_filter - {item[0].matrix_size for item in parsed}
    if unknown_kernels:
        raise ValueError(
            "unknown kernel filters: " + ", ".join(sorted(unknown_kernels))
        )
    if unknown_sizes:
        raise ValueError(
            "unknown size filters: " + ", ".join(map(str, sorted(unknown_sizes)))
        )

    serialized_profiles = {
        json.dumps(item[1], sort_keys=True, separators=(",", ":"))
        for item in selected
    }
    if len(serialized_profiles) != 1:
        raise ValueError("selected groups do not share one hardware profile")

    groups = tuple(
        sorted((item[0] for item in selected), key=lambda group: group.identity)
    )
    identities = [group.identity for group in groups]
    if len(set(identities)) != len(identities):
        raise ValueError("selected report contains duplicate kernel/size/block groups")
    profile = dict(selected[0][1])
    source_tau = dict(selected[0][2])
    return _RawCorpus(groups, profile, source_tau)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _possible_dominators(
    layouts: Sequence[_Layout],
) -> tuple[tuple[tuple[int, bool], ...], ...]:
    result = []
    for index, layout in enumerate(layouts):
        candidates = []
        for other_index, other in enumerate(layouts):
            if index == other_index:
                continue
            if all(
                left <= right
                for left, right in zip(other.fixed_costs, layout.fixed_costs)
            ):
                candidates.append(
                    (
                        other_index,
                        any(
                            left < right
                            for left, right in zip(
                                other.fixed_costs, layout.fixed_costs
                            )
                        ),
                    )
                )
        result.append(tuple(candidates))
    return tuple(result)


def prepare_corpus(
    report: Mapping[str, object],
    *,
    kernels: Sequence[str] = (),
    sizes: Sequence[int] = (),
) -> PreparedCorpus:
    """Validate a report and construct its normalized global feature basis."""

    raw = _parse_report(report, kernels=kernels, sizes=sizes)
    feature_names = sorted(
        {
            name
            for group in raw.groups
            for layout in group.layouts
            for name in layout.features
        }
    )
    rows = [layout for group in raw.groups for layout in group.layouts]
    raw_columns = {
        name: tuple(layout.features.get(name, 0.0) for layout in rows)
        for name in feature_names
    }

    informative = []
    dropped = []
    row_start = 0
    group_ranges = []
    for group in raw.groups:
        group_ranges.append(range(row_start, row_start + len(group.layouts)))
        row_start += len(group.layouts)
    for name in feature_names:
        column = raw_columns[name]
        varies = any(
            any(not _close(column[index], column[indices.start]) for index in indices)
            for indices in group_ranges
        )
        if varies:
            informative.append(name)
        else:
            dropped.append(name)
    if not informative:
        raise ValueError(
            "selected groups have no layout-varying excess-footprint features"
        )

    scales = {name: max(raw_columns[name]) for name in informative}
    normalized = {
        name: tuple(value / scales[name] for value in raw_columns[name])
        for name in informative
    }
    classes: list[list[str]] = []
    signatures: dict[tuple[float, ...], list[int]] = {}
    for name in informative:
        vector = normalized[name]
        signature = tuple(round(value, 11) for value in vector)
        matching = None
        for class_index in signatures.get(signature, []):
            representative = classes[class_index][0]
            if all(
                _close(left, right)
                for left, right in zip(vector, normalized[representative])
            ):
                matching = class_index
                break
        if matching is None:
            matching = len(classes)
            classes.append([name])
            signatures.setdefault(signature, []).append(matching)
        else:
            classes[matching].append(name)

    columns = tuple(
        FeatureColumn(
            name=members[0],
            scale=scales[members[0]],
            aliases=tuple(
                (name, scales[name] / scales[members[0]]) for name in members
            ),
        )
        for members in classes
    )
    representative_index = {
        alias: index
        for index, column in enumerate(columns)
        for alias, _ in column.aliases
    }
    prepared_groups = []
    row_index = 0
    for raw_group in raw.groups:
        layouts = []
        for raw_layout in raw_group.layouts:
            values = tuple(
                normalized[column.name][row_index] for column in columns
            )
            layouts.append(
                _Layout(
                    raw_layout.name,
                    raw_layout.runtime_ms,
                    raw_layout.fixed_costs,
                    values,
                )
            )
            row_index += 1
        layout_tuple = tuple(layouts)
        prepared_groups.append(
            _Group(
                raw_group.kernel,
                raw_group.display_name,
                raw_group.matrix_size,
                raw_group.block,
                raw_group.fine_component,
                layout_tuple,
                _possible_dominators(layout_tuple),
            )
        )

    initial = [0.0] * len(columns)
    for name, tau in raw.source_tau.items():
        index = representative_index.get(name)
        if index is not None:
            initial[index] += tau * scales[name]
    source_weights = _normalize_weights(initial) if any(initial) else None
    return PreparedCorpus(
        groups=tuple(prepared_groups),
        columns=columns,
        source_weights=source_weights,
        input_feature_count=len(feature_names),
        informative_feature_count=len(informative),
        dropped_groupwise_constant=tuple(dropped),
        profile=raw.profile,
    )


def _normalize_weights(values: Sequence[float]) -> tuple[float, ...]:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("tau candidates must be finite and nonnegative")
    maximum = max(values, default=0.0)
    if maximum <= 0:
        raise ValueError("a tau candidate must contain a positive finite weight")
    cleaned = tuple(
        0.0 if value <= maximum * _WEIGHT_REL_FLOOR else value
        for value in values
    )
    total = sum(cleaned)
    return tuple(0.0 if value == 0 else value / total for value in cleaned)


def evaluate_weights(
    corpus: PreparedCorpus, weights: Sequence[float]
) -> Evaluation:
    """Evaluate a normalized tau candidate on every selected instance."""

    if len(weights) != len(corpus.columns):
        raise ValueError("tau candidate dimension does not match the feature basis")
    normalized_weights = _normalize_weights(weights)
    active = [
        (index, weight)
        for index, weight in enumerate(normalized_weights)
        if weight > 0
    ]
    instances = []
    for group in corpus.groups:
        areas = [
            sum(weight * layout.features[index] for index, weight in active)
            for layout in group.layouts
        ]
        frontier_indices = []
        for index, possible in enumerate(group.possible_dominators):
            dominated = any(
                areas[other] <= areas[index]
                and (fixed_strict or areas[other] < areas[index])
                for other, fixed_strict in possible
            )
            if not dominated:
                frontier_indices.append(index)
        optimum = min(layout.runtime_ms for layout in group.layouts)
        frontier_runtime = min(
            group.layouts[index].runtime_ms for index in frontier_indices
        )
        instances.append(
            InstanceMetrics(
                kernel=group.kernel,
                display_name=group.display_name,
                matrix_size=group.matrix_size,
                block=group.block,
                layout_count=len(group.layouts),
                candidate_count=len(frontier_indices),
                frontier_names=tuple(
                    group.layouts[index].name for index in frontier_indices
                ),
                optimal_runtime_ms=optimum,
                best_frontier_runtime_ms=frontier_runtime,
                best_frontier_names=tuple(
                    group.layouts[index].name
                    for index in frontier_indices
                    if group.layouts[index].runtime_ms == frontier_runtime
                ),
                oracle_regret=max(0.0, frontier_runtime / optimum - 1.0),
            )
        )
    return Evaluation(tuple(instances))


def _fitness(
    evaluation: Evaluation, config: CalibrationConfig
) -> tuple[float, ...]:
    regrets = [instance.oracle_regret for instance in evaluation.instances]
    regret_excesses = [max(0.0, regret - config.max_regret) for regret in regrets]
    count_distances = [
        max(
            config.min_candidates - instance.candidate_count,
            instance.candidate_count - config.max_candidates,
            0,
        )
        for instance in evaluation.instances
    ]
    return (
        float(sum(excess > _ABS_TOL for excess in regret_excesses)),
        max(regret_excesses, default=0.0),
        sum(regret_excesses),
        float(sum(distance > 0 for distance in count_distances)),
        float(sum(count_distances)),
        float(sum(instance.candidate_count for instance in evaluation.instances)),
        max(regrets, default=0.0),
        sum(regrets),
    )


def _constraints_met(evaluation: Evaluation, config: CalibrationConfig) -> bool:
    return all(
        config.min_candidates <= instance.candidate_count <= config.max_candidates
        and instance.oracle_regret <= config.max_regret + _ABS_TOL
        for instance in evaluation.instances
    )


def _support(weights: Sequence[float]) -> int:
    return sum(weight > 0 for weight in weights)


def _candidate_key(
    evaluation: Evaluation,
    weights: Sequence[float],
    config: CalibrationConfig,
) -> tuple[object, ...]:
    return (
        _fitness(evaluation, config),
        _support(weights),
        tuple(round(weight, 15) for weight in weights),
    )


def _random_weights(
    generator: random.Random, dimensions: int, support_limit: int
) -> tuple[float, ...]:
    support = generator.randint(1, min(dimensions, support_limit))
    indices = generator.sample(range(dimensions), support)
    values = [0.0] * dimensions
    for index in indices:
        values[index] = 10.0 ** generator.uniform(-2.0, 2.0)
    return _normalize_weights(values)


def _mutate_weights(
    generator: random.Random,
    weights: Sequence[float],
    support_limit: int,
) -> tuple[float, ...]:
    values = list(weights)
    active = [index for index, value in enumerate(values) if value > 0]
    inactive = [index for index, value in enumerate(values) if value == 0]
    operation = generator.randrange(4)
    if operation == 0 and inactive and len(active) < support_limit:
        values[generator.choice(inactive)] = 10.0 ** generator.uniform(-2.0, 2.0)
    elif operation == 1 and len(active) > 1:
        values[generator.choice(active)] = 0.0
    elif operation == 2 and active:
        index = generator.choice(active)
        values[index] *= 10.0 ** generator.uniform(-1.0, 1.0)
    elif inactive:
        values[generator.choice(active)] = 0.0
        values[generator.choice(inactive)] = 10.0 ** generator.uniform(-2.0, 2.0)
    else:
        index = generator.choice(active)
        values[index] *= 10.0 ** generator.uniform(-1.0, 1.0)
    return _normalize_weights(values)


def _search(
    corpus: PreparedCorpus, config: CalibrationConfig
) -> tuple[tuple[float, ...], Evaluation, int]:
    dimensions = len(corpus.columns)
    support_limit = min(dimensions, config.search_support)
    seeds = []
    if dimensions <= support_limit:
        seeds.append(tuple(1.0 / dimensions for _ in range(dimensions)))
    seeds.extend(
        tuple(1.0 if index == active else 0.0 for index in range(dimensions))
        for active in range(dimensions)
    )
    if corpus.source_weights is not None:
        source_indices = sorted(
            range(dimensions),
            key=lambda index: (-corpus.source_weights[index], index),
        )[:support_limit]
        source_seed = [0.0] * dimensions
        for index in source_indices:
            source_seed[index] = corpus.source_weights[index]
        if any(source_seed):
            seeds.append(_normalize_weights(source_seed))

    seen: set[tuple[float, ...]] = set()
    elites: list[tuple[tuple[object, ...], tuple[float, ...], Evaluation]] = []

    def consider(weights: tuple[float, ...]) -> None:
        signature = tuple(round(weight, 14) for weight in weights)
        if signature in seen:
            return
        seen.add(signature)
        evaluation = evaluate_weights(corpus, weights)
        item = (_candidate_key(evaluation, weights, config), weights, evaluation)
        elites.append(item)
        elites.sort(key=lambda entry: entry[0])
        del elites[8:]

    for weights in seeds:
        consider(weights)
    generator = random.Random(config.seed)
    for _ in range(config.iterations):
        if elites and generator.random() < 0.65:
            parent = generator.choice(elites)[1]
            weights = _mutate_weights(generator, parent, support_limit)
        else:
            weights = _random_weights(generator, dimensions, support_limit)
        consider(weights)
    _, best_weights, best_evaluation = min(elites, key=lambda entry: entry[0])
    return best_weights, best_evaluation, len(seen)


def _sparsify(
    corpus: PreparedCorpus,
    weights: tuple[float, ...],
    evaluation: Evaluation,
    config: CalibrationConfig,
) -> tuple[tuple[float, ...], Evaluation, tuple[str, ...]]:
    removed = []
    while _support(weights) > 1:
        candidates = []
        currently_feasible = _constraints_met(evaluation, config)
        for index, value in enumerate(weights):
            if value == 0:
                continue
            trial_values = list(weights)
            trial_values[index] = 0.0
            trial_weights = _normalize_weights(trial_values)
            trial_evaluation = evaluate_weights(corpus, trial_weights)
            if currently_feasible:
                acceptable = _constraints_met(trial_evaluation, config)
            else:
                acceptable = _fitness(trial_evaluation, config) <= _fitness(
                    evaluation, config
                )
            if acceptable:
                candidates.append(
                    (
                        _candidate_key(trial_evaluation, trial_weights, config),
                        corpus.columns[index].name,
                        trial_weights,
                        trial_evaluation,
                    )
                )
        if not candidates:
            break
        _, name, weights, evaluation = min(candidates, key=lambda item: item[:2])
        removed.append(name)
    return weights, evaluation, tuple(removed)


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _evaluation_dict(
    evaluation: Evaluation, config: CalibrationConfig
) -> dict[str, object]:
    counts = [float(instance.candidate_count) for instance in evaluation.instances]
    regrets = [instance.oracle_regret for instance in evaluation.instances]
    return {
        "constraints_met": _constraints_met(evaluation, config),
        "candidate_count": _metric_summary(counts),
        "oracle_regret": _metric_summary(regrets),
        "exact_winner_covered_instances": sum(regret == 0.0 for regret in regrets),
        "instances": [
            {
                "kernel": instance.kernel,
                "display_name": instance.display_name,
                "matrix_size": instance.matrix_size,
                "block": instance.block,
                "layout_count": instance.layout_count,
                "candidate_count": instance.candidate_count,
                "retained_fraction": (
                    instance.candidate_count / instance.layout_count
                ),
                "frontier_names": list(instance.frontier_names),
                "optimal_runtime_ms": instance.optimal_runtime_ms,
                "best_frontier_runtime_ms": instance.best_frontier_runtime_ms,
                "best_frontier_names": list(instance.best_frontier_names),
                "oracle_regret": instance.oracle_regret,
            }
            for instance in evaluation.instances
        ],
    }


def _basis_diagnostic(corpus: PreparedCorpus) -> dict[str, object]:
    """Test whether a measured winner survives the uncompressed full basis."""

    instances = []
    for group in corpus.groups:
        vectors = [(*layout.fixed_costs, *layout.features) for layout in group.layouts]
        frontier_indices = []
        dominators: dict[int, list[int]] = {}
        for index, vector in enumerate(vectors):
            better = [
                other_index
                for other_index, other in enumerate(vectors)
                if other_index != index
                and all(left <= right for left, right in zip(other, vector))
                and any(left < right for left, right in zip(other, vector))
            ]
            if better:
                dominators[index] = better
            else:
                frontier_indices.append(index)

        optimum = min(layout.runtime_ms for layout in group.layouts)
        winner_indices = [
            index
            for index, layout in enumerate(group.layouts)
            if layout.runtime_ms == optimum
        ]
        frontier_set = set(frontier_indices)
        covered = any(index in frontier_set for index in winner_indices)
        instances.append(
            {
                "kernel": group.kernel,
                "display_name": group.display_name,
                "matrix_size": group.matrix_size,
                "block": group.block,
                "objective_count": 4 + len(corpus.columns),
                "frontier_size": len(frontier_indices),
                "frontier_names": [
                    group.layouts[index].name for index in frontier_indices
                ],
                "measured_winners": [
                    group.layouts[index].name for index in winner_indices
                ],
                "winner_covered": covered,
                "dominated_winner_certificates": [
                    {
                        "winner": group.layouts[index].name,
                        "dominator": group.layouts[dominators[index][0]].name,
                    }
                    for index in winner_indices
                    if index in dominators
                ],
            }
        )
    covered_count = sum(bool(instance["winner_covered"]) for instance in instances)
    return {
        "definition": (
            "Pareto frontier over fine Q, fixed hardware peak, every independent "
            "normalized excess-footprint column, codegen runs, and codegen XORs"
        ),
        "covered_instances": covered_count,
        "instance_count": len(instances),
        "all_instances_covered": covered_count == len(instances),
        "instances": instances,
    }


def _recommendation(
    corpus: PreparedCorpus, weights: Sequence[float]
) -> tuple[dict[str, float], dict[str, float]]:
    maximum = max(weights)
    normalized = {
        column.name: weight / maximum
        for column, weight in zip(corpus.columns, weights)
        if weight > 0
    }
    tau = {}
    for column in corpus.columns:
        response = normalized.get(column.name)
        if response is None:
            continue
        share = response / len(column.aliases)
        for name, relative_scale in column.aliases:
            tau[name] = share / (column.scale * relative_scale)
    return dict(sorted(tau.items())), dict(sorted(normalized.items()))


def calibrate_report(
    report: Mapping[str, object],
    *,
    kernels: Sequence[str] = (),
    sizes: Sequence[int] = (),
    config: CalibrationConfig = CalibrationConfig(),
) -> dict[str, object]:
    """Search and sparsify one global tau over selected report groups."""

    corpus = prepare_corpus(report, kernels=kernels, sizes=sizes)
    searched_weights, searched_evaluation, evaluated = _search(corpus, config)
    weights, evaluation, removed = _sparsify(
        corpus, searched_weights, searched_evaluation, config
    )
    tau, normalized = _recommendation(corpus, weights)
    basis_diagnostic = _basis_diagnostic(corpus)
    constraints_met = _constraints_met(evaluation, config)
    passes_validation = bool(
        constraints_met and basis_diagnostic["all_instances_covered"]
    )
    warnings = []
    if not constraints_met:
        warnings.append(
            "No searched and sparsified tau met every candidate-count/regret target; "
            "the recommendation is best effort."
        )
    if not basis_diagnostic["all_instances_covered"]:
        warnings.append(
            "At least one measured winner is dominated in the full selected basis; "
            "nonnegative tau calibration alone cannot recover it."
        )
    fine_components = sorted({group.fine_component for group in corpus.groups})
    return {
        "experiment": "hardware-profile-tau-calibration",
        "method": (
            "normalize excess-footprint columns by their selected-corpus maxima, "
            "merge globally proportional columns, search nonnegative L1-normalized "
            "responses, and greedily remove support while preserving constraints"
        ),
        "source_profile": {
            "profile_id": corpus.profile["profile_id"],
            "device": corpus.profile.get("device"),
            "byte_scales": corpus.profile["byte_scales"],
            "fine_components": fine_components,
        },
        "configuration": {
            "kernels": sorted(set(kernels)) or None,
            "matrix_sizes": sorted(set(sizes)) or None,
            "seed": config.seed,
            "iterations": config.iterations,
            "min_candidates": config.min_candidates,
            "max_candidates": config.max_candidates,
            "max_regret": config.max_regret,
            "search_support": config.search_support,
        },
        "corpus": {
            "instance_count": len(corpus.groups),
            "layout_count": sum(len(group.layouts) for group in corpus.groups),
            "input_feature_count": corpus.input_feature_count,
            "informative_feature_count": corpus.informative_feature_count,
            "deduplicated_feature_count": len(corpus.columns),
            "dropped_groupwise_constant": list(
                corpus.dropped_groupwise_constant
            ),
            "normalized_columns": [
                {
                    "name": column.name,
                    "scale": column.scale,
                    "proportional_cells": [
                        {"name": name, "relative_scale": relative_scale}
                        for name, relative_scale in column.aliases
                    ],
                }
                for column in corpus.columns
            ],
        },
        "search": {
            "evaluated_weight_vectors": evaluated,
            "best_before_sparsification": _evaluation_dict(
                searched_evaluation, config
            ),
            "support_before_sparsification": _support(searched_weights),
            "greedily_removed": list(removed),
        },
        "full_basis_diagnostic": basis_diagnostic,
        "recommendation": {
            "status": "passing" if passes_validation else "best-effort",
            "passes_validation": passes_validation,
            "tau": tau,
            "normalized_feature_response": normalized,
            "independent_support_size": _support(weights),
            "cell_support_size": len(tau),
            "normalization": (
                "max normalized-feature response is one; tau[name] equals "
                "an equal share of that response divided by the cell scale. "
                "Any common positive rescaling has the same Pareto frontier."
            ),
            "proportional_cell_policy": (
                "Responses are split equally across empirically proportional "
                "cells so every cell remains supported on held-out kernels."
            ),
            "kappa": corpus.profile.get("kappa", {}),
            "kappa_note": "unchanged from source profile",
        },
        "metrics": _evaluation_dict(evaluation, config),
        "warnings": warnings,
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a sparse global hardware tau from a completed universal "
            "layout-ranking JSON report."
        )
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON here (default: stdout)")
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--size", type=int, action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=7)
    parser.add_argument(
        "--max-regret",
        type=float,
        default=0.01,
        help="maximum fractional oracle regret per instance (default: %(default)s)",
    )
    parser.add_argument("--search-support", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        report = json.loads(args.report.read_text())
        if not isinstance(report, Mapping):
            raise ValueError("input report must contain a JSON object")
        result = calibrate_report(
            report,
            kernels=args.kernel,
            sizes=args.size,
            config=CalibrationConfig(
                seed=args.seed,
                iterations=args.iterations,
                min_candidates=args.min_candidates,
                max_candidates=args.max_candidates,
                max_regret=args.max_regret,
                search_support=args.search_support,
            ),
        )
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            args.output.write_text(text)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
