"""Runtime-facing evaluation of RELAY candidate frontiers.

The analytical frontier is a candidate set, not a total runtime ordering.
This module evaluates that candidate set through best-in-frontier regret,
epsilon-optimal coverage, retained fraction, purity, enrichment, and a
budgeted top-k scalar-score diagnostic.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Mapping, Sequence


DEFAULT_EPSILONS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)
TAU_PERTURBATION_FACTORS = (0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)
DEFAULT_TAU_PERTURBATION_TRIALS = 128


def _runtime(record: Mapping[str, object]) -> float:
    timing = record["timing"]
    assert isinstance(timing, dict)
    return float(timing["median_ms"])


def _score_components(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    score = record["score"]
    assert isinstance(score, dict)
    components = score["components"]
    assert isinstance(components, list)
    return components


def _base_score_vector(
    record: Mapping[str, object], *, include_fine: bool
) -> tuple[float, ...]:
    fine = next(
        float(component["raw_region_count"])
        for component in _score_components(record)
        if component["name"] == "wave_load.64B"
    )
    score = record["score"]
    assert isinstance(score, dict)
    aggregates = score["aggregates"]
    codegen = score["codegen"]
    assert isinstance(aggregates, dict) and isinstance(codegen, dict)
    vector = (
        float(aggregates["peak_normalized_excess"]),
        float(aggregates["weighted_normalized_excess"]),
        float(codegen["runs"]),
        float(codegen["xors"]),
    )
    return (fine, *vector) if include_fine else vector


def _pareto_names(
    named_vectors: Sequence[tuple[str, tuple[float, ...]]],
) -> set[str]:
    """Return names with non-dominated vectors, retaining exact ties."""

    vectors = {vector for _, vector in named_vectors}
    non_dominated = {
        vector
        for vector in vectors
        if not any(
            other != vector
            and all(left <= right for left, right in zip(other, vector))
            and any(left < right for left, right in zip(other, vector))
            for other in vectors
        )
    }
    return {name for name, vector in named_vectors if vector in non_dominated}


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("a metric summary requires at least one value")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _choose(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def random_subset_coverage_probability(
    layout_count: int,
    optimal_layout_count: int,
    retained_count: int,
) -> float:
    """Return the exact chance that a size-matched random subset hits once."""

    if not 0 < optimal_layout_count <= layout_count:
        raise ValueError("optimal-layout count must be between one and N")
    if not 0 < retained_count <= layout_count:
        raise ValueError("retained count must be between one and N")
    misses = _choose(layout_count - optimal_layout_count, retained_count)
    return 1.0 - misses / _choose(layout_count, retained_count)


def poisson_binomial_upper_tail(
    probabilities: Sequence[float], observed_hits: int
) -> float:
    """Return ``Pr[X >= observed_hits]`` for independent Bernoulli trials."""

    if not 0 <= observed_hits <= len(probabilities):
        raise ValueError("observed hits must lie between zero and trial count")
    distribution = [1.0]
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Bernoulli probabilities must lie in [0, 1]")
        updated = [0.0] * (len(distribution) + 1)
        for hits, mass in enumerate(distribution):
            updated[hits] += mass * (1.0 - probability)
            updated[hits + 1] += mass * probability
        distribution = updated
    return sum(distribution[observed_hits:])


def analyze_frontier_group(
    group: Mapping[str, object],
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
) -> dict[str, object]:
    """Evaluate one completed kernel/size frontier using median runtime."""

    records = group.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError("frontier analysis requires a nonempty result list")
    if any(
        not isinstance(record, dict) or record.get("timing") is None
        for record in records
    ):
        raise ValueError("frontier analysis requires complete timing records")

    frontier = [record for record in records if record["pareto_frontier_member"]]
    if not frontier:
        raise ValueError("frontier analysis requires a nonempty Pareto frontier")

    def runtime(record: Mapping[str, object]) -> float:
        timing = record["timing"]
        assert isinstance(timing, dict)
        return float(timing["median_ms"])

    layout_count = len(records)
    frontier_size = len(frontier)
    optimum = min(runtime(record) for record in records)
    frontier_optimum = min(runtime(record) for record in frontier)
    optimal_layouts = [
        str(record["name"]) for record in records if runtime(record) == optimum
    ]
    best_frontier_layouts = [
        str(record["name"])
        for record in frontier
        if runtime(record) == frontier_optimum
    ]
    oracle_regret = frontier_optimum / optimum - 1.0

    epsilon_metrics = []
    for epsilon in epsilons:
        if epsilon < 0:
            raise ValueError("epsilon values must be nonnegative")
        optimal = [
            record
            for record in records
            if runtime(record) <= (1.0 + epsilon) * optimum
        ]
        frontier_optimal = [
            record for record in optimal if record["pareto_frontier_member"]
        ]
        purity = len(frontier_optimal) / frontier_size
        prevalence = len(optimal) / layout_count
        epsilon_metrics.append(
            {
                "epsilon": float(epsilon),
                "epsilon_percent": 100.0 * float(epsilon),
                "optimal_layout_count": len(optimal),
                "frontier_optimal_layout_count": len(frontier_optimal),
                "covered": bool(frontier_optimal),
                "purity": purity,
                "search_space_prevalence": prevalence,
                "enrichment": purity / prevalence,
                "random_size_matched_coverage_probability": (
                    random_subset_coverage_probability(
                        layout_count,
                        len(optimal),
                        frontier_size,
                    )
                ),
            }
        )

    ordered = sorted(
        records,
        key=lambda record: (float(record["selected_score"]), str(record["name"])),
    )
    best_so_far = math.inf
    best_names: list[str] = []
    top_k = []
    for k, record in enumerate(ordered, 1):
        candidate_runtime = runtime(record)
        if candidate_runtime < best_so_far:
            best_so_far = candidate_runtime
            best_names = [str(record["name"])]
        elif candidate_runtime == best_so_far:
            best_names.append(str(record["name"]))
        top_k.append(
            {
                "k": k,
                "best_runtime_ms": best_so_far,
                "best_layouts": list(best_names),
                "regret": best_so_far / optimum - 1.0,
            }
        )

    return {
        "runtime_statistic": "median_ms",
        "layout_count": layout_count,
        "frontier_size": frontier_size,
        "retained_fraction": frontier_size / layout_count,
        "optimal_runtime_ms": optimum,
        "optimal_layouts": optimal_layouts,
        "best_frontier_runtime_ms": frontier_optimum,
        "best_frontier_layouts": best_frontier_layouts,
        "oracle_regret": oracle_regret,
        "epsilon_metrics": epsilon_metrics,
        "top_k_tie_break": "ascending selected score, then layout name",
        "top_k": top_k,
    }


def analyze_frontier_report(
    groups: Sequence[Mapping[str, object]],
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
) -> dict[str, object]:
    """Aggregate candidate-generation metrics across benchmark instances."""

    if not groups:
        raise ValueError("report analysis requires at least one run")

    instances = []
    group_analyses = []
    for group in groups:
        analysis = analyze_frontier_group(group, epsilons)
        group_analyses.append(analysis)
        instances.append(
            {
                "kernel": str(group["kernel"]),
                "display_name": str(group["display_name"]),
                "matrix_size": int(group["matrix_size"]),
                "layout_count": analysis["layout_count"],
                "frontier_size": analysis["frontier_size"],
                "retained_fraction": analysis["retained_fraction"],
                "optimal_runtime_ms": analysis["optimal_runtime_ms"],
                "optimal_layouts": analysis["optimal_layouts"],
                "best_frontier_runtime_ms": analysis[
                    "best_frontier_runtime_ms"
                ],
                "best_frontier_layouts": analysis["best_frontier_layouts"],
                "oracle_regret": analysis["oracle_regret"],
            }
        )

    regrets = [float(analysis["oracle_regret"]) for analysis in group_analyses]
    retained_fractions = [
        float(analysis["retained_fraction"]) for analysis in group_analyses
    ]
    frontier_sizes = [
        float(analysis["frontier_size"]) for analysis in group_analyses
    ]
    exact_hits = sum(regret == 0.0 for regret in regrets)
    exact_random_probabilities = [
        random_subset_coverage_probability(
            int(analysis["layout_count"]),
            len(analysis["optimal_layouts"]),
            int(analysis["frontier_size"]),
        )
        for analysis in group_analyses
    ]

    epsilon_summary = []
    for index, epsilon in enumerate(epsilons):
        metrics = [
            analysis["epsilon_metrics"][index] for analysis in group_analyses
        ]
        coverage = [bool(metric["covered"]) for metric in metrics]
        random_probabilities = [
            float(metric["random_size_matched_coverage_probability"])
            for metric in metrics
        ]
        purities = [float(metric["purity"]) for metric in metrics]
        enrichments = [float(metric["enrichment"]) for metric in metrics]
        prevalences = [
            float(metric["search_space_prevalence"]) for metric in metrics
        ]
        epsilon_summary.append(
            {
                "epsilon": float(epsilon),
                "epsilon_percent": 100.0 * float(epsilon),
                "covered_instances": sum(coverage),
                "coverage": statistics.fmean(coverage),
                "random_expected_covered_instances": sum(random_probabilities),
                "random_expected_coverage": statistics.fmean(
                    random_probabilities
                ),
                "purity": _summary(purities),
                "enrichment": _summary(enrichments),
                "search_space_prevalence": _summary(prevalences),
            }
        )

    maximum_k = max(int(analysis["layout_count"]) for analysis in group_analyses)
    top_k_summary = []
    for k in range(1, maximum_k + 1):
        entries = [
            analysis["top_k"][min(k, int(analysis["layout_count"])) - 1]
            for analysis in group_analyses
        ]
        k_regrets = [float(entry["regret"]) for entry in entries]
        top_k_summary.append(
            {
                "k": k,
                "mean_candidates_evaluated": statistics.fmean(
                    min(k, int(analysis["layout_count"]))
                    for analysis in group_analyses
                ),
                "regret": _summary(k_regrets),
                "epsilon_coverage": [
                    {
                        "epsilon": float(epsilon),
                        "covered_instances": sum(
                            regret <= epsilon for regret in k_regrets
                        ),
                        "coverage": statistics.fmean(
                            regret <= epsilon for regret in k_regrets
                        ),
                    }
                    for epsilon in epsilons
                ],
            }
        )

    return {
        "runtime_statistic": "median_ms",
        "instance_count": len(groups),
        "epsilon_values": [float(epsilon) for epsilon in epsilons],
        "definitions": {
            "oracle_regret": (
                "best frontier median runtime / best evaluated median runtime - 1"
            ),
            "retained_fraction": "frontier size / feasible layout count",
            "epsilon_optimal": "runtime <= (1 + epsilon) * best runtime",
            "purity": "epsilon-optimal frontier members / frontier size",
            "enrichment": "frontier purity / evaluated-set prevalence",
            "top_k": (
                "best runtime among the first k layouts ordered by selected "
                "scalar score, with layout name as an exact-tie breaker"
            ),
        },
        "oracle_regret": _summary(regrets),
        "retained_fraction": _summary(retained_fractions),
        "frontier_size": _summary(frontier_sizes),
        "exact_winner_coverage": {
            "covered_instances": exact_hits,
            "coverage": exact_hits / len(groups),
        },
        "random_exact_winner_baseline": {
            "expected_covered_instances": sum(exact_random_probabilities),
            "expected_coverage": statistics.fmean(exact_random_probabilities),
            "probability_at_least_observed_hits": poisson_binomial_upper_tail(
                exact_random_probabilities,
                exact_hits,
            ),
        },
        "epsilon_optimal": epsilon_summary,
        "top_k": top_k_summary,
        "instances": instances,
    }


def _equivalence_groups(
    records: Sequence[Mapping[str, object]],
    *,
    include_fine: bool,
) -> dict[str, object]:
    by_vector: dict[tuple[float, ...], list[Mapping[str, object]]] = {}
    for record in records:
        by_vector.setdefault(
            _base_score_vector(record, include_fine=include_fine), []
        ).append(record)

    groups = []
    non_singleton_spreads = []
    for vector, members in sorted(by_vector.items()):
        runtimes = [_runtime(record) for record in members]
        spread = max(runtimes) / min(runtimes) - 1.0
        if len(members) > 1:
            non_singleton_spreads.append(spread)
        groups.append(
            {
                "score_vector": list(vector),
                "layout_count": len(members),
                "layouts": [str(record["name"]) for record in members],
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
        "non_singleton_runtime_spread": (
            _summary(non_singleton_spreads) if non_singleton_spreads else None
        ),
        "groups": groups,
    }


def analyze_score_equivalence(
    group: Mapping[str, object],
) -> dict[str, object]:
    """Measure runtime spread among layouts with identical score vectors."""

    records = group["results"]
    gated = group["fine_locality_gated_frontiers"]
    assert isinstance(records, list) and isinstance(gated, list)
    main = _equivalence_groups(records, include_fine=True)
    main["objectives"] = [
        "wave_load.64B.raw-region-count",
        "peak-normalized-excess",
        "weighted-normalized-excess",
        "codegen-runs",
        "codegen-xors",
    ]
    gated_results = []
    for frontier in gated:
        assert isinstance(frontier, dict)
        limit = float(frontier["fine_limit"])
        eligible = [
            record
            for record in records
            if _base_score_vector(record, include_fine=True)[0] <= limit
        ]
        result = _equivalence_groups(eligible, include_fine=False)
        result.update(
            {
                "delta": float(frontier["delta"]),
                "delta_percent": float(frontier["delta_percent"]),
                "eligible_count": len(eligible),
                "objectives": [
                    "peak-normalized-excess",
                    "weighted-normalized-excess",
                    "codegen-runs",
                    "codegen-xors",
                ],
            }
        )
        gated_results.append(result)
    return {
        "definition": "max median runtime / min median runtime - 1",
        "equality": "exact equality of every score-vector coordinate",
        "main_frontier_vector": main,
        "fine_locality_gated_vectors": gated_results,
    }


def analyze_tau_weight_robustness(
    groups: Sequence[Mapping[str, object]],
    *,
    trials: int = DEFAULT_TAU_PERTURBATION_TRIALS,
    seed: int = 0,
    factors: Sequence[float] = TAU_PERTURBATION_FACTORS,
) -> dict[str, object]:
    """Ablate nonzero tau weights and rebuild the five-cost frontier."""

    if trials <= 0:
        raise ValueError("tau perturbation trials must be positive")
    if not factors or any(factor <= 0 for factor in factors):
        raise ValueError("tau perturbation factors must be positive")

    instances = []
    all_regrets = []
    all_retained = []
    for group in groups:
        records = group["results"]
        assert isinstance(records, list)
        optimum = min(_runtime(record) for record in records)
        active_names = [
            str(component["name"])
            for component in _score_components(records[0])
            if float(component["weight"]) > 0.0
        ]
        base_frontier = [
            record for record in records if record["pareto_frontier_member"]
        ]
        baseline = {
            "frontier_size": len(base_frontier),
            "retained_fraction": len(base_frontier) / len(records),
            "oracle_regret": (
                min(_runtime(record) for record in base_frontier) / optimum - 1.0
            ),
        }
        trial_results = []
        for trial in range(trials):
            digest = hashlib.sha256(
                (
                    f"{seed}:{group['kernel']}:{group['matrix_size']}:{trial}"
                ).encode()
            ).digest()
            generator = random.Random(int.from_bytes(digest[:8], "big"))
            multipliers = {
                name: float(generator.choice(factors)) for name in active_names
            }
            named_vectors = []
            for record in records:
                base = _base_score_vector(record, include_fine=True)
                area = sum(
                    float(component["weight"])
                    * multipliers.get(str(component["name"]), 1.0)
                    * float(component["normalized_excess"])
                    for component in _score_components(record)
                    if float(component["weight"]) > 0.0
                )
                named_vectors.append(
                    (
                        str(record["name"]),
                        (base[0], base[1], area, base[3], base[4]),
                    )
                )
            frontier_names = _pareto_names(named_vectors)
            frontier_records = [
                record
                for record in records
                if str(record["name"]) in frontier_names
            ]
            regret = (
                min(_runtime(record) for record in frontier_records) / optimum
                - 1.0
            )
            retained = len(frontier_records) / len(records)
            all_regrets.append(regret)
            all_retained.append(retained)
            trial_results.append(
                {
                    "trial": trial,
                    "multipliers": multipliers,
                    "frontier_size": len(frontier_records),
                    "retained_fraction": retained,
                    "oracle_regret": regret,
                    "best_frontier_layouts": [
                        str(record["name"])
                        for record in frontier_records
                        if _runtime(record)
                        == min(_runtime(item) for item in frontier_records)
                    ],
                }
            )
        instances.append(
            {
                "kernel": str(group["kernel"]),
                "display_name": str(group["display_name"]),
                "matrix_size": int(group["matrix_size"]),
                "active_tau_names": active_names,
                "baseline": baseline,
                "oracle_regret": _summary(
                    [float(item["oracle_regret"]) for item in trial_results]
                ),
                "retained_fraction": _summary(
                    [float(item["retained_fraction"]) for item in trial_results]
                ),
                "trials": trial_results,
            }
        )
    return {
        "method": (
            "independently draw one multiplier per nonzero tau and instance "
            "from the discrete factor set, then rebuild the notes frontier"
        ),
        "seed": seed,
        "trials_per_instance": trials,
        "factors": [float(factor) for factor in factors],
        "oracle_regret": _summary(all_regrets),
        "retained_fraction": _summary(all_retained),
        "instances": instances,
    }


def render_frontier_plots(
    analysis: Mapping[str, object], output_directory: Path
) -> dict[str, Path]:
    """Render frontier and tau-robustness scorecard plots."""

    matplotlib_cache = Path(tempfile.gettempdir()) / "relay-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "frontier plots require matplotlib and seaborn; install the "
            "project's experiment optional dependencies"
        ) from error

    output_directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths = {
        "epsilon_optimal_coverage": output_directory
        / "epsilon_optimal_coverage.png",
        "retained_fraction_vs_regret": output_directory
        / "retained_fraction_vs_regret.png",
        "purity_and_enrichment": output_directory / "purity_and_enrichment.png",
        "top_k_regret": output_directory / "top_k_regret.png",
        "tau_weight_robustness": output_directory
        / "tau_weight_robustness.png",
        "pareto_depth_regret": output_directory / "pareto_depth_regret.png",
    }

    epsilon = analysis["epsilon_optimal"]
    assert isinstance(epsilon, list)
    epsilon_percent = [float(item["epsilon_percent"]) for item in epsilon]
    coverage = [100.0 * float(item["coverage"]) for item in epsilon]
    random_coverage = [
        100.0 * float(item["random_expected_coverage"]) for item in epsilon
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(epsilon_percent, coverage, marker="o", label="RELAY frontier")
    axis.plot(
        epsilon_percent,
        random_coverage,
        marker="s",
        linestyle="--",
        label="Size-matched random subset",
    )
    axis.set(xlabel="Epsilon tolerance (%)", ylabel="Instance coverage (%)")
    axis.set_ylim(0.0, 102.0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["epsilon_optimal_coverage"], dpi=180)
    plt.close(figure)

    instances = analysis["instances"]
    assert isinstance(instances, list)
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    sns.scatterplot(
        x=[100.0 * float(item["retained_fraction"]) for item in instances],
        y=[100.0 * float(item["oracle_regret"]) for item in instances],
        hue=[str(item["display_name"]) for item in instances],
        style=[str(item["matrix_size"]) for item in instances],
        s=90,
        ax=axis,
    )
    axis.set(
        xlabel="Retained fraction (%)",
        ylabel="Best-in-frontier regret (%)",
    )
    axis.legend(title="Kernel / N", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(paths["retained_fraction_vs_regret"], dpi=180)
    plt.close(figure)

    purity_mean = [100.0 * float(item["purity"]["mean"]) for item in epsilon]
    prevalence_mean = [
        100.0 * float(item["search_space_prevalence"]["mean"])
        for item in epsilon
    ]
    enrichment_mean = [
        float(item["enrichment"]["mean"]) for item in epsilon
    ]
    enrichment_median = [
        float(item["enrichment"]["median"]) for item in epsilon
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(epsilon_percent, purity_mean, marker="o", label="Frontier purity")
    axes[0].plot(
        epsilon_percent,
        prevalence_mean,
        marker="s",
        linestyle="--",
        label="Feasible-set prevalence",
    )
    axes[0].set(xlabel="Epsilon tolerance (%)", ylabel="Mean fraction (%)")
    axes[0].legend()
    axes[1].plot(epsilon_percent, enrichment_mean, marker="o", label="Mean")
    axes[1].plot(epsilon_percent, enrichment_median, marker="s", label="Median")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Epsilon tolerance (%)", ylabel="Frontier enrichment")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(paths["purity_and_enrichment"], dpi=180)
    plt.close(figure)

    top_k = analysis["top_k"]
    assert isinstance(top_k, list)
    k_values = [int(item["k"]) for item in top_k]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for statistic, label in (
        ("maximum", "Maximum"),
        ("mean", "Mean"),
        ("median", "Median"),
    ):
        axis.plot(
            k_values,
            [100.0 * float(item["regret"][statistic]) for item in top_k],
            marker="o" if statistic == "maximum" else None,
            label=label,
        )
    axis.set(
        xlabel="Scalar-score candidate budget k",
        ylabel="Top-k best-runtime regret (%)",
    )
    if len(k_values) <= 24:
        axis.set_xticks(k_values)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["top_k_regret"], dpi=180)
    plt.close(figure)

    robustness = analysis.get("tau_weight_robustness")
    if not isinstance(robustness, dict):
        paths.pop("tau_weight_robustness")
    else:
        robustness_instances = robustness["instances"]
        assert isinstance(robustness_instances, list)
        labels = []
        regret_labels = []
        regrets = []
        retained_labels = []
        retained = []
        baseline_regret = []
        baseline_retained = []
        for instance in robustness_instances:
            assert isinstance(instance, dict)
            label = f"{instance['display_name']}\nN={instance['matrix_size']}"
            labels.append(label)
            trials = instance["trials"]
            baseline = instance["baseline"]
            assert isinstance(trials, list) and isinstance(baseline, dict)
            for trial in trials:
                assert isinstance(trial, dict)
                regret_labels.append(label)
                regrets.append(100.0 * float(trial["oracle_regret"]))
                retained_labels.append(label)
                retained.append(100.0 * float(trial["retained_fraction"]))
            baseline_regret.append(100.0 * float(baseline["oracle_regret"]))
            baseline_retained.append(100.0 * float(baseline["retained_fraction"]))

        figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
        sns.boxplot(x=regret_labels, y=regrets, order=labels, ax=axes[0])
        sns.scatterplot(
            x=labels,
            y=baseline_regret,
            marker="D",
            color="black",
            label="Unperturbed tau",
            ax=axes[0],
        )
        axes[0].set(xlabel="", ylabel="Oracle regret (%)")
        axes[0].legend()
        sns.boxplot(x=retained_labels, y=retained, order=labels, ax=axes[1])
        sns.scatterplot(
            x=labels,
            y=baseline_retained,
            marker="D",
            color="black",
            label="Unperturbed tau",
            ax=axes[1],
        )
        axes[1].set(
            xlabel="Kernel / matrix size", ylabel="Retained fraction (%)"
        )
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(paths["tau_weight_robustness"], dpi=180)
        plt.close(figure)

    information = analysis.get("frontier_information")
    if not isinstance(information, dict):
        paths.pop("pareto_depth_regret")
    else:
        representations = information["representations"]
        assert isinstance(representations, list)
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for representation in representations:
            depth = representation["cumulative_pareto_depth"]
            label = str(representation["label"])
            x_values = [
                100.0 * float(item["retained_fraction"]["mean"])
                for item in depth
            ]
            axes[0].plot(
                x_values,
                [
                    100.0 * float(item["oracle_regret"]["mean"])
                    for item in depth
                ],
                marker="o",
                label=label,
            )
            axes[1].plot(
                x_values,
                [
                    100.0 * float(item["oracle_regret"]["maximum"])
                    for item in depth
                ],
                marker="o",
                label=label,
            )
        axes[0].set(
            xlabel="Mean cumulative retained fraction (%)",
            ylabel="Mean oracle regret (%)",
        )
        axes[1].set(
            xlabel="Mean cumulative retained fraction (%)",
            ylabel="Maximum oracle regret (%)",
        )
        axes[0].legend()
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(paths["pareto_depth_regret"], dpi=180)
        plt.close(figure)

    return paths
