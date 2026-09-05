#!/usr/bin/env python3
"""Export raw tables, Spearman correlations, and plots for experiments 1--3."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence

from layout_panels import STRATIFICATION_MODES


EXPERIMENT_NAMES = {
    1: r"Memory Counters Over Whole-Tensor $G_C$",
    2: r"Memory Counters Over $G_C$ Tiles",
    3: r"Memory Counters Over $G_{OC}$",
}

PLATFORM_LABELS = {
    "tuolumne": "MI300A (Tuolumne)",
    "matrix": "H100 (Matrix)",
}


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + 1 + end)
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return tie-aware Spearman rho without requiring SciPy."""

    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return float(statistics.correlation(_ranks(left), _ranks(right)))


def _component_map(candidate: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(component["name"]): component
        for component in candidate["score"]["components"]
    }


def _counter_values(
    candidate: Mapping[str, object], names: Iterable[str]
) -> dict[str, float]:
    if "counters" not in candidate:
        return {}
    summary = candidate["counters"]["steady_state"]
    return {
        name: float(summary[name])
        for name in names
        if summary.get(name) is not None
    }


def _predictors(candidate: Mapping[str, object]) -> dict[str, float]:
    components = _component_map(candidate)
    result = {
        "J_area": float(candidate["j_area"]),
        "peak_normalized_excess": float(candidate["peak_normalized_excess"]),
    }
    for name, component in components.items():
        result[f"Q:{name}"] = float(component["raw_region_count"])
        result[f"excess:{name}"] = float(component["excess_footprint"])
    return result


def _quotient_coordinate(predictor: str) -> tuple[str | None, int | None]:
    if not predictor.startswith("Q:"):
        return None, None
    scope, separator, scale = predictor.removeprefix("Q:").rpartition(".")
    if not separator or not scale.endswith("B") or not scale[:-1].isdigit():
        return None, None
    return scope, int(scale[:-1])


def _observations(report: Mapping[str, object]):
    profile = report["panel"]["score_profile"]
    counter_names = tuple(profile["counter_components"])
    observations = []
    for candidate in report["candidates"]:
        counters = _counter_values(candidate, counter_names)
        if not counters:
            continue
        observations.append(
            {
                "candidate": candidate,
                "predictors": _predictors(candidate),
                "counters": counters,
            }
        )
    return observations


def correlation_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    observations = _observations(report)
    if not observations:
        return []
    profile = report["panel"]["score_profile"]
    matches = profile["counter_components"]
    focus_counters = set(profile.get("focus_counters", ()))
    predictors = tuple(observations[0]["predictors"])
    rows = []
    for counter, matched_component in matches.items():
        available = [item for item in observations if counter in item["counters"]]
        matched_scope, _ = _quotient_coordinate(f"Q:{matched_component}")
        for predictor in predictors:
            left = [item["predictors"][predictor] for item in available]
            right = [item["counters"][counter] for item in available]
            predictor_scope, predictor_byte_scale = _quotient_coordinate(predictor)
            rows.append(
                {
                    "experiment": report["final_experiment"],
                    "platform": profile["platform"],
                    "case": report["case"],
                    "stratification": report["panel"]["stratification"]["mode"],
                    "counter": counter,
                    "counter_role": (
                        "focus" if counter in focus_counters else "diagnostic"
                    ),
                    "predictor": predictor,
                    "predictor_scope": predictor_scope,
                    "predictor_byte_scale": predictor_byte_scale,
                    "relationship": (
                        "counter_component"
                        if predictor == f"Q:{matched_component}"
                        else "counter_scope_bytescale"
                        if predictor_scope == matched_scope
                        else "aggregate"
                        if predictor == "J_area"
                        else "diagnostic"
                    ),
                    "observations": len(available),
                    "spearman_rho": spearman(left, right),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _raw_rows(
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    observations = _observations(report)
    if not observations:
        return [], []
    predictor_names = tuple(observations[0]["predictors"])
    counter_names = tuple(report["panel"]["score_profile"]["counter_components"])
    fields = [
        "experiment",
        "platform",
        "case",
        "stratification",
        "candidate_id",
        "mapping_id",
        "sample_index",
        "candidate_pool_index",
        "sampling_origin",
        "grammar",
        "inner_tile_shape",
        "inner_word",
        "a_rows",
        "address_expression_runs",
        "xor_count",
        "completed_profile_launches",
        *predictor_names,
        *counter_names,
    ]
    rows = []
    for item in observations:
        candidate = item["candidate"]
        row = {
            "experiment": report["final_experiment"],
            "platform": report["panel"]["score_profile"]["platform"],
            "case": report["case"],
            "stratification": report["panel"]["stratification"]["mode"],
            "candidate_id": candidate["candidate_id"],
            "mapping_id": candidate["mapping_id"],
            "sample_index": candidate["sample_index"],
            "candidate_pool_index": candidate["candidate_pool_index"],
            "sampling_origin": candidate["sampling_origin"],
            "grammar": candidate["grammar"],
            "inner_tile_shape": json.dumps(candidate["inner_tile_shape"]),
            "inner_word": candidate["inner_word"],
            "a_rows": json.dumps(candidate["a_rows"]),
            "address_expression_runs": candidate["address_expression_runs"],
            "xor_count": candidate["xor_count"],
            "completed_profile_launches": candidate[
                "completed_profile_launches"
            ],
            **item["predictors"],
            **item["counters"],
        }
        rows.append(row)
    return rows, fields


def _plot_report(
    report: Mapping[str, object], correlations, path: Path
) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / f"relay-matplotlib-{os.getuid()}"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    observations = _observations(report)
    if not observations:
        return
    profile = report["panel"]["score_profile"]
    lookup = {
        (row["counter"], row["predictor"]): row["spearman_rho"]
        for row in correlations
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        focus = set(profile.get("focus_counters", ()))
        counter_components = profile["counter_components"]
        counter_order = sorted(
            counter_components,
            key=lambda counter: (
                counter not in focus,
                list(counter_components).index(counter),
            ),
        )
        for counter in counter_order:
            component = counter_components[counter]
            available = [item for item in observations if counter in item["counters"]]
            if not available:
                continue
            matched_scope, _ = _quotient_coordinate(f"Q:{component}")
            quotient_predictors = sorted(
                (
                    predictor
                    for predictor in available[0]["predictors"]
                    if _quotient_coordinate(predictor)[0] == matched_scope
                ),
                key=lambda predictor: _quotient_coordinate(predictor)[1],
            )
            specifications = [
                (
                    predictor,
                    rf"$Q_{{{_quotient_coordinate(predictor)[1]}B}}$ at "
                    f"{matched_scope}",
                )
                for predictor in quotient_predictors
            ]
            specifications.append(
                ("J_area", r"Aggregate selection score: $J_{area}$")
            )
            columns = 3
            rows = (len(specifications) + columns - 1) // columns
            figure, axes = plt.subplots(
                rows, columns, figsize=(16.5, 5.0 * rows), squeeze=False
            )
            flat_axes = list(axes.flat)
            for axis, (predictor, title) in zip(flat_axes, specifications):
                x = [item["predictors"][predictor] for item in available]
                y = [item["counters"][counter] for item in available]
                axis.scatter(
                    x,
                    y,
                    s=48,
                    marker="o",
                    facecolor="#0072B2",
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.78,
                )
                rho = lookup[(counter, predictor)]
                rho_text = "undefined" if rho is None else f"{rho:.3f}"
                axis.set_title(f"{title}\nSpearman $\\rho$ = {rho_text}", fontsize=12)
                x_label = (
                    r"$J_{area}$"
                    if predictor == "J_area"
                    else "Quotient component " + predictor.removeprefix("Q:")
                )
                axis.set_xlabel(x_label, fontsize=11)
                axis.set_ylabel(counter.replace("_", " "), fontsize=11)
                axis.grid(True, alpha=0.25)
                axis.set_xlim(left=0)
                axis.set_ylim(bottom=0)
                axis.tick_params(labelsize=10)
            for axis in flat_axes[len(specifications):]:
                axis.set_visible(False)
            experiment = int(report["final_experiment"])
            shape = "x".join(str(value) for value in report["operand_shape"])
            figure.suptitle(
                f"{EXPERIMENT_NAMES[experiment]} — {report['case']} on "
                f"{PLATFORM_LABELS[profile['platform']]}\n"
                f"{len(available)} independently profiled layout mappings; "
                f"target {shape}; stratified over "
                f"{report['panel']['stratification']['mode']} components",
                fontsize=14,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.91))
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def analyze_report(report_path: Path, plot_path: Path) -> dict[str, object]:
    """Analyze one case report and write all final artifacts."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "final_experiment" not in report:
        raise ValueError(f"{report_path} is not a final-experiment report")
    rows, fields = _raw_rows(report)
    correlations = correlation_rows(report)
    raw_path = report_path.with_name("raw-data.csv")
    spearman_path = report_path.with_name("spearman.csv")
    summary_path = report_path.with_name("analysis.json")
    _write_csv(raw_path, rows, fields)
    correlation_fields = (
        "experiment",
        "platform",
        "case",
        "stratification",
        "counter",
        "counter_role",
        "predictor",
        "predictor_scope",
        "predictor_byte_scale",
        "relationship",
        "observations",
        "spearman_rho",
    )
    _write_csv(spearman_path, correlations, correlation_fields)
    _plot_report(report, correlations, plot_path)

    primary = [
        row
        for row in correlations
        if row["relationship"]
        in {"counter_component", "counter_scope_bytescale", "aggregate"}
    ]
    result = {
        "experiment": report["final_experiment"],
        "platform": report["panel"]["score_profile"]["platform"],
        "case": report["case"],
        "stratification": report["panel"]["stratification"]["mode"],
        "complete": report["complete"],
        "correct": report["correct"],
        "profiled_mapping_count": len(rows),
        "requested_mapping_count": report["panel"]["requested_sample_count"],
        "realized_mapping_count": report["panel"]["realized_sample_count"],
        "score_profile": report["panel"]["score_profile"],
        "primary_correlations": primary,
        "artifacts": {
            "profiler_report": str(report_path),
            "raw_data": str(raw_path),
            "spearman": str(spearman_path),
            "plot": str(plot_path),
        },
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def analyze_suite(
    results_root: Path,
    plots_root: Path,
    *,
    experiment: int,
    platform: str,
    stratification: str,
    regenerate_reports: bool = True,
) -> Path:
    """Combine per-kernel correlations, optionally regenerating each case."""

    rows = []
    suite_root = (
        results_root
        / f"experiment-{experiment}"
        / platform
        / f"stratified-{stratification}"
    )
    for report_path in sorted(suite_root.glob("*/report.json")):
        plot = (
            plots_root
            / f"experiment-{experiment}"
            / platform
            / f"stratified-{stratification}"
            / f"{report_path.parent.name}.pdf"
        )
        if regenerate_reports:
            analyze_report(report_path, plot)
        with report_path.with_name("spearman.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise FileNotFoundError("no per-kernel reports were found")
    output = suite_root / "spearman.csv"
    _write_csv(output, rows, rows[0].keys())
    return output


def parse_arguments(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument(
        "--stratification", choices=STRATIFICATION_MODES, default="all"
    )
    parser.add_argument("--case")
    parser.add_argument("--results-root", type=Path, default=root / "results")
    parser.add_argument("--plots-root", type=Path, default=root / "plots")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    if args.case:
        report = (
            args.results_root
            / f"experiment-{args.experiment}"
            / args.platform
            / f"stratified-{args.stratification}"
            / args.case
            / "report.json"
        )
        plot = (
            args.plots_root
            / f"experiment-{args.experiment}"
            / args.platform
            / f"stratified-{args.stratification}"
            / f"{args.case}.pdf"
        )
        result = analyze_report(report, plot)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        output = analyze_suite(
            args.results_root,
            args.plots_root,
            experiment=args.experiment,
            platform=args.platform,
            stratification=args.stratification,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
