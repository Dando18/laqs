#!/usr/bin/env python3
"""Plot quotient level against steady-state first-level memory activity."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from stage1_counter_analysis import rank_correlation


METRICS = {
    "TCP_TOTAL_CACHE_ACCESSES_sum": {
        "field": "l1_cache_line_accesses",
        "title": "First Level Memory Activity (read events)",
        "raw_label": "First Level Memory Activity\n(million read events)",
        "normalized_label": (
            "First Level Memory Activity\n(read events per issue cohort)"
        ),
        "suffix": "tcp",
    },
    "TCP_TOTAL_READ_sum": {
        "field": "first_level_read_events",
        "title": "First Level Read Activity",
        "raw_label": "First Level Read Activity\n(million read events)",
        "normalized_label": (
            "First Level Read Activity\n(read events per issue cohort)"
        ),
        "suffix": "tcp-total-read",
    },
    "TCP_TCC_READ_REQ_sum": {
        "field": "l1_to_l2_read_requests",
        "title": "L1-to-L2 Read Requests",
        "raw_label": "L1-to-L2 Read Requests\n(million read requests)",
        "normalized_label": "L1-to-L2 read requests\nper issue cohort",
        "suffix": "tcp-tcc-read-req",
    },
    "TCC_REQ_sum": {
        "field": "l2_tag_requests",
        "title": "L2 Tag Requests",
        "raw_label": "L2 Tag Requests\n(million requests)",
        "normalized_label": "L2 tag requests\nper issue cohort",
        "suffix": "tcc-req",
    },
    "TCC_READ_sum": {
        "field": "second_level_read_requests",
        "title": "Second Level Read Activity",
        "raw_label": "Second Level Read Activity\n(million read requests)",
        "normalized_label": (
            "Second Level Read Activity\n(read requests per issue cohort)"
        ),
        "suffix": "tcc-read",
    },
}


def report_architecture(report: dict[str, object]) -> dict[str, str]:
    configuration = report.get("configuration", {})
    backend = configuration.get("profiler_backend")
    if backend == "nvidia_ncu":
        return {"display": "NVIDIA H100", "slug": "h100"}
    if backend in (None, "amd_rocprof") and configuration.get("rocprof"):
        return {"display": "AMD MI300A", "slug": "mi300a"}
    raise ValueError("report does not identify an H100 or MI300A profiler")


def report_metric_spec(
    report: dict[str, object], metric: str
) -> dict[str, str]:
    architecture = report_architecture(report)
    if architecture["slug"] != "h100":
        return METRICS[metric]
    if metric != "TCP_TOTAL_CACHE_ACCESSES_sum":
        raise ValueError(f"H100 report does not contain {metric}")
    first_summary = next(
        candidate["counters"]["steady_state"]
        for candidate in report["candidates"]
        if candidate["complete"]
    )
    field = (
        "first_level_memory_accesses"
        if "first_level_memory_accesses" in first_summary
        else "l1_cache_line_accesses"
    )
    return {
        **METRICS[metric],
        "field": field,
        "suffix": "l1tex-global-load-sectors",
    }


def report_first_level_field(report: dict[str, object]) -> str:
    first_summary = next(
        candidate["counters"]["steady_state"]
        for candidate in report["candidates"]
        if candidate["complete"]
    )
    for field in (
        "first_level_memory_accesses",
        "first_level_read_events",
        "l1_cache_line_accesses",
    ):
        if field in first_summary:
            return field
    raise ValueError("report does not contain a first-level memory metric")


def default_output_path(
    report_path: Path,
    report: dict[str, object],
    *,
    metric: str,
    normalize_per_issue: bool,
    label_layout_words: bool = False,
) -> Path:
    architecture = report_architecture(report)
    stem = report_path.stem
    for existing_tag in ("-matrix", "-h100", "-mi300a"):
        if stem.endswith(existing_tag):
            stem = stem[: -len(existing_tag)]
            break
    suffixes = [
        architecture["slug"],
        report_metric_spec(report, metric)["suffix"],
    ]
    if normalize_per_issue:
        suffixes.append("normalized")
    if label_layout_words:
        suffixes.append("layout-words")
    return report_path.with_name(f"{stem}-{'-'.join(suffixes)}.pdf")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="output PDF (default includes GPU, metric, and normalization)",
    )
    parser.add_argument(
        "--normalize-per-issue",
        action="store_true",
        help="divide memory activity by the number of dynamic issue cohorts",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="TCP_TOTAL_CACHE_ACCESSES_sum",
        help="hardware counter to plot",
    )
    parser.add_argument(
        "--label-layout-words",
        action="store_true",
        help="label points with inner layout words ordered from LSB to MSB",
    )
    return parser.parse_args(argv)


def _configure_font(plt) -> str:
    from matplotlib import font_manager

    for family in ("Gill Sans", "Gill Sans MT"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        return family
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


def load_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("experiment") != "triton_stage1_quotient_level_counters":
        raise ValueError(f"{path} is not a quotient-level counter report")
    candidates = [
        candidate for candidate in report["candidates"] if candidate["complete"]
    ]
    if len(candidates) < 2:
        raise ValueError(f"{path} needs at least two fully profiled candidates")
    return report


def _nice_ticks(maximum: float, intervals: int):
    import numpy as np

    if maximum <= 0.0:
        return 1.0, np.asarray((0.0, 1.0))
    raw_step = maximum / intervals
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    step = next(
        candidate * magnitude
        for candidate in (1.0, 2.0, 2.5, 5.0, 10.0)
        if candidate >= normalized
    )
    upper = math.ceil(maximum / step) * step
    return upper, np.arange(0.0, upper + 0.5 * step, step)


def _issue_axis(
    report: dict[str, object], candidates: list[dict[str, object]]
) -> tuple[list[float], float] | None:
    theory = report.get("theory_validation") or {}
    validation = {
        candidate["candidate_id"]: candidate
        for candidate in theory.get("candidates", ())
    }
    missing = [
        candidate["candidate_id"]
        for candidate in candidates
        if candidate["candidate_id"] not in validation
    ]
    if missing:
        if report["case"] != "gemv":
            return None
        shape = [int(value) for value in report["operand_shape"]]
        cohorts = set()
        for candidate in candidates:
            execution = candidate["structural_validation"]["by_launch"][0][
                "reference_execution_layout"
            ]
            output_elements = math.prod(execution["output_shape"])
            if shape[0] % output_elements:
                raise ValueError("GEMV output tile does not divide its row count")
            cohorts.add(
                (shape[0] // output_elements)
                * shape[1]
                * int(execution["input_sizes"]["register"])
            )
        if len(cohorts) != 1 or next(iter(cohorts)) <= 0:
            raise ValueError("GEMV candidates disagree on issue-cohort count")
        dynamic_cohorts = float(cohorts.pop())
        cardinalities = [
            float(candidate["quotient_score"]) / dynamic_cohorts
            for candidate in candidates
        ]
        if any(
            value <= 0.0 or not value.is_integer()
            for value in cardinalities
        ):
            raise ValueError("GEMV quotient score is not an issue cardinality")
        return cardinalities, dynamic_cohorts

    cohorts = {
        float(validation[candidate["candidate_id"]]["dynamic_issue_cohorts"])
        for candidate in candidates
    }
    if len(cohorts) != 1 or next(iter(cohorts)) <= 0.0:
        raise ValueError("candidates must share one positive issue-cohort count")
    cardinalities = [
        float(
            validation[candidate["candidate_id"]][
                "issue_quotient_cardinality"
            ]
        )
        for candidate in candidates
    ]
    return cardinalities, cohorts.pop()


def _saturated_suffix(values):
    if len(values) < 2:
        return None
    level = float(values[-1])
    tolerance = max(1.0, abs(level) * 0.005)
    start = len(values) - 1
    while start > 0 and abs(float(values[start - 1]) - level) <= tolerance:
        start -= 1
    if len(values) - start < 2:
        return None
    return start, level


def _compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.3g}k"
    return f"{value:g}"


def render(
    report: dict[str, object],
    output: Path,
    *,
    metric: str = "TCP_TOTAL_CACHE_ACCESSES_sum",
    normalize_per_issue: bool = False,
    label_layout_words: bool = False,
) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    font = _configure_font(plt)
    architecture = report_architecture(report)
    candidates = sorted(
        (candidate for candidate in report["candidates"] if candidate["complete"]),
        key=lambda candidate: candidate["quotient_score"],
    )
    if label_layout_words:
        missing_words = [
            candidate["candidate_id"]
            for candidate in candidates
            if not candidate.get("inner_word")
        ]
        if missing_words:
            raise ValueError(
                "report is missing inner layout words for: "
                + ", ".join(missing_words)
            )
    issue_axis = _issue_axis(report, candidates)
    if issue_axis is None:
        if normalize_per_issue:
            raise ValueError(
                "report does not record the dynamic issue-cohort count needed "
                "for --normalize-per-issue"
            )
        cardinalities = [
            float(candidate["quotient_score"]) for candidate in candidates
        ]
        dynamic_issue_cohorts = 1.0
        x_label = "Quotient Score"
    else:
        cardinalities, dynamic_issue_cohorts = issue_axis
        transaction_bytes = int(report["configuration"]["transaction_bytes"])
        x_label = (
            f"Quotient Score: {transaction_bytes}-B regions per issue"
        )
    metric_spec = report_metric_spec(report, metric)
    metric_field = metric_spec["field"]
    x = np.arange(len(candidates), dtype=float)
    launch_values = [
        candidate["counters"]["steady_state"][
            f"{metric_field}_by_launch"
        ]
        for candidate in candidates
    ]
    y = np.asarray(
        [
            candidate["counters"]["steady_state"][metric_field]
            for candidate in candidates
        ]
    )
    first_level_field = report_first_level_field(report)
    first_level_y = np.asarray(
        [
            candidate["counters"]["steady_state"][first_level_field]
            for candidate in candidates
        ]
    )
    y_min = np.asarray([min(values) for values in launch_values])
    y_max = np.asarray([max(values) for values in launch_values])
    if normalize_per_issue:
        y /= dynamic_issue_cohorts
        y_min /= dynamic_issue_cohorts
        y_max /= dynamic_issue_cohorts
    else:
        y /= 1_000_000.0
        y_min /= 1_000_000.0
        y_max /= 1_000_000.0
    rho = rank_correlation(cardinalities, y.tolist())
    saturation = _saturated_suffix(first_level_y)
    pre_saturation_end = saturation[0] if saturation is not None else len(y)
    pre_saturation_rho = rank_correlation(
        cardinalities[:pre_saturation_end],
        y[:pre_saturation_end].tolist(),
    )

    figure_size = (5.0, 4.5) if label_layout_words else (4.4, 4.0)
    figure, axis = plt.subplots(figsize=figure_size, layout="constrained")
    ranges = y_max - y_min
    if np.any(ranges > 0.0):
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - y_min, y_max - y)),
            fmt="o",
            markersize=8,
            markerfacecolor="#0072B2",
            markeredgecolor="black",
            markeredgewidth=0.8,
            ecolor="#4d4d4d",
            elinewidth=1.4,
            capsize=4,
            label="Median and min–max across profiler launches",
            zorder=3,
        )
    else:
        axis.scatter(
            x,
            y,
            s=64,
            marker="o",
            facecolor="#0072B2",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
    y_upper, y_ticks = _nice_ticks(float(max(y)), intervals=5)
    axis.set_xlim(-0.5, len(x) - 0.5)
    axis.set_ylim(0.0, y_upper)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{value:g}" if issue_axis is not None else _compact_number(value)
            for value in cardinalities
        ]
    )
    axis.set_yticks(y_ticks)

    def formatter(value, _pos):
        if normalize_per_issue:
            return f"{value:,.0f}"
        if math.isclose(value * 10.0, round(value * 10.0)):
            return f"{value:.1f}"
        return f"{value:.2f}"

    axis.yaxis.set_major_formatter(FuncFormatter(formatter))
    if label_layout_words:
        x_label += (
            "\nPoint labels: inner layout word "
            "(LSB $\\rightarrow$ MSB)"
        )
    axis.set_xlabel(
        x_label,
        fontsize=13,
        labelpad=64 if label_layout_words else None,
    )
    axis.set_ylabel(
        (
            metric_spec["normalized_label"]
            if normalize_per_issue
            else metric_spec["raw_label"]
        ),
        fontsize=13,
    )
    axis.set_title(
        (
            "Quotient Score"
            if issue_axis is not None
            else "Quotient Score"
        )
        + f" versus {metric_spec['title']}\n"
        f"for {report['case'].replace('_', ' ').upper()} Kernel "
        f"on {architecture['display']}",
        fontsize=12.5,
        pad=8,
    )
    rho_text = "n/a" if rho is None else f"{rho:.3f}"
    pre_saturation_rho_text = (
        "n/a"
        if pre_saturation_rho is None
        else f"{pre_saturation_rho:.3f}"
    )
    correlation_text = f"Spearman $\\rho$ = {rho_text}"
    if rho is None or not math.isclose(rho, 1.0, abs_tol=1e-12):
        correlation_text += (
            "\nCoalesced region\nSpearman "
            f"$\\rho$ = {pre_saturation_rho_text}"
        )
    axis.text(
        0.97,
        0.03,
        correlation_text,
        transform=axis.transAxes,
        ha="right",
        multialignment="left",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        zorder=4,
    )
    if label_layout_words:
        for point_x, candidate in zip(x, candidates):
            axis.text(
                float(point_x),
                -0.105,
                candidate["inner_word"],
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                rotation=45,
                rotation_mode="anchor",
                fontsize=8.5,
                fontfamily="monospace",
                clip_on=False,
                zorder=5,
            )
    if (
        saturation is not None
        and metric
        in ("TCP_TOTAL_CACHE_ACCESSES_sum", "TCP_TOTAL_READ_sum")
    ):
        saturation_start, _tcp_saturation_level = saturation
        saturation_level = float(y[-1])
        axis.axhline(
            saturation_level,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        axis.annotate(
            "Fully uncoalesced: one access per lane\n"
            + (
                "for both matrix loads"
                if report["case"] == "gesummv"
                else "for the matrix load"
            ),
            xy=(float(x[saturation_start]), saturation_level),
            xycoords="data",
            xytext=(0.98, 0.88),
            textcoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            arrowprops={
                "arrowstyle": "-",
                "color": "#666666",
                "linewidth": 1.0,
            },
        )
    axis.grid(axis="both", color="#dddddd", linewidth=0.7, zorder=0)
    axis.tick_params(labelsize=12)
    axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return font


def main() -> None:
    args = parse_arguments()
    cache_dir = Path(tempfile.gettempdir()) / "relay-matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    report_path = args.report.resolve()
    report = load_report(report_path)
    output = (
        args.output.resolve()
        if args.output
        else default_output_path(
            report_path,
            report,
            metric=args.metric,
            normalize_per_issue=args.normalize_per_issue,
            label_layout_words=args.label_layout_words,
        )
    )
    font = render(
        report,
        output,
        metric=args.metric,
        normalize_per_issue=args.normalize_per_issue,
        label_layout_words=args.label_layout_words,
    )
    print(f"Wrote {output} using {font}")


if __name__ == "__main__":
    main()
