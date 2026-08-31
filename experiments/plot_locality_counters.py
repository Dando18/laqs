#!/usr/bin/env python3
"""Render PDF comparisons from a completed LAQS locality-counter report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    ("modeled_quotient", "Modeled $Q_{fine}$"),
    ("l1_cache_line_accesses", "TCP cache-line accesses"),
    ("l1_to_l2_read_requests", "TCP$\\to$TCC read requests"),
    ("l1_to_l2_total_requests", "TCP$\\to$TCC total requests"),
    ("l2_tag_requests", "L2 tag requests"),
    ("l2_misses", "L2 misses"),
    ("hbm_read_bytes", "HBM read bytes"),
    ("duration_ns", "Profiled duration"),
)
SCATTER_METRICS = (
    ("l1_cache_line_accesses", "TCP cache-line accesses", "o", "#0072B2"),
    ("l1_to_l2_read_requests", "TCP$\\to$TCC reads", "s", "#D55E00"),
    ("l2_tag_requests", "L2 tag requests", "^", "#009E73"),
    ("l2_misses", "L2 misses", "D", "#CC79A7"),
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PDF directory (default: REPORT stem plus _plots)",
    )
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text())
    if report.get("experiment") != "laqs-locality-hardware-counters":
        raise ValueError(f"{path} is not a LAQS locality-counter report")
    completed = [
        record
        for record in report.get("kernels", [])
        if record.get("hardware_comparison") is not None
    ]
    if not completed:
        raise ValueError(f"{path} contains no completed kernel pairs")
    report["kernels"] = completed
    return report


def _reductions(record: dict[str, object]) -> list[float]:
    steady = record["hardware_comparison"]["steady_state"]
    values = []
    for field, _label in METRICS:
        if field == "modeled_quotient":
            reduction = record["model_comparison"]["predicted_reduction"]
        else:
            reduction = steady[field]["reduction"]
        values.append(float("nan") if reduction is None else 100.0 * float(reduction))
    return values


def render_reduction_matrix(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    kernels = report["kernels"]
    values = np.asarray([_reductions(record) for record in kernels])
    finite = np.abs(values[np.isfinite(values)])
    limit = max(10.0, float(np.max(finite)) if finite.size else 10.0)
    figure, axis = plt.subplots(figsize=(13.2, 4.8 + 0.38 * len(kernels)))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(range(len(METRICS)), [label for _field, label in METRICS])
    axis.set_yticks(
        range(len(kernels)), [record["display_name"] for record in kernels]
    )
    axis.tick_params(axis="x", labelrotation=31, labelsize=10)
    axis.tick_params(axis="y", labelsize=11)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "n/a" if not np.isfinite(value) else f"{value:+.1f}%"
            color = "white" if np.isfinite(value) and abs(value) > 0.55 * limit else "black"
            axis.text(column, row, label, ha="center", va="center", fontsize=9, color=color)
    axis.set_title(
        "LAQS-selected versus row-major: positive values mean fewer requests",
        fontsize=14,
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Reduction relative to row-major (%)", fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def render_prediction_scatter(report: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    kernels = report["kernels"]
    figure, axis = plt.subplots(figsize=(8.4, 6.5))
    all_values = [0.0]
    for field, label, marker, color in SCATTER_METRICS:
        x_values = [
            100.0 * float(record["model_comparison"]["predicted_reduction"])
            for record in kernels
        ]
        y_values = [
            100.0
            * float(record["hardware_comparison"]["steady_state"][field]["reduction"])
            for record in kernels
        ]
        all_values.extend(x_values)
        all_values.extend(y_values)
        axis.scatter(
            x_values,
            y_values,
            marker=marker,
            s=76,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            label=label,
        )
        if field == "l1_to_l2_read_requests":
            for x_value, y_value, record in zip(x_values, y_values, kernels):
                right_side = x_value > 0.75 * max(x_values)
                near_top = y_value > 0.85 * max(y_values)
                axis.annotate(
                    record["display_name"],
                    (x_value, y_value),
                    xytext=(-5 if right_side else 5, -12 if near_top else 4),
                    textcoords="offset points",
                    ha="right" if right_side else "left",
                    fontsize=9,
                )
    lower = min(all_values) - 5.0
    upper = max(all_values) + 5.0
    axis.plot([lower, upper], [lower, upper], color="0.45", linestyle="--", label="equal reduction")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Predicted $Q_{fine}$ reduction (%)", fontsize=12)
    axis.set_ylabel("Measured steady-state reduction (%)", fontsize=12)
    axis.set_title("Does lower quotient cost predict fewer hardware requests?", fontsize=14)
    axis.grid(True, linestyle=":", linewidth=0.7)
    axis.legend(fontsize=9, loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    report_path = args.report.expanduser().resolve()
    report = _load(report_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else report_path.with_suffix("").with_name(report_path.stem + "_plots")
    )
    matrix = output_dir / "locality_counter_reductions.pdf"
    scatter = output_dir / "quotient_vs_hardware_reductions.pdf"
    render_reduction_matrix(report, matrix)
    render_prediction_scatter(report, scatter)
    print(f"Wrote {matrix}")
    print(f"Wrote {scatter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
