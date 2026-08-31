#!/usr/bin/env python3
"""Create PDF figures from a Triton Stage-1 locality-counter report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


HEATMAP_METRICS = (
    ("issue_quotient", "Modeled issue quotient"),
    ("objective_quotient", "Full objective quotient"),
    ("l1_cache_line_accesses", "TCP cache-line accesses"),
    ("l1_to_l2_read_requests", "TCP→TCC read requests"),
    ("l2_tag_requests", "L2 tag requests"),
    ("l2_misses", "L2 misses"),
    ("hbm_read_bytes", "HBM read bytes"),
    ("duration_ns", "Profiled duration"),
)

SCATTER_METRICS = (
    ("l1_cache_line_accesses", "TCP cache-line accesses", "o", "#0072B2"),
    ("l1_to_l2_read_requests", "TCP→TCC reads", "s", "#D55E00"),
    ("l2_tag_requests", "L2 tag requests", "^", "#009E73"),
    ("l2_misses", "L2 misses", "D", "#CC79A7"),
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PDF directory (default: REPORT stem plus -plots)",
    )
    return parser.parse_args(argv)


def load_pairs(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("experiment") != "triton_stage1_locality_counters":
        raise ValueError(f"{path} is not a Triton locality-counter report")
    pairs = []
    for case, case_record in report.get("cases", {}).items():
        for cache_mode, pair in case_record["cache_modes"].items():
            pairs.append((case, cache_mode, pair))
    if not pairs:
        raise ValueError(f"{path} contains no complete layout pairs")
    return report, pairs


def reduction(pair: dict[str, object], metric: str) -> float | None:
    if metric == "issue_quotient":
        return pair["quotient"]["issue"]["reduction"]
    if metric == "objective_quotient":
        return pair["quotient"]["objective"]["reduction"]
    return pair["hardware_comparison"][metric]["reduction"]


def render_reduction_matrix(pairs, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    labels = [f"{case} ({cache_mode})" for case, cache_mode, _pair in pairs]
    values = np.asarray(
        [
            [
                (
                    np.nan
                    if reduction(pair, metric) is None
                    else 100 * reduction(pair, metric)
                )
                for metric, _label in HEATMAP_METRICS
            ]
            for _case, _cache_mode, pair in pairs
        ]
    )
    finite = np.abs(values[np.isfinite(values)])
    limit = max(10.0, float(np.max(finite)) if finite.size else 10.0)
    figure, axis = plt.subplots(figsize=(14.2, 4.0 + 0.42 * len(pairs)))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(
        range(len(HEATMAP_METRICS)),
        [label for _metric, label in HEATMAP_METRICS],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(labelsize=10)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "n/a" if not np.isfinite(value) else f"{value:+.1f}%"
            color = (
                "white"
                if np.isfinite(value) and abs(value) > 0.58 * limit
                else "black"
            )
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    axis.set_title(
        "LAQS-selected versus Triton row-major: locality predictions and counters",
        fontsize=15,
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Reduction relative to row-major (%)", fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def render_prediction_scatter(pairs, output: Path) -> None:
    import matplotlib.pyplot as plt

    cache_modes = [
        mode for mode in ("warm", "thrashed") if any(pair[1] == mode for pair in pairs)
    ]
    figure, axes = plt.subplots(
        1,
        len(cache_modes),
        figsize=(7.2 * len(cache_modes), 6.2),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    all_values = [0.0]
    for column, cache_mode in enumerate(cache_modes):
        axis = axes[0][column]
        selected_pairs = [pair for pair in pairs if pair[1] == cache_mode]
        for metric, label, marker, color in SCATTER_METRICS:
            points = []
            for case, _mode, pair in selected_pairs:
                x_value = reduction(pair, "issue_quotient")
                y_value = reduction(pair, metric)
                if x_value is None or y_value is None:
                    continue
                points.append((100 * x_value, 100 * y_value, case))
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
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
            if metric == "l1_cache_line_accesses":
                for x_value, y_value, case in points:
                    axis.annotate(
                        case,
                        (x_value, y_value),
                        xytext=(5, 4),
                        textcoords="offset points",
                        fontsize=9,
                    )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.grid(True, linestyle=":", linewidth=0.7)
        axis.set_title(f"{cache_mode.capitalize()} cache", fontsize=14)
        axis.set_xlabel("Modeled issue-quotient reduction (%)", fontsize=12)
        if column == 0:
            axis.set_ylabel("Measured counter reduction (%)", fontsize=12)
        axis.legend(fontsize=9, loc="best")
    lower = min(all_values) - 5.0
    upper = max(all_values) + 5.0
    for axis in axes[0]:
        axis.plot(
            [lower, upper],
            [lower, upper],
            color="0.45",
            linestyle="--",
            label="equal reduction",
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
    figure.suptitle(
        "Does a lower LAQS transaction quotient produce fewer memory requests?",
        fontsize=15,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    cache_dir = Path(tempfile.gettempdir()) / "relay-matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    report_path = args.report.resolve()
    _report, pairs = load_pairs(report_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else report_path.with_suffix("").with_name(report_path.stem + "-plots")
    )
    matrix = output_dir / "locality-counter-reductions.pdf"
    scatter = output_dir / "quotient-vs-memory-requests.pdf"
    render_reduction_matrix(pairs, matrix)
    render_prediction_scatter(pairs, scatter)
    print(f"Wrote {matrix}")
    print(f"Wrote {scatter}")


if __name__ == "__main__":
    main()
