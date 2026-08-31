#!/usr/bin/env python3
"""Create a paper-ready heatmap from Stage-1 locality counters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


HEATMAP_METRICS = (
    ("issue_quotient", "Quotient score"),
    ("l1_cache_line_accesses", "TCP_TOTAL_CACHE_ACCESSES_sum"),
    ("l1_to_l2_read_requests", "TCP_TCC_READ_REQ_sum"),
    ("l2_tag_requests", "TCC_REQ_sum"),
)

CASE_LABELS = {
    "bias_relu": "Bias + ReLU",
    "softmax_bias": "Softmax + bias",
    "embedding_bag": "Embedding bag",
    "gemv": "GEMV",
    "mvt": "MVT",
    "gesummv": "GESUMMV",
    "stencil5": "5-point stencil",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PDF directory (default: REPORT stem plus -plots)",
    )
    return parser.parse_args(argv)


def load_report(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("experiment") != "triton_stage1_locality_counters":
        raise ValueError(f"{path} is not a Triton locality-counter report")
    if not report.get("cases"):
        raise ValueError(f"{path} contains no complete layout pairs")
    return report


def reduction(pair: dict[str, object], metric: str) -> float | None:
    if metric == "issue_quotient":
        return pair["quotient"]["issue"]["reduction"]
    return pair["hardware_comparison"][metric]["reduction"]


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


def _values(report, case_order, cache_mode):
    import numpy as np

    return np.asarray(
        [
            [
                (
                    np.nan
                    if reduction(
                        report["cases"][case]["cache_modes"][cache_mode],
                        metric,
                    )
                    is None
                    else 100.0
                    * reduction(
                        report["cases"][case]["cache_modes"][cache_mode],
                        metric,
                    )
                )
                for metric, _label in HEATMAP_METRICS
            ]
            for case in case_order
        ]
    )


def render_reduction_matrix(report, output: Path) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    font_family = _configure_font(plt)
    cases = report["cases"]
    case_order = [
        case
        for case in report["configuration"].get("case_order", cases)
        if case in cases
    ]
    labels = [CASE_LABELS.get(case, case) for case in case_order]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.8),
        sharey=True,
        layout="constrained",
    )
    color_map = plt.get_cmap("Blues").copy()
    color_map.set_bad("#f2f2f2")
    normalization = Normalize(vmin=0.0, vmax=100.0)
    image = None

    for axis, cache_mode, title in zip(
        axes,
        ("warm", "thrashed"),
        ("Warm", "Cache-thrashed"),
    ):
        values = _values(report, case_order, cache_mode)
        image = axis.pcolormesh(
            np.arange(values.shape[1] + 1),
            np.arange(values.shape[0] + 1),
            np.clip(values, 0.0, 100.0),
            cmap=color_map,
            norm=normalization,
            shading="flat",
            edgecolors="white",
            linewidth=1.6,
        )
        axis.set_xlim(0, values.shape[1])
        axis.set_ylim(values.shape[0], 0)
        axis.set_xticks(
            np.arange(len(HEATMAP_METRICS)) + 0.5,
            [label for _metric, label in HEATMAP_METRICS],
            rotation=27,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(np.arange(len(labels)) + 0.5, labels)
        axis.tick_params(axis="both", labelsize=10.5, length=0)
        axis.set_title(title, fontsize=14, pad=9)
        for spine in axis.spines.values():
            spine.set_visible(False)

        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = "n/a" if not np.isfinite(value) else f"{value:.1f}%"
                color = (
                    "white"
                    if np.isfinite(value) and value >= 55.0
                    else "#1a1a1a"
                )
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=color,
                )

    axes[0].set_ylabel("Kernel", fontsize=12)
    figure.suptitle(
        "LAQS-selected versus Triton default: % change in quotient and counter values",
        fontsize=15.5,
    )
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=axes,
        location="right",
        pad=0.025,
        shrink=0.91,
        aspect=28,
    )
    colorbar.set_label(
        "Reduction relative to Triton default (%)", fontsize=11.5
    )
    colorbar.ax.tick_params(labelsize=10, length=0)
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return font_family


def main() -> None:
    args = parse_arguments()
    cache_dir = Path(tempfile.gettempdir()) / "relay-matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    report_path = args.report.resolve()
    report = load_report(report_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else report_path.with_suffix("").with_name(report_path.stem + "-plots")
    )
    matrix = output_dir / "locality-counter-reductions.pdf"
    font_family = render_reduction_matrix(report, matrix)
    print(f"Wrote {matrix} using {font_family}")


if __name__ == "__main__":
    main()
