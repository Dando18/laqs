#!/usr/bin/env python3
"""Render paper-ready plots for the exhaustive shared-word G_S sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "results" / "standard_scoring_mi300a.jsonl"
KERNEL_ORDER = ("atax", "gemm", "gesummv", "mvt", "syrk")
KERNEL_LABELS = {
    "atax": "ATAX",
    "gemm": "GEMM",
    "gesummv": "GESUMMV",
    "mvt": "MVT",
    "syrk": "SYRK",
}
SIZE_ORDER = (512, 1024)
METHODS = (
    ("frontier", "Pareto frontier"),
    ("fine_gated_5pct_frontier", "5% fine-gated frontier"),
    ("top5_hardware_area", r"Top-5 $J_{\mathrm{area}}$"),
    ("lowest_hardware_area", r"Minimum $J_{\mathrm{area}}$"),
    ("lexicographic_five_cost", "Lexicographic cost"),
)
ORACLE_COLOR = "#3B5B92"
FRONTIER_COLOR = "#D17A22"


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="completed G_S summary JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIRECTORY,
        help="directory for PDF and PNG plots (default: script directory)",
    )
    return parser.parse_args(argv)


def load_results(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if record.get("grammar") != "G_S":
                raise ValueError(f"{path}:{line_number}: expected a G_S record")
            if record.get("complete") is not True:
                raise ValueError(f"{path}:{line_number}: sweep is incomplete")
            if record.get("oracle", {}).get("complete") is not True:
                raise ValueError(f"{path}:{line_number}: oracle is incomplete")
            records.append(record)

    by_key = {
        (str(record["kernel"]), int(record["matrix_size"])): record
        for record in records
    }
    expected = {
        (kernel, size) for kernel in KERNEL_ORDER for size in SIZE_ORDER
    }
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if missing or extra or len(by_key) != len(records):
        raise ValueError(
            "summary must contain exactly one record for every expected "
            f"kernel/size pair; missing={missing}, extra={extra}"
        )
    return [by_key[(kernel, size)] for kernel in KERNEL_ORDER for size in SIZE_ORDER]


def selection_by_name(record: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    selections = record["selection_mechanisms"]
    assert isinstance(selections, list)
    return {
        str(selection["name"]): selection
        for selection in selections
        if isinstance(selection, dict)
    }


def method_regret(record: Mapping[str, object], method: str) -> float:
    if method == "frontier":
        frontier = record["frontier"]
        assert isinstance(frontier, dict)
        value = frontier["regret"]
    else:
        value = selection_by_name(record)[method]["regret"]
    if value is None:
        instance = f"{record['kernel']} N={record['matrix_size']}"
        raise ValueError(f"{instance}: missing regret")
    return 100.0 * float(value)


def configure_matplotlib() -> None:
    cache = Path("/tmp/relay-matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure, output_directory: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, options in (
        (".pdf", {}),
        (".png", {"dpi": 400}),
    ):
        path = output_directory / f"{stem}{suffix}"
        temporary = output_directory / f".{stem}.tmp{suffix}"
        figure.savefig(temporary, format=suffix[1:], **options)
        temporary.replace(path)
        paths.append(path)
    return paths


def configure_instance_axis(axis, records: Sequence[Mapping[str, object]]) -> None:
    axis.set_xticks(range(len(records)))
    axis.set_xticklabels([str(record["matrix_size"]) for record in records])
    for kernel_index, kernel in enumerate(KERNEL_ORDER):
        axis.text(
            2 * kernel_index + 0.5,
            -0.16,
            KERNEL_LABELS[kernel],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            fontweight="bold",
            clip_on=False,
        )
    axis.set_xlabel("Matrix size $N$", labelpad=29)


def render_regret_heatmap(
    records: Sequence[Mapping[str, object]], output_directory: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    values = [
        [method_regret(record, method) for record in records]
        for method, _label in METHODS
    ]
    maximum = max(max(row) for row in values)
    figure, axis = plt.subplots(figsize=(7.2, 2.65))
    image = axis.imshow(
        values,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(50.0, maximum),
        interpolation="nearest",
        aspect="auto",
    )

    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            normalized = value / max(50.0, maximum)
            color = "white" if normalized > 0.53 else "#202020"
            label = "0" if value < 0.05 else f"{value:.1f}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=7.0,
            )

    configure_instance_axis(axis, records)
    axis.set_yticks(range(len(METHODS)))
    axis.set_yticklabels([label for _method, label in METHODS])
    axis.tick_params(axis="both", length=0)
    for boundary in range(2, len(records), 2):
        axis.axvline(boundary - 0.5, color="white", linewidth=2.0)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015, fraction=0.035)
    colorbar.set_label("Oracle regret (%)")
    colorbar.outline.set_linewidth(0.6)
    figure.subplots_adjust(left=0.24, right=0.94, bottom=0.25, top=0.98)
    paths = save_figure(figure, output_directory, "gs_selection_regret")
    plt.close(figure)
    return paths


def render_frontier_speedup(
    records: Sequence[Mapping[str, object]], output_directory: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    oracle_speedups = []
    frontier_speedups = []
    frontier_labels = []
    for record in records:
        selections = selection_by_name(record)
        row_major_ms = float(selections["row_major_baseline"]["best_time_ms"])
        oracle = record["oracle"]
        frontier = record["frontier"]
        assert isinstance(oracle, dict) and isinstance(frontier, dict)
        oracle_speedups.append(row_major_ms / float(oracle["best_time_ms"]))
        frontier_speedups.append(row_major_ms / float(frontier["best_time_ms"]))
        frontier_labels.append(f"{frontier['size']}/{record['layout_count']}")

    positions = list(range(len(records)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 3.05))
    oracle_bars = axis.bar(
        [position - width / 2 for position in positions],
        oracle_speedups,
        width,
        label="Exhaustive oracle",
        color=ORACLE_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    frontier_bars = axis.bar(
        [position + width / 2 for position in positions],
        frontier_speedups,
        width,
        label="Score Pareto frontier",
        color=FRONTIER_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )

    axis.axhline(1.0, color="#555555", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_ylabel("Speedup over row-major ($\times$)")
    configure_instance_axis(axis, records)
    axis.set_ylim(0.0, max(oracle_speedups) * 1.19)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper left")

    for bar, label in zip(frontier_bars, frontier_labels):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.045,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.2,
            color="#5A3514",
        )
    axis.text(
        0.995,
        0.985,
        "Labels: frontier layouts / all layouts",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.25, top=0.98)
    paths = save_figure(figure, output_directory, "gs_frontier_speedup")
    plt.close(figure)
    return paths


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    input_path = args.input.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records = load_results(input_path)
    configure_matplotlib()
    paths = [
        *render_regret_heatmap(records, output_directory),
        *render_frontier_speedup(records, output_directory),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
