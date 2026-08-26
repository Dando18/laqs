#!/usr/bin/env python3
"""Plot the effect of sparse flag-preserving G_S materialization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from plot_gs_results import (
    REPOSITORY_ROOT,
    SCRIPT_DIRECTORY,
    configure_instance_axis,
    configure_matplotlib,
    load_results,
    save_figure,
)


DEFAULT_INPUT = (
    REPOSITORY_ROOT / "results" / "standard_fiber_scoring_mi300a.jsonl"
)


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="completed flag-fiber G_S summary JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIRECTORY,
        help="directory for PDF and PNG plots (default: script directory)",
    )
    return parser.parse_args(argv)


def render_comparison(records, output_directory: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    if not all(record.get("flag_fiber", {}).get("enabled") for record in records):
        raise ValueError("every record must contain an enabled flag-fiber search")
    base = [
        100.0 * float(record["base_colored_frontier"]["regret"])
        for record in records
    ]
    fiber = [
        100.0 * float(record["frontier"]["regret"])
        for record in records
    ]
    labels = [
        (
            f"{record['base_colored_frontier']['size']}"
            f"→{record['frontier']['size']}"
        )
        for record in records
    ]
    positions = list(range(len(records)))
    width = 0.37
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    axis.bar(
        [position - width / 2 for position in positions],
        base,
        width,
        label=r"Canonical representative + $J_{\mathrm{place}}$",
        color="#4C78A8",
        edgecolor="#202020",
        linewidth=0.6,
        hatch="//",
    )
    fiber_bars = axis.bar(
        [position + width / 2 for position in positions],
        fiber,
        width,
        label=r"One-$T_{ij}$ flag-fiber realization",
        color="#F2A541",
        edgecolor="#202020",
        linewidth=0.6,
        hatch="..",
    )
    axis.axhline(0.0, color="#303030", linewidth=0.8)
    axis.set_ylabel("Time relative to canonical $G_S$ oracle (%)")
    axis.set_title(
        r"Sparse flag-preserving $T_{ij}$ materialization improves all ten selections"
    )
    configure_instance_axis(axis, records)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    lower = min(fiber)
    upper = max(base)
    axis.set_ylim(min(-8.5, lower - 2.0), max(24.0, upper + 2.0))
    for bar, label, value in zip(fiber_bars, labels, fiber):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            -0.65 if value < 0 else value + 0.65,
            label,
            ha="center",
            va="top" if value < 0 else "bottom",
            rotation=90,
            fontsize=7.0,
            color="#4A2C0A",
        )
    axis.text(
        0.995,
        0.965,
        "Negative values beat the canonical oracle; labels show base→fiber frontier size",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#4A4A4A",
    )
    figure.subplots_adjust(left=0.1, right=0.995, bottom=0.25, top=0.9)
    paths = save_figure(
        figure, output_directory, "gs_flag_fiber_comparison"
    )
    plt.close(figure)
    return paths


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    records = load_results(args.input.expanduser().resolve())
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    for path in render_comparison(records, output_directory):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
