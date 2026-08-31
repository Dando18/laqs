#!/usr/bin/env python3
"""Scatter persistent-layout quotient scores against a hardware counter."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from stage1_counter_analysis import SUMMARY_METRICS, rank_correlation


METRIC_LABELS = {
    "l1_cache_line_accesses": "First-Level Cache Accesses per Kernel",
    "l1_to_l2_read_requests": "TCP_TCC_READ_REQ_sum per dispatch",
    "l1_to_l2_write_requests": "TCP_TCC_WRITE_REQ_sum per dispatch",
    "l1_to_l2_total_requests": "TCP-to-TCC requests per dispatch",
    "l2_tag_requests": "TCC_REQ_sum per dispatch",
    "l2_hits": "TCC_HIT_sum per dispatch",
    "l2_misses": "TCC_MISS_sum per dispatch",
    "hbm_read_bytes": "HBM read bytes per dispatch",
    "hbm_write_bytes": "HBM write bytes per dispatch",
    "hbm_total_bytes": "HBM bytes per dispatch",
    "duration_ns": "Profiled duration (ns)",
    "hbm_bandwidth_gbps": "Achieved HBM bandwidth (GB/s)",
    "l2_hit_rate_percent": "L2 hit rate (%)",
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--metric", choices=SUMMARY_METRICS, default="l1_cache_line_accesses"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PDF (default: REPORT stem plus -METRIC.pdf)",
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
    if report.get("experiment") != "triton_stage1_layout_counter_scatter":
        raise ValueError(f"{path} is not a layout-counter scatter report")
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


def render(report: dict[str, object], metric: str, output: Path) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    font = _configure_font(plt)
    candidates = [
        candidate
        for candidate in report["candidates"]
        if candidate["complete"]
        and candidate["counters"]["steady_state"][metric] is not None
    ]
    if len(candidates) < 2:
        total = len(report["candidates"])
        pending = len(report.get("missing_profiles", ()))
        case = report["case"]
        raise ValueError(
            "scatter report has only "
            f"{len(candidates)} of {total} profiled mappings ({pending} "
            "profiles pending); continue it with "
            "triton/run-stage1-layout-counter-scatter.py "
            f"--case {case} --max-profiles 6"
        )
    tiles = sorted(
        {tuple(candidate["inner_tile_shape"]) for candidate in candidates}
    )
    colors = plt.get_cmap("Dark2")
    markers = ("o", "s", "^", "D", "P", "v", "X", "<", ">")
    figure, axis = plt.subplots(figsize=(8.2, 5.6), layout="constrained")
    for index, tile in enumerate(tiles):
        members = [
            candidate
            for candidate in candidates
            if tuple(candidate["inner_tile_shape"]) == tile
        ]
        axis.scatter(
            [candidate["quotient_score"] for candidate in members],
            [candidate["counters"]["steady_state"][metric] for candidate in members],
            s=68,
            marker=markers[index % len(markers)],
            color=colors(index % colors.N),
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85,
            label="×".join(str(value) for value in tile),
            zorder=3,
        )
    x = [float(candidate["quotient_score"]) for candidate in candidates]
    y = [
        float(candidate["counters"]["steady_state"][metric])
        for candidate in candidates
    ]
    rho = rank_correlation(x, y)
    rho_text = "n/a" if rho is None else f"{rho:.3f}"
    axis.text(
        0.03,
        0.97,
        f"Spearman $\\rho$ = {rho_text}\n$n$ = {len(candidates)} mappings",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    x_upper, x_ticks = _nice_ticks(max(x), intervals=6)
    y_upper, y_ticks = _nice_ticks(max(y), intervals=5)
    axis.set_xlim(0.0, x_upper)
    axis.set_ylim(0.0, y_upper)
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    axis.set_xlabel("Issue quotient score", fontsize=13)
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=13)
    axis.set_title(
        f"{report['case'].upper()}: persistent tile layouts versus memory traffic",
        fontsize=14,
        pad=10,
    )
    axis.grid(axis="both", color="#dddddd", linewidth=0.7, zorder=0)
    axis.tick_params(labelsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        title="Inner tile",
        frameon=False,
        fontsize=9.5,
        title_fontsize=10,
        ncols=2,
        loc="lower right",
    )
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
        else report_path.with_name(report_path.stem + f"-{args.metric}.pdf")
    )
    font = render(report, args.metric, output)
    print(f"Wrote {output} using {font}")


if __name__ == "__main__":
    main()
