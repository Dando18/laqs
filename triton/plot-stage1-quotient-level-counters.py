#!/usr/bin/env python3
"""Plot quotient level against steady-state TCP cache-line accesses."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="output PDF (default: REPORT stem plus -tcp.pdf)",
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


def render(report: dict[str, object], output: Path) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    font = _configure_font(plt)
    candidates = sorted(
        (candidate for candidate in report["candidates"] if candidate["complete"]),
        key=lambda candidate: candidate["quotient_score"],
    )
    x = np.asarray([candidate["quotient_score"] for candidate in candidates])
    launch_values = [
        candidate["counters"]["steady_state"][
            "l1_cache_line_accesses_by_launch"
        ]
        for candidate in candidates
    ]
    y = np.asarray(
        [
            candidate["counters"]["steady_state"]["l1_cache_line_accesses"]
            for candidate in candidates
        ]
    )
    y_min = np.asarray([min(values) for values in launch_values])
    y_max = np.asarray([max(values) for values in launch_values])
    rho = report["statistics"]["tie_aware_spearman"]

    figure, axis = plt.subplots(figsize=(8.0, 5.4), layout="constrained")
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
    x_upper, x_ticks = _nice_ticks(float(max(x)), intervals=6)
    y_upper, y_ticks = _nice_ticks(float(max(y)), intervals=5)
    axis.set_xlim(0.0, x_upper)
    axis.set_ylim(0.0, y_upper)
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    axis.set_xlabel("Quotient Score", fontsize=13)
    axis.set_ylabel(
        "Median First-Level Cache Accesses per Kernel",
        fontsize=13,
    )
    axis.set_title(
        "Quotient Score versus First-level Cache Accesses "
        f"for {report['case'].upper()} Kernel",
        fontsize=14,
        pad=10,
    )
    rho_text = "n/a" if rho is None else f"{rho:.3f}"
    axis.text(
        0.03,
        0.97,
        f"Spearman $\\rho$ = {rho_text}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    axis.grid(axis="both", color="#dddddd", linewidth=0.7, zorder=0)
    axis.tick_params(labelsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    if np.any(ranges > 0.0):
        axis.legend(frameon=False, loc="lower right", fontsize=10.5)

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
        else report_path.with_name(report_path.stem + "-tcp.pdf")
    )
    font = render(report, output)
    print(f"Wrote {output} using {font}")


if __name__ == "__main__":
    main()
