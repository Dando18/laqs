#!/usr/bin/env python3
"""Aggregate Experiment 12 reports into a suite CSV and PDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument(
        "--search-experiment", type=int, choices=(4, 5, 6), default=4
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results"
    )
    parser.add_argument("--plots-root", type=Path, default=ROOT / "plots")
    return parser.parse_args(argv)


def _row(report, counterpart_status: str):
    row = {
        "experiment": 12,
        "platform": report["platform"],
        "operator": report["operator"],
        "config": report["config"],
        "description": report["description"],
        "search_experiment": report["search_experiment"],
        "status": report["status"],
        "counterpart_status": counterpart_status,
        "cross_platform_eligible": report["status"] == "complete"
        and counterpart_status == "complete",
    }
    if report["status"] != "complete":
        row.update(
            {
                "exclusion_category": report.get("exclusion", {}).get("category"),
                "exclusion_message": report.get("exclusion", {}).get("message"),
            }
        )
        return row
    row.update(
        {
            "trace_capture_seconds": report["trace_capture_seconds"],
            "graph_construction_median_seconds": report["graph_construction"][
                "median_seconds"
            ],
            "graph_construction_min_seconds": report["graph_construction"][
                "min_seconds"
            ],
            "quotient_score_median_seconds": report["quotient_score"][
                "median_seconds"
            ],
            "quotient_score_min_seconds": report["quotient_score"]["min_seconds"],
            "solve_median_seconds": report["solve"]["median_seconds"],
            "solve_min_seconds": report["solve"]["min_seconds"],
            **report["trace"],
            "optimized_array_count": report["selection"]["optimized_array_count"],
            "transformed_array_count": report["selection"][
                "transformed_array_count"
            ],
            "selected_j_area": report["selection"]["score"]["hardware_area"],
        }
    )
    return row


def _plot(path: Path, rows, platform: str, search_experiment: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    complete = [row for row in rows if row["status"] == "complete"]
    if not complete:
        return
    labels = [f"{row['operator']}\n{row['config']}" for row in complete]
    phases = (
        (
            "graph_construction_median_seconds",
            "Universal graph construction (ms)",
            "#0072B2",
            "//",
        ),
        (
            "quotient_score_median_seconds",
            "One-layout quotient score (ms)",
            "#009E73",
            "..",
        ),
        (
            "solve_median_seconds",
            "Exact solve time (s)",
            "#E69F00",
            "xx",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for key, ylabel, color, hatch in phases:
            scale = 1_000.0 if key != "solve_median_seconds" else 1.0
            values = [scale * float(row[key]) for row in complete]
            figure, axis = plt.subplots(
                figsize=(max(12.0, len(labels) * 0.5), 6.8)
            )
            axis.bar(
                range(len(labels)),
                values,
                color=color,
                edgecolor="black",
                hatch=hatch,
            )
            axis.set_ylim(bottom=0)
            axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(
                f"Experiment 12 {platform}: E{search_experiment} {ylabel.lower()} "
                f"({len(complete)} completed cases)"
            )
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    args = arguments()
    suite = (
        args.results_root
        / "experiment-12"
        / args.platform
        / f"grammar-e{args.search_experiment}"
    )
    paths = sorted(suite.glob("*--*/report.json"))
    if not paths:
        raise SystemExit(f"no reports found under {suite}")
    other = "matrix" if args.platform == "tuolumne" else "tuolumne"
    rows = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        counterpart_path = (
            args.results_root
            / "experiment-12"
            / other
            / f"grammar-e{args.search_experiment}"
            / path.parent.name
            / "report.json"
        )
        counterpart = (
            json.loads(counterpart_path.read_text(encoding="utf-8"))
            if counterpart_path.is_file()
            else None
        )
        rows.append(
            _row(
                report,
                "missing" if counterpart is None else counterpart.get("status"),
            )
        )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = suite / "raw-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pdf_path = (
        args.plots_root
        / "experiment-12"
        / args.platform
        / f"grammar-e{args.search_experiment}"
        / "summary.pdf"
    )
    _plot(pdf_path, rows, args.platform, args.search_experiment)
    completed = sum(row["status"] == "complete" for row in rows)
    print(f"Cases complete: {completed}/{len(rows)}")
    print(f"Raw data: {csv_path}")
    if pdf_path.is_file():
        print(f"Plot: {pdf_path}")


if __name__ == "__main__":
    main()
