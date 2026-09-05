#!/usr/bin/env python3
"""Aggregate completed TritonBench search jobs into paper-ready CSV and PDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(4, 5, 6), required=True)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--plots-root", type=Path, default=ROOT / "plots")
    return parser.parse_args()


def main():
    args = arguments()
    suite = args.results_root / f"experiment-{args.experiment}" / args.platform
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(suite.glob("*--*/report.json"))
    ]
    if not reports:
        raise SystemExit(f"no reports found under {suite}")
    rows = []
    other_platform = "matrix" if args.platform == "tuolumne" else "tuolumne"
    for report in reports:
        case_id = f"{report['operator']}--{report['config']}"
        counterpart_path = (
            args.results_root
            / f"experiment-{args.experiment}"
            / other_platform
            / case_id
            / "report.json"
        )
        counterpart = (
            json.loads(counterpart_path.read_text(encoding="utf-8"))
            if counterpart_path.is_file()
            else None
        )
        cross_platform = (
            report["status"] == "complete"
            and counterpart is not None
            and counterpart.get("status") == "complete"
        )
        row = {
            "experiment": args.experiment,
            "platform": args.platform,
            "operator": report["operator"],
            "config": report["config"],
            "description": report["description"],
            "status": report["status"],
            "counterpart_status": (
                "missing" if counterpart is None else counterpart.get("status")
            ),
            "cross_platform_eligible": cross_platform,
            "exclusion_category": (report.get("exclusion") or {}).get("category"),
            "exclusion_message": (report.get("exclusion") or {}).get("message"),
        }
        if report["status"] == "complete":
            row.update({
                "baseline_median_ms": report["timing"]["baseline"]["median_ms"],
                "selected_median_ms": report["timing"]["selected"]["median_ms"],
                "speedup": report["timing"]["speedup"],
                "analysis_seconds": report["analysis_seconds"],
                "search_seconds": report["search"].get("elapsed_seconds"),
                "optimized_array_count": report["search"]["optimized_array_count"],
                "transformed_array_count": report["search"]["transformed_array_count"],
                "selected_j_area": report["search"]["score"]["hardware_area"],
            })
            row.update({
                f"selected_Q:{component['name']}": component["raw_region_count"]
                for component in report["search"]["score"]["components"]
            })
            if report["counters"] is not None:
                for layout in ("baseline", "selected"):
                    summary = report["counters"][layout]["steady_state"]
                    row.update({
                        f"{layout}_counter:{name}": value
                        for name, value in summary.items()
                        if isinstance(value, (int, float))
                    })
            row.update({
                f"reduction_percent:{name}": value
                for name, value in report["counter_reductions_percent"].items()
            })
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = suite / "raw-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    complete = [row for row in rows if row["cross_platform_eligible"]]
    if complete:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = args.plots_root / f"experiment-{args.experiment}" / args.platform / "summary.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        labels = [f"{row['operator']}\n{row['config']}" for row in complete]
        with PdfPages(pdf_path) as pdf:
            figure, axis = plt.subplots(figsize=(max(12, len(labels) * 0.48), 6.5))
            colors = ["#0072B2" if float(row["speedup"]) >= 1 else "#D55E00" for row in complete]
            axis.bar(range(len(labels)), [row["speedup"] for row in complete],
                     color=colors, edgecolor="black", hatch="//")
            axis.axhline(1.0, color="black", linewidth=1)
            axis.set_ylim(bottom=0)
            axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
            axis.set_ylabel("Speedup over ordinary Triton")
            axis.set_title(f"Experiment {args.experiment} {args.platform}: selected-layout runtime")
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

            counter = "l1_miss_demand_to_l2"
            key = f"reduction_percent:{counter}"
            counter_rows = [row for row in complete if row.get(key) is not None]
            if counter_rows:
                figure, axis = plt.subplots(figsize=(max(12, len(counter_rows) * 0.48), 6.5))
                values = [row[key] for row in counter_rows]
                axis.bar(range(len(counter_rows)), values, color="#009E73",
                         edgecolor="black", hatch="..")
                axis.axhline(0, color="black", linewidth=1)
                axis.set_xticks(range(len(counter_rows)),
                    [f"{row['operator']}\n{row['config']}" for row in counter_rows],
                    rotation=55, ha="right")
                axis.set_ylabel("Reduction from ordinary Triton (%)")
                axis.set_title(
                    f"Experiment {args.experiment} {args.platform}: L1-miss demand to L2"
                )
                figure.tight_layout()
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        print(f"Plot: {pdf_path}")
    print(f"Raw data: {csv_path}")
    print(
        f"Cross-platform eligible: {len(complete)}; "
        f"excluded/incomplete/missing counterpart: {len(rows) - len(complete)}"
    )


if __name__ == "__main__":
    main()
