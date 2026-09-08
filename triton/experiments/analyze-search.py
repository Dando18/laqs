#!/usr/bin/env python3
"""Aggregate completed TritonBench search jobs into paper-ready CSV and PDF."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
import math
import statistics
from pathlib import Path

from tau_profiles import TAU_NAMES
from tritonbench_cases import CASES


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=(4, 5, 6), required=True)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--plots-root", type=Path, default=ROOT / "plots")
    parser.add_argument("--tau-name", choices=TAU_NAMES, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    results_root = args.results_root / "tau-profiles" / args.tau_name
    plots_root = args.plots_root / "tau-profiles" / args.tau_name
    suite = results_root / f"experiment-{args.experiment}" / args.platform
    reports = []
    for case in CASES:
        path = suite / case.case_id / "report.json"
        reports.append(json.loads(path.read_text()) if path.exists() else {
            "operator": case.operator, "config": case.config, "description": case.description,
            "status": "missing", "exclusion": {"category": "missing", "message": "No report for a declared case"}})
    suite.mkdir(parents=True, exist_ok=True)
    rows = []
    other_platform = "matrix" if args.platform == "tuolumne" else "tuolumne"
    for report in reports:
        case_id = f"{report['operator']}--{report['config']}"
        counterpart_path = (
            results_root
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
            and report.get("source_identity", {}).get("source_hash") is not None
            and report.get("source_identity", {}).get("source_hash") == counterpart.get("source_identity", {}).get("source_hash")
            and report.get("hardware_profile", {}).get("device", {}).get("tau_semantics")
                == counterpart.get("hardware_profile", {}).get("device", {}).get("tau_semantics")
        )
        row = {
            "experiment": args.experiment,
            "platform": args.platform,
            "operator": report["operator"],
            "config": report["config"],
            "description": report["description"],
            "status": report["status"],
            "stage": report.get("stage"),
            "tau_name": args.tau_name,
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
                "baseline_j_area": report["search"].get("baseline_score", {}).get("hardware_area"),
                "speedup_ci95": report["timing"].get("speedup_ci95"),
                "identity_max_deviation": report["timing"].get("identity_max_deviation"),
                "realization_rejections": report.get("realization", {}).get("rejections"),
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

    common = [row for row in rows if row["cross_platform_eligible"]]
    complete = [row for row in rows if row["status"] == "complete"]
    def summary(panel):
        by_family = defaultdict(list)
        for row in panel:
            by_family[row["operator"]].append(math.log(row["speedup"]))
        return {"cases": len(panel), "families": len(by_family),
            "case_geomean_speedup": math.exp(statistics.fmean(math.log(r["speedup"]) for r in panel)) if panel else None,
            "family_macro_geomean_speedup": math.exp(statistics.fmean(statistics.fmean(v) for v in by_family.values())) if by_family else None}
    coverage = {"declared_cases": len(CASES), "statuses": dict(Counter(r["status"] for r in rows)),
               "device_specific": summary(complete), "common_panel": summary(common),
               "meets_twelve_common_families": len({r["operator"] for r in common}) >= 12,
               "rows": rows}
    (suite / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if complete:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14})
        pdf_path = plots_root / f"experiment-{args.experiment}" / args.platform / "summary.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        labels = [f"{row['operator']}\n{row['config']}" for row in complete]
        with PdfPages(pdf_path) as pdf:
            figure, axis = plt.subplots(figsize=(max(12, len(labels) * 0.8), 6.5))
            colors = ["#0072B2" if float(row["speedup"]) >= 1 else "#D55E00" for row in complete]
            axis.bar(range(len(labels)), [row["speedup"] for row in complete],
                     color=colors, edgecolor="black", hatch="//")
            axis.axhline(1.0, color="black", linewidth=1)
            axis.set_ylim(bottom=0)
            axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
            axis.set_ylabel("Speedup over ordinary Triton")
            axis.set_title(
                f"Experiment {args.experiment} {args.platform} ({args.tau_name} tau): "
                f"all {len(complete)} completed device cases; {len(rows) - len(complete)} unavailable"
            )
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

            counter = "l1_miss_demand_to_l2"
            key = f"reduction_percent:{counter}"
            counter_rows = [row for row in complete if row.get(key) is not None]
            if counter_rows:
                figure, axis = plt.subplots(figsize=(max(12, len(counter_rows) * 0.8), 6.5))
                values = [row[key] for row in counter_rows]
                axis.bar(range(len(counter_rows)), values, color="#009E73",
                         edgecolor="black", hatch="..")
                axis.axhline(0, color="black", linewidth=1)
                axis.set_xticks(range(len(counter_rows)),
                    [f"{row['operator']}\n{row['config']}" for row in counter_rows],
                    rotation=55, ha="right")
                axis.set_ylabel("Reduction from ordinary Triton (%)")
                axis.set_title(
                    f"Experiment {args.experiment} {args.platform} "
                    f"({args.tau_name} tau): L1-miss demand to L2"
                )
                figure.tight_layout()
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
            for start in range(0, len(rows), 16):
                page = rows[start:start + 16]
                figure, axis = plt.subplots(figsize=(14, 8))
                axis.axis("off")
                table = axis.table(cellText=[[f"{r['operator']} / {r['config']}", r["status"], r["counterpart_status"]] for r in page],
                    colLabels=["Declared configuration", args.platform, other_platform], loc="center", cellLoc="left",
                    colWidths=[.6, .2, .2])
                table.auto_set_font_size(False)
                table.set_fontsize(14)
                table.scale(1, 1.7)
                axis.set_title(f"E{args.experiment} / {args.tau_name}: complete case inventory\n"
                               f"{coverage['common_panel']['families']}/12 required common families; missing cells are not speedups")
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        print(f"Plot: {pdf_path}")
    print(f"Raw data: {csv_path}")
    print(
        f"Device-complete: {len(complete)}; cross-platform eligible: {len(common)}; "
        f"excluded/incomplete/missing counterpart: {len(rows) - len(common)}"
    )


if __name__ == "__main__":
    main()
