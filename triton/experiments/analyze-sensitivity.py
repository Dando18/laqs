#!/usr/bin/env python3
"""Aggregate Experiment 10 reports into a suite CSV and PDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    parser.add_argument(
        "--search-experiment", type=int, choices=(4, 5, 6), default=5
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results"
    )
    parser.add_argument("--plots-root", type=Path, default=ROOT / "plots")
    return parser.parse_args(argv)


def _rows(reports, args):
    rows = []
    other = "matrix" if args.platform == "tuolumne" else "tuolumne"
    for report in reports:
        counterpart_path = (
            args.results_root
            / "experiment-10"
            / other
            / f"grammar-e{args.search_experiment}"
            / f"{report['operator']}--{report['config']}"
            / "report.json"
        )
        counterpart = (
            json.loads(counterpart_path.read_text(encoding="utf-8"))
            if counterpart_path.is_file()
            else None
        )
        common = {
            "experiment": 10,
            "platform": args.platform,
            "operator": report["operator"],
            "config": report["config"],
            "search_experiment": args.search_experiment,
            "status": report["status"],
            "counterpart_status": (
                "missing" if counterpart is None else counterpart.get("status")
            ),
            "cross_platform_eligible": report["status"] == "complete"
            and counterpart is not None
            and counterpart.get("status") == "complete",
        }
        if report["status"] != "complete":
            rows.append(
                {
                    **common,
                    "exclusion_category": report.get("exclusion", {}).get(
                        "category"
                    ),
                    "exclusion_message": report.get("exclusion", {}).get(
                        "message"
                    ),
                }
            )
            continue
        nominal = next(
            trial for trial in report["trials"] if trial["trial_id"] == "nominal"
        )
        nominal_speedup = float(nominal["speedup"])
        for trial in report["trials"]:
            rows.append(
                {
                    **common,
                    "trial_id": trial["trial_id"],
                    "magnitude": trial["magnitude"],
                    "trial_index": trial["trial_index"],
                    "selection_id": trial["selection_id"],
                    "agrees_with_nominal": trial["selection_id"]
                    == nominal["selection_id"],
                    "solve_seconds": trial["solve_seconds"],
                    "baseline_median_ms": trial["timing"]["baseline"][
                        "median_ms"
                    ],
                    "selected_median_ms": trial["timing"]["selected"][
                        "median_ms"
                    ],
                    "speedup": trial["speedup"],
                    "speedup_change_percent": 100.0
                    * (float(trial["speedup"]) / nominal_speedup - 1.0),
                    "tau": json.dumps(trial["tau"], sort_keys=True),
                    "factors": json.dumps(trial["factors"], sort_keys=True),
                }
            )
    return rows


def _plot(path: Path, rows, platform: str, search_experiment: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    complete = [row for row in rows if row.get("speedup") is not None]
    if not complete:
        return
    magnitudes = sorted({float(row["magnitude"]) for row in complete})
    labels = ["Nominal" if value == 0 else f"±{100 * value:.0f}%" for value in magnitudes]
    colors = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00")
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        nominal_rows = sorted(
            (row for row in complete if float(row["magnitude"]) == 0),
            key=lambda row: (row["operator"], row["config"]),
        )
        case_labels = [
            f"{row['operator']}\n{row['config']}" for row in nominal_rows
        ]
        figure, axis = plt.subplots(
            figsize=(max(10.5, len(case_labels) * 0.65), 6.5)
        )
        axis.bar(
            range(len(nominal_rows)),
            [float(row["speedup"]) for row in nominal_rows],
            color=[
                "#0072B2" if float(row["speedup"]) >= 1.0 else "#D55E00"
                for row in nominal_rows
            ],
            edgecolor="black",
            hatch="//",
        )
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_ylim(bottom=0)
        axis.set_xticks(
            range(len(case_labels)), case_labels, rotation=55, ha="right"
        )
        axis.set_ylabel("Speedup over ordinary Triton")
        axis.set_title(
            f"Experiment 10 {platform}: nominal E{search_experiment} performance "
            f"({len(nominal_rows)} completed cases)"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.5, 6.5))
        groups = [
            [float(row["speedup"]) for row in complete if float(row["magnitude"]) == value]
            for value in magnitudes
        ]
        try:
            artists = axis.boxplot(
                groups,
                tick_labels=labels,
                patch_artist=True,
            )
        except TypeError:
            artists = axis.boxplot(groups, labels=labels, patch_artist=True)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_hatch("//")
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Independent tau perturbation bound")
        axis.set_ylabel("Speedup over ordinary Triton")
        axis.set_title(
            f"Experiment 10 {platform}: runtime sensitivity of E{search_experiment} "
            f"selections ({len(nominal_rows)} completed cases)"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        perturbed = [value for value in magnitudes if value > 0]
        agreement = []
        for value in perturbed:
            group = [
                row for row in complete if float(row["magnitude"]) == value
            ]
            agreement.append(
                100.0
                * sum(bool(row["agrees_with_nominal"]) for row in group)
                / len(group)
            )
        figure, axis = plt.subplots(figsize=(10.5, 6.5))
        axis.bar(
            range(len(perturbed)),
            agreement,
            color="#009E73",
            edgecolor="black",
            hatch="..",
        )
        axis.set_xticks(
            range(len(perturbed)),
            [f"±{100 * value:.0f}%" for value in perturbed],
        )
        axis.set_ylim(0, 100)
        axis.set_xlabel("Independent tau perturbation bound")
        axis.set_ylabel("Selections matching nominal (%)")
        axis.set_title(
            f"Experiment 10 {platform}: E{search_experiment} layout-selection "
            f"stability ({len(nominal_rows)} completed cases)"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        changes = [
            row for row in complete if float(row["magnitude"]) > 0
        ]
        figure, axis = plt.subplots(figsize=(10.5, 6.5))
        for index, value in enumerate(perturbed):
            values = [
                float(row["speedup_change_percent"])
                for row in changes
                if float(row["magnitude"]) == value
            ]
            x = [index + 1] * len(values)
            axis.scatter(
                x,
                values,
                marker=("o", "s", "^")[index % 3],
                color=colors[(index + 1) % len(colors)],
                edgecolor="black",
                alpha=0.65,
                label=f"±{100 * value:.0f}%",
            )
            axis.plot(
                (index + 0.8, index + 1.2),
                (statistics.median(values),) * 2,
                color="black",
                linewidth=2,
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(
            range(1, len(perturbed) + 1),
            [f"±{100 * value:.0f}%" for value in perturbed],
        )
        axis.set_xlabel("Independent tau perturbation bound")
        axis.set_ylabel("Speedup change from nominal selection (%)")
        axis.set_title(
            f"Experiment 10 {platform}: E{search_experiment} speedup robustness "
            f"({len(nominal_rows)} completed cases)"
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = arguments()
    suite = (
        args.results_root
        / "experiment-10"
        / args.platform
        / f"grammar-e{args.search_experiment}"
    )
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(suite.glob("*--*/report.json"))
    ]
    if not reports:
        raise SystemExit(f"no reports found under {suite}")
    rows = _rows(reports, args)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = suite / "raw-data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pdf_path = (
        args.plots_root
        / "experiment-10"
        / args.platform
        / f"grammar-e{args.search_experiment}"
        / "summary.pdf"
    )
    _plot(pdf_path, rows, args.platform, args.search_experiment)
    completed = sum(report["status"] == "complete" for report in reports)
    print(f"Cases complete: {completed}/{len(reports)}")
    print(f"Raw data: {csv_path}")
    if pdf_path.is_file():
        print(f"Plot: {pdf_path}")


if __name__ == "__main__":
    main()
