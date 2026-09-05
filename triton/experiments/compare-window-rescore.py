#!/usr/bin/env python3
"""Compare GEMV baseline and extended-window Spearman correlations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from analyze import correlation_rows


WINDOW = re.compile(
    r"^Q:(?:lane|simd|workgroup)_window\.t(?P<window>\d+)\."
    r"(?:stream|array)\.(?:load|store|atomic)\.(?P<bytes>\d+)B$"
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--extended-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _report(root: Path, k: int, stratification: str) -> Path:
    return (
        root
        / f"k-{k}"
        / "experiment-1"
        / "matrix"
        / f"stratified-{stratification}"
        / "gemv"
        / "report.json"
    )


def _best(rows, counter: str, *, longer_than: int | None = None):
    choices = []
    for row in rows:
        if row["counter"] != counter or row["spearman_rho"] is None:
            continue
        match = WINDOW.match(str(row["predictor"]))
        if match is None:
            continue
        window = int(match.group("window"))
        if longer_than is not None and window <= longer_than:
            continue
        choices.append(row)
    if not choices:
        return None, None
    best = max(choices, key=lambda row: float(row["spearman_rho"]))
    return best["predictor"], float(best["spearman_rho"])


def _j_area(rows, counter: str):
    matches = [
        row
        for row in rows
        if row["counter"] == counter and row["predictor"] == "J_area"
    ]
    return None if not matches else matches[0]["spearman_rho"]


def main() -> None:
    args = parse_arguments()
    records = []
    pattern = "k-*/experiment-1/matrix/stratified-*/gemv/report.json"
    for baseline_path in sorted(args.baseline_root.glob(pattern)):
        k = int(baseline_path.parts[-6].removeprefix("k-"))
        stratification = baseline_path.parts[-3].removeprefix("stratified-")
        extended_path = _report(args.extended_root, k, stratification)
        if not extended_path.is_file():
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        extended = json.loads(extended_path.read_text(encoding="utf-8"))
        baseline_rows = correlation_rows(baseline)
        extended_rows = correlation_rows(extended)
        baseline_limit = max(
            baseline["panel"]["score_profile"]["component_model"]
            .get("scope_basis", {})
            .get("temporal_windows", [4, 16])
        )
        counters = extended["panel"]["score_profile"].get(
            "focus_counters", ()
        )
        for counter in counters:
            baseline_predictor, baseline_rho = _best(
                baseline_rows, counter
            )
            extended_predictor, extended_rho = _best(
                extended_rows, counter
            )
            long_predictor, long_rho = _best(
                extended_rows, counter, longer_than=baseline_limit
            )
            records.append(
                {
                    "k": k,
                    "stratification": stratification,
                    "counter": counter,
                    "baseline_best_predictor": baseline_predictor,
                    "baseline_best_rho": baseline_rho,
                    "extended_best_predictor": extended_predictor,
                    "extended_best_rho": extended_rho,
                    "long_window_best_predictor": long_predictor,
                    "long_window_best_rho": long_rho,
                    "baseline_j_area_rho": _j_area(baseline_rows, counter),
                    "extended_j_area_rho": _j_area(extended_rows, counter),
                }
            )

    if not records:
        raise SystemExit("no matching completed baseline/extended reports")
    fields = tuple(records[0])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, dialect="excel-tab")
    writer.writeheader()
    writer.writerows(records)
    for counter in sorted({record["counter"] for record in records}):
        matching = [record for record in records if record["counter"] == counter]
        baseline = [
            record["baseline_best_rho"]
            for record in matching
            if record["baseline_best_rho"] is not None
        ]
        extended = [
            record["extended_best_rho"]
            for record in matching
            if record["extended_best_rho"] is not None
        ]
        wins = sum(
            record["baseline_best_rho"] is not None
            and record["extended_best_rho"] is not None
            and record["extended_best_rho"] > record["baseline_best_rho"]
            for record in matching
        )
        print(
            f"summary\t{counter}\tpanels={len(matching)}\t"
            f"baseline_median={statistics.median(baseline):.3f}\t"
            f"extended_median={statistics.median(extended):.3f}\t"
            f"improved_panels={wins}"
        )


if __name__ == "__main__":
    main()
