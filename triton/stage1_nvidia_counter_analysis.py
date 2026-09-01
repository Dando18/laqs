"""Parse and aggregate NVIDIA Nsight Compute Stage-1 counters."""

from __future__ import annotations

import csv
from math import isfinite
from pathlib import Path
import statistics
from typing import Mapping, Sequence


DURATION_METRIC = "gpu__time_duration.sum"
SUMMARY_METRICS = (
    "l1_cache_line_accesses",
    "duration_ns",
)


def counter_definitions(metric: str) -> dict[str, str]:
    return {
        metric: (
            "Nsight Compute sum of 32-byte L1TEX sectors requested by "
            "global load instructions"
        ),
        DURATION_METRIC: "Nsight Compute kernel duration in nanoseconds",
        "l1_cache_line_accesses": (
            f"analysis alias for {metric}; values are sectors when the "
            "H100 fallback metric is selected"
        ),
    }


def _number(record: Mapping[str, str], field: str, path: Path) -> float:
    value = record.get(field, "").replace(",", "").strip()
    if not value or value.lower() == "n/a":
        raise ValueError(f"{path} has an empty profiler field {field!r}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"{path} has invalid {field!r} value {value!r}"
        ) from error
    if not isfinite(parsed):
        raise ValueError(f"{path} has non-finite {field!r} value {value!r}")
    return parsed


def _records(path: Path, metric: str) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = None
    for index, line in enumerate(lines):
        fields = next(csv.reader((line,)), ())
        if "Kernel Name" in fields and metric in fields:
            header = index
            break
    if header is None:
        raise ValueError(
            f"{path} has no Nsight Compute raw-page header for {metric}"
        )
    return list(csv.DictReader(lines[header:]))


def _dispatch(
    record: Mapping[str, str], path: Path, metric: str
) -> dict[str, object]:
    return {
        "index": int(record["ID"]),
        "kernel_name": record["Kernel Name"],
        "duration_ns": _number(record, DURATION_METRIC, path),
        "l1_cache_line_accesses": _number(record, metric, path),
        "primary_counter": metric,
        "counters": {
            metric: _number(record, metric, path),
            DURATION_METRIC: _number(record, DURATION_METRIC, path),
        },
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def summarize_dispatches(
    dispatches: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not dispatches:
        raise ValueError("cannot summarize an empty dispatch population")
    summary = {
        field: _median([float(dispatch[field]) for dispatch in dispatches])
        for field in SUMMARY_METRICS
    }
    summary["dispatch_count"] = len(dispatches)
    return summary


def parse_counter_csv(
    path: Path,
    *,
    kernel_name: str,
    profile_iterations: int,
    metric: str,
) -> dict[str, object]:
    """Return the final steady-state target launches from ncu raw CSV."""

    records = [
        record
        for record in _records(path, metric)
        if kernel_name in record.get("Kernel Name", "")
    ]
    records.sort(key=lambda record: int(record["ID"]))
    if len(records) < profile_iterations:
        raise ValueError(
            f"{path} contains {len(records)} {kernel_name} launches; "
            f"expected at least {profile_iterations}"
        )
    dispatches = [
        _dispatch(record, path, metric)
        for record in records[-profile_iterations:]
    ]
    return {
        "target_dispatch_count": len(records),
        "steady_state": summarize_dispatches(dispatches),
        "steady_dispatches": dispatches,
    }


def aggregate_profiles(
    profiles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate launches by their within-launch and cross-launch medians."""

    if not profiles:
        raise ValueError("counter aggregation requires at least one profile")
    launch_summaries = [profile["steady_state"] for profile in profiles]
    aggregate = {}
    for field in SUMMARY_METRICS:
        values = [float(summary[field]) for summary in launch_summaries]
        aggregate[field] = _median(values)
        aggregate[f"{field}_by_launch"] = values
    aggregate["profile_launch_count"] = len(profiles)
    aggregate["dispatches_per_launch"] = [
        int(summary["dispatch_count"]) for summary in launch_summaries
    ]
    return {"steady_state": aggregate, "profiles": list(profiles)}
