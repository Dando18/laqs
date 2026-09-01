"""Parse and compare MI300A counters for Triton Stage-1 layouts."""

from __future__ import annotations

import csv
from math import isfinite
from pathlib import Path
import statistics
from typing import Mapping, Sequence


COUNTER_FIELDS = (
    "TCP_TOTAL_CACHE_ACCESSES_sum",
    "TCP_TOTAL_READ_sum",
    "TCP_TCC_READ_REQ_sum",
    "TCP_TCC_WRITE_REQ_sum",
    "TCC_REQ_sum",
    "TCC_READ_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "FETCH_SIZE",
    "WRITE_SIZE",
)

COUNT_METRICS = (
    "l1_cache_line_accesses",
    "first_level_read_events",
    "l1_to_l2_read_requests",
    "l1_to_l2_write_requests",
    "l1_to_l2_total_requests",
    "l2_tag_requests",
    "second_level_read_requests",
    "l2_hits",
    "l2_misses",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbm_total_bytes",
)

SUMMARY_METRICS = (
    *COUNT_METRICS,
    "duration_ns",
    "hbm_bandwidth_gbps",
    "l2_hit_rate_percent",
)

COUNTER_DEFINITIONS = {
    "TCP_TOTAL_CACHE_ACCESSES_sum": (
        "64-byte vector L1 data-cache accesses, including hits and misses"
    ),
    "TCP_TOTAL_READ_sum": "read pixels or buffers received by TCP from TA",
    "TCP_TCC_READ_REQ_sum": "read requests sent from TCP/L1 to TCC/L2",
    "TCP_TCC_WRITE_REQ_sum": "write requests sent from TCP/L1 to TCC/L2",
    "TCC_REQ_sum": "requests processed by the TCC/L2 tag blocks",
    "TCC_READ_sum": (
        "TCC read requests, including compressed reads but excluding metadata"
    ),
    "TCC_HIT_sum": "TCC/L2 cache hits",
    "TCC_MISS_sum": "TCC/L2 cache misses",
    "FETCH_SIZE": "KiB fetched from memory",
    "WRITE_SIZE": "KiB written to memory",
}


def _number(record: Mapping[str, str], field: str, path: Path) -> float:
    value = record.get(field, "")
    if value == "":
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


def _dispatch(record: Mapping[str, str], path: Path) -> dict[str, object]:
    counters = {field: _number(record, field, path) for field in COUNTER_FIELDS}
    duration_ns = _number(record, "EndNs", path) - _number(
        record, "BeginNs", path
    )
    if duration_ns <= 0.0:
        raise ValueError(f"{path} has a non-positive dispatch duration")
    hits = counters["TCC_HIT_sum"]
    misses = counters["TCC_MISS_sum"]
    hbm_read_bytes = counters["FETCH_SIZE"] * 1024.0
    hbm_write_bytes = counters["WRITE_SIZE"] * 1024.0
    hbm_total_bytes = hbm_read_bytes + hbm_write_bytes
    return {
        "index": int(record["Index"]),
        "kernel_name": record["KernelName"],
        "duration_ns": duration_ns,
        "l1_cache_line_accesses": counters[
            "TCP_TOTAL_CACHE_ACCESSES_sum"
        ],
        "first_level_read_events": counters["TCP_TOTAL_READ_sum"],
        "l1_to_l2_read_requests": counters["TCP_TCC_READ_REQ_sum"],
        "l1_to_l2_write_requests": counters["TCP_TCC_WRITE_REQ_sum"],
        "l1_to_l2_total_requests": (
            counters["TCP_TCC_READ_REQ_sum"]
            + counters["TCP_TCC_WRITE_REQ_sum"]
        ),
        "l2_tag_requests": counters["TCC_REQ_sum"],
        "second_level_read_requests": counters["TCC_READ_sum"],
        "l2_hits": hits,
        "l2_misses": misses,
        "l2_hit_rate_percent": (
            100.0 * hits / (hits + misses) if hits + misses else None
        ),
        "hbm_read_bytes": hbm_read_bytes,
        "hbm_write_bytes": hbm_write_bytes,
        "hbm_total_bytes": hbm_total_bytes,
        "hbm_bandwidth_gbps": hbm_total_bytes / duration_ns,
        "counters": counters,
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
        if field != "l2_hit_rate_percent"
    }
    hit_rates = [
        float(dispatch["l2_hit_rate_percent"])
        for dispatch in dispatches
        if dispatch["l2_hit_rate_percent"] is not None
    ]
    summary["l2_hit_rate_percent"] = _median(hit_rates) if hit_rates else None
    summary["dispatch_count"] = len(dispatches)
    return summary


def parse_counter_csv(
    path: Path, *, kernel_name: str, profile_iterations: int
) -> dict[str, object]:
    """Return the final steady-state target dispatches from rocprof v1 CSV."""

    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"{path} contains no profiler records")
    required = {"Index", "KernelName", "BeginNs", "EndNs", *COUNTER_FIELDS}
    missing = sorted(required - set(records[0]))
    if missing:
        raise ValueError(f"{path} is missing profiler columns: {missing}")
    records = [
        record
        for record in records
        if kernel_name in record.get("KernelName", "")
    ]
    records.sort(key=lambda record: int(record["Index"]))
    if len(records) < profile_iterations:
        raise ValueError(
            f"{path} contains {len(records)} {kernel_name} dispatches; "
            f"expected at least {profile_iterations}"
        )
    dispatches = [
        _dispatch(record, path) for record in records[-profile_iterations:]
    ]
    return {
        "target_dispatch_count": len(records),
        "steady_state": summarize_dispatches(dispatches),
        "steady_dispatches": dispatches,
    }


def aggregate_profiles(
    profiles: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate independent profiler launches by their within-launch medians."""

    if not profiles:
        raise ValueError("counter aggregation requires at least one profile")
    launch_summaries = [profile["steady_state"] for profile in profiles]
    aggregate = {}
    for field in SUMMARY_METRICS:
        values = [
            float(summary[field])
            for summary in launch_summaries
            if summary[field] is not None
        ]
        aggregate[field] = _median(values) if values else None
        aggregate[f"{field}_by_launch"] = values
    aggregate["profile_launch_count"] = len(profiles)
    aggregate["dispatches_per_launch"] = [
        int(summary["dispatch_count"]) for summary in launch_summaries
    ]
    return {"steady_state": aggregate, "profiles": list(profiles)}


def _reduction(baseline: float, selected: float) -> float | None:
    if baseline == 0.0:
        return None
    return 1.0 - selected / baseline


def compare_summaries(
    baseline: Mapping[str, object], selected: Mapping[str, object]
) -> dict[str, object]:
    comparisons = {}
    for field in (*COUNT_METRICS, "duration_ns", "hbm_bandwidth_gbps"):
        baseline_value = float(baseline[field])
        selected_value = float(selected[field])
        comparisons[field] = {
            "default": baseline_value,
            "selected": selected_value,
            "selected_to_default_ratio": (
                selected_value / baseline_value if baseline_value else None
            ),
            "reduction": _reduction(baseline_value, selected_value),
            "fewer": selected_value < baseline_value,
        }
    baseline_hit_rate = baseline["l2_hit_rate_percent"]
    selected_hit_rate = selected["l2_hit_rate_percent"]
    comparisons["l2_hit_rate_percent"] = {
        "default": baseline_hit_rate,
        "selected": selected_hit_rate,
        "percentage_point_change": (
            float(selected_hit_rate) - float(baseline_hit_rate)
            if baseline_hit_rate is not None and selected_hit_rate is not None
            else None
        ),
    }
    return comparisons


def rank_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Spearman correlation with average ranks for ties."""

    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = 0.5 * (start + 1 + end)
            for position in range(start, end):
                result[order[position]] = rank
            start = end
        return result

    return float(statistics.correlation(ranks(left), ranks(right)))


def linear_fit(
    predictor: Sequence[float], response: Sequence[float]
) -> dict[str, float | None]:
    """Fit ``response = intercept + slope * predictor`` by least squares."""

    if len(predictor) != len(response):
        raise ValueError("linear-fit inputs must have equal lengths")
    if len(predictor) < 2:
        raise ValueError("linear fit requires at least two observations")
    x = [float(value) for value in predictor]
    y = [float(value) for value in response]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    x_variation = sum((value - x_mean) ** 2 for value in x)
    if x_variation == 0.0:
        raise ValueError("linear fit requires varying predictor values")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    ) / x_variation
    intercept = y_mean - slope * x_mean
    residual = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x, y)
    )
    total = sum((value - y_mean) ** 2 for value in y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total) if total else None,
    }
