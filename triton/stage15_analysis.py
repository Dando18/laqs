"""Aggregation and ROCm counter analysis for the Stage 1.5 GEMM study."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics


COUNTER_FIELDS = (
    "TCP_TCC_READ_REQ_sum",
    "TCC_REQ_sum",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
    "FETCH_SIZE",
    "WRITE_SIZE",
    "MemUnitStalled",
    "TCP_TCP_TA_DATA_STALL_CYCLES_sum",
    "MfmaUtil",
    "TOTAL_16_OPS",
)


def timing_summary(samples_ms: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "stdev_ms": (
            statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
        ),
        "samples_ms": samples_ms,
    }


def aggregate_rankings(results: list[dict[str, object]]) -> dict[str, object]:
    from relay import summarize_rank_quality

    if not results:
        raise ValueError("Stage 1.5 aggregation requires at least one result")
    first = results[0]
    candidate_ids = [
        str(candidate["candidate_id"]) for candidate in first["candidates"]
    ]
    for result in results[1:]:
        observed = [
            str(candidate["candidate_id"]) for candidate in result["candidates"]
        ]
        if observed != candidate_ids:
            raise ValueError(
                "retained GEMM candidates changed between process launches"
            )

    candidates = []
    for index, candidate_id in enumerate(candidate_ids):
        process_candidates = [result["candidates"][index] for result in results]
        process_medians = [
            float(candidate["timing"]["median_ms"])
            for candidate in process_candidates
        ]
        all_samples = [
            float(sample)
            for candidate in process_candidates
            for sample in candidate["timing"]["samples_ms"]
        ]
        codegen = [candidate["compiled_codegen"] for candidate in process_candidates]
        candidate = {
            key: value
            for key, value in process_candidates[0].items()
            if key not in {"compiled_codegen", "runtime_ms", "timing"}
        }
        candidate["runtime_ms"] = statistics.median(process_medians)
        candidate["timing"] = {
            **timing_summary(all_samples),
            "aggregation_runtime_ms": statistics.median(process_medians),
            "aggregation_method": "median_of_process_medians",
            "process_medians_ms": process_medians,
            "process_launch_count": len(process_medians),
        }
        candidate["compiled_codegen"] = codegen[0]
        candidate["compiled_codegen_consistent"] = all(
            item == codegen[0] for item in codegen[1:]
        )
        if not candidate["compiled_codegen_consistent"]:
            candidate["compiled_codegen_by_process"] = codegen
        candidates.append(candidate)

    rank_quality = summarize_rank_quality(candidates)
    default = next(
        candidate for candidate in candidates if candidate["layout"] == "row_major"
    )
    selected = candidates[0]
    process_speedups = []
    for result in results:
        process_default = next(
            candidate
            for candidate in result["candidates"]
            if candidate["layout"] == "row_major"
        )
        process_selected = result["candidates"][0]
        process_speedups.append(
            float(process_default["runtime_ms"])
            / float(process_selected["runtime_ms"])
        )
    return {
        "process_launch_count": len(results),
        "process_ids": [result["process"]["pid"] for result in results],
        "process_speedups": process_speedups,
        "default_candidate_id": default["candidate_id"],
        "selected_candidate_id": selected["candidate_id"],
        "default": default,
        "selected": selected,
        "candidates": candidates,
        "rank_quality": rank_quality,
        "measured_speedup": float(default["runtime_ms"])
        / float(selected["runtime_ms"]),
        "correct": all(bool(result["correct"]) for result in results),
    }


def _median(records, field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def parse_counter_csv(
    path: Path, *, profile_iterations: int
) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    records = [
        record
        for record in records
        if "gemm_prepacked_b_kernel" in record.get("KernelName", "")
    ]
    records.sort(key=lambda record: int(record["Index"]))
    if len(records) < profile_iterations:
        raise ValueError(
            f"{path} contains {len(records)} GEMM dispatches; "
            f"expected at least {profile_iterations}"
        )
    records = records[-profile_iterations:]
    missing = [field for field in COUNTER_FIELDS if not records[0].get(field)]
    if missing:
        raise ValueError(f"{path} is missing profiler counters: {missing}")

    dispatches = []
    for record in records:
        duration_ns = float(record["EndNs"]) - float(record["BeginNs"])
        counters = {field: float(record[field]) for field in COUNTER_FIELDS}
        hits = counters["TCC_HIT_sum"]
        misses = counters["TCC_MISS_sum"]
        read_bytes = counters["FETCH_SIZE"] * 1024.0
        write_bytes = counters["WRITE_SIZE"] * 1024.0
        dispatches.append(
            {
                "index": int(record["Index"]),
                "duration_ns": duration_ns,
                "l2_hit_rate_percent": 100.0 * hits / (hits + misses),
                "hbm_read_bytes": read_bytes,
                "hbm_write_bytes": write_bytes,
                "hbm_bandwidth_gbps": (read_bytes + write_bytes) / duration_ns,
                "achieved_fp16_tops": counters["TOTAL_16_OPS"]
                / duration_ns
                / 1000.0,
                "counters": counters,
            }
        )

    summary = {
        "profiled_dispatch_count": len(dispatches),
        "duration_ns": _median(dispatches, "duration_ns"),
        "l1_to_l2_read_requests": _median(
            [dispatch["counters"] for dispatch in dispatches],
            "TCP_TCC_READ_REQ_sum",
        ),
        "l2_tag_requests": _median(
            [dispatch["counters"] for dispatch in dispatches], "TCC_REQ_sum"
        ),
        "l2_hits": _median(
            [dispatch["counters"] for dispatch in dispatches], "TCC_HIT_sum"
        ),
        "l2_misses": _median(
            [dispatch["counters"] for dispatch in dispatches], "TCC_MISS_sum"
        ),
        "l2_hit_rate_percent": _median(dispatches, "l2_hit_rate_percent"),
        "hbm_read_bytes": _median(dispatches, "hbm_read_bytes"),
        "hbm_write_bytes": _median(dispatches, "hbm_write_bytes"),
        "hbm_bandwidth_gbps": _median(dispatches, "hbm_bandwidth_gbps"),
        "memory_unit_stalled_percent": _median(
            [dispatch["counters"] for dispatch in dispatches], "MemUnitStalled"
        ),
        "memory_unit_stall_cycles_sum": _median(
            [dispatch["counters"] for dispatch in dispatches],
            "TCP_TCP_TA_DATA_STALL_CYCLES_sum",
        ),
        "mfma_util_percent": _median(
            [dispatch["counters"] for dispatch in dispatches], "MfmaUtil"
        ),
        "fp16_operations": _median(
            [dispatch["counters"] for dispatch in dispatches], "TOTAL_16_OPS"
        ),
        "achieved_fp16_tops": _median(dispatches, "achieved_fp16_tops"),
    }
    return {"summary": summary, "dispatches": dispatches}


def counter_comparison(
    default: dict[str, object],
    selected: dict[str, object],
    *,
    selected_to_default_b_request_ratio: float | None = None,
) -> dict[str, object]:
    default_summary = default["summary"]
    selected_summary = selected["summary"]

    def reduction(field: str) -> float | None:
        baseline = float(default_summary[field])
        if baseline == 0.0:
            return None
        return 1.0 - float(selected_summary[field]) / baseline

    comparison = {
        "l1_to_l2_read_request_reduction": reduction(
            "l1_to_l2_read_requests"
        ),
        "l2_tag_request_reduction": reduction("l2_tag_requests"),
        "l2_miss_reduction": reduction("l2_misses"),
        "hbm_read_byte_reduction": reduction("hbm_read_bytes"),
        "profiled_duration_speedup": float(default_summary["duration_ns"])
        / float(selected_summary["duration_ns"]),
    }
    if selected_to_default_b_request_ratio is not None:
        ratio = selected_to_default_b_request_ratio
        if not 0.0 <= ratio < 1.0:
            raise ValueError("the inferred B request ratio must be in [0, 1)")

        def decompose(field: str) -> dict[str, float]:
            default_total = float(default_summary[field])
            selected_total = float(selected_summary[field])
            default_b = (default_total - selected_total) / (1.0 - ratio)
            return {
                "fixed_kernel_requests": default_total - default_b,
                "default_b_requests": default_b,
                "selected_b_requests": ratio * default_b,
                "selected_to_default_b_ratio": ratio,
            }

        comparison["inferred_request_decomposition"] = {
            "assumption": (
                "A and other request streams are fixed, and B follows the "
                "quotient-score request ratio"
            ),
            "l1_to_l2_reads": decompose("l1_to_l2_read_requests"),
            "l2_tag_requests": decompose("l2_tag_requests"),
        }
    return comparison
