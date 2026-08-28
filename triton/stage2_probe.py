"""Pure construction and analysis helpers for the controlled Stage-2 probe."""

from __future__ import annotations

from collections import defaultdict
import statistics


def build_gemm_b_resource_groups(
    matrix,
    execution,
    *,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    resource_map,
):
    """Construct exact translated four-issue B cohorts for a tiled GEMM."""

    from relay import ResourceCohortGroup, ResourceCohortOccurrence

    if matrix.shape != (k, n):
        raise ValueError("the Stage-2 probe expects a non-transposed KxN B")
    if execution.output_shape != (block_k, block_n):
        raise ValueError("the execution layout does not cover the configured B tile")
    if any(
        extent % block
        for extent, block in ((m, block_m), (n, block_n), (k, block_k))
    ):
        raise ValueError("GEMM dimensions must be divisible by their block sizes")

    occurrences_by_shape = defaultdict(list)
    registers = execution.input_size("register")
    warps = execution.input_size("warp")
    if registers != 4:
        raise ValueError(
            "the controlled service probe requires four B issues per warp"
        )
    for warp in range(warps):
        logical_bits = sorted(
            {
                matrix.coord_to_bits(execution.apply(location))
                for register in range(registers)
                for location in execution.locations(
                    fixed={"register": register, "warp": warp, "block": 0}
                )
            }
        )
        anchor = logical_bits[0]
        relative = tuple(sorted(value ^ anchor for value in logical_bits))
        occurrences = occurrences_by_shape[relative]
        for k_base in range(0, k, block_k):
            for n_base in range(0, n, block_n):
                base = matrix.coord_to_bits((k_base, n_base))
                occurrences.append(
                    ResourceCohortOccurrence(
                        anchors=((matrix.name, base ^ anchor),),
                        weight=m // block_m,
                        source=f"B.w{warp}.k{k_base}.n{n_base}",
                    )
                )

    return tuple(
        ResourceCohortGroup(
            resource_map.cohort_family,
            tuple((matrix.name, value) for value in relative),
            tuple(occurrences),
        )
        for relative, occurrences in sorted(occurrences_by_shape.items())
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


def _relative_spread(values: list[float]) -> float:
    minimum = min(values)
    return max(values) / minimum - 1.0


def analyze_fiber_candidates(
    candidates: list[dict[str, object]],
    *,
    meaningful_spread: float = 0.02,
    predictive_correlation: float = 0.5,
    maximum_service_regret: float = 0.02,
) -> dict[str, object]:
    """Summarize same-flag runtime variation and service-score prediction."""

    from relay import spearman_rank_correlation

    if not candidates:
        raise ValueError("the Stage-2 probe requires candidates")
    quotients = {float(candidate["quotient_score"]) for candidate in candidates}
    flags = {str(candidate["flag_id"]) for candidate in candidates}
    if len(quotients) != 1 or len(flags) != 1:
        raise ValueError("Stage-2 probe candidates must share one flag and quotient")

    identity = next(
        candidate for candidate in candidates if not candidate["shears"]
    )
    fastest = min(candidates, key=lambda candidate: float(candidate["runtime_ms"]))
    ordered_by_service = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["resource_service_score"]),
            int(candidate["codegen_cost"]["xors"]),
            int(candidate["codegen_cost"]["runs"]),
            tuple(tuple(shear) for shear in candidate["shears"]),
        ),
    )
    service_selected = ordered_by_service[0]
    minimum_service = float(service_selected["resource_service_score"])
    service_optimal = [
        candidate
        for candidate in candidates
        if float(candidate["resource_service_score"]) == minimum_service
    ]
    fastest_runtime = float(fastest["runtime_ms"])
    service_best = min(
        service_optimal, key=lambda candidate: float(candidate["runtime_ms"])
    )

    levels = []
    by_service = defaultdict(list)
    for candidate in candidates:
        by_service[float(candidate["resource_service_score"])].append(candidate)
    for score, members in sorted(by_service.items()):
        runtimes = [float(member["runtime_ms"]) for member in members]
        levels.append(
            {
                "resource_service_score": score,
                "candidate_count": len(members),
                "median_runtime_ms": statistics.median(runtimes),
                "minimum_runtime_ms": min(runtimes),
                "maximum_runtime_ms": max(runtimes),
                "relative_runtime_spread": _relative_spread(runtimes),
                "candidate_ids": [str(member["candidate_id"]) for member in members],
            }
        )

    service_correlation = spearman_rank_correlation(
        [float(candidate["resource_service_score"]) for candidate in candidates],
        [float(candidate["runtime_ms"]) for candidate in candidates],
    )
    codegen_xor_correlation = spearman_rank_correlation(
        [float(candidate["codegen_cost"]["xors"]) for candidate in candidates],
        [float(candidate["runtime_ms"]) for candidate in candidates],
    )
    codegen_run_correlation = spearman_rank_correlation(
        [float(candidate["codegen_cost"]["runs"]) for candidate in candidates],
        [float(candidate["runtime_ms"]) for candidate in candidates],
    )
    compiled_instruction_correlation = None
    compiled_register_correlation = None
    if all(
        "compiled_codegen" in candidate
        and "assembly_instruction_count" in candidate["compiled_codegen"]
        and "n_regs" in candidate["compiled_codegen"]
        for candidate in candidates
    ):
        compiled_instruction_correlation = spearman_rank_correlation(
            [
                float(candidate["compiled_codegen"]["assembly_instruction_count"])
                for candidate in candidates
            ],
            [float(candidate["runtime_ms"]) for candidate in candidates],
        )
        compiled_register_correlation = spearman_rank_correlation(
            [
                float(candidate["compiled_codegen"]["n_regs"])
                for candidate in candidates
            ],
            [float(candidate["runtime_ms"]) for candidate in candidates],
        )
    runtime_spread = _relative_spread(
        [float(candidate["runtime_ms"]) for candidate in candidates]
    )
    service_selected_regret = (
        float(service_selected["runtime_ms"]) / fastest_runtime - 1.0
    )
    service_set_regret = float(service_best["runtime_ms"]) / fastest_runtime - 1.0
    meaningful = runtime_spread >= meaningful_spread
    predictive = (
        service_correlation is not None
        and service_correlation >= predictive_correlation
        and service_set_regret <= maximum_service_regret
    )
    return {
        "candidate_count": len(candidates),
        "quotient_invariant": True,
        "quotient_score": next(iter(quotients)),
        "flag_id": next(iter(flags)),
        "identity_candidate_id": identity["candidate_id"],
        "fastest_candidate_id": fastest["candidate_id"],
        "service_selected_candidate_id": service_selected["candidate_id"],
        "service_best_candidate_id": service_best["candidate_id"],
        "identity_runtime_ms": float(identity["runtime_ms"]),
        "fastest_runtime_ms": fastest_runtime,
        "identity_to_fastest_speedup": (
            float(identity["runtime_ms"]) / fastest_runtime
        ),
        "runtime_relative_spread": runtime_spread,
        "resource_service_rank_correlation": service_correlation,
        "codegen_xor_rank_correlation": codegen_xor_correlation,
        "codegen_run_rank_correlation": codegen_run_correlation,
        "compiled_instruction_rank_correlation": (
            compiled_instruction_correlation
        ),
        "compiled_register_rank_correlation": compiled_register_correlation,
        "service_selected_regret": service_selected_regret,
        "service_optimal_set_regret": service_set_regret,
        "service_levels": levels,
        "gate": {
            "thresholds": {
                "meaningful_runtime_spread": meaningful_spread,
                "predictive_rank_correlation": predictive_correlation,
                "maximum_service_optimal_set_regret": maximum_service_regret,
            },
            "meaningful_runtime_variation": meaningful,
            "service_predictive": predictive,
            "develop_stage_2": meaningful and predictive,
        },
    }


def aggregate_probe_results(
    results: list[dict[str, object]],
    *,
    meaningful_spread: float = 0.02,
    predictive_correlation: float = 0.5,
    maximum_service_regret: float = 0.02,
) -> dict[str, object]:
    """Aggregate a Stage-2 probe across independent Python processes."""

    if not results:
        raise ValueError("Stage-2 probe aggregation requires results")
    candidate_ids = [
        str(candidate["candidate_id"]) for candidate in results[0]["candidates"]
    ]
    for result in results[1:]:
        observed = [
            str(candidate["candidate_id"]) for candidate in result["candidates"]
        ]
        if observed != candidate_ids:
            raise ValueError("fiber candidates changed between processes")

    candidates = []
    for index, candidate_id in enumerate(candidate_ids):
        process_candidates = [result["candidates"][index] for result in results]
        medians = [
            float(candidate["timing"]["median_ms"])
            for candidate in process_candidates
        ]
        samples = [
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
        runtime = statistics.median(medians)
        candidate["runtime_ms"] = runtime
        candidate["timing"] = {
            **timing_summary(samples),
            "aggregation_runtime_ms": runtime,
            "aggregation_method": "median_of_process_medians",
            "process_medians_ms": medians,
            "process_launch_count": len(medians),
        }
        candidate["compiled_codegen"] = codegen[0]
        candidate["compiled_codegen_consistent"] = all(
            item == codegen[0] for item in codegen[1:]
        )
        if not candidate["compiled_codegen_consistent"]:
            candidate["compiled_codegen_by_process"] = codegen
        candidates.append(candidate)

    aggregate = {
        key: value
        for key, value in results[0].items()
        if key not in {"analysis", "candidates", "correct", "process"}
    }
    aggregate.update(
        {
            "process_launch_count": len(results),
            "processes": [result["process"] for result in results],
            "candidates": candidates,
            "analysis": analyze_fiber_candidates(
                candidates,
                meaningful_spread=meaningful_spread,
                predictive_correlation=predictive_correlation,
                maximum_service_regret=maximum_service_regret,
            ),
            "correct": all(bool(result["correct"]) for result in results),
        }
    )
    return aggregate
