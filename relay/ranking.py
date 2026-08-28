"""Metrics for comparing analytical layout rankings with measured runtimes."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import fmean
from typing import Hashable, Mapping, Sequence


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return one-based ranks, assigning the average rank to tied values."""

    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return tuple(ranks)


def spearman_rank_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Return tie-aware Spearman rho, or ``None`` for a constant ranking."""

    if len(left) != len(right):
        raise ValueError("rank-correlation inputs must have the same length")
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = fmean(left_ranks)
    right_mean = fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_norm = sum((value - left_mean) ** 2 for value in left_ranks)
    right_norm = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = sqrt(left_norm * right_norm)
    if denominator == 0.0:
        return None
    return numerator / denominator


def _spread(
    records: Sequence[Mapping[str, object]],
    *,
    id_key: str,
    runtime_key: str,
) -> dict[str, object]:
    runtimes = [float(record[runtime_key]) for record in records]
    minimum = min(runtimes)
    maximum = max(runtimes)
    return {
        "candidate_ids": [str(record[id_key]) for record in records],
        "count": len(records),
        "min_runtime_ms": minimum,
        "max_runtime_ms": maximum,
        "absolute_spread_ms": maximum - minimum,
        "relative_spread": maximum / minimum - 1.0,
    }


def summarize_rank_quality(
    records: Sequence[Mapping[str, object]],
    *,
    id_key: str = "candidate_id",
    score_key: str = "quotient_score",
    runtime_key: str = "runtime_ms",
    flag_key: str = "flag_id",
    mapping_key: str = "mapping_id",
) -> dict[str, object]:
    """Summarize regret, rank agreement, and tied-layout runtime spreads.

    Input order resolves quotient-score ties. Stage 1 supplies solver order,
    whose secondary keys are address-expression runs and XOR count.
    """

    if not records:
        raise ValueError("rank-quality analysis requires at least one candidate")
    for record in records:
        for key in (id_key, score_key, runtime_key, flag_key, mapping_key):
            if key not in record:
                raise ValueError(f"candidate record is missing {key!r}")

    indexed = list(enumerate(records))
    ordered = [
        record
        for _, record in sorted(
            indexed,
            key=lambda item: (float(item[1][score_key]), item[0]),
        )
    ]
    fastest = min(records, key=lambda record: float(record[runtime_key]))
    fastest_runtime = float(fastest[runtime_key])

    regrets: dict[str, object] = {}
    for requested in (1, 3):
        count = min(requested, len(ordered))
        selected = min(
            ordered[:count], key=lambda record: float(record[runtime_key])
        )
        selected_runtime = float(selected[runtime_key])
        regrets[f"top_{requested}"] = {
            "candidate_count": count,
            "selected_candidate_id": str(selected[id_key]),
            "selected_runtime_ms": selected_runtime,
            "regret": selected_runtime / fastest_runtime - 1.0,
        }

    quotient_groups: dict[Hashable, list[Mapping[str, object]]] = defaultdict(list)
    flag_groups: dict[Hashable, list[Mapping[str, object]]] = defaultdict(list)
    mapping_groups: dict[Hashable, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        quotient_groups[record[score_key]].append(record)  # type: ignore[index]
        flag_groups[record[flag_key]].append(record)  # type: ignore[index]
        mapping_groups[record[mapping_key]].append(record)  # type: ignore[index]

    equal_quotient = []
    for score, group in sorted(quotient_groups.items(), key=lambda item: float(item[0])):
        if len(group) < 2:
            continue
        item = _spread(group, id_key=id_key, runtime_key=runtime_key)
        item["quotient_score"] = float(score)
        equal_quotient.append(item)

    same_flag = []
    for flag, group in flag_groups.items():
        distinct_mappings = {record[mapping_key] for record in group}
        if len(distinct_mappings) < 2:
            continue
        item = _spread(group, id_key=id_key, runtime_key=runtime_key)
        item["flag_id"] = str(flag)
        item["distinct_mapping_count"] = len(distinct_mappings)
        same_flag.append(item)
    same_flag.sort(key=lambda item: str(item["flag_id"]))

    duplicate_mapping = []
    for mapping, group in mapping_groups.items():
        if len(group) < 2:
            continue
        item = _spread(group, id_key=id_key, runtime_key=runtime_key)
        item["mapping_id"] = str(mapping)
        duplicate_mapping.append(item)
    duplicate_mapping.sort(key=lambda item: str(item["mapping_id"]))

    scores = [float(record[score_key]) for record in records]
    runtimes = [float(record[runtime_key]) for record in records]
    return {
        "runtime_metric": runtime_key,
        "ranking_population": "retained_candidates",
        "tie_breaking": "input solver order",
        "candidate_count": len(records),
        "distinct_mapping_count": len(mapping_groups),
        "fastest_candidate_id": str(fastest[id_key]),
        "fastest_runtime_ms": fastest_runtime,
        "quotient_order_candidate_ids": [
            str(record[id_key]) for record in ordered
        ],
        "regret": regrets,
        "rank_correlation": {
            "method": "spearman_tie_aware",
            "rho": spearman_rank_correlation(scores, runtimes),
            "sample_count": len(records),
            "lower_is_better": True,
        },
        "equal_quotient_score_runtime_spread": equal_quotient,
        "same_flag_runtime_spread": {
            "available": bool(same_flag),
            "groups": same_flag,
        },
        "duplicate_mapping_runtime_spread": duplicate_mapping,
    }
