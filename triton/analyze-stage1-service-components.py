#!/usr/bin/env python3
"""Re-score MI300A random layouts with hardware-conditioned service edges."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from math import log2
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_service import hardware_service_model, lane_bit_subsets


@dataclass(frozen=True)
class CaseSpec:
    element_bytes: int
    mode_names: tuple[str, ...]
    output_names: tuple[str, ...]


CASE_SPECS = {
    "bias_relu": CaseSpec(4, ("feature",), ("feature",)),
    "softmax_bias": CaseSpec(4, ("row", "feature"), ("feature",)),
    "embedding_bag": CaseSpec(2, ("row", "dimension"), ("dimension",)),
    "gemv": CaseSpec(4, ("row", "column"), ("row",)),
    "mvt": CaseSpec(4, ("row", "column"), ("row",)),
    "gesummv": CaseSpec(4, ("row", "column"), ("row",)),
    "stencil5": CaseSpec(4, ("row", "column"), ("column",)),
}

COUNTERS = (
    "l1_cache_line_accesses",
    "l1_to_l2_read_requests",
    "l2_tag_requests",
    "duration_ns",
)


def _execution_record(report):
    for candidate in report["candidates"]:
        launches = candidate.get("structural_validation", {}).get(
            "by_launch", ()
        )
        if launches:
            return launches[0]["reference_execution_layout"]
    raise ValueError(f"{report['case']}: no execution layout in counter report")


def _issue_events(execution, matrix, *, prefix, site, weight, coordinate_map):
    from relay import induce_memory_event

    events = []
    for warp in range(execution.input_size("warp")):
        for register in range(execution.input_size("register")):
            locations = execution.locations(
                fixed={"register": register, "warp": warp, "block": 0}
            )
            events.append(
                induce_memory_event(
                    execution,
                    matrix,
                    locations,
                    id=f"{prefix}.w{warp}.r{register}",
                    site=site,
                    weight=weight,
                    coordinate_map=coordinate_map,
                )
            )
    return tuple(events)


def _case_events(case, execution, matrix):
    if case == "bias_relu":
        return _issue_events(
            execution,
            matrix,
            prefix="bias_relu.bias",
            site="bias_relu.bias.load",
            weight=4096,
            coordinate_map=lambda coord: coord,
        )
    if case == "softmax_bias":
        return _issue_events(
            execution,
            matrix,
            prefix="softmax_bias.bias",
            site="softmax_bias.bias.load",
            weight=1024,
            coordinate_map=lambda coord: (0, coord[0]),
        )
    if case == "embedding_bag":
        return _issue_events(
            execution,
            matrix,
            prefix="embedding_bag.weight",
            site="embedding_bag.weight.load",
            weight=16384,
            coordinate_map=lambda coord: (0, coord[0]),
        )
    if case in {"gemv", "gesummv"}:
        return _issue_events(
            execution,
            matrix,
            prefix=f"{case}.target",
            site=f"{case}.target.load",
            weight=16384,
            coordinate_map=lambda coord: (coord[0], 0),
        )
    if case == "mvt":
        return (
            *_issue_events(
                execution,
                matrix,
                prefix="mvt.row",
                site="mvt.row.load",
                weight=16384,
                coordinate_map=lambda coord: (coord[0], 0),
            ),
            *_issue_events(
                execution,
                matrix,
                prefix="mvt.column",
                site="mvt.column.load",
                weight=16384,
                coordinate_map=lambda coord: (0, coord[0]),
            ),
        )
    if case != "stencil5":
        raise ValueError(f"unsupported case {case!r}")

    events = tuple(
        event
        for site in ("center", "up", "down")
        for event in _issue_events(
            execution,
            matrix,
            prefix=f"stencil5.{site}",
            site=f"stencil5.{site}.load",
            weight=4096,
            coordinate_map=lambda coord: (0, coord[0]),
        )
    )
    for block_index in range(8):
        left_base = block_index * 64 - 1
        right_base = block_index * 64 + 1
        events += _issue_events(
            execution,
            matrix,
            prefix=f"stencil5.left.block{block_index}",
            site="stencil5.left.load",
            weight=512,
            coordinate_map=lambda coord, base=left_base: (
                0,
                max(base + coord[0], 0),
            ),
        )
        events += _issue_events(
            execution,
            matrix,
            prefix=f"stencil5.right.block{block_index}",
            site="stencil5.right.load",
            weight=512,
            coordinate_map=lambda coord, base=right_base: (
                0,
                min(base + coord[0], 511),
            ),
        )
    return events


def _instruction_window_edges(case, execution, matrix):
    from relay import Hyperedge

    if case not in {"gemv", "mvt", "gesummv"}:
        return ()
    locations = execution.locations(fixed={"block": 0})
    rows = tuple(execution.apply(location)[0] for location in locations)
    window = 128 // matrix.element_bytes
    stream_count = 2 if case == "mvt" else 1
    weight = 16 * (1024 // window)
    edges = []
    for stream in range(stream_count):
        points = (
            tuple(
                (row, step) if stream == 0 else (step, row)
                for row in rows
                for step in range(window)
            )
        )
        edges.append(
            Hyperedge.make(
                points,
                weight=weight,
                source=f"{case}.reuse.stream{stream}.window{window}",
            )
        )
    return tuple(edges)


def _lane_stream_temporal_edges(case, execution, matrix):
    from stage1_common import compressed_temporal_edges, temporal_loop_edges

    window = 128 // matrix.element_bytes
    if case in {"gemv", "gesummv"}:
        edges, _ = temporal_loop_edges(
            execution,
            matrix,
            prefix=f"{case}.weight.load",
            steps=1024,
            window=window,
            program_instances=16,
            coordinate_map=lambda coord, step: (coord[0], step),
        )
        return edges
    if case == "mvt":
        row, _ = temporal_loop_edges(
            execution,
            matrix,
            prefix="mvt.row.load",
            steps=1024,
            window=window,
            program_instances=16,
            coordinate_map=lambda coord, step: (coord[0], step),
        )
        column, _ = temporal_loop_edges(
            execution,
            matrix,
            prefix="mvt.column.load",
            steps=1024,
            window=window,
            program_instances=16,
            coordinate_map=lambda coord, step: (step, coord[0]),
        )
        return row + column
    if case == "stencil5":
        edges, _ = compressed_temporal_edges(
            matrix,
            prefix="stencil5.neighborhood",
            point_sets=(
                (
                    (row, column),
                    (row, max(column - 1, 0)),
                    (row, min(column + 1, 511)),
                    (max(row - 1, 0), column),
                    (min(row + 1, 511), column),
                )
                for row in range(512)
                for column in range(512)
            ),
        )
        return edges
    return ()


def _layout(matrix, candidate):
    from relay import LinearInnerLayout

    layout = LinearInnerLayout(
        str(candidate["candidate_id"]),
        matrix.name,
        matrix.mode_bits,
        tuple(int(row) for row in candidate["a_rows"]),
        tuple(reversed(range(matrix.rank))),
    )
    layout.validate(matrix)
    return layout


def _strict_inversion_rate(left, right):
    comparable = 0
    inversions = 0
    for index, left_value in enumerate(left):
        for other in range(index + 1, len(left)):
            left_delta = left_value - left[other]
            right_delta = right[index] - right[other]
            if left_delta == 0 or right_delta == 0:
                continue
            comparable += 1
            inversions += (left_delta < 0) != (right_delta < 0)
    return {
        "comparable_pair_count": comparable,
        "inversion_count": inversions,
        "inversion_rate": inversions / comparable if comparable else None,
    }


def _equal_signature_spread(signatures, values):
    groups = defaultdict(list)
    for signature, value in zip(signatures, values):
        groups[signature].append(float(value))
    tied = [items for items in groups.values() if len(items) > 1]
    spreads = [max(items) - min(items) for items in tied]
    relative = [
        spread / statistics.median(items)
        for spread, items in zip(spreads, tied)
        if statistics.median(items)
    ]
    return {
        "tied_group_count": len(tied),
        "tied_observation_count": sum(len(items) for items in tied),
        "maximum_spread": max(spreads, default=0.0),
        "maximum_relative_spread": max(relative, default=0.0),
        "median_relative_spread": statistics.median(relative) if relative else 0.0,
    }


def _predictor_statistics(scores, candidates):
    from relay import spearman_rank_correlation

    result = {}
    for counter in COUNTERS:
        observed = [
            float(candidate["counters"]["steady_state"][counter])
            for candidate in candidates
        ]
        result[counter] = {
            "spearman_rho": spearman_rank_correlation(scores, observed),
            "strict_pairs": _strict_inversion_rate(scores, observed),
            "equal_score_spread": _equal_signature_spread(scores, observed),
        }
    return result


def analyze_report(report):
    from relay import (
        MatrixSpec,
        ObjectiveComponent,
        TritonLinearLayout,
        low_address_flag,
        weighted_component_region_count,
    )

    case = str(report["case"])
    spec = CASE_SPECS[case]
    matrix = MatrixSpec(
        str(report["target_operand"]),
        tuple(int(size) for size in report["operand_shape"]),
        spec.element_bytes,
        spec.mode_names,
    )
    execution_record = _execution_record(report)
    execution = TritonLinearLayout.from_bases(
        execution_record["bases"],
        tuple(zip(spec.output_names, execution_record["output_shape"])),
    )
    events = _case_events(case, execution, matrix)
    subsets = lane_bit_subsets(execution, exhaustive=True)
    service = hardware_service_model(
        execution,
        events,
        name=f"{case}.service",
        lane_subsets=subsets,
    )
    issue = ObjectiveComponent(
        f"{case}.issue.64B",
        64,
        {matrix.name: tuple(event.hyperedge for event in events)},
    )
    instruction_components = {
        scale: service.instruction.at_scale(scale).build(
            {matrix.name: matrix}, {}, ()
        )[0]
        for scale in (32, 64, 128)
    }
    cohort_components = {
        tuple(dict(fiber.varying_bits)["lane"]): fiber.at_scale(64).build(
            {matrix.name: matrix}, {}, ()
        )[0]
        for fiber in service.lane_cohorts
    }
    instruction_window_edges = _instruction_window_edges(
        case, execution, matrix
    )
    instruction_window = (
        ObjectiveComponent(
            f"{case}.instruction_window.128B",
            128,
            {matrix.name: instruction_window_edges},
        )
        if instruction_window_edges
        else None
    )
    temporal_edges = _lane_stream_temporal_edges(case, execution, matrix)
    lane_stream_window = ObjectiveComponent(
        f"{case}.lane_stream_window.128B",
        128,
        {matrix.name: temporal_edges},
    )

    scored = []
    for candidate in report["candidates"]:
        layout = _layout(matrix, candidate)
        offset_cache = {}
        m0 = weighted_component_region_count(
            matrix, layout, issue, offset_cache=offset_cache
        )
        m1 = {
            scale: weighted_component_region_count(
                matrix,
                layout,
                component,
                offset_cache=offset_cache,
            )
            for scale, component in instruction_components.items()
        }
        if abs(m0 - float(candidate["quotient_score"])) > 1e-9:
            raise ValueError(f"{case}: reconstructed M0 score changed")
        if abs(m0 - m1[64]) > 1e-9:
            raise ValueError(f"{case}: M1 instruction partition changed M0")
        cohort_scores = {
            "_".join(str(bit) for bit in bits): weighted_component_region_count(
                matrix,
                layout,
                component,
                offset_cache=offset_cache,
            )
            for bits, component in cohort_components.items()
        }
        instruction_window_score = (
            weighted_component_region_count(
                matrix,
                layout,
                instruction_window,
                offset_cache=offset_cache,
            )
            if instruction_window is not None
            else None
        )
        lane_stream_score = weighted_component_region_count(
            matrix,
            layout,
            lane_stream_window,
            offset_cache=offset_cache,
        )
        tuned_score = (
            m1[128]
            + lane_stream_score
            + m1[64]
            + 0.0625 * cohort_scores["2_3"]
        )
        flag = low_address_flag(matrix, layout)
        flag_signature = {
            f"{scale}B": list(
                flag[int(log2(scale // matrix.element_bytes))]
            )
            for scale in (32, 64, 128)
        }
        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "mapping_id": candidate["mapping_id"],
                "m0_issue_64B": m0,
                "m1_instruction": {
                    f"{scale}B": score for scale, score in m1.items()
                },
                "m2_lane_cohorts_64B": cohort_scores,
                "m3_instruction_window_128B": instruction_window_score,
                "m3_lane_stream_window_128B": lane_stream_score,
                "mi300a_v1_weighted_score": tuned_score,
                "multiscale_flag": flag_signature,
            }
        )

    predictors = {
        "m0_issue_64B": [record["m0_issue_64B"] for record in scored],
    }
    for scale in (32, 64, 128):
        predictors[f"m1_instruction_{scale}B"] = [
            record["m1_instruction"][f"{scale}B"] for record in scored
        ]
    for bits in cohort_components:
        label = "_".join(str(bit) for bit in bits)
        predictors[f"m2_lane_bits_{label}_64B"] = [
            record["m2_lane_cohorts_64B"][label] for record in scored
        ]
    if instruction_window is not None:
        predictors["m3_instruction_window_128B"] = [
            record["m3_instruction_window_128B"] for record in scored
        ]
    predictors["m3_lane_stream_window_128B"] = [
        record["m3_lane_stream_window_128B"] for record in scored
    ]
    predictors["mi300a_v1_weighted_score"] = [
        record["mi300a_v1_weighted_score"] for record in scored
    ]

    statistics_by_predictor = {
        name: _predictor_statistics(values, report["candidates"])
        for name, values in predictors.items()
    }
    nested_signature = [
        tuple(
            record["m2_lane_cohorts_64B"][
                "_".join(str(bit) for bit in bits)
            ]
            for bits in lane_bit_subsets(execution)
        )
        for record in scored
    ]
    exact_flag_signature = [
        tuple(
            tuple(record["multiscale_flag"][f"{scale}B"])
            for scale in (32, 64, 128)
        )
        for record in scored
    ]
    residuals = {}
    for counter in COUNTERS:
        observed = [
            float(candidate["counters"]["steady_state"][counter])
            for candidate in report["candidates"]
        ]
        residuals[counter] = {
            "m0_scalar": _equal_signature_spread(
                predictors["m0_issue_64B"], observed
            ),
            "m2_nested_signature": _equal_signature_spread(
                nested_signature, observed
            ),
            "exact_multiscale_flag": _equal_signature_spread(
                exact_flag_signature, observed
            ),
        }
    return {
        "case": case,
        "observation_count": len(scored),
        "service_model": service.record(),
        "predictor_statistics": statistics_by_predictor,
        "residual_spread": residuals,
        "candidates": scored,
    }


def summarize(cases):
    predictors = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for predictor, counters in case["predictor_statistics"].items():
            for counter, values in counters.items():
                rho = values["spearman_rho"]
                if rho is not None:
                    predictors[predictor][counter].append(float(rho))
    return {
        predictor: {
            counter: {
                "case_count": len(values),
                "median_case_spearman_rho": statistics.median(values),
                "minimum_case_spearman_rho": min(values),
                "maximum_case_spearman_rho": max(values),
                "by_case": {
                    case["case"]: case["predictor_statistics"][predictor][
                        counter
                    ]["spearman_rho"]
                    for case in cases
                    if predictor in case["predictor_statistics"]
                },
            }
            for counter, values in counters.items()
        }
        for predictor, counters in predictors.items()
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    cases = [
        analyze_report(json.loads(path.read_text(encoding="utf-8")))
        for path in args.reports
    ]
    result = {
        "experiment": "triton_stage1_hardware_service_analysis",
        "inputs": [str(path) for path in args.reports],
        "cases": cases,
        "summary": summarize(cases),
        "correct": True,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json is None:
        print(payload, end="")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
