#!/usr/bin/env python3
"""Run the resumable focused Stage-1 persistent-operand kernel suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_common import positive_integer
from stage1_operand import aggregate_persistent_rankings
from run_stage1_kernel_cases import CASES


def current_case_result(path: Path, args) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        scope = result["ranking"]["search_scope"]
    except (KeyError, TypeError, ValueError):
        return False
    service = result["ranking"].get("hardware_service_model", {})
    matches = (
        scope.get("grammar") == "canonical_inner_tile"
        and scope.get("tile_policy") == "explicit_hypothesis_sweep_v1"
        and scope.get("temporal_mode") == args.temporal_mode
        and service.get("name") == args.service_model
    )
    if not matches or args.service_model == "none":
        return matches
    return (
        service.get("lane_cohort_bits") == list(args.lane_cohort_bits)
        and service.get("instruction_region_bytes")
        == args.instruction_bytes
        and service.get("lane_cohort_region_bytes")
        == args.lane_cohort_bytes
        and service.get("tuning")
        == {
            "issue_tau": args.issue_tau,
            "temporal_tau": args.temporal_tau,
            "instruction_tau": args.instruction_tau,
            "lane_cohort_tau": args.lane_cohort_tau,
        }
    )


def case_summary(result: dict[str, object]) -> dict[str, object]:
    ranking = result["ranking"]
    quality = ranking["rank_quality"]
    return {
        "target_operand": result["target_operand"],
        "operand_shape": result["operand_shape"],
        "inner_tile_shapes": ranking["search_scope"]["inner_tile_shapes"],
        "selected_inner_tile_shape": ranking["selected"][
            "inner_tile_shape"
        ],
        "fixed_outer_order": ranking["search_scope"]["fixed_outer_order"],
        "default_quotient": ranking["default_quotient"],
        "selected_quotient": ranking["selected_quotient"],
        "default_quotient_components": ranking["default"][
            "quotient_components"
        ],
        "selected_quotient_components": ranking["selected"][
            "quotient_components"
        ],
        "temporal_model": ranking["temporal_model"],
        "hardware_service_model": ranking["hardware_service_model"],
        "default_runtime_ms": ranking["default_runtime_ms"],
        "selected_runtime_ms": ranking["selected_runtime_ms"],
        "measured_speedup": ranking["measured_speedup"],
        "top_1_regret": quality["regret"]["top_1"]["regret"],
        "top_2_regret": quality["regret"]["top_2"]["regret"],
        "top_3_regret": quality["regret"]["top_3"]["regret"],
        "top_4_regret": quality["regret"]["top_4"]["regret"],
        "top_5_regret": quality["regret"]["top_5"]["regret"],
        "rank_correlation": quality["rank_correlation"]["rho"],
        "ranking_candidate_count": quality["candidate_count"],
        "removed_duplicate_mapping_count": quality[
            "removed_duplicate_mapping_count"
        ],
        "laqs_made_no_change": ranking["laqs_made_no_change"],
        "selected_word": ranking["selected"]["word"],
        "correct": result["correct"],
    }


def run_suite(args) -> dict[str, object]:
    worker = Path(__file__).with_name("run-stage1-kernel-case.py")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    requested = list(CASES) if args.cases == ["all"] else args.cases
    unknown = sorted(set(requested) - set(CASES))
    if unknown:
        raise ValueError(f"unknown kernel breadth cases: {unknown}")
    for index, name in enumerate(requested, start=1):
        case_dir = args.results_dir / name
        case_dir.mkdir(exist_ok=True)
        for process_launch in range(1, args.process_launches + 1):
            output = case_dir / f"process-{process_launch}.json"
            if current_case_result(output, args) and not args.rerun:
                print(
                    f"Kernel breadth: reuse {name} process {process_launch}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(
                f"Kernel breadth: {index}/{len(requested)} {name}, process "
                f"{process_launch}/{args.process_launches}",
                file=sys.stderr,
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--case",
                    name,
                    "--transaction-bytes",
                    str(args.transaction_bytes),
                    "--candidates",
                    str(args.candidates),
                    "--samples",
                    str(args.samples),
                    "--iterations",
                    str(args.iterations),
                    "--warmup",
                    str(args.warmup),
                    "--temporal-mode",
                    args.temporal_mode,
                    "--service-model",
                    args.service_model,
                    "--issue-tau",
                    str(args.issue_tau),
                    "--temporal-tau",
                    str(args.temporal_tau),
                    "--instruction-tau",
                    str(args.instruction_tau),
                    "--lane-cohort-tau",
                    str(args.lane_cohort_tau),
                    "--instruction-bytes",
                    str(args.instruction_bytes),
                    "--lane-cohort-bytes",
                    str(args.lane_cohort_bytes),
                    "--lane-cohort-bits",
                    *(str(bit) for bit in args.lane_cohort_bits),
                    "--json",
                    str(output),
                    "--quiet",
                ],
                check=True,
            )

    completed = {}
    missing = []
    for name in CASES:
        outputs = [
            args.results_dir / name / f"process-{process_launch}.json"
            for process_launch in range(1, args.process_launches + 1)
        ]
        absent = [
            path
            for path in outputs
            if not current_case_result(path, args)
        ]
        if absent:
            missing.extend(str(path.relative_to(args.results_dir)) for path in absent)
            continue
        processes = [
            json.loads(output.read_text(encoding="utf-8")) for output in outputs
        ]
        first = processes[0]
        completed[name] = {
            key: value
            for key, value in first.items()
            if key not in {"correct", "process", "ranking"}
        }
        completed[name]["ranking"] = aggregate_persistent_rankings(
            [process["ranking"] for process in processes]
        )
        completed[name]["processes"] = [
            process["process"] for process in processes
        ]
        completed[name]["correct"] = all(
            bool(process["correct"]) for process in processes
        )
    return {
        "stage": 1,
        "experiment": "triton_stage1_kernel_breadth",
        "configuration": {
            "transaction_bytes": args.transaction_bytes,
            "requested_retained_candidates": args.candidates,
            "process_launches": args.process_launches,
            "samples": args.samples,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "temporal_mode": args.temporal_mode,
            "service_model": args.service_model,
            "issue_tau": args.issue_tau,
            "temporal_tau": args.temporal_tau,
            "instruction_tau": args.instruction_tau,
            "lane_cohort_tau": args.lane_cohort_tau,
            "instruction_bytes": args.instruction_bytes,
            "lane_cohort_bytes": args.lane_cohort_bytes,
            "lane_cohort_bits": list(args.lane_cohort_bits),
        },
        "case_order": list(CASES),
        "completed_cases": list(completed),
        "missing_cases": missing,
        "summary": {
            name: case_summary(result) for name, result in completed.items()
        },
        "cases": completed,
        "complete": not missing,
        "correct": all(bool(result["correct"]) for result in completed.values()),
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["all"])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("triton/results/stage1-kernel-breadth-cases"),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--process-launches", type=positive_integer, default=3)
    parser.add_argument("--samples", type=positive_integer, default=9)
    parser.add_argument("--iterations", type=positive_integer, default=20)
    parser.add_argument("--warmup", type=positive_integer, default=5)
    parser.add_argument(
        "--temporal-mode",
        choices=("issue", "union", "split"),
        default="issue",
    )
    parser.add_argument(
        "--service-model", choices=("none", "mi300a_v1"), default="none"
    )
    parser.add_argument("--issue-tau", type=float, default=1.0)
    parser.add_argument("--temporal-tau", type=float, default=1.0)
    parser.add_argument("--instruction-tau", type=float, default=1.0)
    parser.add_argument("--lane-cohort-tau", type=float, default=0.0625)
    parser.add_argument("--instruction-bytes", type=positive_integer, default=64)
    parser.add_argument("--lane-cohort-bytes", type=positive_integer, default=64)
    parser.add_argument(
        "--lane-cohort-bits", type=int, nargs="+", default=(2, 3)
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    result = run_suite(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
