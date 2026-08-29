#!/usr/bin/env python3
"""Run a resumable sweep of generalized Stage-1 GEMM regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_common import positive_integer
from stage1_operand import aggregate_persistent_rankings


CASES = {
    "square_512_warm": {},
    "square_1024_warm": {"m": 1024, "n": 1024, "k": 1024},
    "square_2048_warm": {"m": 2048, "n": 2048, "k": 2048},
    "skinny_m64_n1024_k1024_warm": {"m": 64, "n": 1024, "k": 1024},
    "skinny_m32_n2048_k1024_warm": {"m": 32, "n": 2048, "k": 1024},
    "square_512_thrashed": {"cache_mode": "thrashed"},
    "square_1024_thrashed": {
        "m": 1024,
        "n": 1024,
        "k": 1024,
        "cache_mode": "thrashed",
    },
    "skinny_m64_n1024_k1024_thrashed": {
        "m": 64,
        "n": 1024,
        "k": 1024,
        "cache_mode": "thrashed",
    },
    "transposed_a_512_warm": {"trans_a": True},
    "transposed_b_512_warm": {"trans_b": True},
    "block_m16_n32_k16_w4": {"block_m": 16, "block_k": 16},
    "block_m64_n64_k64_w8": {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "num_warps": 8,
    },
    "block_m32_n64_k32_w8": {"block_n": 64, "num_warps": 8},
}


def current_case_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        scope = result["ranking"]["search_scope"]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        scope.get("grammar") == "canonical_inner_tile"
        and scope.get("tile_policy") == "explicit_hypothesis_sweep_v1"
    )


def case_command(worker: Path, args, name: str, output: Path) -> list[str]:
    configuration = {
        "m": 512,
        "n": 512,
        "k": 512,
        "block_m": 32,
        "block_n": 32,
        "block_k": 32,
        "num_warps": 4,
        "trans_a": False,
        "trans_b": False,
        "cache_mode": "warm",
        **CASES[name],
    }
    cold = configuration["cache_mode"] == "thrashed"
    command = [
        sys.executable,
        str(worker),
        "--m",
        str(configuration["m"]),
        "--n",
        str(configuration["n"]),
        "--k",
        str(configuration["k"]),
        "--block-m",
        str(configuration["block_m"]),
        "--block-n",
        str(configuration["block_n"]),
        "--block-k",
        str(configuration["block_k"]),
        "--num-warps",
        str(configuration["num_warps"]),
        "--cache-mode",
        str(configuration["cache_mode"]),
        "--cache-thrash-bytes",
        str(args.cache_thrash_bytes),
        "--transaction-bytes",
        str(args.transaction_bytes),
        "--candidates",
        str(args.candidates),
        "--samples",
        str(args.cold_samples if cold else args.warm_samples),
        "--iterations",
        str(args.cold_iterations if cold else args.warm_iterations),
        "--warmup",
        str(args.cold_warmup if cold else args.warm_warmup),
        "--json",
        str(output),
        "--quiet",
    ]
    if configuration["trans_a"]:
        command.append("--trans-a")
    if configuration["trans_b"]:
        command.append("--trans-b")
    return command


def case_summary(result: dict[str, object]) -> dict[str, object]:
    ranking = result["ranking"]
    quality = ranking["rank_quality"]
    return {
        "configuration": result["configuration"],
        "inner_tile_shapes": ranking["search_scope"]["inner_tile_shapes"],
        "selected_inner_tile_shape": ranking["selected"][
            "inner_tile_shape"
        ],
        "fixed_outer_order": ranking["search_scope"]["fixed_outer_order"],
        "default_quotient": ranking["default_quotient"],
        "selected_quotient": ranking["selected_quotient"],
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


def run_sweep(args) -> dict[str, object]:
    worker = Path(__file__).with_name("run-stage1-gemm-case.py")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    requested = list(CASES) if args.cases == ["all"] else args.cases
    unknown = sorted(set(requested) - set(CASES))
    if unknown:
        raise ValueError(f"unknown GEMM breadth cases: {unknown}")
    for index, name in enumerate(requested, start=1):
        case_dir = args.results_dir / name
        case_dir.mkdir(exist_ok=True)
        for process_launch in range(1, args.process_launches + 1):
            output = case_dir / f"process-{process_launch}.json"
            if current_case_result(output) and not args.rerun:
                print(
                    f"GEMM breadth: reuse {name} process {process_launch}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(
                f"GEMM breadth: {index}/{len(requested)} {name}, process "
                f"{process_launch}/{args.process_launches}",
                file=sys.stderr,
                flush=True,
            )
            subprocess.run(case_command(worker, args, name, output), check=True)

    completed = {}
    missing = []
    for name in CASES:
        outputs = [
            args.results_dir / name / f"process-{process_launch}.json"
            for process_launch in range(1, args.process_launches + 1)
        ]
        absent = [path for path in outputs if not current_case_result(path)]
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
        "experiment": "triton_stage1_gemm_breadth",
        "configuration": {
            "transaction_bytes": args.transaction_bytes,
            "requested_retained_candidates": args.candidates,
            "process_launches": args.process_launches,
            "warm_timing": {
                "samples": args.warm_samples,
                "iterations": args.warm_iterations,
                "warmup": args.warm_warmup,
            },
            "thrashed_timing": {
                "samples": args.cold_samples,
                "iterations": args.cold_iterations,
                "warmup": args.cold_warmup,
                "cache_thrash_bytes": args.cache_thrash_bytes,
                "timing_scope": "GEMM only; cache-thrash kernel excluded",
            },
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
        default=Path("triton/results/stage1-gemm-breadth-cases"),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--transaction-bytes", type=positive_integer, default=128)
    parser.add_argument("--candidates", type=positive_integer, default=8)
    parser.add_argument("--process-launches", type=positive_integer, default=3)
    parser.add_argument("--warm-samples", type=positive_integer, default=9)
    parser.add_argument("--warm-iterations", type=positive_integer, default=20)
    parser.add_argument("--warm-warmup", type=positive_integer, default=5)
    parser.add_argument("--cold-samples", type=positive_integer, default=5)
    parser.add_argument("--cold-iterations", type=positive_integer, default=1)
    parser.add_argument("--cold-warmup", type=positive_integer, default=1)
    parser.add_argument(
        "--cache-thrash-bytes", type=positive_integer, default=256 << 20
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    result = run_sweep(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if not args.quiet:
        print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
