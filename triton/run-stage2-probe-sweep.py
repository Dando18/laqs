#!/usr/bin/env python3
"""Run and aggregate the controlled Stage-2 probe in fresh processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_common import positive_integer
from stage2_probe import aggregate_probe_results


def worker_command(worker: Path, args, output: Path) -> list[str]:
    return [
        sys.executable,
        str(worker),
        "--cache-mode",
        args.cache_mode,
        "--cache-thrash-bytes",
        str(args.cache_thrash_bytes),
        "--samples",
        str(args.samples),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--meaningful-spread",
        str(args.meaningful_spread),
        "--predictive-correlation",
        str(args.predictive_correlation),
        "--maximum-service-regret",
        str(args.maximum_service_regret),
        "--json",
        str(output),
        "--quiet",
    ]


def run_sweep(args) -> dict[str, object]:
    worker = Path(__file__).with_name("run-stage2-probe.py")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for process_launch in range(1, args.process_launches + 1):
        output = args.results_dir / f"process-{process_launch}.json"
        outputs.append(output)
        if output.exists() and not args.rerun:
            print(
                f"Stage-2 probe: reuse process {process_launch}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"Stage-2 probe: process {process_launch}/{args.process_launches}",
            file=sys.stderr,
            flush=True,
        )
        subprocess.run(worker_command(worker, args, output), check=True)

    results = [
        json.loads(output.read_text(encoding="utf-8")) for output in outputs
    ]
    aggregate = aggregate_probe_results(
        results,
        meaningful_spread=args.meaningful_spread,
        predictive_correlation=args.predictive_correlation,
        maximum_service_regret=args.maximum_service_regret,
    )
    aggregate["complete"] = True
    return aggregate


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-launches", type=positive_integer, default=3)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("triton/results/stage2-probe-processes"),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--cache-mode", choices=("warm", "thrashed"), default="thrashed")
    parser.add_argument(
        "--cache-thrash-bytes", type=positive_integer, default=256 << 20
    )
    parser.add_argument("--samples", type=positive_integer, default=9)
    parser.add_argument("--iterations", type=positive_integer, default=1)
    parser.add_argument("--warmup", type=positive_integer, default=1)
    parser.add_argument("--meaningful-spread", type=float, default=0.02)
    parser.add_argument("--predictive-correlation", type=float, default=0.5)
    parser.add_argument("--maximum-service-regret", type=float, default=0.02)
    parser.add_argument(
        "--json", type=Path, default=Path("triton/results/stage2-probe.json")
    )
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
