#!/usr/bin/env python3
"""Account for every declared case and the feasibility of the named reserves."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from tritonbench_cases import CASES
from tau_profiles import TAU_NAMES


def reserve_inventory():
    # This is the pinned TritonBench implementation inventory, not a promise
    # that an optional package or torch.compile is a direct portable JIT case.
    return [
        {"operator": "mamba2_chunk_state", "status": "unavailable_in_pinned_panel",
         "reason": "TritonBench imports its Triton implementation from optional mamba_ssm; no in-tree JIT kernel"},
        {"operator": "mamba2_chunk_scan", "status": "unavailable_in_pinned_panel",
         "reason": "TritonBench imports its Triton implementation from optional mamba_ssm; no in-tree JIT kernel"},
        {"operator": "jagged_layer_norm", "status": "unavailable_in_pinned_panel",
         "reason": "Pinned operator has PyTorch and torch.compile backends, no direct Triton JIT launch"},
    ]


def coverage(root, platform, experiment, tau, *, cases=CASES):
    other = "matrix" if platform == "tuolumne" else "tuolumne"
    rows = []
    for case in cases:
        reports = {}
        for device in (platform, other):
            path = root / "tau-profiles" / tau / f"experiment-{experiment}" / device / case.case_id / "report.json"
            reports[device] = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        row = {"case": case.case_id, "operator": case.operator, "status": reports[platform]["status"],
               "counterpart_status": reports[other]["status"], "stage": reports[platform].get("stage"),
               "exclusion": reports[platform].get("exclusion")}
        row["local_validated"] = row["status"] in {"validated", "complete"}
        row["common_validated"] = (row["local_validated"] and row["counterpart_status"] in {"validated", "complete"}
            and reports[platform].get("source_identity", {}).get("source_hash") is not None
            and reports[platform].get("source_identity", {}).get("source_hash")
                == reports[other].get("source_identity", {}).get("source_hash")
            and reports[platform].get("hardware_profile", {}).get("device", {}).get("tau_semantics")
                == reports[other].get("hardware_profile", {}).get("device", {}).get("tau_semantics"))
        rows.append(row)
    common = sorted({r["operator"] for r in rows if r["common_validated"]})
    return {"experiment": experiment, "tau_name": tau, "platform": platform,
        "declared_cases": len(cases), "statuses": dict(Counter(r["status"] for r in rows)),
        "local_families": sorted({r["operator"] for r in rows if r["local_validated"]}),
        "common_families": common, "meets_twelve_common_families": len(common) >= 12,
        "scope": "per-device diagnostics; broad cross-platform claim requires twelve common families",
        "reserves": reserve_inventory() if len(common) < 12 else [], "cases": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-root", type=Path, required=True)
    p.add_argument("--platform", choices=("tuolumne", "matrix"), required=True)
    p.add_argument("--require-broad-panel", action="store_true")
    args = p.parse_args()
    panels = [coverage(args.suite_root, args.platform, experiment, tau)
              for experiment in (4, 5, 6) for tau in TAU_NAMES]
    path = args.suite_root / "shared" / args.platform / "preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"panels": panels}, indent=2) + "\n")
    print(path)
    for panel in panels:
        print(f"E{panel['experiment']}/{panel['tau_name']}: {len(panel['local_families'])} local, "
              f"{len(panel['common_families'])} common families; {panel['statuses']}")
    if args.require_broad_panel and not all(p["meets_twelve_common_families"] for p in panels):
        raise SystemExit("The twelve-family cross-platform requirement is not met; see the complete preflight inventory.")


if __name__ == "__main__":
    main()
