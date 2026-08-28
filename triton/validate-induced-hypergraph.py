#!/usr/bin/env python3
"""Validate RELAY's induced hypergraph with one observed Triton tile load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
from triton.language import core
from triton.tools import LinearLayout as NativeLinearLayout


TILE_ELEMENTS = 64
ELEMENT_BYTES = 4


@core.extern
def local_workitem_id(axis, _semantic=None):
    """Return the AMD work-item ID without reconstructing it from EXEC."""

    return core.extern_elementwise(
        "",
        "",
        [axis],
        {
            (core.dtype("int32"),): (
                "__ockl_get_local_id",
                core.dtype("int64"),
            )
        },
        is_pure=True,
        _semantic=_semantic,
    ).to(core.int32, _semantic=_semantic)


@triton.jit
def trace_tile_load(
    source,
    loaded_values,
    lanes_by_coord,
    byte_offsets,
    BLOCK_SIZE: tl.constexpr,
    ELEMENT_WIDTH: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(source + offsets)
    lane = local_workitem_id(0)
    tl.store(loaded_values + offsets, values)
    tl.store(lanes_by_coord + offsets, lane)
    tl.store(byte_offsets + offsets, offsets * ELEMENT_WIDTH)


def validate_lane_trace(
    observed_lanes: list[int], expected_lanes: set[int]
) -> None:
    coords_by_lane: dict[int, list[int]] = {}
    for coord, lane in enumerate(observed_lanes):
        coords_by_lane.setdefault(lane, []).append(coord)

    duplicates = {
        lane: coords
        for lane, coords in coords_by_lane.items()
        if len(coords) > 1
    }
    missing = sorted(expected_lanes - coords_by_lane.keys())
    unexpected = sorted(coords_by_lane.keys() - expected_lanes)
    if duplicates or missing or unexpected:
        raise ValueError(
            "observed work-item IDs do not match the compiled lane cohort: "
            f"duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )


def run_probe(transaction_bytes: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the stage-0 probe requires a Flux GPU allocation")

    source = torch.arange(TILE_ELEMENTS, dtype=torch.int32, device="cuda")
    loaded = torch.empty_like(source)
    lanes = torch.empty_like(source)
    byte_offsets = torch.empty_like(source)
    compiled = trace_tile_load[(1,)](
        source,
        loaded,
        lanes,
        byte_offsets,
        BLOCK_SIZE=TILE_ELEMENTS,
        ELEMENT_WIDTH=ELEMENT_BYTES,
        num_warps=1,
    )
    torch.cuda.synchronize()

    if not torch.equal(source, loaded):
        raise ValueError("the instrumented Triton load returned incorrect values")

    # Import RELAY only after Triton because this script lives in relay/triton.
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from relay import (
        HardwareLocation,
        MatrixSpec,
        ObservedAccess,
        TritonLinearLayout,
        extract_blocked_layout,
        induce_memory_event,
        row_major_layout,
        validate_induced_hypergraph,
    )

    blocked = extract_blocked_layout(compiled.asm["ttgir"])
    extracted = TritonLinearLayout.from_blocked(
        (TILE_ELEMENTS,),
        size_per_thread=blocked.size_per_thread,
        threads_per_warp=blocked.threads_per_warp,
        warps_per_cta=blocked.warps_per_cta,
        order=blocked.order,
    )
    native = NativeLinearLayout.from_bases(
        extracted.bases,
        [name for name, _ in extracted.out_dims],
        [size for _, size in extracted.out_dims],
    )
    execution_layout = TritonLinearLayout.from_triton(native)
    cohort = execution_layout.locations(
        fixed={"register": 0, "warp": 0, "block": 0}
    )
    matrix = MatrixSpec("source", (TILE_ELEMENTS,), ELEMENT_BYTES, ("i",))
    induced = induce_memory_event(
        execution_layout,
        matrix,
        cohort,
        id="tile.load.wave0",
        site="trace_tile_load.load",
    )

    observed_lanes = lanes.cpu().tolist()
    validate_lane_trace(
        observed_lanes,
        {location.value("lane") for location in cohort},
    )
    observed_offsets = byte_offsets.cpu().tolist()
    observations = tuple(
        ObservedAccess(
            HardwareLocation.make(
                {
                    "register": 0,
                    "lane": observed_lanes[coord],
                    "warp": 0,
                    "block": 0,
                }
            ),
            (coord,),
            observed_offsets[coord],
        )
        for coord in range(TILE_ELEMENTS)
    )
    validation = validate_induced_hypergraph(
        induced,
        matrix,
        row_major_layout(matrix),
        observations,
        transaction_bytes=transaction_bytes,
    )
    validation.require_valid()

    return {
        "valid": validation.valid,
        "event": validation.event_id,
        "tile_shape": list(matrix.shape),
        "element_bytes": matrix.element_bytes,
        "hardware_locations": len(cohort),
        "compiled_blocked_layout": {
            name: list(values) for name, values in blocked.as_dict().items()
        },
        "transaction_bytes": transaction_bytes,
        "expected_transaction_ids": list(
            validation.expected_transaction_ids
        ),
        "observed_transaction_ids": list(
            validation.observed_transaction_ids
        ),
        "quotient_count": validation.expected_quotient_count,
        "observed_transaction_count": validation.observed_transaction_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction-bytes", type=int, default=128)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = run_probe(args.transaction_bytes)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
