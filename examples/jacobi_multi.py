from __future__ import annotations

import argparse

from relay import (
    Access,
    EventFilter,
    EventSequence,
    GroupedRegions,
    LanePrefixRegions,
    MatrixSpec,
    MemoryEvent,
    RelayProblem,
    ScorePolicy,
    SimultaneousRegions,
    SolverConfig,
    TemporalWindowRegions,
    dump_json,
    print_report,
    solve,
)


def build_problem() -> RelayProblem:
    # Model one interior step of the five-point Jacobi stencil:
    #
    #   output[i, j] = f(input[i, j], input[i-1, j], input[i+1, j],
    #                    input[i, j-1], input[i, j+1])
    #
    # Both arrays are targets because the best layout for the five overlapping
    # input loads need not be the best layout for the single output store.
    matrices = (
        MatrixSpec("input", (256, 256), 4, ("i", "j"), target=True, role="read"),
        MatrixSpec("output", (256, 256), 4, ("i", "j"), target=True, role="write"),
    )
    events: list[MemoryEvent] = []
    sequences: list[EventSequence] = []
    offsets = {
        "center": (0, 0),
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }

    order = 0

    # The modeled 16x16 workgroup is split into four 8x8 wave-owned quadrants.
    # Within a wave, divmod(lane, 8) maps lanes 0..63 row-major over its 8x8
    # output patch.  This is the example's explicit scheduling assumption; the
    # solver consumes these logical coordinates and does not derive lane mapping
    # from a HIP launch or compiler IR.
    for wave in range(4):
        quadrant_i = (wave // 2) * 8
        quadrant_j = (wave % 2) * 8
        sequence_ids: list[str] = []

        # Starting at (64, 64) keeps every +/-1 neighbor in bounds and makes this
        # a representative interior workgroup, free of boundary-condition events.
        # Each named site represents one dynamic vector memory instruction: the
        # same 64 output sites shifted by that stencil neighbor's (di, dj).
        for site, (di, dj) in offsets.items():
            event_id = f"{site}.w{wave}"
            accesses = []
            for lane in range(64):
                li, lj = divmod(lane, 8)
                accesses.append(
                    Access(
                        "input",
                        (64 + quadrant_i + li + di, 64 + quadrant_j + lj + dj),
                        lane=lane,
                        kind="read",
                    )
                )
            events.append(
                MemoryEvent.make(
                    event_id,
                    f"input.{site}",
                    accesses,
                    group=f"wg0.wave{wave}",
                    order=order,
                    # One interior workgroup geometry stands in for repeated
                    # workgroups.  The uniform weight scales scores only; it
                    # neither adds lanes nor changes the logical footprint.
                    weight=256.0,
                    metadata={"workgroup": "wg0", "wave": wave, "phase": "stencil"},
                )
            )
            sequence_ids.append(event_id)
            order += 1

        # The store is centered on the unshifted 8x8 patch owned by this wave,
        # matching the output coordinate on the left-hand side of the stencil.
        store_id = f"store.w{wave}"
        stores = []
        for lane in range(64):
            li, lj = divmod(lane, 8)
            stores.append(
                Access(
                    "output",
                    (64 + quadrant_i + li, 64 + quadrant_j + lj),
                    lane=lane,
                    kind="write",
                )
            )
        events.append(
            MemoryEvent.make(
                store_id,
                "output.store",
                stores,
                group=f"wg0.wave{wave}",
                order=order,
                weight=256.0,
                metadata={"workgroup": "wg0", "wave": wave, "phase": "stencil"},
            )
        )
        sequence_ids.append(store_id)

        # The sequence records modeled program order for temporal objectives.
        # Waves get separate sequences because no global ordering between them is
        # assumed.  Temporal builders use sequence.weight, while simultaneous
        # and grouped builders use event weights, so these values are not doubled.
        sequences.append(EventSequence.make(f"wave{wave}", sequence_ids, weight=256.0))
        order += 1

    input_reads = EventFilter.make(arrays=("input",), kinds=("read",))
    output_writes = EventFilter.make(arrays=("output",), kinds=("write",))

    # Each objective defines (1) which logical accesses should stay together and
    # (2) the size of an aligned interval in the candidate physical layout.  The
    # score is the weighted number of such intervals touched.  A 64-lane FP32
    # event contains 256 B, so input.wave128/output.wave128 have an ideal lower
    # bound of two 128 B regions; "wave128" does not mean the wave carries 128 B.
    objectives = (
        SimultaneousRegions("input.wave128", 128, event_filter=input_reads, provenance="grounded"),
        # Split each input instruction into contiguous lane prefixes.  At every
        # level the region matches the subgroup's scalar payload, from
        # 4 lanes * 4 B = 16 B through 64 lanes * 4 B = 256 B.
        LanePrefixRegions(
            "input.lane",
            levels=((4, 16), (8, 32), (16, 64), (32, 128), (64, 256)),
            event_filter=input_reads,
            provenance="hypothesis",
        ),
        # A five-event sliding window captures temporal reuse among neighboring
        # stencil instructions.  There are six events in a sequence (five loads
        # plus one store), so two windows are formed.  The input filter makes the
        # first contain all five loads and the second contain the final four loads
        # (the store occupies the fifth sequence position but contributes no
        # input point).  Both are scored in aligned 128 B units.
        TemporalWindowRegions(
            "input.window128",
            128,
            window=5,
            event_filter=input_reads,
            provenance="hypothesis",
        ),
        # All four waves share workgroup=wg0 and phase=stencil.  Grouping on those
        # fields therefore unions the five input footprints over the full 16x16
        # output tile.  The distinct input footprint is the tile plus a one-cell
        # north/south/east/west halo (320 floats = 1280 B), evaluated in 1024 B
        # regions to favor workgroup-scale clustering without requiring one-region
        # packing, which is impossible by capacity alone.
        GroupedRegions(
            "input.workgroup1024",
            1024,
            group_by=("workgroup", "phase"),
            event_filter=input_reads,
            provenance="hypothesis",
        ),
        SimultaneousRegions("output.wave128", 128, event_filter=output_writes, provenance="grounded"),
        # Stores have no cross-instruction stencil reuse, so their hypotheses stop
        # at simultaneous contiguous-lane groupings.  The 4-lane input-only level
        # is omitted here to keep the output objective family smaller.
        LanePrefixRegions(
            "output.lane",
            levels=((8, 32), (16, 64), (32, 128), (64, 256)),
            event_filter=output_writes,
            provenance="hypothesis",
        ),
        # With window=None, each wave contributes its complete five-load-plus-store
        # sequence.  Accesses are still separated by allocation into one input edge
        # and one output edge; this does not claim that the arrays share a physical
        # 1024 B region.  It is report-only so it cannot steer the layout search.
        TemporalWindowRegions(
            "joint.phase1024",
            1024,
            window=None,
            provenance="report-only",
            search=False,
            description="input and output footprint in one wave sequence",
        ),
    )

    # Only names in this order participate in Pareto dominance and candidate
    # retention.  LanePrefixRegions still emits its other levels for reporting.
    # The ordering keeps the grounded wave scores prominent, includes one lane16
    # checkpoint and the broader reuse scopes, then prefers simpler/less-strided
    # address maps via runs, xors, and adj_gap.
    policy = ScorePolicy(
        kind="pareto",
        order=(
            "input.wave128",
            "input.lane.lane16.64B",
            "input.window128",
            "output.wave128",
            "output.lane.lane16.64B",
            "input.workgroup1024",
            "runs",
            "xors",
            "adj_gap",
        ),
        frontier_limit=32,
        paths_per_state=8,
    )

    # These canonical tile shapes cover the 4/8-lane substructure, an 8x8 wave
    # patch, and the 16x16 workgroup scale, with rectangular alternatives for
    # trading row and column locality.  General bit-linear search is restricted
    # to input's 8x8 tile because it is costlier and the input stencil has the
    # richer reuse pattern; output's empty tuple disables that branch explicitly.
    tile_family = ((4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (16, 16), (32, 8))
    config = SolverConfig(
        policy=policy,
        tile_shapes={"input": tile_family, "output": tile_family},
        general_tile_shapes={"input": ((8, 8),), "output": ()},
        canonical_candidates_per_tile=8,
        general_max_inner_bits=6,
        general_exact_rank=6,
        general_candidates_per_tile=4,
        primary_tolerance=0.05,
        per_array_candidates=14,
        joint_candidates=16,
    )

    # solve() expands the access scopes into per-array hyperedges, searches the
    # two arrays independently, then combines their retained layouts into bounded
    # joint candidates.  It never runs the Jacobi kernel in this example.
    return RelayProblem(
        matrices=matrices,
        events=tuple(events),
        sequences=tuple(sequences),
        objectives=objectives,
        config=config,
        name="jacobi_input_and_output",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument("--layouts", type=int, default=2)
    args = parser.parse_args()

    result = solve(build_problem())
    print_report(result, max_candidates=args.show, show_layouts=args.layouts)
    if args.json:
        dump_json(result, args.json)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
