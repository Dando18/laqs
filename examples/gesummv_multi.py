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
    PerLaneTemporalRegions,
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
    N = 512
    sampled_j = 32
    repetition_weight = float(N // sampled_j)

    # This is a hand-written, compressed logical trace of the inner loop of
    #
    #   tmp[i] += A[i, j] * x[j]
    #   y[i]   += B[i, j] * x[j]
    #
    # with lanes parallelized over i.  MatrixSpec describes the logical arrays;
    # only target=True arrays get new layouts synthesized.  x stays row-major so
    # its fixed cost can be included as context, while y is present to document
    # the kernel signature but has no modeled store event in this read-focused
    # example.  role is descriptive metadata rather than an access generator.
    matrices = (
        MatrixSpec("A", (N, N), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("B", (N, N), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("x", (N,), 8, ("j",), target=False, role="read"),
        MatrixSpec("y", (N,), 8, ("i",), target=False, role="write"),
    )

    events: list[MemoryEvent] = []
    sequences: list[EventSequence] = []
    order = 0

    # Model one 128-thread workgroup as two 64-lane waves and sample the first
    # sampled_j iterations.  For a matrix load all lanes hold the same j and lane L
    # handles i = 64 * wave + L.  Thus a row-major layout makes adjacent lanes
    # stride by an entire N-element row; a layout with low i bits can instead
    # place this wave's accesses near one another.
    for wave in range(2):
        sequence_ids: list[str] = []
        for j in range(sampled_j):
            for array in ("A", "B"):
                event_id = f"{array}.w{wave}.j{j}"
                accesses = [
                    Access(array, (64 * wave + lane, j), lane=lane, kind="read")
                    for lane in range(64)
                ]
                events.append(
                    MemoryEvent.make(
                        event_id,
                        f"{array}.load",
                        accesses,
                        group=f"wg0.wave{wave}",
                        order=order,
                        # The explicit j values stand in for a repeated
                        # inner-loop pattern.  Weight changes an edge's score,
                        # not its access set or its number of lanes.
                        weight=repetition_weight,
                        metadata={"workgroup": "wg0", "wave": wave, "step": j, "phase": "sample"},
                    )
                )
                sequence_ids.append(event_id)
                order += 1
            x_id = f"x.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    x_id,
                    "x.load",
                    # x[j] is a wave broadcast: all 64 Access objects refer to
                    # one logical element.  Hyperedges deduplicate that point.
                    [Access("x", (j,), lane=lane, kind="read") for lane in range(64)],
                    group=f"wg0.wave{wave}",
                    order=order,
                    weight=repetition_weight,
                    metadata={"workgroup": "wg0", "wave": wave, "step": j, "phase": "sample"},
                )
            )
            sequence_ids.append(x_id)
            order += 1

        # Temporal objectives only combine events named by an explicit local
        # sequence.  Keeping one sequence per wave avoids inventing an execution
        # order between independent waves.  Its weight serves the same compressed
        # multiplicity purpose as event.weight; the builders do not multiply the
        # two weights together.
        sequences.append(EventSequence.make(f"wave{wave}.phase", sequence_ids, weight=repetition_weight))

    matrix_reads = EventFilter.make(arrays=("A", "B"), kinds=("read",))

    # An objective first turns a selected access scope into a logical hyperedge,
    # then counts how many aligned intervals of region_bytes that edge intersects
    # after applying a candidate layout.  region_bytes is therefore not the total
    # footprint of the scope.  For example, one 64-lane FP64 matrix event carries
    # 512 B and has a packing lower bound of eight 64 B regions in fine64.
    objectives = (
        SimultaneousRegions(
            "fine64",
            64,
            event_filter=matrix_reads,
            provenance="grounded",
            description="one full wave load",
        ),
        # These hypotheses split each wave event into aligned contiguous lane
        # groups.  Each region has exactly enough capacity for that group's
        # scalar payload: 8 lanes * 8 B = 64 B, ..., 64 lanes * 8 B = 512 B.
        # This exposes locality at several possible coalescing granularities.
        LanePrefixRegions(
            "lane",
            levels=((8, 64), (16, 128), (32, 256), (64, 512)),
            event_filter=matrix_reads,
            provenance="hypothesis",
        ),
        # Follow each lane through consecutive inner-loop iterations. In
        # particular, window16 asks whether one lane's 16 FP64 matrix loads can
        # fit in one aligned 128 B region.
        PerLaneTemporalRegions(
            "lane_window128",
            region_bytes=128,
            windows=(2, 4, 8, 16, 32),
            event_filter=matrix_reads,
        ),
        # Unlike fine64, this keeps the whole 64-lane event as one edge and asks
        # whether its complete 512 B payload can occupy one aligned 512 B region.
        SimultaneousRegions(
            "payload512",
            512,
            event_filter=matrix_reads,
            provenance="hypothesis",
        ),
        # group_by(workgroup, step) joins the two waves at a fixed j.  A and B
        # remain separate edges because they are separate allocations.  Each
        # per-matrix edge is the 128-row column panel
        # {(i, j) | i = 0..127}, whose FP64 payload is exactly 1024 B.
        GroupedRegions(
            "panel1024",
            1024,
            group_by=("workgroup", "step"),
            event_filter=matrix_reads,
            provenance="hypothesis",
            #search=False,
        ),
        # window=None unions every selected event in each wave's local sequence.
        # Per target matrix this is sampled_j columns x 64 rows. The 4096 B
        # granularity is a broader phase-locality hypothesis;
        # A and B are still scored independently, not packed into one allocation.
        TemporalWindowRegions(
            "phase4096",
            4096,
            window=None,
            event_filter=matrix_reads,
            provenance="hypothesis",
        ),
        # Rebuild the same complete-sequence scope with all reads, which adds the
        # broadcast x points.  x has a fixed row-major layout.  search=False means
        # this component is reported after layouts are chosen but does not guide
        # either A's or B's layout search.
        TemporalWindowRegions(
            "joint_context4096",
            4096,
            window=None,
            event_filter=EventFilter.make(kinds=("read",)),
            provenance="report-only",
            search=False,
            description="includes fixed context vector x",
        ),
    )

    # The Pareto dimensions are the objective components that actually rank and
    # retain candidates.  LanePrefixRegions creates all four lane-level scores,
    # but lane16 is the representative checkpoint named here; the other levels
    # remain visible in the report.  runs/xors approximate address-generation
    # complexity, and adj_gap penalizes distance between consecutive lanes.
    policy = ScorePolicy(
        kind="pareto",
        order=(
            "lane_window128.window16",
            "fine64",
            "lane.lane16.128B",
            "payload512",
            "panel1024",
            "phase4096",
            "runs",
            "xors",
            "adj_gap",
        ),
        frontier_limit=24,
        paths_per_state=8,
    )

    # The canonical tile hypotheses are deliberately tall: the matrix events
    # vary i while holding j fixed. Wider tiles let the solver trade wave
    # locality against per-lane temporal locality; the 16-column hypotheses can
    # place a lane's key 16-j window within one 128 B region. Arbitrary bit-linear
    # layouts are much more expensive to enumerate, so they are tried only on
    # the two small 4-column shapes. The solver also adds its standard
    # row-major/column-major controls (and a global canonical hypothesis).
    config = SolverConfig(
        policy=policy,
        tile_shapes={
            "A": (
                (8, 1),
                (8, 8),
                (16, 1),
                (16, 4),
                (16, 8),
                (16, 16),
                (32, 2),
                (64, 1),
                (64, 4),
                (64, 16),
                (128, 4),
                (128, 16),
            ),
            "B": (
                (8, 1),
                (8, 8),
                (16, 1),
                (16, 4),
                (16, 8),
                (16, 16),
                (32, 2),
                (64, 1),
                (64, 4),
                (64, 16),
                (128, 4),
                (128, 16),
            ),
        },
        general_tile_shapes={"A": ((8, 4), (16, 4)), "B": ((8, 4), (16, 4))},
        max_inner_bits=9,
        canonical_candidates_per_tile=6,
        general_max_inner_bits=6,
        general_exact_rank=6,
        general_candidates_per_tile=3,
        primary_tolerance=0.05,
        per_array_candidates=12,
        joint_candidates=12,
    )

    # solve() expands the objective builders, searches A and B independently,
    # retains bounded candidate families, and finally forms bounded joint A/B
    # choices while adding the fixed context-array scores.
    return RelayProblem(
        matrices=matrices,
        events=tuple(events),
        sequences=tuple(sequences),
        objectives=objectives,
        config=config,
        name="gesummv_two_target_matrices",
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
