from __future__ import annotations

from relay import (
    Access,
    MatrixSpec,
    MemoryEvent,
    RelayProblem,
    ScorePolicy,
    SimultaneousRegions,
    SolverConfig,
    print_report,
    solve,
)


def main() -> None:
    # This deliberately tiny problem isolates a layout conflict.  A 2x2 FP32
    # matrix has four elements, while an aligned 8 B physical region can contain
    # exactly two of them.
    matrix = MatrixSpec("M", (2, 2), 4, ("i", "j"))

    # Think of each tuple as the two lanes of one vector memory instruction.  The
    # first two instructions access logical rows; the last two access logical
    # columns.  MemoryEvent.make preserves this simultaneous access scope, and
    # the lane ids let the report also compute consecutive-lane gap metrics.
    point_sets = (
        ((0, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
    )
    events = tuple(
        MemoryEvent.make(
            f"event{index}",
            f"site{index}",
            [Access("M", point, lane=lane) for lane, point in enumerate(points)],
        )
        for index, points in enumerate(point_sets)
    )

    # SimultaneousRegions makes each two-point event one hyperedge and scores how
    # many aligned 8 B regions it intersects after layout.  Any single linear
    # ordering partitions the four elements into only two aligned pairs, so it
    # cannot make all four requested row/column pairs one-region accesses.  The
    # packing lower bound is four total regions (one per event), but the best
    # realizable score is six: two pairs cost one region and two cost two regions.
    problem = RelayProblem(
        matrices=(matrix,),
        events=events,
        sequences=(),
        objectives=(SimultaneousRegions("pair8", 8),),
        config=SolverConfig(
            # First minimize the access-region count.  runs then favors fewer
            # alternations between i/j bits in a canonical address expression;
            # xors penalizes general bit-linear maps that mix logical bits.
            policy=ScorePolicy("lexicographic", ("pair8", "runs", "xors")),
            # The only tile is the complete matrix.  Searching it with both the
            # canonical bit-interleaving grammar and the general invertible GF(2)
            # grammar shows that the conflict is intrinsic, not just a failure to
            # consider row-major versus column-major.
            tile_shapes={"M": ((2, 2),)},
            general_tile_shapes={"M": ((2, 2),)},
            max_inner_bits=2,
            general_exact_rank=2,
            per_array_candidates=8,
        ),
        name="row_column_conflict",
    )
    print_report(solve(problem), max_candidates=8, show_layouts=4)


if __name__ == "__main__":
    main()
