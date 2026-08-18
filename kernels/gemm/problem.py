"""Construct memory events and objectives for a simple square GEMM.

Each thread computes one output ``C[i, j]``.  A wave first reads C, then loads
``A[i, k]`` and ``B[k, j]`` for every inner-loop step, and finally writes C.
The trace models one representative workgroup, with x / matrix-j threads
varying fastest.  ``Config`` controls the square problem size and workgroup
shape; ``HardwareConfig`` currently fixes a 64-lane wavefront.
"""


from dataclasses import dataclass
import math
from relay import (
    Access,
    EventFilter,
    EventSequence,
    LanePrefixRegions,
    MatrixSpec,
    MemoryEvent,
    PerLaneTemporalRegions,
    SimultaneousRegions,
)
from relay.objectives import ObjectiveSpec


@dataclass(frozen=True)
class Config:
    problem_size: int = 256
    block_size: tuple[int, int, int] = (32, 32, 1)


@dataclass(frozen=True)
class HardwareConfig:
    # TODO later replace this with a more general abstraction so 
    # we can easily switch between different GPU targets.
    wavefront_size: int = 64


def build_config(**kwargs) -> Config:
    return Config(**kwargs)


def get_matrices(config: Config) -> tuple[MatrixSpec, ...]:
    N = config.problem_size
    return (
        MatrixSpec("A", (N, N), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("B", (N, N), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("C", (N, N), 8, ("i", "j"), target=True, role="write"),
    )

def get_events_and_sequences(
    config: Config,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    N = config.problem_size
    blk_sz = config.block_size
    total_threads_per_blk = blk_sz[0] * blk_sz[1] * blk_sz[2]
    hardware = HardwareConfig()

    events: list[MemoryEvent] = []
    order = 0

    # there are ceil(total_threads_per_blk / wavefront_size) wavefronts per block
    total_wavefronts_per_blk = math.ceil(total_threads_per_blk / hardware.wavefront_size)
    sequences: list[EventSequence] = []

    for wave in range(total_wavefronts_per_blk):
        # Model a representative workgroup at blockIdx=(0, 0, 0).  HIP/CUDA
        # linearizes threads with x (the matrix j dimension here) varying
        # fastest.  A final, partial wave contains only the lanes backed by a
        # thread in the workgroup; the kernel bounds check removes any lanes
        # whose output coordinate is outside the matrix.
        lane_coordinates: list[tuple[int, int, int]] = []
        for lane in range(hardware.wavefront_size):
            thread = wave * hardware.wavefront_size + lane
            if thread >= total_threads_per_blk:
                break

            thread_j = thread % blk_sz[0]
            thread_i = (thread // blk_sz[0]) % blk_sz[1]
            if thread_i < N and thread_j < N:
                lane_coordinates.append((lane, thread_i, thread_j))

        # A wave can be completely inactive when a block is larger than a
        # small problem. MemoryEvent does not permit empty access sets.
        if not lane_coordinates:
            continue

        group = f"wg0.wave{wave}"
        common_metadata = {"workgroup": "wg0", "wave": wave}
        sequence_ids: list[str] = []

        c_load_id = f"C.w{wave}.load"
        events.append(
            MemoryEvent.make(
                c_load_id,
                "C.load",
                [
                    Access("C", (i, j), lane=lane, kind="read")
                    for lane, i, j in lane_coordinates
                ],
                group=group,
                order=order,
                metadata={**common_metadata, "phase": "prologue"},
            )
        )
        sequence_ids.append(c_load_id)
        order += 1

        # Every loop iteration is represented by one wave-wide load from A and
        # one from B.  Lanes in a wave share k, while A varies with output row
        # i and B varies with output column j.
        for k in range(N):
            for array in ("A", "B"):
                event_id = f"{array}.w{wave}.k{k}"
                if array == "A":
                    accesses = [
                        Access("A", (i, k), lane=lane, kind="read")
                        for lane, i, _ in lane_coordinates
                    ]
                else:
                    accesses = [
                        Access("B", (k, j), lane=lane, kind="read")
                        for lane, _, j in lane_coordinates
                    ]

                events.append(
                    MemoryEvent.make(
                        event_id,
                        f"{array}.load",
                        accesses,
                        group=group,
                        order=order,
                        metadata={**common_metadata, "step": k, "phase": "inner"},
                    )
                )
                sequence_ids.append(event_id)
                order += 1

        c_store_id = f"C.w{wave}.store"
        events.append(
            MemoryEvent.make(
                c_store_id,
                "C.store",
                [
                    Access("C", (i, j), lane=lane, kind="write")
                    for lane, i, j in lane_coordinates
                ],
                group=group,
                order=order,
                metadata={**common_metadata, "phase": "epilogue"},
            )
        )
        sequence_ids.append(c_store_id)
        sequences.append(
            EventSequence.make(
                f"wave{wave}",
                sequence_ids,
                metadata=common_metadata,
            )
        )
        order += 1

    return tuple(events), tuple(sequences)


def get_objectives(config: Config) -> tuple[ObjectiveSpec, ...]:
    """Describe the GEMM access scopes and aligned byte granularities."""

    del config
    matrix_reads = EventFilter.make(arrays=("A", "B", "C"), kinds=("read",))
    A_reads = EventFilter.make(arrays=("A",), kinds=("read",))
    B_reads = EventFilter.make(arrays=("B",), kinds=("read",))
    C_writes = EventFilter.make(arrays=("C",), kinds=("write",))

    return (
        SimultaneousRegions(
            "fine64",
            64,
            event_filter=matrix_reads,
            provenance="grounded",
            description="one full wave load",
        ),
        SimultaneousRegions(
            "C.store64",
            64,
            event_filter=C_writes,
            provenance="grounded",
            description="one full wave store",
        ),
        PerLaneTemporalRegions(
            "A.lane_temporal128",
            128,
            event_filter=A_reads,
            windows=(16,),
            provenance="grounded",
            description="per-lane temporal reuse of A",
        ),
        PerLaneTemporalRegions(
            "B.lane_temporal128",
            128,
            event_filter=B_reads,
            windows=(16,),
            provenance="grounded",
            description="per-lane temporal reuse of B",
        ),
        LanePrefixRegions(
            "B.lane",
            levels=((8, 64), (16, 128), (32, 256), (64, 256)),
            event_filter=B_reads,
            provenance="grounded",
        ),
        LanePrefixRegions(
            "C.lane",
            levels=((8, 64), (16, 128), (32, 256), (64, 512)),
            event_filter=C_writes,
            provenance="grounded",
        ),
    )
