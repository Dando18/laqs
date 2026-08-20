"""Construct matrices, memory events, and sequences for a simple square GEMM.

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
    EventSequence,
    MatrixSpec,
    MemoryEvent,
)


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
        MatrixSpec(
            "C", (N, N), 8, ("i", "j"), target=True, role="read_write"
        ),
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

    # There are ceil(total_threads_per_blk / wavefront_size) waves per block.
    total_wavefronts_per_blk = math.ceil(
        total_threads_per_blk / hardware.wavefront_size
    )
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
                metadata={**common_metadata, "phase": "prologue", "step": 0},
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
                metadata={**common_metadata, "phase": "epilogue", "step": 0},
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
