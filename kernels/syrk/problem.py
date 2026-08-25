"""Construct matrices, memory events, and sequences for a square SYRK kernel.

Each thread computes one element of the full symmetric result

``C[i, j] = alpha * sum_k A[i, k] * A[j, k] + beta * C[i, j]``.

The trace models one representative two-dimensional workgroup. Threads vary
``j`` fastest, matching the HIP evaluator. Both logical operand streams refer
to A, which makes SYRK useful for studying the compromise between wave-wide
accesses down A's first dimension and per-thread traversal along its second
dimension. A and C are layout targets.
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
    wavefront_size: int = 64


def build_config(**kwargs) -> Config:
    return Config(**kwargs)


def get_matrices(config: Config) -> tuple[MatrixSpec, ...]:
    n = config.problem_size
    return (
        MatrixSpec("A", (n, n), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("C", (n, n), 8, ("i", "j"), target=True, role="read_write"),
    )


def get_events_and_sequences(
    config: Config,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    """Model one workgroup's C access and complete pair of A row streams."""

    n = config.problem_size
    block_x, block_y, block_z = config.block_size
    threads_per_block = block_x * block_y * block_z
    wavefront_size = HardwareConfig().wavefront_size
    wave_count = math.ceil(threads_per_block / wavefront_size)
    events: list[MemoryEvent] = []
    sequences: list[EventSequence] = []
    order = 0

    for wave in range(wave_count):
        lanes: list[tuple[int, int, int]] = []
        for lane in range(wavefront_size):
            thread = wave * wavefront_size + lane
            if thread >= threads_per_block:
                break
            j = thread % block_x
            i = (thread // block_x) % block_y
            if i < n and j < n:
                lanes.append((lane, i, j))
        if not lanes:
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
                    for lane, i, j in lanes
                ],
                group=group,
                order=order,
                metadata={**common_metadata, "phase": "prologue", "step": 0},
            )
        )
        sequence_ids.append(c_load_id)
        order += 1

        for k in range(n):
            for stream, site in (
                ("row_i", "A.row_i.load"),
                ("row_j", "A.row_j.load"),
            ):
                event_id = f"A.{stream}.w{wave}.k{k}"
                if stream == "row_i":
                    accesses = [
                        Access("A", (i, k), lane=lane, kind="read")
                        for lane, i, _ in lanes
                    ]
                else:
                    accesses = [
                        Access("A", (j, k), lane=lane, kind="read")
                        for lane, _, j in lanes
                    ]
                events.append(
                    MemoryEvent.make(
                        event_id,
                        site,
                        accesses,
                        group=group,
                        order=order,
                        metadata={
                            **common_metadata,
                            "phase": "inner",
                            "step": k,
                            "stream": stream,
                        },
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
                    for lane, i, j in lanes
                ],
                group=group,
                order=order,
                metadata={**common_metadata, "phase": "epilogue", "step": 0},
            )
        )
        sequence_ids.append(c_store_id)
        order += 1
        sequences.append(
            EventSequence.make(
                f"wave{wave}", sequence_ids, metadata=common_metadata
            )
        )

    return tuple(events), tuple(sequences)
