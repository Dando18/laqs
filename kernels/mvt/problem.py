"""Construct matrices, memory events, and sequences for square MVT.

Each thread computes two outputs from one matrix in opposing directions::

    x1[i] = beta * x1[i] + alpha * sum_j A[i, j] * y1[j]
    x2[i] = beta * x2[i] + alpha * sum_j A[j, i] * y2[j]

The trace models one representative one-dimensional workgroup. Consecutive
lanes vary ``i``. At an inner-loop step, the first matrix load therefore
varies down a column across the wave, while the transposed load varies across
a row. ``A`` is the only layout target; the four vectors retain fixed
contiguous layouts as context arrays.
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
    block_size: int = 128


@dataclass(frozen=True)
class HardwareConfig:
    wavefront_size: int = 64


def build_config(**kwargs) -> Config:
    return Config(**kwargs)


def get_matrices(config: Config) -> tuple[MatrixSpec, ...]:
    n = config.problem_size
    return (
        MatrixSpec("A", (n, n), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("y1", (n,), 8, ("j",), target=False, role="read"),
        MatrixSpec("y2", (n,), 8, ("j",), target=False, role="read"),
        MatrixSpec("x1", (n,), 8, ("i",), target=False, role="read_write"),
        MatrixSpec("x2", (n,), 8, ("i",), target=False, role="read_write"),
    )


def get_events_and_sequences(
    config: Config,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    """Model one workgroup's two matrix streams and vector accesses."""

    n = config.problem_size
    hardware = HardwareConfig()
    wave_count = math.ceil(config.block_size / hardware.wavefront_size)
    events: list[MemoryEvent] = []
    sequences: list[EventSequence] = []
    order = 0

    for wave in range(wave_count):
        lanes = [
            (lane, wave * hardware.wavefront_size + lane)
            for lane in range(hardware.wavefront_size)
            if wave * hardware.wavefront_size + lane < config.block_size
            and wave * hardware.wavefront_size + lane < n
        ]
        if not lanes:
            continue

        group = f"wg0.wave{wave}"
        common_metadata = {"workgroup": "wg0", "wave": wave}
        sequence_ids: list[str] = []

        for array in ("x1", "x2"):
            event_id = f"{array}.w{wave}.load"
            events.append(
                MemoryEvent.make(
                    event_id,
                    f"{array}.load",
                    [
                        Access(array, (i,), lane=lane, kind="read")
                        for lane, i in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={**common_metadata, "phase": "prologue"},
                )
            )
            sequence_ids.append(event_id)
            order += 1

        for j in range(n):
            row_id = f"A.row.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    row_id,
                    "A.row.load",
                    [
                        Access("A", (i, j), lane=lane, kind="read")
                        for lane, i in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={
                        **common_metadata,
                        "step": j,
                        "phase": "inner",
                        "pattern": "row",
                    },
                )
            )
            sequence_ids.append(row_id)
            order += 1

            y1_id = f"y1.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    y1_id,
                    "y1.load",
                    [
                        Access("y1", (j,), lane=lane, kind="read")
                        for lane, _ in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={
                        **common_metadata,
                        "step": j,
                        "phase": "inner",
                        "pattern": "row",
                    },
                )
            )
            sequence_ids.append(y1_id)
            order += 1

            transpose_id = f"A.transpose.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    transpose_id,
                    "A.transpose.load",
                    [
                        Access("A", (j, i), lane=lane, kind="read")
                        for lane, i in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={
                        **common_metadata,
                        "step": j,
                        "phase": "inner",
                        "pattern": "transpose",
                    },
                )
            )
            sequence_ids.append(transpose_id)
            order += 1

            y2_id = f"y2.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    y2_id,
                    "y2.load",
                    [
                        Access("y2", (j,), lane=lane, kind="read")
                        for lane, _ in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={
                        **common_metadata,
                        "step": j,
                        "phase": "inner",
                        "pattern": "transpose",
                    },
                )
            )
            sequence_ids.append(y2_id)
            order += 1

        for array in ("x1", "x2"):
            event_id = f"{array}.w{wave}.store"
            events.append(
                MemoryEvent.make(
                    event_id,
                    f"{array}.store",
                    [
                        Access(array, (i,), lane=lane, kind="write")
                        for lane, i in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={**common_metadata, "phase": "epilogue"},
                )
            )
            sequence_ids.append(event_id)
            order += 1

        sequences.append(
            EventSequence.make(
                f"wave{wave}",
                sequence_ids,
                metadata=common_metadata,
            )
        )

    return tuple(events), tuple(sequences)
