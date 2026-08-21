"""Construct matrices, memory events, and sequences for a square ATAX kernel.

ATAX evaluates two dependent matrix-vector products::

    tmp[i] = sum_j A[i, j] * x[j]
    y[j] = sum_i A[i, j] * tmp[i]

The trace models one representative one-dimensional workgroup from each HIP
kernel.  Consecutive lanes vary ``i`` in the first pass and ``j`` in the
second, exposing both row-wise and column-wise accesses to the same target
matrix A.  The vectors retain fixed contiguous layouts as context arrays.
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
    element_bytes: int = 8


@dataclass(frozen=True)
class HardwareConfig:
    wavefront_size: int = 64


def build_config(**kwargs) -> Config:
    return Config(**kwargs)


def get_matrices(config: Config) -> tuple[MatrixSpec, ...]:
    n = config.problem_size
    element_bytes = config.element_bytes
    return (
        MatrixSpec(
            "A", (n, n), element_bytes, ("i", "j"), target=True, role="read"
        ),
        MatrixSpec("x", (n,), element_bytes, ("j",), target=False, role="read"),
        MatrixSpec(
            "tmp", (n,), element_bytes, ("i",), target=False, role="read_write"
        ),
        MatrixSpec("y", (n,), element_bytes, ("j",), target=False, role="write"),
    )


def _active_lanes(config: Config, wave: int) -> list[tuple[int, int]]:
    """Return ``(lane, output index)`` pairs for one representative wave."""

    wavefront_size = HardwareConfig().wavefront_size
    return [
        (lane, wave * wavefront_size + lane)
        for lane in range(wavefront_size)
        if wave * wavefront_size + lane < config.block_size
        and wave * wavefront_size + lane < config.problem_size
    ]


def get_events_and_sequences(
    config: Config,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    """Model complete inner loops and output stores for both ATAX stages."""

    n = config.problem_size
    wave_count = math.ceil(config.block_size / HardwareConfig().wavefront_size)
    events: list[MemoryEvent] = []
    sequences: list[EventSequence] = []
    order = 0

    for wave in range(wave_count):
        lanes = _active_lanes(config, wave)
        if not lanes:
            continue

        group = f"wg0.stage1.wave{wave}"
        common_metadata = {
            "workgroup": "wg0",
            "wave": wave,
            "phase": "stage1",
        }
        sequence_ids: list[str] = []
        for j in range(n):
            a_id = f"A.stage1.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    a_id,
                    "A.stage1.load",
                    [
                        Access("A", (i, j), lane=lane, kind="read")
                        for lane, i in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={**common_metadata, "step": j},
                )
            )
            sequence_ids.append(a_id)
            order += 1

            x_id = f"x.stage1.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    x_id,
                    "x.load",
                    [Access("x", (j,), lane=lane, kind="read") for lane, _ in lanes],
                    group=group,
                    order=order,
                    metadata={**common_metadata, "step": j},
                )
            )
            sequence_ids.append(x_id)
            order += 1

        store_id = f"tmp.stage1.w{wave}.store"
        events.append(
            MemoryEvent.make(
                store_id,
                "tmp.store",
                [
                    Access("tmp", (i,), lane=lane, kind="write")
                    for lane, i in lanes
                ],
                group=group,
                order=order,
                metadata=common_metadata,
            )
        )
        sequence_ids.append(store_id)
        order += 1
        sequences.append(
            EventSequence.make(
                f"stage1.wave{wave}", sequence_ids, metadata=common_metadata
            )
        )

    for wave in range(wave_count):
        lanes = _active_lanes(config, wave)
        if not lanes:
            continue

        group = f"wg0.stage2.wave{wave}"
        common_metadata = {
            "workgroup": "wg0",
            "wave": wave,
            "phase": "stage2",
        }
        sequence_ids = []
        for i in range(n):
            a_id = f"A.stage2.w{wave}.i{i}"
            events.append(
                MemoryEvent.make(
                    a_id,
                    "A.stage2.load",
                    [
                        Access("A", (i, j), lane=lane, kind="read")
                        for lane, j in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={**common_metadata, "step": i},
                )
            )
            sequence_ids.append(a_id)
            order += 1

            tmp_id = f"tmp.stage2.w{wave}.i{i}"
            events.append(
                MemoryEvent.make(
                    tmp_id,
                    "tmp.load",
                    [
                        Access("tmp", (i,), lane=lane, kind="read")
                        for lane, _ in lanes
                    ],
                    group=group,
                    order=order,
                    metadata={**common_metadata, "step": i},
                )
            )
            sequence_ids.append(tmp_id)
            order += 1

        store_id = f"y.stage2.w{wave}.store"
        events.append(
            MemoryEvent.make(
                store_id,
                "y.store",
                [Access("y", (j,), lane=lane, kind="write") for lane, j in lanes],
                group=group,
                order=order,
                metadata=common_metadata,
            )
        )
        sequence_ids.append(store_id)
        order += 1
        sequences.append(
            EventSequence.make(
                f"stage2.wave{wave}", sequence_ids, metadata=common_metadata
            )
        )

    return tuple(events), tuple(sequences)
