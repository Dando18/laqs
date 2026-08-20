"""Construct matrices, memory events, and sequences for a square GESUMMV kernel.

Each thread computes one output element

``y[i] = alpha * sum_j A[i, j] * x[j] + beta * sum_j B[i, j] * x[j]``.

The trace models one representative one-dimensional workgroup.  Consecutive
lanes vary ``i`` while every lane visits the same ``j`` at an inner-loop step.
This makes A and B layout targets; x and y retain fixed row-major vector
layouts as context arrays.
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
        MatrixSpec("B", (n, n), 8, ("i", "j"), target=True, role="read"),
        MatrixSpec("x", (n,), 8, ("j",), target=False, role="read"),
        MatrixSpec("y", (n,), 8, ("i",), target=False, role="write"),
    )


def get_events_and_sequences(
    config: Config,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    """Model one workgroup's complete inner loop and final output store."""

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

        for j in range(n):
            for array in ("A", "B"):
                event_id = f"{array}.w{wave}.j{j}"
                events.append(
                    MemoryEvent.make(
                        event_id,
                        f"{array}.load",
                        [
                            Access(array, (i, j), lane=lane, kind="read")
                            for lane, i in lanes
                        ],
                        group=group,
                        order=order,
                        metadata={
                            **common_metadata,
                            "step": j,
                            "phase": "inner",
                        },
                    )
                )
                sequence_ids.append(event_id)
                order += 1

            x_id = f"x.w{wave}.j{j}"
            events.append(
                MemoryEvent.make(
                    x_id,
                    "x.load",
                    [Access("x", (j,), lane=lane, kind="read") for lane, _ in lanes],
                    group=group,
                    order=order,
                    metadata={
                        **common_metadata,
                        "step": j,
                        "phase": "inner",
                    },
                )
            )
            sequence_ids.append(x_id)
            order += 1

        y_id = f"y.w{wave}.store"
        events.append(
            MemoryEvent.make(
                y_id,
                "y.store",
                [
                    Access("y", (i,), lane=lane, kind="write")
                    for lane, i in lanes
                ],
                group=group,
                order=order,
                metadata={**common_metadata, "phase": "epilogue"},
            )
        )
        sequence_ids.append(y_id)
        order += 1
        sequences.append(
            EventSequence.make(
                f"wave{wave}",
                sequence_ids,
                metadata=common_metadata,
            )
        )

    return tuple(events), tuple(sequences)
