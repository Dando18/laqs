"""Small problem module used by the score CLI tests."""

from __future__ import annotations

from relay import (
    Access,
    EventSequence,
    MatrixSpec,
    MemoryEvent,
    SimultaneousRegions,
)


def build_config(problem_size: int = 4) -> dict[str, int]:
    return {"problem_size": problem_size}


def get_matrices(config: dict[str, int]) -> tuple[MatrixSpec, ...]:
    size = config["problem_size"]
    return (MatrixSpec("A", (size, size), 4, ("i", "j")),)


def get_events_and_sequences(
    config: dict[str, int],
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    size = config["problem_size"]
    event = MemoryEvent.make(
        "A.column",
        "A.load",
        (Access("A", (i, 0), lane=i) for i in range(size)),
    )
    return (event,), ()


def get_objectives(
    config: dict[str, int],
) -> tuple[SimultaneousRegions, ...]:
    del config
    return (SimultaneousRegions("column-wave-16B", 16),)
