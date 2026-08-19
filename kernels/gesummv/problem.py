"""Construct memory events and objectives for a square GESUMMV kernel.

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
    EventFilter,
    EventSequence,
    GroupedRegions,
    LanePrefixRegions,
    MatrixSpec,
    MemoryEvent,
    PerLaneTemporalRegions,
    SimultaneousRegions,
    TemporalWindowRegions,
)
from relay.objectives import ObjectiveSpec


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


def get_objectives(config: Config) -> tuple[ObjectiveSpec, ...]:
    """Describe GESUMMV transaction, reuse, and cache-locality scopes.

    Only the wave load and output-store scopes come directly from the traced
    memory instructions.  Lane grouping and every temporal or larger-region
    scope are explicitly marked as hypotheses: they encode plausible locality
    neighborhoods rather than asserted hardware behavior.
    """

    del config
    matrix_reads = EventFilter.make(arrays=("A", "B"), kinds=("read",))
    inner_matrix_reads = EventFilter.make(
        arrays=("A", "B"),
        kinds=("read",),
        metadata={"phase": "inner"},
    )
    y_writes = EventFilter.make(arrays=("y",), kinds=("write",))
    lane_levels = ((8, 64), (16, 128), (32, 256), (64, 512))

    return (
        SimultaneousRegions(
            "wave_load.64B",
            64,
            event_filter=matrix_reads,
            provenance="grounded",
            description="logical addresses issued by one traced wave load",
        ),
        SimultaneousRegions(
            "output_store.64B",
            64,
            event_filter=y_writes,
            provenance="grounded",
            description="logical addresses issued by the traced wave store",
        ),
        LanePrefixRegions(
            "wave_lane_group",
            levels=lane_levels,
            event_filter=matrix_reads,
            provenance="hypothesis",
        ),
        PerLaneTemporalRegions(
            "lane_reuse.128B",
            128,
            windows=(16,),
            stride=16,
            event_filter=matrix_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive inner-loop values used by one lane; "
                "a temporal-reuse neighborhood hypothesis"
            ),
        ),
        SimultaneousRegions(
            "wave_neighborhood.512B",
            512,
            event_filter=matrix_reads,
            provenance="hypothesis",
            description=(
                "one wave's 64 FP64 matrix values in a broader locality region"
            ),
        ),
        GroupedRegions(
            "workgroup_step_panel.1024B",
            1024,
            group_by=("workgroup", "step"),
            event_filter=inner_matrix_reads,
            provenance="hypothesis",
            description=(
                "the 128-row A or B panel used by both waves at one loop step"
            ),
        ),
        TemporalWindowRegions(
            "wave_phase.4096B",
            4096,
            window=None,
            event_filter=matrix_reads,
            provenance="hypothesis",
            description=(
                "one wave's complete matrix-read phase in a cache-scale region"
            ),
        ),
    )


def get_component_weights(config: Config) -> dict[str, float]:
    """Return MI300A-calibrated ``tau`` weights for score aggregation.

    The N=256, 22-layout calibration selected the 512-byte wave neighborhood,
    per-lane reuse, and complete-phase cache scopes. Smaller transaction and
    workgroup-panel terms remain diagnostic at weight zero. The active scopes
    are hypotheses about this machine, not universal cache parameters.
    """

    del config
    return {
        "wave_load.64B": 0.0,
        "output_store.64B": 0.0,
        "wave_lane_group.lane8.64B": 0.0,
        "wave_lane_group.lane16.128B": 0.0,
        "wave_lane_group.lane32.256B": 0.0,
        "wave_lane_group.lane64.512B": 0.5,
        "lane_reuse.128B.window16": 1.0,
        "wave_neighborhood.512B": 0.5,
        "workgroup_step_panel.1024B": 0.0,
        "wave_phase.4096B": 4.0,
    }
