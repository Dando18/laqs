"""Construct memory events and objectives for square MVT.

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
        row_event_ids: list[str] = []
        transpose_event_ids: list[str] = []

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
            row_event_ids.append(row_id)
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
            transpose_event_ids.append(transpose_id)
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
            order += 1

        sequences.extend(
            (
                EventSequence.make(
                    f"wave{wave}.row",
                    row_event_ids,
                    metadata={**common_metadata, "pattern": "row"},
                ),
                EventSequence.make(
                    f"wave{wave}.transpose",
                    transpose_event_ids,
                    metadata={**common_metadata, "pattern": "transpose"},
                ),
            )
        )

    return tuple(events), tuple(sequences)


def get_objectives(config: Config) -> tuple[ObjectiveSpec, ...]:
    """Describe MVT transactions, opposing streams, reuse, and locality.

    The wave load and output-store scopes come directly from the traced memory
    instructions and are marked ``grounded``. Lane grouping, temporal streams,
    cross-direction workgroup panels, and cache-scale neighborhoods are
    explicitly hypotheses about beneficial physical proximity.
    """

    del config
    a_reads = EventFilter.make(arrays=("A",), kinds=("read",))
    row_reads = EventFilter.make(
        arrays=("A",), kinds=("read",), metadata={"pattern": "row"}
    )
    transpose_reads = EventFilter.make(
        arrays=("A",), kinds=("read",), metadata={"pattern": "transpose"}
    )
    inner_a_reads = EventFilter.make(
        arrays=("A",), kinds=("read",), metadata={"phase": "inner"}
    )
    output_writes = EventFilter.make(
        arrays=("x1", "x2"), kinds=("write",)
    )

    return (
        SimultaneousRegions(
            "wave_load.64B",
            64,
            event_filter=a_reads,
            provenance="grounded",
            description=(
                "logical A addresses issued by one traced row or transpose "
                "wave load"
            ),
        ),
        SimultaneousRegions(
            "output_store.64B",
            64,
            event_filter=output_writes,
            provenance="grounded",
            description="logical addresses issued by a traced x1 or x2 wave store",
        ),
        LanePrefixRegions(
            "A.wave_lane_group",
            levels=((8, 64), (16, 128), (32, 256), (64, 512)),
            event_filter=a_reads,
            provenance="hypothesis",
        ),
        PerLaneTemporalRegions(
            "row_lane_stream.128B",
            128,
            windows=(16,),
            stride=16,
            event_filter=row_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive A[i,j] values used by one lane; a "
                "row-stream reuse hypothesis"
            ),
        ),
        PerLaneTemporalRegions(
            "row_lane_stream.512B",
            512,
            windows=(16,),
            stride=16,
            event_filter=row_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive A[i,j] values used by one lane in a "
                "512-byte neighborhood; an empirically calibrated row-stream "
                "reuse hypothesis"
            ),
        ),
        PerLaneTemporalRegions(
            "transpose_lane_stream.128B",
            128,
            windows=(16,),
            stride=16,
            event_filter=transpose_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive A[j,i] values used by one lane; a "
                "column-stream reuse hypothesis"
            ),
        ),
        SimultaneousRegions(
            "wave_neighborhood.512B",
            512,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "one row or transpose wave load in a broader locality region"
            ),
        ),
        SimultaneousRegions(
            "transpose_wave_neighborhood.512B",
            512,
            event_filter=transpose_reads,
            provenance="hypothesis",
            description=(
                "one transpose-stream wave load in a 512-byte cache "
                "neighborhood; an empirically calibrated hypothesis"
            ),
        ),
        SimultaneousRegions(
            "transpose_wave_neighborhood.1024B",
            1024,
            event_filter=transpose_reads,
            provenance="hypothesis",
            description=(
                "one transpose-stream wave load in a 1024-byte cache "
                "neighborhood; an empirically calibrated hypothesis"
            ),
        ),
        SimultaneousRegions(
            "transpose_wave_neighborhood.4096B",
            4096,
            event_filter=transpose_reads,
            provenance="hypothesis",
            description=(
                "one transpose-stream wave load in a 4096-byte cache "
                "neighborhood; an empirically calibrated hypothesis"
            ),
        ),
        SimultaneousRegions(
            "transpose_wave_neighborhood.8192B",
            8192,
            event_filter=transpose_reads,
            provenance="hypothesis",
            description=(
                "one transpose-stream wave load in an 8192-byte cache "
                "neighborhood; an empirically calibrated hypothesis"
            ),
        ),
        GroupedRegions(
            "workgroup_step_cross.2048B",
            2048,
            group_by=("workgroup", "step"),
            event_filter=inner_a_reads,
            provenance="hypothesis",
            description=(
                "the row and column arms touched by a workgroup at one inner "
                "step; a cross-direction cache-reuse hypothesis"
            ),
        ),
        TemporalWindowRegions(
            "wave_pattern_window.4096B",
            4096,
            window=16,
            stride=16,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive loads from one directional matrix stream"
            ),
        ),
        TemporalWindowRegions(
            "wave_pattern_phase.32768B",
            32768,
            window=None,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "one wave's complete row or transpose stream in a broad "
                "cache-scale region"
            ),
        ),
    )


def get_component_weights(config: Config) -> dict[str, float]:
    """Return MI300A-calibrated ``tau`` weights for score aggregation.

    The original N=256 calibration selected two identical full-wave 512-byte
    edge families. The expanded three-size analysis disables both copies:
    their symmetric locality preference incorrectly dominated the N=1024
    winner. Separate row-lane and transpose-wave 512-byte scopes remain
    visible as hypotheses, but stay at weight zero because activating them
    degraded total-score ranks. The remaining directional asymmetry is an
    in-sample, machine-specific hypothesis.
    """

    del config
    return {
        "wave_load.64B": 0.0,
        "output_store.64B": 0.0,
        "A.wave_lane_group.lane8.64B": 0.0,
        "A.wave_lane_group.lane16.128B": 0.0,
        "A.wave_lane_group.lane32.256B": 0.0,
        "A.wave_lane_group.lane64.512B": 0.0,
        "row_lane_stream.128B.window16": 0.0,
        "row_lane_stream.512B.window16": 0.0,
        "transpose_lane_stream.128B.window16": 0.0,
        "wave_neighborhood.512B": 0.0,
        "transpose_wave_neighborhood.512B": 0.0,
        "transpose_wave_neighborhood.1024B": 0.0625,
        "transpose_wave_neighborhood.4096B": 0.0625,
        "transpose_wave_neighborhood.8192B": 0.0625,
        "workgroup_step_cross.2048B": 0.0,
        "wave_pattern_window.4096B": 0.0,
        "wave_pattern_phase.32768B": 0.0,
    }
