"""Construct memory events and objectives for a square ATAX kernel.

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
        MatrixSpec("x", (n,), 8, ("j",), target=False, role="read"),
        MatrixSpec("tmp", (n,), 8, ("i",), target=False, role="read_write"),
        MatrixSpec("y", (n,), 8, ("j",), target=False, role="write"),
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


def get_objectives(config: Config) -> tuple[ObjectiveSpec, ...]:
    """Describe ATAX transactions, reuse, and cache-locality scopes.

    The wave loads and vector stores are grounded in the two traced kernel
    bodies.  Lane grouping and temporal, workgroup, and cache-scale scopes are
    hypotheses about locality that the hardware may exploit.
    """

    del config
    a_reads = EventFilter.make(
        arrays=("A",),
        sites=("A.stage1.load", "A.stage2.load"),
        kinds=("read",),
    )
    stage1_reads = EventFilter.make(
        arrays=("A",),
        sites=("A.stage1.load",),
        kinds=("read",),
    )
    vector_stores = EventFilter.make(arrays=("tmp", "y"), kinds=("write",))
    lane_levels = ((8, 64), (16, 128), (32, 256), (64, 512))

    return (
        SimultaneousRegions(
            "wave_load.64B",
            64,
            event_filter=a_reads,
            provenance="grounded",
            description=(
                "logical A addresses issued by one traced wave load in either pass"
            ),
        ),
        SimultaneousRegions(
            "stage1_wave_load.64B",
            64,
            event_filter=stage1_reads,
            provenance="grounded",
            description=(
                "logical A addresses issued by one traced first-stage wave load"
            ),
        ),
        SimultaneousRegions(
            "output_store.64B",
            64,
            event_filter=vector_stores,
            provenance="grounded",
            description="logical addresses issued by the tmp and y wave stores",
        ),
        LanePrefixRegions(
            "wave_lane_group",
            levels=lane_levels,
            event_filter=a_reads,
            provenance="hypothesis",
        ),
        SimultaneousRegions(
            "stage1_wave_neighborhood.256B",
            256,
            event_filter=stage1_reads,
            provenance="hypothesis",
            description=(
                "first-stage A wave values in a 256-byte neighborhood; the "
                "stage asymmetry is an empirically calibrated cache hypothesis"
            ),
        ),
        PerLaneTemporalRegions(
            "lane_reuse.128B",
            128,
            windows=(16,),
            stride=16,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive reduction values used by one lane in either pass"
            ),
        ),
        SimultaneousRegions(
            "wave_neighborhood.512B",
            512,
            event_filter=a_reads,
            provenance="hypothesis",
            description="one wave's A values in a broader locality region",
        ),
        GroupedRegions(
            "workgroup_step_panel.1024B",
            1024,
            group_by=("workgroup", "phase", "step"),
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "A row or column panel shared by all waves at one reduction step"
            ),
        ),
        TemporalWindowRegions(
            "wave_phase.4096B",
            4096,
            window=None,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "one wave's complete row-wise or column-wise A pass in a cache-scale region"
            ),
        ),
    )


def get_component_weights(config: Config) -> dict[str, float]:
    """Return MI300A-calibrated ``tau`` weights for score aggregation.

    The N=256, 22-layout calibration favored a 128-byte lane neighborhood, a
    complete-pass cache scope, and modest first-stage transaction/panel terms.
    Other scopes remain in reports at weight zero. These weights are empirical
    hypotheses, not universal cache parameters.
    """

    del config
    return {
        "wave_load.64B": 0.0,
        "stage1_wave_load.64B": 0.25,
        "output_store.64B": 0.0,
        "wave_lane_group.lane8.64B": 0.0,
        "wave_lane_group.lane16.128B": 4.0,
        "wave_lane_group.lane32.256B": 0.0,
        "wave_lane_group.lane64.512B": 0.0,
        "stage1_wave_neighborhood.256B": 1.0,
        "lane_reuse.128B.window16": 0.0,
        "wave_neighborhood.512B": 0.0,
        "workgroup_step_panel.1024B": 0.0,
        "wave_phase.4096B": 2.0,
    }
