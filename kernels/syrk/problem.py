"""Construct memory events and objectives for a square SYRK kernel.

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
                metadata={**common_metadata, "phase": "prologue"},
            )
        )
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
                metadata={**common_metadata, "phase": "epilogue"},
            )
        )
        order += 1
        sequences.append(
            EventSequence.make(
                f"wave{wave}", sequence_ids, metadata=common_metadata
            )
        )

    return tuple(events), tuple(sequences)


def get_objectives(config: Config) -> tuple[ObjectiveSpec, ...]:
    """Describe SYRK transaction, reuse, and cache-locality scopes.

    Wave loads and stores correspond directly to traced memory instructions
    and are grounded. Every wider lane, temporal, workgroup, or cache-scale
    scope is a hypothesis about useful physical proximity.
    """

    del config
    reads = EventFilter.make(arrays=("A", "C"), kinds=("read",))
    a_reads = EventFilter.make(arrays=("A",), kinds=("read",))
    row_j_reads = EventFilter.make(
        arrays=("A",), sites=("A.row_j.load",), kinds=("read",)
    )
    inner_a_reads = EventFilter.make(
        arrays=("A",), kinds=("read",), metadata={"phase": "inner"}
    )
    c_writes = EventFilter.make(arrays=("C",), kinds=("write",))

    return (
        SimultaneousRegions(
            "wave_load.64B",
            64,
            event_filter=reads,
            provenance="grounded",
            description="logical addresses issued by one traced wave load",
        ),
        SimultaneousRegions(
            "output_store.64B",
            64,
            event_filter=c_writes,
            provenance="grounded",
            description="logical addresses issued by the traced C wave store",
        ),
        LanePrefixRegions(
            "A.row_j_lane_group",
            levels=((8, 64), (16, 128), (32, 256), (64, 512)),
            event_filter=row_j_reads,
            provenance="hypothesis",
        ),
        PerLaneTemporalRegions(
            "A.paired_row_reuse.128B",
            128,
            windows=(16,),
            stride=16,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "eight consecutive k steps from both A row streams used by "
                "one lane; a temporal-reuse neighborhood hypothesis"
            ),
        ),
        SimultaneousRegions(
            "A.wave_neighborhood.512B",
            512,
            event_filter=a_reads,
            provenance="hypothesis",
            description="one A wave load at a broader locality scale",
        ),
        GroupedRegions(
            "A.workgroup_k_column.256B",
            256,
            group_by=("workgroup", "step"),
            event_filter=inner_a_reads,
            provenance="hypothesis",
            description=(
                "unique A rows used across the workgroup at one k step"
            ),
        ),
        TemporalWindowRegions(
            "A.wave_k_window.4096B",
            4096,
            window=32,
            stride=32,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "sixteen consecutive k steps from both A row streams in a "
                "cache-scale region"
            ),
        ),
        TemporalWindowRegions(
            "A.wave_inner_phase.32768B",
            32768,
            window=None,
            event_filter=a_reads,
            provenance="hypothesis",
            description=(
                "one wave's complete pair of A row streams in a broad "
                "cache-scale region"
            ),
        ),
    )


def get_component_weights(config: Config) -> dict[str, float]:
    """Return MI300A-calibrated ``tau`` weights for score aggregation.

    The N=256, 22-layout calibration retained the grounded wave term, strongly
    weighted the finest row-j lane scope, and kept weak temporal/full-phase
    terms to order tile sizes. Unsupported broader hypotheses remain visible
    at weight zero. These weights are empirical rather than hardware claims.
    """

    del config
    return {
        "wave_load.64B": 1.0,
        "output_store.64B": 0.0,
        "A.row_j_lane_group.lane8.64B": 4.0,
        "A.row_j_lane_group.lane16.128B": 0.25,
        "A.row_j_lane_group.lane32.256B": 0.0,
        "A.row_j_lane_group.lane64.512B": 0.0,
        "A.paired_row_reuse.128B.window16": 0.25,
        "A.wave_neighborhood.512B": 0.0,
        "A.workgroup_k_column.256B": 0.0,
        "A.wave_k_window.4096B": 0.0,
        "A.wave_inner_phase.32768B": 1.0,
    }
