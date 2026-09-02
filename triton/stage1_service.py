"""Hardware-conditioned service components for Triton Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from relay import (
    LinearLayoutResourceFiber,
    TritonLinearLayout,
    linear_layout_resource_fiber,
)


@dataclass(frozen=True)
class HardwareServiceModel:
    """Instruction and lane-subspace fibers from one execution layout."""

    instruction: LinearLayoutResourceFiber
    lane_cohorts: tuple[LinearLayoutResourceFiber, ...]

    @property
    def fibers(self) -> tuple[LinearLayoutResourceFiber, ...]:
        return (self.instruction, *self.lane_cohorts)

    def record(self) -> dict[str, object]:
        def fiber_record(fiber: LinearLayoutResourceFiber) -> dict[str, object]:
            return {
                "name": fiber.name,
                "varying_dimensions": list(fiber.varying_dimensions),
                "varying_bits": {
                    dimension: list(bits)
                    for dimension, bits in fiber.varying_bits
                },
                "hardware_fiber_count": fiber.hardware_fiber_count,
                "omitted_singleton_count": fiber.omitted_singleton_count,
                "merges_event_instances": fiber.merges_event_instances,
            }

        return {
            "instruction": fiber_record(self.instruction),
            "lane_cohorts": [
                fiber_record(fiber) for fiber in self.lane_cohorts
            ],
        }


def lane_bit_subsets(
    execution: TritonLinearLayout, *, exhaustive: bool = False
) -> tuple[tuple[int, ...], ...]:
    """Return proper nonempty lane-bit subspaces in stable size order."""

    width = execution.input_size("lane").bit_length() - 1
    if width <= 1:
        return ()
    if exhaustive:
        return tuple(
            bits
            for count in range(1, width)
            for bits in combinations(range(width), count)
        )
    return tuple(tuple(range(count)) for count in range(1, width))


def hardware_service_model(
    execution: TritonLinearLayout,
    induced_events: Iterable,
    *,
    name: str,
    lane_subsets: Sequence[Sequence[int]] | None = None,
) -> HardwareServiceModel:
    """Build per-operation wave and algebraic lane-cohort fibers."""

    events = tuple(induced_events)
    instruction = linear_layout_resource_fiber(
        execution,
        events,
        varying_dimensions=("lane",),
        name=f"{name}.instruction",
        merge_events=False,
    )
    subsets = (
        lane_bit_subsets(execution)
        if lane_subsets is None
        else tuple(tuple(int(bit) for bit in bits) for bits in lane_subsets)
    )
    cohorts = tuple(
        linear_layout_resource_fiber(
            execution,
            events,
            varying_bits={"lane": bits},
            name=f"{name}.lane_bits_" + "_".join(str(bit) for bit in bits),
            merge_events=False,
        )
        for bits in subsets
    )
    return HardwareServiceModel(instruction, cohorts)
