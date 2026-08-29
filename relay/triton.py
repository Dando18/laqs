"""Execution-conditioned access events induced by Triton LinearLayouts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Protocol, Sequence

from .layouts import Layout
from .model import (
    Access,
    Coord,
    MatrixSpec,
    MemoryEvent,
    exact_log2,
    is_power_of_two,
)
from .objectives import Hyperedge
from .scoring import transaction_region_ids

if TYPE_CHECKING:
    from .solver import RelayProblem, SolverConfig


class TritonLinearLayoutLike(Protocol):
    """The part of Triton's Python LinearLayout binding that RELAY uses."""

    @property
    def bases(self) -> Sequence[tuple[str, Sequence[Sequence[int]]]]: ...

    @property
    def out_dims(self) -> Sequence[tuple[str, int]]: ...


@dataclass(frozen=True)
class BlockedLayoutParameters:
    """The hardware factors of one compiled Triton blocked encoding."""

    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        return {
            "sizePerThread": self.size_per_thread,
            "threadsPerWarp": self.threads_per_warp,
            "warpsPerCTA": self.warps_per_cta,
            "order": self.order,
        }


def _parse_int_list(body: str, field: str) -> tuple[int, ...]:
    match = re.search(rf"\b{field}\s*=\s*\[([^]]*)\]", body)
    if match is None:
        raise ValueError(f"compiled blocked layout has no {field}")
    values = match.group(1).strip()
    return tuple(
        int(value.strip()) for value in values.split(",") if value.strip()
    )


def extract_blocked_layout(ttgir: str) -> BlockedLayoutParameters:
    """Extract the named ``#blocked`` encoding from compiled TTGIR text."""

    match = re.search(r"#blocked\s*=\s*#ttg\.blocked<\{([^}]+)\}>", ttgir)
    if match is None:
        raise ValueError("compiled Triton module does not contain #blocked")
    body = match.group(1)
    return BlockedLayoutParameters(
        size_per_thread=_parse_int_list(body, "sizePerThread"),
        threads_per_warp=_parse_int_list(body, "threadsPerWarp"),
        warps_per_cta=_parse_int_list(body, "warpsPerCTA"),
        order=_parse_int_list(body, "order"),
    )


@dataclass(frozen=True, order=True)
class HardwareLocation:
    """One named point in Triton's hardware-location space."""

    coordinates: tuple[tuple[str, int], ...]

    @classmethod
    def make(cls, coordinates: Mapping[str, int]) -> "HardwareLocation":
        return cls(
            tuple(
                sorted(
                    (str(name), int(value))
                    for name, value in coordinates.items()
                )
            )
        )

    def value(self, dimension: str) -> int:
        for name, value in self.coordinates:
            if name == dimension:
                return value
        raise KeyError(dimension)

    def as_dict(self) -> dict[str, int]:
        return dict(self.coordinates)


@dataclass(frozen=True)
class TritonLinearLayout:
    """Dependency-free representation of Triton's basis-vector convention.

    ``bases[in_dim][bit]`` is the output of ``L`` when that input dimension is
    ``2**bit`` and every other input is zero. Output coordinates and input
    values are low-bit first, exactly as in Triton's ``LinearLayout``.
    """

    bases: tuple[tuple[str, tuple[Coord, ...]], ...]
    out_dims: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        input_names = [name for name, _ in self.bases]
        output_names = [name for name, _ in self.out_dims]
        if len(input_names) != len(set(input_names)):
            raise ValueError("Triton input dimension names must be unique")
        if len(output_names) != len(set(output_names)):
            raise ValueError("Triton output dimension names must be unique")
        if not self.out_dims:
            raise ValueError("a Triton execution layout needs an output dimension")
        if any(not name for name in (*input_names, *output_names)):
            raise ValueError("Triton dimension names cannot be empty")
        if any(not is_power_of_two(size) for _, size in self.out_dims):
            raise ValueError("Triton output dimension sizes must be powers of two")

        output_sizes = tuple(size for _, size in self.out_dims)
        for input_name, input_bases in self.bases:
            for basis in input_bases:
                if len(basis) != len(output_sizes):
                    raise ValueError(
                        f"{input_name}: basis rank does not match the output rank"
                    )
                if any(
                    value < 0 or value >= size
                    for value, size in zip(basis, output_sizes)
                ):
                    raise ValueError(
                        f"{input_name}: basis {basis} lies outside output "
                        f"shape {output_sizes}"
                    )

    @classmethod
    def from_bases(
        cls,
        bases: Mapping[str, Sequence[Sequence[int]]]
        | Iterable[tuple[str, Sequence[Sequence[int]]]],
        out_dims: Mapping[str, int] | Iterable[tuple[str, int]],
    ) -> "TritonLinearLayout":
        basis_items = bases.items() if isinstance(bases, Mapping) else bases
        output_items = (
            out_dims.items() if isinstance(out_dims, Mapping) else out_dims
        )
        return cls(
            tuple(
                (
                    str(name),
                    tuple(
                        tuple(int(value) for value in basis)
                        for basis in input_bases
                    ),
                )
                for name, input_bases in basis_items
            ),
            tuple((str(name), int(size)) for name, size in output_items),
        )

    @classmethod
    def from_triton(cls, layout: TritonLinearLayoutLike) -> "TritonLinearLayout":
        """Copy a layout from Triton's optional Python binding."""

        return cls.from_bases(layout.bases, layout.out_dims)

    @classmethod
    def from_blocked(
        cls,
        shape: Sequence[int],
        *,
        size_per_thread: Sequence[int],
        threads_per_warp: Sequence[int],
        warps_per_cta: Sequence[int],
        order: Sequence[int],
        output_dim_names: Sequence[str] | None = None,
    ) -> "TritonLinearLayout":
        """Construct a blocked CTA LinearLayout, including register repeats.

        This is the common no-CGA case from Triton's
        ``BlockedEncodingAttr::toLinearLayout``. If the hardware factors cover
        less than ``shape``, Triton extends the register dimension in logical
        output-dimension order. Truncation and non-integral repetition remain
        unsupported.
        """

        output_shape = tuple(int(size) for size in shape)
        factors = tuple(
            tuple(int(size) for size in factor)
            for factor in (size_per_thread, threads_per_warp, warps_per_cta)
        )
        rank = len(output_shape)
        if rank == 0 or any(len(factor) != rank for factor in factors):
            raise ValueError("blocked layout factors must match the tensor rank")
        if tuple(sorted(order)) != tuple(range(rank)):
            raise ValueError("blocked layout order must be a dimension permutation")
        if any(
            not is_power_of_two(size)
            for sizes in (output_shape, *factors)
            for size in sizes
        ):
            raise ValueError("blocked layout shapes must contain powers of two")

        covered = tuple(
            factors[0][dim] * factors[1][dim] * factors[2][dim]
            for dim in range(rank)
        )
        if any(
            covered_extent > output_extent
            or output_extent % covered_extent
            for covered_extent, output_extent in zip(covered, output_shape)
        ):
            raise ValueError(
                "blocked hardware factors must tile the tensor shape: "
                f"{covered} does not tile {output_shape}"
            )

        strides = [1] * rank
        input_bases: list[tuple[str, tuple[Coord, ...]]] = []
        for input_name, factor in zip(
            ("register", "lane", "warp"), factors
        ):
            bases: list[Coord] = []
            for dim in order:
                for bit in range(exact_log2(factor[dim])):
                    basis = [0] * rank
                    basis[dim] = strides[dim] << bit
                    bases.append(tuple(basis))
            input_bases.append((input_name, tuple(bases)))
            for dim in range(rank):
                strides[dim] *= factor[dim]

        register_bases = list(input_bases[0][1])
        for dim in range(rank):
            repetitions = output_shape[dim] // covered[dim]
            for bit in range(exact_log2(repetitions)):
                basis = [0] * rank
                basis[dim] = covered[dim] << bit
                register_bases.append(tuple(basis))
        input_bases[0] = ("register", tuple(register_bases))
        input_bases.append(("block", ()))

        names = (
            tuple(str(name) for name in output_dim_names)
            if output_dim_names is not None
            else tuple(f"dim{dim}" for dim in range(rank))
        )
        if len(names) != rank:
            raise ValueError("output_dim_names must match the tensor rank")
        return cls(tuple(input_bases), tuple(zip(names, output_shape)))

    @property
    def input_dims(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.bases)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(size for _, size in self.out_dims)

    def input_size(self, dimension: str) -> int:
        for name, input_bases in self.bases:
            if name == dimension:
                return 1 << len(input_bases)
        raise KeyError(dimension)

    def apply(self, location: HardwareLocation | Mapping[str, int]) -> Coord:
        """Apply ``L`` to one complete hardware location."""

        values = (
            location.as_dict()
            if isinstance(location, HardwareLocation)
            else dict(location)
        )
        if set(values) != set(self.input_dims):
            missing = sorted(set(self.input_dims) - set(values))
            unexpected = sorted(set(values) - set(self.input_dims))
            raise ValueError(
                "hardware location dimensions do not match the layout: "
                f"missing={missing}, unexpected={unexpected}"
            )

        result = [0] * len(self.out_dims)
        for input_name, input_bases in self.bases:
            input_value = int(values[input_name])
            if input_value < 0 or input_value >= 1 << len(input_bases):
                raise ValueError(
                    f"{input_name}={input_value} is outside "
                    f"[0, {1 << len(input_bases)})"
                )
            for bit, basis in enumerate(input_bases):
                if (input_value >> bit) & 1:
                    for output, value in enumerate(basis):
                        result[output] ^= value
        return tuple(result)

    def locations(
        self, *, fixed: Mapping[str, int] | None = None
    ) -> tuple[HardwareLocation, ...]:
        """Enumerate all input points consistent with ``fixed`` dimensions."""

        fixed_values = {
            str(name): int(value) for name, value in (fixed or {}).items()
        }
        unexpected = sorted(set(fixed_values) - set(self.input_dims))
        if unexpected:
            raise ValueError(f"unknown fixed input dimensions: {unexpected}")

        dimensions: list[tuple[str, range | tuple[int, ...]]] = []
        for name in self.input_dims:
            size = self.input_size(name)
            if name in fixed_values:
                value = fixed_values[name]
                if value < 0 or value >= size:
                    raise ValueError(f"{name}={value} is outside [0, {size})")
                dimensions.append((name, (value,)))
            else:
                dimensions.append((name, range(size)))
        names = tuple(name for name, _ in dimensions)
        return tuple(
            HardwareLocation.make(dict(zip(names, values)))
            for values in product(*(values for _, values in dimensions))
        )


@dataclass(frozen=True)
class InducedMemoryEvent:
    """A RELAY event paired with the hardware locations that induced it."""

    event: MemoryEvent
    locations: tuple[HardwareLocation, ...]

    def __post_init__(self) -> None:
        if len(self.event.accesses) != len(self.locations):
            raise ValueError("induced accesses and hardware locations must align")
        if len(set(self.locations)) != len(self.locations):
            raise ValueError(
                "an induced hardware cohort cannot contain duplicate locations"
            )

    @property
    def hyperedge(self) -> Hyperedge:
        return Hyperedge.make(
            (access.coord for access in self.event.accesses),
            weight=self.event.weight,
            source=self.event.id,
        )


def induce_memory_event(
    execution_layout: TritonLinearLayout,
    matrix: MatrixSpec,
    locations: Iterable[HardwareLocation],
    *,
    id: str,
    site: str,
    kind: str = "read",
    group: str = "",
    order: int = 0,
    weight: float = 1.0,
    lane_dimension: str = "lane",
    coordinate_map: Callable[[Coord], Coord] | None = None,
) -> InducedMemoryEvent:
    """Map one hardware issue cohort through ``L`` into a RELAY event.

    ``coordinate_map`` composes a tensor-local Triton result with the pointer
    expression that embeds it in an array. Without it, the LinearLayout output
    is itself the array coordinate.
    """

    if coordinate_map is None and execution_layout.output_shape != matrix.shape:
        raise ValueError(
            "Triton output shape does not match the matrix: "
            f"{execution_layout.output_shape} != {matrix.shape}"
        )
    cohort = tuple(locations)
    if not cohort:
        raise ValueError("an induced hardware cohort cannot be empty")

    accesses: list[Access] = []
    for location in cohort:
        tensor_coord = execution_layout.apply(location)
        coord = (
            tensor_coord
            if coordinate_map is None
            else tuple(coordinate_map(tensor_coord))
        )
        matrix.validate_coord(coord)
        try:
            lane = location.value(lane_dimension)
        except KeyError:
            lane = None
        accesses.append(
            Access(
                matrix.name,
                coord,
                lane=lane,
                kind=kind,
                width_bytes=matrix.element_bytes,
            )
        )
    return InducedMemoryEvent(
        MemoryEvent.make(
            id,
            site,
            accesses,
            group=group,
            order=order,
            weight=weight,
            metadata={"provenance": "triton-linear-layout"},
        ),
        cohort,
    )


def execution_conditioned_quotient_problem(
    matrices: Iterable[MatrixSpec],
    induced_events: Iterable[InducedMemoryEvent],
    *,
    transaction_bytes: int,
    temporal_edges: Mapping[str, Sequence[Hyperedge]] | None = None,
    temporal_mode: str = "issue",
    temporal_objective_name: str | None = None,
    config: "SolverConfig | None" = None,
    objective_name: str | None = None,
    name: str = "triton_execution_conditioned_quotient",
) -> "RelayProblem":
    """Build Stage 1's issue-only or space-time quotient-locality problem."""

    from .objectives import ExplicitRegions
    from .solver import RelayProblem, SolverConfig

    matrix_items = tuple(matrices)
    matrix_by_name = {matrix.name: matrix for matrix in matrix_items}
    if not matrix_items:
        raise ValueError("an execution-conditioned problem needs a matrix")
    if len(matrix_by_name) != len(matrix_items):
        raise ValueError("execution-conditioned matrix names must be unique")
    if not any(matrix.target for matrix in matrix_items):
        raise ValueError("an execution-conditioned problem needs a target matrix")
    if transaction_bytes <= 0 or not is_power_of_two(transaction_bytes):
        raise ValueError("transaction_bytes must be a positive power of two")
    if temporal_mode not in {"issue", "union", "split"}:
        raise ValueError(
            "temporal_mode must be one of 'issue', 'union', or 'split'"
        )
    for matrix in matrix_items:
        if transaction_bytes % matrix.element_bytes:
            raise ValueError(
                f"{matrix.name}: transaction_bytes must be divisible by "
                "element_bytes"
            )

    event_items = tuple(induced_events)
    if not event_items:
        raise ValueError("an execution-conditioned problem needs an induced event")
    events: list[MemoryEvent] = []
    edges: dict[str, list[Hyperedge]] = {}
    for induced in event_items:
        arrays = {access.array for access in induced.event.accesses}
        if len(arrays) != 1:
            raise ValueError(
                f"induced event {induced.event.id} must access exactly one array"
            )
        array = next(iter(arrays))
        if array not in matrix_by_name:
            raise ValueError(
                f"induced event {induced.event.id}: unknown array {array}"
            )
        events.append(induced.event)
        edges.setdefault(array, []).append(induced.hyperedge)

    temporal: dict[str, tuple[Hyperedge, ...]] = {}
    for array, array_edges in (temporal_edges or {}).items():
        if array not in matrix_by_name:
            raise ValueError(f"temporal edges reference unknown array {array}")
        temporal[array] = tuple(array_edges)
        for edge in temporal[array]:
            for point in edge.points:
                matrix_by_name[array].validate_coord(point)

    issue = {array: tuple(items) for array, items in edges.items()}
    component_name = objective_name or f"triton.issue.{transaction_bytes}B"
    if temporal_mode == "issue":
        objectives = (
            ExplicitRegions(
                component_name,
                transaction_bytes,
                issue,
                provenance="triton-linear-layout",
                description=(
                    "Exact hardware issue cohorts induced by Triton "
                    "LinearLayouts"
                ),
            ),
        )
    elif temporal_mode == "union":
        combined = {array: list(items) for array, items in issue.items()}
        for array, items in temporal.items():
            combined.setdefault(array, []).extend(items)
        objectives = (
            ExplicitRegions(
                component_name,
                transaction_bytes,
                {
                    array: tuple(items)
                    for array, items in combined.items()
                },
                provenance="triton-linear-layout-space-time",
                description=(
                    "Union of exact hardware issue cohorts and per-location "
                    "temporal fibers"
                ),
            ),
        )
    else:
        temporal_name = temporal_objective_name
        if temporal_name is None:
            temporal_name = f"triton.temporal.{transaction_bytes}B"
        objectives = (
            ExplicitRegions(
                component_name,
                transaction_bytes,
                issue,
                provenance="triton-linear-layout",
                description=(
                    "Exact hardware issue cohorts induced by Triton "
                    "LinearLayouts"
                ),
            ),
            ExplicitRegions(
                temporal_name,
                transaction_bytes,
                temporal,
                provenance="triton-linear-layout-time",
                description=(
                    "Non-overlapping per-location temporal fibers over "
                    "ordered issue steps"
                ),
            ),
        )
    return RelayProblem(
        matrices=matrix_items,
        events=tuple(events),
        sequences=(),
        objectives=objectives,
        config=config or SolverConfig(),
        name=name,
    )


@dataclass(frozen=True)
class ObservedAccess:
    """One independently observed hardware-to-address mapping."""

    location: HardwareLocation
    logical_coord: Coord
    byte_offset: int

    def __post_init__(self) -> None:
        if self.byte_offset < 0:
            raise ValueError("observed byte offsets must be nonnegative")


@dataclass(frozen=True)
class ValidationMismatch:
    kind: str
    location: HardwareLocation
    expected: str
    observed: str


@dataclass(frozen=True)
class HypergraphValidation:
    """Detailed stage-0 comparison of induced and observed accesses."""

    event_id: str
    transaction_bytes: int
    expected_transaction_ids: tuple[int, ...]
    observed_transaction_ids: tuple[int, ...]
    expected_groups: tuple[tuple[int, tuple[HardwareLocation, ...]], ...]
    observed_groups: tuple[tuple[int, tuple[HardwareLocation, ...]], ...]
    mismatches: tuple[ValidationMismatch, ...]

    @property
    def expected_quotient_count(self) -> int:
        return len(self.expected_transaction_ids)

    @property
    def observed_transaction_count(self) -> int:
        return len(self.observed_transaction_ids)

    @property
    def valid(self) -> bool:
        return not self.mismatches

    def require_valid(self) -> None:
        if self.valid:
            return
        details = "; ".join(
            f"{mismatch.kind} at {mismatch.location.as_dict()}: "
            f"expected {mismatch.expected}, observed {mismatch.observed}"
            for mismatch in self.mismatches[:4]
        )
        remaining = len(self.mismatches) - 4
        if remaining > 0:
            details += f"; and {remaining} more"
        raise ValueError(f"induced hypergraph validation failed: {details}")


def _transaction_groups(
    offsets: Mapping[HardwareLocation, int], transaction_bytes: int
) -> tuple[tuple[int, tuple[HardwareLocation, ...]], ...]:
    groups: dict[int, list[HardwareLocation]] = {}
    for location, offset in offsets.items():
        groups.setdefault(offset // transaction_bytes, []).append(location)
    return tuple(
        (transaction, tuple(sorted(locations)))
        for transaction, locations in sorted(groups.items())
    )


def validate_induced_hypergraph(
    induced: InducedMemoryEvent,
    matrix: MatrixSpec,
    memory_layout: Layout,
    observed_accesses: Iterable[ObservedAccess],
    *,
    transaction_bytes: int,
) -> HypergraphValidation:
    """Validate ``H_e -> L(H_e) -> A_default`` against an address trace."""

    if transaction_bytes <= 0 or not is_power_of_two(transaction_bytes):
        raise ValueError("transaction_bytes must be a positive power of two")
    if transaction_bytes % matrix.element_bytes:
        raise ValueError("transaction_bytes must be divisible by element_bytes")
    if any(access.array != matrix.name for access in induced.event.accesses):
        raise ValueError(
            "the induced event must access exactly the validated matrix"
        )

    observations = tuple(observed_accesses)
    observed_by_location: dict[HardwareLocation, ObservedAccess] = {}
    for observed in observations:
        if observed.location in observed_by_location:
            raise ValueError(
                "duplicate observed hardware location "
                f"{observed.location.as_dict()}"
            )
        matrix.validate_coord(observed.logical_coord)
        observed_by_location[observed.location] = observed

    expected_by_location = dict(zip(induced.locations, induced.event.accesses))
    expected_offsets = {
        location: memory_layout.offset(matrix, access.coord)
        * matrix.element_bytes
        for location, access in expected_by_location.items()
    }
    observed_offsets = {
        location: observed.byte_offset
        for location, observed in observed_by_location.items()
    }

    mismatches: list[ValidationMismatch] = []
    for location in sorted(
        set(expected_by_location) | set(observed_by_location)
    ):
        expected = expected_by_location.get(location)
        observed = observed_by_location.get(location)
        if expected is None:
            mismatches.append(
                ValidationMismatch(
                    "unexpected-location", location, "absent", "present"
                )
            )
            continue
        if observed is None:
            mismatches.append(
                ValidationMismatch(
                    "missing-location", location, "present", "absent"
                )
            )
            continue
        if expected.coord != observed.logical_coord:
            mismatches.append(
                ValidationMismatch(
                    "logical-coordinate",
                    location,
                    str(expected.coord),
                    str(observed.logical_coord),
                )
            )
        expected_offset = expected_offsets[location]
        if expected_offset != observed.byte_offset:
            mismatches.append(
                ValidationMismatch(
                    "byte-offset",
                    location,
                    str(expected_offset),
                    str(observed.byte_offset),
                )
            )

    expected_ids = tuple(
        sorted(
            transaction_region_ids(
                matrix,
                memory_layout,
                induced.hyperedge,
                transaction_bytes,
            )
        )
    )
    observed_ids = tuple(
        sorted(
            {offset // transaction_bytes for offset in observed_offsets.values()}
        )
    )
    if expected_ids != observed_ids:
        mismatches.append(
            ValidationMismatch(
                "transaction-ids",
                HardwareLocation(()),
                str(expected_ids),
                str(observed_ids),
            )
        )

    return HypergraphValidation(
        event_id=induced.event.id,
        transaction_bytes=transaction_bytes,
        expected_transaction_ids=expected_ids,
        observed_transaction_ids=observed_ids,
        expected_groups=_transaction_groups(
            expected_offsets, transaction_bytes
        ),
        observed_groups=_transaction_groups(
            observed_offsets, transaction_bytes
        ),
        mismatches=tuple(mismatches),
    )
