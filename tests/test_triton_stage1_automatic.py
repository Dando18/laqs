"""Host-only integration oracles for the automatic Stage-1 frontend.

These tests compile the unmodified Stage-1 JIT functions for the pinned gfx942
target.  Fake tensors supply only launch allocation metadata, so no GPU or
PyTorch runtime is involved. Independent host traces are converted through the
same public universal construction and compared at the edge-family, canonical
layout quotient, hardware-profile response, and eligible resource-cohort
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from typing import Any, Iterable, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
PINNED_TRITON_PYTHON = REPOSITORY / "triton" / "triton-lang" / "python"
STAGE1_SOURCE = REPOSITORY / "triton"
for source in (str(PINNED_TRITON_PYTHON), str(STAGE1_SOURCE)):
    while source in sys.path:
        sys.path.remove(source)
sys.path[:0] = (str(PINNED_TRITON_PYTHON), str(STAGE1_SOURCE))

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from relay import (
    Access,
    CanonicalLayout,
    EventSequence,
    MatrixSpec,
    MemoryEvent,
    MI300A_V1,
    build_edge_families,
    build_resource_cohorts,
    layout_matrix_rows,
    materialize_edge_families,
    row_major_layout,
    score_layouts,
    weighted_component_region_count,
)
from relay.triton_frontend import (
    _manifest_compilation,
    analyze_compiled_manifest,
)
from stage1_kernels import (
    bias_relu_kernel,
    embedding_bag_kernel,
    gemm_prepacked_b_kernel,
    gemv_kernel,
    gesummv_kernel,
    mvt_kernel,
    softmax_bias_kernel,
    stencil5_kernel,
)


PLUGIN = (
    REPOSITORY
    / "triton"
    / "triton-lang"
    / "python"
    / "triton"
    / "plugins"
    / "libLAQSTritonAccessManifest.so"
)
TARGET = GPUTarget("hip", "gfx942", 64)


@dataclass(frozen=True)
class FakeDType:
    name: str
    itemsize: int

    def __str__(self) -> str:
        return self.name


class FakeStorage:
    def __init__(self, pointer: int):
        self.pointer = pointer

    def data_ptr(self) -> int:
        return self.pointer


class FakeTensor:
    def __init__(
        self,
        shape: Sequence[int],
        pointer: int,
        *,
        dtype: FakeDType,
        values: Sequence[int] | None = None,
    ) -> None:
        self.shape = tuple(shape)
        self.pointer = pointer
        self.dtype = dtype
        self._values = None if values is None else tuple(values)
        stride = 1
        strides = []
        for extent in reversed(self.shape):
            strides.append(stride)
            stride *= extent
        self._strides = tuple(reversed(strides))

    def data_ptr(self) -> int:
        return self.pointer

    def element_size(self) -> int:
        return self.dtype.itemsize

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def untyped_storage(self) -> FakeStorage:
        return FakeStorage(self.pointer)

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def reshape(self, *_shape: int) -> "FakeTensor":
        return self

    def tolist(self) -> list[int]:
        if self._values is None:
            raise TypeError("this fake tensor has no concrete host values")
        return list(self._values)


FP32 = FakeDType("float32", 4)
I32 = FakeDType("int32", 4)


Operation = tuple[str, str, tuple[tuple[int, ...], ...]]


@dataclass(frozen=True)
class ExpectedTrace:
    workgroup: tuple[int, int, int]
    wave: int
    multiplicity: int
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class Stage1Case:
    kernel: Any
    signature: dict[str, str]
    constants: dict[str, Any]
    grid: tuple[int, ...]
    tensors: dict[str, FakeTensor]
    traces: tuple[ExpectedTrace, ...]
    event_count: int
    access_count: int
    loop_site_count: int = 0


def _rows(name: str, shape: tuple[int, ...], modes: tuple[str, ...]) -> tuple[int, ...]:
    matrix = MatrixSpec(name, shape, 4, modes)
    return layout_matrix_rows(matrix, row_major_layout(matrix))


def _coords(values: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    return tuple((value,) for value in values)


def _matrix_coords(
    rows: Iterable[int], columns: Iterable[int]
) -> tuple[tuple[int, int], ...]:
    return tuple((row, column) for row in rows for column in columns)


def _operation(array: str, kind: str, coordinates: Iterable[Sequence[int]]) -> Operation:
    return array, kind, tuple(sorted(tuple(coord) for coord in coordinates))


def _pointer_signature(kernel: Any, pointer_types: dict[str, str], constants: dict[str, Any]) -> dict[str, str]:
    return {
        name: pointer_types.get(name, "constexpr" if name in constants else "i32")
        for name in kernel.arg_names
    }


def _tensors(specifications: Sequence[tuple[str, tuple[int, ...], FakeDType, Sequence[int] | None]]) -> dict[str, FakeTensor]:
    return {
        name: FakeTensor(shape, 0x10000 + index * 0x10000, dtype=dtype, values=values)
        for index, (name, shape, dtype, values) in enumerate(specifications)
    }


def _single_trace(operations: Sequence[Operation], *, multiplicity: int = 1, wave: int = 0) -> tuple[ExpectedTrace, ...]:
    return (ExpectedTrace((0, 0, 0), wave, multiplicity, tuple(operations)),)


def bias_relu_case() -> Stage1Case:
    constants = {"B_ROWS": _rows("bias", (8,), ("feature",)), "N": 8, "ELEMENTS": 16, "BLOCK": 8}
    operations = (
        _operation("source", "load", _coords(range(8))),
        _operation("bias", "load", _coords(range(8))),
        _operation("output", "store", _coords(range(8))),
    )
    tensors = _tensors((("source", (16,), FP32, None), ("bias", (8,), FP32, None), ("output", (16,), FP32, None)))
    return Stage1Case(
        bias_relu_kernel,
        _pointer_signature(bias_relu_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (2,),
        tensors,
        _single_trace(operations, multiplicity=2),
        3,
        24,
    )


def softmax_bias_case() -> Stage1Case:
    constants = {"B_ROWS": _rows("bias", (2, 16), ("row", "feature")), "ROW_BITS": 1, "N": 16}
    row = _matrix_coords((0,), range(16))
    operations = tuple(_operation(name, kind, row) for name, kind in (("source", "load"), ("bias", "load"), ("output", "store")))
    tensors = _tensors(tuple((name, (2, 16), FP32, None) for name in ("source", "bias", "output")))
    return Stage1Case(
        softmax_bias_kernel,
        _pointer_signature(softmax_bias_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (2,),
        tensors,
        _single_trace(operations, multiplicity=2),
        3,
        48,
    )


def gemv_case() -> Stage1Case:
    constants = {"W_ROWS": _rows("weight", (8, 8), ("row", "column")), "ROW_BITS": 3, "M": 8, "K": 8, "BLOCK": 8}
    operations = []
    for column in range(8):
        operations.append(_operation("weight", "load", _matrix_coords(range(8), (column,))))
        operations.append(_operation("vector", "load", ((column,),)))
    operations.append(_operation("output", "store", _coords(range(8))))
    tensors = _tensors((("weight", (8, 8), FP32, None), ("vector", (8,), FP32, None), ("output", (8,), FP32, None)))
    return Stage1Case(
        gemv_kernel,
        _pointer_signature(gemv_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (1,),
        tensors,
        _single_trace(operations),
        17,
        80,
        2,
    )


def mvt_case() -> Stage1Case:
    constants = {"A_ROWS": _rows("matrix", (8, 8), ("row", "column")), "ROW_BITS": 3, "N": 8, "BLOCK": 8}
    operations = []
    for column in range(8):
        operations.extend(
            (
                _operation("matrix", "load", _matrix_coords(range(8), (column,))),
                _operation("x", "load", ((column,),)),
                _operation("matrix", "load", _matrix_coords((column,), range(8))),
                _operation("y", "load", ((column,),)),
            )
        )
    operations.append(_operation("output", "store", _coords(range(8))))
    tensors = _tensors((("matrix", (8, 8), FP32, None), ("x", (8,), FP32, None), ("y", (8,), FP32, None), ("output", (8,), FP32, None)))
    return Stage1Case(
        mvt_kernel,
        _pointer_signature(mvt_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (1,),
        tensors,
        _single_trace(operations),
        33,
        152,
        4,
    )


def gesummv_case() -> Stage1Case:
    rows = _rows("a", (8, 8), ("row", "column"))
    constants = {"A_ROWS": rows, "B_ROWS": rows, "MODE_BITS": 3, "N": 8, "BLOCK": 8, "ALPHA": 1.0, "BETA": 1.0}
    operations = []
    for column in range(8):
        operations.extend(
            (
                _operation("x", "load", ((column,),)),
                _operation("a", "load", _matrix_coords(range(8), (column,))),
                _operation("b", "load", _matrix_coords(range(8), (column,))),
            )
        )
    operations.append(_operation("output", "store", _coords(range(8))))
    tensors = _tensors((("a", (8, 8), FP32, None), ("b", (8, 8), FP32, None), ("x", (8,), FP32, None), ("output", (8,), FP32, None)))
    return Stage1Case(
        gesummv_kernel,
        _pointer_signature(gesummv_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (1,),
        tensors,
        _single_trace(operations),
        25,
        144,
        3,
    )


def stencil5_case() -> Stage1Case:
    constants = {"A_ROWS": _rows("source", (4, 8), ("row", "column")), "ROW_BITS": 2, "M": 4, "N": 8, "BLOCK": 8}
    traces = []
    for row in range(4):
        columns = tuple(range(8))
        operations = (
            _operation("source", "load", ((row, column) for column in columns)),
            _operation("source", "load", ((row, max(column - 1, 0)) for column in columns)),
            _operation("source", "load", ((row, min(column + 1, 7)) for column in columns)),
            _operation("source", "load", ((max(row - 1, 0), column) for column in columns)),
            _operation("source", "load", ((min(row + 1, 3), column) for column in columns)),
            _operation("output", "store", ((row, column) for column in columns)),
        )
        traces.append(ExpectedTrace((row, 0, 0), 0, 1, operations))
    tensors = _tensors((("source", (4, 8), FP32, None), ("output", (4, 8), FP32, None)))
    return Stage1Case(
        stencil5_kernel,
        _pointer_signature(stencil5_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (4,),
        tensors,
        tuple(traces),
        24,
        192,
    )


def embedding_bag_case() -> Stage1Case:
    constants = {"W_ROWS": _rows("weight", (4, 8), ("row", "dimension")), "ROW_BITS": 2, "D": 8, "BAG_SIZE": 2}
    operations = (
        _operation("indices", "load", ((0,),)),
        _operation("weight", "load", _matrix_coords((0,), range(8))),
        _operation("indices", "load", ((1,),)),
        _operation("weight", "load", _matrix_coords((1,), range(8))),
        _operation("output", "store", _matrix_coords((0,), range(8))),
    )
    tensors = _tensors((("weight", (4, 8), FP32, None), ("indices", (4,), I32, (0, 1, 2, 3)), ("output", (2, 8), FP32, None)))
    return Stage1Case(
        embedding_bag_kernel,
        _pointer_signature(embedding_bag_kernel, {"weight": "*fp32", "indices": "*i32", "output": "*fp32"}, constants),
        constants,
        (2,),
        tensors,
        _single_trace(operations, multiplicity=2),
        5,
        26,
    )


def gemm_prepacked_b_case() -> Stage1Case:
    constants = {"B_ROWS": _rows("b", (16, 16), ("k", "column")), "MODE_BITS": 4, "N": 16, "BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}
    tensors = _tensors((("a", (16, 16), FP32, None), ("b", (16, 16), FP32, None), ("c", (16, 16), FP32, None)))
    # The post-coalesce blocked encoding assigns one consecutive four-row
    # stripe to each wave. The trace keeps wave zero and multiplicity four.
    stripe = _matrix_coords(range(4), range(16))
    operations = (
        _operation("a", "load", stripe),
        _operation("b", "load", stripe),
        _operation("c", "store", stripe),
    )
    return Stage1Case(
        gemm_prepacked_b_kernel,
        _pointer_signature(gemm_prepacked_b_kernel, {name: "*fp32" for name in tensors}, constants),
        constants,
        (1, 1),
        tensors,
        _single_trace(operations, multiplicity=4),
        3,
        192,
    )


CASES = {
    "bias_relu": bias_relu_case,
    "softmax_bias": softmax_bias_case,
    "gemv": gemv_case,
    "mvt": mvt_case,
    "gesummv": gesummv_case,
    "stencil5": stencil5_case,
    "gemm_prepacked_b": gemm_prepacked_b_case,
    "embedding_bag": embedding_bag_case,
}


def _bound_arguments(case: Stage1Case) -> dict[int | str, Any]:
    names = tuple(case.kernel.arg_names)
    bound: dict[int | str, Any] = {
        "__names__": {index: name for index, name in enumerate(names)}
    }
    for index, name in enumerate(names):
        if name in case.tensors:
            bound[index] = case.tensors[name]
            bound[name] = case.tensors[name]
        elif name in case.constants:
            bound[index] = case.constants[name]
            bound[name] = case.constants[name]
    return bound


def _observed_operations(events: Sequence[Any]) -> tuple[Operation, ...]:
    operations: list[tuple[str, str, list[tuple[int, ...]]]] = []
    previous_site = None
    for event in events:
        arrays = {access.array for access in event.accesses}
        kinds = {access.kind for access in event.accesses}
        if len(arrays) != 1 or len(kinds) != 1:
            raise AssertionError(f"memory event {event.id} mixes allocations or operations")
        array = next(iter(arrays))
        kind = next(iter(kinds))
        if event.site != previous_site:
            operations.append((array, kind, []))
            previous_site = event.site
        elif operations[-1][:2] != (array, kind):
            raise AssertionError(f"site {event.site} changes allocation or operation")
        operations[-1][2].extend(access.coord for access in event.accesses)
    return tuple(
        (array, kind, tuple(sorted(coordinates)))
        for array, kind, coordinates in operations
    )


def _oracle_matrices(case: Stage1Case) -> tuple[MatrixSpec, ...]:
    operations = {
        name: {
            kind
            for trace in case.traces
            for operation_name, kind, _coordinates in trace.operations
            if operation_name == name
        }
        for name in case.tensors
    }
    result = []
    for name, tensor in case.tensors.items():
        kinds = operations[name]
        role = (
            "read_write"
            if kinds == {"load", "store"}
            else "read"
            if kinds == {"load"}
            else "write"
        )
        result.append(
            MatrixSpec(
                name,
                tensor.shape,
                tensor.element_size(),
                tuple(f"dim{dimension}" for dimension in range(len(tensor.shape))),
                target=role == "read",
                role=role,
            )
        )
    return tuple(result)


def _oracle_site_and_phase(
    case: Stage1Case, operation_index: int
) -> tuple[str, str]:
    if case.loop_site_count and operation_index < len(case.traces[0].operations) - 1:
        site = operation_index % case.loop_site_count
        iteration = operation_index // case.loop_site_count
        return f"memory.{site}", f"root/n0/iter{iteration}.sync0"
    site = case.loop_site_count if case.loop_site_count else operation_index
    return f"memory.{site}", "root.sync0"


def _oracle_events(
    case: Stage1Case,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    events = []
    sequences = []
    for trace_index, trace in enumerate(case.traces):
        event_ids = []
        workgroup = ".".join(str(value) for value in trace.workgroup)
        for operation_index, (array, kind, coordinates) in enumerate(
            trace.operations
        ):
            site, phase = _oracle_site_and_phase(case, operation_index)
            event_id = f"oracle{trace_index}.event{operation_index}.{site}"
            events.append(
                MemoryEvent.make(
                    event_id,
                    site,
                    (
                        Access(
                            array,
                            coordinate,
                            lane=lane,
                            kind=kind,
                            width_bytes=case.tensors[array].element_size(),
                        )
                        for lane, coordinate in enumerate(coordinates)
                    ),
                    order=operation_index,
                    metadata={
                        "block": 0,
                        "phase": phase,
                        "step": operation_index,
                        "wave": trace.wave,
                        "workgroup": workgroup,
                    },
                )
            )
            event_ids.append(event_id)
        sequences.append(
            EventSequence.make(
                f"triton.trace.{trace_index}",
                event_ids,
                weight=trace.multiplicity,
            )
        )
    return tuple(events), tuple(sequences)


def _family_signature(families: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        (
            family.scope,
            family.normalization_bytes,
            tuple(
                (
                    array,
                    tuple(
                        sorted(
                            (edge.points, edge.weight)
                            for edge in edges
                        )
                    ),
                )
                for array, edges in sorted(family.edges_by_array.items())
            ),
        )
        for family in families
    )


def _canonical_candidates(matrix: MatrixSpec) -> tuple[CanonicalLayout, ...]:
    candidates = []

    def enumerate_words(prefix: tuple[int, ...], remaining: tuple[int, ...]) -> None:
        if not any(remaining):
            word = "".join(matrix.mode_names[mode] for mode in prefix)
            candidates.append(
                CanonicalLayout(
                    f"canonical_{word}",
                    matrix.name,
                    matrix.mode_bits,
                    prefix,
                    tuple(reversed(range(matrix.rank))),
                )
            )
            return
        for mode, count in enumerate(remaining):
            if count == 0:
                continue
            next_remaining = list(remaining)
            next_remaining[mode] -= 1
            enumerate_words(prefix + (mode,), tuple(next_remaining))

    enumerate_words((), matrix.mode_bits)
    for candidate in candidates:
        candidate.validate(matrix)
    return tuple(candidates)


def _resource_signature(cohorts: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        (
            cohort.family,
            cohort.weight,
            tuple(
                (
                    access.array,
                    access.coord,
                    access.lane,
                    access.kind,
                    access.width_bytes,
                )
                for access in cohort.accesses
            ),
        )
        for cohort in cohorts
    )


def _score_signature(score: Any) -> tuple[Any, ...]:
    return (
        tuple(
            sorted(
                (
                    component.name,
                    component.region_bytes,
                    component.weight,
                    component.raw_region_count,
                    component.packing_lower_bound,
                    component.normalized_excess,
                    component.normalization_bytes,
                    component.excess_footprint,
                    component.peak_tolerance,
                    tuple(
                        sorted(
                            (
                                array.array,
                                array.raw_region_count,
                                array.packing_lower_bound,
                                array.normalized_excess,
                            )
                            for array in component.arrays
                        )
                    ),
                )
                for component in score.components
            )
        ),
        tuple(sorted(score.codegen.arrays, key=lambda item: item.array)),
        score.weighted_region_count,
        score.peak_normalized_excess,
        score.weighted_normalized_excess,
        score.hardware_peak,
        score.hardware_area,
    )


class Stage1AutomaticFrontendTests(unittest.TestCase):
    maxDiff = None
    _analyses: dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if not PLUGIN.is_file():
            raise unittest.SkipTest(
                "LAQS Triton access-manifest plugin has not been built"
            )
        source = Path(triton.__file__).resolve()
        expected = (PINNED_TRITON_PYTHON / "triton").resolve()
        if not source.is_relative_to(expected):
            raise RuntimeError(f"tests imported {source}, not pinned Triton {expected}")

    def _analysis(self, name: str) -> tuple[Stage1Case, Any]:
        case = CASES[name]()
        if name not in self._analyses:
            with _manifest_compilation(PLUGIN):
                compiled = triton.compile(
                    ASTSource(case.kernel, case.signature, case.constants),
                    target=TARGET,
                )
            self._analyses[name] = analyze_compiled_manifest(
                compiled,
                compiled.metadata.laqs_access_manifest,
                case.grid,
                _bound_arguments(case),
            )
        return case, self._analyses[name]

    def _check(self, name: str) -> None:
        case, analysis = self._analysis(name)
        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(len(analysis.events), case.event_count)
        self.assertEqual(
            sum(len(event.accesses) for event in analysis.events),
            case.access_count,
        )
        events = {event.id: event for event in analysis.events}
        observed = []
        for sequence in analysis.sequences:
            metadata = dict(sequence.metadata)
            workgroup = tuple(
                int(value)
                for value in metadata["representative_workgroup"].split(".")
            )
            observed.append(
                ExpectedTrace(
                    workgroup,
                    int(metadata["representative_wave"]),
                    int(sequence.multiplicity),
                    _observed_operations(
                        tuple(events[event_id] for event_id in sequence.event_ids)
                    ),
                )
            )
        self.assertEqual(tuple(observed), case.traces)

        oracle_matrices = {
            matrix.name: matrix for matrix in _oracle_matrices(case)
        }
        automatic_matrices = {
            matrix.name: matrix for matrix in analysis.matrices
        }
        self.assertEqual(automatic_matrices, oracle_matrices)
        oracle_events, oracle_sequences = _oracle_events(case)
        oracle_families = build_edge_families(
            oracle_matrices,
            {event.id: event for event in oracle_events},
            oracle_sequences,
        )
        self.assertEqual(
            _family_signature(analysis.edge_families),
            _family_signature(oracle_families),
        )

        automatic_components = materialize_edge_families(
            analysis.edge_families,
            automatic_matrices,
            MI300A_V1.byte_scales,
        )
        oracle_components = materialize_edge_families(
            oracle_families,
            oracle_matrices,
            MI300A_V1.byte_scales,
        )
        self.assertEqual(
            tuple(component.name for component in automatic_components),
            tuple(component.name for component in oracle_components),
        )
        default_layouts = {
            matrix.name: row_major_layout(matrix)
            for matrix in automatic_matrices.values()
            if matrix.target
        }
        for matrix in automatic_matrices.values():
            if not matrix.target:
                continue
            for candidate in _canonical_candidates(matrix):
                for automatic, expected in zip(
                    automatic_components, oracle_components
                ):
                    with self.subTest(
                        case=name,
                        array=matrix.name,
                        layout=candidate.name,
                        component=automatic.name,
                    ):
                        self.assertEqual(
                            weighted_component_region_count(
                                matrix, candidate, automatic
                            ),
                            weighted_component_region_count(
                                matrix, candidate, expected
                            ),
                        )
                layouts = dict(default_layouts)
                layouts[matrix.name] = candidate
                automatic_score = score_layouts(
                    automatic_matrices,
                    automatic_components,
                    layouts,
                    hardware_profile=MI300A_V1,
                )
                oracle_score = score_layouts(
                    oracle_matrices,
                    oracle_components,
                    layouts,
                    hardware_profile=MI300A_V1,
                )
                self.assertEqual(
                    _score_signature(automatic_score),
                    _score_signature(oracle_score),
                )

        if all(trace.multiplicity == 1 for trace in case.traces):
            families = tuple(
                resource_map.cohort_family
                for resource_map in MI300A_V1.resource_maps
            )
            automatic_cohorts = build_resource_cohorts(
                automatic_matrices,
                {event.id: event for event in analysis.events},
                analysis.sequences,
                families,
            )
            oracle_cohorts = build_resource_cohorts(
                oracle_matrices,
                {event.id: event for event in oracle_events},
                oracle_sequences,
                families,
            )
            self.assertEqual(
                {
                    family: _resource_signature(cohorts)
                    for family, cohorts in automatic_cohorts.items()
                },
                {
                    family: _resource_signature(cohorts)
                    for family, cohorts in oracle_cohorts.items()
                },
            )
            all_row_major = {
                matrix.name: row_major_layout(matrix)
                for matrix in automatic_matrices.values()
            }
            automatic_resource_score = score_layouts(
                automatic_matrices,
                automatic_components,
                all_row_major,
                hardware_profile=MI300A_V1,
                resource_cohorts=automatic_cohorts,
            )
            oracle_resource_score = score_layouts(
                oracle_matrices,
                oracle_components,
                all_row_major,
                hardware_profile=MI300A_V1,
                resource_cohorts=oracle_cohorts,
            )
            self.assertEqual(
                automatic_resource_score.placements,
                oracle_resource_score.placements,
            )
            self.assertEqual(
                automatic_resource_score.hardware_place,
                oracle_resource_score.hardware_place,
            )

    def test_bias_relu(self) -> None:
        self._check("bias_relu")

    def test_softmax_bias(self) -> None:
        self._check("softmax_bias")

    def test_gemv(self) -> None:
        self._check("gemv")

    def test_mvt(self) -> None:
        self._check("mvt")

    def test_gesummv(self) -> None:
        self._check("gesummv")

    def test_stencil5(self) -> None:
        self._check("stencil5")

    def test_gemm_prepacked_b(self) -> None:
        self._check("gemm_prepacked_b")

    def test_embedding_bag(self) -> None:
        self._check("embedding_bag")


if __name__ == "__main__":
    unittest.main()
