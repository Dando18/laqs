from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from relay.triton_frontend import (
    MANIFEST_SCHEMA,
    AnalysisOptions,
    EvaluationLimits,
    ExpressionEvaluator,
    ManifestError,
    TensorValue,
    UnsupportedTritonAnalysis,
    _EvaluationContext,
    analyze_compiled_manifest,
    analyze_launch,
    parse_access_manifest,
)
from relay.hardware_profiles import MI300A_V1
from relay.simple_solver import SimpleRelayProblem


class FakeDType:
    itemsize = 4

    def __str__(self) -> str:
        return "float32"


class FakeStorage:
    def __init__(self, pointer: int):
        self.pointer = pointer

    def data_ptr(self) -> int:
        return self.pointer


class FakeTensor:
    dtype = FakeDType()

    def __init__(
        self,
        shape: tuple[int, ...],
        pointer: int,
        *,
        strides: tuple[int, ...] | None = None,
        storage_pointer: int | None = None,
    ) -> None:
        self.shape = shape
        self.pointer = pointer
        self.strides = strides or self._contiguous_strides(shape)
        self.storage_pointer = pointer if storage_pointer is None else storage_pointer

    @staticmethod
    def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        stride = 1
        result = []
        for extent in reversed(shape):
            result.append(stride)
            stride *= extent
        return tuple(reversed(result))

    def data_ptr(self) -> int:
        return self.pointer

    def element_size(self) -> int:
        return 4

    def stride(self) -> tuple[int, ...]:
        return self.strides

    def untyped_storage(self) -> FakeStorage:
        return FakeStorage(self.storage_pointer)


class FakeIntegerTensor(FakeTensor):
    def __init__(
        self,
        values: tuple[int, ...],
        pointer: int,
        *,
        storage_pointer: int | None = None,
    ) -> None:
        super().__init__(
            (len(values),),
            pointer,
            storage_pointer=storage_pointer,
        )
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, _extent: int):
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeDescriptor:
    def __init__(
        self,
        base: FakeTensor,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        block_shape: tuple[int, ...],
        padding: str = "zero",
        round_f32_to_tf32: bool = False,
    ) -> None:
        self.base = base
        self.shape = shape
        self.strides = strides
        self.block_shape = block_shape
        self.padding = padding
        self.round_f32_to_tf32 = round_f32_to_tf32


def blocked_layout(size: int, *, replicated_lane_bit: bool = False) -> dict[str, object]:
    basis = [[1 << bit] for bit in range(size.bit_length() - 1)]
    free_masks = {}
    if replicated_lane_bit:
        basis.append([0])
        free_masks = {"lane": 1 << (len(basis) - 1)}
    return {
        "id": "layout0",
        "bases": [
            {"input": "register", "basis": []},
            {"input": "lane", "basis": basis},
            {"input": "warp", "basis": []},
            {"input": "block", "basis": []},
        ],
        "input_dims": [
            {"name": "register", "size": 1},
            {"name": "lane", "size": 1 << len(basis)},
            {"name": "warp", "size": 1},
            {"name": "block", "size": 1},
        ],
        "output_dims": [{"name": "element", "size": size}],
        "free_variable_masks": free_masks,
    }


def vector_manifest(block: int = 8) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "version": 1,
        "status": "supported",
        "args": [
            {"index": 0, "name": "x", "kind": "pointer"},
            {"index": 1, "name": "y", "kind": "pointer"},
            {"index": 2, "name": "n", "kind": "scalar"},
        ],
        "layouts": [blocked_layout(block)],
        "expressions": [
            {"id": 0, "op": "program_id", "type": "i32", "attributes": {"axis": 0}},
            {"id": 1, "op": "constant", "type": "i32", "attributes": {"value": str(block)}},
            {"id": 2, "op": "mul", "type": "i32", "operands": [0, 1]},
            {"id": 3, "op": "make_range", "type": f"tensor<{block}xi32>", "attributes": {"start": 0, "end": block}},
            {"id": 4, "op": "add", "type": f"tensor<{block}xi32>", "operands": [2, 3]},
            {"id": 5, "op": "arg", "type": "i32", "attributes": {"arg": 2}},
            {"id": 6, "op": "cmp", "type": f"tensor<{block}xi1>", "operands": [4, 5], "attributes": {"predicate": "slt"}},
        ],
        "body": [
            {
                "kind": "memory",
                "site_id": "x.load",
                "op": "load",
                "source": "kernel.py:10:4",
                "base": {"arg_index": 0, "name": "x", "path": []},
                "offset": 4,
                "mask": 6,
                "layout": "layout0",
                "shape": [block],
                "element_type": "f32",
                "element_bytes": 4,
                "lexical_order": 0,
                "issue": {},
            },
            {
                "kind": "memory",
                "site_id": "y.store",
                "op": "store",
                "base": {"arg_index": 1, "name": "y", "path": []},
                "offset": 4,
                "mask": 6,
                "layout": "layout0",
                "shape": [block],
                "element_type": "f32",
                "element_bytes": 4,
                "lexical_order": 1,
                "issue": {},
            },
        ],
        "diagnostics": [],
    }


def bound_arguments(n: int, *, aliases: bool = False) -> dict[int | str, object]:
    x = FakeTensor((n,), 0x1000, storage_pointer=0x1000)
    y = FakeTensor((n,), 0x2000, storage_pointer=0x1000 if aliases else 0x2000)
    return {
        0: x,
        1: y,
        2: n,
        "x": x,
        "y": y,
        "n": n,
        "__names__": {0: "x", 1: "y", 2: "n"},
    }


class ManifestParserAndExpressionTests(unittest.TestCase):
    def test_strict_version_and_deterministic_layout_parse(self) -> None:
        payload = vector_manifest()
        manifest = parse_access_manifest(json.dumps(payload, sort_keys=True))

        self.assertEqual(manifest.schema, MANIFEST_SCHEMA)
        self.assertEqual(manifest.layouts[0].layout.apply({"register": 0, "lane": 5, "warp": 0, "block": 0}), (5,))
        self.assertEqual(manifest.body[0].site_id, "x.load")
        bad = dict(payload, version=2)
        with self.assertRaisesRegex(ManifestError, "unsupported.*version"):
            parse_access_manifest(bad)

    def test_compiler_unsupported_diagnostic_is_structured(self) -> None:
        payload = vector_manifest()
        payload.update(
            status="unsupported",
            diagnostics=[
                {
                    "category": "unsupported.ambiguous_pointer_provenance",
                    "message": "memory.0 may address multiple allocations",
                    "site": "memory.0",
                }
            ],
        )
        analysis = analyze_compiled_manifest(
            object(), payload, (1,), bound_arguments(8)
        )

        self.assertFalse(analysis.supported)
        self.assertEqual(
            analysis.unsupported.category,
            "unsupported.ambiguous_pointer_provenance",
        )

    def test_expression_dag_shape_operations_and_integer_semantics(self) -> None:
        payload = vector_manifest(8)
        payload["expressions"] = [
            {"id": 0, "op": "make_range", "type": "tensor<8xi32>", "attributes": {"start": 0, "end": 8}},
            {"id": 1, "op": "reshape", "type": "tensor<2x4xi32>", "shape": [2, 4], "operands": [0]},
            {"id": 2, "op": "transpose", "type": "tensor<4x2xi32>", "operands": [1], "attributes": {"order": [1, 0]}},
            {"id": 3, "op": "constant", "type": "i32", "attributes": {"value": "0x3"}},
            {"id": 4, "op": "xor", "type": "tensor<4x2xi32>", "operands": [2, 3]},
        ]
        payload["body"] = [dict(payload["body"][0], offset=4, mask=None)]
        manifest = parse_access_manifest(payload)
        evaluator = ExpressionEvaluator(manifest, _EvaluationContext({}, (0, 0, 0), {}, {}))

        self.assertEqual(evaluator.evaluate(2), TensorValue((4, 2), (0, 4, 1, 5, 2, 6, 3, 7)))
        self.assertEqual(evaluator.evaluate(4).values, (3, 7, 2, 6, 1, 5, 0, 4))

    def test_fixed_width_overflow_unsigned_and_cast_semantics(self) -> None:
        payload = vector_manifest()
        payload["expressions"] = [
            {"id": 0, "op": "constant", "type": "i8", "attributes": {"value": "250"}},
            {"id": 1, "op": "constant", "type": "i8", "attributes": {"value": "10"}},
            {"id": 2, "op": "add", "type": "i8", "operands": [0, 1]},
            {"id": 3, "op": "udiv", "type": "i8", "operands": [0, 1]},
            {"id": 4, "op": "sdiv", "type": "i8", "operands": [0, 1]},
            {"id": 5, "op": "cmp", "type": "i1", "operands": [0, 1], "attributes": {"predicate": "ult"}},
            {"id": 6, "op": "cmp", "type": "i1", "operands": [0, 1], "attributes": {"predicate": "slt"}},
            {"id": 7, "op": "sext", "type": "i16", "operands": [0]},
            {"id": 8, "op": "zext", "type": "i16", "operands": [0]},
        ]
        payload["body"] = [dict(payload["body"][0], offset=2, mask=None)]
        manifest = parse_access_manifest(payload)
        evaluator = ExpressionEvaluator(manifest, _EvaluationContext({}, (0, 0, 0), {}, {}))

        self.assertEqual(evaluator.evaluate(2), 4)
        self.assertEqual(evaluator.evaluate(3), 25)
        self.assertEqual(evaluator.evaluate(4), 0)
        self.assertFalse(evaluator.evaluate(5))
        self.assertTrue(evaluator.evaluate(6))
        self.assertEqual(evaluator.evaluate(7), 65_530)
        self.assertEqual(evaluator.evaluate(8), 250)

    def test_pointer_typed_argument_is_a_base_relative_zero(self) -> None:
        payload = vector_manifest()
        payload["expressions"].extend(
            [
                {
                    "id": 7,
                    "op": "arg",
                    "type": "!tt.ptr<f32>",
                    "attributes": {"arg": "x", "path": []},
                },
                {
                    "id": 8,
                    "op": "splat",
                    "type": "tensor<8x!tt.ptr<f32>>",
                    "operands": [7],
                    "shape": [8],
                },
                {
                    "id": 9,
                    "op": "addptr",
                    "type": "tensor",
                    "operands": [8, 4],
                    "shape": [8],
                    "attributes": {"integer_width": 64},
                },
            ]
        )
        payload["body"][0]["offset"] = 9
        analysis = analyze_compiled_manifest(
            object(),
            payload,
            (100,),
            bound_arguments(800),
            options=AnalysisOptions(
                limits=EvaluationLimits(max_trace_contexts=8)
            ),
        )

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(sorted(sequence.weight for sequence in analysis.sequences), [100])

    def test_expression_limit_is_enforced_before_large_range_materialization(self) -> None:
        payload = vector_manifest()
        payload["expressions"] = [
            {
                "id": 0,
                "op": "make_range",
                "type": "tensor<1024xi32>",
                "attributes": {"start": 0, "end": 1024},
            }
        ]
        payload["body"] = [
            dict(payload["body"][0], offset=0, mask=None)
        ]
        evaluator = ExpressionEvaluator(
            parse_access_manifest(payload),
            _EvaluationContext({}, (0, 0, 0), {}, {}),
            max_tensor_elements=8,
        )

        with self.assertRaisesRegex(
            UnsupportedTritonAnalysis, "1024 tensor elements"
        ) as raised:
            evaluator.evaluate(0)
        self.assertEqual(raised.exception.category, "expression_bound")

    def test_tensor_descriptor_ir_field_uses_runtime_alias(self) -> None:
        payload = vector_manifest()
        payload["args"].append(
            {"index": 3, "name": "descriptor", "kind": "tensor_descriptor"}
        )
        payload["expressions"].append(
            {
                "id": 7,
                "op": "arg",
                "type": "i1",
                "attributes": {
                    "arg": 3,
                    "path": ["roundF32ToTF32"],
                },
            }
        )
        manifest = parse_access_manifest(payload)
        descriptor = FakeDescriptor(
            FakeTensor((8,), 0x3000),
            (8,),
            (1,),
            (8,),
            round_f32_to_tf32=True,
        )
        evaluator = ExpressionEvaluator(
            manifest,
            _EvaluationContext({3: descriptor}, (0, 0, 0), {}, {}),
        )

        self.assertTrue(evaluator.evaluate(7))

        unrelated = ExpressionEvaluator(
            manifest,
            _EvaluationContext(
                {3: SimpleNamespace(round_f32_to_tf32=True)},
                (0, 0, 0),
                {},
                {},
            ),
        )
        with self.assertRaises(AttributeError):
            unrelated.evaluate(7)

    def test_replicated_owner_mask_elects_one_lane(self) -> None:
        payload = vector_manifest(4)
        payload["layouts"] = [blocked_layout(4, replicated_lane_bit=True)]
        payload["expressions"][1]["attributes"]["value"] = "4"
        payload["expressions"][3]["attributes"]["end"] = 4
        payload["body"] = [dict(payload["body"][0], shape=[4])]
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound_arguments(4))

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual([access.lane for access in analysis.events[0].accesses], [0, 1, 2, 3])

    def test_linearly_dependent_nonzero_free_basis_is_accepted(self) -> None:
        payload = vector_manifest(4)
        layout = blocked_layout(4, replicated_lane_bit=True)
        layout["bases"][1]["basis"][-1] = [1]
        payload["layouts"] = [layout]
        payload["expressions"][1]["attributes"]["value"] = "4"
        payload["expressions"][3]["attributes"]["end"] = 4
        payload["body"] = [dict(payload["body"][0], shape=[4])]
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound_arguments(4))

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual([access.coord for access in analysis.events[0].accesses], [(0,), (1,), (2,), (3,)])


class StructuredTraceTests(unittest.TestCase):
    def test_boundary_trace_classes_envelope_and_address_oracle(self) -> None:
        analysis = analyze_compiled_manifest(object(), vector_manifest(), (4,), bound_arguments(29))

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(analysis.allocations[0].true_shape, (29,))
        self.assertEqual(analysis.matrices[0].shape, (32,))
        self.assertTrue(analysis.matrices[0].target)
        self.assertFalse(analysis.matrices[1].target)
        self.assertEqual(sorted(sequence.weight for sequence in analysis.sequences), [1, 3])
        observed = [
            (event.site, tuple((access.lane, access.coord[0]) for access in event.accesses))
            for event in analysis.events
        ]
        expected = [
            (site, tuple((lane, base + lane) for lane in range(width)))
            for base, width in ((0, 8), (24, 5))
            for site in ("x.load", "y.store")
        ]
        self.assertEqual(observed, expected)
        self.assertTrue(analysis.edge_families)
        self.assertEqual(
            [event.meta("phase") for event in analysis.events[:2]],
            ["root.sync0", "root.sync0"],
        )

    def test_hardware_profile_materializes_existing_universal_components(self) -> None:
        analysis = analyze_compiled_manifest(
            object(),
            vector_manifest(),
            (1,),
            bound_arguments(8),
            options=AnalysisOptions(hardware_profile=MI300A_V1),
        )

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertTrue(analysis.components)
        self.assertEqual(
            {component.region_bytes for component in analysis.components},
            set(MI300A_V1.byte_scales),
        )
        problem = analysis.relay_problem(hardware_profile=MI300A_V1)
        self.assertIsInstance(problem, SimpleRelayProblem)
        self.assertIs(problem.hardware_profile, MI300A_V1)
        self.assertEqual(problem.grammar, "standard")
        self.assertEqual(problem.events, analysis.events)
        self.assertEqual(problem.sequences, analysis.sequences)

    def test_resource_profile_preserves_absolute_trace_anchors(self) -> None:
        analysis = analyze_compiled_manifest(
            object(),
            vector_manifest(),
            (4,),
            bound_arguments(32),
            options=AnalysisOptions(hardware_profile=MI300A_V1),
        )

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(len(analysis.sequences), 4)
        self.assertEqual([sequence.weight for sequence in analysis.sequences], [1] * 4)

    def test_resource_profile_cannot_be_added_after_translation_compression(self) -> None:
        analysis = analyze_compiled_manifest(
            object(), vector_manifest(), (4,), bound_arguments(32)
        )

        self.assertTrue(analysis.supported, analysis.unsupported)
        with self.assertRaisesRegex(ValueError, "resource-color anchors"):
            analysis.relay_problem(hardware_profile=MI300A_V1)

    def test_structured_loop_carried_pointer_and_scalar_if(self) -> None:
        payload = vector_manifest(4)
        payload["args"].append({"index": 3, "name": "enabled", "kind": "scalar"})
        payload["layouts"] = [blocked_layout(4)]
        payload["expressions"] = [
            {"id": 0, "op": "constant", "type": "i32", "attributes": {"value": 0}},
            {"id": 1, "op": "constant", "type": "i32", "attributes": {"value": 2}},
            {"id": 2, "op": "constant", "type": "i32", "attributes": {"value": 1}},
            {"id": 3, "op": "loop_carried", "type": "i32", "attributes": {"name": "pointer"}},
            {"id": 4, "op": "make_range", "type": "tensor<4xi32>", "attributes": {"start": 0, "end": 4}},
            {"id": 5, "op": "add", "type": "tensor<4xi32>", "operands": [3, 4]},
            {"id": 6, "op": "constant", "type": "i32", "attributes": {"value": 4}},
            {"id": 7, "op": "add", "type": "i32", "operands": [3, 6]},
            {"id": 8, "op": "arg", "type": "i1", "attributes": {"arg": 3}},
        ]
        memory = dict(payload["body"][0], offset=5, mask=None, shape=[4])
        payload["body"] = [
            {
                "kind": "for",
                "iv": "iteration",
                "lower": 0,
                "upper": 1,
                "step": 2,
                "iter_args": [{"name": "pointer", "init": 0, "yield": 7}],
                "body": [
                    {
                        "kind": "if",
                        "condition": 8,
                        "then": [memory],
                        "else": [],
                    }
                ],
            }
        ]
        bound = bound_arguments(8)
        bound.update({3: 1, "enabled": 1, "__names__": {0: "x", 1: "y", 2: "n", 3: "enabled"}})
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound)

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual([[access.coord for access in event.accesses] for event in analysis.events], [[(0,), (1,), (2,), (3,)], [(4,), (5,), (6,), (7,)]])
        self.assertTrue(all("/iter" in event.meta("phase") for event in analysis.events))

    def test_runtime_loop_bound_is_checked_without_host_integer_overflow(self) -> None:
        payload = vector_manifest(4)
        payload["expressions"] = [
            {"id": 0, "op": "constant", "type": "i64", "attributes": {"value": 0}},
            {"id": 1, "op": "constant", "type": "i64", "attributes": {"value": 1 << 63}},
            {"id": 2, "op": "constant", "type": "i64", "attributes": {"value": 1}},
            {"id": 3, "op": "make_range", "type": "tensor<4xi64>", "attributes": {"start": 0, "end": 4}},
        ]
        payload["body"] = [
            {
                "kind": "for",
                "iv": "iteration",
                "lower": 0,
                "upper": 1,
                "step": 2,
                "body": [dict(payload["body"][0], offset=3, mask=None, shape=[4])],
            }
        ]
        analysis = analyze_compiled_manifest(
            object(), payload, (1,), bound_arguments(4)
        )

        self.assertFalse(analysis.supported)
        self.assertEqual(analysis.unsupported.category, "enumeration_bound")

    def test_permuted_dense_offset_inversion(self) -> None:
        payload = vector_manifest()
        payload["body"] = [payload["body"][0]]
        bound = bound_arguments(8)
        permuted = FakeTensor((2, 4), 0x1000, strides=(1, 2))
        bound[0] = bound["x"] = permuted
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound)

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(analysis.allocations[0].dense_status, "permuted_dense")
        self.assertEqual([access.coord for access in analysis.events[0].accesses], [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)])

    def test_dense_offset_inversion_ignores_singleton_dimensions(self) -> None:
        payload = vector_manifest()
        payload["body"] = [payload["body"][0]]
        bound = bound_arguments(8)
        degenerate = FakeTensor((8, 1), 0x1000, strides=(1, 1))
        bound[0] = bound["x"] = degenerate
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound)

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(analysis.allocations[0].dense_status, "dense")
        self.assertEqual(
            [access.coord for access in analysis.events[0].accesses],
            [(index, 0) for index in range(8)],
        )

    def test_tensor_descriptor_uses_concrete_shape_and_strides(self) -> None:
        payload = vector_manifest(4)
        payload["args"] = [{"index": 0, "name": "descriptor", "kind": "tensor_descriptor"}]
        payload["layouts"] = [
            {
                "id": "layout0",
                "bases": [
                    {"input": "register", "basis": []},
                    {"input": "lane", "basis": [[1, 0], [0, 1]]},
                    {"input": "warp", "basis": []},
                    {"input": "block", "basis": []},
                ],
                "input_dims": [
                    {"name": "register", "size": 1},
                    {"name": "lane", "size": 4},
                    {"name": "warp", "size": 1},
                    {"name": "block", "size": 1},
                ],
                "output_dims": [{"name": "row", "size": 2}, {"name": "column", "size": 2}],
                "free_variable_masks": {},
            }
        ]
        payload["expressions"] = [
            {"id": 0, "op": "constant", "type": "i64", "attributes": {"value": 1}},
            {"id": 1, "op": "constant", "type": "i64", "attributes": {"value": 3}},
        ]
        payload["body"] = [
            {
                "kind": "memory",
                "site_id": "descriptor.load",
                "op": "load",
                "base": {"arg_index": 0, "path": []},
                "offset": None,
                "descriptor": 0,
                "indices": [0, 1],
                "mask": None,
                "layout": "layout0",
                "shape": [2, 2],
                "element_type": "f32",
                "element_bytes": 4,
                "issue": {},
            }
        ]
        base = FakeTensor((3, 5), 0x3000, strides=(5, 1))
        descriptor = FakeDescriptor(base, (3, 5), (5, 1), (2, 2))
        bound = {0: descriptor, "descriptor": descriptor, "__names__": {0: "descriptor"}}
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound)

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(analysis.allocations[0].true_shape, (3, 5))
        self.assertEqual([access.coord for access in analysis.events[0].accesses], [(1, 3), (2, 3), (1, 4), (2, 4)])

        payload["expressions"][0]["attributes"]["value"] = 2
        payload["expressions"][1]["attributes"]["value"] = 4
        boundary = analyze_compiled_manifest(object(), payload, (1,), bound)
        self.assertTrue(boundary.supported, boundary.unsupported)
        self.assertEqual(
            [access.coord for access in boundary.events[0].accesses], [(2, 4)]
        )

    def test_out_of_bounds_descriptor_store_is_rejected(self) -> None:
        payload = vector_manifest(4)
        payload["args"] = [
            {"index": 0, "name": "output", "kind": "tensor_descriptor"},
            {"index": 1, "name": "input", "kind": "tensor_descriptor"},
        ]
        payload["layouts"] = [
            {
                "id": "layout0",
                "bases": [
                    {"input": "register", "basis": []},
                    {"input": "lane", "basis": [[1, 0], [0, 1]]},
                    {"input": "warp", "basis": []},
                    {"input": "block", "basis": []},
                ],
                "input_dims": [
                    {"name": "register", "size": 1},
                    {"name": "lane", "size": 4},
                    {"name": "warp", "size": 1},
                    {"name": "block", "size": 1},
                ],
                "output_dims": [
                    {"name": "row", "size": 2},
                    {"name": "column", "size": 2},
                ],
                "free_variable_masks": {},
            }
        ]
        payload["expressions"] = [
            {"id": 0, "op": "constant", "type": "i64", "attributes": {"value": 2}},
            {"id": 1, "op": "constant", "type": "i64", "attributes": {"value": 4}},
            {"id": 2, "op": "constant", "type": "i64", "attributes": {"value": 0}},
        ]

        def descriptor_memory(site: str, operation: str, argument: int, indices: list[int]) -> dict[str, object]:
            return {
                "kind": "memory",
                "site_id": site,
                "op": operation,
                "base": {"arg_index": argument, "path": []},
                "offset": None,
                "descriptor": argument,
                "indices": indices,
                "mask": None,
                "layout": "layout0",
                "shape": [2, 2],
                "element_type": "f32",
                "element_bytes": 4,
                "issue": {},
            }

        payload["body"] = [
            descriptor_memory("output.store", "store", 0, [0, 1]),
            descriptor_memory("input.load", "load", 1, [2, 2]),
        ]
        output = FakeDescriptor(FakeTensor((3, 5), 0x3000), (3, 5), (5, 1), (2, 2))
        input_value = FakeDescriptor(FakeTensor((3, 5), 0x4000), (3, 5), (5, 1), (2, 2))
        bound = {
            0: output,
            "output": output,
            1: input_value,
            "input": input_value,
            "__names__": {0: "output", 1: "input"},
        }
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound)

        self.assertFalse(analysis.supported)
        self.assertEqual(
            analysis.unsupported.category, "descriptor_out_of_bounds_store"
        )

    def test_aliases_are_fixed_and_report_no_candidate(self) -> None:
        analysis = analyze_compiled_manifest(object(), vector_manifest(), (1,), bound_arguments(8, aliases=True))

        self.assertFalse(analysis.supported)
        self.assertEqual(analysis.unsupported.category, "eligibility")

    def test_gather_rejects_index_storage_aliased_by_a_writer(self) -> None:
        payload = vector_manifest()
        payload["args"].append(
            {"index": 3, "name": "data", "kind": "pointer"}
        )
        payload["expressions"].append(
            {
                "id": 7,
                "op": "gather",
                "type": "tensor<8xi32>",
                "operands": [3],
                "attributes": {"arg": 0, "integer_width": 32},
            }
        )
        payload["body"].insert(
            1,
            {
                **payload["body"][0],
                "site_id": "data.load",
                "base": {"arg_index": 3, "name": "data", "path": []},
                "offset": 7,
                "mask": None,
                "lexical_order": 1,
            },
        )
        indices = FakeIntegerTensor(
            tuple(range(8)), 0x1000, storage_pointer=0x1000
        )
        output = FakeTensor((8,), 0x2000, storage_pointer=0x1000)
        data = FakeTensor((8,), 0x3000)
        bound = {
            0: indices,
            1: output,
            2: 8,
            3: data,
            "x": indices,
            "y": output,
            "n": 8,
            "data": data,
            "__names__": {0: "x", 1: "y", 2: "n", 3: "data"},
        }
        analysis = analyze_compiled_manifest(object(), payload, (1,), bound)

        self.assertFalse(analysis.supported)
        self.assertEqual(
            analysis.unsupported.category, "data_dependent_index"
        )

    def test_alias_group_names_are_launch_address_independent(self) -> None:
        first = analyze_compiled_manifest(
            object(), vector_manifest(), (1,), bound_arguments(8)
        )
        second = analyze_compiled_manifest(
            object(), vector_manifest(), (1,), bound_arguments(8)
        )

        self.assertTrue(first.supported, first.unsupported)
        self.assertEqual(
            [allocation.alias_group for allocation in first.allocations],
            ["alias.0", "alias.1"],
        )
        self.assertEqual(
            [allocation.alias_group for allocation in first.allocations],
            [allocation.alias_group for allocation in second.allocations],
        )

    def test_large_aligned_launch_uses_proved_trace_classes(self) -> None:
        grid = 100_001
        n = 800_003
        analysis = analyze_compiled_manifest(
            object(),
            vector_manifest(),
            (grid,),
            bound_arguments(n),
            options=AnalysisOptions(limits=EvaluationLimits(max_trace_contexts=8)),
        )

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(sorted(sequence.weight for sequence in analysis.sequences), [1, 100_000])
        self.assertEqual([len(event.accesses) for event in analysis.events], [8, 8, 3, 3])

    def test_large_aligned_launch_accepts_compiler_layout_wrappers(self) -> None:
        payload = vector_manifest()
        payload["expressions"].extend(
            [
                {
                    "id": 7,
                    "op": "convert_layout",
                    "type": "tensor",
                    "operands": [4],
                    "shape": [8],
                    "attributes": {"integer_width": 64},
                },
                {
                    "id": 8,
                    "op": "convert_layout",
                    "type": "tensor<8xi1>",
                    "operands": [6],
                    "shape": [8],
                },
            ]
        )
        payload["body"] = [
            dict(memory, offset=7, mask=8) for memory in payload["body"]
        ]
        analysis = analyze_compiled_manifest(
            object(),
            payload,
            (100,),
            bound_arguments(800),
            options=AnalysisOptions(
                limits=EvaluationLimits(max_trace_contexts=8)
            ),
        )

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual([sequence.weight for sequence in analysis.sequences], [100])

    def test_large_nonaffine_launch_is_categorized_unsupported(self) -> None:
        payload = vector_manifest()
        payload["expressions"].append({"id": 7, "op": "urem", "type": "i32", "operands": [0, 1]})
        payload["expressions"][2] = {"id": 2, "op": "mul", "type": "i32", "operands": [7, 1]}
        analysis = analyze_compiled_manifest(
            object(), payload, (100,), bound_arguments(800), options=AnalysisOptions(limits=EvaluationLimits(max_trace_contexts=8))
        )

        self.assertFalse(analysis.supported)
        self.assertEqual(analysis.unsupported.category, "enumeration_bound")

    def test_large_aligned_launch_proves_all_addresses_are_in_bounds(self) -> None:
        payload = vector_manifest()
        payload["body"] = [dict(payload["body"][0], mask=None)]
        analysis = analyze_compiled_manifest(
            object(),
            payload,
            (100,),
            bound_arguments(8),
            options=AnalysisOptions(
                limits=EvaluationLimits(max_trace_contexts=8)
            ),
        )

        self.assertFalse(analysis.supported)
        self.assertEqual(analysis.unsupported.category, "out_of_bounds")

    def test_large_wrapping_affine_launch_is_not_extrapolated(self) -> None:
        payload = vector_manifest()
        for expression in payload["expressions"]:
            if expression["id"] in {0, 1, 2, 3, 4}:
                expression["type"] = expression["type"].replace("i32", "i8")
        analysis = analyze_compiled_manifest(
            object(),
            payload,
            (100,),
            bound_arguments(800),
            options=AnalysisOptions(
                limits=EvaluationLimits(max_trace_contexts=8)
            ),
        )

        self.assertFalse(analysis.supported)
        self.assertEqual(analysis.unsupported.category, "enumeration_bound")

    def test_large_resource_profile_launch_is_categorized_unsupported(self) -> None:
        analysis = analyze_compiled_manifest(
            object(),
            vector_manifest(),
            (100,),
            bound_arguments(800),
            options=AnalysisOptions(
                hardware_profile=MI300A_V1,
                limits=EvaluationLimits(max_trace_contexts=8),
            ),
        )

        self.assertFalse(analysis.supported)
        self.assertEqual(analysis.unsupported.category, "resource_trace_bound")


class LaunchWrapperTests(unittest.TestCase):
    def test_autotune_heuristic_callable_grid_and_single_launch(self) -> None:
        manifest = json.dumps(vector_manifest())
        compiled = SimpleNamespace(metadata=SimpleNamespace(laqs_access_manifest=manifest))

        class Parameter:
            def __init__(self, name: str) -> None:
                self.name = name
                self.has_default = False

        class JITFunction:
            arg_names = ["x", "y", "n", "BLOCK", "EVEN"]
            params = [Parameter(name) for name in arg_names]

            def __init__(self) -> None:
                self.launch_count = 0
                self.pre_run_hooks = []

            def run(self, *args, **kwargs):
                self.launch_count += 1
                self.launch_kwargs = kwargs
                for hook in self.pre_run_hooks:
                    hook(*args, **kwargs)
                kwargs["grid"](
                    {
                        **dict(zip(self.arg_names, args)),
                        **kwargs,
                    }
                )
                return compiled

        class Heuristics:
            arg_names = JITFunction.arg_names

            def __init__(self, fn) -> None:
                self.fn = fn
                self.values = {"EVEN": lambda values: values["n"] % values["BLOCK"] == 0}

            def run(self, *args, **kwargs):
                kwargs["EVEN"] = self.values["EVEN"]({**dict(zip(self.arg_names, args)), **kwargs})
                return self.fn.run(*args, **kwargs)

        class Config:
            def all_kwargs(self):
                return {"BLOCK": 8, "num_warps": 4}

        class Autotuner:
            arg_names = JITFunction.arg_names

            def __init__(self, fn) -> None:
                self.fn = fn

            def run(self, *args, **kwargs):
                self.best_config = Config()
                kwargs.update(self.best_config.all_kwargs())
                return self.fn.run(*args, **kwargs)

        jit = JITFunction()
        kernel = Autotuner(Heuristics(jit))
        grid_calls = []

        def launch_grid(meta):
            grid_calls.append(dict(meta))
            return ((meta["n"] + meta["BLOCK"] - 1) // meta["BLOCK"],)

        @contextmanager
        def no_plugin(_path):
            yield

        with mock.patch("relay.triton_frontend._manifest_compilation", no_plugin):
            analysis = analyze_launch(
                kernel,
                launch_grid,
                *tuple(bound_arguments(13)[index] for index in range(3)),
            )

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertIs(analysis.compiled_kernel, compiled)
        self.assertEqual(jit.launch_count, 1)
        self.assertEqual(len(grid_calls), 1)
        self.assertEqual(analysis.grid, (2,))
        self.assertEqual(analysis.selected_config["BLOCK"], 8)
        self.assertFalse(analysis.selected_config["EVEN"])

    def test_launch_binding_retains_omitted_jit_defaults(self) -> None:
        payload = vector_manifest()
        payload["args"][2] = {
            "index": 2,
            "name": "n",
            "kind": "scalar",
        }
        manifest = json.dumps(payload)
        compiled = SimpleNamespace(
            metadata=SimpleNamespace(laqs_access_manifest=manifest)
        )

        class Parameter:
            def __init__(self, name: str, default=None) -> None:
                self.name = name
                self.has_default = default is not None
                self.default = default

        class JITFunction:
            arg_names = ["x", "y", "n"]
            params = [Parameter("x"), Parameter("y"), Parameter("n", 8)]

            def __init__(self) -> None:
                self.pre_run_hooks = []

            def run(self, *args, **kwargs):
                for hook in self.pre_run_hooks:
                    hook(*args, **kwargs)
                return compiled

        jit = JITFunction()

        @contextmanager
        def no_plugin(_path):
            yield

        arguments = bound_arguments(8)
        with mock.patch("relay.triton_frontend._manifest_compilation", no_plugin):
            analysis = analyze_launch(jit, (1,), arguments[0], arguments[1])

        self.assertTrue(analysis.supported, analysis.unsupported)
        self.assertEqual(analysis.bound_arguments[2], 8)
        self.assertEqual(analysis.bound_arguments["n"], 8)


if __name__ == "__main__":
    unittest.main()
