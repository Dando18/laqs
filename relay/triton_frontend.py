"""Automatic conversion of a compiled Triton access manifest into LAQS traces.

The compiler boundary is deliberately small: an analysis pass serializes a
typed expression DAG, native LinearLayouts, and a structured access program.
This module binds that target-independent description to one concrete launch
and evaluates it without inspecting printed IR.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .access_scopes import build_edge_families, materialize_edge_families
from .hardware import HardwareProfile
from .model import Access, EventSequence, MatrixSpec, MemoryEvent
from .triton import HardwareLocation, TritonLinearLayout


MANIFEST_SCHEMA = "laqs.triton.access_manifest"
MANIFEST_VERSION = 1
MANIFEST_METADATA_KEY = "laqs_access_manifest"


class ManifestError(ValueError):
    """A malformed or internally inconsistent compiler manifest."""


class UnsupportedTritonAnalysis(RuntimeError):
    """An exact trace cannot be constructed for a supported launch subset."""

    def __init__(self, category: str, message: str, *, site: str | None = None):
        super().__init__(message)
        self.category = category
        self.site = site


@dataclass(frozen=True)
class UnsupportedReason:
    category: str
    message: str
    site: str | None = None


def _freeze(value: Any, path: str = "manifest") -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ManifestError(f"{path}: object keys must be strings")
        return MappingProxyType(
            {key: _freeze(item, f"{path}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    raise ManifestError(f"{path}: value of type {type(value).__name__} is not JSON-compatible")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{path}: expected an object")
    return value


def _array(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"{path}: expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{path}: expected a nonempty string")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{path}: expected an integer")
    return value


def _path(value: Any, path: str) -> tuple[str | int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if not value:
            return ()
        return tuple(part for part in value.split(".") if part)
    items = _array(value, path)
    result: list[str | int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ManifestError(f"{path}[{index}]: path elements must be strings or integers")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class ManifestArgument:
    index: int
    name: str
    kind: str
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class ManifestExpression:
    id: int
    op: str
    result_type: str
    operands: tuple[int, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestLinearLayout:
    id: str
    layout: TritonLinearLayout
    input_dims: tuple[tuple[str, int], ...]
    free_variable_masks: tuple[tuple[str, int], ...]

    def free_mask(self, dimension: str) -> int:
        return dict(self.free_variable_masks).get(dimension, 0)


@dataclass(frozen=True)
class ManifestFor:
    iv: str
    lower: int
    upper: int
    step: int
    body: tuple["ManifestNode", ...]
    iter_args: tuple[tuple[str, int, int], ...] = ()
    lexical_order: int = 0


@dataclass(frozen=True)
class ManifestIf:
    condition: int
    then_body: tuple["ManifestNode", ...]
    else_body: tuple["ManifestNode", ...] = ()
    lexical_order: int = 0


@dataclass(frozen=True)
class ManifestMemory:
    site_id: str
    operation: str
    source: str
    base_arg: int | str
    base_path: tuple[str | int, ...]
    offset: int | None
    descriptor: int | None
    indices: tuple[int, ...]
    mask: int | None
    control_predicates: tuple[tuple[int, bool], ...]
    layout: str
    shape: tuple[int, ...]
    element_type: str
    element_bytes: int
    lexical_order: int
    issue: Mapping[str, Any]
    cache: str | None = None
    eviction: str | None = None


@dataclass(frozen=True)
class ManifestBarrier:
    lexical_order: int = 0


ManifestNode = ManifestFor | ManifestIf | ManifestMemory | ManifestBarrier


@dataclass(frozen=True)
class AccessManifest:
    arguments: tuple[ManifestArgument, ...]
    layouts: tuple[ManifestLinearLayout, ...]
    expressions: tuple[ManifestExpression, ...]
    body: tuple[ManifestNode, ...]
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    schema: str = MANIFEST_SCHEMA
    version: int = MANIFEST_VERSION
    status: str = "supported"

    @property
    def expression_map(self) -> dict[int, ManifestExpression]:
        return {expression.id: expression for expression in self.expressions}

    @property
    def layout_map(self) -> dict[str, ManifestLinearLayout]:
        return {layout.id: layout for layout in self.layouts}


def _parse_argument(value: Any, index: int) -> ManifestArgument:
    record = _object(value, f"args[{index}]")
    arg_index = _integer(record.get("index", index), f"args[{index}].index")
    name = record.get("name", f"arg{arg_index}")
    return ManifestArgument(
        arg_index,
        _string(name, f"args[{index}].name"),
        _string(record.get("kind", "scalar"), f"args[{index}].kind"),
        _path(record.get("path"), f"args[{index}].path"),
    )


def _parse_layout(value: Any, index: int) -> ManifestLinearLayout:
    record = _object(value, f"layouts[{index}]")
    raw_identifier = record.get("id")
    if isinstance(raw_identifier, bool) or not isinstance(raw_identifier, (str, int)):
        raise ManifestError(f"layouts[{index}].id: expected a string or integer")
    identifier = str(raw_identifier)
    basis_records = _array(record.get("bases"), f"layouts[{index}].bases")
    bases: list[tuple[str, tuple[tuple[int, ...], ...]]] = []
    for basis_index, raw_basis in enumerate(basis_records):
        basis_record = _object(raw_basis, f"layouts[{index}].bases[{basis_index}]")
        input_name = basis_record.get("input", basis_record.get("name"))
        vectors = []
        for vector_index, raw_vector in enumerate(
            _array(basis_record.get("basis", ()), f"layouts[{index}].bases[{basis_index}].basis")
        ):
            vectors.append(
                tuple(
                    _integer(component, f"layouts[{index}].bases[{basis_index}].basis[{vector_index}]")
                    for component in _array(raw_vector, "basis vector")
                )
            )
        bases.append((_string(input_name, "layout input dimension"), tuple(vectors)))

    raw_input_dims = record.get("input_dims")
    if raw_input_dims is None:
        input_dims = tuple((name, 1 << len(vectors)) for name, vectors in bases)
    else:
        input_dims = tuple(
            (
                _string(_object(item, "input dimension").get("name"), "input dimension name"),
                _integer(_object(item, "input dimension").get("size"), "input dimension size"),
            )
            for item in _array(raw_input_dims, f"layouts[{index}].input_dims")
        )
    out_key = "output_dims" if "output_dims" in record else "out_dims"
    output_dims = tuple(
        (
            _string(_object(item, "output dimension").get("name"), "output dimension name"),
            _integer(_object(item, "output dimension").get("size"), "output dimension size"),
        )
        for item in _array(record.get(out_key), f"layouts[{index}].{out_key}")
    )
    layout = TritonLinearLayout.from_bases(bases, output_dims)
    if tuple((name, layout.input_size(name)) for name in layout.input_dims) != input_dims:
        raise ManifestError(f"layout {identifier}: input sizes disagree with basis counts")

    raw_masks = _object(record.get("free_variable_masks", {}), "free_variable_masks")
    masks: list[tuple[str, int]] = []
    for name in layout.input_dims:
        raw_mask = raw_masks.get(name)
        if raw_mask is None:
            mask = sum(
                1 << bit
                for bit, vector in enumerate(dict(layout.bases)[name])
                if not any(vector)
            )
        elif isinstance(raw_mask, (list, tuple)):
            mask = sum(1 << _integer(bit, f"free_variable_masks.{name}") for bit in raw_mask)
        else:
            mask = _integer(raw_mask, f"free_variable_masks.{name}")
        if mask < 0 or mask >= layout.input_size(name):
            raise ManifestError(f"layout {identifier}: free-variable mask for {name!r} is out of range")
        masks.append((name, mask))
    return ManifestLinearLayout(identifier, layout, input_dims, tuple(masks))


_EXPRESSION_CORE_FIELDS = frozenset(("id", "op", "type", "result_type", "args", "operands", "attributes"))


def _parse_expression(value: Any, index: int) -> ManifestExpression:
    record = _object(value, f"expressions[{index}]")
    identifier = _integer(record.get("id"), f"expressions[{index}].id")
    raw_operands = record.get("operands", record.get("args", ()))
    operands = tuple(
        _integer(item, f"expressions[{index}].operands")
        for item in _array(raw_operands, f"expressions[{index}].operands")
    )
    nested_attributes = record.get("attributes", {})
    attributes = dict(_object(nested_attributes, f"expressions[{index}].attributes"))
    attributes.update({key: item for key, item in record.items() if key not in _EXPRESSION_CORE_FIELDS})
    return ManifestExpression(
        identifier,
        _string(record.get("op"), f"expressions[{index}].op"),
        _string(record.get("result_type", record.get("type", "index")), f"expressions[{index}].type"),
        operands,
        _freeze(attributes, f"expressions[{index}].attributes"),
    )


def _expression_ref(value: Any, path: str) -> int:
    if isinstance(value, Mapping):
        value = value.get("expr", value.get("id"))
    return _integer(value, path)


def _control_predicate(value: Any, path: str) -> tuple[int, bool]:
    if isinstance(value, Mapping):
        return (
            _expression_ref(value.get("expression", value.get("expr")), path),
            bool(value.get("polarity", True)),
        )
    return (_expression_ref(value, path), True)


def _parse_body(values: Any, path: str = "body") -> tuple[ManifestNode, ...]:
    result: list[ManifestNode] = []
    for index, value in enumerate(_array(values, path)):
        node_path = f"{path}[{index}]"
        record = _object(value, node_path)
        kind = _string(record.get("kind"), f"{node_path}.kind")
        lexical = _integer(record.get("lexical_order", index), f"{node_path}.lexical_order")
        if kind == "for":
            loop_id = record.get("iv", record.get("id"))
            loop_name = _string(loop_id, f"{node_path}.id")
            carried = []
            for carried_index, raw_carried in enumerate(_array(record.get("iter_args", ()), f"{node_path}.iter_args")):
                item = _object(raw_carried, f"{node_path}.iter_args[{carried_index}]")
                carried.append(
                    (
                        str(item.get("name", f"{loop_name}.slot{item.get('slot', carried_index)}")),
                        _expression_ref(item.get("init", item.get("initial")), "loop-carried init"),
                        _expression_ref(item.get("yield"), "loop-carried yield"),
                    )
                )
            result.append(
                ManifestFor(
                    loop_name,
                    _expression_ref(record.get("lower"), f"{node_path}.lower"),
                    _expression_ref(record.get("upper"), f"{node_path}.upper"),
                    _expression_ref(record.get("step"), f"{node_path}.step"),
                    _parse_body(record.get("body", ()), f"{node_path}.body"),
                    tuple(carried),
                    lexical,
                )
            )
        elif kind == "if":
            result.append(
                ManifestIf(
                    _expression_ref(record.get("condition"), f"{node_path}.condition"),
                    _parse_body(record.get("then", record.get("body", ())), f"{node_path}.then"),
                    _parse_body(record.get("else", ()), f"{node_path}.else"),
                    lexical,
                )
            )
        elif kind == "memory":
            operation = _string(record.get("op"), f"{node_path}.op")
            if operation not in {"load", "store", "atomic"}:
                raise ManifestError(f"{node_path}: unsupported memory operation {operation!r}")
            base = _object(record.get("base"), f"{node_path}.base")
            base_arg = base.get("arg", base.get("arg_index", base.get("index", base.get("name"))))
            if isinstance(base_arg, bool) or not isinstance(base_arg, (str, int)):
                raise ManifestError(f"{node_path}.base.arg: expected an argument name or index")
            raw_path = base.get("path")
            descriptor_path = raw_path == "descriptor.base"
            result.append(
                ManifestMemory(
                    _string(record.get("site_id"), f"{node_path}.site_id"),
                    operation,
                    str(record.get("source", "")),
                    base_arg,
                    () if descriptor_path else _path(raw_path, f"{node_path}.base.path"),
                    None if record.get("offset") is None else _expression_ref(record.get("offset"), f"{node_path}.offset"),
                    None if record.get("descriptor") is None else _expression_ref(record.get("descriptor"), f"{node_path}.descriptor"),
                    tuple(
                        _expression_ref(item, f"{node_path}.indices")
                        for item in _array(record.get("indices", ()), f"{node_path}.indices")
                    ),
                    None if record.get("mask") is None else _expression_ref(record.get("mask"), f"{node_path}.mask"),
                    tuple(
                        _control_predicate(item, f"{node_path}.control_predicates")
                        for item in _array(record.get("control_predicates", ()), f"{node_path}.control_predicates")
                    ),
                    str(record.get("layout")),
                    tuple(_integer(item, f"{node_path}.logical_shape") for item in _array(record.get("logical_shape", record.get("shape")), f"{node_path}.logical_shape")),
                    _string(record.get("element_type", "unknown"), f"{node_path}.element_type"),
                    _integer(record.get("element_bytes"), f"{node_path}.element_bytes"),
                    lexical,
                    _freeze(_object(record.get("issue", {}), f"{node_path}.issue")),
                    None if record.get("cache", record.get("attributes", {}).get("cache")) is None else str(record.get("cache", record.get("attributes", {}).get("cache"))),
                    None if record.get("eviction", record.get("attributes", {}).get("eviction")) is None else str(record.get("eviction", record.get("attributes", {}).get("eviction"))),
                )
            )
        elif kind == "barrier":
            result.append(ManifestBarrier(lexical))
        else:
            raise ManifestError(f"{node_path}: unknown structured node kind {kind!r}")
    return tuple(result)


def _iter_nodes(nodes: Iterable[ManifestNode]) -> Iterator[ManifestNode]:
    for node in nodes:
        yield node
        if isinstance(node, ManifestFor):
            yield from _iter_nodes(node.body)
        elif isinstance(node, ManifestIf):
            yield from _iter_nodes(node.then_body)
            yield from _iter_nodes(node.else_body)


def parse_access_manifest(payload: str | bytes | Mapping[str, Any]) -> AccessManifest:
    """Parse and validate the versioned compiler-to-runtime boundary."""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
            if isinstance(payload, str):
                payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ManifestError(f"access manifest is not valid JSON: {error}") from error
    record = _object(payload, "manifest")
    schema = record.get("schema")
    version = record.get("version")
    if schema != MANIFEST_SCHEMA:
        raise ManifestError(f"unknown access-manifest schema {schema!r}")
    if version != MANIFEST_VERSION:
        raise ManifestError(f"unsupported access-manifest version {version!r}")
    status = _string(record.get("status", "supported"), "manifest.status")
    if status not in {"supported", "unsupported"}:
        raise ManifestError(f"manifest.status: unknown status {status!r}")
    raw_arguments = record.get("args", record.get("arguments", ()))
    arguments = tuple(_parse_argument(item, index) for index, item in enumerate(_array(raw_arguments, "manifest.arguments")))
    layouts = tuple(_parse_layout(item, index) for index, item in enumerate(_array(record.get("layouts", ()), "manifest.layouts")))
    expressions = tuple(
        _parse_expression(item, index)
        for index, item in enumerate(_array(record.get("expressions", ()), "manifest.expressions"))
    )
    if len({argument.index for argument in arguments}) != len(arguments):
        raise ManifestError("manifest argument indices must be unique")
    if len({layout.id for layout in layouts}) != len(layouts):
        raise ManifestError("manifest layout ids must be unique")
    expression_ids = {expression.id for expression in expressions}
    if len(expression_ids) != len(expressions):
        raise ManifestError("manifest expression ids must be unique")
    for expression in expressions:
        missing = set(expression.operands) - expression_ids
        if missing:
            raise ManifestError(f"expression {expression.id}: unknown operands {sorted(missing)}")
    raw_body = _array(record.get("body", ()), "manifest.body")
    functions = [item for item in raw_body if isinstance(item, Mapping) and item.get("kind") == "function"]
    if functions:
        if len(functions) != 1 or len(functions) != len(raw_body):
            raise ManifestError("access manifest must contain exactly one structured entry function")
        raw_body = _array(functions[0].get("body", ()), "manifest.body[0].body")
    body = () if status == "unsupported" else _parse_body(raw_body)
    layout_ids = {layout.id for layout in layouts}
    references: list[tuple[str, int]] = []
    for node in _iter_nodes(body):
        if isinstance(node, ManifestMemory):
            if node.layout not in layout_ids:
                raise ManifestError(f"memory site {node.site_id}: unknown layout {node.layout!r}")
            references.extend(
                (node.site_id, ref)
                for ref in (
                    *((node.offset,) if node.offset is not None else ()),
                    *((node.descriptor,) if node.descriptor is not None else ()),
                    *node.indices,
                    *(expression for expression, _polarity in node.control_predicates),
                )
            )
            if node.mask is not None:
                references.append((node.site_id, node.mask))
        elif isinstance(node, ManifestFor):
            references.extend((f"loop {node.iv}", ref) for ref in (node.lower, node.upper, node.step))
            references.extend((f"loop {node.iv}", ref) for _, init, yielded in node.iter_args for ref in (init, yielded))
        elif isinstance(node, ManifestIf):
            references.append(("if", node.condition))
    for owner, expression_id in references:
        if expression_id not in expression_ids:
            raise ManifestError(f"{owner}: unknown expression {expression_id}")
    diagnostics = tuple(
        _freeze(_object(item, f"diagnostics[{index}]"))
        for index, item in enumerate(_array(record.get("diagnostics", ()), "manifest.diagnostics"))
    )
    if status == "supported" and not body:
        raise ManifestError("a supported access manifest must have a structured body")
    return AccessManifest(arguments, layouts, expressions, body, diagnostics, schema, version, status)


@dataclass(frozen=True)
class TensorValue:
    shape: tuple[int, ...]
    values: tuple[int | bool, ...]

    def __post_init__(self) -> None:
        size = math.prod(self.shape)
        if any(extent <= 0 for extent in self.shape) or len(self.values) != size:
            raise UnsupportedTritonAnalysis("expression_shape", f"tensor value shape {self.shape} has {len(self.values)} elements")

    @classmethod
    def full(cls, shape: Sequence[int], value: int | bool) -> "TensorValue":
        resolved = tuple(int(extent) for extent in shape)
        return cls(resolved, (value,) * math.prod(resolved))

    def at(self, coord: Sequence[int]) -> int | bool:
        if len(coord) != len(self.shape):
            raise UnsupportedTritonAnalysis("expression_shape", f"coordinate {tuple(coord)} does not match tensor shape {self.shape}")
        flat = 0
        for component, extent in zip(coord, self.shape):
            if component < 0 or component >= extent:
                raise UnsupportedTritonAnalysis("expression_shape", f"coordinate {tuple(coord)} is outside {self.shape}")
            flat = flat * extent + component
        return self.values[flat]


ScalarValue = int | bool
RuntimeValue = ScalarValue | TensorValue


def _coords(shape: Sequence[int]) -> Iterator[tuple[int, ...]]:
    if not shape:
        yield ()
        return
    from itertools import product

    yield from product(*(range(extent) for extent in shape))


def _flat_index(coord: Sequence[int], shape: Sequence[int]) -> int:
    result = 0
    for component, extent in zip(coord, shape):
        result = result * extent + component
    return result


def _broadcast(value: RuntimeValue, shape: tuple[int, ...]) -> TensorValue:
    if not isinstance(value, TensorValue):
        return TensorValue.full(shape, value)
    if value.shape == shape:
        return value
    if len(value.shape) > len(shape):
        raise UnsupportedTritonAnalysis("expression_shape", f"cannot broadcast {value.shape} to {shape}")
    padded = (1,) * (len(shape) - len(value.shape)) + value.shape
    if any(source not in (1, target) for source, target in zip(padded, shape)):
        raise UnsupportedTritonAnalysis("expression_shape", f"cannot broadcast {value.shape} to {shape}")
    values = []
    for coord in _coords(shape):
        source_coord = tuple(0 if extent == 1 else component for component, extent in zip(coord, padded))
        source_coord = source_coord[len(padded) - len(value.shape) :]
        values.append(value.at(source_coord))
    return TensorValue(shape, tuple(values))


def _elementwise(function: Any, *values: RuntimeValue) -> RuntimeValue:
    tensor_shapes = [value.shape for value in values if isinstance(value, TensorValue)]
    if not tensor_shapes:
        return function(*values)
    shape = _elementwise_shape(tensor_shapes)
    broadcast = [_broadcast(value, shape) for value in values]
    return TensorValue(shape, tuple(function(*(value.values[index] for value in broadcast)) for index in range(math.prod(shape))))


def _elementwise_shape(tensor_shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    rank = max(len(shape) for shape in tensor_shapes)
    result_shape = [1] * rank
    for shape in tensor_shapes:
        padded = (1,) * (rank - len(shape)) + shape
        for dimension, extent in enumerate(padded):
            if result_shape[dimension] not in (1, extent) and extent != 1:
                raise UnsupportedTritonAnalysis("expression_shape", f"incompatible broadcast shapes {tensor_shapes}")
            result_shape[dimension] = max(result_shape[dimension], extent)
    return tuple(result_shape)


def _signed_div(left: int, right: int) -> int:
    if right == 0:
        raise UnsupportedTritonAnalysis("division_by_zero", "division by zero while evaluating manifest")
    quotient = abs(left) // abs(right)
    return -quotient if (left < 0) != (right < 0) else quotient


def _signed_rem(left: int, right: int) -> int:
    return left - _signed_div(left, right) * right


def _integer_width(type_name: str) -> int | None:
    if type_name == "index":
        return 64
    for index, character in enumerate(type_name):
        if character != "i" or index + 1 >= len(type_name) or not type_name[index + 1].isdigit():
            continue
        end = index + 1
        while end < len(type_name) and type_name[end].isdigit():
            end += 1
        width = int(type_name[index + 1 : end])
        if width > 0:
            return width
    return None


def _unsigned(value: int | bool, width: int) -> int:
    return int(value) & ((1 << width) - 1)


def _signed(value: int | bool, width: int) -> int:
    result = _unsigned(value, width)
    return result - (1 << width) if result & (1 << (width - 1)) else result


def _require_integer_width(expression: ManifestExpression) -> int:
    width = expression.attributes.get("integer_width")
    if width is None:
        width = _integer_width(expression.result_type)
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise UnsupportedTritonAnalysis(
            "integer_type",
            f"expression {expression.id}: cannot determine fixed integer width from {expression.result_type!r}",
        )
    return width


def _optional_integer_width(expression: ManifestExpression) -> int | None:
    width = expression.attributes.get("integer_width")
    if width is None:
        width = _integer_width(expression.result_type)
    if width is None:
        return None
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ManifestError(
            f"expression {expression.id}: integer_width must be a positive integer"
        )
    return width


_BINARY_OPS = {
    "add": lambda a, b: a + b,
    "addi": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "subi": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "muli": lambda a, b: a * b,
    "shl": lambda a, b: a << b,
    "shr": lambda a, b: a >> b,
    "ashr": lambda a, b: a >> b,
    "lshr": lambda a, b: a >> b if a >= 0 else (_ for _ in ()).throw(UnsupportedTritonAnalysis("integer_semantics", "logical shift requires a declared integer width")),
    "div": _signed_div,
    "divsi": _signed_div,
    "floordiv": lambda a, b: a // b,
    "udiv": lambda a, b: a // b,
    "divui": lambda a, b: a // b,
    "mod": _signed_rem,
    "remsi": _signed_rem,
    "remui": lambda a, b: a % b,
    "and": lambda a, b: a & b,
    "andi": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "ori": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "xori": lambda a, b: a ^ b,
    "min": min,
    "max": max,
    "minimum": min,
    "maximum": max,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "slt": lambda a, b: a < b,
    "sle": lambda a, b: a <= b,
    "sgt": lambda a, b: a > b,
    "sge": lambda a, b: a >= b,
    "ult": lambda a, b: a >= 0 and b >= 0 and a < b,
    "ule": lambda a, b: a >= 0 and b >= 0 and a <= b,
    "ugt": lambda a, b: a >= 0 and b >= 0 and a > b,
    "uge": lambda a, b: a >= 0 and b >= 0 and a >= b,
}


_FIXED_WIDTH_BINARY_OPS = frozenset(
    {
        "add",
        "addi",
        "sub",
        "subi",
        "mul",
        "muli",
        "shl",
        "shr",
        "ashr",
        "lshr",
        "div",
        "divsi",
        "floordiv",
        "udiv",
        "divui",
        "mod",
        "remsi",
        "remui",
        "and",
        "andi",
        "or",
        "ori",
        "xor",
        "xori",
        "smin",
        "umin",
        "smax",
        "umax",
    }
)


def _canonical_op(op: str) -> str:
    result = op.lower().replace("tt.", "").replace("arith.", "")
    aliases = {
        "addptr": "add",
        "add_ptr": "add",
        "make_range": "make_range",
        "broadcast_to": "broadcast",
        "expand_dims": "expand_dims",
        "convert_layout": "convert_layout",
        "bitcast": "cast",
        "extsi": "cast",
        "extui": "cast",
        "trunci": "cast",
        "index_cast": "cast",
        "cmpi": "cmp",
        "loop_iv": "iv",
        "loop_carried": "variable",
        "sdiv": "divsi",
        "urem": "remui",
        "srem": "remsi",
        "sext": "cast",
        "zext": "cast",
        "trunc": "cast",
        "ptr_to_int": "cast",
        "int_to_ptr": "cast",
    }
    return aliases.get(result, result)


@dataclass
class _EvaluationContext:
    arguments: Mapping[int | str, Any]
    pid: tuple[int, int, int]
    variables: dict[str, RuntimeValue]
    readonly_tensors: Mapping[int | str, Any]
    grid: tuple[int, int, int] = (1, 1, 1)


class ExpressionEvaluator:
    """Exact interpreter for the access-manifest integer expression subset."""

    def __init__(self, manifest: AccessManifest, context: _EvaluationContext, *, max_tensor_elements: int = 1 << 18):
        self.expressions = manifest.expression_map
        self.context = context
        self.max_tensor_elements = max_tensor_elements
        self._active: set[int] = set()
        self._cache: dict[int, RuntimeValue] = {}

    def evaluate(self, expression_id: int) -> RuntimeValue:
        cached = self._cache.get(expression_id)
        if cached is not None:
            return cached
        if expression_id in self._active:
            raise ManifestError(f"expression DAG contains a cycle at {expression_id}")
        try:
            expression = self.expressions[expression_id]
        except KeyError as error:
            raise ManifestError(f"unknown expression {expression_id}") from error
        self._active.add(expression_id)
        operands = tuple(self.evaluate(operand) for operand in expression.operands)
        try:
            result = self._apply(expression, operands)
        finally:
            self._active.remove(expression_id)
        if isinstance(result, TensorValue) and len(result.values) > self.max_tensor_elements:
            raise UnsupportedTritonAnalysis("expression_bound", f"expression {expression_id} materializes {len(result.values)} tensor elements; limit is {self.max_tensor_elements}")
        self._cache[expression_id] = result
        return result

    def _check_shape_bound(self, shape: Sequence[int], owner: str) -> None:
        elements = math.prod(shape)
        if elements > self.max_tensor_elements:
            raise UnsupportedTritonAnalysis(
                "expression_bound",
                f"{owner} materializes {elements} tensor elements; "
                f"limit is {self.max_tensor_elements}",
            )

    def _bounded_broadcast(
        self, value: RuntimeValue, shape: tuple[int, ...], owner: str
    ) -> TensorValue:
        self._check_shape_bound(shape, owner)
        return _broadcast(value, shape)

    def _bounded_elementwise(
        self, function: Any, *values: RuntimeValue, owner: str
    ) -> RuntimeValue:
        tensor_shapes = [
            value.shape for value in values if isinstance(value, TensorValue)
        ]
        if tensor_shapes:
            self._check_shape_bound(_elementwise_shape(tensor_shapes), owner)
        return _elementwise(function, *values)

    def _apply(self, expression: ManifestExpression, operands: tuple[RuntimeValue, ...]) -> RuntimeValue:
        op = _canonical_op(expression.op)
        attrs = expression.attributes
        if op in {"constant", "const"}:
            value = attrs.get("value")
            width = _optional_integer_width(expression)
            if isinstance(value, bool):
                return value if width == 1 else int(value)
            if isinstance(value, int):
                return _unsigned(value, width) if width is not None else value
            if isinstance(value, str):
                try:
                    parsed = int(value, 0)
                    return _unsigned(parsed, width) if width is not None else parsed
                except ValueError as error:
                    raise ManifestError(f"expression {expression.id}: invalid integer spelling {value!r}") from error
            if value is None and isinstance(attrs.get("values"), tuple):
                value = attrs["values"]
            if isinstance(value, tuple):
                shape = tuple(int(extent) for extent in attrs.get("shape", (len(value),)))
                self._check_shape_bound(shape, f"expression {expression.id}")
                flattened = tuple(
                    int(item, 0) if isinstance(item, str) else int(item) if not isinstance(item, bool) else item
                    for item in value
                )
                if width is not None:
                    flattened = tuple(_unsigned(item, width) for item in flattened)
                return TensorValue(shape, flattened)
            raise ManifestError(f"expression {expression.id}: integer constant has invalid value {value!r}")
        if op in {"argument", "arg"}:
            key = attrs.get("arg", attrs.get("index", attrs.get("name")))
            if key not in self.context.arguments:
                raise UnsupportedTritonAnalysis("argument_binding", f"expression {expression.id}: launch argument {key!r} is unavailable")
            value = _follow_path(self.context.arguments[key], tuple(attrs.get("path", ())))
            if "ptr<" in expression.result_type or expression.result_type.startswith("ptr"):
                # Pointer expressions are represented relative to the unique
                # allocation selected by the manifest's provenance record.
                return 0
            if isinstance(value, bool):
                return value
            if isinstance(value, int):
                width = _optional_integer_width(expression)
                return _unsigned(value, width) if width is not None else value
            if hasattr(value, "value") and isinstance(value.value, (bool, int)):
                return value.value
            raise UnsupportedTritonAnalysis("runtime_scalar", f"expression {expression.id}: argument {key!r} is not an integer scalar")
        if op in {"program_id", "programid"}:
            axis = int(attrs.get("axis", 0))
            if axis < 0 or axis >= 3:
                raise ManifestError(f"expression {expression.id}: program_id axis must be 0, 1, or 2")
            width = _optional_integer_width(expression)
            value = self.context.pid[axis]
            return _unsigned(value, width) if width is not None else value
        if op in {"num_programs", "numprograms"}:
            axis = int(attrs.get("axis", 0))
            if axis < 0 or axis >= 3:
                raise ManifestError(
                    f"expression {expression.id}: num_programs axis must be 0, 1, or 2"
                )
            width = _optional_integer_width(expression)
            value = self.context.grid[axis]
            return _unsigned(value, width) if width is not None else value
        if op in {"iv", "symbol", "variable"}:
            if op == "variable" and "loop" in attrs and "slot" in attrs:
                name = f"{attrs['loop']}.slot{attrs['slot']}"
            else:
                name = str(attrs.get("name", attrs.get("loop")))
            try:
                return self.context.variables[name]
            except KeyError as error:
                raise ManifestError(f"expression {expression.id}: unbound structured variable {name!r}") from error
        if op == "make_range":
            if operands:
                if len(operands) != 2 or any(isinstance(value, TensorValue) for value in operands):
                    raise ManifestError(f"expression {expression.id}: make_range needs two scalar operands")
                start, end = (int(value) for value in operands)
            else:
                start, end = int(attrs.get("start", 0)), int(attrs["end"])
            if end <= start:
                raise UnsupportedTritonAnalysis("expression_shape", f"expression {expression.id}: empty make_range is unsupported")
            self._check_shape_bound(
                (end - start,), f"expression {expression.id}"
            )
            width = _optional_integer_width(expression)
            values = tuple(range(start, end))
            if width is not None:
                values = tuple(_unsigned(value, width) for value in values)
            return TensorValue((end - start,), values)
        if op in _FIXED_WIDTH_BINARY_OPS:
            if len(operands) != 2:
                raise ManifestError(f"expression {expression.id}: {op} needs two operands")
            width = _require_integer_width(expression)

            def binary(left: int, right: int) -> int:
                if op in {"add", "addi"}:
                    result = int(left) + int(right)
                elif op in {"sub", "subi"}:
                    result = int(left) - int(right)
                elif op in {"mul", "muli"}:
                    result = int(left) * int(right)
                elif op == "shl":
                    if int(right) < 0 or int(right) >= width:
                        raise UnsupportedTritonAnalysis("integer_semantics", f"expression {expression.id}: shift amount {right} is outside [0, {width})")
                    result = int(left) << int(right)
                elif op in {"shr", "ashr"}:
                    if int(right) < 0 or int(right) >= width:
                        raise UnsupportedTritonAnalysis("integer_semantics", f"expression {expression.id}: shift amount {right} is outside [0, {width})")
                    result = _signed(left, width) >> int(right)
                elif op == "lshr":
                    if int(right) < 0 or int(right) >= width:
                        raise UnsupportedTritonAnalysis("integer_semantics", f"expression {expression.id}: shift amount {right} is outside [0, {width})")
                    result = _unsigned(left, width) >> int(right)
                elif op in {"div", "divsi"}:
                    result = _signed_div(_signed(left, width), _signed(right, width))
                elif op == "floordiv":
                    divisor = _signed(right, width)
                    if divisor == 0:
                        raise UnsupportedTritonAnalysis("division_by_zero", "division by zero while evaluating manifest")
                    result = _signed(left, width) // divisor
                elif op in {"udiv", "divui"}:
                    divisor = _unsigned(right, width)
                    if divisor == 0:
                        raise UnsupportedTritonAnalysis("division_by_zero", "division by zero while evaluating manifest")
                    result = _unsigned(left, width) // divisor
                elif op in {"mod", "remsi"}:
                    result = _signed_rem(_signed(left, width), _signed(right, width))
                elif op == "remui":
                    divisor = _unsigned(right, width)
                    if divisor == 0:
                        raise UnsupportedTritonAnalysis("division_by_zero", "division by zero while evaluating manifest")
                    result = _unsigned(left, width) % divisor
                elif op in {"and", "andi"}:
                    result = int(left) & int(right)
                elif op in {"or", "ori"}:
                    result = int(left) | int(right)
                elif op in {"xor", "xori"}:
                    result = int(left) ^ int(right)
                elif op == "smin":
                    result = min(_signed(left, width), _signed(right, width))
                elif op == "smax":
                    result = max(_signed(left, width), _signed(right, width))
                elif op == "umin":
                    result = min(_unsigned(left, width), _unsigned(right, width))
                elif op == "umax":
                    result = max(_unsigned(left, width), _unsigned(right, width))
                else:
                    raise AssertionError(op)
                return _unsigned(result, width)

            return self._bounded_elementwise(
                binary, *operands, owner=f"expression {expression.id}"
            )
        if op in {"min", "max", "minimum", "maximum"}:
            signedness = expression.attributes.get("signed")
            if signedness is None:
                raise UnsupportedTritonAnalysis(
                    "integer_semantics",
                    f"expression {expression.id}: min/max signedness is not encoded",
                )
            resolved = ("s" if signedness else "u") + ("min" if op in {"min", "minimum"} else "max")
            equivalent = replace(expression, op=resolved)
            return self._apply(equivalent, operands)
        if op == "cmp":
            if len(operands) != 2:
                raise ManifestError(f"expression {expression.id}: cmp needs two operands")
            predicate = str(attrs.get("predicate", attrs.get("pred", ""))).lower()
            operand_width = _require_integer_width(
                self.expressions[expression.operands[0]]
            )
            comparisons = {
                "eq": lambda a, b: _unsigned(a, operand_width) == _unsigned(b, operand_width),
                "ne": lambda a, b: _unsigned(a, operand_width) != _unsigned(b, operand_width),
                "slt": lambda a, b: _signed(a, operand_width) < _signed(b, operand_width),
                "sle": lambda a, b: _signed(a, operand_width) <= _signed(b, operand_width),
                "sgt": lambda a, b: _signed(a, operand_width) > _signed(b, operand_width),
                "sge": lambda a, b: _signed(a, operand_width) >= _signed(b, operand_width),
                "ult": lambda a, b: _unsigned(a, operand_width) < _unsigned(b, operand_width),
                "ule": lambda a, b: _unsigned(a, operand_width) <= _unsigned(b, operand_width),
                "ugt": lambda a, b: _unsigned(a, operand_width) > _unsigned(b, operand_width),
                "uge": lambda a, b: _unsigned(a, operand_width) >= _unsigned(b, operand_width),
            }
            try:
                operation = comparisons[predicate]
            except KeyError as error:
                raise UnsupportedTritonAnalysis("expression_operation", f"expression {expression.id}: unsupported comparison {predicate!r}") from error
            return self._bounded_elementwise(
                operation, *operands, owner=f"expression {expression.id}"
            )
        if op in {"select", "where"}:
            if len(operands) != 3:
                raise ManifestError(f"expression {expression.id}: select needs three operands")
            return self._bounded_elementwise(
                lambda condition, yes, no: yes if condition else no,
                *operands,
                owner=f"expression {expression.id}",
            )
        if op in {"splat", "broadcast"}:
            if len(operands) != 1:
                raise ManifestError(f"expression {expression.id}: {op} needs one operand")
            shape = tuple(int(extent) for extent in attrs.get("shape", ()))
            if not shape:
                raise ManifestError(f"expression {expression.id}: {op} requires shape")
            return self._bounded_broadcast(
                operands[0], shape, f"expression {expression.id}"
            )
        if op == "expand_dims":
            if len(operands) != 1 or not isinstance(operands[0], TensorValue):
                raise ManifestError(f"expression {expression.id}: expand_dims needs one tensor")
            axis = int(attrs.get("axis"))
            source = operands[0]
            if axis < 0:
                axis += len(source.shape) + 1
            if axis < 0 or axis > len(source.shape):
                raise ManifestError(f"expression {expression.id}: invalid expand_dims axis {axis}")
            shape = source.shape[:axis] + (1,) + source.shape[axis:]
            return TensorValue(shape, source.values)
        if op == "reshape":
            if len(operands) != 1 or not isinstance(operands[0], TensorValue):
                raise ManifestError(f"expression {expression.id}: reshape needs one tensor")
            shape = tuple(int(extent) for extent in attrs.get("shape", ()))
            if math.prod(shape) != len(operands[0].values):
                raise UnsupportedTritonAnalysis("expression_shape", f"expression {expression.id}: reshape changes element count")
            self._check_shape_bound(shape, f"expression {expression.id}")
            return TensorValue(shape, operands[0].values)
        if op == "transpose":
            if len(operands) != 1 or not isinstance(operands[0], TensorValue):
                raise ManifestError(f"expression {expression.id}: transpose needs one tensor")
            source = operands[0]
            permutation = tuple(int(item) for item in attrs.get("order", attrs.get("permutation", ())))
            if sorted(permutation) != list(range(len(source.shape))):
                raise ManifestError(f"expression {expression.id}: invalid transpose permutation")
            shape = tuple(source.shape[dimension] for dimension in permutation)
            values = []
            for coord in _coords(shape):
                source_coord = [0] * len(shape)
                for output_dimension, source_dimension in enumerate(permutation):
                    source_coord[source_dimension] = coord[output_dimension]
                values.append(source.at(source_coord))
            return TensorValue(shape, tuple(values))
        if op == "convert_layout":
            if len(operands) != 1:
                raise ManifestError(f"expression {expression.id}: convert_layout needs one operand")
            return operands[0]
        if op == "cast":
            if len(operands) != 1:
                raise ManifestError(f"expression {expression.id}: cast needs one operand")
            target_width = _require_integer_width(expression)
            source_width = _require_integer_width(
                self.expressions[expression.operands[0]]
            )
            raw_op = expression.op.lower().split(".")[-1]

            def cast(value: int) -> int:
                if raw_op in {"extsi", "sext"}:
                    value = _signed(value, source_width)
                else:
                    value = _unsigned(value, source_width)
                return _unsigned(value, target_width)
            return self._bounded_elementwise(
                cast, operands[0], owner=f"expression {expression.id}"
            )
        if op in {"gather", "load_readonly_integer"}:
            return self._gather(expression, operands)
        raise UnsupportedTritonAnalysis("expression_operation", f"expression {expression.id}: unsupported operation {expression.op!r}")

    def _gather(self, expression: ManifestExpression, operands: tuple[RuntimeValue, ...]) -> RuntimeValue:
        if len(operands) != 1:
            raise ManifestError(f"expression {expression.id}: gather needs one index operand")
        key = expression.attributes.get("arg", expression.attributes.get("index"))
        try:
            tensor = self.context.readonly_tensors[key]
        except KeyError as error:
            raise UnsupportedTritonAnalysis("data_dependent_index", f"expression {expression.id}: read-only integer tensor {key!r} was not copied") from error
        flat = _copy_integer_tensor(tensor)
        def load(index: int) -> int:
            if index < 0 or index >= len(flat):
                raise UnsupportedTritonAnalysis("data_dependent_index", f"expression {expression.id}: gather index {index} is out of bounds")
            return flat[index]
        return self._bounded_elementwise(
            load, operands[0], owner=f"expression {expression.id}"
        )


def _copy_integer_tensor(value: Any) -> tuple[int, ...]:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "reshape"):
        current = current.reshape(-1)
    if hasattr(current, "tolist"):
        current = current.tolist()
    if not isinstance(current, (list, tuple)):
        raise UnsupportedTritonAnalysis("data_dependent_index", "read-only gather argument cannot be copied to a host integer sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in current):
        raise UnsupportedTritonAnalysis("data_dependent_index", "read-only gather values must be integers")
    return tuple(current)


_TENSOR_DESCRIPTOR_FIELDS = ("base", "shape", "strides", "block_shape")
_TENSOR_DESCRIPTOR_FIELD_ALIASES = {
    # TTIR preserves Triton's C++/IR spelling while the Python runtime object
    # deliberately exposes the same field in snake case.
    "roundF32ToTF32": "round_f32_to_tf32",
}


def _is_tensor_descriptor(value: Any) -> bool:
    return all(hasattr(value, field) for field in _TENSOR_DESCRIPTOR_FIELDS)


def _follow_path(value: Any, path: Sequence[str | int]) -> Any:
    for part in path:
        if isinstance(part, int):
            value = value[part]
        elif isinstance(value, Mapping):
            value = value[part]
        else:
            runtime_part = (
                _TENSOR_DESCRIPTOR_FIELD_ALIASES.get(part, part)
                if _is_tensor_descriptor(value)
                else part
            )
            value = getattr(value, runtime_part)
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype if dtype is not None else "unknown")


def _element_bytes(value: Any, fallback: int | None = None) -> int:
    element_size = getattr(value, "element_size", None)
    if callable(element_size):
        result = int(element_size())
        if result > 0:
            return result
    dtype = getattr(value, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        result = int(itemsize)
        if result > 0:
            return result
    if fallback is not None and fallback > 0:
        return fallback
    raise UnsupportedTritonAnalysis("allocation_metadata", "cannot infer allocation element width")


def _shape(value: Any) -> tuple[int, ...]:
    try:
        result = tuple(int(extent) for extent in value.shape)
    except (AttributeError, TypeError) as error:
        raise UnsupportedTritonAnalysis("allocation_metadata", "pointer argument has no logical shape") from error
    if not result or any(extent <= 0 for extent in result):
        raise UnsupportedTritonAnalysis("allocation_metadata", f"unsupported logical shape {result}")
    return result


def _strides(value: Any, rank: int) -> tuple[int, ...]:
    stride = getattr(value, "stride", None)
    result = stride() if callable(stride) else getattr(value, "strides", None)
    if result is None:
        raise UnsupportedTritonAnalysis("allocation_metadata", "pointer argument has no logical strides")
    resolved = tuple(int(item) for item in result)
    if len(resolved) != rank:
        raise UnsupportedTritonAnalysis("allocation_metadata", "allocation shape/stride rank mismatch")
    return resolved


def _dense_status(shape: tuple[int, ...], strides: tuple[int, ...]) -> str:
    if any(stride < 0 for stride in strides):
        return "negative_stride"
    expected = 1
    order = []
    for dimension in sorted(range(len(shape)), key=lambda dim: (strides[dim], dim)):
        if shape[dimension] == 1:
            continue
        if strides[dimension] != expected:
            return "unsupported_view"
        expected *= shape[dimension]
        order.append(dimension)
    row_order = [
        dimension
        for dimension in reversed(range(len(shape)))
        if shape[dimension] != 1
    ]
    return "dense" if order == row_order else "permuted_dense"


def _data_pointer(value: Any) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if not callable(data_ptr):
        raise UnsupportedTritonAnalysis("allocation_metadata", "pointer argument has no data_ptr()")
    return int(data_ptr())


def _storage_identity(value: Any) -> tuple[str, int]:
    storage = getattr(value, "untyped_storage", None)
    if callable(storage):
        underlying = storage()
        data_ptr = getattr(underlying, "data_ptr", None)
        if callable(data_ptr):
            return ("storage", int(data_ptr()))
    base = getattr(value, "_base", None)
    if base is not None:
        return ("base", _data_pointer(base))
    return ("pointer", _data_pointer(value))


@dataclass(frozen=True)
class AllocationMetadata:
    name: str
    argument: int | str
    path: tuple[str | int, ...]
    base_pointer: int
    storage_identity: tuple[str, int]
    dtype: str
    element_bytes: int
    true_shape: tuple[int, ...]
    envelope_shape: tuple[int, ...]
    strides: tuple[int, ...]
    role: str
    alias_group: str
    dense_status: str
    eligible: bool

    def offset_to_coord(self, element_offset: int) -> tuple[int, ...]:
        if element_offset < 0:
            raise UnsupportedTritonAnalysis("negative_offset", f"{self.name}: active access has negative element offset {element_offset}")
        remaining = element_offset
        coord = [0] * len(self.true_shape)
        for dimension in sorted(range(len(coord)), key=lambda dim: self.strides[dim], reverse=True):
            if self.true_shape[dimension] == 1:
                continue
            stride = self.strides[dimension]
            if stride <= 0:
                raise UnsupportedTritonAnalysis("unsupported_view", f"{self.name}: cannot invert stride {stride}")
            coord[dimension], remaining = divmod(remaining, stride)
        if remaining or any(component >= extent for component, extent in zip(coord, self.true_shape)):
            raise UnsupportedTritonAnalysis("out_of_bounds", f"{self.name}: active element offset {element_offset} is outside logical shape {self.true_shape}")
        return tuple(coord)


def _argument_name(argument: int | str, arguments: Mapping[int | str, Any]) -> str:
    if isinstance(argument, str):
        return argument
    names = arguments.get("__names__", {})
    return str(names.get(argument, f"arg{argument}"))


def infer_allocations(manifest: AccessManifest, arguments: Mapping[int | str, Any]) -> tuple[AllocationMetadata, ...]:
    """Infer all global allocation metadata and one kernel-independent policy."""

    sites_by_base: dict[tuple[int | str, tuple[str | int, ...]], list[ManifestMemory]] = {}
    for node in _iter_nodes(manifest.body):
        if isinstance(node, ManifestMemory):
            sites_by_base.setdefault((node.base_arg, node.base_path), []).append(node)
    provisional = []
    for (argument, path), sites in sites_by_base.items():
        if argument not in arguments:
            raise UnsupportedTritonAnalysis("argument_binding", f"manifest base argument {argument!r} is unavailable")
        root_argument = arguments[argument]
        original = _follow_path(root_argument, path)
        descriptor_value = root_argument if _is_tensor_descriptor(root_argument) else original
        descriptor = _is_tensor_descriptor(descriptor_value)
        value = descriptor_value.base if descriptor else original
        shape = tuple(int(extent) for extent in descriptor_value.shape) if descriptor else _shape(value)
        strides = tuple(int(stride) for stride in descriptor_value.strides) if descriptor else _strides(value, len(shape))
        dense = _dense_status(shape, strides)
        operations = {site.operation for site in sites}
        role = "atomic" if "atomic" in operations else "read_write" if operations == {"load", "store"} else "read" if operations == {"load"} else "write"
        fallback_widths = {site.element_bytes for site in sites}
        if len(fallback_widths) != 1:
            raise UnsupportedTritonAnalysis("allocation_metadata", f"argument {argument!r} is accessed with inconsistent element widths")
        manifest_width = next(iter(fallback_widths))
        runtime_width = _element_bytes(value, manifest_width)
        if runtime_width != manifest_width:
            raise UnsupportedTritonAnalysis(
                "allocation_metadata",
                f"argument {argument!r}: runtime element width {runtime_width} B "
                f"does not match manifest width {manifest_width} B",
            )
        argument_name = _argument_name(argument, arguments)
        suffix = "" if not path else "." + ".".join(str(item) for item in path)
        provisional.append(
            {
                "name": argument_name + suffix,
                "argument": argument,
                "path": path,
                "base_pointer": _data_pointer(value),
                "storage_identity": _storage_identity(value),
                "dtype": _dtype_name(value),
                "element_bytes": runtime_width,
                "true_shape": shape,
                "envelope_shape": tuple(_next_power_of_two(extent) for extent in shape),
                "strides": strides,
                "role": role,
                "dense_status": dense,
            }
        )
    identities: dict[tuple[str, int], list[int]] = {}
    for index, allocation in enumerate(provisional):
        identities.setdefault(allocation["storage_identity"], []).append(index)
    alias_names = {
        identity: f"alias.{group_index}"
        for group_index, identity in enumerate(identities)
    }
    result = []
    for index, allocation in enumerate(provisional):
        alias_members = identities[allocation["storage_identity"]]
        alias_group = alias_names[allocation["storage_identity"]]
        eligible = (
            allocation["role"] == "read"
            and allocation["dense_status"] in {"dense", "permuted_dense"}
            and len(alias_members) == 1
        )
        result.append(AllocationMetadata(**allocation, alias_group=alias_group, eligible=eligible))
    return tuple(result)


def _readonly_launch_tensors(
    allocations: Sequence[AllocationMetadata],
    arguments: Mapping[int | str, Any],
) -> dict[int | str, Any]:
    writable_storage = {
        allocation.storage_identity
        for allocation in allocations
        if allocation.role != "read"
    }
    return {
        allocation.argument: _follow_path(
            arguments[allocation.argument], allocation.path
        )
        for allocation in allocations
        if allocation.role == "read"
        and allocation.storage_identity not in writable_storage
    }


@dataclass(frozen=True)
class EvaluationLimits:
    max_trace_contexts: int = 4096
    max_dynamic_events: int = 1 << 18
    max_loop_iterations: int = 1 << 16
    max_tensor_elements: int = 1 << 18


@dataclass
class _ConcreteSequence:
    events: list[MemoryEvent]
    pid: tuple[int, int, int]
    block: int
    wave: int
    multiplicity: int = 1


@dataclass
class _TraceState:
    manifest: AccessManifest
    arguments: Mapping[int | str, Any]
    allocations: Mapping[tuple[int | str, tuple[str | int, ...]], AllocationMetadata]
    readonly_tensors: Mapping[int | str, Any]
    limits: EvaluationLimits
    pid: tuple[int, int, int]
    block: int
    wave: int
    grid: tuple[int, int, int]
    events: list[MemoryEvent]
    variables: dict[str, RuntimeValue] = field(default_factory=dict)
    phase_counter: int = 0
    dynamic_events: int = 0

    def evaluator(self) -> ExpressionEvaluator:
        return ExpressionEvaluator(
            self.manifest,
            _EvaluationContext(
                self.arguments,
                self.pid,
                self.variables,
                self.readonly_tensors,
                self.grid,
            ),
            max_tensor_elements=self.limits.max_tensor_elements,
        )


def _scalar(value: RuntimeValue, owner: str) -> int | bool:
    if isinstance(value, TensorValue):
        raise UnsupportedTritonAnalysis("tensor_control", f"{owner} requires a scalar control value")
    return value


def _at(value: RuntimeValue, coord: tuple[int, ...]) -> int | bool:
    return value.at(coord) if isinstance(value, TensorValue) else value


def _register_slices(node: ManifestMemory, register_count: int) -> tuple[tuple[int, ...], ...]:
    raw = node.issue.get("register_slices")
    if raw is not None:
        slices = tuple(tuple(int(register) for register in item) for item in raw)
    else:
        width = node.issue.get(
            "register_slice_elements",
            node.issue.get("register_slice_size", node.issue.get("vector_size", 1)),
        )
        width = int(width)
        if width <= 0:
            raise ManifestError(f"memory site {node.site_id}: issue register width must be positive")
        slices = tuple(tuple(range(start, min(start + width, register_count))) for start in range(0, register_count, width))
    flattened = tuple(register for item in slices for register in item)
    if tuple(sorted(flattened)) != tuple(range(register_count)) or len(flattened) != len(set(flattened)):
        raise ManifestError(f"memory site {node.site_id}: register issue slices must partition [0, {register_count})")
    return slices


def _memory_events(node: ManifestMemory, state: _TraceState, structural_phase: str) -> list[MemoryEvent]:
    layout_record = state.manifest.layout_map[node.layout]
    layout = layout_record.layout
    supported_dimensions = {"register", "lane", "warp", "block"}
    unknown = set(layout.input_dims) - supported_dimensions
    if unknown:
        raise UnsupportedTritonAnalysis("layout_dimension", f"memory site {node.site_id}: unsupported LinearLayout input dimensions {sorted(unknown)}", site=node.site_id)
    sizes = {name: layout.input_size(name) for name in layout.input_dims}
    if state.wave >= sizes.get("warp", 1) or state.block >= sizes.get("block", 1):
        return []
    register_count = sizes.get("register", 1)
    lane_count = sizes.get("lane", 1)
    evaluator = state.evaluator()
    offset = None if node.offset is None else evaluator.evaluate(node.offset)
    descriptor_indices = tuple(evaluator.evaluate(index) for index in node.indices)
    mask = True if node.mask is None else evaluator.evaluate(node.mask)
    controls = tuple(
        (evaluator.evaluate(predicate), polarity)
        for predicate, polarity in node.control_predicates
    )
    if any(
        bool(_scalar(control, f"memory site {node.site_id} control predicate"))
        != polarity
        for control, polarity in controls
    ):
        return []
    allocation = state.allocations[(node.base_arg, node.base_path)]
    result = []
    for slice_index, registers in enumerate(_register_slices(node, register_count)):
        accesses = []
        for register in registers:
            for lane in range(lane_count):
                coordinates = {name: 0 for name in layout.input_dims}
                coordinates.update(register=register, lane=lane, warp=state.wave, block=state.block)
                coordinates = {name: coordinates[name] for name in layout.input_dims}
                if any(coordinates[name] & layout_record.free_mask(name) for name in layout.input_dims):
                    continue
                location = HardwareLocation.make(coordinates)
                tensor_coord = layout.apply(location)
                if node.shape and tensor_coord and tuple(node.shape) != layout.output_shape:
                    raise ManifestError(f"memory site {node.site_id}: operation shape {node.shape} disagrees with LinearLayout output {layout.output_shape}")
                if not bool(_at(mask, tensor_coord)):
                    continue
                root_argument = state.arguments[node.base_arg]
                runtime_descriptor = all(
                    hasattr(root_argument, field)
                    for field in ("base", "shape", "strides", "block_shape")
                )
                if runtime_descriptor and descriptor_indices:
                    block_shape = tuple(int(extent) for extent in root_argument.block_shape)
                    if tuple(node.shape) != block_shape:
                        raise UnsupportedTritonAnalysis(
                            "descriptor_metadata",
                            f"memory site {node.site_id}: compiled block shape "
                            f"{node.shape} does not match runtime descriptor "
                            f"block shape {block_shape}",
                            site=node.site_id,
                        )
                    if len(descriptor_indices) != len(allocation.true_shape):
                        raise UnsupportedTritonAnalysis(
                            "descriptor_metadata",
                            f"memory site {node.site_id}: descriptor index rank does not match allocation",
                            site=node.site_id,
                        )
                    starts = tuple(
                        int(_at(index, tensor_coord))
                        for index in descriptor_indices
                    )
                    if len(tensor_coord) != len(starts):
                        raise UnsupportedTritonAnalysis(
                            "descriptor_metadata",
                            f"memory site {node.site_id}: descriptor block rank does not match its indices",
                            site=node.site_id,
                        )
                    coord = tuple(
                        start + component
                        for start, component in zip(starts, tensor_coord)
                    )
                    if any(
                        component < 0 or component >= extent
                        for component, extent in zip(coord, allocation.true_shape)
                    ):
                        if node.operation != "load":
                            raise UnsupportedTritonAnalysis(
                                "descriptor_out_of_bounds_store",
                                f"memory site {node.site_id}: active descriptor "
                                f"{node.operation} coordinate {coord} is outside "
                                f"logical shape {allocation.true_shape}",
                                site=node.site_id,
                            )
                        padding = getattr(root_argument, "padding", None)
                        if padding not in {"zero", "nan"}:
                            raise UnsupportedTritonAnalysis(
                                "descriptor_boundary_semantics",
                                f"memory site {node.site_id}: out-of-bounds descriptor "
                                "load has no concrete zero/nan padding policy",
                                site=node.site_id,
                            )
                        continue
                else:
                    if offset is None:
                        raise ManifestError(
                            f"memory site {node.site_id}: direct operation has no element offset"
                        )
                    element_offset = int(_at(offset, tensor_coord))
                    offset_expression = state.manifest.expression_map[node.offset]
                    offset_width = _require_integer_width(offset_expression)
                    element_offset = _signed(element_offset, offset_width)
                    coord = allocation.offset_to_coord(element_offset)
                accesses.append(Access(allocation.name, coord, lane=lane, kind=node.operation, width_bytes=node.element_bytes))
        if not accesses:
            continue
        order = len(state.events) + len(result)
        metadata = {
            "provenance": "triton-access-manifest-v1",
            "parent_operation": node.site_id,
            "workgroup": ".".join(str(value) for value in state.pid),
            "wave": str(state.wave),
            "block": str(state.block),
            "step": str(order),
            "phase": f"{structural_phase}.sync{state.phase_counter}",
            "issue_slice": str(slice_index),
        }
        if node.source:
            metadata["source"] = node.source
        if node.cache is not None:
            metadata["cache"] = node.cache
        if node.eviction is not None:
            metadata["eviction"] = node.eviction
        result.append(MemoryEvent.make(f"pending.{order}", node.site_id, accesses, group=f"b{state.block}.w{state.wave}", order=order, metadata=metadata))
    return result


def _execute_nodes(nodes: Sequence[ManifestNode], state: _TraceState, path: tuple[str, ...] = ("root",)) -> None:
    for node_index, node in enumerate(nodes):
        region = path + (f"n{node.lexical_order if hasattr(node, 'lexical_order') else node_index}",)
        if isinstance(node, ManifestMemory):
            events = _memory_events(node, state, "/".join(path))
            state.dynamic_events += len(events)
            if state.dynamic_events > state.limits.max_dynamic_events:
                raise UnsupportedTritonAnalysis("enumeration_bound", f"dynamic event count exceeds exact bound {state.limits.max_dynamic_events}", site=node.site_id)
            state.events.extend(events)
        elif isinstance(node, ManifestBarrier):
            state.phase_counter += 1
        elif isinstance(node, ManifestIf):
            condition = bool(_scalar(state.evaluator().evaluate(node.condition), "structured if"))
            branch = node.then_body if condition else node.else_body
            _execute_nodes(branch, state, region + (("then" if condition else "else"),))
        elif isinstance(node, ManifestFor):
            evaluator = state.evaluator()
            lower = int(_scalar(evaluator.evaluate(node.lower), f"loop {node.iv} lower bound"))
            upper = int(_scalar(evaluator.evaluate(node.upper), f"loop {node.iv} upper bound"))
            step = int(_scalar(evaluator.evaluate(node.step), f"loop {node.iv} step"))
            if step == 0:
                raise UnsupportedTritonAnalysis("loop_control", f"loop {node.iv} has zero step")
            if step > 0:
                iterations = max(0, (upper - lower + step - 1) // step)
            else:
                magnitude = -step
                iterations = max(0, (lower - upper + magnitude - 1) // magnitude)
            if iterations > state.limits.max_loop_iterations:
                raise UnsupportedTritonAnalysis("enumeration_bound", f"loop {node.iv} has {iterations} iterations; limit is {state.limits.max_loop_iterations}")
            initial = state.evaluator()
            for name, init, _yielded in node.iter_args:
                state.variables[name] = initial.evaluate(init)
            for iteration, iv in enumerate(range(lower, upper, step)):
                state.variables[node.iv] = iv
                _execute_nodes(node.body, state, region + (f"iter{iteration}",))
                if node.iter_args:
                    yielded = state.evaluator()
                    updates = {name: yielded.evaluate(yield_expression) for name, _init, yield_expression in node.iter_args}
                    state.variables.update(updates)
            state.variables.pop(node.iv, None)
            for name, _init, _yielded in node.iter_args:
                state.variables.pop(name, None)
        else:
            raise AssertionError(type(node))


def _sequence_signature(
    sequence: _ConcreteSequence,
    matrices: Mapping[str, MatrixSpec],
    *,
    normalize_translations: bool,
) -> tuple[Any, ...]:
    anchors: dict[str, int] = {}
    for event in sequence.events:
        for access in event.accesses:
            anchors.setdefault(access.array, matrices[access.array].coord_to_bits(access.coord))
    signature = []
    for event in sequence.events:
        metadata = tuple((key, value) for key, value in event.metadata if key not in {"workgroup", "wave", "block", "step"})
        accesses = tuple(
            (
                access.array,
                access.lane,
                access.kind,
                access.width_bytes,
                (
                    matrices[access.array].coord_to_bits(access.coord)
                    ^ (anchors[access.array] if normalize_translations else 0)
                ),
            )
            for access in event.accesses
        )
        signature.append((event.site, event.group.split(".w", 1)[0], accesses, metadata))
    return tuple(signature)


def _compress_sequences(
    concrete: Sequence[_ConcreteSequence],
    matrices: Mapping[str, MatrixSpec],
    *,
    normalize_translations: bool,
) -> tuple[tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    groups: dict[tuple[Any, ...], tuple[_ConcreteSequence, int]] = {}
    for sequence in concrete:
        if not sequence.events:
            continue
        signature = _sequence_signature(
            sequence,
            matrices,
            normalize_translations=normalize_translations,
        )
        if signature in groups:
            representative, multiplicity = groups[signature]
            groups[signature] = (
                representative,
                multiplicity + sequence.multiplicity,
            )
        else:
            groups[signature] = (sequence, sequence.multiplicity)
    events = []
    sequences = []
    for class_index, (_signature, (concrete_sequence, multiplicity)) in enumerate(groups.items()):
        event_ids = []
        for event_index, event in enumerate(concrete_sequence.events):
            event_id = f"trace{class_index}.event{event_index}.{event.site}"
            events.append(replace(event, id=event_id, order=event_index))
            event_ids.append(event_id)
        sequences.append(
            EventSequence.make(
                f"triton.trace.{class_index}",
                event_ids,
                weight=multiplicity,
                metadata={
                    "provenance": "triton-access-manifest-v1",
                    "representative_workgroup": ".".join(str(value) for value in concrete_sequence.pid),
                    "representative_wave": concrete_sequence.wave,
                    "representative_block": concrete_sequence.block,
                    "trace_class": class_index,
                },
            )
        )
    if not events:
        raise UnsupportedTritonAnalysis("empty_trace", "the concrete launch has no active global-memory accesses")
    return tuple(events), tuple(sequences)


def _grid_points(grid: tuple[int, int, int]) -> Iterator[tuple[int, int, int]]:
    from itertools import product

    yield from product(range(grid[0]), range(grid[1]), range(grid[2]))


def _depends_on_program_id(
    expression_id: int,
    expressions: Mapping[int, ManifestExpression],
    active: set[int] | None = None,
) -> bool:
    active = set() if active is None else active
    if expression_id in active:
        raise ManifestError(f"expression DAG contains a cycle at {expression_id}")
    expression = expressions[expression_id]
    if _canonical_op(expression.op) in {"program_id", "programid"}:
        return True
    active.add(expression_id)
    try:
        return any(
            _depends_on_program_id(operand, expressions, active)
            for operand in expression.operands
        )
    finally:
        active.remove(expression_id)


def _proves_affine_program_translation(
    expression_id: int,
    expressions: Mapping[int, ManifestExpression],
) -> bool:
    expression = expressions[expression_id]
    op = _canonical_op(expression.op)
    dependent = _depends_on_program_id(expression_id, expressions)
    if not dependent:
        return True
    if op in {"program_id", "programid"}:
        return int(expression.attributes.get("axis", 0)) == 0
    if op in {"add", "addi", "sub", "subi"}:
        return all(
            _proves_affine_program_translation(operand, expressions)
            for operand in expression.operands
        )
    if op in {"mul", "muli"}:
        if len(expression.operands) != 2:
            return False
        left, right = expression.operands
        return (
            _depends_on_program_id(left, expressions)
            != _depends_on_program_id(right, expressions)
            and all(
                _proves_affine_program_translation(operand, expressions)
                for operand in expression.operands
            )
        )
    if op in {
        "splat",
        "broadcast",
        "expand_dims",
        "reshape",
        "transpose",
        "convert_layout",
    }:
        return len(expression.operands) == 1 and _proves_affine_program_translation(
            expression.operands[0], expressions
        )
    return False


def _value_items(value: RuntimeValue) -> tuple[int, ...]:
    if isinstance(value, TensorValue):
        return tuple(int(item) for item in value.values)
    return (int(value),)


def _zero_like(value: RuntimeValue) -> RuntimeValue:
    if isinstance(value, TensorValue):
        return TensorValue.full(value.shape, 0)
    return 0


def _reshape_runtime(value: RuntimeValue, shape: tuple[int, ...]) -> RuntimeValue:
    if not isinstance(value, TensorValue) or math.prod(shape) != len(value.values):
        raise UnsupportedTritonAnalysis(
            "expression_shape", "affine proof encountered an invalid reshape"
        )
    return TensorValue(shape, value.values)


def _transpose_runtime(
    value: RuntimeValue, permutation: tuple[int, ...]
) -> RuntimeValue:
    if not isinstance(value, TensorValue) or sorted(permutation) != list(
        range(len(value.shape))
    ):
        raise UnsupportedTritonAnalysis(
            "expression_shape", "affine proof encountered an invalid transpose"
        )
    shape = tuple(value.shape[dimension] for dimension in permutation)
    values = []
    for coord in _coords(shape):
        source_coord = [0] * len(shape)
        for output_dimension, source_dimension in enumerate(permutation):
            source_coord[source_dimension] = coord[output_dimension]
        values.append(value.at(source_coord))
    return TensorValue(shape, tuple(values))


def _affine_pid_terms(
    expression_id: int,
    expressions: Mapping[int, ManifestExpression],
    evaluator: ExpressionEvaluator,
    last_pid: int,
    cache: dict[int, tuple[RuntimeValue, RuntimeValue] | None],
) -> tuple[RuntimeValue, RuntimeValue] | None:
    """Return unbounded ``base + pid(0) * coefficient`` and prove no wrap.

    Every program-dependent intermediate is checked at both endpoints against
    its declared MLIR integer width. Since the accepted expression is affine,
    those endpoint checks prove that modular integer evaluation agrees with the
    mathematical translation throughout the launch.
    """

    if expression_id in cache:
        return cache[expression_id]
    expression = expressions[expression_id]
    if not _depends_on_program_id(expression_id, expressions):
        value = evaluator.evaluate(expression_id)
        result = (value, _zero_like(value))
        cache[expression_id] = result
        return result

    op = _canonical_op(expression.op)
    result: tuple[RuntimeValue, RuntimeValue] | None = None
    if op in {"program_id", "programid"}:
        if int(expression.attributes.get("axis", 0)) == 0:
            result = (0, 1)
    elif op in {"add", "addi", "sub", "subi"} and len(expression.operands) == 2:
        left = _affine_pid_terms(
            expression.operands[0], expressions, evaluator, last_pid, cache
        )
        right = _affine_pid_terms(
            expression.operands[1], expressions, evaluator, last_pid, cache
        )
        if left is not None and right is not None:
            operation = (
                (lambda first, second: int(first) + int(second))
                if op in {"add", "addi"}
                else (lambda first, second: int(first) - int(second))
            )
            result = (
                evaluator._bounded_elementwise(
                    operation,
                    left[0],
                    right[0],
                    owner=f"affine expression {expression.id}",
                ),
                evaluator._bounded_elementwise(
                    operation,
                    left[1],
                    right[1],
                    owner=f"affine expression {expression.id}",
                ),
            )
    elif op in {"mul", "muli"} and len(expression.operands) == 2:
        left_id, right_id = expression.operands
        left_dependent = _depends_on_program_id(left_id, expressions)
        right_dependent = _depends_on_program_id(right_id, expressions)
        if left_dependent != right_dependent:
            dependent_id = left_id if left_dependent else right_id
            independent_id = right_id if left_dependent else left_id
            dependent = _affine_pid_terms(
                dependent_id, expressions, evaluator, last_pid, cache
            )
            if dependent is not None:
                independent = evaluator.evaluate(independent_id)
                multiply = lambda first, second: int(first) * int(second)
                result = (
                    evaluator._bounded_elementwise(
                        multiply,
                        dependent[0],
                        independent,
                        owner=f"affine expression {expression.id}",
                    ),
                    evaluator._bounded_elementwise(
                        multiply,
                        dependent[1],
                        independent,
                        owner=f"affine expression {expression.id}",
                    ),
                )
    elif op in {
        "splat",
        "broadcast",
        "expand_dims",
        "reshape",
        "transpose",
        "convert_layout",
    } and len(expression.operands) == 1:
        operand = _affine_pid_terms(
            expression.operands[0], expressions, evaluator, last_pid, cache
        )
        if operand is not None:
            attributes = expression.attributes
            if op in {"splat", "broadcast"}:
                shape = tuple(int(extent) for extent in attributes.get("shape", ()))
                if shape:
                    result = (
                        evaluator._bounded_broadcast(
                            operand[0],
                            shape,
                            f"affine expression {expression.id}",
                        ),
                        evaluator._bounded_broadcast(
                            operand[1],
                            shape,
                            f"affine expression {expression.id}",
                        ),
                    )
            elif op == "expand_dims":
                if isinstance(operand[0], TensorValue) and isinstance(
                    operand[1], TensorValue
                ):
                    axis = int(attributes.get("axis"))
                    if axis < 0:
                        axis += len(operand[0].shape) + 1
                    shape = operand[0].shape[:axis] + (1,) + operand[0].shape[axis:]
                    result = (
                        TensorValue(shape, operand[0].values),
                        TensorValue(shape, operand[1].values),
                    )
            elif op == "reshape":
                shape = tuple(int(extent) for extent in attributes.get("shape", ()))
                if shape:
                    result = (
                        _reshape_runtime(operand[0], shape),
                        _reshape_runtime(operand[1], shape),
                    )
            elif op == "transpose":
                permutation = tuple(
                    int(item)
                    for item in attributes.get(
                        "order", attributes.get("permutation", ())
                    )
                )
                result = (
                    _transpose_runtime(operand[0], permutation),
                    _transpose_runtime(operand[1], permutation),
                )
            else:
                result = operand

    if result is not None:
        width = _optional_integer_width(expression)
        if width is None:
            result = None
        else:
            maximum = 1 << width
            base_items = _value_items(result[0])
            coefficient_items = _value_items(result[1])
            if len(base_items) != len(coefficient_items) or any(
                endpoint < 0 or endpoint >= maximum
                for base, coefficient in zip(base_items, coefficient_items)
                for endpoint in (base, base + last_pid * coefficient)
            ):
                result = None
    cache[expression_id] = result
    return result


def _ceil_div(upper: int, lower: int) -> int:
    return -((-upper) // lower)


def _unwrap_predicate_expression(
    expression_id: int,
    expressions: Mapping[int, ManifestExpression],
) -> int:
    """Peel only logical-value-preserving unary tensor wrappers."""

    current = expression_id
    while True:
        expression = expressions[current]
        if (
            _canonical_op(expression.op)
            not in {"convert_layout", "reshape", "broadcast"}
            or len(expression.operands) != 1
        ):
            return current
        current = expression.operands[0]


def _translation_launch_classes(
    manifest: AccessManifest,
    arguments: Mapping[int | str, Any],
    allocations: Sequence[AllocationMetadata],
    grid: tuple[int, int, int],
    limits: EvaluationLimits,
) -> tuple[tuple[tuple[int, int, int], int], ...] | None:
    """Prove and compress the common one-dimensional aligned-tile case.

    The proof accepts a flat structured body whose offsets are affine in
    ``program_id(0)`` with one positive power-of-two stride. Canonical upper
    bound masks may create a full-tile interval, one boundary interval, and an
    empty interval. Each returned representative therefore stands for an exact
    range of program IDs, not a sampled range.
    """

    if grid[1:] != (1, 1):
        return None
    nodes = tuple(manifest.body)
    if any(not isinstance(node, (ManifestMemory, ManifestBarrier)) for node in nodes):
        return None
    memories = tuple(node for node in nodes if isinstance(node, ManifestMemory))
    if not memories or any(len(allocation.true_shape) != 1 for allocation in allocations):
        return None
    expressions = manifest.expression_map
    if any(
        node.control_predicates
        or node.offset is None
        or node.indices
        for node in memories
    ):
        return None
    if any(not _proves_affine_program_translation(node.offset, expressions) for node in memories):
        return None
    readonly = _readonly_launch_tensors(allocations, arguments)
    contexts = (
        _EvaluationContext(arguments, (0, 0, 0), {}, readonly, grid),
        _EvaluationContext(arguments, (1, 0, 0), {}, readonly, grid),
    )
    evaluators = tuple(
        ExpressionEvaluator(manifest, context, max_tensor_elements=limits.max_tensor_elements)
        for context in contexts
    )
    breakpoints = {0, grid[0]}
    stride_by_base: dict[tuple[int | str, tuple[str | int, ...]], int] = {}
    offsets_by_base: dict[tuple[int | str, tuple[str | int, ...]], list[int]] = {}
    for node in memories:
        if (
            _affine_pid_terms(
                node.offset,
                expressions,
                evaluators[0],
                grid[0] - 1,
                {},
            )
            is None
        ):
            return None
        first = _value_items(evaluators[0].evaluate(node.offset))
        second = _value_items(evaluators[1].evaluate(node.offset))
        if len(first) != len(second):
            return None
        differences = {right - left for left, right in zip(first, second)}
        if len(differences) != 1:
            return None
        stride = next(iter(differences))
        if stride <= 0 or stride & (stride - 1):
            return None
        if max(first) - min(first) >= stride or min(first) < 0:
            return None
        base_key = (node.base_arg, node.base_path)
        existing = stride_by_base.setdefault(base_key, stride)
        if existing != stride:
            return None
        offsets_by_base.setdefault(base_key, []).extend(first)
        if node.mask is None:
            continue
        mask_expression = expressions[
            _unwrap_predicate_expression(node.mask, expressions)
        ]
        if _canonical_op(mask_expression.op) != "cmp" or len(mask_expression.operands) != 2:
            return None
        predicate = str(mask_expression.attributes.get("predicate", mask_expression.attributes.get("pred", ""))).lower()
        if predicate not in {"slt", "ult", "lt", "signed_less_than", "unsigned_less_than"}:
            return None
        address, limit_expression = mask_expression.operands
        if _depends_on_program_id(limit_expression, expressions):
            return None
        if (
            not _proves_affine_program_translation(address, expressions)
            or _affine_pid_terms(
                address,
                expressions,
                evaluators[0],
                grid[0] - 1,
                {},
            )
            is None
        ):
            return None
        address_first = _value_items(evaluators[0].evaluate(address))
        address_second = _value_items(evaluators[1].evaluate(address))
        if address_first != first or address_second != second:
            return None
        limit = evaluators[0].evaluate(limit_expression)
        if isinstance(limit, TensorValue):
            if len(set(limit.values)) != 1:
                return None
            limit = limit.values[0]
        full_end = (int(limit) - 1 - max(first)) // stride
        empty_start = _ceil_div(int(limit) - min(first), stride)
        breakpoints.update(
            max(0, min(grid[0], point))
            for point in (full_end + 1, empty_start)
        )
    for base_key, offsets in offsets_by_base.items():
        stride = stride_by_base[base_key]
        if len({offset // stride for offset in offsets}) != 1:
            return None
    ordered = sorted(breakpoints)
    classes = []
    for start, end in zip(ordered, ordered[1:]):
        if start < end:
            classes.append(((start, 0, 0), end - start))

    # A representative can stand for an interval only if its active register
    # set is constant and every translated address remains within the concrete
    # allocation.  The accepted offsets have one positive affine coefficient,
    # so checking both interval endpoints proves the address bound throughout.
    allocation_map = {
        (allocation.argument, allocation.path): allocation
        for allocation in allocations
    }
    for (start, _y, _z), multiplicity in classes:
        endpoint_active: list[tuple[tuple[int, ...], ...]] = []
        for pid in (start, start + multiplicity - 1):
            evaluator = ExpressionEvaluator(
                manifest,
                _EvaluationContext(arguments, (pid, 0, 0), {}, readonly, grid),
                max_tensor_elements=limits.max_tensor_elements,
            )
            active_by_site: list[tuple[int, ...]] = []
            for node in memories:
                assert node.offset is not None
                offset = evaluator.evaluate(node.offset)
                mask = True if node.mask is None else evaluator.evaluate(node.mask)
                layout_shape = manifest.layout_map[node.layout].layout.output_shape
                offset_width = _require_integer_width(expressions[node.offset])
                active = tuple(
                    _signed(int(_at(offset, coord)), offset_width)
                    for coord in _coords(layout_shape)
                    if bool(_at(mask, coord))
                )
                allocation = allocation_map[(node.base_arg, node.base_path)]
                for element_offset in active:
                    allocation.offset_to_coord(element_offset)
                active_by_site.append(active)
            endpoint_active.append(tuple(active_by_site))
        first, last = endpoint_active
        if any(
            len(first_site) != len(last_site)
            for first_site, last_site in zip(first, last)
        ):
            return None
    return tuple(classes)


def evaluate_manifest(
    manifest: AccessManifest,
    arguments: Mapping[int | str, Any],
    grid: Sequence[int],
    *,
    limits: EvaluationLimits = EvaluationLimits(),
    preserve_resource_anchors: bool = False,
) -> tuple[tuple[AllocationMetadata, ...], tuple[MatrixSpec, ...], tuple[MemoryEvent, ...], tuple[EventSequence, ...]]:
    """Evaluate one launch exactly, or raise a categorized unsupported result."""

    if manifest.status == "unsupported":
        diagnostic = manifest.diagnostics[0] if manifest.diagnostics else {}
        raise UnsupportedTritonAnalysis(str(diagnostic.get("category", "compiler_unsupported")), str(diagnostic.get("message", "compiler analysis rejected this kernel")), site=diagnostic.get("site"))
    normalized_grid = tuple(int(extent) for extent in grid)
    if not 1 <= len(normalized_grid) <= 3 or any(extent <= 0 for extent in normalized_grid):
        raise UnsupportedTritonAnalysis("launch_grid", f"invalid Triton launch grid {normalized_grid}")
    normalized_grid += (1,) * (3 - len(normalized_grid))
    allocations = infer_allocations(manifest, arguments)
    matrices = tuple(
        MatrixSpec(
            allocation.name,
            allocation.envelope_shape,
            allocation.element_bytes,
            tuple(f"dim{dimension}" for dimension in range(len(allocation.true_shape))),
            target=allocation.eligible,
            role=allocation.role,
        )
        for allocation in allocations
    )
    if not any(matrix.target for matrix in matrices):
        raise UnsupportedTritonAnalysis("eligibility", "the launch has no unaliased dense read-only layout candidate")
    matrix_map = {matrix.name: matrix for matrix in matrices}
    allocation_map = {(allocation.argument, allocation.path): allocation for allocation in allocations}
    max_blocks = max((layout.layout.input_size("block") if "block" in layout.layout.input_dims else 1) for layout in manifest.layouts)
    max_waves = max((layout.layout.input_size("warp") if "warp" in layout.layout.input_dims else 1) for layout in manifest.layouts)
    if max_blocks != 1:
        raise UnsupportedTritonAnalysis(
            "clustered_cta",
            "the first exact frontend subset requires a singleton LinearLayout block dimension",
        )
    context_count = math.prod(normalized_grid) * max_blocks * max_waves
    launch_classes = None
    if context_count > limits.max_trace_contexts:
        if preserve_resource_anchors:
            raise UnsupportedTritonAnalysis(
                "resource_trace_bound",
                f"launch needs {context_count} exact workgroup/wave contexts; "
                f"limit is {limits.max_trace_contexts}, and hardware resource "
                "color phases cannot be discarded by translation compression",
            )
        launch_classes = _translation_launch_classes(
            manifest, arguments, allocations, normalized_grid, limits
        )
        if launch_classes is None:
            raise UnsupportedTritonAnalysis("enumeration_bound", f"launch needs {context_count} exact workgroup/wave contexts; limit is {limits.max_trace_contexts}, and aligned-translation exactness was not proved")
        if len(launch_classes) * max_blocks * max_waves > limits.max_trace_contexts:
            raise UnsupportedTritonAnalysis("enumeration_bound", "aligned-translation trace classes still exceed the exact context bound")
    readonly = _readonly_launch_tensors(allocations, arguments)
    concrete = []
    total_events = 0
    pid_classes = (
        tuple((pid, 1) for pid in _grid_points(normalized_grid))
        if launch_classes is None
        else launch_classes
    )
    for pid, multiplicity in pid_classes:
        for block in range(max_blocks):
            for wave in range(max_waves):
                state = _TraceState(
                    manifest,
                    arguments,
                    allocation_map,
                    readonly,
                    limits,
                    pid,
                    block,
                    wave,
                    normalized_grid,
                    [],
                )
                _execute_nodes(manifest.body, state)
                total_events += len(state.events)
                if total_events > limits.max_dynamic_events:
                    raise UnsupportedTritonAnalysis("enumeration_bound", f"launch dynamic event count exceeds exact bound {limits.max_dynamic_events}")
                concrete.append(
                    _ConcreteSequence(
                        state.events,
                        pid,
                        block,
                        wave,
                        multiplicity,
                    )
                )
    events, sequences = _compress_sequences(
        concrete,
        matrix_map,
        normalize_translations=not preserve_resource_anchors,
    )
    return allocations, matrices, events, sequences


@dataclass(frozen=True)
class AnalysisOptions:
    plugin_path: str | os.PathLike[str] | None = None
    hardware_profile: HardwareProfile | None = None
    limits: EvaluationLimits = field(default_factory=EvaluationLimits)


@dataclass(frozen=True)
class TritonLaunchAnalysis:
    supported: bool
    compiled_kernel: Any = None
    manifest: AccessManifest | None = None
    grid: tuple[int, ...] = ()
    bound_arguments: Mapping[int | str, Any] = field(default_factory=dict)
    selected_config: Mapping[str, Any] = field(default_factory=dict)
    allocations: tuple[AllocationMetadata, ...] = ()
    matrices: tuple[MatrixSpec, ...] = ()
    events: tuple[MemoryEvent, ...] = ()
    sequences: tuple[EventSequence, ...] = ()
    edge_families: tuple[Any, ...] = ()
    components: tuple[Any, ...] = ()
    hardware_profile: HardwareProfile | None = None
    resource_anchors_preserved: bool = False
    unsupported: UnsupportedReason | None = None

    def require_supported(self) -> "TritonLaunchAnalysis":
        if not self.supported:
            assert self.unsupported is not None
            raise UnsupportedTritonAnalysis(self.unsupported.category, self.unsupported.message, site=self.unsupported.site)
        return self

    def relay_problem(
        self,
        *,
        hardware_profile: HardwareProfile | None = None,
        grammar: str = "standard",
        config: Any = None,
        byte_scales: Sequence[int] | None = None,
    ) -> Any:
        self.require_supported()
        from .access_scopes import UniversalScopeObjectives
        from .simple_solver import SimpleRelayProblem
        from .solver import RelayProblem, SolverConfig

        profile = hardware_profile or self.hardware_profile
        scales = tuple(byte_scales or (() if profile is None else profile.byte_scales))
        if not scales:
            raise ValueError("relay_problem requires a hardware profile or explicit byte_scales")
        objectives = (UniversalScopeObjectives(scales),)
        if profile is not None:
            if profile.resource_maps and not self.resource_anchors_preserved:
                raise ValueError(
                    "this trace was compressed without hardware resource-color "
                    "anchors; pass the profile in AnalysisOptions when analyzing "
                    "the launch"
                )
            if config is not None:
                raise ValueError(
                    "the exact hardware-profile adapter uses SimpleRelayProblem; "
                    "pass grammar/frontier settings instead of SolverConfig"
                )
            return SimpleRelayProblem(
                matrices=self.matrices,
                events=self.events,
                sequences=self.sequences,
                objectives=objectives,
                grammar=grammar,
                hardware_profile=profile,
                fine_component=profile.fine_component,
                name="triton_automatic_hypergraph",
            )
        return RelayProblem(
            self.matrices,
            self.events,
            self.sequences,
            objectives,
            config or SolverConfig(),
            "triton_automatic_hypergraph",
        )


def _unwrap_jit(kernel: Any) -> Any:
    current = kernel
    seen = set()
    while not hasattr(current, "arg_names") or not hasattr(current, "params"):
        if id(current) in seen or not hasattr(current, "fn"):
            raise UnsupportedTritonAnalysis("launch_wrapper", f"cannot find a Triton JITFunction under {type(kernel).__name__}")
        seen.add(id(current))
        current = current.fn
    return current


def _resolved_launch_arguments(kernel: Any, args: Sequence[Any], kwargs: Mapping[str, Any]) -> tuple[dict[int | str, Any], dict[str, Any]]:
    jit = _unwrap_jit(kernel)
    names = tuple(jit.arg_names)
    parameters = {parameter.name: parameter for parameter in jit.params}
    named = dict(zip(names, args))
    named.update({key: value for key, value in kwargs.items() if key in names})
    current = kernel
    seen = set()
    selected: dict[str, Any] = {}
    while current is not jit:
        if id(current) in seen:
            raise UnsupportedTritonAnalysis("launch_wrapper", "cyclic Triton launch wrapper chain")
        seen.add(id(current))
        class_name = type(current).__name__
        if class_name == "Autotuner":
            config = getattr(current, "best_config", None)
            if config is None:
                raise UnsupportedTritonAnalysis("autotune_config", "the concrete autotune configuration was not selected by the launch")
            config_values = dict(config.all_kwargs())
            named.update(config_values)
            selected.update(config_values)
        elif class_name == "Heuristics":
            for name, heuristic in current.values.items():
                value = heuristic({**named, **kwargs})
                named[name] = value
                selected[name] = value
        current = current.fn
    missing = [
        name
        for name in names
        if name not in named and not parameters[name].has_default
    ]
    if missing:
        raise UnsupportedTritonAnalysis("argument_binding", f"missing concrete Triton arguments: {missing}")
    for parameter in jit.params:
        if parameter.name not in named and parameter.has_default:
            named[parameter.name] = parameter.default
    bound: dict[int | str, Any] = {"__names__": {index: name for index, name in enumerate(names)}}
    for index, name in enumerate(names):
        bound[index] = named[name]
        bound[name] = named[name]
    return bound, selected


def _bound_from_named(jit: Any, named: Mapping[str, Any]) -> dict[int | str, Any]:
    names = tuple(jit.arg_names)
    resolved = dict(named)
    for parameter in jit.params:
        if parameter.name not in resolved and parameter.has_default:
            resolved[parameter.name] = parameter.default
    missing = [name for name in names if name not in resolved]
    if missing:
        raise UnsupportedTritonAnalysis(
            "argument_binding",
            f"concrete JIT launch did not bind arguments {missing}",
        )
    bound: dict[int | str, Any] = {
        "__names__": {index: name for index, name in enumerate(names)}
    }
    for index, name in enumerate(names):
        bound[index] = resolved[name]
        bound[name] = resolved[name]
    return bound


def _selected_from_capture(
    kernel: Any,
    jit: Any,
    captured: Mapping[str, Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    provided = set(tuple(jit.arg_names)[: len(args)]) | (
        set(kwargs) & set(jit.arg_names)
    )
    selected = {
        name: value
        for name, value in captured.items()
        if name in jit.arg_names and name not in provided
    }
    current = kernel
    seen = set()
    while current is not jit and id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "best_config", None)
        if config is not None and hasattr(config, "all_kwargs"):
            selected.update(config.all_kwargs())
        current = current.fn
    return selected


def _normalize_grid(grid: Any, named_arguments: Mapping[int | str, Any]) -> tuple[int, ...]:
    if callable(grid):
        names = named_arguments.get("__names__", {})
        grid = grid({name: named_arguments[index] for index, name in names.items()})
    if isinstance(grid, int):
        grid = (grid,)
    try:
        result = tuple(int(extent) for extent in grid)
    except TypeError as error:
        raise UnsupportedTritonAnalysis("launch_grid", f"invalid launch grid {grid!r}") from error
    if not 1 <= len(result) <= 3 or any(extent <= 0 for extent in result):
        raise UnsupportedTritonAnalysis("launch_grid", f"invalid launch grid {result}")
    return result


def _default_plugin_path() -> Path | None:
    configured = os.environ.get("LAQS_TRITON_PLUGIN_PATH")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[1]
    active_candidate = None
    try:
        import triton

        module_file = getattr(triton, "__file__", None)
        if module_file is not None:
            active_candidate = (
                Path(module_file).resolve().parent
                / "plugins"
                / "libLAQSTritonAccessManifest.so"
            )
    except ImportError:
        pass
    if active_candidate is not None:
        return active_candidate if active_candidate.is_file() else None
    candidates = tuple(path for path in (
        root / "triton" / "triton-lang" / "python" / "triton" / "plugins" / "libLAQSTritonAccessManifest.so",
        root / "triton" / "plugins" / "tuolumne" / "libLAQSTritonAccessManifest.so",
        root / "triton" / "plugins" / "matrix" / "libLAQSTritonAccessManifest.so",
        root / "triton" / "access-manifest" / "build" / "libLaqsTritonPlugin.so",
        root / "triton" / "access-manifest" / "build" / "libTritonLAQS.so",
    ) if path is not None)
    return next((path for path in candidates if path.is_file()), None)


_LOADED_PLUGIN_PATHS: set[Path] = set()


@contextmanager
def _manifest_compilation(plugin_path: str | os.PathLike[str] | None) -> Iterator[None]:
    try:
        from triton import knobs
        from triton._C.libtriton import passes
    except (ImportError, AttributeError) as error:
        raise UnsupportedTritonAnalysis("plugin_unavailable", "the pinned Triton checkout with the LAQS post-coalesce hook is unavailable") from error
    if not hasattr(knobs.runtime, "post_coalesce_hook"):
        raise UnsupportedTritonAnalysis("plugin_unavailable", "pinned Triton was built without the disabled-by-default post-coalesce analysis hook")
    path = Path(plugin_path) if plugin_path is not None else _default_plugin_path()
    if path is None or not path.is_file():
        raise UnsupportedTritonAnalysis("plugin_unavailable", "LAQS Triton access-manifest plugin is not built; set LAQS_TRITON_PLUGIN_PATH")
    resolved_path = path.resolve()
    if resolved_path not in _LOADED_PLUGIN_PATHS:
        try:
            passes.plugin.extend_with(str(resolved_path))
        except Exception as error:
            raise UnsupportedTritonAnalysis("plugin_unavailable", f"failed to load Triton pass plugin {resolved_path}: {error}") from error
        _LOADED_PLUGIN_PATHS.add(resolved_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def hook(pm: Any = None) -> str:
        if pm is None:
            return f"laqs-access-manifest-v1:{digest}"
        passes.plugin.add_laqs_access_manifest(pm)
        return f"laqs-access-manifest-v1:{digest}"

    def collect_metadata(module: Any, metadata: dict[str, Any]) -> None:
        payload = module.get_str_attr("laqs.access_manifest")
        if payload is None:
            raise RuntimeError("LAQS access-manifest pass did not set laqs.access_manifest")
        metadata[MANIFEST_METADATA_KEY] = payload

    hook.collect_metadata = collect_metadata

    with knobs.runtime.scope():
        knobs.runtime.post_coalesce_hook = hook
        yield


def analyze_compiled_manifest(
    compiled_kernel: Any,
    manifest_payload: str | bytes | Mapping[str, Any],
    grid: Sequence[int],
    bound_arguments: Mapping[int | str, Any],
    *,
    selected_config: Mapping[str, Any] | None = None,
    options: AnalysisOptions = AnalysisOptions(),
) -> TritonLaunchAnalysis:
    """CPU-testable half of :func:`analyze_launch` after concrete compilation."""

    try:
        manifest = parse_access_manifest(manifest_payload)
        allocations, matrices, events, sequences = evaluate_manifest(
            manifest,
            bound_arguments,
            grid,
            limits=options.limits,
            preserve_resource_anchors=bool(
                options.hardware_profile is not None
                and options.hardware_profile.resource_maps
            ),
        )
        matrix_map = {matrix.name: matrix for matrix in matrices}
        event_map = {event.id: event for event in events}
        families = build_edge_families(matrix_map, event_map, sequences)
        components = ()
        if options.hardware_profile is not None:
            components = materialize_edge_families(families, matrix_map, options.hardware_profile.byte_scales)
            options.hardware_profile.component_weights(components)
        return TritonLaunchAnalysis(
            supported=True,
            compiled_kernel=compiled_kernel,
            manifest=manifest,
            grid=tuple(int(extent) for extent in grid),
            bound_arguments=MappingProxyType(dict(bound_arguments)),
            selected_config=MappingProxyType(dict(selected_config or {})),
            allocations=allocations,
            matrices=matrices,
            events=events,
            sequences=sequences,
            edge_families=families,
            components=components,
            hardware_profile=options.hardware_profile,
            resource_anchors_preserved=bool(
                options.hardware_profile is not None
                and options.hardware_profile.resource_maps
            ),
        )
    except (
        ManifestError,
        UnsupportedTritonAnalysis,
        LookupError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, UnsupportedTritonAnalysis):
            reason = UnsupportedReason(error.category, str(error), error.site)
        elif isinstance(error, ManifestError):
            reason = UnsupportedReason("invalid_manifest", str(error))
        else:
            reason = UnsupportedReason("trace_construction", str(error))
        return TritonLaunchAnalysis(False, compiled_kernel=compiled_kernel, grid=tuple(int(extent) for extent in grid), bound_arguments=MappingProxyType(dict(bound_arguments)), selected_config=MappingProxyType(dict(selected_config or {})), unsupported=reason)


def analyze_launch(kernel: Any, grid: Any, *args: Any, _laqs_options: AnalysisOptions | None = None, **kwargs: Any) -> TritonLaunchAnalysis:
    """Launch an unmodified Triton kernel once and build its automatic LAQS graph.

    The ordinary launch selects heuristics and autotune configuration exactly as
    Triton normally does. The returned ``CompiledKernel`` is the one used for
    that launch; the kernel is not launched a second time for analysis.
    """

    options = _laqs_options or AnalysisOptions()
    compiled = None
    bound: Mapping[int | str, Any] = {}
    selected: Mapping[str, Any] = {}
    resolved_grid: tuple[int, ...] = ()
    try:
        jit = _unwrap_jit(kernel)
        captured_named: dict[str, Any] = {}
        captured_grid: list[tuple[int, ...]] = []

        def capture_arguments(*launch_args: Any, **launch_kwargs: Any) -> None:
            captured_named.update(dict(zip(jit.arg_names, launch_args)))
            captured_named.update(
                {
                    name: launch_kwargs[name]
                    for name in jit.arg_names
                    if name in launch_kwargs
                }
            )

        effective_grid = grid
        if callable(grid):
            def capture_grid(meta: Mapping[str, Any]) -> Any:
                value = grid(meta)
                normalized = (value,) if isinstance(value, int) else tuple(value)
                captured_grid.append(tuple(int(extent) for extent in normalized))
                return value

            effective_grid = capture_grid

        hooks = getattr(jit, "pre_run_hooks", None)
        if isinstance(hooks, list):
            hooks.append(capture_arguments)
        try:
            with _manifest_compilation(options.plugin_path):
                compiled = kernel.run(
                    *args,
                    grid=effective_grid,
                    warmup=False,
                    **kwargs,
                )
        finally:
            if isinstance(hooks, list) and capture_arguments in hooks:
                hooks.remove(capture_arguments)
        if compiled is None:
            raise UnsupportedTritonAnalysis("compilation", "Triton launch hook suppressed compilation")
        if captured_named:
            bound = _bound_from_named(jit, captured_named)
            selected = _selected_from_capture(
                kernel, jit, captured_named, args, kwargs
            )
        else:
            bound, selected = _resolved_launch_arguments(kernel, args, kwargs)
        resolved_grid = (
            captured_grid[-1]
            if captured_grid
            else _normalize_grid(grid, bound)
        )
        metadata = getattr(compiled, "metadata", None)
        payload = (
            metadata.get(MANIFEST_METADATA_KEY)
            if isinstance(metadata, Mapping)
            else getattr(metadata, MANIFEST_METADATA_KEY, None)
        )
        if payload is None:
            raise UnsupportedTritonAnalysis("manifest_missing", f"CompiledKernel.metadata has no {MANIFEST_METADATA_KEY!r}")
        return analyze_compiled_manifest(compiled, payload, resolved_grid, bound, selected_config=selected, options=options)
    except UnsupportedTritonAnalysis as error:
        return TritonLaunchAnalysis(False, compiled_kernel=compiled, grid=resolved_grid, bound_arguments=MappingProxyType(dict(bound)), selected_config=MappingProxyType(dict(selected)), unsupported=UnsupportedReason(error.category, str(error), error.site))
