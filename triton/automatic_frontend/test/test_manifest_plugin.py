import json
from pathlib import Path
import unittest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from relay.triton_frontend import _manifest_compilation, parse_access_manifest


PLUGIN = (
    Path(__file__).parents[1]
    / "triton-lang/python/triton/plugins/libLAQSTritonAccessManifest.so"
)
TARGET = GPUTarget("hip", "gfx942", 64)
HOPPER = GPUTarget("cuda", 90, 32)


@triton.jit
def vector_copy(source, output, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < size
    tl.store(output + offsets, tl.load(source + offsets, mask=mask), mask=mask)


@triton.jit
def scalar_copy(source, output):
    tl.store(output, tl.load(source))


@triton.jit
def scalar_sum(source, output, count):
    accumulator = 0.0
    for index in range(0, count):
        accumulator += tl.load(source + index)
    tl.store(output, accumulator)


@triton.jit
def pointer_walk(source, output, count):
    pointer = source
    accumulator = 0.0
    for _ in range(0, count):
        accumulator += tl.load(pointer)
        pointer += 1
    tl.store(output, accumulator)


@triton.jit
def selected_pointer_copy(source, output, block: tl.constexpr):
    offsets = tl.arange(0, block)
    negative_offsets = offsets - block
    pointers = tl.where(
        (offsets & 1) == 0,
        source + negative_offsets,
        source + offsets,
    )
    tl.store(output + offsets, tl.load(pointers))


@triton.jit
def ambiguous_pointer_copy(left, right, output, block: tl.constexpr):
    offsets = tl.arange(0, block)
    pointers = tl.where(
        (offsets & 1) == 0,
        left + offsets,
        right + offsets,
    )
    tl.store(output + offsets, tl.load(pointers))


@triton.jit
def transformed_tile_copy(source, output):
    linear = tl.arange(0, 64)
    tile = tl.reshape(linear, (8, 8))
    transposed = tl.trans(tile)
    rows = tl.expand_dims(tl.arange(0, 8), 1)
    columns = tl.expand_dims(tl.arange(0, 8), 0)
    offsets = rows * 8 + columns
    value = tl.load(source + transposed) + tl.load(source + offsets)
    tl.store(output + offsets, value)


@triton.jit
def scalar_branch_copy(source, output, select_second):
    if select_second != 0:
        tl.store(output + 1, tl.load(source + 1))
    else:
        tl.store(output, tl.load(source))


@triton.jit
def atomic_increment(output, block: tl.constexpr):
    offsets = tl.arange(0, block)
    tl.atomic_add(output + offsets, 1)


@triton.jit
def descriptor_copy(source, output):
    value = source.load([0, 0])
    output.store([0, 0], value)


@triton.jit
def one_level_gather(indices, source, output, block: tl.constexpr):
    offsets = tl.arange(0, block)
    selected = tl.load(indices + offsets)
    tl.store(output + offsets, tl.load(source + selected))


@triton.jit
def opaque_inline_assembly():
    tl.inline_asm_elementwise(
        "mov.u32 $0, %smid;", "=r", [], tl.int32, is_pure=False, pack=1
    )


def compile_manifest(function, signature, constants=None, target=TARGET):
    if not PLUGIN.is_file():
        raise unittest.SkipTest("LAQS Triton access-manifest plugin is not built")
    source = ASTSource(function, signature, constants or {})
    with _manifest_compilation(PLUGIN):
        compiled = triton.compile(source, target=target)
    payload = compiled.metadata.laqs_access_manifest
    return json.loads(payload), parse_access_manifest(payload)


def test_direct_memory_manifest_is_deterministic():
    signature = {
        "source": "*fp32",
        "output": "*fp32",
        "size": "constexpr",
        "block": "constexpr",
    }
    constants = {"size": 127, "block": 128}
    raw, manifest = compile_manifest(vector_copy, signature, constants)
    second, _ = compile_manifest(vector_copy, signature, constants)

    assert raw == second
    assert manifest.status == "supported"
    assert [argument.name for argument in manifest.arguments] == ["source", "output"]
    sites = [node for node in manifest.body if hasattr(node, "site_id")]
    assert [site.operation for site in sites] == ["load", "store"]
    assert all(site.shape == (128,) for site in sites)
    assert all(site.mask is not None for site in sites)
    assert all(site.base_path == () for site in sites)
    expressions = {expression["id"]: expression for expression in raw["expressions"]}
    offsets = [expressions[node["offset"]] for node in raw["body"]]
    assert all(expression["op"] == "add" for expression in offsets)
    assert all(expression["attributes"]["integer_width"] == 32 for expression in offsets)


def test_cuda_post_coalesce_hook_direct_memory():
    _, manifest = compile_manifest(
        vector_copy,
        {
            "source": "*fp32",
            "output": "*fp32",
            "size": "constexpr",
            "block": "constexpr",
        },
        {"size": 127, "block": 128},
        target=HOPPER,
    )

    assert manifest.status == "supported"
    sites = [node for node in manifest.body if hasattr(node, "site_id")]
    assert [site.operation for site in sites] == ["load", "store"]


def test_scalar_memory_uses_replicated_owner_election():
    raw, manifest = compile_manifest(
        scalar_copy, {"source": "*fp32", "output": "*fp32"}
    )

    assert manifest.status == "supported"
    scalar_layout = next(layout for layout in raw["layouts"] if layout["origin"] == "scalar_semantics")
    sizes = {item["name"]: item["size"] for item in scalar_layout["input_dims"]}
    assert sizes == {"register": 1, "lane": 64, "warp": 4, "block": 1}
    assert scalar_layout["free_variable_masks"] == {
        "register": 0,
        "lane": 63,
        "warp": 3,
        "block": 0,
    }
    sites = [node for node in manifest.body if hasattr(node, "site_id")]
    assert all(site.shape == (1,) for site in sites)


def test_loop_manifest_slices_unrelated_float_state():
    _, manifest = compile_manifest(
        scalar_sum, {"source": "*fp32", "output": "*fp32", "count": "i32"}
    )

    assert manifest.status == "supported"
    loop = next(node for node in manifest.body if hasattr(node, "iv"))
    assert loop.iter_args == ()
    sites = [node for node in loop.body if hasattr(node, "site_id")]
    assert [site.operation for site in sites] == ["load"]


def test_loop_carried_pointer_is_an_offset_variable():
    raw, manifest = compile_manifest(
        pointer_walk,
        {"source": "*fp32", "output": "*fp32", "count": "i32"},
    )

    assert manifest.status == "supported"
    loop = next(node for node in manifest.body if hasattr(node, "iv"))
    assert len(loop.iter_args) == 1
    name, _initial, _yielded = loop.iter_args[0]
    assert name.startswith("loop.0.iter")
    extensions = [
        expression
        for expression in raw["expressions"]
        if expression["op"] == "sext"
    ]
    assert len(extensions) == 1
    assert extensions[0]["attributes"]["integer_width"] == 64


def test_pointer_select_sign_extends_branch_offsets():
    raw, manifest = compile_manifest(
        selected_pointer_copy,
        {
            "source": "*fp32",
            "output": "*fp32",
            "block": "constexpr",
        },
        {"block": 64},
    )

    assert manifest.status == "supported"
    expressions = {expression["id"]: expression for expression in raw["expressions"]}
    load = next(node for node in raw["body"] if node.get("op") == "load")
    selected = expressions[load["offset"]]
    assert selected["op"] == "select"
    assert selected["attributes"]["integer_width"] == 64
    branches = [expressions[operand] for operand in selected["operands"][1:]]
    assert [branch["op"] for branch in branches] == ["sext", "sext"]
    assert all(branch["attributes"]["integer_width"] == 64 for branch in branches)


def test_ambiguous_pointer_provenance_is_diagnostic():
    raw, manifest = compile_manifest(
        ambiguous_pointer_copy,
        {
            "left": "*fp32",
            "right": "*fp32",
            "output": "*fp32",
            "block": "constexpr",
        },
        {"block": 64},
    )

    assert manifest.status == "unsupported"
    categories = {diagnostic["category"] for diagnostic in raw["diagnostics"]}
    assert "unsupported.ambiguous_pointer_provenance" in categories


def test_tensor_shape_operations_are_serialized():
    raw, manifest = compile_manifest(
        transformed_tile_copy,
        {"source": "*fp32", "output": "*fp32"},
    )

    assert manifest.status == "supported"
    operations = {expression["op"] for expression in raw["expressions"]}
    assert "reshape" in operations
    assert "transpose" in operations
    assert {"expand_dims", "broadcast"} & operations
    sites = [node for node in manifest.body if hasattr(node, "site_id")]
    assert all(site.shape == (8, 8) for site in sites)


def test_scalar_branch_is_a_structured_access_program():
    raw, manifest = compile_manifest(
        scalar_branch_copy,
        {"source": "*fp32", "output": "*fp32", "select_second": "i32"},
    )

    assert manifest.status == "supported"
    branch = next(node for node in raw["body"] if node["kind"] == "if")
    assert [node["op"] for node in branch["then"]] == ["load", "store"]
    assert [node["op"] for node in branch["else"]] == ["load", "store"]
    assert all(node["control_predicates"] for node in branch["then"])
    assert all(node["control_predicates"] for node in branch["else"])


def test_atomic_manifest_site():
    _, manifest = compile_manifest(
        atomic_increment,
        {"output": "*i32", "block": "constexpr"},
        {"block": 64},
    )

    site = next(node for node in manifest.body if hasattr(node, "site_id"))
    assert manifest.status == "supported"
    assert site.operation == "atomic"


def test_descriptor_load_store_manifest():
    _, manifest = compile_manifest(
        descriptor_copy,
        {
            "source": "tensordesc<fp32[16,16]>",
            "output": "tensordesc<fp32[16,16]>",
        },
        target=HOPPER,
    )

    sites = [node for node in manifest.body if hasattr(node, "site_id")]
    assert manifest.status == "supported"
    assert [site.operation for site in sites] == ["load", "store"]
    assert [site.base_arg for site in sites] == ["source", "output"]
    assert all(site.shape == (16, 16) for site in sites)


def test_one_level_integer_gather_expression():
    raw, manifest = compile_manifest(
        one_level_gather,
        {
            "indices": "*i32",
            "source": "*fp32",
            "output": "*fp32",
            "block": "constexpr",
        },
        {"block": 64},
    )

    assert manifest.status == "supported"
    gather = next(expression for expression in raw["expressions"] if expression["op"] == "gather")
    assert gather["attributes"]["arg"] == "indices"
    assert len(gather["operands"]) == 1


def test_opaque_memory_diagnostic():
    raw, manifest = compile_manifest(opaque_inline_assembly, {}, target=HOPPER)

    assert manifest.status == "unsupported"
    assert raw["diagnostics"][0]["category"] == "unsupported.opaque_memory_operation"


class ManifestPluginTest(unittest.TestCase):
    def test_direct_memory_manifest_is_deterministic(self):
        test_direct_memory_manifest_is_deterministic()

    def test_cuda_post_coalesce_hook_direct_memory(self):
        test_cuda_post_coalesce_hook_direct_memory()

    def test_scalar_memory_uses_replicated_owner_election(self):
        test_scalar_memory_uses_replicated_owner_election()

    def test_loop_manifest_slices_unrelated_float_state(self):
        test_loop_manifest_slices_unrelated_float_state()

    def test_loop_carried_pointer_is_an_offset_variable(self):
        test_loop_carried_pointer_is_an_offset_variable()

    def test_pointer_select_sign_extends_branch_offsets(self):
        test_pointer_select_sign_extends_branch_offsets()

    def test_ambiguous_pointer_provenance_is_diagnostic(self):
        test_ambiguous_pointer_provenance_is_diagnostic()

    def test_tensor_shape_operations_are_serialized(self):
        test_tensor_shape_operations_are_serialized()

    def test_scalar_branch_is_a_structured_access_program(self):
        test_scalar_branch_is_a_structured_access_program()

    def test_atomic_manifest_site(self):
        test_atomic_manifest_site()

    def test_descriptor_load_store_manifest(self):
        test_descriptor_load_store_manifest()

    def test_one_level_integer_gather_expression(self):
        test_one_level_integer_gather_expression()

    def test_opaque_memory_diagnostic(self):
        test_opaque_memory_diagnostic()


if __name__ == "__main__":
    unittest.main()
