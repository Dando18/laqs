"""Compile and launch Triton kernels over packed LAQS persistent layouts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from math import prod
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class RuntimeLayout:
    name: str
    argument: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    envelope_shape: tuple[int, ...]
    rows: tuple[int, ...]

    def pass_argument(self) -> str:
        fields = (
            str(self.argument),
            ",".join(map(str, self.shape)),
            ",".join(map(str, self.strides)),
            ",".join(map(str, self.rows)),
        )
        return "|".join(fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argument": self.argument,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "envelope_shape": list(self.envelope_shape),
            "rows": list(self.rows),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeLayout":
        return cls(
            str(value["name"]),
            int(value["argument"]),
            tuple(map(int, value["shape"])),
            tuple(map(int, value["strides"])),
            tuple(map(int, value["envelope_shape"])),
            tuple(map(int, value["rows"])),
        )


def _plugin_path() -> Path:
    configured = os.environ.get("LAQS_TRITON_LAYOUT_PLUGIN_PATH")
    if configured:
        return Path(configured).resolve()
    package_paths = tuple(Path(path) for path in getattr(triton, "__path__", ()))
    candidates = tuple(
        path / "plugins" / "libLAQSTritonLayoutRewrite.so"
        for path in package_paths
    )
    repository = Path(__file__).resolve().parents[2]
    candidates += (
        repository / "triton" / "triton-lang" / "python" / "triton"
        / "plugins" / "libLAQSTritonLayoutRewrite.so",
        repository / "triton" / "plugins" / "matrix"
        / "libLAQSTritonLayoutRewrite.so",
    )
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            "libLAQSTritonLayoutRewrite.so is not installed; run the platform "
            "setup command documented in triton/experiments/README.md"
        )
    return found.resolve()


_LOADED: set[Path] = set()


@contextmanager
def rewrite_layouts(layouts: Sequence[RuntimeLayout]) -> Iterator[None]:
    """Select the post-coalescing address rewrite for one compilation/cache key."""

    if not layouts:
        yield
        return
    from triton import knobs
    from triton._C.libtriton import passes

    path = _plugin_path()
    if path not in _LOADED:
        passes.plugin.extend_with(str(path))
        _LOADED.add(path)
    arguments = [layout.pass_argument() for layout in layouts]
    digest = hashlib.sha256(path.read_bytes() + "\0".join(arguments).encode()).hexdigest()

    def hook(pm=None):
        key = f"laqs-layout-rewrite-v1:{digest}"
        if pm is not None:
            passes.plugin.add_laqs_layout_rewrite(pm, arguments)
        return key

    with knobs.runtime.scope():
        knobs.runtime.post_coalesce_hook = hook
        yield


@triton.jit
def _pack_kernel(source, target, logical_size: tl.constexpr,
                 shape: tl.constexpr, strides: tl.constexpr,
                 mode_shifts: tl.constexpr, rows: tl.constexpr,
                 row_count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < logical_size
    logical = tl.zeros((BLOCK,), dtype=tl.int64)
    for dimension in tl.static_range(0, len(shape)):
        coordinate = (offsets // strides[dimension]) % shape[dimension]
        logical |= coordinate.to(tl.int64) << mode_shifts[dimension]
    physical = tl.zeros((BLOCK,), dtype=tl.int64)
    for physical_bit in tl.static_range(0, row_count):
        parity = logical & rows[physical_bit]
        parity ^= parity >> 32
        parity ^= parity >> 16
        parity ^= parity >> 8
        parity ^= parity >> 4
        parity ^= parity >> 2
        parity ^= parity >> 1
        physical |= (parity & 1) << physical_bit
    value = tl.load(source + offsets, mask=mask)
    tl.store(target + physical, value, mask=mask)


def pack_tensor(source: torch.Tensor, layout: RuntimeLayout) -> torch.Tensor:
    """Pack one contiguous logical tensor into its power-of-two envelope."""

    if tuple(source.shape) != layout.shape:
        raise ValueError(
            f"{layout.name}: runtime shape {tuple(source.shape)} != {layout.shape}"
        )
    if tuple(source.stride()) != layout.strides or not source.is_contiguous():
        raise ValueError(f"{layout.name}: only ordinary dense inputs are realizable")
    target = torch.empty(prod(layout.envelope_shape), device=source.device,
                         dtype=source.dtype)
    shifts = []
    shift = 0
    for extent in layout.envelope_shape:
        shifts.append(shift)
        shift += extent.bit_length() - 1
    if shift != len(layout.rows):
        raise ValueError(f"{layout.name}: envelope and address matrix disagree")
    block = 256
    _pack_kernel[(triton.cdiv(source.numel(), block),)](
        source.reshape(-1), target, source.numel(),
        shape=layout.shape, strides=layout.strides,
        mode_shifts=tuple(shifts), rows=layout.rows,
        row_count=len(layout.rows), BLOCK=block,
    )
    return target


def unwrap_jit(kernel: Any) -> Any:
    current = kernel
    seen = set()
    while not hasattr(current, "arg_names") or not hasattr(current, "params"):
        if id(current) in seen or not hasattr(current, "fn"):
            raise TypeError(f"cannot locate a Triton JITFunction below {type(kernel).__name__}")
        seen.add(id(current))
        current = current.fn
    return current


@dataclass
class FrozenLaunch:
    jit: Any
    grid: tuple[int, ...]
    values: list[Any]
    options: dict[str, Any]

    def clone(self) -> "FrozenLaunch":
        return FrozenLaunch(self.jit, self.grid, list(self.values), dict(self.options))

    def run(self):
        return self.jit.run(*self.values, grid=self.grid, warmup=False, **self.options)


def freeze_launch(spec, selected_config: Mapping[str, Any]) -> FrozenLaunch:
    """Bypass an autotuner using the configuration selected by ordinary Triton."""

    jit = unwrap_jit(spec.kernel)
    names = tuple(jit.arg_names)
    named = dict(zip(names, spec.args))
    named.update({key: value for key, value in spec.kwargs.items() if key in names})
    named.update({key: value for key, value in selected_config.items() if key in names})
    for parameter in jit.params:
        if parameter.name not in named and parameter.has_default:
            named[parameter.name] = parameter.default
    missing = [name for name in names if name not in named]
    if missing:
        raise ValueError(f"frozen launch is missing JIT arguments: {missing}")
    options = {
        key: value
        for key, value in {**spec.kwargs, **selected_config}.items()
        if key not in names and value is not None
    }
    grid = spec.grid(named) if callable(spec.grid) else spec.grid
    if isinstance(grid, int):
        grid = (grid,)
    return FrozenLaunch(jit, tuple(map(int, grid)), [named[name] for name in names], options)


def replace_inputs(launch: FrozenLaunch, layouts: Sequence[RuntimeLayout]) -> FrozenLaunch:
    result = launch.clone()
    for layout in layouts:
        source = result.values[layout.argument]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"{layout.name}: argument {layout.argument} is not a tensor")
        result.values[layout.argument] = pack_tensor(source, layout)
    return result


def fresh_outputs(launch: FrozenLaunch, output_arguments: Sequence[int]) -> FrozenLaunch:
    result = launch.clone()
    for argument in output_arguments:
        value = result.values[argument]
        if isinstance(value, torch.Tensor):
            result.values[argument] = torch.empty_like(value)
    return result
