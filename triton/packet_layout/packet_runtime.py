"""Structured packet compilation and narrow memory-primitive validation."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

from layout_contract import RuntimeLayout, compiler_statistics

_LOADED = set()


def plugin_path():
    import triton
    path = Path(triton.__file__).resolve().parent / 'plugins/libLAQSTritonPacketLayout.so'
    if not path.is_file():
        raise FileNotFoundError(f"build the platform packet plugin first: {path}")
    return path


@contextmanager
def structured_layouts(layouts, *, inspect=False, mode='repaired', launch=None):
    """Compile a frozen launch; scalar/grid bindings are proof assumptions.

    A caller supplying ``launch`` must run that launch unchanged in this scope.
    Different scalar arguments or grids require a new scope/compilation key.
    Without bindings, recurrence proofs use only facts present in the IR.
    """
    from triton import knobs
    from triton._C.libtriton import passes
    from layout_runtime import launch_guard
    path = plugin_path()
    if path not in _LOADED:
        passes.plugin.extend_with(str(path))
        _LOADED.add(path)
    if mode not in ('legacy', 'transparent', 'repaired'):
        raise ValueError(f'unknown packet address mode: {mode}')
    bindings = {} if launch is None else {
        name: int(value) for name, value in zip(launch.jit.arg_names, launch.values)
        if isinstance(value, (int, bool)) and -(1 << 31) <= value < (1 << 31)}
    if launch is not None:
        bindings.update({f'$grid{i}': int(n) for i, n in enumerate(launch.grid)})
    expected_grid = tuple(launch.grid) if launch is not None else ()
    expected_jit = launch.jit if launch is not None else None
    def guard(actual):
        if launch is None:
            return
        if actual.jit is not expected_jit or tuple(actual.grid) != expected_grid:
            raise ValueError('launch does not match the packet recurrence proof')
        values = dict(zip(actual.jit.arg_names, actual.values))
        if any(values[name] != value for name, value in bindings.items() if not name.startswith('$')):
            raise ValueError('scalar arguments changed since the packet recurrence proof')
    arguments = (["inspect"] if inspect else []) + [mode] + [
        f'bind:{name}:{value}' for name, value in sorted(bindings.items())] + [layout.pass_argument() for layout in layouts]
    signature = hashlib.sha256(path.read_bytes() + json.dumps(arguments).encode()).hexdigest()

    def hook(pm=None):
        if pm is not None:
            passes.plugin.add_laqs_packet_layout(pm, arguments)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_licm(pm)
            passes.common.add_canonicalizer(pm)
        return f"laqs-packet-v1:{signature}"

    def collect(module, metadata):
        metadata['laqs_packet_contracts'] = module.get_str_attr('laqs.packet_contract_json') or '[]'

    hook.collect_metadata = collect
    with knobs.runtime.scope():
        knobs.runtime.post_coalesce_hook = hook
        token = launch_guard.set(guard)
        try:
            yield
        finally:
            launch_guard.reset(token)


def statistics(kernel, directory, label):
    record = compiler_statistics(kernel, directory, label)
    record['packet_sites'] = json.loads(getattr(kernel.metadata, 'laqs_packet_contracts', '[]'))
    # Ignore dependence-barrier pseudo-memory names and half-precision FMA suffixes.
    record['memory_primitives'] = {k: v for k, v in record['memory_instruction_widths'].items()
                                   if not k.startswith('LDGDEPBAR')}
    record['matrix_primitives'] = {k: v for k, v in record['matrix_instruction_forms'].items()
                                   if not k.startswith('HFMA2')}
    record['communication'] = {k: v for k, v in record['opcode_counts'].items()
                               if re.search(r'shfl|ds_bpermute|ds_swizzle', k, re.I)}
    record['opaque_identity_calls'] = len(re.findall(
        r'\basm(?:\s+\w+)*\s+""\s*,\s*"=[rv],0"', kernel.asm.get('llir', '')))
    recurrence = re.search(r'laqs.physical_recurrences = (\d+)', kernel.asm.get('ttgir', ''))
    record['physical_recurrences'] = int(recurrence[1]) if recurrence else 0
    return record


def primitive_rejections(baseline, candidate):
    """Hard semantic/primitive checks; registers and static cost remain diagnostics."""
    reasons = []
    if not baseline.get('final_assembly') or not candidate.get('final_assembly'):
        reasons.append('final device assembly unavailable')
    for field in ['memory_primitives', 'matrix_primitives']:
        if field not in baseline or field not in candidate or set(baseline[field]) != set(candidate[field]):
            reasons.append(f'{field} changed')
    if any(count > baseline.get('communication', {}).get(op, 0)
           for op, count in candidate.get('communication', {}).items()):
        reasons.append('possible additional cross-lane communication')
    if not candidate.get('memory_primitives'):
        reasons.append('no final memory primitive evidence')
    if not baseline.get('packet_sites') or baseline.get('packet_sites') != candidate.get('packet_sites'):
        reasons.append('native per-site packet/ownership/mask contract changed or missing')
    for field in ['n_spills', 'shared_bytes']:
        if field not in baseline or field not in candidate:
            reasons.append(f'missing {field}')
        elif candidate[field] > baseline[field]:
            reasons.append(f'{field} increased')
    return reasons


def runtime_layouts(candidate):
    return tuple(RuntimeLayout.from_dict(value) for value in candidate['runtime_layouts'])
