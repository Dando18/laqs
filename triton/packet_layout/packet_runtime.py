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
def structured_layouts(layouts, *, inspect=False):
    from triton import knobs
    from triton._C.libtriton import passes
    path = plugin_path()
    if path not in _LOADED:
        passes.plugin.extend_with(str(path))
        _LOADED.add(path)
    arguments = (["inspect"] if inspect else []) + [layout.pass_argument() for layout in layouts]
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
        yield


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
