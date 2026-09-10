"""Evidence-bound layout constraints, separate from the locality objective."""
from dataclasses import replace
import hashlib
import json
import re


def native_contracts(manifest, asm, gpu):
    """Protect H100 source sectors only at identified native 16-byte cg copies.

    PTX source locations connect final instructions to manifest sites. This is a
    compatibility restriction, not a model of lane grouping or transaction count.
    """
    if isinstance(manifest, (str, bytes)):
        manifest = json.loads(manifest)
    ptx = asm.get('ptx', '')
    if 'H100' not in gpu or not re.search(r'\.target\s+sm_90a?\b', ptx):
        return []
    files = {int(i): path for i, path in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', ptx)}
    locations, location = set(), None
    for line in ptx.splitlines():
        match = re.search(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if match:
            index, row, column = map(int, match.groups())
            location = (files.get(index), row, column)
        if re.search(r'cp\.async\.cg\.shared\.global\s+[^;]*,\s*(?:0x10|16)\s*[,;]', line):
            if location is None or location[0] is None:
                raise ValueError('native H100 async copy has no source location')
            locations.add(location)
    sites = []
    def visit(value):
        if isinstance(value, dict):
            if value.get('op') == 'load' and isinstance(value.get('source'), dict):
                source = value['source']
                key = (source.get('file'), source.get('line'), source.get('column'))
                if key in locations:
                    element_bytes = int(value['element_bytes'])
                    if element_bytes != 2:
                        raise ValueError('H100 sector contract currently supports FP16-sized elements only')
                    sites.append({'site': value['site_id'], 'source': source,
                                  'protected_element_bits': 4, 'source_sector_bytes': 32,
                                  'primitive': 'cp.async.cg.shared.global.16B',
                                  'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest()})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    matched = {(s['source']['file'], s['source']['line'], s['source']['column']) for s in sites}
    if locations - matched:
        raise ValueError('native H100 async copy could not be matched to a manifest load')
    return sites


def constrain_graph(analysis, contracts):
    """Attach per-site constraints without regrouping events or changing scores."""
    if not contracts:
        return analysis
    by_site = {c['site']: c for c in contracts}
    seen = set()
    events = []
    for event in analysis.events:
        contract = by_site.get(event.meta('parent_operation', event.site))
        if contract:
            seen.add(contract['site'])
            metadata = dict(event.metadata)
            metadata.update(protected_element_bits=str(contract['protected_element_bits']),
                            compatibility_primitive=contract['primitive'], source_sector_bytes='32')
            event = replace(event, metadata=tuple(sorted(metadata.items())))
        events.append(event)
    if set(by_site) - seen:
        raise ValueError('native source-sector contract has no corresponding graph event')
    return replace(analysis, events=tuple(events)) if contracts else analysis


def protected_bits(events):
    result = {}
    for event in events:
        bits = max((int(event.meta('vector_elements', '1')) - 1).bit_length(),
                   int(event.meta('protected_element_bits', '0')))
        for access in event.accesses:
            result[access.array] = max(result.get(access.array, 0), bits)
    return result
