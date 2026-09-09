"""Reproducibility, portable capture records, and independent-process statistics."""
from __future__ import annotations

import copyreg
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
import subprocess
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "relay.search.compiler-contract.v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity():
    """Hash working sources, including uncommitted changes, not just Git HEAD."""
    paths = sorted(set(ROOT.joinpath("relay").glob("**/*.py")) |
                   set(ROOT.joinpath("triton/experiments").glob("*.py")) |
                   set(ROOT.joinpath("triton").glob("*.py")) |
                   set(ROOT.joinpath("triton/automatic_frontend").glob("*.cpp")) |
                   set(ROOT.joinpath("triton/layout_rewrite").glob("*.cpp")))
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for checkout in ("triton/triton-lang", "triton/tritonbench"):
        files[checkout + ":revision"] = subprocess.check_output(
            ["git", "-C", str(ROOT / checkout), "rev-parse", "HEAD"], text=True).strip()
        patch = subprocess.check_output(["git", "-C", str(ROOT / checkout), "diff", "HEAD"])
        files[checkout + ":patch"] = hashlib.sha256(patch).hexdigest()
    return {"protocol": PROTOCOL, "source_hash": digest(files), "sources": files}


def compiler_source_identity(package_file):
    root = Path(package_file).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
    return {"path": str(root), "revision": revision, "changes_sha256": hashlib.sha256(changes).hexdigest()}


def proposal_key(proposal):
    """Bind resumable measurements to semantics, excluding elapsed setup time."""
    fields = ("schema", "experiment", "platform", "operator", "config", "tau_name",
              "source_identity", "capture_identity", "hardware_profile", "tau_profile_hash",
              "selected_config", "output_arguments", "runtime_layouts", "baseline_layouts", "kernel_name")
    identity = {name: proposal[name] for name in fields}
    identity["selection_policy"] = {name: proposal["search"].get(name)
                                    for name in ("algorithm", "minimum_score_gain", "search_scope")}
    return digest(identity)


@dataclass(frozen=True)
class HostTensor:
    """Only tensor metadata and integer index data needed by the CPU frontend."""
    shape: tuple
    strides: tuple
    dtype: str
    width: int
    pointer: int
    storage_pointer: int
    data: tuple | None = None

    def stride(self):
        return self.strides

    def element_size(self):
        return self.width

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return replace(self, pointer=self.storage_pointer)

    def detach(self):
        return self

    def cpu(self):
        return self

    def flatten(self):
        return self

    def tolist(self):
        if self.data is None:
            raise ValueError("capture has no integer index payload for this allocation")
        return list(self.data)


def host_arguments(arguments):
    import torch

    result = {}
    tensors = {}
    for key, value in arguments.items():
        if isinstance(value, torch.Tensor):
            if id(value) not in tensors:
                data = None
                # Indirection tensors in this panel are small integer vectors;
                # large quantized weights need metadata, never a host copy.
                if not value.is_floating_point() and value.numel() <= (1 << 20):
                    data = tuple(value.detach().cpu().flatten().tolist())
                tensors[id(value)] = HostTensor(tuple(value.shape), tuple(value.stride()),
                    str(value.dtype), value.element_size(), value.data_ptr(),
                    value.untyped_storage().data_ptr(), data)
            value = tensors[id(value)]
        result[key] = value
    return result


def _reduce_proxy(value):
    return dict, (dict(value),)


copyreg.pickle(type(MappingProxyType({})), _reduce_proxy)


def save_graph(path, value):
    """Write an internal, trusted cache atomically; return its content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(pickle.dumps(value, protocol=5), compresslevel=1, mtime=0)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)
    return hashlib.sha256(data).hexdigest()


def load_graph(path, expected_hash):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"cache content hash mismatch: {path}")
    return pickle.loads(gzip.decompress(data))


def process_summary(records):
    """Paired ratios with processes as the replication unit for uncertainty."""
    result = {}
    for label in ("baseline", "selected", "identity"):
        values = [v for r in records for v in r["timings"][label]["samples_ms"]]
        medians = [r["timings"][label]["median_ms"] for r in records]
        result[label] = {"median_ms": statistics.median(medians),
                         "mean_ms": statistics.fmean(values), "min_ms": min(values),
                         "process_medians_ms": medians, "samples_ms": values}
    ratios = [r["timings"]["baseline"]["median_ms"] / r["timings"]["selected"]["median_ms"] for r in records]
    identity_ratios = [r["timings"]["baseline"]["median_ms"] / r["timings"]["identity"]["median_ms"] for r in records]
    result["speedup"] = math.exp(statistics.fmean(map(math.log, ratios)))
    result["process_speedups"] = ratios
    result["identity_speedups"] = identity_ratios
    result["confidence_unit"] = "independent timing process; log-ratio t interval"
    # Student t critical values avoid treating 21 correlated samples as 21 runs.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if len(ratios) > 1:
        logs = list(map(math.log, ratios))
        margin = critical.get(len(logs) - 1, 2.228) * statistics.stdev(logs) / math.sqrt(len(logs))
        center = statistics.fmean(logs)
        result["speedup_ci95"] = [math.exp(center - margin), math.exp(center + margin)]
    else:
        result["speedup_ci95"] = None
    result["identity_max_deviation"] = max(abs(r - 1) for r in identity_ratios)
    result["processes"] = records
    return result
