"""Bounded, ordered CPU work using clean spawned processes."""
import copyreg
from contextlib import ExitStack
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import gzip
import hashlib
import io
from itertools import islice
import multiprocessing
import os
from pathlib import Path
import pickle
from types import MappingProxyType


def _proxy_dict(value):
    return dict, (dict(value),)


def _pack(value):
    stream = io.BytesIO()
    pickler = pickle.Pickler(stream, protocol=5)
    pickler.dispatch_table = {**copyreg.dispatch_table, type(MappingProxyType({})): _proxy_dict}
    pickler.dump(value)
    return gzip.compress(stream.getvalue(), compresslevel=1, mtime=0)


def _unpack(value):
    return pickle.loads(gzip.decompress(value))


def _initialize(function, context):
    global _function, _context
    _function, _context = function, _unpack(context)


def _run(item):
    item, path = item
    result = _pack(_function(_context, item))
    if path is not None:
        _atomic_write(path, result)
        return None
    return result


def _atomic_write(path, data):
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def _fingerprint(value):
    """Hash semantic fields, excluding cached properties and hash-seed ordering."""
    result = hashlib.sha256()
    result.update(type(value).__qualname__.encode())
    if is_dataclass(value) and not isinstance(value, type):
        children = [(field.name, getattr(value, field.name)) for field in fields(value)]
    elif isinstance(value, Mapping):
        children = sorted(value.items(), key=lambda pair: _fingerprint(pair[0]))
    elif isinstance(value, (set, frozenset)):
        children = sorted(_fingerprint(item) for item in value)
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        result.update(pickle.dumps(value, protocol=5))
        return result.digest()
    for child in children:
        result.update(_fingerprint(child))
    return result.digest()


def ordered_parallel_map(function, context, items, workers, *, checkpoint_dir=None):
    """Keep at most one result per worker in flight and preserve input order.

    Spawn avoids inheriting an initialized GPU runtime. Large read-only inputs
    are serialized once, and compressed outputs bound IPC overhead. Exiting on
    failure terminates the pool instead of waiting for unrelated queued work.
    Optional checkpoints commit each item in its worker before ordered delivery.
    The caller must exclusively own the directory and version it with its source
    identity; context checks additionally reject changed inputs.
    """
    if workers < 1:
        raise ValueError("CPU worker count must be positive")
    if workers == 1 and checkpoint_dir is None:
        for item in items:
            yield function(context, item)
        return
    context_blob = _pack(context)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        binding = hashlib.sha256(function.__module__.encode() + function.__qualname__.encode()
                                 + _fingerprint(context)).hexdigest().encode()
        header = checkpoint_dir / 'context.sha256'
        if header.exists() and header.read_bytes() != binding:
            raise ValueError(f'checkpoint context changed: {checkpoint_dir}')
        _atomic_write(header, binding)
    items = enumerate(items)
    with ExitStack() as stack:
        pool = None
        while batch := tuple(islice(items, workers)):
            tasks, ordered = [], []
            for index, item in batch:
                path = None if checkpoint_dir is None else checkpoint_dir / (
                    f'{index:08d}-{_fingerprint(item).hex()}.pkl.gz')
                cached = path is not None and path.exists()
                ordered.append((path, cached))
                if not cached:
                    tasks.append((item, path))
            if workers == 1:
                results = []
                for item, path in tasks:
                    result = _pack(function(context, item))
                    _atomic_write(path, result)
                    results.append(None)
                results = iter(results)
            elif tasks:
                if pool is None:
                    pool = stack.enter_context(multiprocessing.get_context('spawn').Pool(
                        min(workers, len(batch)), initializer=_initialize,
                        initargs=(function, context_blob)))
                results = pool.imap(_run, tasks, chunksize=1)
            else:
                results = iter(())
            for path, cached in ordered:
                result = None if cached else next(results)
                yield _unpack(path.read_bytes() if path is not None else result)
