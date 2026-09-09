# CPU search optimizations

The CPU path now avoids repeated static tracing work and uses exact prepared
region counts in the packet-layout search. Candidate families, partial-flag
equivalence, numerical objective values, multiplicities, and tie-breaking are
unchanged. No accesses are sampled and no search budget is reduced.

## Implemented changes

- Cache native register/lane ownership per manifest node, wave, and block.
  Masks, offsets, control predicates, and loop-carried values are still evaluated
  dynamically. Distinct nodes with the same site label cannot share an entry.
- Cache small broadcast index maps, with bounds on both entry count and size.
  Tensor values are never cached by shape. Cache immutable matrix bit metadata.
- Reuse coordinate encodings and proved affine signatures while compressing
  edge families. Cache sizes are bounded; eviction only repeats exact work.
- Prepare each scored edge once. A proved affine coset uses the exact rank of
  its mapped quotient basis. Other sets retain exact point enumeration.
  Small integer lookup tables map coordinates without per-point layout
  validation, and mapped offsets are shared across overlapping edges/scales.
- Compute native tile hypotheses once per matrix and reuse packing bounds.
- Write internal graphs with lossless gzip level 1 instead of level 9.
- Log CPU phases and persist `search-timings.json` independently of the signed
  search result. Add `benchmark-cpu.py` for replaying frozen inputs using current
  CPU sources, including a selection-only mode that reuses a saved graph.

## Measurements and correctness

These are individual CPU checks on `tuolumne2151`, using the archived fixed
captures in `packet-implementation-checks-2026-09-08`. They are exploratory
measurements, not controlled repeated performance estimates. The row/column
phases were measured separately; their sum is not presented as a fresh full
workflow timing. No GPU job was required.

| Row/column small phase | Before | After |
|---|---:|---:|
| Concrete tracing | 87.3 s | 39.4 s |
| Edge-family construction | 215.4 s | 175.4 s |
| Complete analytical selection and score diagnostics | 79.1 s | 56.7 s |
| Compression of identical serialized graph bytes | 67.6 s | 0.93 s |

The reconstructed row/column events, sequences, and objective components matched
the archive exactly. Final selections, candidate scores, and search records
also matched after JSON normalization. The compressed graph grew from 26.6 MB
to 30.5 MB; decompression reproduced the identical original byte string.

For TritonBench `sum--small`, current capture-to-selection CPU replay took
57.8 seconds, versus 154.5 seconds in the archived search record. That comparison
uses the same frozen input but includes different run conditions and timer
boundaries: replay excludes graph writing. Its full selection and scores
matched the archive. Reusing its saved graph took about 0.3 seconds including
loading. Row/column graph loading plus selection took about 66.6 seconds.

147 targeted tests passed across scoring, layouts, objectives, solver,
persistence, frontend, access scopes, and experiment modules. New checks compare
prepared counts to direct exhaustive counts for affine translations, non-affine
sets (including power-of-two cardinalities), XOR layouts spanning lookup-table
boundaries, partial tiles, multiple arrays, and byte scales. Broadcast tests
exercise changing values, rank padding, scalar tensors, invalid shapes, and the
uncached large-shape path. A frontend regression covers repeated site labels
with different ownership maps and phase callbacks.

All-in-one unittest discovery was not clean: 18 errors involved the `experiments`
namespace, missing PyTorch, and missing HIP backend discovery; 10 tests were
skipped. Relevant CPU modules passed when run independently. GPU compilation and
timing were not rerun for these CPU changes.

Machine-readable measurements: [cpu-search-optimization-validation.json](cpu-search-optimization-validation.json).

## Rapid CPU iteration

From the repository root, this existing TritonBench example requires no GPU:

```bash
.venv/bin/python triton/packet_layout/benchmark-cpu.py \
  --platform tuolumne \
  --directory triton/experiments/results/packet-implementation-checks-2026-09-08/final-workflow/tuolumne/sum--small/expert \
  --selection-only
```

Omit `--selection-only` to include tracing and graph construction. Substitute
another trusted case directory and its platform to benchmark another capture.
The command reports whether the selection matches the prior search, and writes
no experiment artifacts. It intentionally allows current CPU code to consume a
frozen older capture; payload hashes are still checked. Normal experiment runs
retain their source-fingerprint checks and require a fresh capture/results root
after these source changes.

The largest remaining row/column cost is edge-family construction. Loop kernels
still expand concrete workgroups and iterations before merging equivalent
accesses. Exact compression before that expansion is the next substantial
improvement; it needs proofs covering loop-carried state, masks, temporal
scopes, and multiplicities. These changes do not make CPU cost independent of
problem size.
