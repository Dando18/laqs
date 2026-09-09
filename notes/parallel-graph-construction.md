# Exact parallel graph construction

Graph construction now supports multiple CPU workers, and new packet-suite
submissions request four CPU cores for their search stages by default.
The direct driver and CPU replay utility retain a one-worker default so an
existing one-core allocation is not oversubscribed.

## Changes and exactness

Launch interpretation partitions the original program-ID order into contiguous
batches of four workgroups. Each worker evaluates every workgroup, wave, loop,
predicate, mask, and memory access using the existing interpreter. It merges
equivalent traces within its batch. The parent merges batches in their original
order, retaining the first representative and summing integer multiplicities.
Both local and global retained-event limits remain enforced. Resource-anchor
preservation disables translation normalization exactly as before.

Edge construction partitions the existing scope grammar by temporal window,
issue lane-group size, workgroup scopes, and phase scopes. Partitions have no
overlapping output scope keys. Each preserves its original edge insertion order
and floating-point weight-addition order. Final families retain the original
global schema order. Single-worker construction also processes and compresses
scope groups separately, reducing simultaneous raw-edge retention.

The interpreter additionally caches program-ID-dependent expressions that are
proved independent of all loop variables and loop-carried state. These caches
are confined to one program ID. It caches allocation stride order, avoids
recomputing an already-established translation anchor for every access, and
avoids a second normalization/merging pass over already-unique trace classes.
Temporal scope builders reuse per-event access selection across windows.

Workers use clean spawned processes, including when the parent has previously
used a GPU. Immutable mapping wrappers are serialized as dictionaries. Contexts
and returned batches use fast lossless compression. The parent permits at most
one submitted batch result per worker in flight, and consumes results in order.
Pool exit terminates workers on failure. Categorized frontend exceptions retain
their category and site when transported to the parent.

No template, access, scope, byte scale, score, or packing bound is removed or
approximated. This is still exhaustive tracing; CPU work is not independent of
launch size. Larger worker counts also increase host memory use.

## Validation and timing

Single CPU measurements on `tuolumne2151`, using frozen MI300A captures:

| Case | Reference graph time | Current time with 4 workers |
|---|---:|---:|
| FP8 GEMM, small | 855 s, archived v2 analysis | 137 s |
| Row/column, small | 215 s, previous trace + edge phase measurements | 135 s |

FP8's current phases were 134.5 seconds of tracing and 3.0 seconds of edge
construction. Row/column's phases were 21.5 and 113.9 seconds. The FP8 comparison
includes the earlier CPU optimizations as well as this change; these are not
controlled repeated scaling measurements. Neither timing includes layout
selection, GPU compilation/evaluation, queue delay, or graph-file writing.

Both runs matched the archived allocations, matrix metadata, memory events,
event sequences, edge families, and objective components exactly. This includes
representatives, ordering, source strings, and weights. Raw measurements and
test results are in [parallel-graph-validation.json](parallel-graph-validation.json).

151 targeted tests passed. New serial/parallel comparisons cover masked loops,
multiple workgroups, resource-anchor preservation, fractional weights, scope
ordering and provenance, retained-event limits, out-of-bounds rejection, and
exception propagation through worker pools. No GPU job was needed for these
CPU changes.

## Running

Submit a fresh panel from the repository root:

```bash
triton/packet_layout/submit-tuolumne.bash \
  --cpu-workers 4 --root triton/experiments/results/packet-parallel
```

Use `submit-matrix.bash` for Matrix. Both schedulers request the same number of
cores that the search driver uses. Tuolumne search jobs request no GPU; Matrix
retains its required GPU allocation. GPU-only stages still request one CPU core.
Use `--cases ...` to change the existing panel; this change does not expand it
or make it split-only. Increasing `--cpu-workers` to 8 is supported, but its
speedup and memory requirements have not been benchmarked here.

For a short end-to-end user-run debug allocation:

```bash
relay_debug_root="triton/experiments/results/packet-parallel-debug-$(date +%Y%m%d-%H%M%S)"
flux run -n1 -c4 -g1 -q pdebug -t 30m \
  triton/packet_layout/job-tuolumne.bash \
  --stage all --platform tuolumne --case fp8_gemm--small \
  --cpu-workers 4 --selection analytical --root "$relay_debug_root"
```

On Matrix, replace the launcher with `srun -n1 -c4 -G1 -p pdebug -t 00:30:00`,
the wrapper with `job-matrix.bash`, and the platform with `matrix`.
Thirty minutes is a cap for this user-run command, not a measured full-workflow
bound. CPU replay also accepts `--cpu-workers 4`; reserve four cores and omit
`--selection-only` to exercise tracing and graph construction.

Already-queued jobs retain their old core requests and one-worker commands.
They were not canceled or resubmitted. To use parallelism, submit fresh jobs
with a fresh capture/results root; normal source-fingerprint checks remain in
force. `search-timings.json` records the requested worker count and phase times.
