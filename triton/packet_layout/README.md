# LAQS packet-layout workflow

This implements the packet-preserving optimization boundary proposed in the
attached design: select an explicit storage-address template, materialize its
integer packet bases, preserve native vector primitives, and keep analytical
selection separate from a bounded measured choice.

This is a new protocol under `results/packet-v1`. It does not redefine the old
Experiments 4–6 or overwrite their reports. The new plugin is
`libLAQSTritonPacketLayout.so`; the generic rewrite remains available for
same-layout diagnostics. The sources and binaries used by the queued v2 jobs
are unchanged.

## Run the declared panel

From the repository root on the respective cluster:

```bash
triton/packet_layout/submit-tuolumne.bash
triton/packet_layout/submit-matrix.bash
```

The default panel is FP8 small, asymmetric GEMM, sum small, row/column matrix
products at N=1024 and N=4096, and large vector-add as a negative control. Each
case has separate capture, CPU search, GPU validation, tuning, and held-out
evaluation jobs. Row/column cases additionally get an equal-budget conventional
storage comparison job. Tuolumne CPU search reserves no GPU; Matrix reserves one GPU
because its partition rejected the CPU-only allocation. Every GPU job uses one
task and one GPU. Full jobs are user-submitted; agent smoke tests must still
follow the repository's five-minute limit.

For a targeted run:

```bash
triton/packet_layout/submit-matrix.bash --cases sum--small
```

Use `--selection analytical` to deploy the analytical result without a timing
choice, `--root /shared/path` for an isolated suite, and `--queue pdebug` to
change the queue. Inspect commands without building/submitting:

```bash
.venv/bin/python triton/packet_layout/submit.py --platform matrix --dry-run
```

## Search boundary

The first family enumerates all two-dimensional split-coordinate templates

```text
i = 2^a I + i0
j = 2^v J + s
p(i,j) = I*2^(M+a) + J*2^(a+v) + i0*2^v + s
```

where the minor dimension has M bits, and v protects the native vector packet.
Degenerate parameters include ordinary storage. The richer family additionally
interleaves whole chunks whose boundaries come from native access footprints,
packet width, and allocation extents, with at most six deposit fields. Both
families minimize the same LAQS objective; no timing data enters that search.

Supported refinements are deduplicated by `A^-1 U_d` at every declared byte
scale and the packet depth. The representative is chosen for address-template
simplicity within that equivalence class, rather than choosing an arbitrary
unrealizable flag first. Ordinary storage wins score ties. Families yielding
the same physical mapping share one candidate.

The search returns at most three distinct physical candidates: ordinary, the
split-family optimum, and the chunk-family optimum. Its exactness claim is
limited to the explicit enumerated families and supplied graph. Compiler
validation remains necessary; the result is not an optimum over arbitrary
layouts or all possible compiler implementations.

Initial automatic templates support eligible, direct, dense, read-only 2D
allocations with power-of-two extents and fewer than 31 element-address bits.
Other operands remain ordinary with an explicit reason. Descriptor/TMA and
transformed producer stores are unsupported. This is an intentional initial
compiler family, not an approximation to those access paths.

## Compiler realization and contracts

`OffsetBuilder` recovers base-relative integer SSA through `tt.addptr`, shape
operations, same-width pointer bitcasts, selects, and supported loop-carried
pointers. Loops receive parallel integer offset state. Unsupported provenance
fails compilation; there is no pointer-to-integer reconstruction fallback.
The address envelope has fewer than 31 bits, so 32-bit modular offset state
preserves every bit used by the permutation even when intermediate additions
wrap. Original index arithmetic retains its original integer semantics.

`PacketAddressBuilder` deposits disjoint coordinate fields, retaining the
ordinary low packet field. Shape operations preserve broadcasts. Integer sums
are split only when conservative possible-bit masks prove there can be no
carry. Canonicalization, common-subexpression elimination, and loop-invariant
code motion simplify the resulting address expressions.

At each changed load, the plugin checks the original packet width against the
relative storage permutation. It attaches only the contiguity/divisibility
bounds and constancy facts implied by that proof, then reruns AxisInfo to check
that the packet is recognized. Masks and distributed value types are unchanged.
Per-site packet, alignment, mask, and ownership records are retained in kernel
metadata.

A pure, tied-register identity carries the proved integer-offset facts through
arithmetic reassociation and AMD pointer-to-buffer conversion, which otherwise
discard attributes on arithmetic/addptr operations. It uses an empty inline
assembly body and emits no address arithmetic or memory operation. This is the
initial proof carrier using Triton's existing IR; an opaque late LLVM rewrite
would miss the earlier vectorization and pipelining decisions.

Deployment validation checks numerical outputs, final memory and matrix
primitive forms, packet-site records, spills, shared-memory growth, and
recorded communication instructions. Register and static instruction counts
are diagnostics, not blanket vetoes. Per-site records describe the
post-coalescing contract; the final ISA check also matters because later
compiler passes can change lowering. An opcode histogram alone is not claimed
to prove every final dynamic memory grouping.

## Timing and interpretation

Tuning captures graphs for no more than three eligible physical candidates.
It includes independent ordinary-identity and same-pointer controls, rotates
execution order, and rotates output allocation assignments across processes.
Ordinary remains selected unless a candidate's paired-process confidence
interval clears the minimum gain and the controls. The choice is written and
hashed before fresh evaluation processes start.
With maximum observed control deviation `d`, the required lower-bound speedup
is `(1 + minimum_gain) * (1 + d) / (1 - d)`; `d >= 1` forces ordinary storage.
This empirical two-sided allowance rejects gains comparable to control noise
without rejecting a large gain merely because a control varies by a few percent.
It is a conservative selection policy, not a statistical bound on all systematic
measurement error.

Reports retain the analytical top-1, every candidate's compiler disposition,
tuning measurements, held-out evaluation, confidence intervals, and controls.
Rejected analytical layouts remain visible. Final evaluation does not change
the selected layout. The `speedup` and `l1_to_l2` tau maps are still legacy pilot
ablations, not recalibrated models.

Packing is outside kernel timing. Warm packing is measured separately with both
GPU events and synchronous wall time including allocation, excluding input
generation and compilation. Reports include one-use conversion-inclusive
speedup and the estimated number of reuses needed to amortize packing. This is
not a producer–consumer pipeline measurement. No positive amortization estimate
is reported when a changed kernel fails to save time.

## Conventional storage comparison

The row/column operator computes both `A @ x` and `A.T @ z`. Its explicit
row-major, column-major, and 16×16 tiled implementations receive the same six
schedule choices: `BLOCK_M` in {1, 4, 16}, `BLOCK_K` in {64, 128}, four warps.
Every changed LAQS map receives those same six choices through the automatic
packet pass, with correctness and primitive validation at each schedule.
Invalid schedules count against the offered budget and remain in the record.

Each storage's schedule is frozen after tuning, then evaluated in fresh
processes against the tuned ordinary baseline. Results are in `conventional.md`
and `conventional.json`. This comparison is separate from the bounded deployment
selection. It is a declared schedule budget, not a claim to the best possible
conventional kernel or tile. Column/tile kernels use explicit address functions;
they are never counted as automatic compiler results.

## Same-layout diagnosis

After capture and search, run inside a platform GPU allocation:

```bash
triton/packet_layout/job-matrix.bash --stage diagnose --platform matrix \
  --case sum--small
```

This compares ordinary storage, the generic rewrite, and the structured rewrite
of the **same map**. These measurements do not participate in deployment
selection. Code and timings are saved separately under `diagnose/`.

To diagnose a previously saved E4–6 proposal, supply its
`--diagnostic-selection .../proposed-selection.json` at capture and diagnosis,
using a fresh `--root`. Capture then freezes that proposal's configuration.
The diagnostic rejects mismatched cases/configurations and unsupported maps.
A handwritten address realization is a diagnostic baseline, not an automatic
search result.
With an explicit old proposal, capture and diagnosis can run directly without
repeating CPU search. For example, inside a Tuolumne GPU allocation:

```bash
relay_proposal=triton/experiments/results/search-v2/tau-profiles/expert/experiment-4/tuolumne/fp8_gemm--small/proposed-selection.json
triton/packet_layout/job-tuolumne.bash --stage capture --platform tuolumne \
  --case fp8_gemm--small --root triton/experiments/results/packet-fp8-diagnostic \
  --diagnostic-selection "$relay_proposal"
triton/packet_layout/job-tuolumne.bash --stage diagnose --platform tuolumne \
  --case fp8_gemm--small --root triton/experiments/results/packet-fp8-diagnostic \
  --diagnostic-selection "$relay_proposal"
```

## Artifacts and checks

Each `<root>/<platform>/<case>/<tau>/` contains:

- `capture.json`, `capture.pkl.gz`: frozen native configuration, integer inputs,
  input byte probes, source/compiler/plugin fingerprints, independent reference.
- `search.json`, `graph.pkl.gz`: candidates, partial-flag counts, per-array
  templates, full LAQS score vectors, and graph identity.
- `validated.json`, `codegen/`: numerical checks, exclusions, saved IR/ISA/binary
  hashes, primitive contracts, and resource counts.
- `choice.json`, `tune/`: bounded measured or analytical deployment decision.
- `evaluation.json`, `evaluate/`: fresh evaluation after the decision is frozen.
- `report.json`, `analysis.md`: assembled results with packing interpretation.
- `status.json`: the first causal stage failure; downstream stages preserve it.

Search, validation and choice records have checked content hashes. Input shape,
strides, dtype and deterministic byte probes must match capture. A timed-out
stage that leaves `running` status is reported as interrupted before downstream
work starts. Use a fresh result root after changing the sources or compiler.

Run CPU checks with:

```bash
.venv/bin/python -m unittest discover -s tests -p test_packet_layout_search.py
```

GPU checks use the platform Triton environment and one debug GPU:

```bash
flux run -n1 -g1 -t 5m -q pdebug triton/packet_layout/smoke-tuolumne.bash
```

The packet witness exhaustively checks bijection, vector alignment, region
counts, and equal partial flags with different complete flags. GPU regressions
check vectorized copying, loop-carried pointers, masks, and additions with
carries, rejected atomic storage, conventional schedules, and FP8 vectorization
through the complete MI300A pipeline. The same test file also compiles an H100
packet copy to PTX/cubin without needing a GPU. Passing these tests does not
itself establish a full-panel speedup. Implementation results are recorded in
[`notes/packet-layout-implementation.md`](../../notes/packet-layout-implementation.md).
