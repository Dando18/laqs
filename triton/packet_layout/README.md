# LAQS packet-layout workflow

This implements the packet-preserving optimization boundary proposed in the
attached design: select an explicit storage-address template, materialize its
integer packet bases, preserve native vector primitives, and keep analytical
selection separate from a bounded measured choice.

This is a new protocol under `results/packet-v1`. It does not redefine the old
Experiments 4–6 or overwrite their reports. The new plugin is
`libLAQSTritonPacketLayout.so`; the generic rewrite remains available for
same-layout diagnostics. Compiler or source changes require a fresh result root.

## Evaluate the address-generation repair in resumable debug jobs

Start with the saved **Matrix asymmetric GEMM**. This freezes its original
launch, inputs, objective and large layout, reuses its completed CPU graph,
and compares six variants:

| Variant | Storage | Lowering |
| --- | --- | --- |
| ordinary | ordinary | native |
| current | saved large LAQS map | legacy packet implementation |
| repaired | exactly the same large map | transparent carrier and proved physical recurrence |
| smaller | smaller equal-J map | repaired |
| identity_legacy | ordinary, through the rewrite | legacy; same input/output addresses as ordinary |
| identity_repaired | ordinary, through the rewrite | repaired; same input/output addresses as ordinary |

Build once on Matrix, from the repository root. The build script applies the
checked-in compiler patch to the source used by that platform's CMake build
and rebuilds **libtriton and the packet plugin**. Rebuilding only the plugin is
insufficient.

```bash
triton/packet_layout/build-matrix.bash
srun -N1 -n1 -c8 -G1 -p pdebug -t 00:30:00 \
  triton/packet_layout/run-debug-matrix.bash --minutes 25 \
  --root triton/experiments/results/packet-affine-debug \
  --address-reference triton/experiments/results/packet-debug-split \
  --cases gemm--asymmetric --profile-addresses
```

**Repeat the exact allocation command to resume.** Completed stages, timing
processes, and individual counter variants are retained. Use `--retry-failed`
after fixing a failed stage's environment. The default eight CPU workers are
ample for this panel: the old graph is reused and only layout selection is
repeated. There is no new tracing or graph construction. The old results are
read without alteration. A different launch, input, grid, scalar argument,
kernel source or objective makes the comparison fail explicitly.

`--profile-addresses` appends a separate, resumable Nsight Compute stage after
unprofiled timing. The Matrix wrappers load `nsight-compute/2025.3.0` after
CUDA and check `ncu --version` before starting. CUDA's bundled `ncu` launcher
alone may point to a missing Nsight installation. Set `RELAY_NCU_MODULE` to
override the profiler module. GPU counter access must also be permitted.
If profiling fails, the completed timing results remain available; repeat the
same command with `--retry-failed` after correcting the environment.
To collect timing alone, omit that flag **on the first run**, then keep the
same settings when resuming. Profiling can subsequently be run directly in an
allocation with `job-matrix.bash --stage profile`, the same root/reference/case,
and `--grammar split --selection analytical --resume`.

Profiling records one dispatch after repeated same-candidate warmups, with
application replay, `--cache-control none` and `--clock-control none`. Warmup
therefore runs again for every counter pass. This targets the repeated warm
kernel regime used for timing; it is not a cold-cache or producer–consumer
measurement. See the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
for the replay/cache distinction. Counter durations never enter speedups.
The profiled cubin must match the graph-captured timing cubin by hash.

The counter panel contains absolute L1TEX global-load sectors, requests and
wavefronts, TEX-source L2 read requests/sectors, L2 read misses and DRAM read
bytes. These are whole-kernel counts. The historical `counter_components`
association between a 128-byte issue footprint and `global_load_requests`
is a heuristic association, **not equality with the L1TEX request counter**.
Coarser quotient counts remain locality/service proxies, not cache-miss
predictions. This experiment preserves the existing tau weights.

Results are under
`triton/experiments/results/packet-affine-debug/matrix/gemm--asymmetric/expert/`:
`analysis.md`, `report.json`, the actual graph-captured binaries under
`evaluate/process-*-codegen/`, and optional absolute counts in `profile.json`
with raw CSVs under `profile/`. Rejected variants are listed explicitly; a
partial comparison cannot be presented as a successful repair.

On Tuolumne, `sum--small` has the completed reference search needed for this
quick comparison. Its loop rebuilds addresses from an integer induction
variable. The recurrence pass now handles this form as well as loop-carried
pointers. The changed repaired variants should report one physical recurrence
and no permutation shifts inside the reduction loop:

```bash
triton/packet_layout/build-tuolumne.bash
flux run -N1 -n1 -c8 -g1 -q pdebug -t 30m \
  triton/packet_layout/run-debug-tuolumne.bash --minutes 25 \
  --root triton/experiments/results/packet-affine-debug \
  --address-reference triton/experiments/results/packet-debug-split \
  --cases sum--small
```

Use a **fresh root** for this revision; completed `packet-address-debug`
checkpoints describe the previous compiler and timing protocol. The old
`packet-debug-split` reference stays unchanged.

The six-variant comparison now times every variant at three shared input
placements in each process. Prepared bytes are copied into the same input
buffers outside timing, followed by same-candidate warmup. All candidates use
the same output buffers except the independent-output identity control.
Placement order and variant order rotate. `evaluate/process-*.json` records
each placement's timings, addresses, and device identity; `analysis.md` exposes
placement and process speedups. Placements are repeated measures, not extra
independent processes. These are warm steady-state measurements.

Separate tuning also freezes a practical selection among ordinary, repaired,
and smaller. It retains ordinary unless the existing confidence/control rule
establishes a gain. This selection is evaluated in fresh processes and reported
separately from the unchanged analytical ablation choice. Neither the objective
nor the smaller-representative tie-break changes.

For an hour, use `-t 01:00:00` / `-t 60m` and `--minutes 55` if the debug queue
allows it. On Matrix, try `pdebug` first; the same short allocation with
`-p pbatch` is an alternative when both debug nodes are occupied. Within an
existing allocation, invoke the runner via `srun` or `flux run` with one task,
eight cores and one GPU and a work budget fitting the remaining time.

To collect all 29 cases using the repaired implementation, omit
`--address-reference`, `--cases` and `--profile-addresses`, and use a new
`--root triton/experiments/results/packet-debug-repaired`. This broader run
performs exact CPU search and resumes across debug allocations as below.

## Isolate Matrix copy-service work in a short debug job

This A-only diagnostic compares the ordinary, large, and smaller GEMM source
maps with the saved 128-thread ownership, 16-byte packets and shared-memory
swizzle. It compares asynchronous copies against synchronous global-load /
shared-store copies. Address increments are constant in both versions. The
matrix pipeline and B traffic are deliberately omitted to isolate copy service.

```bash
srun -n1 -G1 -c8 -p pdebug -t 00:30:00 \
  triton/packet_layout/run-copy-probe-matrix.bash --minutes 25 \
  --root triton/experiments/results/copy-service-debug/matrix
```

The wrapper loads CUDA and the working Nsight Compute module. The probe builds
itself; it does not require rebuilding Triton. It validates copied values and
checks final SASS for exactly two 128-bit global copies per loop body. Each of
the six variants has independent timing and counter checkpoints. Repeat the
same command to resume or retry an interrupted stage; source/tool changes
require a fresh root. For an existing allocation, invoke the script directly
with a budget shorter than the remaining allocation time.

Inspect `analysis.md`, `report.json`, raw `.csv` counters and `codegen.sass` in
the probe root. A warp-union model predicts 4,194,304 load sectors for each
variant. Inflation in the large async variant alone would implicate the async
service path; inflation in both copy forms suggests a broader source-service
issue. Failure to reproduce the GEMM inflation points back to its surrounding
pipeline or cache behavior. Four-lane footprint counts are included as a
diagnostic hypothesis; the probe does not install a new tau weight or claim a
universal hardware grouping. Single-process probe timings are diagnostics,
not full-kernel speedup evidence.

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

## Collect all 29 TritonBench cases in debug allocations

The resumable runner uses one GPU, eight CPU workers, the split-only grammar,
and analytical selection by default. It runs capture, exact CPU search,
primitive/numerical validation, selection, and fresh held-out evaluation for all
29 configured TritonBench cases across 15 operators. The synthetic row/column
cases are not part of this panel. Evaluation retains the regular three timing
processes, 21 samples, 50 iterations, and 10 warmups; these are not shortened
smoke-test measurements. The GPU is idle during CPU search.

Build once on the respective system before starting the first allocation:

```bash
triton/packet_layout/build-tuolumne.bash   # Tuolumne
triton/packet_layout/build-matrix.bash     # Matrix
```

Run the following from the repository root. These are commands for the user to
submit; a 30-minute allocation gets a 25-minute work budget and cleanup margin.

Tuolumne, queued debug submission:

```bash
flux submit -N1 -n1 -c8 -g1 -q pdebug -t 30m \
  --output='triton/experiments/results/packet-debug-tuolumne-{{id}}.log' \
  triton/packet_layout/run-debug-tuolumne.bash --minutes 25
```

Matrix, foreground debug allocation:

```bash
srun -N1 -n1 -c8 -G1 -p pdebug -t 00:30:00 \
  triton/packet_layout/run-debug-matrix.bash --minutes 25
```

For an hour, use `-t 60m` / `-t 01:00:00` and `--minutes 55`, if the queue
permits an hour. On Matrix, try `pdebug` first; if its debug nodes are occupied,
the same command with `-p pbatch` is an alternative. For interactive shells:

```bash
# Tuolumne: allocate, then launch the runner inside that allocation.
flux alloc -N1 -n1 -c8 -g1 -q pdebug -t 30m
flux run -n1 -c8 -g1 triton/packet_layout/run-debug-tuolumne.bash --minutes 25

# Matrix: allocate, then launch the runner as a job step.
salloc -N1 -n1 -c8 -G1 -p pdebug -t 00:30:00
srun -N1 -n1 -c8 -G1 triton/packet_layout/run-debug-matrix.bash --minutes 25
```

If you spend time in an interactive shell before launching, reduce `--minutes`
to fit the remaining allocation. The runner also handles SIGINT, SIGTERM and
SIGUSR1, stopping the stage's process group, including spawned workers.

**Repeat the same command to continue. Keep the same result root.** The default
is `triton/experiments/results/packet-debug-split`, with separate platform
subdirectories, so Matrix and Tuolumne may run concurrently. Only one allocation
may own a given platform/root at once. Do not generate a fresh timestamp on each
restart. Large trace and graph checkpoints live under this shared results root,
not node-local `/tmp`.

Progress and results appear in:

```text
<root>/<platform>/debug-summary.md
<root>/<platform>/debug-state.json
<root>/<platform>/<case>/expert/analysis.md
<root>/<platform>/<case>/expert/debug-<stage>.log
```

Completed stages have dependency-bound output checksums. During CPU search,
completed four-workgroup trace batches and individual scope partitions are
written atomically by workers and reused after interruption. Their order,
representatives, weights, and exact scores are preserved. A completed graph is
saved before selection. Completed independent timing processes are also reused.
The unfinished batch/partition, graph serialization or selection pass may need
to run again; this does not resume arbitrary instructions. If an individual
unit cannot finish within an allocation, that case needs a longer allocation.
Checkpoints add shared-filesystem I/O and disk usage.

Untouched cases get a turn before retrying an interrupted long case. A failed
case is retained in the summary and other cases continue. Repeat with
`--retry-failed` to retry failed stages; completed earlier stages stay cached.
Failures/exclusions are not counted as completed results. A suite ending with
only complete/failed cases exits nonzero if any failed. A budget-limited partial
run exits successfully and reports how much remains.

The manifest freezes sources, GPU/compiler/plugin identity, case panel, grammar,
tau profile and measurement settings. Changing those requires a fresh `--root`;
fixing source code therefore requires a new run. Worker count and allocation
budget may change when resuming. Old `packet-v1` roots are not imported.

Eight workers is the recommended starting point on both systems, **not a
measured optimum**. Scope construction currently has eight independent
partitions; tracing can use more workers. Sixteen is a reasonable next tracing
experiment: change the reservation to `-c16` and pass `--cpu-workers 16` on the
same root. More workers replicate graph/interpreter state and increase memory
and I/O demands, so using all 96/112 cores is unlikely to help this runner.
The wrappers cap nested OpenMP/BLAS threading at one thread per worker.

Inspect the full command list without building, submitting, or requiring a GPU:

```bash
.venv/bin/python triton/packet_layout/debug-suite.py --platform tuolumne --dry-run
```

For a separate smaller panel, pass `--cases sum--small fp8_gemm--small` together
with a fresh `--root`. `--grammar split-chunks` and `--selection measured` are
available for separate experiments.

## CPU iteration and timing

Every search prints phase timings and writes `search-timings.json` beside
`search.json`: capture loading, tracing, edge-family construction, component
materialization, selection, graph writing, and total time. Timing diagnostics
are separate from the signed analytical selection.

Submission scripts reserve four CPU cores for each search and pass
`--cpu-workers 4` to the driver. Override this with, for example,
`submit-tuolumne.bash --cpu-workers 8`. GPU stages retain their one-task,
one-GPU allocation. Direct driver and CPU benchmark commands default to one
CPU worker; reserve cores before increasing their worker count.
Already-submitted one-core jobs retain their original requests.

Tracing is divided into contiguous workgroup batches and merged in launch
order. Scope construction runs in independent partitions, preserving original
within-scope edge and floating-point accumulation order. Spawned workers avoid
inheriting the GPU runtime. Pools bound in-flight results and terminate when
analysis fails. More workers require more host memory because each owns its
trace/graph working state; they are not GPU timing processes.

For CPU development, replay an existing trusted capture without a GPU or a
new capture job. Run from the repository root, substituting the case directory:

```bash
.venv/bin/python triton/packet_layout/benchmark-cpu.py \
  --platform tuolumne \
  --directory triton/experiments/results/packet-v1/tuolumne/row_column--small/expert
```

Add `--selection-only` to reuse a completed graph and skip tracing and edge
construction. This diagnostic prints timings, the current selection, and
whether it matches the saved selection. It verifies cached payload hashes but
intentionally runs current CPU sources against the frozen input. It writes no
workflow artifacts and does not certify compiler realization or GPU speedup.
Normal experiment runs still require a fresh capture/root after source changes.

To measure graph construction using a four-core CPU allocation on Tuolumne:

```bash
flux run -n1 -c4 -q pdebug -t 30m \
  .venv/bin/python triton/packet_layout/benchmark-cpu.py \
  --cpu-workers 4 --platform tuolumne \
  --directory triton/experiments/results/packet-v1/tuolumne/sum--small/expert
```

For an interactive GPU allocation, use `-n1 -c4 -g1` with Flux, or
`-n1 -c4 -G1` with Slurm, and pass `--cpu-workers 4` to the job wrapper.

The packet search uses exact prepared region counts: affine cosets are scored
by the rank of their mapped quotient basis; non-affine sets use exact encoded
point counts. Static ownership, broadcast indexing, coordinate metadata, and
repeated affine proofs are cached. These optimizations preserve templates,
partial-flag equivalence, edge multiplicities, weights, and score tie-breaking.
Graph caches use lossless gzip level 1 to avoid expensive maximum compression.

Loop-containing kernels still use concrete tracing. The optimization does not
claim size-independent CPU time or introduce access sampling.

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
scale and the packet depth. Representatives prefer ordinary storage, then
fewer moved address bits, then fewer address fields and a deterministic row
order. This ordering also applies to equal-J candidates within and across
families. Moved bits are a tie-break heuristic, not a runtime cost model.
Ordinary storage wins score ties. Families yielding
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
pointers. In the default repaired lowering, a selected loop pointer must have
a bounded, constant positive power-of-two stride. Every source bit that can
carry during the frozen loop must map into a contiguous increasing physical
field. The initial transformed offset is computed before the loop and the
loop carries physical offset state with a constant increment. Its proved
no-overflow bounds survive LLVM lowering, allowing pointer strength reduction.
No per-element pointer state is introduced beyond the native tensor state.

Loop bounds and strides come from the IR or frozen scalar/grid bindings, which
are included in the compilation key and checked by `FrozenLaunch.run`.
Unsupported carries, unknown bounds, nonconstant updates or escaping final
pointers reject that candidate; the workflow retains ordinary storage. This
initial restriction can exclude useful layouts. It does not retry additional
templates or synthesize periodic address state machines. Original integer
addition is never treated as GF(2) addition without a carry proof.

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
discard attributes on arithmetic/addptr operations. The tagged
`laqs.packet_identity` carrier remains through those analyses, then lowers to
its operand during TritonGPU-to-LLVM conversion. The lowering checks its exact
single-input, single-output, pure i32 tied-register identity form. Untagged
assembly is unaffected, and malformed tags fail compilation. No opaque proof
assembly reaches LLVM optimization in the repaired path. Diagnostic modes
`legacy` and `transparent` retain the old recurrence implementation, with the
old or repaired carrier respectively; ordinary experiments use `repaired`.

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
