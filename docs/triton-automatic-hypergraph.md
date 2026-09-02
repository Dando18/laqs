# Automatic Triton-to-LAQS hypergraphs

The automatic frontend analyzes an ordinary concrete Triton launch. Kernel
source is unchanged and contains no LAQS coordinate maps, labels, weights, or
target-operand annotations:

```text
specialized Triton launch
  -> post-coalesce TTGIR access manifest
  -> launch-bound exact trace classes
  -> universal LAQS edge families
  -> optional HardwareProfile components and the existing solver
```

## Compiler and runtime boundary

The out-of-tree `LAQSTritonAccessManifest` pass runs immediately after the
first TritonGPU coalescing pass in both pinned AMD and NVIDIA pipelines. This
is late enough to observe specialized constants and compiler-selected
encodings, but precedes thread-locality optimization, loop scheduling,
pipelining, and LLVM lowering. A disabled-by-default `post_coalesce_hook` is the
only pipeline extension. It contributes to compilation cache identity when
enabled and copies the pass result from the TTGIR module into
`CompiledKernel.metadata`; ordinary Triton compilation is unchanged when it is
unset.

The module attribute contains version 1 of
`laqs.triton.access_manifest`. Its target-independent JSON records launch
arguments and paths; a typed expression DAG; a structured body of `for`, `if`,
barrier, and memory nodes; stable sites and pointer provenance; operation,
shape, type, cache, and conservative issue information; and native
`ttg::toLinearLayout` bases, dimensions, sizes, and free-variable masks. It has
no transaction sizes, cache capacities, architecture-specific owner bits,
counter names, or objective weights. Unsupported compiler constructs become
diagnostics rather than approximations or text-parsed fallbacks.

At launch time the binder captures the concrete callable grid, heuristic
values, autotuner configuration, runtime scalars, tensors, and tensor
descriptors. Allocation metadata includes logical shape and strides, element
width, storage identity, role, aliases, and dense-view status. Dense and
permuted-dense strides are inverted exactly. Non-power-of-two extents retain
their true bounds and use a next-power-of-two solver envelope.

The evaluator executes each exact representative program/wave class. It
evaluates loops, scalar control, masks, and offsets; maps register/lane/warp/
block locations through the recorded LinearLayout; and elects one replicated
owner with its free-variable masks. Dynamic issue order and mechanically
derived structured phases are retained. The bounded fallback enumerates every
context. A proved aligned-translation fast path represents full and boundary
program intervals without sampling. The proof checks fixed-width integer
expressions for wrap over the complete interval. If neither route is exact
within `EvaluationLimits`, analysis returns a categorized unsupported result.
When a hardware profile has resource maps, absolute logical anchors are kept
so resource-color phases remain available; an oversized resource trace is
rejected instead of translation-compressed.

## Public API

```python
from relay import AnalysisOptions, analyze_launch, get_hardware_profile

analysis = analyze_launch(
    kernel,
    grid,
    *kernel_args,
    _laqs_options=AnalysisOptions(
        hardware_profile=get_hardware_profile("mi300a"),
    ),
    **kernel_kwargs,
)

if analysis.supported:
    print(analysis.events, analysis.sequences)
    print(analysis.edge_families, analysis.components)
else:
    print(analysis.unsupported.category, analysis.unsupported.message)
```

The wrapper calls the normal JIT/heuristic/autotune stack once and analyzes the
`CompiledKernel` returned by that call. Triton autotuning may itself perform its
usual benchmark launches. The callable grid and final launch arguments are
captured at the underlying JIT call, after heuristics and the selected autotune
configuration have been applied.

A hardware profile materializes objective components; without one, the
target-neutral trace and universal edge families are still produced. For an
analysis created with a hardware profile,
`analysis.relay_problem(grammar="standard")` returns the existing
`SimpleRelayProblem` with `UniversalScopeObjectives` and that full profile, so
the simple frontier uses its tau, kappa, and fine component. The problem also
retains the resource maps for the existing profile-aware scoring path.
The explicit target-neutral
`analysis.relay_problem(byte_scales=(...))` form returns the existing
`RelayProblem`. A profile containing resource maps cannot be attached after a
target-neutral trace was already translation-compressed; pass it in
`AnalysisOptions` at launch analysis time.

## Exact supported subset and multiplicity

The implemented exact contract accepts tensor-of-pointer global loads, stores,
and atomics; ordinary descriptor block loads/stores; static and runtime
structured `scf.for` loops; scalar structured branches; loop-carried scalar and
pointer offsets; and masks in the integer expression language. Expressions
include constants, runtime integer scalars, `program_id`, `num_programs`,
ranges, splat/broadcast, expand-dims, reshape, transpose, layout conversion,
integer arithmetic and bitwise operations, comparisons, select,
signed/unsigned min/max, address-preserving casts, and `tt.addptr`. Layout
conversion changes ownership, not logical tensor values. CPU manifest tests
exercise these evaluator paths; successful compilation of a particular kernel
is reported separately by the platform coverage driver.

One level of data-dependent indexing is exact when a load from an unambiguous
read-only integer launch allocation directly supplies a later address. The
compiler represents that dependency as a `gather` DAG node. Outside kernel
execution, the evaluator copies that integer allocation to a flat host integer
sequence, resolves every active lookup with concrete values, and checks its
bounds. The allocation must remain read-only and host-copyable, and only this
single lookup level is admitted.

Allocation views must have positive, invertible dense or permuted-dense
strides. The global eligibility policy is uniform: a layout candidate is
read-only, dense or permuted-dense, and has no storage alias in the launch.
Outputs, read-write values, atomics, aliases, and unsupported views remain
fixed, while every global allocation remains in the trace.

An `EventSequence` is one exact trace class. Its multiplicity is the number of
represented program/wave occurrences. A memory event may carry an intra-class
base weight; every universal builder uses
`event.weight * sequence.weight` exactly once. IDs may be shared across classes
but may not repeat within one sequence. Issue, temporal, workgroup, phase,
persistence, useful-byte, and resource-cohort builders share this contract.

## Explicit unsupported results

The frontend rejects rather than guesses on data-dependent `scf.while`,
loaded-data control, pointer chasing, ambiguous pointer provenance, descriptor
gather/scatter, opaque custom memory, address-affecting inline assembly,
negative or overlapping views, clustered-CTA layouts, unsupported LinearLayout
dimensions, inconsistent descriptor metadata, non-integral address semantics,
active direct out-of-bounds accesses, active descriptor stores outside the
true shape, or exact bounds exceeded without a translation proof. Non-dense
views remain fixed rather than layout candidates and are accepted only when
every active offset can be inverted unambiguously. Descriptor loads outside
the true shape are omitted only when the concrete `TensorDescriptor` supplies
Triton's zero or NaN padding policy.

Loaded values used for control, a second loaded-data lookup, arbitrary pointer
chasing, and non-integer or mutable lookup allocations remain unsupported.
There is no runtime heuristic or sampling fallback: a memory-result dependency
must match the compiler's explicit one-level `gather` form. The frontend never
falls back to annotations, regex IR parsing, or a Triton-specific score.

## Build and validation

Both cluster installers verify the exact pinned revision containing the
committed backend-neutral hook, enable extension symbols, and include
`triton/automatic_frontend` through `TRITON_PASS_PLUGIN_DIRS`. They load the
resulting library and verify pass registration. Matrix builds the same commit
in its platform-isolated CUDA clone and exposes the library under the ignored
`triton/plugins/matrix` convenience path. Automatic discovery first resolves
the library beside the Triton package imported by the active environment. It
does not fall back to a library built for another checkout, which prevents a
Tuolumne ROCm extension from being loaded into a Matrix CUDA process.

CPU tests cover schema/expression evaluation, native ownership, structured
execution, stride inversion, descriptor bounds, fixed-width overflow,
boundary trace classes, multiplicity, universal edges, resource-anchor
retention, and small independent address oracles. These tests use hand-authored
manifests and therefore validate the runtime boundary independently of plugin
compilation. GPU support is reported as a coverage distribution rather than
inferred from those tests:

```bash
# Run inside one task/one GPU allocation of at most five minutes.
triton/.venv/bin/python triton/run-automatic-frontend-coverage.py \
  --json triton/results/automatic-frontend-coverage-mi300a.json

triton/.venv-matrix/bin/python triton/run-automatic-frontend-coverage.py \
  --json triton/results/automatic-frontend-coverage-h100.json
```

The driver records `supported`, categorized `unsupported`, or `error`, plus
allocation, trace-class, multiplicity, edge-family, and component counts.
`--require-supported` turns any non-supported result into a command failure.
It imports the editable Triton selected by the active Python environment before
adding the RELAY source tree. It then verifies that package's enclosing Git
checkout is exactly commit `b3376d6459bfb14f2500c1c20b3948ad59649bf8`.
This admits the shared Tuolumne checkout and the Matrix installer's
platform-isolated clone while rejecting an installed upstream wheel, a checkout
at another revision, or the repository's `triton/` namespace directory.

`analyze_launch` and this coverage driver are the production construction path.
The pre-existing `run-stage1*.py`, `stage1_common.py`, and `stage1_operand.py`
manual builders remain measurement and equivalence oracles only; the automatic
frontend never calls them and has no annotation-based fallback.

The checked-in Tuolumne records are the measured MI300A launch results from
September 2, 2026. The compiler-only rows use the same pinned revision and
plugin but do not claim execution on the named target:

| Case | Feature exercised | Current evidence |
| --- | --- | --- |
| Hand-authored vector manifest and fake launch wrapper | callable-grid binding, masked boundary, read/write trace | CPU tests pass, including an independent address oracle |
| Hand-authored descriptor manifest | concrete shape/strides, padded load, out-of-bounds store rejection | CPU tests pass |
| TritonBench vector add | callable grid, masked boundary, read/read/write | MI300A ordinary launch supported; 2 trace classes and 24 edge families ([record](../triton/results/automatic-frontend-coverage-mi300a.json)) |
| TritonBench fused softmax | reduction and non-power-of-two row | MI300A ordinary launch supported; 2 trace classes and 24 edge families ([record](../triton/results/automatic-frontend-coverage-mi300a.json)) |
| TritonBench tutorial matmul | autotune, runtime strides, PID mapping, inner loop | MI300A ordinary launch supported; selected `32x16x128`, 2-warps configuration, 2 trace classes, and 24 edge families ([record](../triton/results/automatic-frontend-coverage-mi300a.json)) |
| TritonBench layer norm | repeated loops and scalar stores | MI300A ordinary launch supported; 4 trace classes and 24 edge families ([record](../triton/results/automatic-frontend-coverage-mi300a.json)) |
| Descriptor block copy | native descriptor load/store and concrete runtime descriptor metadata | MI300A ordinary launch with the `mi300a` hardware profile supported; 4 trace classes, 24 edge families, and 288 materialized objective components ([record](../triton/results/automatic-frontend-coverage-mi300a-descriptor.json)); pinned CUDA `sm90` compiler manifest test also passes |
| Stage-1 bias + ReLU, softmax + bias, GEMV, MVT, GESUMMV, stencil, and prepacked-B GEMM | structured addressing, reductions, loops, neighborhood accesses, and dot tiles | pinned HIP `gfx942` compilation plus independent host address traces match all 24 universal families and all 12 MI300A quotient scales for every full-envelope canonical word; GEMV, MVT, GESUMMV, and stencil also match resource cohorts and robust placement scores |
| Stage-1 embedding bag | one-level concrete read-only integer gather | pinned HIP `gfx942` compilation with host-copyable indices matches the independent address trace, all universal families, and every MI300A quotient component; absolute resource placement is not claimed after translation compression |

The platform JSON is the authoritative support matrix: a row is not claimed
supported until that platform's driver reports it as such.
