# Packet address-generation repair

The implementation now retains the selected physical address recurrence and
removes the LAQS proof carrier before LLVM optimization. The locality objective
and tau weights are unchanged. The next useful performance result is the
frozen six-variant Matrix asymmetric-GEMM comparison documented in
[the debug-run instructions](../triton/packet_layout/README.md#evaluate-the-address-generation-repair-in-resumable-debug-jobs).

## Compiler changes

The `laqs.packet_identity` tag identifies the pure, single-input/single-output
i32 tied-register carrier. TritonGPU-to-LLVM conversion verifies its precise
identity form and substitutes the converted operand. The carrier still exists
when packet selection, pipelining and AMD buffer conversion consume its axis
facts. Ordinary untagged inline assembly retains its original behavior.

The packet pass now computes a transformed initial offset outside the loop
and carries physical offset state with a constant increment. For the first
implementation, a selected stream must use a uniform, positive power-of-two
logical stride and a bounded loop. Every bit that can carry must map into a
contiguous increasing physical field. This proves ordinary integer addition;
it does not distribute a GF(2) map over an arbitrary sum. Unknown bounds,
unsupported carries/strides, derived updates and escaping final pointers fail
validation, leaving ordinary storage as the deployment fallback.

Bounds can use the frozen launch's scalar arguments and grid. They are part of
the compilation key, and `FrozenLaunch.run` rejects a call that differs from
the bound launch. Direct callers of `structured_layouts(..., launch=...)` must
use that frozen-launch contract; an arbitrary direct JIT call bypasses the
launcher guard. Without bindings, only IR-provable bounds are available.

There was a second transparency problem in the compiler: generic integer-add
lowering copied attributes but lost the MLIR overflow **properties**. The
physical induction has a proved nonnegative, nonwrapping range; retaining
`nsw`/`nuw` through LLVM lowering permits it to widen without repeated signed
index extension. The core compiler patch therefore repairs integer-add
overflow-property lowering as well as the identity carrier. Both platform
build scripts apply this patch to their actual source clone and rebuild
libtriton and the packet plugin.

The compiler still has freedom to emit residual base/byte-address arithmetic.
This is a physical-offset recurrence implementation, not a promise of
instruction-for-instruction identity with the original pointer loop.

## Selection and experiment

Partial-flag representatives and equal-score winners within and across search
families now prefer ordinary storage, then fewer moved address bits, then
fewer address fields, then deterministic row order. This changes the tie-break,
not LAQS scores or the enumerated grammar. Recurrence validation is still a
compiler obligation after candidate selection; a rejected winner falls back
to ordinary instead of triggering more searches or new compiler strategies.

`--address-reference` freezes the saved launch and reuses its completed graph.
It builds the ordinary/current/repaired/smaller panel, plus two nonempty
identity-map specifications using legacy and repaired lowering. During graph
timing both forced-identity variants share the ordinary variant's actual input
and output addresses. Selecting the ordinary fallback cannot masquerade as
this control.

The current and repaired variants use identical saved layout rows. The smaller
variant must attain the same objective value. Capture verifies configuration,
input probes, scalar arguments, grid and TritonBench source identity. The debug
manifest freezes the reference records as well as current source/toolchain
identity. Old reference files are read without modification.

Code is saved from the kernels returned during actual timing-graph capture.
The optional Matrix counter stage runs afterwards, with independent resumable
checkpoints for each candidate. It requires the profiled cubin to match the
unprofiled timing cubin. Profiling duration does not enter reported speedup.
Absolute sectors, requests/wavefronts, L2 read sectors/misses and DRAM read bytes
are kept separate from quotient footprints. In particular, a 128-byte issue
footprint is not an L1TEX request count or a cache-miss prediction.

## Validation performed

The offline H100 compile used the saved asymmetric-GEMM dimensions
M=512, N=4096, K=4096, tiles 64×128×32, four warps and four stages. The saved
large map is `split-a9-v3`; the new smaller representative is `split-a1-v5`.
Their J values are both 0, versus 0.1166667 for ordinary storage in this graph.

| Variant | Opaque identity calls in LLVM | Cubin registers | Hot-loop `and.b32` | Hot-loop `shr.u32` | Hot-loop `mad.wide.s32` |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary | 0 | 117 | 0 | 0 | 0 |
| current | 8 | 114 | 4 | 2 | 2 |
| repaired | 0 | 96 | 0 | 0 | 0 |
| smaller | 0 | 96 | 0 | 0 | 0 |
| identity_legacy | 8 | 96 | 2 | 0 | 2 |
| identity_repaired | 0 | 117 | 0 | 0 | 0 |

These are **offline compiler diagnostics**, not H100 measurements. All variants
retain six static `cp.async.cg.shared.global` sites and two WGMMA matrix
instructions in the hot loop. The repaired identity matches the ordinary
hot-loop opcode histogram. The repaired changed maps still have two additional
64-bit shifts and two additional 64-bit additions there; their binaries are
not instruction-identical to ordinary. Resource inspection reports no local
memory or stack allocation. Total static opcode counts alone do not establish
dynamic pipeline neutrality or speedup.

Validation also included:

- 25 CPU tests covering exact search/ties, reference binding, counter parsing
  and interruption/resume behavior.
- All ten compiler/GPU tests on one MI300A debug GPU, including numerical
  copying through physical recurrences and masks, FP8 GEMM, same-pointer
  transformed identity controls, unsupported-carry rejection, and the frozen
  H100 offline compilation. The job completed within its five-minute limit.
- Offline compilation of all six packet pilot kernels on CUDA sm90 and AMD
  gfx942, retaining their packet contracts.
- A separate five-minute, one-GPU end-to-end smoke run through capture, reused
  search, validation, analytical choice and evaluation for Tuolumne sum small.
  All six variants passed validation and the comparison report was complete.
  CPU graph reuse plus selection took approximately 0.2 seconds. Its shortened
  measurement settings were for harness validation, not performance conclusions.

These initial implementation checks did not establish a speedup. The sum
smoke did contain a loop, but its addresses were rebuilt from the induction
variable and the first recurrence implementation did not transform them.
See [the subsequent result analysis](packet-address-results-analysis.md) and
[the updated debug commands](../triton/packet_layout/README.md#evaluate-the-address-generation-repair-in-resumable-debug-jobs).

## Follow-up implementation

The recurrence pass now also recognizes affine integer expressions of a loop
induction variable, using the existing integer-carry proof. Nonlinear or
unproved expressions retain their original address calculation. The actual
TritonBench sum's large and smaller layouts now produce one recurrence and no
inner-loop permutation shifts on both compiler targets.

The address panel compares three shared input placements per process, records
addresses and placement timings, and reports paired process ratios. Separate
tuning freezes a practical ordinary/repaired/smaller choice before evaluation;
the analytical ablation remains visible. The standalone Matrix copy probe
checks the saved A ownership and shared swizzle with async and synchronous
copies, without introducing an unverified hardware penalty into the objective.

Follow-up checks passed: 34 CPU tests (14 environment-dependent skips), all
13 compiler/GPU regressions across two one-GPU MI300A jobs, and a complete
six-variant sum workflow with shortened timing settings. The smoke run selected
ordinary storage; it establishes correctness and fallback behavior, not a
performance improvement. The CUDA diagnostic compiles for sm90a and all six
specializations pass final SASS primitive checks. H100 numerical execution and
counter collection still require the Matrix debug command.
