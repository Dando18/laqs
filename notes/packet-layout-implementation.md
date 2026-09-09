# Packet-preserving LAQS implementation

The implementation is in [`triton/packet_layout`](../triton/packet_layout/README.md).
It keeps Triton and LAQS, replaces the storage-address realization, and separates
analytical layout selection from a small measured deployment choice. This is a
new experimental protocol, `laqs.packet.v1`, with results under `packet-v1`.
The old Experiments 4–6 sources and compiler plugins remain unchanged because
Tuolumne v2 jobs were still queued during this work.

## Implemented changes

| Recommendation | Implementation |
| --- | --- |
| Preserve native vector packets | Check the relative permutation at each load, retain low packet bits, masks and distributed value ownership, and propagate proved AxisInfo facts. |
| Remove pointer round-tripping | Recover integer SSA through addptr, broadcasts, reshapes, selects and supported loop-carried pointers. Unsupported provenance is rejected. |
| Retain index decomposition | Project and deposit coordinate fields through shape operations; split integer sums only with a conservative no-carry proof. Run CSE and loop-invariant code motion. |
| Search supported templates directly | Enumerate split-coordinate templates and a bounded grammar of native-footprint chunks. Ordinary wins objective ties. Other allocations stay fixed with a recorded reason. |
| Use partial flags | Deduplicate supported refinements by `A^-1 U_d` at every declared scored scale and the protected packet depth, then prefer simpler address templates. |
| Bound runtime selection | At most three distinct physical candidates: ordinary, split optimum and chunk optimum. Preserve analytical top-1, and freeze an optional measured choice before fresh evaluation processes. |
| Narrow compiler guards | Require correctness, packet contracts, final memory/matrix primitive forms, no additional spills, shared-memory growth or detected cross-lane communication. Register counts and total static instructions are diagnostic. |
| Diagnose the same map | Compare ordinary, generic and structured lowering using one frozen configuration and identical inputs; accept saved v2 proposals directly. |
| Add conflicting storage preferences | Add simultaneous `A @ x` and `A.T @ z`, with ordinary, column-major and 16×16 tiled baselines. Offer the same six schedules to every storage and LAQS map, then evaluate the frozen schedules separately. |
| Account for conversion | Measure warm packing separately and report kernel-only speedup, one-use conversion-inclusive speedup and estimated amortization reuse count. |
| Preserve experimental provenance | Check source/compiler/plugin fingerprints, input metadata and byte probes, artifact content hashes, and the first failed or interrupted stage. |

Search optimality is limited to the explicitly enumerated template families and
captured graph. It is not a claim about all of GL, all canonical layouts, or all
compiler-feasible kernels. Initial automatic support is direct, dense, read-only,
power-of-two 2D allocations with fewer than 31 element-address bits. Descriptor/TMA
and transformed producer/store paths remain explicitly unsupported, as proposed.

## A compiler failure found and fixed during implementation

Removing `ptr_to_int` and attaching facts to the new pointer was insufficient for
FP8 GEMM. Later arithmetic reassociation and AMD pointer-to-buffer conversion
discarded those facts. The final code scalarized even though post-coalescing
AxisInfo had accepted the packet contract.

The new pass carries the proved facts on a pure tied-register identity around
the integer offset. Its assembly body is empty; the carrier survives those
rewrites and is visible to earlier scheduling and vectorization decisions. It
can still constrain optimization, so its profitability is measured. Offset
state uses 32-bit modular arithmetic: the admitted permutation only reads fewer
than 31 low bits, making this exact even through carries and intermediate wrap.

A same-map MI300A diagnostic used the saved expert E4 FP8-small proposal with
its original launch configuration. The completed two-process smoke measurement
used nine graph samples per process and twenty launches per graph:

| Realization | Final global load form | Registers | Spills | Static instructions | Speedup vs ordinary |
| --- | --- | ---: | ---: | ---: | ---: |
| Ordinary Triton | `buffer_load_dwordx2` | 60 | 0 | 556 | 1.000× |
| Generic rewrite, same map | `buffer_load_ubyte` | 64 | 6 | 655 | 0.228× |
| Structured rewrite, same map | `buffer_load_dwordx2` | 58 | 0 | 472 | 0.880× |

The structured layout's process-level 95% interval was [0.860, 0.901]. The largest
identity-control deviation was 0.35%; the same-pointer control deviation was
0.64%. Numerical checks passed. The restored load is eight bytes wide. Matrix
instruction forms also matched; static multiplicities differed with pipeline
structure, which is why raw histogram equality is not a blanket legality test.

This establishes a concrete compiler repair, **not a speedup over ordinary
Triton for this FP8 layout**. The structured realization is about 3.86× faster
than the generic rewrite, but still loses to ordinary storage. A deployment
run should retain ordinary unless independent tuning provides evidence for
another candidate. These short diagnostics do not replace the declared panel.

The complete sum-small workflow also finished capture, CPU search, numerical and
primitive validation, tuning, and independent evaluation. LAQS selected a split
layout with J=0 versus ordinary J=0.3. Both realizations passed the new guards:
the split used 11 registers versus 8 and 76 static instructions versus 69, with
the same load/store forms and no spills. Tuning retained ordinary. The held-out
split result was 0.8259×, with interval [0.8232, 0.8286]; identity-control deviation
was 2.30%, so this remains a diagnostic rather than a precision performance
claim. A separate same-map run found generic and structured sum lowering both
near 0.83×. The FP8 integration repair therefore does not imply that address
reconstruction was the limiting problem for sum.

Validation includes fourteen CPU tests, six MI300A GPU tests, and H100 offline
PTX/cubin compilation. The H100 packet-copy check retains `ld.global.v4.b32`
for both ordinary and transformed storage. This checks the NVIDIA proof-carrier
lowering; it does not replace numerical or performance testing on H100 hardware.
Compact measurements and fingerprints are in
[`packet-layout-validation.json`](packet-layout-validation.json).

## Positive result from conflicting storage preferences

For the 1024×1024 row/column operator, analytical selection lowered J from 2.00
to 1.08 and produced one changed split layout; the chunk optimum deduplicated
to the same physical map. All 24 offered storage/schedule combinations passed
numerical validation, and all automatic-layout schedules passed the primitive
checks. Each storage received six tuning schedules. Three fresh evaluation
processes then measured the frozen schedule choices, using 21 samples and 50
launches per graph:

| Storage | Chosen `(BLOCK_M, BLOCK_K)` | Paired speedup vs tuned ordinary | Process-level 95% interval |
| --- | --- | ---: | --- |
| Ordinary | (16, 128) | 1.000× | baseline |
| Column-major | (16, 128) | 0.988× | [0.979, 0.997] |
| Conventional 16×16 tile | (16, 128) | 1.473× | [1.429, 1.517] |
| LAQS split, automatic compiler pass | (4, 64) | 1.527× | [1.474, 1.582] |

This is a positive persistent-storage result with an equal schedule budget.
The LAQS map approximately matches the conventional tile; the intervals overlap,
so this is not evidence that it beats the best conventional tile. The fixed
ordinary anchor's largest identity deviation was 1.85%, and its same-pointer
deviation was 0.93%. Those controls concern the fixed anchor, rather than a
separate clone of the retuned ordinary winner.

Conversion changes the conclusion for transient inputs. Warm packing plus
allocation measured about 65 microseconds for the split map. Including packing
for one use gives only 0.069×; the median-cost estimate needs at least 41 uses
to amortize conversion. This estimate uses synchronous packing cost and median
kernel times, whereas the table uses geometric means of paired process ratios.
It does not include a producer/store pipeline.

The initial one-process smoke trial was retained alongside the confirming
three-process run. Both used the same declared candidates and schedule budget.
Full-panel measurements remain necessary before generalizing this result.

A separate analytical-only evaluation held the original `(4, 128)` schedule
fixed for both ordinary storage and the LAQS split. No timing selected its
layout or schedule. Three fresh processes measured **1.935×**, with interval
**[1.858, 2.014]**. Numerical and primitive checks passed; maximum identity and
same-pointer deviations were 1.91% and 0.45%. This isolates a storage-layout
benefit from schedule tuning. Packing still loses for one use (0.094×), with a
median-cost break-even estimate of 21 uses for this slower fixed baseline.

Measured deployment uses tuning data only. Its final policy requires the lower
speedup interval to exceed `(1 + minimum_gain) * (1 + d) / (1 - d)`, where `d` is
the maximum observed identity/same-pointer deviation; `d >= 1` abstains. This
scales the required gain with control variation rather than imposing a fixed
one-percent noise veto on arbitrarily large gains. That policy adjustment is
covered by CPU tests; the analytical-only result above does not depend on it.

Raw IR, ISA, captures and measurement records are archived under
[`packet-implementation-checks-2026-09-08`](../triton/experiments/results/packet-implementation-checks-2026-09-08/README.md).

## Running the new protocol

Run from the repository root on the respective cluster:

```bash
triton/packet_layout/submit-tuolumne.bash
triton/packet_layout/submit-matrix.bash
```

These build only the new packet plugin and submit the declared six-case panel.
Use `--cases sum--small row_column--small` for a smaller initial run. CPU search
is separate from GPU work; the Matrix request includes a GPU to avoid the
previously rejected CPU-only node configuration. The README documents direct
same-map diagnosis, analytical-only deployment and individual stages.

The row/column sizes are 1024 and 4096, corresponding to 4 MiB and 64 MiB input
matrices. The small exact graph/search took about 430 seconds in the development
check. Larger sizes increase CPU tracing costs as well as GPU work; the default
search job has a four-hour budget and must remain separate from short GPU tests.

The conventional comparison is a declared six-schedule budget with one fixed
16×16 tile baseline, not an exhaustive claim about the best conventional kernel.
Full-panel performance and H100 GPU validation require runs on the respective
clusters. Each agent GPU check used one task, one GPU and at most five minutes.
