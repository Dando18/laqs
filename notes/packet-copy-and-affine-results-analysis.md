# Matrix copy probe and Tuolumne affine-recurrence results

Analysis of the completed `copy-service-debug/matrix` and
`packet-affine-debug/tuolumne/sum--small` runs.

**Matrix now isolates a concrete hardware-model failure: the large layout
doubles sector demand specifically on the asynchronous source-to-shared copy
path. Tuolumne confirms that fixing address arithmetic does not recover the
lost performance: both transformed layouts lose at every matched input
placement, even though their repaired inner loops differ only in the address
increment.**

The smaller Matrix layout does improve the isolated copy kernel, modestly.
That improvement did not translate into a speedup in the previously measured
full GEMM. These results support retaining the exact LAQS machinery while
correcting its hardware constraints and being more selective about what the
objective can predict. They do not support another round of generic
address-code optimization as the main remedy.

Only this analysis file was added. Experiment code and saved results were not
changed, and no GPU jobs were submitted for this analysis.

## Inputs and verification

- [Matrix probe report](../triton/experiments/results/copy-service-debug/matrix/report.json),
  [SASS](../triton/experiments/results/copy-service-debug/matrix/codegen.sass),
  and [primitive checks](../triton/experiments/results/copy-service-debug/matrix/primitives.json).
- [Tuolumne report](../triton/experiments/results/packet-affine-debug/tuolumne/sum--small/expert/report.json)
  and [per-process/per-placement summary](../triton/experiments/results/packet-affine-debug/tuolumne/sum--small/expert/analysis.md).
- [Earlier full-GEMM address comparison](../triton/experiments/results/packet-address-debug/matrix/gemm--asymmetric/expert/report.json)
  and [previous diagnosis](packet-address-results-analysis.md).

The Matrix probe completed all 12 stages: unprofiled timing and profiling for
three layouts and two copy forms. All copied-value checks passed. I verified
all saved probe checkpoint output hashes. Async specializations each contain
two 128-bit `LDGSTS` instructions; synchronous specializations each contain
two 128-bit global loads and two 128-bit shared stores. Resources are identical
across layouts within each copy form: 22 registers for async, 28 for sync,
5,120 reported shared bytes, and zero stack/local allocation.

All six Tuolumne variants passed validation. Saved executable hashes match the
actual files and are identical across timing processes for each variant. All
three processes ran on `tuolumne1017`, GPU index 0, with the same recorded GPU
UUID. Within each placement, all variants used the same input address.

## Matrix: the async copy path reproduces the GEMM failure

Speedups below compare each layout with ordinary storage using the **same
copy form**. They are diagnostic ratios from one timing process per variant,
not replicated benchmark speedup estimates. Counter durations are excluded.

| Layout | Copy form | Unprofiled median, µs | Same-form speedup | Load requests | 32-byte load sectors | Output load wavefronts | L2 read requests |
|---|---|---:|---:|---:|---:|---:|---:|
| Ordinary | Async | 32.902 | 1.000× | 262,144 | 4,194,304 | 1,048,576 | 1,844,732 |
| Large | Async | 81.864 | **0.402×** | 262,144 | **8,388,608** | 1,048,576 | **8,169,803** |
| Smaller | Async | 31.629 | **1.040×** | 262,144 | 4,194,304 | 262,144 | 638,088 |
| Ordinary | Sync | 36.920 | 1.000× | 262,144 | 4,194,304 | 524,288 | 1,712,669 |
| Large | Sync | 42.889 | **0.861×** | 262,144 | **4,194,304** | 262,144 | 604,701 |
| Smaller | Sync | 33.577 | **1.100×** | 262,144 | 4,194,304 | 262,144 | 641,220 |

The warp-union model predicts 16 sectors/request for every layout. Five
variants achieve that count; **large async requires 32**. Switching the large
map to synchronous copies removes the entire sector excess and lowers its
runtime from 81.864 to 42.889 µs. The large async variant also presents 4.43×
as many L2 read requests as ordinary async.

The profiler records zero DRAM read bytes for five variants and only 256 bytes
for smaller sync. This is effectively a cache-resident experiment. The large
async slowdown therefore does not require HBM bandwidth pressure, B-matrix
traffic, tensor-core computation, or the original GEMM pipeline. It survives
in a much smaller kernel with constant address increments and the same
per-thread vector width.

This localizes the excess work to the interaction between the source layout
and the asynchronous copy/service path. It does **not** yet identify the exact
internal replay or grouping rule. A synchronous copy changes the instruction
path, register use and scheduling as well as the service behavior; its timing
is not a pure measurement of the cost of those extra sectors.

### Connection to the earlier full GEMM

The probe isolates the GEMM's A stream with the same source ownership and
shared-memory swizzle. It executes 256 CTAs × 128 tiles × four warps × two A
copy instructions = 262,144 warp requests. Its excess is:

```
8,388,608 - 4,194,304 = 4,194,304 sectors
```

The earlier full GEMM's excess was exactly the same:

```
16,777,216 - 12,582,912 = 4,194,304 sectors
```

This is substantially stronger evidence than the earlier whole-kernel
correlation: the isolated A copy reproduces all of the excess that the
previous analysis tentatively attributed to A.

In the large layout, two lanes contributing opposite 16-byte halves of a
32-byte source sector are separated by four lanes. Ordinary and smaller use
adjacent lane pairs. A four-lane footprint sum predicts the observed
doubling, but the six probe variants cannot uniquely establish that as the
hardware rule. Source/destination alignment and internal service splitting
remain alternative descriptions requiring further discrimination.

### Lower traffic is still not a sufficient runtime model

The smaller async map cuts output wavefronts by 75% and L2 requests by about
65%, yet improves this copy kernel by only 4%. Previously it achieved 0.986×
in the full GEMM. The copy probe therefore supplies a real local improvement,
not evidence for a full-GEMM gain.

The synchronous large map is an additional counterexample to a simplistic
request-count objective: it reduces L2 requests by about 65% and output
wavefronts by 50%, with unchanged sector count, yet runs about 16% longer than
ordinary sync. Work counts do not capture service latency, distribution or
the amount of work overlapped by execution. Existing counters do not identify
which of those explains that residual loss.

Nsight distinguishes requests, sectors and service wavefronts; these are not
interchangeable notions of a memory transaction. Its guidance also separates
memory bandwidth, memory-instruction throughput and other execution limits.
See the [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Tuolumne: the repair works, but address arithmetic was not the main cost

The actual 4096×1024 fp32 sum now has a proved physical recurrence in both
changed repaired variants. Register usage falls from 14 to 11, matching
ordinary; every variant has zero spills, 64 shared bytes, and the same
`buffer_load_dwordx2` / `buffer_store_dword` memory forms.

Most tellingly, the repaired large and smaller kernels have **the same 29
inner-loop instructions**, except for this immediate:

```
large:    v_add_u32_e32 v4, 0x40000, v4   // 262,144-byte advance
smaller:  v_add_u32_e32 v4, 0x80, v4      // 128-byte advance
```

Their initial address setup differs, but the repeated address reconstruction
is gone. The remaining large/smaller performance difference cannot be
explained by more permutation instructions in one inner loop.

| Variant | Reported speedup | Reported 95% interval | Recurrences |
|---|---:|---:|---:|
| Large, legacy | 0.643× | 0.413–1.001 | 0 |
| Large, repaired | **0.649×** | 0.417–1.010 | 1 |
| Smaller, repaired | **0.858×** | 0.676–1.090 | 1 |
| Identity, legacy | 0.978× | 0.969–0.988 | 0 |
| Identity, repaired | **0.9995×** | 0.99895–1.00014 | 1 |

Repairing the same large map improves the reported paired-process comparison
by only about 0.9%. The repaired identity control is essentially neutral.
Together with the generated code, this substantially weakens the hypothesis
that generic rewrite overhead explains the large losses. Layout-dependent
memory behavior is the leading remaining explanation, although AMD counters
are still needed to identify the hardware mechanism.

### Matched placements remove the apparent wins

The nine rows below are three placements within each of three processes, not
nine independent processes. Times are medians in µs.

| Process | Placement | Ordinary | Repaired large | Smaller | Smaller speedup |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 15.055 | 25.564 | 16.586 | 0.908× |
| 0 | 1 | 20.486 | 25.562 | 25.846 | 0.793× |
| 0 | 2 | 15.053 | 25.873 | 16.591 | 0.907× |
| 1 | 0 | 15.028 | 25.735 | 16.567 | 0.907× |
| 1 | 1 | 20.589 | 25.792 | 26.857 | 0.767× |
| 1 | 2 | 20.506 | 25.503 | 26.705 | 0.768× |
| 2 | 0 | 20.574 | 25.811 | 26.035 | 0.790× |
| 2 | 1 | 15.056 | 25.886 | 16.599 | 0.907× |
| 2 | 2 | 15.042 | 25.888 | 16.578 | 0.907× |

Both transformed layouts are slower in **every** matched placement. The
earlier isolated 1.20× observation does not survive this comparison.

Ordinary and smaller have two clear regimes: roughly 15/16.6 µs versus
20.5/26–27 µs. Large stays around 25.5–25.9 µs. In the five faster-ordinary
placements, smaller averages approximately 0.907×; in the four slower-ordinary
placements it averages 0.779×. Thus placement or process state changes the
size of the penalty, but is not hiding a winning layout in these observations.

The three input virtual addresses are identical across processes:
`0x153b60200000`, `0x153b5f000000`, and `0x153b5de00000`. The output address
also repeats. Yet, for example, placement 0 changes from the fast to the slow
regime in process 2. This rules out a deterministic explanation based solely
on the recorded virtual address bits. It does not rule out different physical
backing, cache state, or other device/process state. There is no direct
evidence here for a particular NUMA, cache-slice or memory-partition mechanism.

The headline summary pools samples across placements, takes a median per
process, and then combines process ratios. With two regimes, that emphasizes
whichever regime occupies more placements in a process. Taking ratios within
each matched placement first, then equally weighting placements and processes,
gives **0.671×** for repaired large and **0.848×** for smaller. Recomputing the
same log-ratio Student-t interval with that pairing order gives:

| Variant | Matched-placement speedup | Recomputed 95% interval |
|---|---:|---:|
| Large, legacy | 0.665× | 0.571–0.776 |
| Large, repaired | **0.671×** | **0.576–0.783** |
| Smaller, repaired | **0.848×** | **0.771–0.933** |
| Identity, repaired | 0.9998× | 0.99919–1.00034 |

For reproducibility: within each process, average the three log ratios
`log(ordinary_median / candidate_median)` from matched placements. Average
those three process values, and use a margin of
`4.303 * sample_standard_deviation(process_values) / sqrt(3)` before
exponentiating the interval endpoints. The three processes remain the
replication units; placements are not promoted to nine independent trials.

This removes much of the artificial process variation caused by pooling
different placement regimes. Under the same interval assumptions, both
changed layouts now have intervals entirely below one. The current report's
pooled-median estimand and these matched-placement estimates differ; the
saved report has not been rewritten. Future reporting should preserve this
pairing order. With only three processes on one GPU, these remain estimates
for this controlled experiment, not a statement about all MI300A executions.

Separate tuning selected ordinary storage before held-out evaluation. Its
reported 1.000× is a successful fallback, not an analytical LAQS speedup.

## Recommended next step

**For Matrix, strengthen the native copy contract before changing the
objective weights.** A simple conservative candidate rule is to preserve
which cooperative lanes share each native 32-byte source sector for this
16-byte async-copy form. The existing contract preserves each thread's vector
packet but allows that inter-lane sector grouping to change.

For this GEMM's fp16 A stream, retaining the native low four element-address
bits would preserve the ordinary adjacent-lane sector pair; the old large
map retains only three, while the smaller map retains five. This can be
expressed using the existing protected-bit machinery rather than a learned
penalty or an alternate kernel implementation. Validate the boundary with a
few neighboring templates before treating it as a generally sufficient
H100 rule. It is a conservative restriction, not a proof of globally optimal
copy layouts. Do not apply it to AMD buffer loads by analogy.

This would exclude the demonstrated pathological layout. It would **not**
make the already-tested smaller GEMM layout faster: the existing tie-break
already selects it, and its full-kernel result is still slightly below one.

**For Tuolumne, profile the existing ordinary/repaired/smaller kernels at the
matched placements.** The useful measurements are memory/cache requests and
hit/miss behavior, memory-instruction stalls, and achieved utilization. They
would discriminate additional memory work from slower service or an execution
limit. Repeating more timing samples or rewriting the now-identical inner
loops is unlikely to answer that question. Preserve the per-placement pairing
in the analysis.

I would retain the recurrence implementation, small-representative tie-break
and explicit ordinary fallback. I would postpone the next broad sweep until
the async contract and the AMD service behavior are understood.

The evidence now challenges the **sufficiency of the current cost model**,
not the exactness of the quotient computation. A geometric union can be
computed exactly and still undercount hardware service work; reducing that
work can also fail to improve the limiting part of a kernel. These benchmarks
are useful for exposing those failures. The runs neither establish that all
problems are too small nor that Triton is globally optimal. Larger inputs
could change cache residency and the bottleneck, but would not correct the
demonstrated async-sector mismatch.
