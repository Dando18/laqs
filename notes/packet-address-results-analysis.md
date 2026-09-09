# Address-repair results: Matrix GEMM and Tuolumne sum

Analysis of the completed `packet-address-debug` runs, 2026-09-09.

The results expose two different problems. **On Matrix, fixing address generation barely helps the large layout; the selected layout increases measured memory-service work. On Tuolumne, the intended loop-address optimization did not apply at all.** The smaller Matrix representative removes almost all of the regression, but does not beat ordinary Triton. Tuolumne also has substantial variation between processes that the headline average obscures.

These findings revise the earlier diagnosis: address-generation overhead is not the main explanation for the large Matrix slowdown. The Tuolumne experiment has not yet tested that explanation successfully.

This analysis covers the two completed six-variant address comparisons, not a new run of all 29 cases. Experiment code and result files were left unchanged.

## Evidence and timing

Primary records:

- [Matrix GEMM report](../triton/experiments/results/packet-address-debug/matrix/gemm--asymmetric/expert/report.json), [counters](../triton/experiments/results/packet-address-debug/matrix/gemm--asymmetric/expert/profile.json), and [search scores](../triton/experiments/results/packet-address-debug/matrix/gemm--asymmetric/expert/search.json).
- [Tuolumne sum report](../triton/experiments/results/packet-address-debug/tuolumne/sum--small/expert/report.json) and [first timing process](../triton/experiments/results/packet-address-debug/tuolumne/sum--small/expert/evaluate/process-0.json). Processes 1 and 2 are adjacent files.

Speedup means ordinary runtime divided by candidate runtime; greater than one is better. These are geometric means of paired ratios across three fresh processes, excluding packing. Intervals are the harness's 95% intervals over process log-ratios.

| Variant | Matrix speedup | Tuolumne speedup |
|---|---:|---:|
| Ordinary | 1.000 | 1.000 |
| Large layout, legacy address code | 0.478 | 0.648 |
| Same large layout, repaired address code | 0.480 | 0.647 |
| Smaller equal-score layout, repaired address code | **0.986** | **0.736** |
| Identity layout, legacy address code | 0.979 | 0.975 |
| Identity layout, repaired address code | 1.002 | 0.980 |

All six variants passed numerical and primitive validation on both platforms. I verified the saved executable hashes against the actual files: each variant uses an identical binary across its three timing processes. All six Matrix profile records also report that their binary matches the timed binary. Thus different compilations between timing processes do not explain the variation.

The address-comparison harness deliberately labels the repaired **old large** layout as its analytical selection. That is an ablation choice, not evidence that the new smaller-representative tie-break failed to run. Inspect the `smaller` row separately.

## Matrix: equal LAQS scores conceal very different hardware costs

This GEMM has M=512, N=4096, K=4096, fp16 inputs, a 64×128×32 tile, four warps, and four stages. Only A is repacked. Ordinary takes approximately 38.4 µs, the repaired large layout 80.0 µs, and the smaller layout 38.9 µs.

The repair creates one physical pointer recurrence and reduces register usage from 114 to 96 for the large layout. Ordinary uses 117 registers. All variants have zero spills and 49,152 bytes of shared memory. The static memory and matrix instruction forms and counts are preserved: 24 `LDGSTS.E.BYPASS.128`, eight `LDSM.16.M88.4`, eight `STG.E.128`, and two `HGMMA.64x128x16.F32` instructions.

Nevertheless, repairing the same large layout improves runtime by only about 0.5%. Its paired speedup interval remains 0.476–0.485. The smaller layout is about twice as fast as the large layout but remains 1.4% slower than ordinary, with interval 0.983–0.988. The repaired identity control is compatible with parity: 0.994–1.010.

### Counters identify a concrete model mismatch

| Whole-kernel counter | Ordinary | Repaired large | Smaller |
|---|---:|---:|---:|
| L1TEX global-load requests | 786,432 | 786,432 | 786,432 |
| L1TEX global-load 32-byte sectors | 12,582,912 | 16,777,216 | 12,582,912 |
| L1TEX output load wavefronts | 2,097,152 | 2,097,152 | 1,310,720 |
| L2 read requests from TEX | 2,464,423 | 9,101,877 | 1,490,347 |
| L2 read sectors from TEX | 6,288,646 | 13,425,319 | 5,926,896 |
| DRAM bytes read | 33,893,888 | 28,368,384 | 30,787,072 |

The large layout incurs **33% more issued sectors, 3.69× the L2 read requests, and 2.13× the L2 read sectors**, even while reducing DRAM bytes. Legacy large-layout counters show the same sector inflation. Removing address instructions does not remove this cost.

The smaller layout does improve several service measures: unchanged issued sectors, 37.5% fewer output wavefronts, about 40% fewer L2 requests, and about 6% fewer L2 read sectors. Those improvements do not translate into faster execution here. They may affect work that is not limiting performance, or be offset by other costs. Existing counters do not distinguish compute, shared-memory, scheduling, address arithmetic, and pipeline stalls well enough to settle that remaining 1.4% difference.

Requests, sectors, and wavefronts count different things. A warp request can require multiple serialized service wavefronts; an address-set footprint is not itself a service count. See the [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

These counters come from one profiled dispatch per variant after 500 same-variant warmups, with application replay and no profiler cache flush. They are whole-kernel counters, not operand-resolved measurements. Their cache history differs from the interleaved timing panel. Use the graph timings for speedups; do not infer precise cache-traffic confidence intervals from this single profile.

### The missing distinction is visible in the address geometry

The saved graph predicts exactly **12,582,912** sectors for ordinary, large, and smaller layouts in `issue.g32.stream.load.32B`. Both changed layouts receive J=0; ordinary receives J=0.116667. The score improvement comes from larger-region locality terms, since ordinary already meets the modeled sector lower bound.

I checked the graph's first A events against the timed PTX ownership. Within a warp, lane `l` loads eight adjacent fp16 elements starting at:

```
i = row_base + l // 4
k = iteration_base + 8 * (l % 4)
```

The physical element offsets from the saved layout rows are:

```
ordinary: 4096*i + k
large:    4096*(k//8) + 8*i + k%8
smaller:  8192*(i//2) + 64*(k//32) + 32*(i%2) + k%32
```

All three have 16 distinct 32-byte sectors per A warp instruction. However, the large layout places the two halves of a sector in lanes separated by four, whereas ordinary and smaller pair adjacent lanes. Summing footprints over separate contiguous four-lane groups gives:

| A sectors per warp instruction | Ordinary | Large | Smaller |
|---|---:|---:|---:|
| Unique sectors over the entire warp | 16 | 16 | 16 |
| Sum over four-lane groups | 16 | 32 | 16 |
| Sum over eight-lane groups | 16 | 16 | 16 |

There are 256 CTAs × 128 K tiles × four warps = 131,072 warp-tile pairs, each with two A and four B copy instructions. Doubling only A's sector demand would add `131072 * 2 * 16 = 4,194,304` sectors—exactly the observed excess.

**This is a diagnostic witness, not proof that H100 universally services copies in four-lane groups.** Source-to-shared copy geometry, alignment, or internal replay could produce the same aggregate evidence. The current scope basis starts at eight lanes, and its source-address sets do not represent those distinctions. Merely preserving per-thread vector width and the final instruction opcode does not establish equivalent memory service.

The immediate research question is therefore why this `LDGSTS` mapping exceeds the warp-union sector count. Do not add a guessed four-lane weight to the objective and declare the model fixed. Existing data establish that the geometric prediction misses measured work for this layout; they do not establish a universal replacement hardware rule, partition-camping explanation, or TLB explanation.

## Tuolumne: the recurrence repair misses this loop form

The sum input is 4096×1024 fp32, with a 16×16 tile and two warps. It loops over 64 reduction tiles. Its address is rebuilt from the scalar loop induction variable; the loop carries the reduction accumulator, not the input pointer.

[`OffsetSSA::extendLoop`](../triton/packet_layout/PacketLayoutPlugin.cpp) looks for selected input pointers in the loop's initial arguments and rewrites their pointer updates. This sum loop has no such pointer argument, so the pass records **zero physical recurrences**. Its transformed integer address still contains masks and shifts inside the loop.

The saved `current.amdgcn` and `repaired.amdgcn` are textually identical after removing the two empty-assembly marker comments. Both have 88 static instructions and 14 registers; ordinary has 78 instructions and 11 registers. This makes the negligible timing change unsurprising. The `repaired` label means the mode was requested, not that strength reduction occurred.

This is a concrete coverage gap in the previous implementation. The clean fix is to recognize affine integer offsets derived from the loop induction variable and use the **same existing integer-carry proof** to create a physical offset recurrence. For this frozen launch, the reduction bound and step are known. There is no need for a separate kernel implementation or an approximate address transformation. Verify the optimization by checking the emitted loop, not just the mode label or successful compilation.

### Process variation prevents a clean layout conclusion

| Median kernel runtime, µs | Process 0 | Process 1 | Process 2 |
|---|---:|---:|---:|
| Ordinary | 20.569 | 15.076 | 15.094 |
| Independent-output identity control | 20.539 | 15.066 | 15.053 |
| Same-pointer control | 20.555 | 15.052 | 15.063 |
| Large, legacy | 26.083 | 25.630 | 25.767 |
| Large, repaired | 25.791 | 25.756 | 26.053 |
| Smaller | 17.157 | 26.046 | 26.277 |
| Smaller paired speedup | **1.199** | **0.579** | **0.574** |

The smaller layout's aggregate 0.736 speedup has a very wide 95% interval, approximately 0.258–2.103. Within-process samples are tight, and the controls follow ordinary closely. This is a process-dependent performance regime, not just noisy event timing. Identical binaries rule out recompilation differences.

Input allocation placement, cache behavior, or process/device state are plausible explanations, not proven causes. The harness rotates output allocations and timing order, but transformed inputs remain in separate allocations. The identity controls share ordinary's input and therefore cannot test the impact of transformed-input placement.

A small controlled repeat should reuse equal-sized input storage across variants, repack outside timing, balance variant order, and record addresses and device identity. Repeat across a few storage placements. Report paired per-process comparisons explicitly; the existing ratio of aggregate medians for `smaller_vs_repaired` conceals the reversal. Address records alone would not prove physical placement or NUMA locality.

## What to change next

1. **Complete the existing address optimization for affine loop-index expressions.** This is the narrowest concrete implementation fix. Re-run the same sum ablation and require emitted-code evidence that the inner-loop permutation has disappeared. Until then, Tuolumne cannot falsify the address-overhead hypothesis.
2. **Isolate Matrix copy-service behavior with one small diagnostic.** Hold shared-memory destination geometry, vector width, and launch geometry fixed; compare ordinary, large, and smaller source maps using the existing asynchronous copy primitive. A synchronous load/store control would help isolate the async-copy path. Measure sectors, requests, and service wavefronts before choosing a hardware model. If confirmed, express the actual cooperative-copy requirement as a legality constraint or correct service hyperedge, reusing LAQS rather than adding kernel-specific penalties.
3. **Retain the smaller-representative tie-break and ordinary baseline.** The tie-break already removes the catastrophic Matrix regression. It does not establish an improvement over ordinary. For practical execution, keep ordinary when validation timing shows no win, and report this fallback separately from the analytical top-1 result.
4. **Stabilize the Tuolumne comparison before another broad sweep.** Use the bounded allocation experiment above. More samples on the same fixed allocations will not resolve the observed process split.

These runs do not establish that the problems are simply too small, that Triton's defaults are universally optimal, or that the LAQS algebra is wrong. They show that minimizing the current footprint objective is insufficient to predict runtime: one selected map generates unmodeled service work, while another reduces measured memory work without improving the bottleneck. Larger inputs could change cache residency and expose different limits, but would not repair either the missed compiler transformation or the hardware-model mismatch. Resolve those two specific issues before spending more debug allocations on the full suite.
