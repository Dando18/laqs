# Experiments 4–6: speedup and setup analysis

Analysis of the September 5–7, 2026 result snapshot, using the source at `0eae00c232cd830c52814ce8b4456330834dd7cc`.

The implementation follow-up, validation, and new commands are in [the rerun notes](experiments-4-6-rerun.md). The analysis below describes the original snapshot and its original implementation.

## Main findings

The disappointing results are real, but they combine implementation defects, an incomplete evaluation, and limitations of the objective. They do **not** yet constitute a clean test of whether RELAY can improve representative Triton workloads.

The highest-priority findings are:

1. **Experiment 5 does not reliably retain the ordinary baseline.** Its tie handling substitutes tiled row-major for whole-tensor row-major, and the baseline is absent from its candidate set. A CPU reproduction confirms the defect. The most severe observed regression—H100 int4 decode at **0.0295× speedup, or 33.9× longer runtime**—ties the baseline's score.
2. **The issue graph is not faithful to the lowered memory instructions.** It unconditionally splits accesses into one-register slices. H100 vector-add provides a particularly clear counterexample: RELAY predicts half as many 32-byte issue regions, while the transformed kernel actually incurs **twice the L1 sectors and four times the load requests**.
3. **Improving one downstream counter can make the kernel much slower.** MI300A FP8 GEMM cuts L1-to-L2 read requests by 50.7%, but raises L1 cache accesses by 9.8× and runs at 0.277×. H100 sum cuts L1-to-L2 requests while moving substantially more data from HBM.
4. **The four-bit inner bound in Experiment 6 often makes arbitrary inner maps invisible to the score.** For FP32 and narrower operands on MI300A, and FP16 and narrower operands on H100, changing only that inner map cannot improve any positively weighted quotient component in the present profiles.
5. **Coverage and measurement need repair before a broad claim is possible.** Experiments 4–5 retain only 13 common configurations from 10 operator families; Experiment 6 retains only six common configurations from four families. Many missing cases time out or exceed frontend bounds. Configurations are frozen within each comparison, but are retuned across searches and profiles.

My recommendation is to repair baseline handling and lowering fidelity first, then evaluate a conservative, compiler-aware selector. Increasing search size or retuning tau against these evaluation results would not address the main problems.

## 1. Data and interpretation

I read all **432** per-case reports under:

```text
triton/experiments/results/tau-profiles/{expert,l1_to_l2,speedup}/
    experiment-{4,5,6}/{matrix,tuolumne}/*--*/report.json
```

These contain 204 completed measurements and 228 explicit exclusions. The intended panel has 522 cells: 29 configurations × three experiments × three profiles × two devices. Thus **90 intended cells have no report**. Missing reports are not counted as exclusions or successful no-change results. Older results directly under `results/experiment-*` are excluded from this analysis.

All 432 recorded tau maps match their currently named profiles. Every completed report's timing-process records match its corresponding `timings/process-*.json` files. All 204 completed cases report successful numerical comparison. Each contains the specified three timing processes, with 10 warm-ups and 21 samples of 50 launches, and three profiler runs per layout with 20 retained dispatches each.

Speedup below means `baseline_median_ms / selected_median_ms`; less than one is slower. Geometric means use one speedup per completed cell. “Changed” means `transformed_array_count > 0`. The three profiles and two sizes of an operator are not independent benchmark families.

This investigation used existing GPU measurements and CPU checks; it did not run new GPU experiments. Source-based explanations of generated code are distinguished below from mechanisms directly established by counters. The reports lack source and binary fingerprints, so matching tau maps does not prove the exact compiler/plugin revision used by every run.

### Completed results, including platform-specific cases

| Experiment | Tau | H100 complete / changed | H100 geometric mean | MI300A complete / changed | MI300A geometric mean |
|---|---|---:|---:|---:|---:|
| 4: whole tensor | expert | 13 / 3 | 0.8759× | 14 / 4 | 0.8294× |
| 4 | l1_to_l2 | 13 / 3 | 0.8739× | 14 / 3 | 0.8732× |
| 4 | speedup | 13 / 0 | 1.0033× | 14 / 0 | 1.0016× |
| 5: tiles | expert | 13 / 3 | 0.8841× | 14 / 7 | 0.8020× |
| 5 | l1_to_l2 | 13 / 3 | 0.8833× | 14 / 7 | 0.7988× |
| 5 | speedup | 13 / 3 | 0.7520× | 14 / 7 | 0.7950× |
| 6: bounded G_OC | expert | 9 / 7 | 0.7676× | 6 / 0 | 1.0026× |
| 6 | l1_to_l2 | 9 / 0 | 1.0065× | 6 / 0 | 1.0049× |
| 6 | speedup | 6 / 0 | 1.0032× | 6 / 0 | 1.0022× |

Of the 204 completed cells, **154 leave all layouts unchanged**. Of the 50 changed cells:

- 48 are slower by the recorded median.
- 40 take more than 5% longer, i.e. speedup < `1/1.05`.
- None achieves >1.05×.
- The only two improvements are H100 `sum--large` in Experiment 5, at 1.0209× and 1.0214× under expert and l1_to_l2.

The unchanged-layout measurements themselves range from **0.9871–1.0424× on H100** and **0.9935–1.0147× on MI300A**. Their geometric means are 1.0042× and 1.0022×. Consequently, apparent gains around a percent, especially in no-change cells, are not evidence of layout improvement. The two changed-layout wins need more careful measurement before being claimed.

### What the existing cross-platform plots show

[analyze-search.py](../triton/experiments/analyze-search.py) plots only cases completed on both platforms for the same experiment and tau. This removes the MI300A FP8 regressions from Experiments 4–5 and the H100-only completed jagged cases from Experiment 6.

| Experiment | Tau | Common configurations | H100 geometric mean | MI300A geometric mean |
|---|---|---:|---:|---:|
| 4 | expert | 13 | 0.8759× | 0.9023× |
| 4 | l1_to_l2 | 13 | 0.8739× | 0.9600× |
| 4 | speedup | 13 | 1.0033× | 1.0016× |
| 5 | expert | 13 | 0.8841× | 0.8556× |
| 5 | l1_to_l2 | 13 | 0.8833× | 0.8526× |
| 5 | speedup | 13 | 0.7520× | 0.8500× |
| 6 | expert | 6 | 0.9690× | 1.0026× |
| 6 | l1_to_l2 | 6 | 1.0019× | 1.0049× |
| 6 | speedup | 6 | 1.0032× | 1.0022× |

In particular, Experiment 6's near-one cross-platform bars do not demonstrate a successful general solver. They describe four vector cases, small softmax, and small layer norm. Report both common-panel and device-specific outcomes, and show missing cases alongside them.

## 2. Confirmed defect: tile search loses the ordinary baseline

In [_canonical_search](../triton/experiments/search_algorithms.py), two different layouts are involved:

- `baseline[matrix.name] = row_major_layout(matrix)`: ordinary whole-tensor row-major.
- `tile_baseline = _row_major_tile(matrix, tile)`: row-major inside tiles, with tiles also ordered row-major.

They generally have different physical address maps. For example, in an 8×16 matrix partitioned into 2×4 tiles, logical element `(0,4)` has ordinary offset 4 but tiled offset 8.

The current code does:

```python
if abs(candidate_score.hardware_area - baseline_score.hardware_area) <= 1e-12:
    selected = tile_baseline
    candidate_score = baseline_score
```

This substitutes a potentially different mapping and temporarily associates it with the ordinary baseline's score. Moreover, `candidates` contains only optimized tile hypotheses: there is no general “keep ordinary layout” candidate when every tile is worse. The final joint score is recomputed, so the final report is not necessarily falsely scored; the defect is in selection and intermediate ranking.

I reproduced the tie failure with the existing CPU solver on an 8×16 matrix, a 2×4 natural footprint, and a layout-independent one-point objective. Whole-tensor search retains row-major; tile search transforms it despite every score being exactly zero. Appendix A contains the reproduction.

### Evidence in the real results

The following compare Experiment 5 with the unchanged ordinary layout in Experiment 4 under `speedup`, using only cases with identical recorded frozen configurations:

| Device / case | Ordinary J_area | Experiment 5 J_area | Experiment 5 speedup |
|---|---:|---:|---:|
| H100 int4 decode | 0 | 0 | 0.0295× |
| H100 sum, small | 6.975 | 6.975 | 0.8217× |
| H100 sum, large | 6.975 | 6.975 | 0.9746× |
| MI300A FP8, small | 2 | 4 | 0.3333× |
| MI300A int4 decode | 0.6667 | 0.6667 | 0.4022× |
| MI300A jagged mean, small | 2.8376 | 2.8376 | 0.9514× |
| MI300A jagged sum, small | 2.8376 | 2.8376 | 0.9568× |
| MI300A sum, large | 15 | 15 | 0.7281× |

These are score ties or a worse score, not analytically justified improvements. All ten transformed `speedup` Experiment 5 cells use row-major words inside their selected tiles. The two omitted cases have different recorded configurations, so their baseline scores should be recomputed rather than borrowed across experiments.

Sources: [H100 int4 E4](../triton/experiments/results/tau-profiles/speedup/experiment-4/matrix/int4_gemm--decode/report.json), [H100 int4 E5](../triton/experiments/results/tau-profiles/speedup/experiment-5/matrix/int4_gemm--decode/report.json), [MI300A FP8 E4](../triton/experiments/results/tau-profiles/speedup/experiment-4/tuolumne/fp8_gemm--small/report.json), [MI300A FP8 E5](../triton/experiments/results/tau-profiles/speedup/experiment-5/tuolumne/fp8_gemm--small/report.json).

**Fix:** explicitly search `{ordinary baseline} ∪ {tile candidates}` for deployment. Prefer the actual ordinary mapping on score ties, and recompute the score of any replacement. Record whether the result is a grammar optimum or a baseline fallback. Assert that the deployed score never exceeds the ordinary score and that equal-score selection retains the ordinary mapping. This handles both the erroneous assignment and cases where the baseline is outside the restricted tile grammar.

This should eliminate several severe regressions. It will mostly turn them into no-change results; it does not by itself create speedups.

There is a separate tile-hypothesis problem. `natural_tile_hypotheses` takes bounding boxes of `problem.events`, which the frontend has already split by wave and register slice. These can be much smaller than the frozen kernel's full access footprint. MI300A FP8's selected A tile is 16×32 although its kernel block loads 64×32; H100 int4's B tile is 4×128 although the logical packed-B block spans 128×128 for `BLOCK_SIZE_K=256`. Derive the primary tile from the parent memory operation across its register and wave owners, with masks and alignment accounted for. Smaller service footprints can be additional hypotheses, but should be identified as such.

## 3. The graph optimizes register slices, while the GPU executes lowered instructions

The manifest plugin's [serializeIssue](../triton/automatic_frontend/AccessManifestPlugin.cpp) unconditionally emits:

```text
partition = conservative_register_slices
register_slice_elements = 1
register_slice_size = 1
```

It also records contiguity and alignment information, but [_register_slices](../relay/triton_frontend.py) follows the width-one field. Thus an “issue” edge is one abstract register position across lanes. It is not necessarily one machine load instruction.

The distinction matters when several adjacent per-thread elements become a vector load. The compiler's actual instruction partition may also change after the address rewrite. Extracting exact logical accesses from `LinearLayout` establishes neither the hardware issue boundaries nor their invariance under a new layout.

### H100 vector-add is a direct counterexample

Compare the expert Experiment 4 ordinary mapping with the expert Experiment 6 choice for `vector_add--small`. The selected low four address rows are `[1,4,8,2]`, instead of `[1,2,4,8]`.

| Quantity | Ordinary | Selected |
|---|---:|---:|
| Predicted issue Q at 32 B | 262,144 | 131,072 |
| Predicted J_area | 2.4 | 1.4 |
| Measured global-load requests | 4,096 | 16,384 |
| Measured L1 global-load sectors | 65,536 | 131,072 |
| Measured L2 read sectors from TEX | 65,536 | 65,536 |
| Measured L1-to-L2 read requests | 16,384 | 24,751.5 |
| Unprofiled batch time per launch | 9.126 μs | 9.931 μs |

The ordinary graph already overcounts measured L1 sectors by 4×, whereas the transformed graph matches their count in this example. This is consistent with counting scalar register slices before a four-element vector load, then losing that vectorization after rewriting. The request increase strongly supports that explanation; saved ISA would confirm the exact instruction widths.

The reported `sectors_per_request` even improves from 16 to 8 while total sectors double. A per-request ratio alone can therefore give the wrong optimization signal.

NVIDIA documents a global-memory instruction for a warp as generating an L1TEX request, with the request then accessing cache lines and sectors. This makes lowered instruction boundaries essential to interpreting these counters. [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities)

Sources: [ordinary vector score](../triton/experiments/results/tau-profiles/expert/experiment-4/matrix/vector_add--small/report.json), [transformed vector result](../triton/experiments/results/tau-profiles/expert/experiment-6/matrix/vector_add--small/report.json).

### What to change

First introduce a conservative admissibility rule: preserve each original vector access's contiguity and alignment, and reject candidates that increase lowered memory-instruction count or introduce spills without a separately justified benefit. Merely placing the current issue component behind a hard constraint is insufficient: the example shows that this component itself is wrong.

Then construct issue edges from the logical elements serviced together by the lowered load, including vector width, predicates, lane ownership, and target-specific service partition. Preserve the logical-access graph separately from this lowering-dependent graph.

There are two practical implementation paths:

- **Restrict the grammar to preserve the lowering contract.** This makes one baseline issue graph reusable and keeps analytical selection cheap.
- **Compile a small analytical shortlist and inspect each realization.** Reconstruct its issue groups and resource requirements before final scoring. Any timing-based choice should be explicitly labeled measured selection.

The [layout rewrite](../triton/layout_rewrite/LayoutRewritePlugin.cpp) executes after coalescing, but later passes still optimize dot operands, pipeline loads, and lower memory operations. Frozen launch parameters do not freeze those effects. Record final load widths/counts, register counts, spills, shared memory, and dot/matrix-instruction forms for baseline and selected kernels.

## 4. Counter reductions are not interchangeable with performance

### MI300A FP8: less downstream demand, much more first-level work

For expert Experiment 4 `fp8_gemm--small`:

- Runtime increases from 18.111 μs to 65.339 μs: **0.2772×**.
- L1-to-L2 read requests decrease **50.7%**.
- HBM read bytes decrease **24.8%**.
- `TCP_TOTAL_READ` increases **8×**.
- `TCP_TOTAL_CACHE_ACCESSES` increases **9.8×**.

The independent profiler's durations also show a substantial regression: 15.740 μs to 53.341 μs. The three timing processes give 0.2752×, 0.2789×, and 0.2809×, so this is not a marginal median fluctuation.

AMD distinguishes vector-L1 read/cache accesses from requests sent onward to L2. Improving the latter does not imply less work at the former. [AMD MI300/MI200 counter definitions](https://rocmdocs.amd.com/en/docs-7.2.4/conceptual/gpu-arch/mi300-mi200-performance-counters.html#texture-cache-per-pipe-counters)

This supports a first-level service/realization problem. Scalarization, additional address work, and changes to pipelining or register pressure are candidates; their individual contributions cannot be separated from these reports alone. [FP8 report](../triton/experiments/results/tau-profiles/expert/experiment-4/tuolumne/fp8_gemm--small/report.json)

### H100 sum: fewer requests, almost unchanged sectors, more HBM traffic

Expert Experiment 4 `sum--small` reaches J_area = 0, but runs at **0.8794×**:

- L1-to-L2 read requests decrease **43.0%**.
- L2 read sectors from TEX are effectively unchanged: 524,288 versus 524,330.
- HBM read bytes increase from 1,433,984 to 16,786,688: **11.7×**.
- L2 read misses increase **4.15×**.

Some of the request reduction is therefore better grouping into requests, not less downstream byte traffic. The HBM change indicates substantially different cache behavior under the profiling setup. Set conflicts, altered reuse distance, and placement are plausible explanations, but none is established individually.

The selected whole-tensor word puts four column bits first, then all row bits, then the remaining column bits. This changes the global traversal substantially, even though each modeled short window packs well. [H100 sum report](../triton/experiments/results/tau-profiles/expert/experiment-4/matrix/sum--small/report.json)

### H100 int4: a severe regression that a baseline fallback would prevent

The `speedup` Experiment 5 decode case increases unprofiled time from **78.021 μs to 2,647.607 μs**. Its three process speedups round to 0.0295×, 0.0295×, and 0.0294×. Under profiling:

- L1 sectors increase **8.5×**.
- Global-load requests increase **3.33×**.
- L1-to-L2 read requests increase **92.29×**.
- HBM read bytes nevertheless decrease **64.3%**.

This is genuine measured memory-system disruption, alongside any added arithmetic. A smaller HBM-byte count is not sufficient to rescue it. Its zero J_area under the lane-window profile only means the selected lane-window packing bound is attained; it does not mean the kernel is efficient. [H100 int4 report](../triton/experiments/results/tau-profiles/speedup/experiment-5/matrix/int4_gemm--decode/report.json)

### A more useful objective

Keep quotient counts as locality features, but add explicit admission constraints for first-level service, lowering, and resource use. Apply the objective only after those checks. For example:

```text
ordinary layout is always admissible
preserve required vector access structure
bound actual issue-sector and instruction inflation
reject spills and unacceptable register/occupancy changes
require a meaningful analytical benefit to justify address rewriting
```

A later performance model can distinguish memory-issue throughput, L1/L2/HBM traffic, arithmetic, and occupancy. One global weighted sum of overlapping locality windows cannot reliably account for which resource limits each kernel. A guard must also distinguish structural no-regression guarantees from empirical runtime evidence; equal locality does not guarantee equal runtime.

## 5. Why the speedup profile mostly does nothing

The speedup profile minimizes lane-window excess: 100% `lane_window.t16...64B` on MI300A; 97.5% `lane_window.t16...32B` plus 2.5% `lane_window.t4...32B` on H100. It has no positive issue-coalescing weight.

In whole-tensor search it selects the ordinary mapping for **all 27 completed device/case combinations**. Several ordinary cases already attain zero on this objective, and others have no preferred canonical alternative. This is successful abstention in these measurements, but it supplies no evidence of useful speedup selection.

There are also three calibration limitations:

1. **Profiler time is the training target.** [_speedup_groups](../triton/experiments/tune_tau.py) reads `candidate["counters"]["steady_state"]["duration_ns"]`. Experiments 4–6 report separate unprofiled Python-enqueued event batches. These are different measurement regimes.
2. **Finite pilot panels do not validate the exact-search optimum.** Exact search can find previously unmeasured layouts that exploit omissions in the model. A no-regression training panel is not an out-of-sample guarantee.
3. **The realization path changes.** Pilot [stage1_kernels.py](../triton/stage1_kernels.py) constructs physical offsets directly in Triton source, where they participate in compilation from the start. Experiments 4–6 insert a generic pointer rewrite after coalescing. A fit to one path need not transfer to the other.

Retune only after fixing these mismatches. Use the same realization and unprofiled timing path for pilots and evaluation. Validate across held-out pilot operator families, sizes, and access patterns; include layouts obtained by the actual exact selector in pilot validation. Keep the current TritonBench results labeled as development evidence once they influence algorithm changes, and preserve an untouched final evaluation set.

## 6. Experiment 6 has both an expressiveness limit and avoidable solver cost

### Four inner bits often cannot affect any active component

Let the element width be e bytes and a scored region have size b bytes. The quotient ignores the lowest d = log2(b/e) element-address bits. Holding the outer word fixed, replacing an invertible map on the lowest p physical bits changes no b-byte region identifier whenever p ≤ d:

```text
floor(offset_A(x) / 2^d) = floor(offset_B(x) / 2^d)
                    when A and B differ only within p ≤ d low bits.
```

This is an invariance of the objective for every edge, not a hypothesis about the GPU.

The current p ≤ 4 bound therefore has these consequences:

| Main operand type / device | Smallest positively weighted region | d | Can the four-bit arbitrary inner map improve the score over its canonical inner representative, with outer word fixed? |
|---|---:|---:|---|
| FP32 / MI300A | 64 B | 4 | No |
| FP16 or BF16 / MI300A | 64 B | 5 | No |
| FP16 or BF16 / H100 | 32 B | 4 | No |
| FP8 or packed int8 / H100 | 32 B | 5 | No |
| FP32 / H100 expert or speedup | 32 B | 3 | Potentially, at p = 4 |
| FP32 / H100 l1_to_l2 | 128 B | 5 | No |

Wider index/offset operands can be exceptions. The outer canonical suffix remains searchable, so Experiment 6 can still reproduce canonical gains. But arbitrary inner GL maps cannot provide a strict score advantage for most primary operands at this bound. The H100 FP32 vector cases are exactly where the bound can change a scored region—and they expose the issue-model error.

Do not simply increase the exhaustive GL bound. |GL(4,2)| = 20,160, while |GL(5,2)| = 9,999,360 for each tile hypothesis. Prefer structured transformations that preserve vector bits and selectively mix higher address bits, or search score-relevant subspaces/flags with a cheap realizable representative. Larger search should follow evidence of an opportunity the existing grammar cannot express.

### Scalar selection currently pays for a frontier

Experiment 6 calls [simple_solve](../relay/simple_solver.py), builds raw and final Pareto frontiers, retains canonical score ties, and only then chooses the minimum J_area member. This is consistent with obtaining a scalar minimum, but it retains machinery the stated experiment does not need.

For H100 small softmax:

| Tau | Search time | Retained frontier members | Deployed mapping |
|---|---:|---:|---|
| expert | 19.59 s | 1,820 | ordinary |
| l1_to_l2 | 6.43 s | 455 | ordinary |
| speedup | 921.45 s | 18,564 | ordinary |

The speedup case spends over 15 minutes searching to keep the baseline. Small layer norm similarly retains 18,564 members and takes 945.51 s. [Softmax report](../triton/experiments/results/tau-profiles/speedup/experiment-6/matrix/softmax--small/report.json)

With the current fixed profiles, common normalization, and no cross-allocation resource terms, J_area is additive across target arrays:

```text
J_area(A_1, ..., A_n) = constant + sum_a J_a(A_a).
```

Consequently, independent scalar minima can be joined exactly. Experiments 4–5 already exploit this. Implement a scalar G_OC solver that retains only states necessary for the objective and deterministic tie-breaking, explicitly preserves the baseline, and collapses score-equivalent inner mappings. Full per-array and joint frontiers are unnecessary for this experiment. Revisit separability if adding allocation-interaction or cache-conflict terms.

## 7. The setup removes many of the workloads with interesting reuse

For every tau, Experiments 4–5 have:

- H100: 13 complete, 13 excluded, three missing, out of 29.
- MI300A: 14 complete, 12 excluded, three missing, out of 29.

Their common panel comprises vector add, vector exp, softmax, sum, layer norm, int4 GEMM, gather GEMV, jagged sum, jagged mean, and jagged softmax. Several are closely related streaming/reduction patterns. The 12-family minimum in [final-experiments.md](final-experiments.md) is not met, and the reserve operators are absent from [CASES](../triton/experiments/tritonbench_cases.py).

The missing three cases in each Experiment 4–5 group are the large jagged configurations. Matrix logs explicitly show time-limit cancellations. For Experiment 6, H100 has another six to nine missing cells per tau; MI300A has eleven. Several Tuolumne logs are empty, so their termination cause cannot be determined from the saved artifacts.

The explicit exclusions are mostly:

- GEMM, bf16×int16 GEMM, FP8 large, int4 prefill, attention, and large reductions: `launch dynamic event count exceeds exact bound 1048576`.
- H100 FP8 small is also excluded, while MI300A completes it.
- Gather large: an event bound or the 65,536-context limit.
- Both dropout sizes: `unsupported.pointer_provenance: pointer base flows through tt.bitcast`.

Examples: [GEMM exclusion](../triton/experiments/results/tau-profiles/expert/experiment-4/matrix/gemm--square/report.json), [dropout exclusion](../triton/experiments/results/tau-profiles/expert/experiment-4/tuolumne/low_mem_dropout--small/report.json), [large-jagged timeout](../triton/experiments/results/tau-profiles/expert/experiment-4/matrix/logs/relay-e4-jagged_sum--large-h100-expert-325444.log).

### Fix the representation before raising limits

The early [_translation_launch_classes](../relay/triton_frontend.py) shortcut accepts a flat body, one-dimensional allocations, and a narrow affine program-ID form. Loops and multidimensional tensors fall outside it. Later trace compression can merge repetitions, but the frontend has already materialized events and checked its bound by then.

Extend exact symbolic compression to regular tiled loops and multidimensional launches. Build translated edge/window classes with multiplicities directly, including boundary and phase classes, instead of first expanding the full launch. Raising limits alone will increase cost without addressing the structural scaling problem.

Persist the manifest, frozen configuration, and compressed graph once per shape/device. Reuse them for the three grammars and tau profiles. Pure CPU search should run separately from scarce GPU allocations. Long jobs should checkpoint their stage and publish a status record on failure.

Treat dropout pointer bitcasts explicitly, preserving pointee-size and byte-offset semantics in both the frontend and rewriter. The rewriter's provenance allowlist also lacks `tt.bitcast`; changing only the frontend would be incomplete.

Run the planned preflight and reserves before the next full submission. Preserve excluded cases in the coverage report. Do not silently shrink difficult shapes until the headline improves; label diagnostic shapes and explain any revised workload panel.

### The scope names also overstate modeled reuse

The notes describe expert tau as representing workgroup-to-workgroup reuse. In [access_scopes.py](../relay/access_scopes.py), workgroup-step/window keys retain a particular workgroup and sequence. They do not combine accesses from different workgroups.

Furthermore, the automatic frontend builds concrete sequences per wave, and the scope keys include `sequence.name`. In this path, these scopes do not reconstruct a full multi-wave workgroup union either. Their “step” comes from the serial register-slice event order. This makes the names a poor description of a hardware-wide reuse model.

Rename/document the current scopes accurately, and construct explicit within-workgroup unions where intended. For inter-workgroup locality, represent overlap between tiles separately from assumptions about their scheduling or cache residency. Avoid assuming a deterministic execution order among independent workgroups.

## 8. Timing, freezing, and reproducibility gaps

### Event batches can include host-enqueue gaps

[timing_worker](../triton/experiments/run-search.py) records a GPU start event, enqueues 50 calls through `FrozenLaunch.run()` in Python, and records the end event. If the GPU consumes work faster than Python supplies it, the event interval includes idle time between dispatches. The selected path also adds a Triton specialization-hook lookup during dispatch.

There is substantial evidence of a floor in the small-kernel measurements:

- MI300A small vector add: about 14.33 μs per unprofiled batch launch, versus 2.72 μs in the separate profiler.
- H100 small vector add: about 8.97 μs versus 3.09 μs.
- MI300A Experiment 5 jagged sum under speedup: the batch ratio is 0.9568×, while profiler dispatch durations are 6.76 versus 13.46 μs, a 0.5022× ratio.

Profiler durations are **diagnostics, not replacement speedups**: replay, clocks, caches, and instrumentation differ. These discrepancies justify measuring dispatch gaps directly and using a graph replay or equivalent low-overhead launch path for kernel-only throughput. PyTorch documents how CPU launch overhead produces gaps between short kernels and how graph replay reduces them. [PyTorch CUDA Graphs explanation](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

Keep paired alternating order and fresh timing processes. Add identity-versus-identity controls, process-level ratios, and uncertainty estimates that respect the process/sample hierarchy. The 50 launches inside a batch are not 50 independent samples. Declare warm-cache repeated execution explicitly and add cold/rotating-buffer measurements only when relevant to deployment.

### Configurations are frozen only within a cell

Every orchestrator calls `case.factory()` and `analyze_launch(...)` again, allowing the native autotuner to run for each grammar/profile. The subsequent baseline and selected workers share that cell's selected configuration, so the within-cell comparison is controlled. The intended one-time freezing across all searches is not implemented.

This is visible in recorded results:

- MI300A small sum uses `BLOCK_SIZE_NON_REDUCE_DIM=4, num_warps=2` for expert E4, but `16,4` for l1_to_l2 E4 and `16,2` for speedup E5.
- MI300A large sum changes block size and/or warp count across profiles.
- Both platforms' jagged kernels change `num_stages` between 2 and 4.

Freeze and serialize one native configuration per device/shape before branching into searches. Record resolved defaults as well as autotuner-selected values. The present variation confounds comparisons between tau profiles and grammars, although it does not explain the large within-cell regressions.

### Artifact and oracle improvements

The run reports retain analysis and search time, but not separate packing, compilation, autotuning, and validation costs as the plan requests. They record selected scores but not a directly computed ordinary score/vector or a lowering comparison. Resume checks validate tau identity, not the complete source/configuration/selection fingerprint.

Record these artifacts, input seeds/content hashes, true/envelope shapes, source and plugin hashes, GPU identity, and timing methodology. Bind timing/profiler checkpoints to the selection fingerprint. Distinguish unsupported, search failure, realization failure, timeout, and missing submission. Have the aggregator iterate the declared 29 cases so absent reports cannot disappear silently.

The oracle currently compares against ordinary Triton with one broad `rtol=1e-2, atol=5e-2, equal_nan=True` rule. Add finite-output checks and operation-appropriate reference validation, especially for reduced-precision GEMMs. The current allclose success is evidence that transformed outputs match the ordinary implementation under that tolerance; it does not establish a full operator oracle.

## 9. Address realization and workload details worth changing

The generic [pointer rewrite](../triton/layout_rewrite/LayoutRewritePlugin.cpp) recovers an element offset using 64-bit pointer arithmetic, then applies the bit map. Its power-of-two path uses bit runs, but non-power-of-two shapes go through coordinate decomposition with division/remainder.

This matters in the existing panel, even though a later experiment is supposed to study non-power-of-two behavior:

- Int4 packed B has true shape `(4096,1280)` and envelope `(4096,2048)`: **1.6× allocation expansion**.
- Small jagged values have true shape `(16723,256)` and envelope `(32768,256)`: **1.959× expansion**.

The expansion is allocation capacity, not a prediction that every padded byte is read. It can nevertheless change memory placement and requires a different realization path. Jagged row-stride division by 256 may simplify cheaply; int4's 1280 stride deserves particular ISA inspection. These effects must not be conflated with the mathematical quality of a layout.

Prefer rewriting symbolic logical indices while their structure is still available. Keep invariant terms outside loops, use safe narrower offset arithmetic where valid, preserve compiler-provable alignment, and specialize common canonical bit runs. Consider native row pitch with an inner-block transform, explicit divisible tiles plus tails, or a storage ABI that makes padded strides cheap. Measure the address path rather than using descriptor complexity as the sole proxy.

Also audit the workload description against the actual factories. For example, `_fp8` launches the tutorial FP8 matmul without tensor-scale arguments, rather than the tensor-wise-scaling operator described in the plan. The `sum` kernel loops through the reduction in blocks of at most 16 elements, often for hundreds of iterations. Such a baseline is a legitimate selected implementation, but benefits or regressions relative to it do not establish superiority over other row-reduction implementations. Keep any stronger implementation baseline as a separately labeled comparison.

Packing remains excluded from these speedups, as intended. For practical use, report amortization:

```text
T_packed(R) = T_pack + R * T_selected
R_break_even > T_pack / (T_baseline - T_selected), if selected is faster.
```

For an offline layout search, report its deployment cost separately. For dynamic layouts, include search/compilation in the relevant amortization budget. Current regressions have no positive packing break-even point.

## 10. Recommended next steps and research direction

### First: make the experiment trustworthy

1. Fix the ordinary-layout fallback and score consistency in Experiment 5. Add the small CPU regression checks in Appendix A and checks on actual selected mappings.
2. Persist one configuration and trace per case/device, with artifact fingerprints; cache graph construction across tau and grammar runs.
3. Correct the timing path and identity controls. Retain separate profiler measurements.
4. Report all 29 cases, failure stages, matched panels, and device-specific outcomes.

Success at this stage means avoiding unjustified regressions and making comparisons interpretable. Expect many repaired results to become exactly the baseline.

### Second: test the strongest mechanism before another broad sweep

Use a small diagnostic panel:

| Diagnostic | What to compare | What it resolves |
|---|---|---|
| H100 vector add/exp, small and large | ordinary, current E6 map, vector-preserving candidate | Whether corrected issue grouping and preservation prevent the false coalescing gain |
| H100 int4 decode | ordinary, current E5 map, baseline-fallback result | Whether the selector deploys the ordinary mapping on ties; instruction/address costs of the rejected map |
| MI300A FP8 small | ordinary, current selected map, vector-preserving realization | Why first-level work rises despite lower downstream demand |
| Sum on both devices | same frozen config, whole/tile map, ordinary | Request versus sector versus HBM behavior; sensitivity to warm-cache execution |
| MI300A jagged sum/softmax | ordinary, tiled map, low-overhead timing | Whether the current batch timing hides device regressions |

Save lowered IR/ISA and resource metadata for each. Repeat only this bounded set after each relevant implementation change. A future GPU diagnostic command should use the repository's platform environment and the cluster's permitted one-task, one-GPU allocation; full reruns should remain user-submitted batch work.

### Third: improve the chance of meaningful gains

The surviving panel gives RELAY little favorable headroom: one-dimensional streaming has only one canonical ordering; row-contiguous softmax/layer norm already have natural access patterns; many matrix-reuse workloads are excluded. Preserve these as no-harm controls, but restore the planned GEMM, attention, and reserve coverage before making a broad assessment.

I would pursue **layout selection subject to a compiler contract**: preserve vectorized accesses, admissible resource use, and numerical behavior; optimize locality within that feasible set; retain ordinary storage when predicted benefits are small. This aligns directly with the stated goal of some useful speedups and little harm, without requiring locality to rank all runtimes.

If this still produces almost exclusively abstentions, change the research unit:

- **Joint storage and execution-layout selection.** A physical-layout change may require different thread ownership, tiles, or pipelining to pay off. Evaluate a small joint set of schedule/layout choices, and give the baseline an equivalent native tuning budget. Label this separately from the fixed-configuration experiment.
- **Producer–consumer or multi-kernel layout planning.** Persistent weights, transpose-heavy pipelines, repeated gather/reduction consumers, and incompatible access orientations offer an opportunity to amortize storage changes. Include producer packing/output-layout cost and all consumers in the objective.
- **Analytical shortlist followed by limited measurement.** Use RELAY to reduce a large search to a few structurally safe candidates plus the baseline, then time them. This changes the claim from prediction-only optimization to search acceleration; it needs a fair equal-budget autotuning comparison and independent final measurements.

These changes should be motivated by measured headroom, not by replacing unfavorable evaluation cases with favorable ones. If a corrected, compiler-aware implementation cannot beat the ordinary layout on a representative panel, the defensible result is a locality/counter analysis plus a limited-domain optimizer, rather than a general speedup claim.

## Appendix A. CPU reproduction of the baseline tie defect

Run from the repository root with `.venv/bin/python`. This deliberately uses a constant objective to isolate tie semantics; it does not simulate GPU timing.

```python
import sys
sys.path.insert(0, "triton/experiments")

from relay import (
    Access, HardwareProfile, MatrixSpec, MemoryEvent, SimpleRelayProblem,
    layout_matrix_rows, row_major_layout,
)
from relay.objectives import EdgeFamily, Hyperedge, ScopeKey
from search_algorithms import _canonical_search

component = EdgeFamily(
    ScopeKey("issue", 32, "stream", "load"),
    {"x": (Hyperedge.make([(0, 0)]),)},
    normalization_bytes=4,
).at_scale(16)

class FixedObjective:
    def build(self, *args):
        return [component]

matrix = MatrixSpec("x", (8, 16), 4, ("i", "j"), role="read")
event = MemoryEvent.make(
    "load", "load",
    [Access("x", (i, j)) for i in range(2) for j in range(4)],
)
problem = SimpleRelayProblem(
    (matrix,), (event,), (), (FixedObjective(),), "canonical",
)
profile = HardwareProfile(
    profile_id="repro", device={}, byte_scales=(16,),
    tau={component.name: 1.0}, fine_component=component.name,
)
ordinary = layout_matrix_rows(matrix, row_major_layout(matrix))
for whole in (True, False):
    layouts, record = _canonical_search(problem, profile, whole=whole)
    print(whole, record["score"]["hardware_area"],
          layout_matrix_rows(matrix, layouts["x"]) == ordinary,
          layouts["x"].offset(matrix, (0, 4)))
```

Observed output before a fix:

```text
True  0.0 True  4
False 0.0 False 8
```

The second line should retain the ordinary mapping under the documented baseline tie policy.

## Appendix B. Recomputing the aggregate table

This reads per-case JSON files without changing results. The expected snapshot is 432 reports and 204 completed cells; later completed jobs can change the totals.

```python
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics

root = Path("triton/experiments/results/tau-profiles")
groups = defaultdict(list)
for path in sorted(root.glob("*/experiment-[456]/*/*--*/report.json")):
    report = json.loads(path.read_text())
    groups[(report["experiment"], report["tau_name"],
            report["platform"])].append(report)

for key, reports in sorted(groups.items()):
    complete = [r for r in reports if r["status"] == "complete"]
    changed = sum(r["search"]["transformed_array_count"] > 0 for r in complete)
    speedups = [r["timing"]["baseline"]["median_ms"] /
                r["timing"]["selected"]["median_ms"] for r in complete]
    gm = math.exp(statistics.mean(map(math.log, speedups)))
    print(key, "complete", len(complete), "changed", changed,
          "missing", 29 - len(reports), "geomean", round(gm, 4))
```
