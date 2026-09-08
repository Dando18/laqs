# Why Experiments 4–6 still do not show speedups

Snapshot: **September 8, 2026, 14:16 PDT / 21:16 UTC**, from `triton/experiments/results/search-v2`. Matrix has finished submitting/executing its workflow; Tuolumne is still incomplete. This investigation reads reports, proposed and deployed selections, generated code, profiler counters, scheduler logs, and the benchmark/search implementations. It also runs small CPU footprint witnesses; it does not submit new GPU jobs or change the sources used by the active experiments.

## Answer

**Problem size is not the primary explanation.** The more useful distinction is:

1. Many benchmarks already attain the packing lower bound for the objective being optimized. There is no modeled improvement to find, even with an unrestricted layout search at those scales.
2. Other benchmarks have substantial predicted headroom, but the current compiler realization loses efficient memory operations or incurs address-computation costs. Most such candidates are never timed.
3. In the one configuration where changes reach timing, a real reduction in first-level footprint does not reduce the downstream traffic or kernel runtime.

Consequently, these results do **not** show that Triton's ordinary storage is universally best, and they do **not** refute the quotient-space identity. They show that the current combination of benchmark panel, fixed execution schedule, footprint objective, realization, and admission rules produces almost no useful deployed changes.

My previous repair deliberately prevented dangerous transformations, but it also made this a very conservative deployment experiment. In particular, rejecting every register increase and every static instruction increase above 10% leaves important opportunities unmeasured. Those thresholds are heuristics, not consequences of LAQS theory.

The next step should be a small causal experiment separating **footprint headroom → successful realization → relevant traffic reduction → runtime improvement**, rather than another broad rerun at larger dimensions.

## What actually ran

The intended panel is 29 configurations × 3 experiments × 3 profiles = 261 cells per platform. Profiles and experiments reuse configurations; these cells are not independent workload families.

| Matrix outcome | Cells | Interpretation |
|---|---:|---|
| Search proposes ordinary storage | 169 | No nonordinary layout reaches compilation |
| Search proposes a change; realization rejects it | 36 | Timing subsequently measures ordinary storage |
| Changed layout passes realization and reaches timing | 2 | Both are `int4_gemm--decode`, Experiment 6 |
| Failed or incomplete before measurement | 54 | Six configurations; not evidence of no speedup |
| Total | 261 | 207 completed measurements, covering 23 configurations / 14 families |

Thus **205 of 207 completed cells time ordinary storage**. Near-one aggregate bars are largely the result of abstention, not 207 failed trials of optimized layouts.

| Matrix profile | Completed cells | Baseline J = 0 | Changed proposals | Changed deployments |
|---|---:|---:|---:|---:|
| expert | 69 | 42 | 19 | 1 |
| l1_to_l2 | 69 | 30 | 19 | 1 |
| speedup | 69 | 12 | 0 | 0 |

All 207 completed Matrix reports record source hash `c6dba707d21c3fed582d43600814479a55211df8f57c3d11bcab7e29a9cf0692` and Triton revision `b3376d6459bfb14f2500c1c20b3948ad59649bf8`.

At this snapshot, Tuolumne has **153 validated, 54 prepared, 45 running, and 9 failed cells, with no complete timing reports**. Among its 20 changed proposals, 12 have been rejected, two have passed validation (int4 decode), and six await realization. Do not treat its prepared `transformed_array_count` as a deployment count or compare unfinished panels as performance results.

The [audit CSV](experiments-4-6-v2-audit.csv) records every cell and its report path. It distinguishes proposed changes from deployed changes and leaves deployment blank before validation. Counts here are a snapshot, not a statement about subsequent Tuolumne progress.

## Size: several workloads are already large

These are Matrix expert / Experiment 4 baseline graph timings. Input sizes are derived from the actual captured allocations; output traffic is additional.

| Case | Relevant input size | Baseline runtime | Baseline J |
|---|---:|---:|---:|
| vector_add small | 2 × 1 MiB | 1.532 µs | 0 |
| vector_add large | 2 × 256 MiB | 261.148 µs | 0 |
| vector_exp large | 256 MiB | 177.674 µs | 0 |
| softmax large | 128 MiB | 90.641 µs | 0 |
| layer_norm large | 128 MiB input + small weight/bias | 96.100 µs | 0 |
| gather_gemv large | 512 MiB weight allocation; two selected 64 MiB matrices | 56.727 µs | 0 |
| int4 prefill | 64 MiB activations + 5 MiB packed weights | 890.134 µs | 0.5333 |

The H100 has a 50 MB L2 cache. The large streaming cases exceed it by substantial margins; their profiler records also show approximately 512 MiB, 256 MiB, and 128 MiB of HBM reads for vector-add, vector-exp, and softmax, respectively. These are not simply tiny, entirely cache-resident problems. [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.2.0/pdf/Hopper_Tuning_Guide.pdf)

As a sanity check, vector-add's logical input-plus-output traffic is 768 MiB per launch. Dividing by 261.148 µs gives about **3.08 TB/s effective bandwidth**. This is a traffic/time calculation, not a claim about measured write traffic or a precise roofline, but it illustrates why substantially improving a contiguous streaming kernel is difficult.

Increasing the number of identical, aligned blocks scales total work; it need not change the low-address geometry or normalized objective. A larger vector cannot improve an already minimum-sector vector load. Likewise, adding rows to an already well-packed row softmax cannot create an opportunity that the layout objective lacks.

Small cases still matter for interpretation: 1.5 µs vector kernels have little absolute time to save, and repeatedly replaying the same buffers measures warm reuse. For future opportunity cases, sweep the **accessed working set**, not merely allocation size, through cache-resident and streaming regimes. Keep launch shape fixed during the size-only sweep, then study schedule changes separately.

## What “the default is best” does and does not mean

Under the expert profile, **14 of the 23 completed configurations have J = 0**: both vector-add, vector-exp, softmax, layer-norm, and gather-GEMV sizes; square GEMM; and the small jagged sum, mean, and softmax cases.

For these cases the recorded ordinary layout meets the packing lower bound of every positively weighted component. Since the weights and excess costs are nonnegative, this is a certificate of optimality **for that modeled objective**. Increasing search breadth cannot lower it below zero. It is not a certificate of minimum GPU runtime, nor of optimality for unweighted scopes, cache-capacity behavior, stores, bank conflicts, or another schedule.

There are additional structural reasons this panel favors ordinary storage:

- E4/E5 canonical search has only one physical mapping for a truly one-dimensional array: it preserves bit order within its sole logical dimension. E6 can explore additional bit maps, but contiguous streaming controls already coalesce well.
- The factories make dense contiguous inputs; softmax and layer norm reduce the contiguous last dimension. Gather-GEMV selects whole matrices and then reads their contiguous data, rather than performing arbitrary fine-grained gathers.
- These are authored kernel implementations, often with autotuned tiles and explicit reuse strategies. Triton's matrix-multiplication tutorial deliberately chooses block ordering to improve L2 reuse. This is a stronger baseline than an unoptimized generic access schedule. [Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- The experiment freezes the ordinary kernel's launch configuration, rewrites eligible dense input storage, and preserves ordinary outputs. It does not optimize a producer–consumer chain, shared-memory layout, or thread ownership jointly with storage. See [case factories](../triton/experiments/tritonbench_cases.py), [selection](../triton/experiments/search_algorithms.py), and [runtime realization](../triton/experiments/layout_runtime.py).

These workloads are useful **negative controls and breadth checks**, but many are weak positive tests of automatic storage-layout optimization. Keep them in the evaluation; do not require them to supply the speedup story or remove them after seeing their outcomes.

## Where headroom exists, the realization usually blocks it

Representative Matrix expert proposals follow. All rows except int4 decode revert to ordinary storage before timing.

| Candidate | Predicted J, ordinary → proposed | Generated-code evidence |
|---|---:|---|
| asymmetric GEMM, E4 | 0.1167 → 0 | Registers 117 → 128; memory instruction forms change |
| bf16×int16 output, E4 | 0.1500 → 0 | Registers 210 → 254; part of the `LDGSTS` path becomes `LDG.E.U16` |
| FP8 small, E4 | 0.6667 → 0 | Registers 96 → 231; asynchronous memory forms disappear in favor of scalar `LDG.E.U8` |
| jagged softmax large, E4 | 1.4964 → 0.0769 | Registers 36 → 40; vectorized memory forms change |
| sum small, E4 | 0.3000 → 0 | Registers remain 27 and memory forms match; static instructions 202 → 242 |
| int4 prefill, E6 | 0.5333 → 0.2000 | Registers remain 255 and memory/matrix forms match; instructions 2850 → 3490 |

Sources: the respective [Matrix expert reports](../triton/experiments/results/search-v2/tau-profiles/expert/) and their `codegen` directories. The FP8 E5 realization is less expensive than E4, but still raises registers from 96 to 168 and is rejected.

There are two distinct classes here:

**Loss of efficient lowering is a real integration problem.** Preserving low vector bits algebraically has not preserved vector/async-copy realization in these cases. A layout can have the desired address geometry yet become opaque to compiler analyses after pointer rewriting. Saved assembly establishes the loss of efficient operations; determining which rewrite or downstream analysis causes it needs targeted compiler work. This is not solved by scaling the allocation.

**A static budget alone is insufficient grounds to declare an opportunity unprofitable.** Sum and int4 prefill are rejected solely by the 10% instruction budget. Their transformed timing is unknown. More static instructions can be acceptable if they save the limiting memory work; conversely, int4 decode passes that budget and still regresses. The current guard is a conservative deployment policy, not a performance oracle.

Search also keeps one scalar-optimal canonical candidate per tile before admission. A rejected optimum is not followed by a search for every lower-benefit, lower-cost feasible alternative. E6 uses a bounded inner search plus single swap/XOR neighbors, not unrestricted GL search. “Baseline fallback” therefore cannot generally be read as “no feasible beneficial layout exists.”

## The strongest counterexample: int4 decode

Experiment 6 changes only the **16 KiB activation vector**, leaving the **5 MiB packed weight matrix** unchanged. The kernel loads even and odd activation columns separately, so separating those address bits creates an actual first-level packing opportunity.

| Quantity | Ordinary | Expert-selected |
|---|---:|---:|
| Predicted issue regions at 32 B | 327,680 | 245,760 |
| Measured first-level load sectors | 327,680 | 245,760 |
| Measured global-load requests | 61,440 | 61,440 |
| Registers | 167 | 167 |
| Static final instructions | 2,522 | 2,562 |
| Unprofiled median runtime | 76.715 µs | 83.631 µs |

The **25% first-level sector reduction is predicted exactly**. But L1-to-L2 sector traffic rises by about 0.18%, HBM reads by about 0.14%, and the paired-process speedup is **0.91727×**, with 95% interval **[0.91697, 0.91757]**. Runtime grows by about 9.0%. Identity-control deviation is below 0.04%.

The `l1_to_l2` profile chooses a different activation bit swap. It reduces modeled workgroup footprint, but measured downstream traffic changes by only about 0.03%; speedup is **0.91592×**. This variant additionally shows that the chosen temporal/workgroup footprint is not a reliable quantitative predictor of downstream traffic here.

Sources: [expert decode report](../triton/experiments/results/search-v2/tau-profiles/expert/experiment-6/matrix/int4_gemm--decode/report.json), [l1_to_l2 decode report](../triton/experiments/results/search-v2/tau-profiles/l1_to_l2/experiment-6/matrix/int4_gemm--decode/report.json), and [kernel source](../triton/tritonbench/tritonbench/operators/int4_gemm/kernel.py).

The kernel's recorded grid has **only 10 CTAs**, and the frozen block computes 16 rows although logical M is 1. The source uses row indices modulo M and ultimately masks output stores. It is a GEMM-style implementation with very limited grid parallelism for decode, not proof of an optimal GEMV schedule. Layout search does not repair this schedule.

The evidence supports a narrow conclusion: the saved first-level work is not enough to overcome the transformed kernel's costs in this implementation. Added address instructions, dependency chains, and the serial reduction/tensor-core schedule are plausible causes of the slowdown, but the existing counters do not isolate their individual contributions. The profiler also uses separate launches rather than the timed graphs, so its duration is not substituted for the graph timing.

This case is encouraging for **footprint fidelity**, disappointing for **runtime selection**, and insufficient to judge general layout-finding theory. A lower value of an excess-footprint objective is not the same percentage reduction in runtime; J approaching zero means removing modeled excess, not removing compulsory work.

## Small CPU checks distinguish geometry from size

I ran [five reproducible CPU witnesses](experiments-4-6-v2-witness.py) through the current selector and independently counted aligned regions from the returned physical addresses:

```bash
.venv/bin/python notes/experiments-4-6-v2-witness.py
```

| Access set | Search | Ordinary → selected regions | J, ordinary → selected |
|---|---|---:|---:|
| 32 adjacent FP32 elements in a row | E4, 32 B regions | 4 → 4 | 0 → 0 |
| 32 FP32 elements down a column | E4, 32 B regions | 32 → 4 | 7 → 0 |
| Same column pattern, multiplicity 256 | E4, 32 B regions | 8,192 → 1,024 | 7 → 0 |
| Two diagonal points in a 4×2 array | E4, 8 B regions | 2 → 2 | 1 → 1 |
| Same diagonal points | E6, 8 B regions | 2 → 1 | 1 → 0 |

The last mapping has binary rows `[1, 2, 5]`, mixing two logical bits through XOR. These are **algebraic tests, not GPU speedups**. They show that the implementation can find a benefit when the access geometry supplies one, that repeating a pattern does not change its normalized headroom, and that an XOR-sensitive access set can distinguish E6 from canonical layouts.

## Remaining measurement and coverage limitations

**Several important large cases did not finish successfully on Matrix.** Scheduler completion does not imply complete experimental coverage:

| Case | Root evidence in logs |
|---|---|
| sum large | Native-reference comparison fails; maximum absolute error 0.000427 versus absolute tolerance 0.000030 |
| FP8 large | Native-reference comparison fails; reported chunk maxima about 1.0–1.35 |
| bf16×int16 projection | CPU search hits its four-hour limit |
| template attention | CPU search hits its four-hour limit |
| jagged sum large | CPU search is OOM-killed |
| jagged mean large | CPU search is OOM-killed |

See [Matrix scheduler logs](../triton/experiments/results/search-v2/shared/matrix/logs/). Final `capture.json`-missing errors obscure the earlier numerical failures, while `preflight_incomplete` obscures the earlier OOM/timeouts. Preserve the first causal failure in future status reporting.

The sum discrepancy is compatible with reduction-order rounding, but that is a hypothesis to check against a higher-precision reference and a reduction-length/conditioning-aware error budget. The FP8 mismatch needs a separate arithmetic-path audit. Neither failure demonstrates a bad layout, and neither should be “fixed” by blindly loosening a global tolerance. Streaming event compression also has not eliminated all graph-memory or runtime scaling failures; the earlier repair notes were too optimistic about that.

**Unchanged-layout timing still has systematic variation.** Across 205 unchanged cells, ratios span 0.98877–1.05929×. The approximately 5.9% apparent gains are small jagged softmax with no layout change; its identity controls vary by about 2.5–2.7%. Separate output allocations, address-dependent cache behavior, and measurement effects remain possible explanations. A narrow process interval would not remove such systematic differences. Plot no-change decisions explicitly and retain their measured ratios as controls, rather than implying a layout speedup. Rotate output-buffer assignments across labels and include a same-pointer control in the next timing audit.

**The `speedup` tau map is still legacy calibration.** On Matrix it weights only two per-lane temporal scopes (0.975 and 0.025), and produces no changed proposal in this panel. It was not refitted for the corrected issue graph/realization. Its failure to find a layout is not an independent validation of the new model. Keep it labeled as an ablation until new independent pilot measurements exist.

## Recommended next experiments, in priority order

### 1. Measure the promising candidates currently hidden by admission

Start with **sum small**, then **int4 prefill**, in an isolated diagnostic suite. Retain the ordinary baseline, the current proposal, and a few cheaper intermediate layouts. Validate every candidate numerically, record its exact memory instructions and resources, and time it even when the static instruction budget rejects deployment. Do not change the active production suite or silently loosen its policy.

For sum, its proposed interleaving has the direct formula `offset(i,j) = (i >> 1) * 2048 + 2*j + (i & 1)` for the captured 4096×1024 shape. Compare that specialized realization with the generic pointer-rewrite implementation. This tests address-generation overhead without changing the logical mapping. The prediction is about improved 128-byte footprint, not an assured reduction in every traffic counter.

This experiment distinguishes “the filter hid a useful layout” from “even correct, well-realized footprint savings are irrelevant to this kernel.” The present data cannot answer that question.

### 2. Repair realization with one focused compiler case

Use asymmetric GEMM or FP8 small as a compiler diagnostic: preserve a native vector/async-copy baseline, realize the proposed mapping, and inspect exactly where contiguity information or asynchronous copies are lost. Compose the physical mapping into logical index construction, simplify constant bit fields, hoist loop-invariant transformations, and preserve proven alignment/contiguity through later passes. Do not add alignment promises unless they are proved valid for every accessed address.

For a given quotient flag, choose a cheap compatible address basis rather than assuming every realizing matrix has the same execution cost. The theory provides useful equivalence classes; the compiler should exploit them. A shortlist scored for actual address costs is more informative than an absolute prohibition on one extra register.

If efficient realization requires changing thread ownership or shared-memory staging, explicitly move to joint schedule/storage optimization. Retune both ordinary and transformed layouts over the same schedule budget; keep the current frozen-schedule comparison as a separate ablation.

### 3. Add positive controls and genuine layout-choice workloads

Retain the existing coalesced operators as negative controls. Add a separately declared panel with mechanisms that create storage headroom:

| Workload class | What it tests | Required competing baseline |
|---|---|---|
| Fixed strided/column access with an analytical solution | End-to-end recovery of known coalescing headroom | Handwritten column-major or simple tiled layout |
| Same matrix used through row and column views, e.g. MVT-like kernels | A real compromise between incompatible accesses | Best row-major, column-major, and conventional tile; tuned schedules |
| Multi-axis stencils or tensor contractions with incompatible input views | Reuse and layout tradeoffs across directions/scales | Conventional blocked layouts and algorithm-specific kernels |
| Producer followed by several consumers with different preferred layouts | Whether retaining packed storage amortizes conversion | Full pipeline including producers, conversions, and consumers |
| Structured bit/XOR gathers with known affine geometry | Capability beyond canonical bit ordering | Best canonical layout and a handwritten affine layout |

A deliberately strided positive control establishes correctness and realization quality; it is not by itself evidence that LAQS beats a well-tuned application. Since MVT/stencil-like kernels already appear among the old pilots, use independent configurations/access patterns for evaluation and do not train on their evaluation outcomes. Arbitrary random gathers are not guaranteed to supply exploitable bit-linear structure.

Declare the panel and its analytic selection criteria before collecting performance results. Include all outcomes; do not replace failures with newly selected winners. Compare against **the best conventional storage alternatives**, not only ordinary row-major, to show what automatic search contributes.

### 4. Test the connection between footprint and runtime directly

For each positive case, collect a fixed small candidate set with predicted per-scope regions, actual issue groups, measured first-level and downstream traffic, final code costs, and unprofiled runtime. Sweep accessed working sets below and above L2 capacity, with fixed launch shape, and distinguish warm repeated use from rotating-buffer/streaming use. Run an additional schedule sweep separately.

Use this to identify the limiting resource and train any revised cost model on independent pilots. A bottleneck-aware runtime approximation needs compulsory work, request/sector service, cache behavior, compute, address generation, and occupancy effects; a single weighted excess-footprint scalar cannot generally substitute for them. Report per-operand contributions so a small, repeatedly reused activation does not distract from the kernel's limiting work.

For deployment, retain the ordinary choice and evaluate packing amortization: if packing costs P and each use saves Δt, more than P/Δt uses are needed, before other conversion costs. When Δt ≤ 0, no amount of reuse makes that realization profitable. Producer-native layouts can change this equation and deserve a separate pipeline experiment.

The immediate priority is **a cheap, validated realization of known headroom**, followed by evidence that its saved traffic lies on the runtime-critical path. Simply enlarging the present zero-headroom controls will not supply that evidence.
