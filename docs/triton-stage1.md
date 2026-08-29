# Triton integration stage 1: execution-conditioned quotient LAQS

Stage 1 asks whether exact issue cohorts induced by Triton LinearLayouts are
enough for RELAY to select a better persistent memory layout. It optimizes
transaction locality only; resource placement and flag-fiber search remain
Stage 2 concerns.

For induced events \(e\), the implemented objective is

\[
J_Q(A;L)=\sum_e w_e\left|\pi_d A L(H_e)\right|,
\]

where \(L\) is the compiled execution layout, \(A\) is a candidate persistent
layout, and \(\pi_d\) maps element offsets to aligned transactions of the
selected byte size. Event weights retain dynamic instruction multiplicity.

## Library boundary

`induce_memory_event` now accepts `coordinate_map`. Triton's LinearLayout maps
a hardware location to a tensor-local coordinate; the coordinate map models
the pointer expression that embeds that tensor coordinate in an array. This
keeps execution layout, logical indexing, and persistent layout as separate
composable objects.

`execution_conditioned_quotient_problem` collects induced events into exact
`ExplicitRegions` hyperedges and returns an ordinary `RelayProblem`. Stage 1
therefore uses the existing solver, scorer, layouts, and reports. The result is
also the input needed by later flag-fiber work; there is no Triton-specific
search implementation.

Stage 1 searches the canonical grammar only inside a bounded persistent-load
or declared natural-reuse tile. The complete address map is the two-level
composition

\[
A=\begin{bmatrix}A_{\mathrm{inner}}&0\\0&A_{\mathrm{outer}}\end{bmatrix},
\]

where $A_{\mathrm{inner}}$ is searched and $A_{\mathrm{outer}}$ is fixed to
row-major tile order. The flat row-major tensor mapping remains a default
control. Full column-major is not admitted as an alternative outer layout.
This keeps the search on address bits for which the compiled execution layout
provides direct evidence.

The kernel-breadth experiment treats tile shape as an explicit sensitivity
parameter. It does not claim that Triton supplied the added orthogonal tile
extent: the compiled execution cohort stays fixed while each declared shape
changes the bounded domain of $A_{\mathrm{inner}}$. Every resulting complete
mapping is rescored against the same induced events, and the best canonical
candidate from every shape is retained along with flat row-major. Ranking then
deduplicates identical complete physical mappings. This distinguishes a valid
tile-hypothesis sweep from extracting a larger execution layout than the
kernel actually has.

The score policy is lexicographic:

1. minimize the induced quotient transaction count;
2. minimize address-expression runs among locality ties; and
3. minimize XORs among remaining ties.

Only the first item is the Stage 1 objective. The other two choose a simple
canonical inner realization without introducing another memory-performance
score. Reported run counts cover the complete composed address expression,
including the fixed outer order.

## Symmetric row/column experiment

`triton/run-stage1.py` compiles a one-wave Triton kernel that repeatedly loads
both a logical row and the matching logical column of one FP32 matrix. The
kernel's `tl.arange(0, 64)` blocked layout is extracted from TTGIR and passed
through Triton's native `LinearLayout` binding before RELAY induces two exact
64-element hyperedges.

One representative row and column event are sufficient here. Every translated
64-element chunk is a coset of the same direction space, so a bit-linear
persistent layout gives it the same quotient count. Each representative is
weighted by the exact number of dynamic issue instances in one kernel launch.

The experiment then:

- solves the induced 128-byte quotient objective;
- packs identical logical data into row-major and selected physical layouts;
- specializes the same Triton kernel for both address maps;
- checks both outputs against the logical row-plus-column sum; and
- times alternating samples, excluding packing and compilation.

Run it from the repository root in one MI300A allocation:

```bash
module load rocm/7.2.1
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1.py \
  --json triton/results/stage1.json
```

The JSON reports the compiled blocked layout, representative hyperedges,
packing lower bound, searched inner tile and word, fixed outer order, complete
physical word and address rows, predicted transaction counts, raw timing
samples, transaction reduction, and measured speedup. It also reports every
solver-retained candidate as

\[
\left(J_Q(A;L),\ \operatorname{runtime}(A),\
      \operatorname{runs}(A),\ \operatorname{XORs}(A),\
      \operatorname{codegen}(A)\right).
\]

The compiled-codegen record includes Triton's loaded-kernel register and spill
counts, maximum threads, shared-memory bytes, binary size, IR sizes, assembly
instruction totals, and a complete opcode histogram. Candidate timing order is
balanced by rotating alternating forward and reverse orders.

`rank_quality` defines top-\(k\) regret as

\[
\frac{\min_{A\ \text{in the first }k\text{ distinct mappings}} T(A)}
     {\min_{A\ \text{in all distinct retained mappings}} T(A)}-1.
\]

Before ranking, candidates are deduplicated by their physical address mapping;
the first occurrence in solver order is retained. This prevents mandatory
control layouts from consuming a second top-k position when the DP already
produced the same mapping. Regret is reported for every k from one through
five. Solver order resolves quotient ties using runs and then XOR
count. Rank correlation is tie-aware Spearman correlation between quotient
score and median runtime over the same distinct-mapping population, with
smaller values better on both axes. Equal-score spread is reported for every
quotient-score group containing at least two distinct mappings. Same-flag
spread is reported only when a flag has at least two distinct physical
mappings; repeated measurements of an identical mapping are reported
separately as noise controls.

`--matrix-size`, `--transaction-bytes`, and the timing controls are explicit so
larger confirmation runs do not require code changes.

For confirmation across several sizes and independent process launches, run:

```bash
module load rocm/7.2.1
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-ranking.py \
  --matrix-sizes 512 1024 2048 \
  --process-launches 3 \
  --json triton/results/stage1-ranking.json --quiet
```

The aggregate runtime for ranking is the median of the per-process medians.
The output retains every process's candidate metadata and raw samples, so the
aggregate result does not hide launch-to-launch changes in rank ordering.

## Targeted kernel suite

`triton/run-stage1-suite.py` applies the same execution-conditioned objective
to four deliberately selected regimes:

1. **Contiguous vector add.** One input is a layout target. The compiled
   256-element tile has four waves, and each wave's contiguous issue cohort is
   induced separately. This is the negative control.
2. **Distributed 2-D tile.** A custom reduction loads a 32x32 FP32 tile with
   four waves. Triton's blocked encoding directly covers 8x32 elements and
   extends the register dimension by 4x to cover the tensor. All sixteen
   register/warp lane cohorts are represented.
3. **GESUMMV.** A one-wave output tile maps lanes to output rows. At each inner
   step, A and B therefore expose explicit column-oriented issue cohorts. The
   solver chooses A and B independently; timing covers default/default,
   LAQS/default, default/LAQS, and LAQS/LAQS.
4. **Prepacked-B GEMM.** A fixed 32x32x32 FP16 dot kernel retains row-major A
   and C while allowing one reusable B operand to change. B's blocked load
   encoding has four waves and four register elements per lane.

The blocked-layout bridge implements Triton's register repetition when a CTA's
hardware factors cover less than the tensor shape. This preserves Stage 0's
exact no-repetition behavior while allowing Stage 1 to model multiple elements
per lane.

For independent-process confirmation:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-suite-sweep.py \
  --process-launches 3 \
  --json triton/results/stage1-suite-sweep.json --quiet
```

The default suite uses 21 samples, 50 launches per sample, and 10 warmup rounds.
Its aggregate is the median of the per-process medians, while retaining all raw
process samples and codegen statistics.

## Stage 1.5: complete GEMM ranking

`triton/run-stage15-gemm.py` reuses the suite's exact fixed GEMM kernel and
execution-layout helpers but compiles, validates, and benchmarks all eight
retained B layouts. `triton/run-stage15-gemm-sweep.py` repeats that worker in
fresh processes, aggregates candidates by stable layout identity, recomputes
rank quality from the median of process medians, and can profile the default
and selected mappings in isolation.

Keep the timing and counter portions in separate five-minute allocations:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --process-launches 3 \
  --json triton/results/stage15-gemm-ranking.json --quiet

flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --ranking-json triton/results/stage15-gemm-ranking.json \
  --profile --json triton/results/stage15-gemm.json --quiet
```

The counter configuration uses four replay passes because gfx942 cannot
schedule all of the required counter blocks at once. It reports:

- TCP-to-TCC read requests, which are the closest measured analogue of
  coalesced load transactions in the quotient model;
- L2 tag requests, hits, misses, and hit rate;
- HBM read/write bytes and achieved bandwidth;
- memory-unit stall percentage and aggregate stall cycles; and
- MFMA utilization, profiler-derived FP16 operation count, and achieved TOPS.

Only the final steady-state dispatches after explicit warmups are aggregated.
The counters cover the whole GEMM, not B alone. Since A, C, the launch shape,
and the kernel configuration are held fixed, request-count differences are
attributable to B. The JSON also includes an explicitly labeled decomposition
that infers the fixed and B request streams under the quotient score's B ratio.

On the current MI300A run, the selected
`nnnkkkkkkkkknnnnnn` layout is fastest across the aggregate and in each of the
three process launches. Top-1 and top-3 regret are both zero, and tie-aware
Spearman correlation is 0.756. The six score-262,144 candidates span 1.11% in
runtime. Whole-kernel TCP-to-TCC read requests fall 30.37%; the inferred B
stream falls exactly 50%, from 123,168 to 61,584 requests. L2 misses remain
45,216 and HBM read bytes change by only 0.016%. This is evidence that Stage 1
correctly removes B request redundancy, but that the redundant requests were
L2 hits rather than HBM traffic. The equal-score runtime spread remains useful
motivation for realization-aware work, but GEMM does not show that the Stage 1
quotient ranking itself needs repair.

ROCm 7.2's `rocprofv3` currently aborts while profiling this PyTorch/Triton
process during late HIP registration. The driver therefore defaults to the
available ROCm 7.0.2 `rocprof` counter collector; `--rocprof` and
`--profile-config` make that choice explicit and replaceable. The profiler path
and configuration are recorded in the result.

## Stage 1 breadth experiments

Two resumable drivers extend the all-candidate ranking experiment without
turning it into a broad operator-count exercise. Both run each case in three
fresh Python processes, retain raw samples and codegen statistics for every
candidate, and rank the candidates using the median of the process medians.
Existing per-process JSON files are reused, so a sweep can be divided into
several Flux allocations without changing the aggregate.

`triton/run-stage1-gemm-breadth.py` holds one prepacked B operand and sweeps 13
fixed GEMM regimes:

- 512, 1024, and 2048 square problems;
- two lower-arithmetic-intensity skinny problems;
- warm and explicitly cache-thrashed timing;
- transposed A and transposed B storage; and
- three block/warp configurations.

The cache-thrashed path touches a 256 MiB buffer before every measured GEMM.
The thrash kernel is synchronized by stream order but lies outside the GPU
event interval, so the reported runtime covers only GEMM. This is an
operational cache-state control rather than a claim that every level of the
MI300A memory hierarchy is empty.

The current execution-layout bridge extracts the compiled blocked B-load
encoding rather than Triton's dot-operand encoding. The generalized worker
therefore requires `BLOCK_M == BLOCK_K`; the supplied configurations all obey
that restriction. This keeps the induced B tile and the layout extracted from
the compiled kernel aligned until dot-operand extraction is implemented. For
non-transposed B, the searched inner tile is `BLOCK_K x BLOCK_N`; transposed-B
storage uses `BLOCK_N x BLOCK_K`. The layout of tiles within the full operand
is fixed row-major.

Run a subset that fits comfortably within one five-minute allocation, then
continue with another subset using the same results directory:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-gemm-breadth.py \
  --cases square_512_warm square_1024_warm square_2048_warm \
  --process-launches 3 \
  --json triton/results/stage1-gemm-breadth.json --quiet
```

`triton/run-stage1-kernel-breadth.py` covers seven persistent-operand regimes:

- repeated bias with ReLU and matrix bias with row softmax;
- embedding-bag lookup from a persistent weight table;
- GEMV, MVT, and GESUMMV matrix operands; and
- a five-point stencil over a persistent field.

The suite includes contiguous negative controls as well as row, column,
row-plus-column, indirect, and neighborhood access patterns. Affine-translated
cohorts are represented once with their exact dynamic multiplicity. The
stencil represents the clamped left and right boundary cohorts separately.
Bias-ReLU searches 32, 64, 128, and 256-element one-dimensional tiles. The 2-D
cases use the power-of-two ladder 1, 2, 4, 8, 16, 32, and 64 in the orthogonal
reuse dimension: `Rx256` for softmax bias, `Rx128` for embedding bag, `64xC`
for GEMV and GESUMMV, and `Rx64` for stencil. MVT contains both row and column
loads, so it searches both `64xC` and `Rx64`, with duplicate `64x64` removed.
Every hypothesis retains row-major outer tile order.

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-kernel-breadth.py \
  --process-launches 3 \
  --json triton/results/stage1-kernel-breadth.json --quiet
```

Each case summary contains all inner-tile hypotheses, the selected tile, and
the fixed outer order; default and selected quotient scores and runtimes;
speedup; top-1 through top-5 regret; rank correlation; deduplicated mapping
count; selected layout word; and whether LAQS retained the default physical
mapping. Full case records preserve duplicate-mapping, equal-score, and
same-flag runtime spreads plus per-candidate register, spill, instruction,
binary, and opcode statistics.

The current three-process tile-sweep result is stored in
`triton/results/stage1-kernel-breadth.json`. Bias-ReLU, biased softmax, and
embedding bag collapse to one physical mapping after deduplication. All seven
GEMV and GESUMMV shapes collapse to the same selected mapping plus row-major;
GESUMMV improves by 8.77%, while GEMV's 32x quotient reduction is 3.38% slower
than row-major and therefore has zero top-2 regret. MVT produces ten distinct
mappings. Its selected layout improves on row-major by 11.11%, and its
top-1 through top-5 regrets are 1.35%, 1.35%, 0.93%, 0.93%, and 0%; the six
minimum-score mappings span 4.38% in runtime. Stencil produces seven distinct
equal-score mappings: its top-1 regret is 5.06% and top-2 regret is zero. The
selected stencil mapping is physically row-major, so its separately timed
duplicate is also retained as a noise-control measurement.

Results produced before the inner-tile scope change searched every logical bit
of the operand and are not directly comparable. The previous 32x64x32
eight-warp inversion motivated fixing the unsupported high-bit choices. A
three-process controlled rerun fixes the suffix to row-major tile order and
still measures 51.75% top-1 regret and 17.58% top-2 regret. The inversion is
therefore not explained solely by unconstrained outer bits; it remains a
useful counterexample for improving Stage 1's execution model.

## Controlled Stage-2 flag-fiber probe

`triton/run-stage2-probe.py` tests flag realization without integrating a
fiber into the solver DP. It uses the Stage-1-selected prepacked B flag for a
2048x2048x2048 GEMM with a fixed 32x32x32, four-warp configuration. Every GEMM
is preceded by a 256 MiB cache-thrash launch outside the timing interval.

The MI300A proof-of-concept resource map selects element-offset destination
bits 11, 12, and 13, corresponding to its modeled HBM-stack byte-address bits
12, 13, and 14. Identity plus every color-relevant elementary upper-triangular
shear produces 28 realizations. For every realization the probe:

- compares the entire low-address prefix flag with the Stage-1 flag;
- recomputes and requires the exact 128-byte quotient score;
- records analytical runs/XORs and compiled register, spill, instruction, and
  opcode statistics separately;
- scores translated groups of four B issue cohorts per warp and B tile under
  the HBM-stack service sketch; and
- packs B, checks GEMM output, and benchmarks the specialized address map.

The service score covers B only because B is the only changing allocation.
It is a hypothesis from the proof-of-concept HBM-stack interleave map, not a
measured hardware counter. The complete dynamic B-tile translations and their
program multiplicities remain in the grouped service calculation.

Run three fresh processes with the resumable driver:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage2-probe-sweep.py \
  --process-launches 3 \
  --json triton/results/stage2-probe.json --quiet
```

The aggregate gate calls variation meaningful at a 2% total runtime spread.
It calls the service score predictive when its tie-aware rank correlation is
at least 0.5 and the fastest minimum-service realization is within 2% of the
measured fastest realization. All thresholds and the underlying continuous
metrics are recorded, so the binary gate does not replace inspection of the
data.

On the current three-process MI300A run, all 28 realizations preserve the
complete flag and the exact quotient score of 16,777,216. The modeled service
score has two levels: six shears reduce it from 1,048,576 to 449,389.71. The
aggregate runtime nevertheless spans only 3.11%, and the service/runtime rank
correlation is -0.054. The low-service group has a 512.484 us median versus
512.245 us for the higher-service group. Its fastest realization is 1.81%
slower than the measured fastest candidate.

The unsheared Stage-1 realization is fastest in the aggregate at 501.484 us
and fastest in two of three processes. It compiles to 486 instructions; the
low-service shears compile to 496 instructions, while all candidates use 56
registers and have no spills. Compiled instruction count has a modest 0.437
rank correlation with runtime. Thus the probe observes real same-flag
variation but no service-predictable benefit, and the declared gate is false.
This result favors retaining fiber-aware realization as future work rather
than implementing the full Stage-2 DP now.

## Current scope

This experiment deliberately fixes the compute/distributed layout and changes
only the prepacked persistent matrix layout. It uses one transaction scale and
the canonical layouts retained by the quotient solver, exactly isolating the
Stage 1 ranking question. It does not include packing cost, arbitrary
pointer-expression extraction, multiple arrays, resource maps, or sparse
shears. Consequently, same-flag realization groups may be unavailable in the
current candidate set. The problem constructor and `InducedMemoryEvent.event`
are reusable for those extensions without changing the induced-hypergraph
contract.
