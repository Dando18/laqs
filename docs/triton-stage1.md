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

The first experiment searches the complete canonical flag grammar. Its score
policy is lexicographic:

1. minimize the induced quotient transaction count;
2. minimize address-expression runs among locality ties; and
3. minimize XORs among remaining ties.

Only the first item is the Stage 1 objective. The other two choose a simple
canonical realization without introducing another memory-performance score.

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
packing lower bound, default and selected layout words, predicted transaction
counts, raw timing samples, transaction reduction, and measured speedup. It now
also reports every solver-retained candidate as

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
\frac{\min_{A\ \text{in the first }k\text{ solver choices}} T(A)}
     {\min_{A\ \text{in all retained choices}} T(A)}-1.
\]

Solver order resolves quotient ties using runs and then XOR count. Rank
correlation is tie-aware Spearman correlation between quotient score and median
runtime, with smaller values better on both axes. Equal-score spread is reported
for every quotient-score group containing at least two candidates. Same-flag
spread is reported only when a flag has at least two distinct physical mappings;
repeated measurements of an identical mapping are reported separately as noise
controls.

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
