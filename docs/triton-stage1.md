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
counts, raw timing samples, transaction reduction, and measured speedup.
`--matrix-size`, `--transaction-bytes`, and the timing controls are explicit so
larger confirmation runs do not require code changes.

## Current scope

This experiment deliberately fixes the compute/distributed layout and changes
only the prepacked persistent matrix layout. It uses one transaction scale and
one best canonical realization, exactly isolating the Stage 1 question. It
does not include packing cost, arbitrary pointer-expression extraction,
multiple arrays, resource maps, or sparse shears. The problem constructor and
`InducedMemoryEvent.event` are reusable for those extensions without changing
the induced-hypergraph contract.
