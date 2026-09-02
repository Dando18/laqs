# Automatic edge and hardware-service validation on MI300A

This report records the September 2, 2026 validation after moving the pinned
Triton checkout to `Dando18/laqs-triton` commit
`b3376d6459bfb14f2500c1c20b3948ad59649bf8`.

## What was validated

The automatic frontend and the layout-realization experiment have different
boundaries. The frontend obtains the trace and universal edge families from an
ordinary concrete Triton launch. The existing Stage-1 harness packs the chosen
layout and supplies its address matrix so that candidate layouts can be timed.
The small-instance host oracle connects the two: for bias + ReLU, softmax +
bias, embedding bag, GEMV, MVT, GESUMMV, five-point stencil, and prepacked-B
GEMM, it compares
every automatic event and trace class with an independent host trace, then
compares all 24 edge families, every MI300A quotient component for every
canonical layout, and eligible resource-cohort scores. All eight cases passed.

The equality means that a universal-family score recovered automatically is
the same score the independent trace produces on these instances. The larger
performance corpora below still use their existing trace/realization harnesses;
they do not claim that LAQS yet rewrites an arbitrary Triton kernel to realize
the selected storage layout.

## Ordinary-launch coverage

All five ordinary launch cases completed on an AMD Instinct MI300A with ROCm
7.2. The target-neutral run reported 5 supported, 0 unsupported, and 0 errors.
With the MI300A profile attached, every case again succeeded and materialized
24 universal families and 288 objective components:

| Kernel | Trace classes | Dynamic class multiplicity | Events | Components |
|---|---:|---:|---:|---:|
| Vector add | 65 | 65 | 195 | 288 |
| Fused softmax | 32 | 32 | 80 | 288 |
| Tutorial matmul | 64 | 64 | 3,072 | 288 |
| Layer normalization | 32 | 32 | 256 | 288 |
| Descriptor copy | 4 | 4 | 8 | 288 |

These counts are from resource-anchor-preserving evaluation. Target-neutral
evaluation compresses aligned translations more aggressively. The two matmul
runs selected different valid autotune configurations; each concrete choice
was captured in the corresponding result rather than assumed by the frontend.

Raw reports:

- `automatic-frontend-validation-mi300a.json`
- `automatic-frontend-profile-validation-mi300a.json`
- `automatic-frontend-profile-validation-heavy-mi300a.json`

## Current universal-profile rescore

The current `build_edge_families` implementation and `MI300A_V1` response were
rebuilt for all five N=256 HIP kernels. Runtime samples for the same 73 layouts
were reused from `results/layout_ranking.json`; no timing result was refit or
remeasured. This check is materially worse than the stale aggregate in that
file, whose stored objective names come from the older pre-universal builder.

| Kernel | Frontier layouts | Best-frontier regret | Exact winner retained |
|---|---:|---:|---:|
| ATAX | 1/73 | **21.481%** | no |
| GEMM | 6/73 | 0.183% | no |
| GESUMMV | 12/73 | 0.966% | no |
| MVT | 5/73 | **39.035%** | no |
| SYRK | 20/73 | 0.102% | no |

In aggregate, the frontier retains 12.1% of layouts, contains an at-most-1%
candidate for 3/5 kernels, has 0.966% median regret, 12.353% mean regret, and
39.035% maximum regret. This does **not** validate the current universal
profile as a reliable layout selector.

The representation ladder localizes part of the problem. Keeping every
universal component (`F_all`) raises mean retention to 52.3% and recovers three
exact winners, but ATAX and MVT remain large confirmed dominance failures. A
source/stream-split representation recovers all five only because its frontier
retains all 73 layouts in every case. The immediate issue is therefore not the
automatic manifest extraction; it is missing discriminating structure and/or
an unsuitable compression/response in the production universal profile.

The rescore is reproducible with:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --kernel atax --kernel gemm --kernel gesummv --kernel mvt --kernel syrk \
  --size 256 --samples 5 --iterations 3 --warmup 2 \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --reuse-timings results/layout_ranking.json \
  --output results/automatic_edge_layout_ranking_mi300a.json
```

## Hardware-service counter agreement over 700 measured layouts

This is a separate, more lowering-conditioned Stage-1 edge hierarchy, not the
aggregate `MI300A_V1` profile above. Its analysis was recomputed from the seven
100-layout random panels and exactly matched the checked-in analysis. Values
below are medians of the seven per-kernel tie-aware Spearman correlations.

| Predictor | vL1 64 B accesses | L1-to-L2 reads | L2 tag requests | Profiled duration |
|---|---:|---:|---:|---:|
| Previous issue Q64 | 0.652 | 0.940 | 0.939 | 0.730 |
| Per-operation instruction Q128 | 0.645 | **0.976** | **0.975** | 0.656 |
| Lane bits `{2,3}` Q64 | **0.751** | 0.446 | 0.464 | 0.314 |
| Weighted selection objective | 0.582 | 0.831 | 0.837 | 0.649 |

The separated signature is more informative than its weighted scalar. The
lane cohort is the best vector-L1 predictor and is consistently positive in
all seven kernels (rho 0.673--0.829). The coarser instruction edge is the best
downstream-request predictor. MVT remains the weakest downstream correlation
case because its row and transposed streams prefer opposing layouts. GEMV
runtime is also a warning against interpreting request correlation as a full
performance model: its per-operation Q128/runtime rho is -0.214 even though
Q128/request rho is 0.980.

## Fresh hardware-service timing replication

GEMV, MVT, and GESUMMV were rerun in one fresh process with nine timing samples,
20 iterations per sample, and five warmups. All candidates passed numerical
validation, had zero spills, and retained the reference load/store instruction
structure.

| Kernel | Selected tile | Default | Selected | Speedup | Prior 3-process speedup |
|---|---:|---:|---:|---:|---:|
| GEMV | 64 x 4 | 347.644 us | 323.040 us | **1.0762x** | 1.0761x |
| MVT | 4 x 64 | 451.323 us | 406.531 us | **1.1102x** | 1.1108x |
| GESUMMV | 64 x 4 | 472.455 us | 450.749 us | **1.0482x** | 1.0485x |

The geometric-mean speedup of this changed-layout subset is 1.0779x. Each new
default and selected median is within 0.08% of its prior three-process median,
and each speedup is within 0.06% relative. The earlier full seven-kernel result
remains 1.0352x geometric mean, with four improvements and no regressions due
to the default-on-exact-tie policy.

Raw reports:

- `stage1-hardware-service-mi300a-validation.json`
- `stage1-hardware-service-mi300a-validation-cases/`

## Fresh counter pairs

One warm-cache ROCprof launch per layout, with 20 final target dispatches and
four compatible counter passes, was collected for MVT and GESUMMV using the
hardware-service selection itself.

| Kernel | Objective reduction | vL1 access reduction | L1-to-L2 read reduction | L2-tag reduction | L2-miss change | HBM-read change |
|---|---:|---:|---:|---:|---:|---:|
| MVT | 57.07% | 29.41% | 87.38% | 86.91% | -0.001% | -0.002% |
| GESUMMV | 74.63% | 43.75% | 47.26% | 47.22% | -0.028% | -0.025% |

The large reductions stop at the warm L2 boundary: miss and HBM traffic are
unchanged to measurement noise. This supports the intended interpretation
that the edges predict request formation and cache service, not compulsory
memory traffic. ROCprof replay duration is not used as the runtime result: it
reported 1.300x for MVT but only 1.0006x for GESUMMV, whereas the independent
unprofiled timings above are stable at 1.1102x and 1.0482x.

Raw reports:

- `stage1-hardware-service-counter-validation.json`
- `stage1-hardware-service-counter-validation.csv`
- `stage1-hardware-service-counter-validation-profiles/`

## Assessment

The compiler-to-trace frontend is functioning end to end on the tested
conventional kernels, including autotuning, non-power-of-two boundaries,
structured loops, multiple allocations, and descriptors. Its recovered graph
is exactly equivalent to the independent Stage-1 oracle on the supported small
instances.

The current universal MI300A selection profile is not validated: the direct
N=256 rescore has two severe dominance failures. In contrast, the separate
lowering-conditioned Stage-1 service hierarchy has the expected hardware
interpretation: fine lane cohorts track vector-L1 work, coarser instruction
edges track downstream requests, and its selected layouts deliver stable
speedups and large request reductions without falsely predicting HBM changes.

The next model work should focus on why the universal basis makes the ATAX and
MVT winners dominated, using the successful instruction/lane service signature
as the controlled reference. Recalibrating tau alone cannot repair a winner
that remains dominated in the complete component representation. Counter replay
timing is also not a reliable runtime metric. Finally, automatic candidate
layout realization for arbitrary unmodified kernels remains outside this
validation; the Stage-1 packing harness realizes the layouts used in the
performance runs.
