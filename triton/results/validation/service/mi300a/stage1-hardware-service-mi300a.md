# Hardware-conditioned service-edge experiment on MI300A

This experiment implements the event hierarchy proposed in
`notes/linearlayout-edge-construction-part2.md` and evaluates it on the seven
Stage-1 persistent-operand kernels. The selection goal is conservative: keep
the default layout when the model sees no locality benefit, and otherwise
select a layout that balances instruction coalescing with loop-local reuse.

## Construction

`linear_layout_resource_fiber` now supports two capabilities needed by the
model:

- `merge_events=False` keeps every induced memory event as a distinct lowered
  operation instead of merging compatible event streams.
- `varying_bits={"lane": S}` constructs algebraic lane subspaces for an
  arbitrary subset of lane bits while fixing the complementary bits.

The experiment evaluates the following hierarchy offline:

| Model | Edge interpretation |
|---|---|
| M0 | Previous fixed-register, full-wave issue edge |
| M1 | Full-wave edge per retained operation identity at 32, 64, and 128 B |
| M2 | Per-operation lane-bit subspaces; all 62 proper nonempty subsets of the six MI300A lane bits were evaluated at 64 B |
| M3 | Instruction-window unions and per-lane temporal/cache windows at 128 B |

All seven kernels lower the target access with one value per lane
(`register=1`). Preserving operation identity therefore leaves M1 at 64 B
exactly equal to M0 in this suite. This is a useful negative result: the old
construction was not accidentally merging register elements here. The lane
and temporal decompositions are the refinements that add information.

The tuned selection objective is the raw weighted sum below.

| Component | Scale | Tau |
|---|---:|---:|
| Full-wave issue | 128 B | 1 |
| Per-operation full wave | 64 B | 1 |
| Per-operation lane bits `{2,3}` | 64 B | 1/16 |
| Per-lane loop/cache window | 128 B | 1 |

The lane subset was selected from the existing seven 100-layout MI300A
counter panels for robust vector-L1 correlation. The small lane tau gives the
cohort score tie-breaking influence without allowing its many smaller edges
to dominate full-wave and temporal locality. A coarse weight sweep on the old
breadth results selected 1/16; the performance table below is a fresh
three-process validation, not the data used for that choice.

If a candidate and row-major have exactly the same weighted objective, the
selection policy now chooses row-major. This prevents code-layout changes when
the locality model predicts no benefit.

## Counter behavior

The new metrics were recomputed for the same 700 randomly sampled mappings
whose MI300A counters had already been collected. The table reports the
median per-kernel tie-aware Spearman rho.

| Predictor | vL1 64 B accesses | L1-to-L2 reads | L2 tag requests | Profiled duration |
|---|---:|---:|---:|---:|
| M0 issue Q64 | 0.652 | 0.940 | 0.939 | 0.730 |
| M1 instruction Q128 | 0.645 | **0.976** | **0.975** | 0.656 |
| M2 lane bits `{2,3}` Q64 | **0.751** | 0.446 | 0.464 | 0.314 |
| Weighted selection objective | 0.582 | 0.831 | 0.837 | 0.649 |

The service signature improves the counter model when its components remain
separate. The lane-bit component has vector-L1 rho from 0.673 to 0.829 in
every individual kernel (median 0.751), versus a 0.224 worst case for M0.
The 128-byte instruction component reaches about 0.98 L1-to-L2 and L2-tag rho
in GEMV, GESUMMV, embedding bag, and stencil. MVT remains difficult because
its row and transposed streams impose opposing locality requirements, and the
bias L1-to-L2 count has too little variation for a stable rank relationship.

The weighted score should not replace this counter-specific signature: it is
a layout-selection compromise and is a weaker direct counter predictor. The
result supports the intended interpretation that fine lane service predicts
vector-L1 work while a coarser full-instruction quotient predicts downstream
traffic. Exact multiscale-flag residuals could not be estimated because all
100 random mappings in each case had distinct exact flags.

## Fresh performance validation

The full suite ran on one MI300A GPU with three fresh process launches, nine
timing samples per process, 20 iterations per sample, and five warmups. Every
retained candidate was numerically validated. All candidates had zero spills,
and each kernel's load-opcode signature was invariant across layouts.

| Kernel | Selected tile | Default runtime | Selected runtime | Speedup |
|---|---:|---:|---:|---:|
| Bias + ReLU | default | 15.138 us | 15.138 us | 1.0000x |
| Softmax + bias | default | 15.496 us | 15.496 us | 1.0000x |
| Embedding bag | default | 15.334 us | 15.334 us | 1.0000x |
| GEMV | 64 x 4 | 347.884 us | 323.291 us | **1.0761x** |
| MVT | 4 x 64 | 451.629 us | 406.568 us | **1.1108x** |
| GESUMMV | 64 x 4 | 472.731 us | 450.859 us | **1.0485x** |
| 5-point stencil | 2 x 64 | 14.588 us | 14.356 us | **1.0162x** |

The geometric-mean speedup is **1.0352x**. Four of seven kernels improve and
none regress. The three no-change cases are exact modeled-locality ties and
therefore exercise the default-on-tie safety rule. The four selected layouts
reduce the weighted objective by 74.6%, 57.1%, 74.6%, and 21.3% for GEMV,
MVT, GESUMMV, and stencil, respectively.

Compared with the previous issue-only breadth result, the minimum observed
speedup improves from 0.9596x to 1.0000x and the geometric mean from 1.0166x
to 1.0352x. This is encouraging evidence for the requested safety and
frequency-of-improvement criteria on this small suite, but it is not yet a
general no-regression guarantee outside these kernels and sizes.

Raw reports:

- `stage1-hardware-service-analysis-mi300a.json`
- `stage1-hardware-service-mi300a.json`
- `stage1-hardware-service-mi300a-cases/*/process-{1,2,3}.json`
