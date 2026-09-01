# LinearLayout register-fiber experiment on MI300A

This experiment tests whether retaining the register coordinates from Triton's
hardware LinearLayout improves the Stage 1.5 prepacked-B GEMM layout model. The
512-square GEMM uses a 32x32x32 tile, four waves, and four FP16 register values
per lane. Regrouping the existing issue events by fixed lane, warp, and block
creates 256 register-ownership fibers. Each fiber is evaluated at its exact
8-byte footprint. The search minimizes the equal-weight sum of normalized
excess for the 128-byte lane-issue component and the 8-byte register component.

All runtime rankings use three fresh processes, 21 samples per candidate, 50
kernel launches per sample, and the median of process medians. The counter
panel profiles the eight issue-selected layouts for 20 steady-state dispatches
per counter pass.

## Results

| Metric | Issue quotient | Issue + register fiber |
|---|---:|---:|
| Runtime Spearman rho, issue-selected panel | 0.412 | 0.050 |
| Profiled-duration Spearman rho | 0.247 | -0.100 |
| L1-to-L2 read-request Spearman rho | 0.577 | 0.651 |
| L2 tag-request Spearman rho | 0.577 | 0.651 |
| HBM-read-byte Spearman rho | 0.218 | 0.132 |

L2 misses were identical for all eight layouts, so their correlation is
undefined. HBM bytes were also effectively constant. The register component
alone has rho 0.130 with both L1-to-L2 reads and L2 tag requests.

The active register-aware search selects the same physical mapping as the
issue-only search. Its issue quotient is 262,144, versus 524,288 for row major,
and its register component reaches the packing lower bound. The active panel's
runtime rho is 0.577 for both scores because every retained non-control mapping
also reaches the register lower bound. The selected mapping is 1.013x faster
than row major in that run, but is 0.525% slower than the fastest retained
mapping.

Thus, this first test does not improve the yielded issue quotient, selected
mapping, or runtime prediction. It modestly improves correlation with the two
downstream memory-request counters on the broader issue-selected panel, but
that signal does not carry through to runtime. The most direct next test is a
kernel whose compiled LinearLayout exposes nontrivial register fibers that are
not already simultaneously optimal with its lane-issue objective.

Raw reports:

- `stage15-gemm-fresh-baseline-mi300a.json`
- `stage15-gemm-register-fibers-mi300a.json`
- `stage15-gemm-fiber-counter-panel-mi300a.json`
