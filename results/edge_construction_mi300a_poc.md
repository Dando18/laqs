# Universal edge construction and MI300A profile: proof of concept

This report records the August 2026 proof-of-concept evaluation of one
device-independent access-scope construction and one kernel-independent
MI300A response. The checked profile is
`mi300a-gfx942-universal-v1-poc`.

## Experiment boundary

- Hardware timings: MI300A (`gfx942`), ROCm 7.0.2 hipcc, five samples, three
  timed iterations, and two warmups.
- Canonical corpus: 73 uniform-layout controls already measured in
  `results/layout_ranking.json`.
- Calibration instances: ATAX and GESUMMV at `N=256,512`.
- Transfer instances: GEMM, MVT, and SYRK at `N=256,512`; these did not affect
  tau selection.
- Solver probe: exact `G_S` and `G_A` searches at `N=256,512`, followed by
  bounded MI300A timing jobs for every retained candidate.
- `N=1024` transfer rescoring was deliberately deferred: universal objective
  construction already took 34--172 seconds per holdout instance at the two
  smaller sizes.

The primary analytical vector is

```text
(Q_fine, J_peak_hardware, J_area_hardware, codegen runs, codegen XORs)
```

with `J_area = sum(tau[s,b] * b * (Q-LB) / B_K)`. All costs are minimized.

## Canonical-corpus calibration result

| Kernel | N | Candidates | Retained | Oracle regret | Exact winner | Full-basis winner nondominated |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ATAX | 256 | 5/73 | 6.849% | 0% | yes | yes |
| ATAX | 512 | 5/73 | 6.849% | 0% | yes | yes |
| GESUMMV | 256 | 7/73 | 9.589% | 0.965804% | no | yes |
| GESUMMV | 512 | 7/73 | 9.589% | 0.737461% | no | yes |

The mean is 6 candidates (8.219% retained), 0.425816% regret, and all four
instances are within 1% of their measured optimum. The full universal
weight-free frontiers contain 14, 14, 63, and 63 layouts respectively and
contain every exact measured winner. The two GESUMMV misses are therefore
compression misses on this corpus, not basis-dominance failures.

Sixteen seeded global tau-perturbation trials multiplied every active cell by
one of `{0.5,0.8,0.9,1,1.1,1.2,1.5}` and applied the same perturbed response
to all four instances. Across the 64 trial/instance observations, mean regret
was 0.679679%, median regret 0.368730%, maximum regret 2.431812%, and mean
retention 8.048%. This small ablation shows some GESUMMV sensitivity and should
not be read as a confidence interval.

## Unfitted-kernel transfer result

| Kernel | N | Candidates | Retained | Oracle regret | Exact winner | Full-basis winner nondominated |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| GEMM | 256 | 7/73 | 9.589% | 0% | yes | yes |
| GEMM | 512 | 7/73 | 9.589% | 0.118728% | no | yes |
| MVT | 256 | 5/73 | 6.849% | 0% | yes | yes |
| MVT | 512 | 5/73 | 6.849% | 0% | yes | yes |
| SYRK | 256 | 15/73 | 20.548% | 0% | yes | yes |
| SYRK | 512 | 15/73 | 20.548% | 0.458693% | no | yes |

The transfer-set mean is 9 candidates (12.329% retained) and 0.096237%
regret; all six instances are within 1%. The exact winner is nondominated in
the full universal basis for all six. Diagnostic full-basis frontiers retain
38/73 for each GEMM size, 23/73 for each MVT size, and 62/73 for each SYRK
size. SYRK therefore transfers in regret but misses the desired compactness
budget.

As a diagnostic, the same zero-slack fine-locality gate used for solver search
reduces each SYRK frontier from 15 to 3 layouts. Regret becomes 0.792757% at
`N=256` and 0.660192% at `N=512`, so both remain within 1%, but neither exact
winner is retained. This is a useful budget/regret tradeoff, not a global
default: the zero-slack gate is much less reliable for the GESUMMV calibration
instances.

Across calibration and transfer together, the frozen profile retains 7.8/73
layouts on average (10.685%), has 0.228069% mean and 0.965804% maximum regret,
contains 6/10 exact winners, and is within 1% on 10/10. All 10 exact winners
are nondominated in the full universal basis.

The theoretical schema has 864 scope-scale cells for every kernel. Sparse
materialization has two expected shapes: ATAX, GESUMMV, and MVT realize 24
load families (288 components), while GEMM and SYRK realize 24 load and 24
store families (576 components). Every tested instance uses the same ten
nonzero tau cells and six kappa cells; realized name sets are invariant across
sizes.

## Solver-frontier probe

An exact zero-slack fine gate bounds both solver grammars. Each raw
cross-grammar union contains seven mappings. Cross-grammar Pareto filtering
leaves five ATAX mappings and six GESUMMV mappings.

| Kernel | N | `G_S` / `G_A` frontier | Row-major ms | Best union ms | Speedup | `G_A` regret vs union |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ATAX | 256 | 4 / 3 | 0.088587 | 0.063373 | 1.398x | 1.957% |
| ATAX | 512 | 4 / 3 | 0.169254 | 0.120268 | 1.407x | 2.549% |
| GESUMMV | 256 | 6 / 1 | 0.065147 | 0.040693 | 1.601x | 19.922% |
| GESUMMV | 512 | 6 / 1 | 0.121667 | 0.080427 | 1.513x | 20.192% |

Every reported search is exact; the affine dynamic programs are untruncated.
The final reports and plots are:

- [ATAX N=256](solver_frontier_edge_mi300a_final_atax_n256.json) and its
  [plot](solver_frontier_edge_mi300a_final_atax_n256.png);
- [ATAX N=512](solver_frontier_edge_mi300a_final_atax_n512.json) and its
  [plot](solver_frontier_edge_mi300a_final_atax_n512.png);
- [GESUMMV N=256](solver_frontier_edge_mi300a_final_gesummv_n256.json) and its
  [plot](solver_frontier_edge_mi300a_final_gesummv_n256.png); and
- [GESUMMV N=512](solver_frontier_edge_mi300a_final_gesummv_n512.json) and its
  [plot](solver_frontier_edge_mi300a_final_gesummv_n512.png).

## Representation failure exposed by search

An earlier exploratory GESUMMV search used temporal windows formed before the
complete-trace/global-slot contract was corrected. Its candidate timings are
still valid even though its analytical scores are obsolete. It found standard
mappings at 0.036000 ms (`N=256`) and 0.070907 ms (`N=512`), making the best
candidates from the corrected model 13.036% and 13.426% slower.

When those measured mappings are rescored, they are dominated even in the
complete corrected additive universal feature basis. No nonnegative tau over
that basis can recover them. This is evidence for an orthogonal interaction—
plausibly cross-array cache-set/channel balance or allocation-base phase—not
for another kernel-specific locality weight. The diagnostic raw reports are
[N=256](solver_frontier_edge_mi300a_precontract_gesummv_n256.json) and
[N=512](solver_frontier_edge_mi300a_precontract_gesummv_n512.json).

## Conclusions and next experiment

The proof of concept meets the requested 5--7 candidate budget on eight of ten
canonical-corpus instances and stays below 1% regret on all ten using exactly
one MI300A tau. It also generates meaningful speedups over row-major layouts.
The two clear next issues are:

1. add a justified cross-allocation/channel-balance feature and test whether it
   resolves the repeated GESUMMV full-basis dominance failure; and
2. improve compression or adopt a globally validated budget/tie policy for
   SYRK, whose ordinary 15-layout frontier is accurate but too large (and whose
   3-layout fine-gated alternative trades away exact recall).

Before fitting another device, kernel traces must accept that launch's SIMD
width explicitly; the current traces describe a 64-lane MI300A schedule.
Also, the present tau deliberately excludes a non-affine SIMD-window-16 cell
to keep `G_A` applicable. Once affine search can cost arbitrary edge sets, tau
should be recalibrated without that solver-capability restriction.
