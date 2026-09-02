# LinearLayout hardware-hierarchy experiment on MI300A

This experiment extends the Stage 1.5 prepacked-B GEMM graph from its original
128-byte lane-issue edges to a hierarchy of LinearLayout-induced components.
The 512-square GEMM uses a 32x32x32 tile, four waves, and four FP16 register
values per lane.

## Construction and weights

Each edge fixes the hardware coordinates above its scope and varies the
coordinates owned by that scope. The warp and CTA scopes are cumulative: a
warp edge contains all register and lane values owned by one warp, while a CTA
edge contains all register, lane, and warp values in the program instance.

| Scope | Varying hardware coordinates | Fiber count | Scale and tau |
|---|---|---:|---|
| Register ownership | register | 256 | 8 B: 0.25; 16 B: 0.25; 32 B: 0.25 |
| Lane issue | lane | 16 | 64 B: 1.0; 128 B: 1.0; 256 B: 0.5 |
| Warp fragment | register, lane | 4 | 256 B: 0.125; 512 B: 1.0 |
| CTA fragment | register, lane, warp | 1 | 1024 B: 0.0625; 2048 B: 0.03125 |

The 128-byte lane entry is the pre-existing issue component. For component
`s` and scale `d`, the combined objective is

```text
sum_s,d tau[s,d] * (quotient[s,d] / packing_bound[s,d] - 1).
```

The `common-sense` profile starts with scale-decaying weights. The reported
`mi300a` profile was selected using the previously collected eight-canonical-
layout counter panel. On that tuning panel its Spearman correlations were
0.371 with ordinary runtime, 0.575 with profiled duration, 0.898 with L1-to-L2
read requests, and 0.814 with L2 tag requests.

The search also adds the six permutations of the nonempty register, lane, and
warp LinearLayout basis groups as explicit candidates. This matters: the
canonical word grammar can score the hierarchy but cannot put the individual
hardware directions in the discovered lane-first order.

## Fresh validation result

The hierarchy selects `hardware_basis_lane_register_warp`. It reaches the
packing lower bound at every lane, warp, and CTA component, while paying a
normalized excess of 3 at each of the three register scales. Its combined
score is therefore 2.25. The prior canonical winner scores 6.5 and row-major
scores 22.28125.

| Metric | Lane-first selected | Prior canonical winner | Row-major |
|---|---:|---:|---:|
| 128-byte issue quotient | 65,536 | 262,144 | 524,288 |
| Combined hierarchy score | 2.25 | 6.5 | 22.28125 |
| Aggregated runtime | 16.165 us | 16.201 us | 16.210 us |

The selected layout is 1.0028x faster than row-major, but 0.213% slower than
the fastest retained candidate. Across the seven distinct physical mappings,
the fresh runtime Spearman rho is -0.059 for the hierarchy score and -0.139
for the issue quotient. Thus the much better quotient does not improve
ordinary warm-cache runtime prediction for this very short kernel.

The isolated counter panel tells a clearer locality story:

| Counter or metric | Hierarchy-score rho | Issue-quotient rho | Selected vs. row-major |
|---|---:|---:|---:|
| Profiled duration | 0.795 | 0.804 | 1.199x speedup |
| L1-to-L2 read requests | 0.867 | 0.896 | 32.17% reduction |
| L2 tag requests | 0.867 | 0.896 | 27.11% reduction |
| HBM read bytes | 0.617 | 0.624 | 0.016% reduction |
| L2 misses | 0.113 | 0.114 | unchanged at the median |

These correlations remove the duplicate canonical/hardware-basis realization
of one physical mapping; the raw JSON reports the eight profiled candidates.
The hierarchy therefore helps expose and select a mapping with substantially
better cache-request behavior and a 4x lower issue quotient than the old
canonical winner. Relative to that old winner, the lane-first mapping reduces
L1-to-L2 reads by 1.29%, L2 tag requests by 1.08%, and profiled duration by
7.89% (1.086x), while ordinary runtime improves by only 0.22%. It does not yet
beat the single issue quotient as a counter correlate on the fresh panel, nor
does it produce a meaningful improvement in unprofiled end-to-end runtime.

Raw reports:

- `stage15-gemm-hardware-hierarchy-mi300a.json`
- `stage15-gemm-hardware-hierarchy-counter-panel-mi300a.json`
