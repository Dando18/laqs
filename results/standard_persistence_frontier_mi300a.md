# Temporal quotient persistence frontier experiment

This score-only experiment reuses the exhaustive measured G_S corpus; no new timings were collected. J_place is the frozen corrected robust statistic from the input plan.

## Cross-kernel objective tradeoff

A shared combination uses the same objective coordinates for every kernel and size. Worst regret and maximum frontier size are taken over all reported instances.

| Boundary | Objectives | Max samples | Mean samples | Worst regret |
|---|---|---:|---:|---:|
| `smallest_below_one_percent` | `Q_fine, J_persist.d1.64B` | 182 | 93.40 | 0.077% |
| `best_below_ten_samples` | `J_area, J_persist.d16.32768B` | 9 | 5.10 | 13.077% |
| `best_below_five_samples` | `J_persist` | 2 | 1.40 | 146.031% |

### Best shared combination per kernel

| Kernel | Boundary | Objectives | Max samples | Worst regret |
|---|---|---|---:|---:|
| `atax` | `smallest_below_one_percent` | `J_persist.d1.128B` | 20 | 0.000% |
| `atax` | `best_below_ten_samples` | `J_area, J_persist.2048B` | 4 | 1.169% |
| `atax` | `best_below_five_samples` | `J_area, J_persist.2048B` | 4 | 1.169% |
| `gemm` | `smallest_below_one_percent` | `J_area, J_persist.d1.1024B` | 2 | 0.204% |
| `gemm` | `best_below_ten_samples` | `J_place, J_persist.d1` | 4 | 0.055% |
| `gemm` | `best_below_five_samples` | `J_place, J_persist.d1` | 4 | 0.055% |
| `gesummv` | `smallest_below_one_percent` | `J_peak, J_persist.32B` | 10 | 0.000% |
| `gesummv` | `best_below_ten_samples` | `J_area, J_persist.32768B` | 9 | 6.537% |
| `gesummv` | `best_below_five_samples` | `Q_fine, J_persist.32768B` | 4 | 15.189% |
| `mvt` | `smallest_below_one_percent` | `J_persist.d1` | 1 | 0.539% |
| `mvt` | `best_below_ten_samples` | `J_peak, J_persist.simd_schedule.32B` | 9 | 0.211% |
| `mvt` | `best_below_five_samples` | `J_persist.d1` | 1 | 0.539% |
| `syrk` | `smallest_below_one_percent` | `J_peak, J_persist.32768B` | 3 | 0.000% |
| `syrk` | `best_below_ten_samples` | `J_peak, J_persist.32768B` | 3 | 0.000% |
| `syrk` | `best_below_five_samples` | `J_peak, J_persist.32768B` | 3 | 0.000% |

## ATAX N=512

Oracle median: 0.119627 ms over 146 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 4 | 0.858% | no |
| `locality_plus_persist` | 16 | 0.858% | no |
| `locality_plus_place` | 24 | 0.802% | no |
| `all_five` | 25 | 0.802% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiiiiiiiiijjjjjjj` | 0.000% | 2746656.000000 |
| `current_selection` | `iiijjjjiiiiiijjjjj` | 10.644% | 2546808.000000 |
| `near_oracle_dominator` | `iijjiiiiiiijjjjjjj` | 2.942% | 2746656.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_persist.d1` | 2 | 0.802% |
| `J_persist.lane_stream.d1` | 2 | 0.802% |
| `J_persist.mean.d1` | 2 | 0.802% |
| `J_persist.simd_schedule.d1` | 2 | 0.802% |
| `J_persist.simd_stream.d1` | 2 | 0.802% |
| `J_peak, J_persist.d1` | 2 | 0.802% |
| `J_peak, J_persist.lane_stream.d1` | 2 | 0.802% |
| `J_peak, J_persist.mean.d1` | 2 | 0.802% |
| `J_peak, J_persist.simd_schedule.d1` | 2 | 0.802% |
| `J_peak, J_persist.simd_stream.d1` | 2 | 0.802% |
| `Q_fine, J_persist.d1` | 2 | 0.802% |
| `Q_fine, J_persist.lane_stream.d1` | 2 | 0.802% |
| `Q_fine, J_persist.mean.d1` | 2 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d1` | 2 | 0.802% |
| `Q_fine, J_persist.simd_stream.d1` | 2 | 0.802% |
| `Q_fine, J_peak, J_persist.d1` | 2 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 2 | 0.802% |
| `Q_fine, J_peak, J_persist.mean.d1` | 2 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1` | 2 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 2 | 0.802% |
| `Q_fine, J_persist.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.lane_stream.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.lane_stream.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.simd_stream.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_persist.simd_stream.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.2048B` | 4 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.2048B` | 4 | 0.802% |
| `J_peak, J_persist.d1.2048B` | 6 | 0.802% |
| `J_peak, J_persist.d4.2048B` | 6 | 0.802% |
| `J_peak, J_persist.lane_stream.d1.2048B` | 6 | 0.802% |
| `J_peak, J_persist.lane_stream.d4.2048B` | 6 | 0.802% |
| `J_peak, J_persist.simd_schedule.d1.2048B` | 6 | 0.802% |
| `J_peak, J_persist.simd_schedule.d4.2048B` | 6 | 0.802% |
| `J_peak, J_persist.simd_stream.d1.2048B` | 6 | 0.802% |
| `J_peak, J_persist.simd_stream.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.802% |
| `J_area, J_persist.d1` | 8 | 0.802% |
| `J_area, J_persist.lane_stream.d1` | 8 | 0.802% |
| `J_area, J_persist.mean.d1` | 8 | 0.802% |
| `J_area, J_persist.simd_schedule.d1` | 8 | 0.802% |
| `J_area, J_persist.simd_stream.d1` | 8 | 0.802% |
| `Q_fine, J_persist.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.lane_stream.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.lane_stream.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.lane_stream.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.lane_stream.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_schedule.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_stream.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_stream.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_stream.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_persist.simd_stream.d4.4096B` | 8 | 0.802% |
| `J_peak, J_area, J_persist.d1` | 8 | 0.802% |
| `J_peak, J_area, J_persist.lane_stream.d1` | 8 | 0.802% |
| `J_peak, J_area, J_persist.mean.d1` | 8 | 0.802% |
| `J_peak, J_area, J_persist.simd_schedule.d1` | 8 | 0.802% |
| `J_peak, J_area, J_persist.simd_stream.d1` | 8 | 0.802% |
| `Q_fine, J_area, J_persist.d1` | 8 | 0.802% |
| `Q_fine, J_area, J_persist.lane_stream.d1` | 8 | 0.802% |
| `Q_fine, J_area, J_persist.mean.d1` | 8 | 0.802% |
| `Q_fine, J_area, J_persist.simd_schedule.d1` | 8 | 0.802% |
| `Q_fine, J_area, J_persist.simd_stream.d1` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.1024B` | 8 | 0.802% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.4096B` | 8 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.d1` | 8 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1` | 8 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.mean.d1` | 8 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1` | 8 | 0.802% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1` | 8 | 0.802% |
| `J_area, J_persist.2048B` | 4 | 0.858% |
| `J_area, J_persist.4096B` | 4 | 0.858% |
| `J_area, J_persist.d1.128B` | 4 | 0.858% |
| `J_area, J_persist.d1.64B` | 4 | 0.858% |
| `J_area, J_persist.lane_stream.2048B` | 4 | 0.858% |
| `J_area, J_persist.lane_stream.4096B` | 4 | 0.858% |
| `J_area, J_persist.lane_stream.d1.128B` | 4 | 0.858% |
| `J_area, J_persist.lane_stream.d1.64B` | 4 | 0.858% |
| `J_area, J_persist.simd_schedule.2048B` | 4 | 0.858% |
| `J_area, J_persist.simd_schedule.4096B` | 4 | 0.858% |
| `J_area, J_persist.simd_schedule.d1.128B` | 4 | 0.858% |
| `J_area, J_persist.simd_schedule.d1.64B` | 4 | 0.858% |
| `J_area, J_persist.simd_stream.2048B` | 4 | 0.858% |
| `J_area, J_persist.simd_stream.4096B` | 4 | 0.858% |
| `J_area, J_persist.simd_stream.d1.128B` | 4 | 0.858% |
| `J_area, J_persist.simd_stream.d1.64B` | 4 | 0.858% |
| `Q_fine, J_area` | 4 | 0.858% |
| `Q_fine, J_persist.d1.512B` | 4 | 0.858% |
| `Q_fine, J_persist.d4.512B` | 4 | 0.858% |
| `Q_fine, J_persist.lane_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_persist.lane_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_persist.simd_schedule.d1.512B` | 4 | 0.858% |
| `Q_fine, J_persist.simd_schedule.d4.512B` | 4 | 0.858% |
| `Q_fine, J_persist.simd_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_persist.simd_stream.d4.512B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.2048B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.4096B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.d1.128B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.d1.64B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.2048B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.4096B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.d1.128B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.d1.64B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.2048B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.4096B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.d1.128B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.d1.64B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.2048B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.4096B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.d1.128B` | 4 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.d1.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.2048B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.4096B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d1.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d1.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d1.256B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d1.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d1.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d16.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d16.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d16.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d16.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d4.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d4.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d4.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.d4.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.2048B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.4096B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.256B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d16.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d16.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d16.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d16.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d4.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d4.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d4.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.2048B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.4096B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.256B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.2048B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.4096B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.256B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d16.128B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d16.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d16.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d16.64B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d4.16B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d4.32B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d4.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.2048B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.4096B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.256B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d16.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d16.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d16.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d16.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d4.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d4.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d4.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.2048B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.4096B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.256B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.2048B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.4096B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.256B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.2048B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.4096B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.256B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.128B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.64B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.1024B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.16B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.32B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.512B` | 4 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.64B` | 4 | 0.858% |
| `J_area, J_persist.d16.32768B` | 6 | 0.858% |
| `J_area, J_persist.d16.8192B` | 6 | 0.858% |
| `J_area, J_persist.lane_stream.d16.32768B` | 6 | 0.858% |
| `J_area, J_persist.lane_stream.d16.8192B` | 6 | 0.858% |
| `J_area, J_persist.simd_schedule.d16.32768B` | 6 | 0.858% |
| `J_area, J_persist.simd_schedule.d16.8192B` | 6 | 0.858% |
| `J_area, J_persist.simd_stream.d16.32768B` | 6 | 0.858% |
| `J_area, J_persist.simd_stream.d16.8192B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.d16.32768B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.d16.8192B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.d16.32768B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.d16.8192B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.d16.8192B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.d16.32768B` | 6 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.d16.8192B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.128B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.512B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.d1.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.128B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.512B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d1.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.128B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.512B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.128B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.32B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.512B` | 6 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d1.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.128B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.512B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d1.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.128B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.512B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.128B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.512B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.128B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.32B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.512B` | 6 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.32B` | 6 | 0.858% |
| `J_area, J_persist.d16.16384B` | 8 | 0.858% |
| `J_area, J_persist.lane_stream.d16.16384B` | 8 | 0.858% |
| `J_area, J_persist.simd_schedule.d16.16384B` | 8 | 0.858% |
| `J_area, J_persist.simd_stream.d16.16384B` | 8 | 0.858% |
| `Q_fine, J_persist.2048B` | 8 | 0.858% |
| `Q_fine, J_persist.4096B` | 8 | 0.858% |
| `Q_fine, J_persist.lane_stream.2048B` | 8 | 0.858% |
| `Q_fine, J_persist.lane_stream.4096B` | 8 | 0.858% |
| `Q_fine, J_persist.simd_schedule.2048B` | 8 | 0.858% |
| `Q_fine, J_persist.simd_schedule.4096B` | 8 | 0.858% |
| `Q_fine, J_persist.simd_stream.2048B` | 8 | 0.858% |
| `Q_fine, J_persist.simd_stream.4096B` | 8 | 0.858% |
| `J_peak, J_area, J_persist.d16.16384B` | 8 | 0.858% |
| `J_peak, J_area, J_persist.lane_stream.d16.16384B` | 8 | 0.858% |
| `J_peak, J_area, J_persist.simd_schedule.d16.16384B` | 8 | 0.858% |
| `J_peak, J_area, J_persist.simd_stream.d16.16384B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.1024B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.1024B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.lane_stream.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.1024B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.1024B` | 8 | 0.858% |
| `Q_fine, J_area, J_persist.simd_stream.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.2048B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.4096B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.lane_stream.2048B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.lane_stream.4096B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_schedule.2048B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_schedule.4096B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_stream.2048B` | 8 | 0.858% |
| `Q_fine, J_peak, J_persist.simd_stream.4096B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.1024B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.1024B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.1024B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.8192B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.1024B` | 8 | 0.858% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.8192B` | 8 | 0.858% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_persist.d1` | 2 | 0.802% |
| `best_below_ten_samples` | `J_persist.d1` | 2 | 0.802% |
| `best_below_five_samples` | `J_persist.d1` | 2 | 0.802% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_persist.d1` | 2 | 0.802% |
| `J_persist.lane_stream.d1` | 2 | 0.802% |
| `J_persist.mean.d1` | 2 | 0.802% |
| `J_persist.simd_schedule.d1` | 2 | 0.802% |
| `J_persist.simd_stream.d1` | 2 | 0.802% |
| `J_peak, J_persist.d1` | 2 | 0.802% |
| `J_peak, J_persist.lane_stream.d1` | 2 | 0.802% |
| `J_peak, J_persist.mean.d1` | 2 | 0.802% |
| `J_peak, J_persist.simd_schedule.d1` | 2 | 0.802% |
| `J_peak, J_persist.simd_stream.d1` | 2 | 0.802% |

## ATAX N=1024

Oracle median: 0.239680 ms over 182 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 4 | 1.169% | no |
| `locality_plus_persist` | 16 | 1.169% | no |
| `locality_plus_place` | 35 | 1.230% | no |
| `all_five` | 37 | 1.230% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiijjjjjjjjiiiiiiii` | 0.000% | 5641548.000000 |
| `current_selection` | `iiijjjjiiiiiiijjjjjj` | 7.867% | 5160144.000000 |
| `near_oracle_dominator` | `jjiiijjjjjjjjiiiiiii` | 1.169% | 5267064.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| _None_ | — | — |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_persist.d1.128B` | 20 | 0.000% |
| `best_below_ten_samples` | `J_area, J_persist.2048B` | 4 | 1.169% |
| `best_below_five_samples` | `J_area, J_persist.2048B` | 4 | 1.169% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_area, J_persist.2048B` | 4 | 1.169% |
| `J_area, J_persist.4096B` | 4 | 1.169% |
| `J_area, J_persist.d1.128B` | 4 | 1.169% |
| `J_area, J_persist.d1.64B` | 4 | 1.169% |
| `J_area, J_persist.lane_stream.2048B` | 4 | 1.169% |
| `J_area, J_persist.lane_stream.4096B` | 4 | 1.169% |
| `J_area, J_persist.lane_stream.d1.128B` | 4 | 1.169% |
| `J_area, J_persist.lane_stream.d1.64B` | 4 | 1.169% |
| `J_area, J_persist.simd_schedule.2048B` | 4 | 1.169% |
| `J_area, J_persist.simd_schedule.4096B` | 4 | 1.169% |

## GEMM N=512

Oracle median: 0.263214 ms over 146 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 10 | 0.147% | no |
| `locality_plus_persist` | 3 | 0.147% | no |
| `locality_plus_place` | 11 | 8.809% | no |
| `all_five` | 30 | 0.147% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiiijjjjjjjiiiiii` | 0.000% | 20909328.000000 |
| `current_selection` | `jijjjjjjjjiiiiiiii` | 17.643% | 21708016.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_area, J_persist.d1.512B` | 3 | 0.000% |
| `J_area, J_persist.lane_stream.d1.512B` | 3 | 0.000% |
| `J_area, J_persist.simd_schedule.d4.512B` | 3 | 0.000% |
| `J_area, J_persist.simd_stream.d1.512B` | 3 | 0.000% |
| `J_place, J_persist.d1` | 4 | 0.055% |
| `J_place, J_persist.lane_stream.d1` | 4 | 0.055% |
| `J_place, J_persist.mean.d1` | 4 | 0.055% |
| `J_place, J_persist.simd_stream.d1` | 4 | 0.055% |
| `Q_fine, J_place, J_persist.d1` | 8 | 0.055% |
| `Q_fine, J_place, J_persist.lane_stream.d1` | 8 | 0.055% |
| `Q_fine, J_place, J_persist.mean.d1` | 8 | 0.055% |
| `Q_fine, J_place, J_persist.simd_stream.d1` | 8 | 0.055% |
| `J_peak, J_place, J_persist.d1` | 9 | 0.055% |
| `J_peak, J_place, J_persist.lane_stream.d1` | 9 | 0.055% |
| `J_peak, J_place, J_persist.mean.d1` | 9 | 0.055% |
| `J_peak, J_place, J_persist.simd_stream.d1` | 9 | 0.055% |
| `Q_fine, J_peak, J_place, J_persist.d1` | 9 | 0.055% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1` | 9 | 0.055% |
| `Q_fine, J_peak, J_place, J_persist.mean.d1` | 9 | 0.055% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1` | 9 | 0.055% |
| `J_area, J_persist.d1.1024B` | 2 | 0.081% |
| `J_area, J_persist.lane_stream.d1.1024B` | 2 | 0.081% |
| `J_area, J_persist.simd_schedule.d4.1024B` | 2 | 0.081% |
| `J_area, J_persist.simd_stream.d1.1024B` | 2 | 0.081% |
| `J_area, J_persist.d4.1024B` | 3 | 0.081% |
| `J_area, J_persist.lane_stream.d4.1024B` | 3 | 0.081% |
| `J_area, J_persist.simd_stream.d4.1024B` | 3 | 0.081% |
| `Q_fine, J_persist.simd_schedule.512B` | 9 | 0.081% |
| `Q_fine, J_area, J_persist.d1.1024B` | 9 | 0.081% |
| `Q_fine, J_area, J_persist.lane_stream.d1.1024B` | 9 | 0.081% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.1024B` | 9 | 0.081% |
| `Q_fine, J_area, J_persist.simd_stream.d1.1024B` | 9 | 0.081% |
| `Q_fine, J_persist` | 1 | 0.147% |
| `Q_fine, J_persist.lane_stream` | 1 | 0.147% |
| `Q_fine, J_persist.mean.lane_stream` | 1 | 0.147% |
| `Q_fine, J_persist.mean.simd_stream` | 1 | 0.147% |
| `Q_fine, J_persist.mean_cells` | 1 | 0.147% |
| `Q_fine, J_persist.simd_schedule.d16` | 1 | 0.147% |
| `Q_fine, J_persist.simd_stream` | 1 | 0.147% |
| `J_persist` | 2 | 0.147% |
| `J_persist.lane_stream` | 2 | 0.147% |
| `J_persist.mean.lane_stream` | 2 | 0.147% |
| `J_persist.mean.simd_stream` | 2 | 0.147% |
| `J_persist.mean_cells` | 2 | 0.147% |
| `J_persist.simd_schedule.d16` | 2 | 0.147% |
| `J_persist.simd_stream` | 2 | 0.147% |
| `J_area, J_persist` | 2 | 0.147% |
| `J_area, J_persist.32768B` | 2 | 0.147% |
| `J_area, J_persist.d1.32768B` | 2 | 0.147% |
| `J_area, J_persist.d16.32768B` | 2 | 0.147% |
| `J_area, J_persist.d4.32768B` | 2 | 0.147% |
| `J_area, J_persist.lane_stream` | 2 | 0.147% |
| `J_area, J_persist.lane_stream.32768B` | 2 | 0.147% |
| `J_area, J_persist.lane_stream.d1.32768B` | 2 | 0.147% |
| `J_area, J_persist.lane_stream.d16.32768B` | 2 | 0.147% |
| `J_area, J_persist.lane_stream.d4.32768B` | 2 | 0.147% |
| `J_area, J_persist.mean.lane_stream` | 2 | 0.147% |
| `J_area, J_persist.mean.simd_stream` | 2 | 0.147% |
| `J_area, J_persist.mean_cells` | 2 | 0.147% |
| `J_area, J_persist.simd_schedule.32768B` | 2 | 0.147% |
| `J_area, J_persist.simd_schedule.d16` | 2 | 0.147% |
| `J_area, J_persist.simd_schedule.d16.32768B` | 2 | 0.147% |
| `J_area, J_persist.simd_schedule.d4.32768B` | 2 | 0.147% |
| `J_area, J_persist.simd_stream` | 2 | 0.147% |
| `J_area, J_persist.simd_stream.32768B` | 2 | 0.147% |
| `J_area, J_persist.simd_stream.d1.32768B` | 2 | 0.147% |
| `J_area, J_persist.simd_stream.d16.32768B` | 2 | 0.147% |
| `J_area, J_persist.simd_stream.d4.32768B` | 2 | 0.147% |
| `J_peak, J_persist` | 2 | 0.147% |
| `J_peak, J_persist.lane_stream` | 2 | 0.147% |
| `J_peak, J_persist.mean.lane_stream` | 2 | 0.147% |
| `J_peak, J_persist.mean.simd_stream` | 2 | 0.147% |
| `J_peak, J_persist.mean_cells` | 2 | 0.147% |
| `J_peak, J_persist.simd_stream` | 2 | 0.147% |
| `Q_fine, J_area, J_persist` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.d1.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.d16.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.d4.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.mean.lane_stream` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.mean.simd_stream` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.mean_cells` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.32768B` | 2 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.32768B` | 2 | 0.147% |
| `Q_fine, J_peak, J_persist` | 2 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream` | 2 | 0.147% |
| `Q_fine, J_peak, J_persist.mean.lane_stream` | 2 | 0.147% |
| `Q_fine, J_peak, J_persist.mean.simd_stream` | 2 | 0.147% |
| `Q_fine, J_peak, J_persist.mean_cells` | 2 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream` | 2 | 0.147% |
| `J_area, J_persist.d4` | 3 | 0.147% |
| `J_area, J_persist.lane_stream.d4` | 3 | 0.147% |
| `J_area, J_persist.mean.d4` | 3 | 0.147% |
| `J_area, J_persist.mean.simd_schedule` | 3 | 0.147% |
| `J_area, J_persist.simd_schedule` | 3 | 0.147% |
| `J_area, J_persist.simd_stream.d4` | 3 | 0.147% |
| `J_peak, J_persist.simd_schedule.d16` | 3 | 0.147% |
| `J_peak, J_area, J_persist` | 3 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream` | 3 | 0.147% |
| `J_peak, J_area, J_persist.mean.lane_stream` | 3 | 0.147% |
| `J_peak, J_area, J_persist.mean.simd_stream` | 3 | 0.147% |
| `J_peak, J_area, J_persist.mean_cells` | 3 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream` | 3 | 0.147% |
| `Q_fine, J_area, J_persist.d4` | 3 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4` | 3 | 0.147% |
| `Q_fine, J_area, J_persist.mean.d4` | 3 | 0.147% |
| `Q_fine, J_area, J_persist.mean.simd_schedule` | 3 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule` | 3 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4` | 3 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16` | 3 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist` | 3 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream` | 3 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.mean.lane_stream` | 3 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.mean.simd_stream` | 3 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.mean_cells` | 3 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream` | 3 | 0.147% |
| `J_peak, J_area, J_persist.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.d1.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.d16.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.d4` | 4 | 0.147% |
| `J_peak, J_area, J_persist.d4.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d1.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d16.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d4.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.mean.d4` | 4 | 0.147% |
| `J_peak, J_area, J_persist.mean.simd_schedule` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d16` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d4.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d1.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d16.32768B` | 4 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d4.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d1.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d16.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d4` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d4.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.mean.d4` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.mean.simd_schedule` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.32768B` | 4 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.32768B` | 4 | 0.147% |
| `J_area, J_persist.simd_schedule.d4` | 5 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d4` | 5 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d4` | 5 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4` | 5 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4` | 5 | 0.147% |
| `J_area, J_persist.mean.d16` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.d1.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.d1.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.d16.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.d16.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.d4.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.d4.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.mean.d16` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.8192B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.16384B` | 6 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.8192B` | 6 | 0.147% |
| `J_area, J_persist.d16` | 7 | 0.147% |
| `J_area, J_persist.lane_stream.d16` | 7 | 0.147% |
| `J_area, J_persist.simd_stream.d16` | 7 | 0.147% |
| `Q_fine, J_persist.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.d1.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.d1.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.d16.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.d16.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.d4.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.d4.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.d1.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.d1.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.d16.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.d16.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.d4.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.lane_stream.d4.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_schedule.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_schedule.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_schedule.d16.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_schedule.d16.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_schedule.d4.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_schedule.d4.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.d1.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.d1.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.d16.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.d16.8192B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.d4.32768B` | 7 | 0.147% |
| `Q_fine, J_persist.simd_stream.d4.8192B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.d1.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.d16` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.d4.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.4096B` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16` | 7 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.4096B` | 7 | 0.147% |
| `J_area, J_persist.256B` | 8 | 0.147% |
| `J_area, J_persist.64B` | 8 | 0.147% |
| `J_area, J_persist.d16.128B` | 8 | 0.147% |
| `J_area, J_persist.d16.256B` | 8 | 0.147% |
| `J_area, J_persist.d4.128B` | 8 | 0.147% |
| `J_area, J_persist.d4.256B` | 8 | 0.147% |
| `J_area, J_persist.d4.32B` | 8 | 0.147% |
| `J_area, J_persist.d4.64B` | 8 | 0.147% |
| `J_area, J_persist.lane_stream.256B` | 8 | 0.147% |
| `J_area, J_persist.lane_stream.64B` | 8 | 0.147% |
| `J_area, J_persist.lane_stream.d16.256B` | 8 | 0.147% |
| `J_area, J_persist.lane_stream.d4.128B` | 8 | 0.147% |
| `J_area, J_persist.lane_stream.d4.256B` | 8 | 0.147% |
| `J_area, J_persist.lane_stream.d4.64B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.128B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.256B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.32B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.64B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.d16.128B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.d16.256B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.d4.32B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.d4.64B` | 8 | 0.147% |
| `J_area, J_persist.simd_stream.256B` | 8 | 0.147% |
| `J_area, J_persist.simd_stream.64B` | 8 | 0.147% |
| `J_area, J_persist.simd_stream.d16.256B` | 8 | 0.147% |
| `J_area, J_persist.simd_stream.d4.128B` | 8 | 0.147% |
| `J_area, J_persist.simd_stream.d4.256B` | 8 | 0.147% |
| `J_area, J_persist.simd_stream.d4.64B` | 8 | 0.147% |
| `Q_fine, J_area` | 8 | 0.147% |
| `J_peak, J_area, J_persist.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.d1.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.d1.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.d16.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.d16.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.d4.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.d4.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d1.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d1.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d16.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d16.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d4.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d4.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.mean.d16` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d16.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d16.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d4.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d4.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d1.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d1.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d16.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d16.8192B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d4.16384B` | 8 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d4.8192B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.2048B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.512B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d1.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d1.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d1.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d16.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d16.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d16.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d16.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d16.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d4.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d4.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d4.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d4.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.d4.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.2048B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.512B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d1.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d16.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.d4.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.1024B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.1024B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16384B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.2048B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.4096B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.512B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.8192B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.2048B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.512B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d1.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d16.64B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.128B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.16B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.256B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.32B` | 8 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.d4.64B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d1.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d1.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d16.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d16.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d4.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d4.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.mean.d16` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.8192B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.16384B` | 8 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.8192B` | 8 | 0.147% |
| `J_area, J_persist.simd_schedule.d4.128B` | 9 | 0.147% |
| `J_area, J_persist.simd_schedule.d4.256B` | 9 | 0.147% |
| `J_peak, J_persist.32768B` | 9 | 0.147% |
| `J_peak, J_persist.8192B` | 9 | 0.147% |
| `J_peak, J_persist.d1.32768B` | 9 | 0.147% |
| `J_peak, J_persist.d1.8192B` | 9 | 0.147% |
| `J_peak, J_persist.d16.32768B` | 9 | 0.147% |
| `J_peak, J_persist.d16.8192B` | 9 | 0.147% |
| `J_peak, J_persist.d4.32768B` | 9 | 0.147% |
| `J_peak, J_persist.d4.8192B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.32768B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.8192B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.d1.32768B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.d1.8192B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.d16.32768B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.d16.8192B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.d4.32768B` | 9 | 0.147% |
| `J_peak, J_persist.lane_stream.d4.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_schedule.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_schedule.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_schedule.d16.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_schedule.d16.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_schedule.d4.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_schedule.d4.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.d1.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.d1.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.d16.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.d16.8192B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.d4.32768B` | 9 | 0.147% |
| `J_peak, J_persist.simd_stream.d4.8192B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.d1.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.d1.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.d16` | 9 | 0.147% |
| `J_peak, J_area, J_persist.d4.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.d4.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d1.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d1.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d16` | 9 | 0.147% |
| `J_peak, J_area, J_persist.lane_stream.d4.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d16.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d4` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d4.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d1.32B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d1.4096B` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d16` | 9 | 0.147% |
| `J_peak, J_area, J_persist.simd_stream.d4.4096B` | 9 | 0.147% |
| `Q_fine, J_area, J_persist.128B` | 9 | 0.147% |
| `Q_fine, J_area, J_persist.lane_stream.128B` | 9 | 0.147% |
| `Q_fine, J_area, J_persist.simd_stream.128B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.d1.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.d1.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.d16.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.d16.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.d4.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.d4.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.d16.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.d16.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.d16.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.d16.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.32768B` | 9 | 0.147% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.8192B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d1.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d1.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d16` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d4.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.d4.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.32B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.4096B` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16` | 9 | 0.147% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.4096B` | 9 | 0.147% |
| `J_area, J_persist.4096B` | 2 | 0.152% |
| `J_area, J_persist.lane_stream.4096B` | 2 | 0.152% |
| `J_area, J_persist.simd_stream.4096B` | 2 | 0.152% |
| `Q_fine, J_area, J_persist.4096B` | 2 | 0.152% |
| `Q_fine, J_area, J_persist.lane_stream.4096B` | 2 | 0.152% |
| `Q_fine, J_area, J_persist.simd_stream.4096B` | 2 | 0.152% |
| `J_peak, J_area, J_persist.4096B` | 4 | 0.152% |
| `J_peak, J_area, J_persist.lane_stream.4096B` | 4 | 0.152% |
| `J_peak, J_area, J_persist.simd_stream.4096B` | 4 | 0.152% |
| `Q_fine, J_peak, J_area, J_persist.4096B` | 4 | 0.152% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.4096B` | 4 | 0.152% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.4096B` | 4 | 0.152% |
| `J_peak, J_area, J_persist.d16.4096B` | 6 | 0.152% |
| `J_peak, J_area, J_persist.lane_stream.d16.4096B` | 6 | 0.152% |
| `J_peak, J_area, J_persist.simd_stream.d16.4096B` | 6 | 0.152% |
| `Q_fine, J_peak, J_area, J_persist.d16.4096B` | 7 | 0.152% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.4096B` | 7 | 0.152% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.4096B` | 7 | 0.152% |
| `J_area, J_persist.d16.4096B` | 8 | 0.152% |
| `J_area, J_persist.lane_stream.d16.4096B` | 8 | 0.152% |
| `J_area, J_persist.simd_stream.d16.4096B` | 8 | 0.152% |
| `Q_fine, J_area, J_persist.d16.4096B` | 9 | 0.152% |
| `Q_fine, J_area, J_persist.lane_stream.d16.4096B` | 9 | 0.152% |
| `Q_fine, J_area, J_persist.simd_stream.d16.4096B` | 9 | 0.152% |
| `Q_fine, J_persist.d1.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.d4.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.lane_stream.d1.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.lane_stream.d4.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.simd_schedule.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.simd_schedule.d16.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.simd_schedule.d4.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.simd_stream.d1.2048B` | 7 | 0.218% |
| `Q_fine, J_persist.simd_stream.d4.2048B` | 7 | 0.218% |
| `J_peak, J_persist.d1.2048B` | 9 | 0.218% |
| `J_peak, J_persist.d4.2048B` | 9 | 0.218% |
| `J_peak, J_persist.lane_stream.d1.2048B` | 9 | 0.218% |
| `J_peak, J_persist.lane_stream.d4.2048B` | 9 | 0.218% |
| `J_peak, J_persist.simd_schedule.2048B` | 9 | 0.218% |
| `J_peak, J_persist.simd_schedule.d16.2048B` | 9 | 0.218% |
| `J_peak, J_persist.simd_schedule.d4.2048B` | 9 | 0.218% |
| `J_peak, J_persist.simd_stream.d1.2048B` | 9 | 0.218% |
| `J_peak, J_persist.simd_stream.d4.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.d1.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.d4.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.simd_schedule.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.2048B` | 9 | 0.218% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.2048B` | 9 | 0.218% |
| `J_persist.d1` | 2 | 0.238% |
| `J_persist.lane_stream.d1` | 2 | 0.238% |
| `J_persist.mean.d1` | 2 | 0.238% |
| `J_persist.simd_stream.d1` | 2 | 0.238% |
| `Q_fine, J_persist.d1` | 2 | 0.238% |
| `Q_fine, J_persist.lane_stream.d1` | 2 | 0.238% |
| `Q_fine, J_persist.mean.d1` | 2 | 0.238% |
| `Q_fine, J_persist.simd_stream.d1` | 2 | 0.238% |
| `J_peak, J_persist.d1` | 3 | 0.238% |
| `J_peak, J_persist.lane_stream.d1` | 3 | 0.238% |
| `J_peak, J_persist.mean.d1` | 3 | 0.238% |
| `J_peak, J_persist.simd_stream.d1` | 3 | 0.238% |
| `Q_fine, J_peak, J_persist.d1` | 3 | 0.238% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 3 | 0.238% |
| `Q_fine, J_peak, J_persist.mean.d1` | 3 | 0.238% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 3 | 0.238% |
| `Q_fine, J_persist.d1.512B` | 7 | 0.319% |
| `Q_fine, J_persist.d4.512B` | 7 | 0.319% |
| `Q_fine, J_persist.lane_stream.d1.512B` | 7 | 0.319% |
| `Q_fine, J_persist.lane_stream.d4.512B` | 7 | 0.319% |
| `Q_fine, J_persist.simd_schedule.d4.512B` | 7 | 0.319% |
| `Q_fine, J_persist.simd_stream.d1.512B` | 7 | 0.319% |
| `Q_fine, J_persist.simd_stream.d4.512B` | 7 | 0.319% |
| `Q_fine, J_persist.d1.256B` | 8 | 0.319% |
| `Q_fine, J_persist.lane_stream.d1.256B` | 8 | 0.319% |
| `Q_fine, J_persist.simd_schedule.d4.256B` | 8 | 0.319% |
| `Q_fine, J_persist.simd_stream.d1.256B` | 8 | 0.319% |
| `J_peak, J_persist.d1.512B` | 9 | 0.319% |
| `J_peak, J_persist.d4.512B` | 9 | 0.319% |
| `J_peak, J_persist.lane_stream.d1.512B` | 9 | 0.319% |
| `J_peak, J_persist.lane_stream.d4.512B` | 9 | 0.319% |
| `J_peak, J_persist.simd_schedule.d4.512B` | 9 | 0.319% |
| `J_peak, J_persist.simd_stream.d1.512B` | 9 | 0.319% |
| `J_peak, J_persist.simd_stream.d4.512B` | 9 | 0.319% |
| `Q_fine, J_persist.128B` | 9 | 0.319% |
| `Q_fine, J_persist.lane_stream.128B` | 9 | 0.319% |
| `Q_fine, J_persist.simd_stream.128B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.d1.512B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.d4.512B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.512B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.512B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.512B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.512B` | 9 | 0.319% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.512B` | 9 | 0.319% |
| `J_place, J_persist.d16.16384B` | 2 | 0.359% |
| `J_place, J_persist.lane_stream.d16.16384B` | 2 | 0.359% |
| `J_place, J_persist.simd_stream.d16.16384B` | 2 | 0.359% |
| `J_place, J_persist.16384B` | 3 | 0.359% |
| `J_place, J_persist.32768B` | 3 | 0.359% |
| `J_place, J_persist.d1.16384B` | 3 | 0.359% |
| `J_place, J_persist.d1.32768B` | 3 | 0.359% |
| `J_place, J_persist.d16.32768B` | 3 | 0.359% |
| `J_place, J_persist.d4.16384B` | 3 | 0.359% |
| `J_place, J_persist.d4.32768B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.16384B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.32768B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.d1.16384B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.d1.32768B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.d16.32768B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.d4.16384B` | 3 | 0.359% |
| `J_place, J_persist.lane_stream.d4.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.16384B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.d16.16384B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.d16.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.d4.16384B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.d4.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.16384B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.d1.16384B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.d1.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.d16.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.d4.16384B` | 3 | 0.359% |
| `J_place, J_persist.simd_stream.d4.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.d1.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.d1.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.d16.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.d4.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.d4.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d1.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d1.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d16.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d4.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d4.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d1.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d1.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d16.32768B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d4.16384B` | 3 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d4.32768B` | 3 | 0.359% |
| `J_place, J_persist.simd_schedule.d16.8192B` | 4 | 0.359% |
| `J_place, J_persist.simd_schedule.d4` | 4 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d4` | 4 | 0.359% |
| `J_place, J_persist.8192B` | 5 | 0.359% |
| `J_place, J_persist.d1.8192B` | 5 | 0.359% |
| `J_place, J_persist.d4.8192B` | 5 | 0.359% |
| `J_place, J_persist.lane_stream.8192B` | 5 | 0.359% |
| `J_place, J_persist.lane_stream.d1.8192B` | 5 | 0.359% |
| `J_place, J_persist.lane_stream.d4.8192B` | 5 | 0.359% |
| `J_place, J_persist.simd_schedule.8192B` | 5 | 0.359% |
| `J_place, J_persist.simd_schedule.d4.8192B` | 5 | 0.359% |
| `J_place, J_persist.simd_stream.8192B` | 5 | 0.359% |
| `J_place, J_persist.simd_stream.d1.8192B` | 5 | 0.359% |
| `J_place, J_persist.simd_stream.d4.8192B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.d1.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.d1.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.d16.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.d4.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.d4.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d1.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d1.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d16.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d4.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d4.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d16.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d16.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d4` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d4.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d4.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d1.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d1.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d16.32768B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d4.16384B` | 5 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d4.32768B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.d1.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.d16.16384B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.d4.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d1.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d16.16384B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.lane_stream.d4.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d1.8192B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d16.16384B` | 5 | 0.359% |
| `Q_fine, J_place, J_persist.simd_stream.d4.8192B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d1.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d1.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d16.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d4.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d4.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.32768B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.16384B` | 5 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.32768B` | 5 | 0.359% |
| `J_place, J_persist` | 7 | 0.359% |
| `J_place, J_persist.lane_stream` | 7 | 0.359% |
| `J_place, J_persist.mean.lane_stream` | 7 | 0.359% |
| `J_place, J_persist.mean.simd_stream` | 7 | 0.359% |
| `J_place, J_persist.simd_stream` | 7 | 0.359% |
| `J_peak, J_place, J_persist.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.d1.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.d16.16384B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.d4.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d1.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d16.16384B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.lane_stream.d4.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d4.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d1.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d16.16384B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_stream.d4.8192B` | 7 | 0.359% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d1.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d16.16384B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.d4.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.16384B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.8192B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.16384B` | 7 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.8192B` | 7 | 0.359% |
| `J_peak, J_place, J_persist.simd_schedule.d16.8192B` | 9 | 0.359% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.8192B` | 9 | 0.359% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `Q_fine, J_persist` | 1 | 0.147% |
| `best_below_ten_samples` | `J_area, J_persist.d1.512B` | 3 | 0.000% |
| `best_below_five_samples` | `J_area, J_persist.d1.512B` | 3 | 0.000% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_area, J_persist.d1.512B` | 3 | 0.000% |
| `J_area, J_persist.lane_stream.d1.512B` | 3 | 0.000% |
| `J_area, J_persist.simd_schedule.d4.512B` | 3 | 0.000% |
| `J_area, J_persist.simd_stream.d1.512B` | 3 | 0.000% |
| `J_place, J_persist.d1` | 4 | 0.055% |
| `J_place, J_persist.lane_stream.d1` | 4 | 0.055% |
| `J_place, J_persist.mean.d1` | 4 | 0.055% |
| `J_place, J_persist.simd_stream.d1` | 4 | 0.055% |
| `Q_fine, J_place, J_persist.d1` | 8 | 0.055% |
| `Q_fine, J_place, J_persist.lane_stream.d1` | 8 | 0.055% |

## GEMM N=1024

Oracle median: 1.409226 ms over 182 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 11 | 0.204% | no |
| `locality_plus_persist` | 3 | 0.204% | no |
| `locality_plus_place` | 17 | 0.204% | no |
| `all_five` | 38 | 0.204% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiiiiijjjjjjjjiiiii` | 0.000% | 40430464.000000 |
| `current_selection` | `jijjjjjjjjjiiiiiiiii` | 1.210% | 44843776.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_place, J_persist.d1` | 4 | 0.000% |
| `J_place, J_persist.lane_stream.d1` | 4 | 0.000% |
| `J_place, J_persist.mean.d1` | 4 | 0.000% |
| `J_place, J_persist.simd_stream.d1` | 4 | 0.000% |
| `Q_fine, J_place, J_persist.d1` | 8 | 0.000% |
| `Q_fine, J_place, J_persist.lane_stream.d1` | 8 | 0.000% |
| `Q_fine, J_place, J_persist.mean.d1` | 8 | 0.000% |
| `Q_fine, J_place, J_persist.simd_stream.d1` | 8 | 0.000% |
| `J_peak, J_place, J_persist.d1` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d1` | 9 | 0.000% |
| `J_peak, J_place, J_persist.mean.d1` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_place, J_persist.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_place, J_persist.mean.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1` | 9 | 0.000% |
| `Q_fine, J_persist.d1.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.d4.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.lane_stream.d1.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.lane_stream.d4.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.simd_schedule.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.simd_schedule.d16.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.simd_schedule.d4.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.simd_stream.d1.2048B` | 8 | 0.131% |
| `Q_fine, J_persist.simd_stream.d4.2048B` | 8 | 0.131% |
| `J_area, J_persist.simd_schedule.d16.2048B` | 9 | 0.160% |
| `J_area` | 1 | 0.204% |
| `J_area, J_persist.128B` | 1 | 0.204% |
| `J_area, J_persist.16384B` | 1 | 0.204% |
| `J_area, J_persist.16B` | 1 | 0.204% |
| `J_area, J_persist.2048B` | 1 | 0.204% |
| `J_area, J_persist.32B` | 1 | 0.204% |
| `J_area, J_persist.512B` | 1 | 0.204% |
| `J_area, J_persist.8192B` | 1 | 0.204% |
| `J_area, J_persist.d1.16384B` | 1 | 0.204% |
| `J_area, J_persist.d1.16B` | 1 | 0.204% |
| `J_area, J_persist.d1.32B` | 1 | 0.204% |
| `J_area, J_persist.d1.4096B` | 1 | 0.204% |
| `J_area, J_persist.d1.64B` | 1 | 0.204% |
| `J_area, J_persist.d1.8192B` | 1 | 0.204% |
| `J_area, J_persist.d16.16384B` | 1 | 0.204% |
| `J_area, J_persist.d16.16B` | 1 | 0.204% |
| `J_area, J_persist.d16.32B` | 1 | 0.204% |
| `J_area, J_persist.d16.64B` | 1 | 0.204% |
| `J_area, J_persist.d16.8192B` | 1 | 0.204% |
| `J_area, J_persist.d4.16384B` | 1 | 0.204% |
| `J_area, J_persist.d4.16B` | 1 | 0.204% |
| `J_area, J_persist.d4.4096B` | 1 | 0.204% |
| `J_area, J_persist.d4.8192B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.128B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.16384B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.16B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.2048B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.32B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.512B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.8192B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d1.16384B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d1.16B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d1.32B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d1.4096B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d1.64B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d1.8192B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d16.128B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d16.16384B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d16.16B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d16.32B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d16.64B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d16.8192B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d4.16384B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d4.16B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d4.32B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d4.4096B` | 1 | 0.204% |
| `J_area, J_persist.lane_stream.d4.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.1024B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.4096B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.1024B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.128B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.2048B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.256B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.32768B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.32B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.4096B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.512B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.64B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d1.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.32B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.4096B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.64B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.4096B` | 1 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.128B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.2048B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.32B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.512B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d1.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d1.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d1.32B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d1.4096B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d1.64B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d1.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d16.128B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d16.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d16.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d16.32B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d16.64B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d16.8192B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d4.16384B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d4.16B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d4.32B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d4.4096B` | 1 | 0.204% |
| `J_area, J_persist.simd_stream.d4.8192B` | 1 | 0.204% |
| `J_area, J_persist` | 2 | 0.204% |
| `J_area, J_persist.1024B` | 2 | 0.204% |
| `J_area, J_persist.32768B` | 2 | 0.204% |
| `J_area, J_persist.4096B` | 2 | 0.204% |
| `J_area, J_persist.d1.1024B` | 2 | 0.204% |
| `J_area, J_persist.d1.128B` | 2 | 0.204% |
| `J_area, J_persist.d1.2048B` | 2 | 0.204% |
| `J_area, J_persist.d1.256B` | 2 | 0.204% |
| `J_area, J_persist.d1.32768B` | 2 | 0.204% |
| `J_area, J_persist.d16.32768B` | 2 | 0.204% |
| `J_area, J_persist.d4.2048B` | 2 | 0.204% |
| `J_area, J_persist.d4.32768B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.1024B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.32768B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.4096B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d1.1024B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d1.128B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d1.2048B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d1.256B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d1.32768B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d16.32768B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d4.2048B` | 2 | 0.204% |
| `J_area, J_persist.lane_stream.d4.32768B` | 2 | 0.204% |
| `J_area, J_persist.mean.lane_stream` | 2 | 0.204% |
| `J_area, J_persist.mean.simd_stream` | 2 | 0.204% |
| `J_area, J_persist.mean_cells` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.2048B` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.32768B` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.512B` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.d16` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.32768B` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.1024B` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.2048B` | 2 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.32768B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.1024B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.32768B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.4096B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d1.1024B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d1.128B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d1.2048B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d1.256B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d1.32768B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d16.32768B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d4.2048B` | 2 | 0.204% |
| `J_area, J_persist.simd_stream.d4.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.4096B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.d1.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.d16.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.d4.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.4096B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.mean.lane_stream` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.mean.simd_stream` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.mean_cells` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.4096B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.32768B` | 2 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.32768B` | 2 | 0.204% |
| `J_area, J_persist.d1` | 3 | 0.204% |
| `J_area, J_persist.d1.512B` | 3 | 0.204% |
| `J_area, J_persist.d4` | 3 | 0.204% |
| `J_area, J_persist.d4.1024B` | 3 | 0.204% |
| `J_area, J_persist.lane_stream.d1` | 3 | 0.204% |
| `J_area, J_persist.lane_stream.d1.512B` | 3 | 0.204% |
| `J_area, J_persist.lane_stream.d4` | 3 | 0.204% |
| `J_area, J_persist.lane_stream.d4.1024B` | 3 | 0.204% |
| `J_area, J_persist.mean.d1` | 3 | 0.204% |
| `J_area, J_persist.mean.d4` | 3 | 0.204% |
| `J_area, J_persist.mean.simd_schedule` | 3 | 0.204% |
| `J_area, J_persist.simd_schedule` | 3 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.512B` | 3 | 0.204% |
| `J_area, J_persist.simd_stream.d1` | 3 | 0.204% |
| `J_area, J_persist.simd_stream.d1.512B` | 3 | 0.204% |
| `J_area, J_persist.simd_stream.d4` | 3 | 0.204% |
| `J_area, J_persist.simd_stream.d4.1024B` | 3 | 0.204% |
| `J_peak, J_area, J_persist` | 3 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream` | 3 | 0.204% |
| `J_peak, J_area, J_persist.mean.lane_stream` | 3 | 0.204% |
| `J_peak, J_area, J_persist.mean.simd_stream` | 3 | 0.204% |
| `J_peak, J_area, J_persist.mean_cells` | 3 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream` | 3 | 0.204% |
| `Q_fine, J_area, J_persist.d4` | 3 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4` | 3 | 0.204% |
| `Q_fine, J_area, J_persist.mean.d4` | 3 | 0.204% |
| `Q_fine, J_area, J_persist.mean.simd_schedule` | 3 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule` | 3 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4` | 3 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist` | 3 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream` | 3 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.mean.lane_stream` | 3 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.mean.simd_stream` | 3 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.mean_cells` | 3 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream` | 3 | 0.204% |
| `J_peak, J_area, J_persist.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.4096B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.d1.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.d16.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.d4` | 4 | 0.204% |
| `J_peak, J_area, J_persist.d4.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.4096B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d1.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d16.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d4.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.mean.d4` | 4 | 0.204% |
| `J_peak, J_area, J_persist.mean.simd_schedule` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d16` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d4.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.4096B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d1.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d16.32768B` | 4 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d4.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.4096B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d1.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d16.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d4` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d4.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.4096B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.mean.d4` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.mean.simd_schedule` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.4096B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.32768B` | 4 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.32768B` | 4 | 0.204% |
| `J_area, J_persist.simd_schedule.d4` | 5 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d4` | 5 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d4` | 5 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4` | 5 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4` | 5 | 0.204% |
| `J_area, J_persist.mean.d16` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.d1.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.d1.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.d16.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.d16.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.d4.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.d4.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.mean.d16` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.8192B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.16384B` | 7 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.8192B` | 7 | 0.204% |
| `J_area, J_persist.d16` | 8 | 0.204% |
| `J_area, J_persist.lane_stream.d16` | 8 | 0.204% |
| `J_area, J_persist.simd_stream.d16` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.d1.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.d16` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.d4.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.4096B` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16` | 8 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.4096B` | 8 | 0.204% |
| `J_area, J_persist.256B` | 9 | 0.204% |
| `J_area, J_persist.64B` | 9 | 0.204% |
| `J_area, J_persist.d16.128B` | 9 | 0.204% |
| `J_area, J_persist.d16.256B` | 9 | 0.204% |
| `J_area, J_persist.d4.128B` | 9 | 0.204% |
| `J_area, J_persist.d4.256B` | 9 | 0.204% |
| `J_area, J_persist.d4.32B` | 9 | 0.204% |
| `J_area, J_persist.d4.64B` | 9 | 0.204% |
| `J_area, J_persist.lane_stream.256B` | 9 | 0.204% |
| `J_area, J_persist.lane_stream.64B` | 9 | 0.204% |
| `J_area, J_persist.lane_stream.d16.256B` | 9 | 0.204% |
| `J_area, J_persist.lane_stream.d4.128B` | 9 | 0.204% |
| `J_area, J_persist.lane_stream.d4.256B` | 9 | 0.204% |
| `J_area, J_persist.lane_stream.d4.64B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.128B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.256B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.32B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.64B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.128B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.d16.256B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.32B` | 9 | 0.204% |
| `J_area, J_persist.simd_schedule.d4.64B` | 9 | 0.204% |
| `J_area, J_persist.simd_stream.256B` | 9 | 0.204% |
| `J_area, J_persist.simd_stream.64B` | 9 | 0.204% |
| `J_area, J_persist.simd_stream.d16.256B` | 9 | 0.204% |
| `J_area, J_persist.simd_stream.d4.128B` | 9 | 0.204% |
| `J_area, J_persist.simd_stream.d4.256B` | 9 | 0.204% |
| `J_area, J_persist.simd_stream.d4.64B` | 9 | 0.204% |
| `Q_fine, J_area` | 9 | 0.204% |
| `J_peak, J_area, J_persist.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.d1.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.d1.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.d16.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.d16.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.d4.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.d4.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d1.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d1.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d16.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d16.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d4.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.lane_stream.d4.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.mean.d16` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d16.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d16.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d4` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d4.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_schedule.d4.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d1.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d1.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d16.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d16.8192B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d4.16384B` | 9 | 0.204% |
| `J_peak, J_area, J_persist.simd_stream.d4.8192B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.2048B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.512B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d1.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d1.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d1.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d16.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d16.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d16.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d16.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d16.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d4.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d4.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d4.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d4.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.d4.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.2048B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.512B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d1.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d16.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.lane_stream.d4.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.1024B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.1024B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16384B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32768B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.4096B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.512B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.8192B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.2048B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.512B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d1.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d16.64B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.128B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.16B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.256B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.32B` | 9 | 0.204% |
| `Q_fine, J_area, J_persist.simd_stream.d4.64B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d1.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d1.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d16.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d16.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d4.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.d4.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d16.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.mean.d16` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d16.8192B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.16384B` | 9 | 0.204% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.8192B` | 9 | 0.204% |
| `J_place, J_persist.d1.128B` | 1 | 0.260% |
| `J_place, J_persist.d1.256B` | 1 | 0.260% |
| `J_place, J_persist.lane_stream.d1.128B` | 1 | 0.260% |
| `J_place, J_persist.lane_stream.d1.256B` | 1 | 0.260% |
| `J_place, J_persist.simd_schedule.1024B` | 1 | 0.260% |
| `J_place, J_persist.simd_schedule.512B` | 1 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.128B` | 1 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.256B` | 1 | 0.260% |
| `J_place, J_persist.simd_stream.d1.128B` | 1 | 0.260% |
| `J_place, J_persist.simd_stream.d1.256B` | 1 | 0.260% |
| `J_place, J_persist.2048B` | 2 | 0.260% |
| `J_place, J_persist.d1.1024B` | 2 | 0.260% |
| `J_place, J_persist.d1.512B` | 2 | 0.260% |
| `J_place, J_persist.lane_stream.2048B` | 2 | 0.260% |
| `J_place, J_persist.lane_stream.d1.1024B` | 2 | 0.260% |
| `J_place, J_persist.lane_stream.d1.512B` | 2 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.1024B` | 2 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.512B` | 2 | 0.260% |
| `J_place, J_persist.simd_stream.2048B` | 2 | 0.260% |
| `J_place, J_persist.simd_stream.d1.1024B` | 2 | 0.260% |
| `J_place, J_persist.simd_stream.d1.512B` | 2 | 0.260% |
| `Q_fine, J_place` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.2048B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.4096B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d1.1024B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d1.128B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d1.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d1.256B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d1.512B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d1.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d16.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d16.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d16.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d4.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.d4.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.2048B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.4096B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.1024B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.128B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.256B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.512B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d16.128B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d16.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d16.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d16.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d4.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d4.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.1024B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.128B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.16384B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.2048B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.256B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.32768B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.4096B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.512B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.8192B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.1024B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.256B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.512B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.2048B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.4096B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.1024B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.128B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.256B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.512B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d16.128B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d16.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d16.32B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d16.64B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d4.16B` | 2 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d4.32B` | 2 | 0.260% |
| `J_place, J_persist.16384B` | 3 | 0.260% |
| `J_place, J_persist.32768B` | 3 | 0.260% |
| `J_place, J_persist.4096B` | 3 | 0.260% |
| `J_place, J_persist.d1.16384B` | 3 | 0.260% |
| `J_place, J_persist.d1.2048B` | 3 | 0.260% |
| `J_place, J_persist.d1.32768B` | 3 | 0.260% |
| `J_place, J_persist.d1.64B` | 3 | 0.260% |
| `J_place, J_persist.d16.32768B` | 3 | 0.260% |
| `J_place, J_persist.d4.16384B` | 3 | 0.260% |
| `J_place, J_persist.d4.32768B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.16384B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.32768B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.4096B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d1.16384B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d1.2048B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d1.32768B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d1.64B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d16.32768B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d4.16384B` | 3 | 0.260% |
| `J_place, J_persist.lane_stream.d4.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.16384B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.2048B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.d16.16384B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.d16.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.16384B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.2048B` | 3 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.16384B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.4096B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d1.16384B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d1.2048B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d1.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d1.64B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d16.32768B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d4.16384B` | 3 | 0.260% |
| `J_place, J_persist.simd_stream.d4.32768B` | 3 | 0.260% |
| `J_peak, J_place, J_persist.d4.32B` | 3 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.32B` | 3 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.32B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d1.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d1.2048B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d1.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d1.4096B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d16.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d4.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.d4.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.2048B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.4096B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d16.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d4.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d4.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.1024B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.2048B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.4096B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.2048B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.4096B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.2048B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.4096B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d16.32768B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d4.16384B` | 3 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d4.32768B` | 3 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d4.32B` | 3 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.32B` | 3 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.32B` | 3 | 0.260% |
| `J_peak, J_place` | 4 | 0.260% |
| `J_place, J_persist.d1.4096B` | 4 | 0.260% |
| `J_place, J_persist.lane_stream.d1.4096B` | 4 | 0.260% |
| `J_place, J_persist.simd_schedule.4096B` | 4 | 0.260% |
| `J_place, J_persist.simd_schedule.d4` | 4 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.4096B` | 4 | 0.260% |
| `J_place, J_persist.simd_stream.d1.4096B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.2048B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.4096B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d1.1024B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d1.128B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d1.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d1.256B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d1.512B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d1.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d16.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d16.32B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d16.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.d4.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.2048B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.4096B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.1024B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.128B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.256B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.512B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d16.128B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d16.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d16.32B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d16.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d4.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d4.32B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.1024B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.128B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.16384B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.2048B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.256B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.32768B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.32B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.4096B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.512B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d1.8192B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d16.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d16.32B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d16.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.1024B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.256B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.512B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.2048B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.4096B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.1024B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.128B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.256B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.512B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d16.128B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d16.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d16.32B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d16.64B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d4.16B` | 4 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d4.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place` | 4 | 0.260% |
| `Q_fine, J_place, J_persist.128B` | 4 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.128B` | 4 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4` | 4 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.2048B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.4096B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.1024B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.256B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.512B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d16.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d16.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d16.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d4.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.2048B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.4096B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.1024B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.256B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.512B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.1024B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.16384B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.2048B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.256B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.32768B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.4096B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.512B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.8192B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.1024B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.256B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.512B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.2048B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.4096B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.1024B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.256B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.512B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.128B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.32B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.64B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.16B` | 4 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.32B` | 4 | 0.260% |
| `J_place, J_persist.8192B` | 5 | 0.260% |
| `J_place, J_persist.d1.8192B` | 5 | 0.260% |
| `J_place, J_persist.d4.8192B` | 5 | 0.260% |
| `J_place, J_persist.lane_stream.8192B` | 5 | 0.260% |
| `J_place, J_persist.lane_stream.d1.8192B` | 5 | 0.260% |
| `J_place, J_persist.lane_stream.d4.8192B` | 5 | 0.260% |
| `J_place, J_persist.simd_schedule.8192B` | 5 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.8192B` | 5 | 0.260% |
| `J_place, J_persist.simd_stream.8192B` | 5 | 0.260% |
| `J_place, J_persist.simd_stream.d1.8192B` | 5 | 0.260% |
| `J_place, J_persist.simd_stream.d4.8192B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d1.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d1.2048B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d1.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d1.4096B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d16.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d4.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.d4.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.2048B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.4096B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d16.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d4.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d4.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.1024B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.2048B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.4096B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d16.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d16.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.2048B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.4096B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.2048B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.4096B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d16.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d4.16384B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d4.32768B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.32B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.d1.32B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.d1.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.d4.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.32B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.32B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d1.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.d4.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.32B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.32B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d1.8192B` | 5 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.d4.8192B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.2048B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.4096B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d16.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d4.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d4.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.2048B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.4096B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.1024B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.2048B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.4096B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.2048B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.4096B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.2048B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.4096B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.32768B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.16384B` | 5 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.32768B` | 5 | 0.260% |
| `J_peak, J_place, J_persist.128B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.32B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.d1.32B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.128B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.32B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.32B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.128B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.32B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.32B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.128B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.32B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.32B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.128B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.32B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.32B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.128B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.32B` | 6 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.32B` | 6 | 0.260% |
| `J_peak, J_place, J_persist.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.d1.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.d4.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d1.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.d4.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.simd_schedule.d4.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d1.8192B` | 7 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.d4.8192B` | 7 | 0.260% |
| `Q_fine, J_place, J_persist.512B` | 7 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.512B` | 7 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.512B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d1.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.d4.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.8192B` | 7 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.8192B` | 7 | 0.260% |
| `Q_fine, J_place, J_persist.1024B` | 8 | 0.260% |
| `Q_fine, J_place, J_persist.lane_stream.1024B` | 8 | 0.260% |
| `Q_fine, J_place, J_persist.simd_stream.1024B` | 8 | 0.260% |
| `J_place, J_persist.d4.32B` | 9 | 0.260% |
| `J_place, J_persist.simd_schedule.32B` | 9 | 0.260% |
| `J_place, J_persist.simd_schedule.d4.32B` | 9 | 0.260% |
| `J_peak, J_place, J_persist.512B` | 9 | 0.260% |
| `J_peak, J_place, J_persist.lane_stream.512B` | 9 | 0.260% |
| `J_peak, J_place, J_persist.simd_stream.512B` | 9 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.512B` | 9 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.512B` | 9 | 0.260% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.512B` | 9 | 0.260% |
| `J_persist.d1` | 2 | 0.412% |
| `J_persist.lane_stream.d1` | 2 | 0.412% |
| `J_persist.mean.d1` | 2 | 0.412% |
| `J_persist.simd_stream.d1` | 2 | 0.412% |
| `Q_fine, J_persist.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.d1.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.d16.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.d4.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.lane_stream.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.lane_stream.d1.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.lane_stream.d16.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.lane_stream.d4.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_schedule.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_schedule.d16.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_schedule.d4.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_stream.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_stream.d1.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_stream.d16.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.simd_stream.d4.8192B` | 8 | 0.501% |
| `Q_fine, J_persist.d1.512B` | 8 | 0.576% |
| `Q_fine, J_persist.d4.512B` | 8 | 0.576% |
| `Q_fine, J_persist.lane_stream.d1.512B` | 8 | 0.576% |
| `Q_fine, J_persist.lane_stream.d4.512B` | 8 | 0.576% |
| `Q_fine, J_persist.simd_schedule.d4.512B` | 8 | 0.576% |
| `Q_fine, J_persist.simd_stream.d1.512B` | 8 | 0.576% |
| `Q_fine, J_persist.simd_stream.d4.512B` | 8 | 0.576% |
| `Q_fine, J_persist.d1.256B` | 9 | 0.576% |
| `Q_fine, J_persist.lane_stream.d1.256B` | 9 | 0.576% |
| `Q_fine, J_persist.simd_schedule.d4.256B` | 9 | 0.576% |
| `Q_fine, J_persist.simd_stream.d1.256B` | 9 | 0.576% |
| `J_place, J_persist.simd_schedule.d16.2048B` | 9 | 0.722% |
| `J_place, J_persist.simd_schedule.d16.4096B` | 9 | 0.722% |
| `Q_fine, J_persist.d1` | 2 | 0.781% |
| `Q_fine, J_persist.lane_stream.d1` | 2 | 0.781% |
| `Q_fine, J_persist.mean.d1` | 2 | 0.781% |
| `Q_fine, J_persist.simd_stream.d1` | 2 | 0.781% |
| `J_peak, J_persist.d1` | 3 | 0.781% |
| `J_peak, J_persist.lane_stream.d1` | 3 | 0.781% |
| `J_peak, J_persist.mean.d1` | 3 | 0.781% |
| `J_peak, J_persist.simd_stream.d1` | 3 | 0.781% |
| `Q_fine, J_peak, J_persist.d1` | 3 | 0.781% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 3 | 0.781% |
| `Q_fine, J_peak, J_persist.mean.d1` | 3 | 0.781% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 3 | 0.781% |
| `J_place, J_persist` | 7 | 0.909% |
| `J_place, J_persist.lane_stream` | 7 | 0.909% |
| `J_place, J_persist.mean.lane_stream` | 7 | 0.909% |
| `J_place, J_persist.mean.simd_stream` | 7 | 0.909% |
| `J_place, J_persist.simd_stream` | 7 | 0.909% |
| `Q_fine, J_persist.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.d1.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.d16.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.d4.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.lane_stream.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.lane_stream.d1.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.lane_stream.d16.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.lane_stream.d4.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_schedule.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_schedule.d16.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_schedule.d4.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_stream.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_stream.d1.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_stream.d16.32768B` | 8 | 0.909% |
| `Q_fine, J_persist.simd_stream.d4.32768B` | 8 | 0.909% |
| `J_place, J_persist.d16.16384B` | 9 | 0.909% |
| `J_place, J_persist.lane_stream.d16.16384B` | 9 | 0.909% |
| `J_place, J_persist.simd_stream.d16.16384B` | 9 | 0.909% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_area` | 1 | 0.204% |
| `best_below_ten_samples` | `J_place, J_persist.d1` | 4 | 0.000% |
| `best_below_five_samples` | `J_place, J_persist.d1` | 4 | 0.000% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_place, J_persist.d1` | 4 | 0.000% |
| `J_place, J_persist.lane_stream.d1` | 4 | 0.000% |
| `J_place, J_persist.mean.d1` | 4 | 0.000% |
| `J_place, J_persist.simd_stream.d1` | 4 | 0.000% |
| `Q_fine, J_place, J_persist.d1` | 8 | 0.000% |
| `Q_fine, J_place, J_persist.lane_stream.d1` | 8 | 0.000% |
| `Q_fine, J_place, J_persist.mean.d1` | 8 | 0.000% |
| `Q_fine, J_place, J_persist.simd_stream.d1` | 8 | 0.000% |
| `J_peak, J_place, J_persist.d1` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d1` | 9 | 0.000% |

## GESUMMV N=512

Oracle median: 0.062293 ms over 146 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 10 | 13.913% | no |
| `locality_plus_persist` | 41 | 0.193% | no |
| `locality_plus_place` | 19 | 4.667% | no |
| `all_five` | 72 | 0.193% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiiiiiiiijjjjjjji` | 0.000% | 3413656.000000 |
| `current_selection` | `iijiiiiiiijjjjjjjj` | 13.913% | 3982104.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.32B` | 9 | 0.000% |
| `J_peak, J_persist.d1.32B` | 9 | 0.000% |
| `J_peak, J_persist.lane_stream.32B` | 9 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.32B` | 9 | 0.000% |
| `J_peak, J_persist.simd_stream.32B` | 9 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.32B` | 9 | 0.000% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_peak, J_persist.32B` | 9 | 0.000% |
| `best_below_ten_samples` | `J_peak, J_persist.32B` | 9 | 0.000% |
| `best_below_five_samples` | `J_peak, J_persist.16384B` | 3 | 1.777% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.32B` | 9 | 0.000% |
| `J_peak, J_persist.d1.32B` | 9 | 0.000% |
| `J_peak, J_persist.lane_stream.32B` | 9 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.32B` | 9 | 0.000% |
| `J_peak, J_persist.simd_stream.32B` | 9 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.32B` | 9 | 0.000% |
| `J_peak, J_persist.16384B` | 3 | 1.777% |
| `J_peak, J_persist.32768B` | 3 | 1.777% |
| `J_peak, J_persist.d1.16384B` | 3 | 1.777% |
| `J_peak, J_persist.d1.32768B` | 3 | 1.777% |

## GESUMMV N=1024

Oracle median: 0.127280 ms over 182 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 11 | 20.878% | no |
| `locality_plus_persist` | 45 | 0.933% | no |
| `locality_plus_place` | 27 | 20.878% | no |
| `all_five` | 99 | 0.933% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiiiiiiiijjjjjjjjii` | 0.000% | 6893336.000000 |
| `current_selection` | `iijiiiiiiiijjjjjjjjj` | 20.878% | 8266264.000000 |
| `near_oracle_dominator` | `jjiiiiiiijjjjjjjjiii` | 0.482% | 6464152.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| _None_ | — | — |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_peak, J_persist.32B` | 10 | 0.000% |
| `best_below_ten_samples` | `J_area, J_persist.32768B` | 9 | 6.537% |
| `best_below_five_samples` | `Q_fine, J_persist.32768B` | 4 | 15.189% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_area, J_persist.32768B` | 9 | 6.537% |
| `J_area, J_persist.d1.32768B` | 9 | 6.537% |
| `J_area, J_persist.d16.32768B` | 9 | 6.537% |
| `J_area, J_persist.d4.32768B` | 9 | 6.537% |
| `J_area, J_persist.lane_stream.32768B` | 9 | 6.537% |
| `J_area, J_persist.lane_stream.d1.32768B` | 9 | 6.537% |
| `J_area, J_persist.lane_stream.d16.32768B` | 9 | 6.537% |
| `J_area, J_persist.lane_stream.d4.32768B` | 9 | 6.537% |
| `J_area, J_persist.simd_schedule.32768B` | 9 | 6.537% |
| `J_area, J_persist.simd_schedule.d16.32768B` | 9 | 6.537% |

## MVT N=512

Oracle median: 0.069227 ms over 146 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 4 | 13.077% | no |
| `locality_plus_persist` | 9 | 0.539% | no |
| `locality_plus_place` | 14 | 0.539% | no |
| `all_five` | 32 | 0.539% | no |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `iijjjjjjjjjiiiiiii` | 0.000% | 2757173.000000 |
| `current_selection` | `iijjjiiiiiiijjjjjj` | 13.077% | 2612797.312500 |
| `near_oracle_dominator` | `jiijjjjjjjjiiiiiii` | 0.230% | 2741046.906250 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.simd_schedule.d1.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.2048B` | 6 | 0.000% |
| `J_peak, J_persist.simd_schedule.1024B` | 7 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d1.2048B` | 7 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.1024B` | 7 | 0.000% |
| `Q_fine, J_persist.simd_schedule.1024B` | 8 | 0.000% |
| `J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |
| `J_peak, J_persist.d1.256B` | 9 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.d1.256B` | 9 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |
| `J_peak, J_persist.32B` | 2 | 0.077% |
| `J_peak, J_persist.d1.32B` | 2 | 0.077% |
| `J_peak, J_persist.lane_stream.32B` | 2 | 0.077% |
| `J_peak, J_persist.lane_stream.d1.32B` | 2 | 0.077% |
| `J_peak, J_persist.simd_stream.32B` | 2 | 0.077% |
| `J_peak, J_persist.simd_stream.d1.32B` | 2 | 0.077% |
| `Q_fine, J_peak, J_persist.32B` | 2 | 0.077% |
| `Q_fine, J_peak, J_persist.d1.32B` | 2 | 0.077% |
| `Q_fine, J_peak, J_persist.lane_stream.32B` | 2 | 0.077% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.32B` | 2 | 0.077% |
| `Q_fine, J_peak, J_persist.simd_stream.32B` | 2 | 0.077% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.32B` | 2 | 0.077% |
| `J_peak, J_area, J_persist.32B` | 6 | 0.077% |
| `J_peak, J_area, J_persist.d1.32B` | 6 | 0.077% |
| `Q_fine, J_peak, J_area, J_persist.32B` | 6 | 0.077% |
| `Q_fine, J_peak, J_area, J_persist.d1.32B` | 6 | 0.077% |
| `J_peak, J_area, J_persist.lane_stream.32B` | 8 | 0.077% |
| `J_peak, J_area, J_persist.lane_stream.d1.32B` | 8 | 0.077% |
| `J_peak, J_area, J_persist.simd_stream.32B` | 8 | 0.077% |
| `J_peak, J_area, J_persist.simd_stream.d1.32B` | 8 | 0.077% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.32B` | 8 | 0.077% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.32B` | 8 | 0.077% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.32B` | 8 | 0.077% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.32B` | 8 | 0.077% |
| `J_peak, J_persist.d1.64B` | 9 | 0.077% |
| `Q_fine, J_peak, J_persist.d1.64B` | 9 | 0.077% |
| `J_peak, J_persist.simd_schedule.32B` | 8 | 0.211% |
| `J_peak, J_persist.simd_schedule.d1.32B` | 8 | 0.211% |
| `Q_fine, J_peak, J_persist.simd_schedule.32B` | 8 | 0.211% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.32B` | 8 | 0.211% |
| `J_peak, J_persist.16B` | 9 | 0.211% |
| `J_peak, J_persist.d1.16B` | 9 | 0.211% |
| `J_peak, J_persist.simd_schedule.16B` | 9 | 0.211% |
| `J_peak, J_persist.simd_schedule.d1.16B` | 9 | 0.211% |
| `Q_fine, J_persist.simd_schedule.32B` | 9 | 0.211% |
| `Q_fine, J_persist.simd_schedule.d1.32B` | 9 | 0.211% |
| `Q_fine, J_peak, J_persist.16B` | 9 | 0.211% |
| `Q_fine, J_peak, J_persist.d1.16B` | 9 | 0.211% |
| `Q_fine, J_peak, J_persist.simd_schedule.16B` | 9 | 0.211% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.16B` | 9 | 0.211% |
| `J_peak, J_place, J_persist.lane_stream.32B` | 6 | 0.230% |
| `J_peak, J_place, J_persist.lane_stream.d1.32B` | 6 | 0.230% |
| `J_peak, J_place, J_persist.simd_stream.32B` | 6 | 0.230% |
| `J_peak, J_place, J_persist.simd_stream.d1.32B` | 6 | 0.230% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.32B` | 6 | 0.230% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.32B` | 6 | 0.230% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.32B` | 6 | 0.230% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.32B` | 6 | 0.230% |
| `J_persist.d1` | 1 | 0.539% |
| `J_persist.mean.d1` | 1 | 0.539% |
| `J_peak, J_persist.d1` | 1 | 0.539% |
| `J_peak, J_persist.mean.d1` | 1 | 0.539% |
| `Q_fine, J_persist.d1` | 1 | 0.539% |
| `Q_fine, J_persist.mean.d1` | 1 | 0.539% |
| `Q_fine, J_peak, J_persist.d1` | 1 | 0.539% |
| `Q_fine, J_peak, J_persist.mean.d1` | 1 | 0.539% |
| `J_persist.lane_stream.d1` | 2 | 0.539% |
| `J_persist.simd_stream.d1` | 2 | 0.539% |
| `J_peak, J_persist.d1.2048B` | 2 | 0.539% |
| `J_peak, J_persist.d1.4096B` | 2 | 0.539% |
| `J_peak, J_persist.d4.2048B` | 2 | 0.539% |
| `J_peak, J_persist.lane_stream.d1` | 2 | 0.539% |
| `J_peak, J_persist.lane_stream.d1.2048B` | 2 | 0.539% |
| `J_peak, J_persist.lane_stream.d4.2048B` | 2 | 0.539% |
| `J_peak, J_persist.simd_schedule.2048B` | 2 | 0.539% |
| `J_peak, J_persist.simd_schedule.4096B` | 2 | 0.539% |
| `J_peak, J_persist.simd_schedule.d1.8192B` | 2 | 0.539% |
| `J_peak, J_persist.simd_schedule.d16.2048B` | 2 | 0.539% |
| `J_peak, J_persist.simd_schedule.d4.2048B` | 2 | 0.539% |
| `J_peak, J_persist.simd_stream.d1` | 2 | 0.539% |
| `J_peak, J_persist.simd_stream.d1.2048B` | 2 | 0.539% |
| `J_peak, J_persist.simd_stream.d4.2048B` | 2 | 0.539% |
| `Q_fine, J_persist.lane_stream.d1` | 2 | 0.539% |
| `Q_fine, J_persist.simd_stream.d1` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.d1.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.d1.4096B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.d4.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.4096B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.8192B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.2048B` | 2 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.2048B` | 2 | 0.539% |
| `J_peak, J_area, J_persist.d1.4096B` | 3 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.4096B` | 3 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.8192B` | 3 | 0.539% |
| `Q_fine, J_area, J_persist.d1.4096B` | 3 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.4096B` | 3 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.8192B` | 3 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d1.4096B` | 3 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.4096B` | 3 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.8192B` | 3 | 0.539% |
| `J_area, J_persist.d1` | 4 | 0.539% |
| `J_area, J_persist.mean.d1` | 4 | 0.539% |
| `J_peak, J_persist.d4.1024B` | 4 | 0.539% |
| `J_peak, J_persist.d4.4096B` | 4 | 0.539% |
| `J_peak, J_persist.lane_stream.d1.1024B` | 4 | 0.539% |
| `J_peak, J_persist.lane_stream.d1.4096B` | 4 | 0.539% |
| `J_peak, J_persist.lane_stream.d4.1024B` | 4 | 0.539% |
| `J_peak, J_persist.lane_stream.d4.4096B` | 4 | 0.539% |
| `J_peak, J_persist.simd_schedule.d16.4096B` | 4 | 0.539% |
| `J_peak, J_persist.simd_schedule.d4` | 4 | 0.539% |
| `J_peak, J_persist.simd_schedule.d4.1024B` | 4 | 0.539% |
| `J_peak, J_persist.simd_schedule.d4.4096B` | 4 | 0.539% |
| `J_peak, J_persist.simd_stream.d1.1024B` | 4 | 0.539% |
| `J_peak, J_persist.simd_stream.d1.4096B` | 4 | 0.539% |
| `J_peak, J_persist.simd_stream.d4.1024B` | 4 | 0.539% |
| `J_peak, J_persist.simd_stream.d4.4096B` | 4 | 0.539% |
| `Q_fine, J_persist.d1.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.d1.4096B` | 4 | 0.539% |
| `Q_fine, J_persist.d4.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.lane_stream.d1.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.lane_stream.d4.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_schedule.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_schedule.4096B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d1.8192B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d16.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d4` | 4 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d4.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_stream.d1.2048B` | 4 | 0.539% |
| `Q_fine, J_persist.simd_stream.d4.2048B` | 4 | 0.539% |
| `J_peak, J_area, J_persist.d1` | 4 | 0.539% |
| `J_peak, J_area, J_persist.mean.d1` | 4 | 0.539% |
| `Q_fine, J_area, J_persist.d1` | 4 | 0.539% |
| `Q_fine, J_area, J_persist.mean.d1` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.d4.1024B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.d4.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.1024B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.1024B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.1024B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.1024B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.1024B` | 4 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.4096B` | 4 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d1` | 4 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.mean.d1` | 4 | 0.539% |
| `J_peak, J_area, J_persist.d1.2048B` | 5 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.2048B` | 5 | 0.539% |
| `Q_fine, J_area, J_persist.d1.2048B` | 5 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.2048B` | 5 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d1.2048B` | 5 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.2048B` | 5 | 0.539% |
| `J_peak, J_persist.d4` | 6 | 0.539% |
| `J_peak, J_persist.lane_stream.d4` | 6 | 0.539% |
| `J_peak, J_persist.mean.d4` | 6 | 0.539% |
| `J_peak, J_persist.simd_stream.d4` | 6 | 0.539% |
| `J_peak, J_area, J_persist.16384B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.d1.16384B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.d4.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.d4.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.16384B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.16384B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d16.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d16.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.539% |
| `J_peak, J_place, J_persist.simd_schedule.d16.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.16384B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.d1.16384B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.16384B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16384B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.4096B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.16384B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d1.16384B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.16384B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.16384B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.2048B` | 6 | 0.539% |
| `J_peak, J_area, J_persist` | 7 | 0.539% |
| `J_peak, J_area, J_persist.8192B` | 7 | 0.539% |
| `J_peak, J_area, J_persist.d1.8192B` | 7 | 0.539% |
| `J_peak, J_area, J_persist.mean.simd_schedule` | 7 | 0.539% |
| `J_peak, J_area, J_persist.mean_cells` | 7 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule` | 7 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.8192B` | 7 | 0.539% |
| `J_peak, J_place, J_persist.d4.1024B` | 7 | 0.539% |
| `J_peak, J_place, J_persist.lane_stream.d4.1024B` | 7 | 0.539% |
| `J_peak, J_place, J_persist.simd_stream.d4.1024B` | 7 | 0.539% |
| `Q_fine, J_area, J_persist.8192B` | 7 | 0.539% |
| `Q_fine, J_area, J_persist.d1.8192B` | 7 | 0.539% |
| `Q_fine, J_area, J_persist.mean.simd_schedule` | 7 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule` | 7 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.8192B` | 7 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.8192B` | 7 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d1.8192B` | 7 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.mean.simd_schedule` | 7 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule` | 7 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.8192B` | 7 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.d4.1024B` | 7 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.1024B` | 7 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.1024B` | 7 | 0.539% |
| `J_area, J_persist.lane_stream.d1` | 8 | 0.539% |
| `J_area, J_persist.simd_stream.d1` | 8 | 0.539% |
| `Q_fine, J_persist.d4.1024B` | 8 | 0.539% |
| `Q_fine, J_persist.d4.4096B` | 8 | 0.539% |
| `Q_fine, J_persist.lane_stream.d1.1024B` | 8 | 0.539% |
| `Q_fine, J_persist.lane_stream.d1.4096B` | 8 | 0.539% |
| `Q_fine, J_persist.lane_stream.d4.1024B` | 8 | 0.539% |
| `Q_fine, J_persist.lane_stream.d4.4096B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d16.4096B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d4.1024B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_schedule.d4.4096B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_stream.d1.1024B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_stream.d1.4096B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_stream.d4.1024B` | 8 | 0.539% |
| `Q_fine, J_persist.simd_stream.d4.4096B` | 8 | 0.539% |
| `J_peak, J_area, J_persist.32768B` | 8 | 0.539% |
| `J_peak, J_area, J_persist.d1.32768B` | 8 | 0.539% |
| `J_peak, J_area, J_persist.lane_stream.d1` | 8 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.32768B` | 8 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d4` | 8 | 0.539% |
| `J_peak, J_area, J_persist.simd_stream.d1` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.32768B` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.d1.32768B` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.lane_stream.d1` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.32768B` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.simd_schedule.d4` | 8 | 0.539% |
| `Q_fine, J_area, J_persist.simd_stream.d1` | 8 | 0.539% |
| `Q_fine, J_peak, J_persist.d4` | 8 | 0.539% |
| `Q_fine, J_peak, J_persist.lane_stream.d4` | 8 | 0.539% |
| `Q_fine, J_peak, J_persist.mean.d4` | 8 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_stream.d4` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.32768B` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.d1.32768B` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32768B` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1` | 8 | 0.539% |
| `J_peak, J_place, J_persist.d4.2048B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.lane_stream.d4.2048B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_schedule.d4.1024B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_stream.d4.2048B` | 9 | 0.539% |
| `Q_fine, J_area, J_persist.mean_cells` | 9 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist` | 9 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.mean_cells` | 9 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.d4.2048B` | 9 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.2048B` | 9 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.1024B` | 9 | 0.539% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.2048B` | 9 | 0.539% |
| `J_peak, J_persist.d1.1024B` | 2 | 0.731% |
| `Q_fine, J_peak, J_persist.d1.1024B` | 2 | 0.731% |
| `Q_fine, J_persist.d1.1024B` | 4 | 0.731% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_persist.d1` | 1 | 0.539% |
| `best_below_ten_samples` | `J_peak, J_persist.simd_schedule.d1.2048B` | 6 | 0.000% |
| `best_below_five_samples` | `J_peak, J_persist.32B` | 2 | 0.077% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.simd_schedule.d1.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.2048B` | 6 | 0.000% |
| `J_peak, J_persist.simd_schedule.1024B` | 7 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d1.2048B` | 7 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.1024B` | 7 | 0.000% |
| `Q_fine, J_persist.simd_schedule.1024B` | 8 | 0.000% |
| `J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |
| `J_peak, J_persist.d1.256B` | 9 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.2048B` | 9 | 0.000% |

## MVT N=1024

Oracle median: 0.136293 ms over 182 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 4 | 12.522% | no |
| `locality_plus_persist` | 9 | 0.000% | yes |
| `locality_plus_place` | 23 | 0.000% | yes |
| `all_five` | 40 | 0.000% | yes |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `jjiiiijjjjjjjjiiiiii` | 0.000% | 5289351.250000 |
| `current_selection` | `iijjjiiiiiiiijjjjjjj` | 12.522% | 5292415.312500 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_persist.d1` | 1 | 0.000% |
| `J_persist.mean.d1` | 1 | 0.000% |
| `J_peak, J_persist.d1` | 1 | 0.000% |
| `J_peak, J_persist.mean.d1` | 1 | 0.000% |
| `Q_fine, J_persist.d1` | 1 | 0.000% |
| `Q_fine, J_persist.mean.d1` | 1 | 0.000% |
| `Q_fine, J_peak, J_persist.d1` | 1 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d1` | 1 | 0.000% |
| `J_persist.lane_stream.d1` | 2 | 0.000% |
| `J_persist.simd_stream.d1` | 2 | 0.000% |
| `J_peak, J_persist.d1.2048B` | 2 | 0.000% |
| `J_peak, J_persist.d1.4096B` | 2 | 0.000% |
| `J_peak, J_persist.d4.2048B` | 2 | 0.000% |
| `J_peak, J_persist.lane_stream.d1` | 2 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.2048B` | 2 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.2048B` | 2 | 0.000% |
| `J_peak, J_persist.simd_schedule.2048B` | 2 | 0.000% |
| `J_peak, J_persist.simd_schedule.4096B` | 2 | 0.000% |
| `J_peak, J_persist.simd_schedule.d1.8192B` | 2 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.2048B` | 2 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.2048B` | 2 | 0.000% |
| `J_peak, J_persist.simd_stream.d1` | 2 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.2048B` | 2 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.2048B` | 2 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1` | 2 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.d1.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.d1.4096B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.d4.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.4096B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.8192B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.2048B` | 2 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.2048B` | 2 | 0.000% |
| `J_peak, J_area, J_persist.d1.4096B` | 3 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.4096B` | 3 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d1.8192B` | 3 | 0.000% |
| `Q_fine, J_area, J_persist.d1.4096B` | 3 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.4096B` | 3 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.8192B` | 3 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d1.4096B` | 3 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.4096B` | 3 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.8192B` | 3 | 0.000% |
| `J_area, J_persist.d1` | 4 | 0.000% |
| `J_area, J_persist.mean.d1` | 4 | 0.000% |
| `J_peak, J_persist.d4.1024B` | 4 | 0.000% |
| `J_peak, J_persist.d4.4096B` | 4 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.1024B` | 4 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.4096B` | 4 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.1024B` | 4 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.4096B` | 4 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.4096B` | 4 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4` | 4 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.1024B` | 4 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.4096B` | 4 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.1024B` | 4 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.4096B` | 4 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.1024B` | 4 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.4096B` | 4 | 0.000% |
| `Q_fine, J_persist.d1.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.d1.4096B` | 4 | 0.000% |
| `Q_fine, J_persist.d4.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.4096B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d1.8192B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.2048B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.2048B` | 4 | 0.000% |
| `J_peak, J_area, J_persist.d1` | 4 | 0.000% |
| `J_peak, J_area, J_persist.mean.d1` | 4 | 0.000% |
| `Q_fine, J_area, J_persist.d1` | 4 | 0.000% |
| `Q_fine, J_area, J_persist.mean.d1` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.d4.1024B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.d4.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.1024B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.1024B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.1024B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.1024B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.1024B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.4096B` | 4 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d1` | 4 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.mean.d1` | 4 | 0.000% |
| `J_area, J_persist.mean.simd_schedule` | 5 | 0.000% |
| `J_area, J_persist.simd_schedule` | 5 | 0.000% |
| `J_peak, J_area, J_persist.d1.2048B` | 5 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.2048B` | 5 | 0.000% |
| `Q_fine, J_area, J_persist.d1.2048B` | 5 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.2048B` | 5 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d1.2048B` | 5 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.2048B` | 5 | 0.000% |
| `J_peak, J_persist.d4` | 6 | 0.000% |
| `J_peak, J_persist.lane_stream.d4` | 6 | 0.000% |
| `J_peak, J_persist.mean.d4` | 6 | 0.000% |
| `J_peak, J_persist.simd_stream.d4` | 6 | 0.000% |
| `J_peak, J_area, J_persist.16384B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.d1.16384B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.d4.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.d4.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.16384B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d1.16384B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d16.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d16.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.16384B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.d1.16384B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.16384B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16384B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.16384B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d1.16384B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.16384B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.16384B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d16.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.4096B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.2048B` | 6 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d4.4096B` | 6 | 0.000% |
| `J_peak, J_area, J_persist` | 7 | 0.000% |
| `J_peak, J_area, J_persist.8192B` | 7 | 0.000% |
| `J_peak, J_area, J_persist.d1.8192B` | 7 | 0.000% |
| `J_peak, J_area, J_persist.mean.simd_schedule` | 7 | 0.000% |
| `J_peak, J_area, J_persist.mean_cells` | 7 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule` | 7 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.8192B` | 7 | 0.000% |
| `Q_fine, J_area, J_persist.8192B` | 7 | 0.000% |
| `Q_fine, J_area, J_persist.d1.8192B` | 7 | 0.000% |
| `Q_fine, J_area, J_persist.mean.simd_schedule` | 7 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule` | 7 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.8192B` | 7 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.8192B` | 7 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d1.8192B` | 7 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.mean.simd_schedule` | 7 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule` | 7 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.8192B` | 7 | 0.000% |
| `J_area, J_persist.lane_stream.d1` | 8 | 0.000% |
| `J_area, J_persist.simd_stream.d1` | 8 | 0.000% |
| `Q_fine, J_persist.d4.1024B` | 8 | 0.000% |
| `Q_fine, J_persist.d4.4096B` | 8 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.1024B` | 8 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.4096B` | 8 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.1024B` | 8 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.4096B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16.4096B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.1024B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.4096B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.1024B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.4096B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.1024B` | 8 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.4096B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.d1.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d1` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d4` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d1` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.32768B` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.d1.32768B` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.lane_stream.d1` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.32768B` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.simd_schedule.d4` | 8 | 0.000% |
| `Q_fine, J_area, J_persist.simd_stream.d1` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.32768B` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.d1.32768B` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32768B` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.32768B` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.32B` | 9 | 0.000% |
| `J_peak, J_persist.simd_schedule.d1.32B` | 9 | 0.000% |
| `Q_fine, J_area, J_persist.mean_cells` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.32B` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.32B` | 9 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist` | 9 | 0.000% |
| `Q_fine, J_peak, J_area, J_persist.mean_cells` | 9 | 0.000% |
| `J_peak, J_persist.32B` | 2 | 0.675% |
| `J_peak, J_persist.d1.32B` | 2 | 0.675% |
| `J_peak, J_persist.lane_stream.32B` | 2 | 0.675% |
| `J_peak, J_persist.lane_stream.d1.32B` | 2 | 0.675% |
| `J_peak, J_persist.simd_stream.32B` | 2 | 0.675% |
| `J_peak, J_persist.simd_stream.d1.32B` | 2 | 0.675% |
| `Q_fine, J_peak, J_persist.32B` | 2 | 0.675% |
| `Q_fine, J_peak, J_persist.d1.32B` | 2 | 0.675% |
| `Q_fine, J_peak, J_persist.lane_stream.32B` | 2 | 0.675% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.32B` | 2 | 0.675% |
| `Q_fine, J_peak, J_persist.simd_stream.32B` | 2 | 0.675% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.32B` | 2 | 0.675% |
| `J_peak, J_area, J_persist.32B` | 6 | 0.675% |
| `J_peak, J_area, J_persist.d1.32B` | 6 | 0.675% |
| `Q_fine, J_peak, J_area, J_persist.32B` | 6 | 0.675% |
| `Q_fine, J_peak, J_area, J_persist.d1.32B` | 6 | 0.675% |
| `J_peak, J_persist.simd_schedule.d1.2048B` | 7 | 0.675% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.2048B` | 7 | 0.675% |
| `J_peak, J_persist.simd_schedule.1024B` | 8 | 0.675% |
| `Q_fine, J_persist.simd_schedule.d1.2048B` | 8 | 0.675% |
| `J_peak, J_area, J_persist.lane_stream.32B` | 8 | 0.675% |
| `J_peak, J_area, J_persist.lane_stream.d1.32B` | 8 | 0.675% |
| `J_peak, J_area, J_persist.simd_stream.32B` | 8 | 0.675% |
| `J_peak, J_area, J_persist.simd_stream.d1.32B` | 8 | 0.675% |
| `Q_fine, J_peak, J_persist.simd_schedule.1024B` | 8 | 0.675% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.32B` | 8 | 0.675% |
| `Q_fine, J_peak, J_area, J_persist.lane_stream.d1.32B` | 8 | 0.675% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.32B` | 8 | 0.675% |
| `Q_fine, J_peak, J_area, J_persist.simd_stream.d1.32B` | 8 | 0.675% |
| `Q_fine, J_persist.simd_schedule.1024B` | 9 | 0.675% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_persist.d1` | 1 | 0.000% |
| `best_below_ten_samples` | `J_persist.d1` | 1 | 0.000% |
| `best_below_five_samples` | `J_persist.d1` | 1 | 0.000% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_persist.d1` | 1 | 0.000% |
| `J_persist.mean.d1` | 1 | 0.000% |
| `J_peak, J_persist.d1` | 1 | 0.000% |
| `J_peak, J_persist.mean.d1` | 1 | 0.000% |
| `Q_fine, J_persist.d1` | 1 | 0.000% |
| `Q_fine, J_persist.mean.d1` | 1 | 0.000% |
| `Q_fine, J_peak, J_persist.d1` | 1 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d1` | 1 | 0.000% |
| `J_persist.lane_stream.d1` | 2 | 0.000% |
| `J_persist.simd_stream.d1` | 2 | 0.000% |

## SYRK N=512

Oracle median: 0.259600 ms over 146 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 10 | 0.663% | no |
| `locality_plus_persist` | 36 | 0.000% | yes |
| `locality_plus_place` | 22 | 0.596% | no |
| `all_five` | 77 | 0.000% | yes |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `iijjjjjjjjjiiiiiii` | 0.000% | 15720704.000000 |
| `current_selection` | `iijiiiiiiijjjjjjjj` | 1.197% | 31745280.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.16384B` | 3 | 0.000% |
| `J_peak, J_persist.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d1.16384B` | 3 | 0.000% |
| `J_peak, J_persist.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d16.16384B` | 3 | 0.000% |
| `J_peak, J_persist.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d4.16384B` | 3 | 0.000% |
| `J_peak, J_persist.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.16384B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.16384B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d16.16384B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.16384B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.16384B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.16384B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.16384B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.16384B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d16.16384B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.16384B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.32768B` | 3 | 0.000% |
| `Q_fine, J_persist.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.d1.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.d16.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.d4.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d16.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d16.16384B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.16384B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.d1.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.d4.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.32768B` | 4 | 0.000% |
| `J_peak, J_persist` | 5 | 0.000% |
| `J_peak, J_persist.d1` | 5 | 0.000% |
| `J_peak, J_persist.lane_stream` | 5 | 0.000% |
| `J_peak, J_persist.lane_stream.d1` | 5 | 0.000% |
| `J_peak, J_persist.lane_stream.d4` | 5 | 0.000% |
| `J_peak, J_persist.mean.lane_stream` | 5 | 0.000% |
| `J_peak, J_persist.mean.simd_stream` | 5 | 0.000% |
| `J_peak, J_persist.mean_cells` | 5 | 0.000% |
| `J_peak, J_persist.simd_stream` | 5 | 0.000% |
| `J_peak, J_persist.simd_stream.d1` | 5 | 0.000% |
| `J_peak, J_persist.simd_stream.d4` | 5 | 0.000% |
| `Q_fine, J_persist.d4` | 5 | 0.000% |
| `Q_fine, J_persist.mean.d4` | 5 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.d1.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.d16.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.d4.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d16.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d16.16384B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.16384B` | 5 | 0.000% |
| `J_peak, J_persist.d4` | 6 | 0.000% |
| `J_peak, J_persist.mean.d4` | 6 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4` | 6 | 0.000% |
| `Q_fine, J_persist.d16` | 7 | 0.000% |
| `Q_fine, J_persist.lane_stream.d16` | 7 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4` | 7 | 0.000% |
| `Q_fine, J_persist.mean.d16` | 7 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16` | 7 | 0.000% |
| `Q_fine, J_persist.simd_stream.d16` | 7 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4` | 7 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d16.32768B` | 7 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d16.32768B` | 7 | 0.000% |
| `J_peak, J_persist.8192B` | 8 | 0.000% |
| `J_peak, J_persist.d1.8192B` | 8 | 0.000% |
| `J_peak, J_persist.d16` | 8 | 0.000% |
| `J_peak, J_persist.d16.8192B` | 8 | 0.000% |
| `J_peak, J_persist.d4.8192B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.8192B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.8192B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d16` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d16.8192B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.8192B` | 8 | 0.000% |
| `J_peak, J_persist.mean.d16` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.8192B` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.8192B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.8192B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.8192B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d16` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d16.8192B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.8192B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.d1.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.d16.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.d4.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d1.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d16.16384B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d4.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d4.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d1.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d16.16384B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d4.32768B` | 8 | 0.000% |
| `J_peak, J_place, J_persist` | 8 | 0.000% |
| `J_peak, J_place, J_persist.d1` | 8 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream` | 8 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d1` | 8 | 0.000% |
| `J_peak, J_place, J_persist.mean.lane_stream` | 8 | 0.000% |
| `J_peak, J_place, J_persist.mean.simd_stream` | 8 | 0.000% |
| `J_peak, J_place, J_persist.mean_cells` | 8 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream` | 8 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d1` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.d1` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.d16` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d16` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d16` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.lane_stream` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.simd_stream` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.mean_cells` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 8 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d16` | 8 | 0.000% |
| `Q_fine, J_persist.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.d1.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.d16.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.d4.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.d16.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.d16.8192B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.8192B` | 9 | 0.000% |
| `J_peak, J_area, J_persist.d16.16384B` | 9 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d16.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d1.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d1.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d16` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d16.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d16.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d4` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d4.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.d4.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d1.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d1.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d16` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d16.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d16.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d4` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d4.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.lane_stream.d4.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.mean.d16` | 9 | 0.000% |
| `J_peak, J_place, J_persist.mean.d4` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_schedule.d16` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_schedule.d16.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_schedule.d16.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_schedule.d4` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_schedule.d4.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_schedule.d4.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d1.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d1.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d16` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d16.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d16.32768B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d4` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d4.16384B` | 9 | 0.000% |
| `J_peak, J_place, J_persist.simd_stream.d4.32768B` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4` | 9 | 0.000% |
| `Q_fine, J_persist.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.d1.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.d16.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.d4.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.lane_stream.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.lane_stream.d1.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.lane_stream.d16.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.lane_stream.d4.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.simd_schedule.d16.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.simd_schedule.d4.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.simd_stream.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.simd_stream.d1.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.simd_stream.d16.32768B` | 1 | 0.195% |
| `Q_fine, J_persist.simd_stream.d4.32768B` | 1 | 0.195% |
| `Q_fine, J_persist` | 4 | 0.195% |
| `Q_fine, J_persist.d1` | 4 | 0.195% |
| `Q_fine, J_persist.lane_stream` | 4 | 0.195% |
| `Q_fine, J_persist.lane_stream.d1` | 4 | 0.195% |
| `Q_fine, J_persist.mean.lane_stream` | 4 | 0.195% |
| `Q_fine, J_persist.mean.simd_stream` | 4 | 0.195% |
| `Q_fine, J_persist.simd_stream` | 4 | 0.195% |
| `Q_fine, J_persist.simd_stream.d1` | 4 | 0.195% |
| `Q_fine, J_place, J_persist` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.d1` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.lane_stream` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.lane_stream.d1` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.mean.lane_stream` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.mean.simd_stream` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.simd_stream` | 9 | 0.195% |
| `Q_fine, J_place, J_persist.simd_stream.d1` | 9 | 0.195% |
| `Q_fine, J_persist.mean.d1` | 4 | 0.508% |
| `Q_fine, J_persist.mean_cells` | 4 | 0.508% |
| `Q_fine, J_peak, J_persist.mean.d1` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.d1.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.lane_stream.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.lane_stream.d1.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.simd_schedule.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.simd_stream.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.simd_stream.d1.32B` | 7 | 0.508% |
| `Q_fine, J_place, J_persist.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.d1.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.lane_stream.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.lane_stream.d1.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.mean_cells` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.simd_schedule.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.simd_stream.16B` | 8 | 0.508% |
| `Q_fine, J_place, J_persist.simd_stream.d1.16B` | 8 | 0.508% |
| `J_peak, J_persist.simd_schedule.d1` | 3 | 0.539% |
| `J_peak, J_persist.simd_schedule.d1.1024B` | 4 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1` | 4 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.1024B` | 4 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.2048B` | 5 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.4096B` | 6 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.8192B` | 7 | 0.539% |
| `J_peak, J_area, J_persist.simd_schedule.d1.8192B` | 7 | 0.539% |
| `J_peak, J_area, J_persist.mean.d1` | 8 | 0.539% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1` | 8 | 0.539% |
| `J_peak, J_place, J_persist.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.d1.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.d4.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.lane_stream.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.lane_stream.d1.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_schedule.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_schedule.d1.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_schedule.d4.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_stream.64B` | 9 | 0.539% |
| `J_peak, J_place, J_persist.simd_stream.d1.64B` | 9 | 0.539% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1` | 9 | 0.539% |
| `J_peak, J_persist.simd_schedule.16384B` | 3 | 0.549% |
| `J_peak, J_persist.simd_schedule.d1.16384B` | 3 | 0.549% |
| `Q_fine, J_persist.mean.simd_schedule` | 4 | 0.549% |
| `Q_fine, J_persist.simd_schedule` | 4 | 0.549% |
| `J_peak, J_persist.mean.simd_schedule` | 5 | 0.549% |
| `J_peak, J_persist.simd_schedule` | 5 | 0.549% |
| `J_peak, J_area, J_persist.simd_schedule.16384B` | 7 | 0.549% |
| `J_peak, J_area, J_persist.simd_schedule.d1.16384B` | 7 | 0.549% |
| `Q_fine, J_peak, J_persist.mean.simd_schedule` | 7 | 0.549% |
| `Q_fine, J_peak, J_persist.simd_schedule` | 7 | 0.549% |
| `Q_fine, J_peak, J_persist.simd_schedule.16384B` | 9 | 0.549% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.16384B` | 9 | 0.549% |
| `J_peak, J_place, J_persist.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.d1.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.d16.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.d4.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.lane_stream.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.lane_stream.d1.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.lane_stream.d16.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.lane_stream.d4.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.simd_schedule.d16.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.simd_schedule.d4.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.simd_stream.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.simd_stream.d1.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.simd_stream.d16.8192B` | 7 | 0.550% |
| `J_peak, J_place, J_persist.simd_stream.d4.8192B` | 7 | 0.550% |
| `Q_fine, J_place, J_persist.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.d1.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.d16.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.d4.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.lane_stream.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.lane_stream.d1.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.lane_stream.d16.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.lane_stream.d4.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.simd_stream.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.simd_stream.d1.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.simd_stream.d16.8192B` | 8 | 0.550% |
| `Q_fine, J_place, J_persist.simd_stream.d4.8192B` | 8 | 0.550% |
| `J_area, J_persist.32768B` | 3 | 0.580% |
| `J_area, J_persist.d1.32768B` | 3 | 0.580% |
| `J_area, J_persist.d16.32768B` | 3 | 0.580% |
| `J_area, J_persist.d4.32768B` | 3 | 0.580% |
| `J_area, J_persist.lane_stream.32768B` | 3 | 0.580% |
| `J_area, J_persist.lane_stream.d1.32768B` | 3 | 0.580% |
| `J_area, J_persist.lane_stream.d16.32768B` | 3 | 0.580% |
| `J_area, J_persist.lane_stream.d4.32768B` | 3 | 0.580% |
| `J_area, J_persist.simd_schedule.d16.32768B` | 3 | 0.580% |
| `J_area, J_persist.simd_schedule.d4.32768B` | 3 | 0.580% |
| `J_area, J_persist.simd_stream.32768B` | 3 | 0.580% |
| `J_area, J_persist.simd_stream.d1.32768B` | 3 | 0.580% |
| `J_area, J_persist.simd_stream.d16.32768B` | 3 | 0.580% |
| `J_area, J_persist.simd_stream.d4.32768B` | 3 | 0.580% |
| `J_area, J_persist.16384B` | 4 | 0.580% |
| `J_area, J_persist.d1.16384B` | 4 | 0.580% |
| `J_area, J_persist.d16.16384B` | 4 | 0.580% |
| `J_area, J_persist.d4.16384B` | 4 | 0.580% |
| `J_area, J_persist.lane_stream.16384B` | 4 | 0.580% |
| `J_area, J_persist.lane_stream.d1.16384B` | 4 | 0.580% |
| `J_area, J_persist.lane_stream.d16.16384B` | 4 | 0.580% |
| `J_area, J_persist.lane_stream.d4.16384B` | 4 | 0.580% |
| `J_area, J_persist.simd_schedule.d16.16384B` | 4 | 0.580% |
| `J_area, J_persist.simd_schedule.d4.16384B` | 4 | 0.580% |
| `J_area, J_persist.simd_stream.16384B` | 4 | 0.580% |
| `J_area, J_persist.simd_stream.d1.16384B` | 4 | 0.580% |
| `J_area, J_persist.simd_stream.d16.16384B` | 4 | 0.580% |
| `J_area, J_persist.simd_stream.d4.16384B` | 4 | 0.580% |
| `J_area, J_persist.8192B` | 7 | 0.580% |
| `J_area, J_persist.d1.8192B` | 7 | 0.580% |
| `J_area, J_persist.d16.8192B` | 7 | 0.580% |
| `J_area, J_persist.d4.8192B` | 7 | 0.580% |
| `J_area, J_persist.lane_stream.8192B` | 7 | 0.580% |
| `J_area, J_persist.lane_stream.d1.8192B` | 7 | 0.580% |
| `J_area, J_persist.lane_stream.d16.8192B` | 7 | 0.580% |
| `J_area, J_persist.lane_stream.d4.8192B` | 7 | 0.580% |
| `J_area, J_persist.simd_schedule.d16.8192B` | 7 | 0.580% |
| `J_area, J_persist.simd_schedule.d4.8192B` | 7 | 0.580% |
| `J_area, J_persist.simd_stream.8192B` | 7 | 0.580% |
| `J_area, J_persist.simd_stream.d1.8192B` | 7 | 0.580% |
| `J_area, J_persist.simd_stream.d16.8192B` | 7 | 0.580% |
| `J_area, J_persist.simd_stream.d4.8192B` | 7 | 0.580% |
| `J_area, J_persist.4096B` | 9 | 0.580% |
| `J_area, J_persist.d1.4096B` | 9 | 0.580% |
| `J_area, J_persist.d16.4096B` | 9 | 0.580% |
| `J_area, J_persist.d4.4096B` | 9 | 0.580% |
| `J_area, J_persist.lane_stream.4096B` | 9 | 0.580% |
| `J_area, J_persist.lane_stream.d1.4096B` | 9 | 0.580% |
| `J_area, J_persist.lane_stream.d16.4096B` | 9 | 0.580% |
| `J_area, J_persist.lane_stream.d4.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_schedule.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_schedule.d16.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_schedule.d4.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_stream.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_stream.d1.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_stream.d16.4096B` | 9 | 0.580% |
| `J_area, J_persist.simd_stream.d4.4096B` | 9 | 0.580% |
| `J_peak, J_persist.32B` | 9 | 0.580% |
| `J_peak, J_persist.d1.32B` | 9 | 0.580% |
| `J_peak, J_persist.lane_stream.32B` | 9 | 0.580% |
| `J_peak, J_persist.lane_stream.d1.32B` | 9 | 0.580% |
| `J_peak, J_persist.simd_schedule.32B` | 9 | 0.580% |
| `J_peak, J_persist.simd_schedule.d1.32B` | 9 | 0.580% |
| `J_peak, J_persist.simd_stream.32B` | 9 | 0.580% |
| `J_peak, J_persist.simd_stream.d1.32B` | 9 | 0.580% |
| `J_peak, J_persist.simd_schedule.8192B` | 3 | 0.596% |
| `J_peak, J_persist.simd_schedule.d1.8192B` | 3 | 0.596% |
| `J_peak, J_persist.mean.d1` | 5 | 0.596% |
| `J_peak, J_area, J_persist.simd_schedule.32768B` | 6 | 0.596% |
| `J_peak, J_area, J_persist.simd_schedule.d1.32768B` | 6 | 0.596% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.32768B` | 9 | 0.596% |
| `Q_fine, J_peak, J_area, J_persist.simd_schedule.d1.32768B` | 9 | 0.596% |
| `J_area, J_persist.mean.simd_schedule` | 9 | 0.637% |
| `J_area, J_persist.simd_schedule` | 9 | 0.637% |
| `J_peak, J_persist.simd_schedule.d1.4096B` | 3 | 0.662% |
| `Q_fine, J_area, J_persist.simd_schedule.32768B` | 4 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32768B` | 4 | 0.663% |
| `Q_fine, J_persist.simd_schedule.32768B` | 5 | 0.663% |
| `Q_fine, J_persist.simd_schedule.d1` | 5 | 0.663% |
| `Q_fine, J_persist.simd_schedule.d1.32768B` | 5 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.16384B` | 5 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1` | 5 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16384B` | 5 | 0.663% |
| `Q_fine, J_persist.simd_schedule.16384B` | 6 | 0.663% |
| `Q_fine, J_persist.simd_schedule.d1.16384B` | 6 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.8192B` | 6 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.8192B` | 6 | 0.663% |
| `Q_fine, J_persist.simd_schedule.8192B` | 7 | 0.663% |
| `Q_fine, J_persist.simd_schedule.d1.8192B` | 7 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.4096B` | 7 | 0.663% |
| `Q_fine, J_area` | 8 | 0.663% |
| `Q_fine, J_persist.simd_schedule.d1.4096B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.d1.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.d16.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.d16.32B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.d16.64B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.d4.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d1.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d16.128B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d16.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d16.32B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d16.64B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d4.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d4.32B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.1024B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.2048B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.512B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.32B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d16.64B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d1.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d16.128B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d16.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d16.32B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d16.64B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d4.16B` | 8 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d4.32B` | 8 | 0.663% |
| `Q_fine, J_peak, J_persist.simd_schedule.32768B` | 8 | 0.663% |
| `Q_fine, J_peak, J_persist.simd_schedule.d1.32768B` | 8 | 0.663% |
| `Q_fine, J_persist.simd_schedule.d1.2048B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.d1.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.d4.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.lane_stream.d1.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d1.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.simd_schedule.d4.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.32B` | 9 | 0.663% |
| `Q_fine, J_area, J_persist.simd_stream.d1.32B` | 9 | 0.663% |
| `J_peak, J_persist.simd_schedule.32768B` | 3 | 0.734% |
| `J_peak, J_persist.simd_schedule.d1.32768B` | 3 | 0.734% |
| `Q_fine, J_place, J_persist.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.d1.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.d16.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.d4.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.4096B` | 4 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.4096B` | 4 | 0.734% |
| `J_place, J_persist.simd_schedule.d1.512B` | 5 | 0.734% |
| `Q_fine, J_place` | 5 | 0.734% |
| `J_peak, J_place, J_persist.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.d1.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.d16.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.d4.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.4096B` | 5 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.4096B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d1.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d1.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d1.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d1.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d1.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d1.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.32B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d16.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.d4.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d1.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.32B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d16.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.32B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.lane_stream.d4.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.32B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d16.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d1.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.32B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d16.64B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.1024B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.128B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.16B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.2048B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.256B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.32B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.512B` | 5 | 0.734% |
| `Q_fine, J_place, J_persist.simd_stream.d4.64B` | 5 | 0.734% |
| `J_peak, J_place` | 6 | 0.734% |
| `J_peak, J_place, J_persist.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d1.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d1.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d1.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d1.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d1.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.32B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d16.64B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d4.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d4.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d4.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d4.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d4.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.d4.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.32B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d16.64B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.32B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d1.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d1.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.32B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d16.64B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.32B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.512B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d16.64B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.1024B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.128B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.16B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.2048B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.256B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.32B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.512B` | 6 | 0.734% |
| `Q_fine, J_place, J_persist.d4.32B` | 6 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d4.32B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d1.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.4096B` | 6 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.4096B` | 6 | 0.734% |
| `J_peak, J_place, J_persist.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.d1.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d4.64B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d1.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.32B` | 7 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d4.64B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d1.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d1.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d1.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d1.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d1.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.32B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d16.64B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.d4.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d1.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.32B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d16.64B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.32B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d1.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.32B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d16.64B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_schedule.d4.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d1.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.32B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.512B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d16.64B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.1024B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.128B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.16B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.2048B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.256B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.32B` | 7 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.512B` | 7 | 0.734% |
| `J_place, J_persist.simd_schedule.d1.1024B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.d1.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.d4.32B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.lane_stream.d1.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d1.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d4.32B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.16B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.simd_stream.d1.16B` | 8 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.lane_stream.d4.64B` | 8 | 0.734% |
| `Q_fine, J_peak, J_place, J_persist.simd_stream.d4.64B` | 8 | 0.734% |
| `J_peak, J_place, J_persist.simd_schedule.d1.512B` | 9 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.32768B` | 9 | 0.734% |
| `Q_fine, J_place, J_persist.simd_schedule.d1.32768B` | 9 | 0.734% |
| `J_peak, J_persist.simd_schedule.d1.2048B` | 3 | 0.765% |
| `J_persist.simd_schedule.d1` | 1 | 0.909% |
| `J_area, J_persist.32B` | 2 | 0.909% |
| `J_area, J_persist.d1.32B` | 2 | 0.909% |
| `J_area, J_persist.d4.32B` | 2 | 0.909% |
| `J_area, J_persist.lane_stream.32B` | 2 | 0.909% |
| `J_area, J_persist.lane_stream.d1.32B` | 2 | 0.909% |
| `J_area, J_persist.simd_schedule.32B` | 2 | 0.909% |
| `J_area, J_persist.simd_schedule.d1` | 2 | 0.909% |
| `J_area, J_persist.simd_schedule.d1.32B` | 2 | 0.909% |
| `J_area, J_persist.simd_schedule.d4.32B` | 2 | 0.909% |
| `J_area, J_persist.simd_stream.32B` | 2 | 0.909% |
| `J_area, J_persist.simd_stream.d1.32B` | 2 | 0.909% |
| `J_peak, J_area, J_persist.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.d1.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.lane_stream.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.lane_stream.d1.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.simd_schedule.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.simd_schedule.d1.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.simd_stream.32B` | 3 | 0.909% |
| `J_peak, J_area, J_persist.simd_stream.d1.32B` | 3 | 0.909% |
| `J_area, J_persist.mean.d1` | 4 | 0.909% |
| `J_peak, J_area, J_persist.d4.32B` | 4 | 0.909% |
| `J_peak, J_area, J_persist.simd_schedule.d4.32B` | 4 | 0.909% |
| `J_area, J_persist.64B` | 7 | 0.909% |
| `J_area, J_persist.d1.64B` | 7 | 0.909% |
| `J_area, J_persist.d4.64B` | 7 | 0.909% |
| `J_area, J_persist.lane_stream.64B` | 7 | 0.909% |
| `J_area, J_persist.lane_stream.d1.64B` | 7 | 0.909% |
| `J_area, J_persist.simd_schedule.64B` | 7 | 0.909% |
| `J_area, J_persist.simd_schedule.d1.64B` | 7 | 0.909% |
| `J_area, J_persist.simd_schedule.d4.64B` | 7 | 0.909% |
| `J_area, J_persist.simd_stream.64B` | 7 | 0.909% |
| `J_area, J_persist.simd_stream.d1.64B` | 7 | 0.909% |
| `J_area, J_persist.128B` | 9 | 0.909% |
| `J_area, J_persist.d1.128B` | 9 | 0.909% |
| `J_area, J_persist.d4.128B` | 9 | 0.909% |
| `J_area, J_persist.lane_stream.128B` | 9 | 0.909% |
| `J_area, J_persist.lane_stream.d1.128B` | 9 | 0.909% |
| `J_area, J_persist.simd_schedule.128B` | 9 | 0.909% |
| `J_area, J_persist.simd_schedule.d1.128B` | 9 | 0.909% |
| `J_area, J_persist.simd_schedule.d4.128B` | 9 | 0.909% |
| `J_area, J_persist.simd_stream.128B` | 9 | 0.909% |
| `J_area, J_persist.simd_stream.d1.128B` | 9 | 0.909% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `Q_fine, J_persist.32768B` | 1 | 0.195% |
| `best_below_ten_samples` | `J_peak, J_persist.16384B` | 3 | 0.000% |
| `best_below_five_samples` | `J_peak, J_persist.16384B` | 3 | 0.000% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.16384B` | 3 | 0.000% |
| `J_peak, J_persist.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d1.16384B` | 3 | 0.000% |
| `J_peak, J_persist.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d16.16384B` | 3 | 0.000% |
| `J_peak, J_persist.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d4.16384B` | 3 | 0.000% |
| `J_peak, J_persist.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.16384B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.32768B` | 3 | 0.000% |

## SYRK N=1024

Oracle median: 1.418071 ms over 182 layouts.

### Main frontier comparison

| Frontier | Samples | Best regret | Oracle retained |
|---|---:|---:|:---:|
| `current_locality` | 11 | 2.661% | no |
| `locality_plus_persist` | 39 | 0.000% | yes |
| `locality_plus_place` | 35 | 2.507% | no |
| `all_five` | 100 | 0.000% | yes |

### Diagnostic layouts

| Role | Word | Runtime regret | J_persist |
|---|---|---:|---:|
| `oracle` | `iijjjjjjjjjjiiiiiiii` | 0.000% | 31927056.000000 |
| `current_selection` | `iijiiiiiiiijjjjjjjjj` | 3.276% | 65912576.000000 |

### Combinations meeting <1% regret and <10 samples

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.32768B` | 3 | 0.000% |
| `Q_fine, J_persist.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.d1.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.d4.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d16.32768B` | 4 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.32768B` | 4 | 0.000% |
| `J_peak, J_persist` | 5 | 0.000% |
| `J_peak, J_persist.d1` | 5 | 0.000% |
| `J_peak, J_persist.lane_stream` | 5 | 0.000% |
| `J_peak, J_persist.lane_stream.d1` | 5 | 0.000% |
| `J_peak, J_persist.lane_stream.d4` | 5 | 0.000% |
| `J_peak, J_persist.mean.lane_stream` | 5 | 0.000% |
| `J_peak, J_persist.mean.simd_stream` | 5 | 0.000% |
| `J_peak, J_persist.simd_stream` | 5 | 0.000% |
| `J_peak, J_persist.simd_stream.d1` | 5 | 0.000% |
| `J_peak, J_persist.simd_stream.d4` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.d1.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.d16.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.d4.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d16.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d4.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d16.32768B` | 5 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d4.32768B` | 5 | 0.000% |
| `J_peak, J_persist.d4` | 6 | 0.000% |
| `J_peak, J_persist.mean.d4` | 6 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4` | 6 | 0.000% |
| `Q_fine, J_persist.d4` | 6 | 0.000% |
| `Q_fine, J_persist.mean.d4` | 6 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4` | 6 | 0.000% |
| `J_peak, J_persist.16384B` | 8 | 0.000% |
| `J_peak, J_persist.d1.16384B` | 8 | 0.000% |
| `J_peak, J_persist.d16` | 8 | 0.000% |
| `J_peak, J_persist.d16.16384B` | 8 | 0.000% |
| `J_peak, J_persist.d4.16384B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.16384B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.16384B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d16` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d16.16384B` | 8 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.16384B` | 8 | 0.000% |
| `J_peak, J_persist.mean.d16` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.16384B` | 8 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.16384B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.16384B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d1.16384B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d16` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d16.16384B` | 8 | 0.000% |
| `J_peak, J_persist.simd_stream.d4.16384B` | 8 | 0.000% |
| `Q_fine, J_persist.d16` | 8 | 0.000% |
| `Q_fine, J_persist.lane_stream.d16` | 8 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4` | 8 | 0.000% |
| `Q_fine, J_persist.mean.d16` | 8 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16` | 8 | 0.000% |
| `Q_fine, J_persist.simd_stream.d16` | 8 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4` | 8 | 0.000% |
| `J_peak, J_area, J_persist.lane_stream.d16.32768B` | 8 | 0.000% |
| `J_peak, J_area, J_persist.simd_stream.d16.32768B` | 8 | 0.000% |
| `Q_fine, J_persist.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.d1.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.d16.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.d4.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.d1.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.d16.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.lane_stream.d4.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d16.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_schedule.d4.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.d1.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.d16.16384B` | 9 | 0.000% |
| `Q_fine, J_persist.simd_stream.d4.16384B` | 9 | 0.000% |
| `J_peak, J_area, J_persist.d16.32768B` | 9 | 0.000% |
| `J_peak, J_area, J_persist.simd_schedule.d16.32768B` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.d16` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.d4` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.lane_stream.d16` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d16` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.d4` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.lane_stream` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.mean.simd_stream` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d16` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_schedule.d4` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d1` | 9 | 0.000% |
| `Q_fine, J_peak, J_persist.simd_stream.d16` | 9 | 0.000% |

### Target boundary

The objective variants below are post-hoc ablations on this one measured instance; they are diagnostics, not calibrated transferable weights.

| Boundary | Objectives | Samples | Best regret |
|---|---|---:|---:|
| `smallest_below_one_percent` | `J_peak, J_persist.32768B` | 3 | 0.000% |
| `best_below_ten_samples` | `J_peak, J_persist.32768B` | 3 | 0.000% |
| `best_below_five_samples` | `J_peak, J_persist.32768B` | 3 | 0.000% |

### Best compact combinations

| Objectives | Samples | Best regret |
|---|---:|---:|
| `J_peak, J_persist.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d1.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.lane_stream.d4.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d16.32768B` | 3 | 0.000% |
| `J_peak, J_persist.simd_schedule.d4.32768B` | 3 | 0.000% |

