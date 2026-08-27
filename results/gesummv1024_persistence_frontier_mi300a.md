# Temporal quotient persistence frontier experiment

This score-only experiment reuses the exhaustive measured G_S corpus; no new timings were collected. J_place is the frozen corrected robust statistic from the input plan.

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

