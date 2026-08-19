# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runs and XORs are separate address-code generation costs. They are included in the Pareto frontier but are not folded into the scalar locality score or score rank.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| GEMM | 256 | 22 | 1 | — | — |
| GEMM | 512 | 22 | 1 | — | — |
| GEMM | 1024 | 22 | 1 | — | — |
| GESUMMV | 256 | 22 | 6 | — | — |
| GESUMMV | 512 | 22 | 6 | — | — |
| GESUMMV | 1024 | 22 | 6 | — | — |

## GEMM — N=256

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 4 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0.00195312 | logical addresses issued by the traced C wave store |
| `B.wave_lane_group.lane8.64B` | hypothesis | 64 | 0.125 | contiguous groups of 8 lanes |
| `B.wave_lane_group.lane16.128B` | hypothesis | 128 | 0.125 | contiguous groups of 16 lanes |
| `B.wave_lane_group.lane32.256B` | hypothesis | 256 | 0.125 | contiguous groups of 32 lanes |
| `B.wave_lane_group.lane64.512B` | hypothesis | 512 | 0.125 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 2 | sixteen consecutive k-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0.5 | unique A or B values reused across a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 1 | sixteen consecutive A/B load pairs in a cache-scale region |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0.25 | one wave's complete k-loop working set in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile32x8_row_major` | 24704 | 5 | 13.6284 | 6 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 1 | `tile32x8_row_major` | `jjjiiiii` | 13.6284 | 6 | 0 |
| 2 | `tile16x8_row_major` | `jjjiiii` | 13.7117 | 6 | 0 |
| 3 | `tile32_interleaved` | `jijijijiji` | 13.8885 | 30 | 0 |
| 4 | `tile8_row_major` | `jjjiii` | 14.2117 | 6 | 0 |
| 5 | `tile16_interleaved` | `jijijiji` | 14.4718 | 24 | 0 |
| 6 | `tile32x16_row_major` | `jjjjiiiii` | 21.2534 | 6 | 0 |
| 7 | `tile16_row_major` | `jjjjiiii` | 21.3367 | 6 | 0 |
| 8 | `tile8x16_row_major` | `jjjjiii` | 21.8367 | 6 | 0 |
| 9 | `tile32_row_major` | `jjjjjiiiii` | 24.1284 | 6 | 0 |
| 10 | `tile16x32_row_major` | `jjjjjiiii` | 24.7117 | 6 | 0 |
| 11 | `tile8x32_row_major` | `jjjjjiii` | 25.2117 | 6 | 0 |
| 12 | `row_major` | `jjjjjjjjiiiiiiii` | 28.3367 | 6 | 0 |
| 14 | `tile8_column_major` | `iiijjj` | 37.8829 | 6 | 0 |
| 14 | `tile8x16_column_major` | `iiijjjj` | 37.8829 | 6 | 0 |
| 14 | `tile8x32_column_major` | `iiijjjjj` | 37.8829 | 6 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 48.8829 | 6 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 48.8829 | 6 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 48.8829 | 6 | 0 |
| 20 | `tile32_column_major` | `iiiiijjjjj` | 56.5496 | 6 | 0 |
| 20 | `tile32x16_column_major` | `iiiiijjjj` | 56.5496 | 6 | 0 |
| 20 | `tile32x8_column_major` | `iiiiijjj` | 56.5496 | 6 | 0 |
| 22 | `column_major` | `iiiiiiiijjjjjjjj` | 71.7163 | 6 | 0 |

## GEMM — N=512

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 4 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0.000976562 | logical addresses issued by the traced C wave store |
| `B.wave_lane_group.lane8.64B` | hypothesis | 64 | 0.125 | contiguous groups of 8 lanes |
| `B.wave_lane_group.lane16.128B` | hypothesis | 128 | 0.125 | contiguous groups of 16 lanes |
| `B.wave_lane_group.lane32.256B` | hypothesis | 256 | 0.125 | contiguous groups of 32 lanes |
| `B.wave_lane_group.lane64.512B` | hypothesis | 512 | 0.125 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 2 | sixteen consecutive k-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0.5 | unique A or B values reused across a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 1 | sixteen consecutive A/B load pairs in a cache-scale region |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0.25 | one wave's complete k-loop working set in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile32x8_row_major` | 49280 | 5 | 13.7975 | 6 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 1 | `tile32x8_row_major` | `jjjiiiii` | 13.7975 | 6 | 0 |
| 2 | `tile16x8_row_major` | `jjjiiii` | 13.9975 | 6 | 0 |
| 3 | `tile32_interleaved` | `jijijijiji` | 14.065 | 30 | 0 |
| 4 | `tile16_interleaved` | `jijijiji` | 14.765 | 24 | 0 |
| 5 | `tile8_row_major` | `jjjiii` | 16.0475 | 6 | 0 |
| 6 | `tile32x16_row_major` | `jjjjiiiii` | 21.4225 | 6 | 0 |
| 7 | `tile16_row_major` | `jjjjiiii` | 21.6225 | 6 | 0 |
| 8 | `tile8x16_row_major` | `jjjjiii` | 23.6725 | 6 | 0 |
| 9 | `tile32_row_major` | `jjjjjiiiii` | 24.2975 | 6 | 0 |
| 10 | `tile16x32_row_major` | `jjjjjiiii` | 24.9975 | 6 | 0 |
| 11 | `tile8x32_row_major` | `jjjjjiii` | 27.0475 | 6 | 0 |
| 12 | `row_major` | `jjjjjjjjjiiiiiiiii` | 34.6725 | 6 | 0 |
| 14 | `tile8_column_major` | `iiijjj` | 39.7455 | 6 | 0 |
| 14 | `tile8x16_column_major` | `iiijjjj` | 39.7455 | 6 | 0 |
| 14 | `tile8x32_column_major` | `iiijjjjj` | 39.7455 | 6 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 49.1955 | 6 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 49.1955 | 6 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 49.1955 | 6 | 0 |
| 20 | `tile32_column_major` | `iiiiijjjjj` | 56.7455 | 6 | 0 |
| 20 | `tile32x16_column_major` | `iiiiijjjj` | 56.7455 | 6 | 0 |
| 20 | `tile32x8_column_major` | `iiiiijjj` | 56.7455 | 6 | 0 |
| 22 | `column_major` | `iiiiiiiiijjjjjjjjj` | 85.6455 | 6 | 0 |

## GEMM — N=1024

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 4 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0.000488281 | logical addresses issued by the traced C wave store |
| `B.wave_lane_group.lane8.64B` | hypothesis | 64 | 0.125 | contiguous groups of 8 lanes |
| `B.wave_lane_group.lane16.128B` | hypothesis | 128 | 0.125 | contiguous groups of 16 lanes |
| `B.wave_lane_group.lane32.256B` | hypothesis | 256 | 0.125 | contiguous groups of 32 lanes |
| `B.wave_lane_group.lane64.512B` | hypothesis | 512 | 0.125 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 2 | sixteen consecutive k-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0.5 | unique A or B values reused across a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 1 | sixteen consecutive A/B load pairs in a cache-scale region |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0.25 | one wave's complete k-loop working set in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile32x8_row_major` | 98432 | 5 | 13.9099 | 6 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 1 | `tile32x8_row_major` | `jjjiiiii` | 13.9099 | 6 | 0 |
| 2 | `tile32_interleaved` | `jijijijiji` | 14.1811 | 30 | 0 |
| 3 | `tile16x8_row_major` | `jjjiiii` | 14.1876 | 6 | 0 |
| 4 | `tile16_interleaved` | `jijijiji` | 14.9589 | 24 | 0 |
| 5 | `tile8_row_major` | `jjjiii` | 16.4099 | 6 | 0 |
| 6 | `tile32x16_row_major` | `jjjjiiiii` | 21.5349 | 6 | 0 |
| 7 | `tile16_row_major` | `jjjjiiii` | 21.8126 | 6 | 0 |
| 8 | `tile8x16_row_major` | `jjjjiii` | 24.0349 | 6 | 0 |
| 9 | `tile32_row_major` | `jjjjjiiiii` | 24.4099 | 6 | 0 |
| 10 | `tile16x32_row_major` | `jjjjjiiii` | 25.1876 | 6 | 0 |
| 11 | `tile8x32_row_major` | `jjjjjiii` | 27.4099 | 6 | 0 |
| 12 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 38.5626 | 6 | 0 |
| 14 | `tile8_column_major` | `iiijjj` | 40.1214 | 6 | 0 |
| 14 | `tile8x16_column_major` | `iiijjjj` | 40.1214 | 6 | 0 |
| 14 | `tile8x32_column_major` | `iiijjjjj` | 40.1214 | 6 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 49.3991 | 6 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 49.3991 | 6 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 49.3991 | 6 | 0 |
| 20 | `tile32_column_major` | `iiiiijjjjj` | 56.8714 | 6 | 0 |
| 20 | `tile32x16_column_major` | `iiiiijjjj` | 56.8714 | 6 | 0 |
| 20 | `tile32x8_column_major` | `iiiiijjj` | 56.8714 | 6 | 0 |
| 22 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 89.5936 | 6 | 0 |

## GESUMMV — N=256

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 1 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0.00195312 | logical addresses issued by the traced wave store |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0.125 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 0.125 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0.125 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0.125 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 2 | sixteen consecutive inner-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one wave's 64 FP64 matrix values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0.5 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 0.5 | one wave's complete matrix-read phase in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 8192 | 15 | 24.625 | 4 | 0 |
| `tile8x16_column_major` | 8192 | 15 | 24.625 | 4 | 0 |
| `tile8x32_column_major` | 8192 | 15 | 24.625 | 4 | 0 |
| `tile16_interleaved` | 32768 | 15 | 20.75 | 16 | 0 |
| `tile16x8_row_major` | 65536 | 7 | 17.75 | 4 | 0 |
| `tile32x8_row_major` | 65536 | 7 | 17.75 | 4 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 1.5 | `tile16x8_row_major` | `jjjiiii` | 17.75 | 4 | 0 |
| 1.5 | `tile32x8_row_major` | `jjjiiiii` | 17.75 | 4 | 0 |
| 3.5 | `tile16_interleaved` | `jijijiji` | 20.75 | 16 | 0 |
| 3.5 | `tile32_interleaved` | `jijijijiji` | 20.75 | 20 | 0 |
| 5 | `tile8_row_major` | `jjjiii` | 21.75 | 4 | 0 |
| 7 | `tile8_column_major` | `iiijjj` | 24.625 | 4 | 0 |
| 7 | `tile8x16_column_major` | `iiijjjj` | 24.625 | 4 | 0 |
| 7 | `tile8x32_column_major` | `iiijjjjj` | 24.625 | 4 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 24.75 | 4 | 0 |
| 10 | `tile32x16_row_major` | `jjjjiiiii` | 24.75 | 4 | 0 |
| 10 | `tile8x16_row_major` | `jjjjiii` | 24.75 | 4 | 0 |
| 12 | `column_major` | `iiiiiiiijjjjjjjj` | 31.5 | 4 | 0 |
| 14 | `tile32_column_major` | `iiiiijjjjj` | 31.875 | 4 | 0 |
| 14 | `tile32x16_column_major` | `iiiiijjjj` | 31.875 | 4 | 0 |
| 14 | `tile32x8_column_major` | `iiiiijjj` | 31.875 | 4 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 34.75 | 4 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 34.75 | 4 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 34.75 | 4 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 40.75 | 4 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 40.75 | 4 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 40.75 | 4 | 0 |
| 22 | `row_major` | `jjjjjjjjiiiiiiii` | 100.75 | 4 | 0 |

## GESUMMV — N=512

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 1 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0.000976562 | logical addresses issued by the traced wave store |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0.125 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 0.125 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0.125 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0.125 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 2 | sixteen consecutive inner-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one wave's 64 FP64 matrix values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0.5 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 0.5 | one wave's complete matrix-read phase in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 16384 | 15 | 24.625 | 4 | 0 |
| `tile8x16_column_major` | 16384 | 15 | 24.625 | 4 | 0 |
| `tile8x32_column_major` | 16384 | 15 | 24.625 | 4 | 0 |
| `tile16_interleaved` | 65536 | 15 | 20.75 | 16 | 0 |
| `tile16x8_row_major` | 131072 | 7 | 17.75 | 4 | 0 |
| `tile32x8_row_major` | 131072 | 7 | 17.75 | 4 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 1.5 | `tile16x8_row_major` | `jjjiiii` | 17.75 | 4 | 0 |
| 1.5 | `tile32x8_row_major` | `jjjiiiii` | 17.75 | 4 | 0 |
| 3.5 | `tile16_interleaved` | `jijijiji` | 20.75 | 16 | 0 |
| 3.5 | `tile32_interleaved` | `jijijijiji` | 20.75 | 20 | 0 |
| 5 | `tile8_row_major` | `jjjiii` | 21.75 | 4 | 0 |
| 7 | `tile8_column_major` | `iiijjj` | 24.625 | 4 | 0 |
| 7 | `tile8x16_column_major` | `iiijjjj` | 24.625 | 4 | 0 |
| 7 | `tile8x32_column_major` | `iiijjjjj` | 24.625 | 4 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 24.75 | 4 | 0 |
| 10 | `tile32x16_row_major` | `jjjjiiiii` | 24.75 | 4 | 0 |
| 10 | `tile8x16_row_major` | `jjjjiii` | 24.75 | 4 | 0 |
| 13 | `tile32_column_major` | `iiiiijjjjj` | 31.875 | 4 | 0 |
| 13 | `tile32x16_column_major` | `iiiiijjjj` | 31.875 | 4 | 0 |
| 13 | `tile32x8_column_major` | `iiiiijjj` | 31.875 | 4 | 0 |
| 15 | `column_major` | `iiiiiiiiijjjjjjjjj` | 33.5 | 4 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 34.75 | 4 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 34.75 | 4 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 34.75 | 4 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 40.75 | 4 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 40.75 | 4 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 40.75 | 4 | 0 |
| 22 | `row_major` | `jjjjjjjjjiiiiiiiii` | 100.75 | 4 | 0 |

## GESUMMV — N=1024

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 1 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0.000488281 | logical addresses issued by the traced wave store |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0.125 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 0.125 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0.125 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0.125 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 2 | sixteen consecutive inner-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one wave's 64 FP64 matrix values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0.5 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 0.5 | one wave's complete matrix-read phase in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 32768 | 15 | 24.625 | 4 | 0 |
| `tile8x16_column_major` | 32768 | 15 | 24.625 | 4 | 0 |
| `tile8x32_column_major` | 32768 | 15 | 24.625 | 4 | 0 |
| `tile16_interleaved` | 131072 | 15 | 20.75 | 16 | 0 |
| `tile16x8_row_major` | 262144 | 7 | 17.75 | 4 | 0 |
| `tile32x8_row_major` | 262144 | 7 | 17.75 | 4 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 1.5 | `tile16x8_row_major` | `jjjiiii` | 17.75 | 4 | 0 |
| 1.5 | `tile32x8_row_major` | `jjjiiiii` | 17.75 | 4 | 0 |
| 3.5 | `tile16_interleaved` | `jijijiji` | 20.75 | 16 | 0 |
| 3.5 | `tile32_interleaved` | `jijijijiji` | 20.75 | 20 | 0 |
| 5 | `tile8_row_major` | `jjjiii` | 21.75 | 4 | 0 |
| 7 | `tile8_column_major` | `iiijjj` | 24.625 | 4 | 0 |
| 7 | `tile8x16_column_major` | `iiijjjj` | 24.625 | 4 | 0 |
| 7 | `tile8x32_column_major` | `iiijjjjj` | 24.625 | 4 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 24.75 | 4 | 0 |
| 10 | `tile32x16_row_major` | `jjjjiiiii` | 24.75 | 4 | 0 |
| 10 | `tile8x16_row_major` | `jjjjiii` | 24.75 | 4 | 0 |
| 13 | `tile32_column_major` | `iiiiijjjjj` | 31.875 | 4 | 0 |
| 13 | `tile32x16_column_major` | `iiiiijjjj` | 31.875 | 4 | 0 |
| 13 | `tile32x8_column_major` | `iiiiijjj` | 31.875 | 4 | 0 |
| 15 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 33.5 | 4 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 34.75 | 4 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 34.75 | 4 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 34.75 | 4 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 40.75 | 4 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 40.75 | 4 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 40.75 | 4 | 0 |
| 22 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 100.75 | 4 | 0 |
