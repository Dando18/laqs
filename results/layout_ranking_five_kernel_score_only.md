# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runs and XORs are separate address-code generation costs. They are included in the Pareto frontier but are not folded into the scalar locality score or score rank.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| ATAX | 256 | 22 | 4 | — | — |
| ATAX | 512 | 22 | 4 | — | — |
| ATAX | 1024 | 22 | 4 | — | — |
| GEMM | 256 | 22 | 4 | — | — |
| GEMM | 512 | 22 | 4 | — | — |
| GEMM | 1024 | 22 | 4 | — | — |
| GESUMMV | 256 | 22 | 7 | — | — |
| GESUMMV | 512 | 22 | 7 | — | — |
| GESUMMV | 1024 | 22 | 7 | — | — |
| MVT | 256 | 22 | 5 | — | — |
| MVT | 512 | 22 | 5 | — | — |
| MVT | 1024 | 22 | 5 | — | — |
| SYRK | 256 | 22 | 4 | — | — |
| SYRK | 512 | 22 | 4 | — | — |
| SYRK | 1024 | 22 | 4 | — | — |

## ATAX — N=256

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical A addresses issued by one traced wave load in either pass |
| `stage1_wave_load.64B` | grounded | 64 | 0.25 | logical A addresses issued by one traced first-stage wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the tmp and y wave stores |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 4 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `stage1_wave_neighborhood.256B` | hypothesis | 256 | 1 | first-stage A wave values in a 256-byte neighborhood; the stage asymmetry is an empirically calibrated cache hypothesis |
| `lane_reuse.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive reduction values used by one lane in either pass |
| `wave_neighborhood.512B` | hypothesis | 512 | 0 | one wave's A values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | A row or column panel shared by all waves at one reduction step |
| `wave_phase.4096B` | hypothesis | 4096 | 2 | one wave's complete row-wise or column-wise A pass in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16_interleaved` | 24576 | 7 | 19.75 | 8 | 0 |
| `tile8_column_major` | 36864 | 4 | 19 | 2 | 0 |
| `tile8x16_column_major` | 36864 | 4 | 19 | 2 | 0 |
| `tile8x32_column_major` | 36864 | 4 | 19 | 2 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 |
| 2 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 |
| 2 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 |
| 4.5 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 |
| 4.5 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 |
| 7 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 |
| 7 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 |
| 7 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 |
| 10 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 |
| 10 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 |
| 10 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 |
| 13 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 |
| 13 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 |
| 13 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 |
| 15 | `column_major` | `iiiiiiiijjjjjjjj` | 33 | 2 | 0 |
| 17 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 |
| 17 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 |
| 17 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 |
| 22 | `row_major` | `jjjjjjjjiiiiiiii` | 65.75 | 2 | 0 |

## ATAX — N=512

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical A addresses issued by one traced wave load in either pass |
| `stage1_wave_load.64B` | grounded | 64 | 0.25 | logical A addresses issued by one traced first-stage wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the tmp and y wave stores |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 4 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `stage1_wave_neighborhood.256B` | hypothesis | 256 | 1 | first-stage A wave values in a 256-byte neighborhood; the stage asymmetry is an empirically calibrated cache hypothesis |
| `lane_reuse.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive reduction values used by one lane in either pass |
| `wave_neighborhood.512B` | hypothesis | 512 | 0 | one wave's A values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | A row or column panel shared by all waves at one reduction step |
| `wave_phase.4096B` | hypothesis | 4096 | 2 | one wave's complete row-wise or column-wise A pass in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16_interleaved` | 49152 | 7 | 19.75 | 8 | 0 |
| `tile8_column_major` | 73728 | 4 | 19 | 2 | 0 |
| `tile8x16_column_major` | 73728 | 4 | 19 | 2 | 0 |
| `tile8x32_column_major` | 73728 | 4 | 19 | 2 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 |
| 2 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 |
| 2 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 |
| 4.5 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 |
| 4.5 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 |
| 7 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 |
| 7 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 |
| 7 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 |
| 10 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 |
| 10 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 |
| 10 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 |
| 13 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 |
| 13 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 |
| 13 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 |
| 15 | `column_major` | `iiiiiiiiijjjjjjjjj` | 37 | 2 | 0 |
| 17 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 |
| 17 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 |
| 17 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 |
| 22 | `row_major` | `jjjjjjjjjiiiiiiiii` | 69.75 | 2 | 0 |

## ATAX — N=1024

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical A addresses issued by one traced wave load in either pass |
| `stage1_wave_load.64B` | grounded | 64 | 0.25 | logical A addresses issued by one traced first-stage wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the tmp and y wave stores |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 4 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `stage1_wave_neighborhood.256B` | hypothesis | 256 | 1 | first-stage A wave values in a 256-byte neighborhood; the stage asymmetry is an empirically calibrated cache hypothesis |
| `lane_reuse.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive reduction values used by one lane in either pass |
| `wave_neighborhood.512B` | hypothesis | 512 | 0 | one wave's A values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | A row or column panel shared by all waves at one reduction step |
| `wave_phase.4096B` | hypothesis | 4096 | 2 | one wave's complete row-wise or column-wise A pass in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16_interleaved` | 98304 | 7 | 19.75 | 8 | 0 |
| `tile8_column_major` | 147456 | 4 | 19 | 2 | 0 |
| `tile8x16_column_major` | 147456 | 4 | 19 | 2 | 0 |
| `tile8x32_column_major` | 147456 | 4 | 19 | 2 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 |
| 2 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 |
| 2 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 |
| 4.5 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 |
| 4.5 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 |
| 7 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 |
| 7 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 |
| 7 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 |
| 10 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 |
| 10 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 |
| 10 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 |
| 13 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 |
| 13 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 |
| 13 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 |
| 15 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 37 | 2 | 0 |
| 17 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 |
| 17 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 |
| 17 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 |
| 22 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 69.75 | 2 | 0 |

## GEMM — N=256

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 4 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced C wave store |
| `B.wave_lane_group.lane8.64B` | hypothesis | 64 | 2 | contiguous groups of 8 lanes |
| `B.wave_lane_group.lane16.128B` | hypothesis | 128 | 2 | contiguous groups of 16 lanes |
| `B.wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `B.wave_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | sixteen consecutive k-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 1 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0 | unique A or B values reused across a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive A/B load pairs in a cache-scale region |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete k-loop working set in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16x8_row_major` | 24704 | 4 | 8.29503 | 6 | 0 |
| `tile32x8_row_major` | 24704 | 4 | 8.29503 | 6 | 0 |
| `tile8_row_major` | 24704 | 4 | 8.29503 | 6 | 0 |
| `tile16_interleaved` | 36992 | 3 | 15.6801 | 24 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 3.5 | `tile16x32_row_major` | `jjjjjiiii` | 8.29503 | 6 | 0 |
| 3.5 | `tile16x8_row_major` | `jjjiiii` | 8.29503 | 6 | 0 |
| 3.5 | `tile32_row_major` | `jjjjjiiiii` | 8.29503 | 6 | 0 |
| 3.5 | `tile32x8_row_major` | `jjjiiiii` | 8.29503 | 6 | 0 |
| 3.5 | `tile8_row_major` | `jjjiii` | 8.29503 | 6 | 0 |
| 3.5 | `tile8x32_row_major` | `jjjjjiii` | 8.29503 | 6 | 0 |
| 8.5 | `row_major` | `jjjjjjjjiiiiiiii` | 8.79503 | 6 | 0 |
| 8.5 | `tile16_row_major` | `jjjjiiii` | 8.79503 | 6 | 0 |
| 8.5 | `tile32x16_row_major` | `jjjjiiiii` | 8.79503 | 6 | 0 |
| 8.5 | `tile8x16_row_major` | `jjjjiii` | 8.79503 | 6 | 0 |
| 11.5 | `tile16_interleaved` | `jijijiji` | 15.6801 | 24 | 0 |
| 11.5 | `tile32_interleaved` | `jijijijiji` | 15.6801 | 30 | 0 |
| 14 | `tile8_column_major` | `iiijjj` | 55.8354 | 6 | 0 |
| 14 | `tile8x16_column_major` | `iiijjjj` | 55.8354 | 6 | 0 |
| 14 | `tile8x32_column_major` | `iiijjjjj` | 55.8354 | 6 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 77.3354 | 6 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 77.3354 | 6 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 77.3354 | 6 | 0 |
| 20 | `tile32_column_major` | `iiiiijjjjj` | 81.3354 | 6 | 0 |
| 20 | `tile32x16_column_major` | `iiiiijjjj` | 81.3354 | 6 | 0 |
| 20 | `tile32x8_column_major` | `iiiiijjj` | 81.3354 | 6 | 0 |
| 22 | `column_major` | `iiiiiiiijjjjjjjj` | 89.3354 | 6 | 0 |

## GEMM — N=512

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 4 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced C wave store |
| `B.wave_lane_group.lane8.64B` | hypothesis | 64 | 2 | contiguous groups of 8 lanes |
| `B.wave_lane_group.lane16.128B` | hypothesis | 128 | 2 | contiguous groups of 16 lanes |
| `B.wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `B.wave_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | sixteen consecutive k-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 1 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0 | unique A or B values reused across a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive A/B load pairs in a cache-scale region |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete k-loop working set in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16x8_row_major` | 49280 | 4 | 8.29751 | 6 | 0 |
| `tile32x8_row_major` | 49280 | 4 | 8.29751 | 6 | 0 |
| `tile8_row_major` | 49280 | 4 | 8.29751 | 6 | 0 |
| `tile16_interleaved` | 73856 | 3 | 15.69 | 24 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 3.5 | `tile16x32_row_major` | `jjjjjiiii` | 8.29751 | 6 | 0 |
| 3.5 | `tile16x8_row_major` | `jjjiiii` | 8.29751 | 6 | 0 |
| 3.5 | `tile32_row_major` | `jjjjjiiiii` | 8.29751 | 6 | 0 |
| 3.5 | `tile32x8_row_major` | `jjjiiiii` | 8.29751 | 6 | 0 |
| 3.5 | `tile8_row_major` | `jjjiii` | 8.29751 | 6 | 0 |
| 3.5 | `tile8x32_row_major` | `jjjjjiii` | 8.29751 | 6 | 0 |
| 8.5 | `row_major` | `jjjjjjjjjiiiiiiiii` | 8.79751 | 6 | 0 |
| 8.5 | `tile16_row_major` | `jjjjiiii` | 8.79751 | 6 | 0 |
| 8.5 | `tile32x16_row_major` | `jjjjiiiii` | 8.79751 | 6 | 0 |
| 8.5 | `tile8x16_row_major` | `jjjjiii` | 8.79751 | 6 | 0 |
| 11.5 | `tile16_interleaved` | `jijijiji` | 15.69 | 24 | 0 |
| 11.5 | `tile32_interleaved` | `jijijijiji` | 15.69 | 30 | 0 |
| 14 | `tile8_column_major` | `iiijjj` | 55.8676 | 6 | 0 |
| 14 | `tile8x16_column_major` | `iiijjjj` | 55.8676 | 6 | 0 |
| 14 | `tile8x32_column_major` | `iiijjjjj` | 55.8676 | 6 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 77.3676 | 6 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 77.3676 | 6 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 77.3676 | 6 | 0 |
| 20 | `tile32_column_major` | `iiiiijjjjj` | 81.3676 | 6 | 0 |
| 20 | `tile32x16_column_major` | `iiiiijjjj` | 81.3676 | 6 | 0 |
| 20 | `tile32x8_column_major` | `iiiiijjj` | 81.3676 | 6 | 0 |
| 22 | `column_major` | `iiiiiiiiijjjjjjjjj` | 89.3676 | 6 | 0 |

## GEMM — N=1024

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 4 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced C wave store |
| `B.wave_lane_group.lane8.64B` | hypothesis | 64 | 2 | contiguous groups of 8 lanes |
| `B.wave_lane_group.lane16.128B` | hypothesis | 128 | 2 | contiguous groups of 16 lanes |
| `B.wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `B.wave_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | sixteen consecutive k-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 1 | one inner-loop wave load at a broader locality scale |
| `workgroup_k_panel.256B` | hypothesis | 256 | 0 | unique A or B values reused across a workgroup at one k step |
| `wave_k_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive A/B load pairs in a cache-scale region |
| `wave_inner_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete k-loop working set in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16x8_row_major` | 98432 | 4 | 8.29875 | 6 | 0 |
| `tile32x8_row_major` | 98432 | 4 | 8.29875 | 6 | 0 |
| `tile8_row_major` | 98432 | 4 | 8.29875 | 6 | 0 |
| `tile16_interleaved` | 147584 | 3 | 15.695 | 24 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 3.5 | `tile16x32_row_major` | `jjjjjiiii` | 8.29875 | 6 | 0 |
| 3.5 | `tile16x8_row_major` | `jjjiiii` | 8.29875 | 6 | 0 |
| 3.5 | `tile32_row_major` | `jjjjjiiiii` | 8.29875 | 6 | 0 |
| 3.5 | `tile32x8_row_major` | `jjjiiiii` | 8.29875 | 6 | 0 |
| 3.5 | `tile8_row_major` | `jjjiii` | 8.29875 | 6 | 0 |
| 3.5 | `tile8x32_row_major` | `jjjjjiii` | 8.29875 | 6 | 0 |
| 8.5 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 8.79875 | 6 | 0 |
| 8.5 | `tile16_row_major` | `jjjjiiii` | 8.79875 | 6 | 0 |
| 8.5 | `tile32x16_row_major` | `jjjjiiiii` | 8.79875 | 6 | 0 |
| 8.5 | `tile8x16_row_major` | `jjjjiii` | 8.79875 | 6 | 0 |
| 11.5 | `tile16_interleaved` | `jijijiji` | 15.695 | 24 | 0 |
| 11.5 | `tile32_interleaved` | `jijijijiji` | 15.695 | 30 | 0 |
| 14 | `tile8_column_major` | `iiijjj` | 55.8838 | 6 | 0 |
| 14 | `tile8x16_column_major` | `iiijjjj` | 55.8838 | 6 | 0 |
| 14 | `tile8x32_column_major` | `iiijjjjj` | 55.8838 | 6 | 0 |
| 17 | `tile16_column_major` | `iiiijjjj` | 77.3838 | 6 | 0 |
| 17 | `tile16x32_column_major` | `iiiijjjjj` | 77.3838 | 6 | 0 |
| 17 | `tile16x8_column_major` | `iiiijjj` | 77.3838 | 6 | 0 |
| 20 | `tile32_column_major` | `iiiiijjjjj` | 81.3838 | 6 | 0 |
| 20 | `tile32x16_column_major` | `iiiiijjjj` | 81.3838 | 6 | 0 |
| 20 | `tile32x8_column_major` | `iiiiijjj` | 81.3838 | 6 | 0 |
| 22 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 89.3838 | 6 | 0 |

## GESUMMV — N=256

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced wave store |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 0 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0.5 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | sixteen consecutive inner-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.5 | one wave's 64 FP64 matrix values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 4 | one wave's complete matrix-read phase in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 8192 | 7 | 14 | 4 | 0 |
| `tile8x16_column_major` | 8192 | 7 | 14 | 4 | 0 |
| `tile8x32_column_major` | 8192 | 7 | 14 | 4 | 0 |
| `tile16_interleaved` | 32768 | 7 | 10 | 16 | 0 |
| `tile16x8_row_major` | 65536 | 7 | 8 | 4 | 0 |
| `tile32x8_row_major` | 65536 | 7 | 8 | 4 | 0 |
| `tile8_row_major` | 65536 | 7 | 8 | 4 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 |
| 2 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 |
| 2 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 |
| 4.5 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 |
| 4.5 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 |
| 7 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 |
| 7 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 |
| 7 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 |
| 10 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 |
| 10 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 |
| 13 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 |
| 13 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 |
| 13 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 |
| 16 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 |
| 16 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 |
| 16 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 |
| 18 | `column_major` | `iiiiiiiijjjjjjjj` | 27 | 4 | 0 |
| 20 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 |
| 20 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 |
| 20 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 |
| 22 | `row_major` | `jjjjjjjjiiiiiiii` | 63 | 4 | 0 |

## GESUMMV — N=512

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced wave store |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 0 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0.5 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | sixteen consecutive inner-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.5 | one wave's 64 FP64 matrix values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 4 | one wave's complete matrix-read phase in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 16384 | 7 | 14 | 4 | 0 |
| `tile8x16_column_major` | 16384 | 7 | 14 | 4 | 0 |
| `tile8x32_column_major` | 16384 | 7 | 14 | 4 | 0 |
| `tile16_interleaved` | 65536 | 7 | 10 | 16 | 0 |
| `tile16x8_row_major` | 131072 | 7 | 8 | 4 | 0 |
| `tile32x8_row_major` | 131072 | 7 | 8 | 4 | 0 |
| `tile8_row_major` | 131072 | 7 | 8 | 4 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 |
| 2 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 |
| 2 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 |
| 4.5 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 |
| 4.5 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 |
| 7 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 |
| 7 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 |
| 7 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 |
| 10 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 |
| 10 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 |
| 13 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 |
| 13 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 |
| 13 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 |
| 16 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 |
| 16 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 |
| 16 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 |
| 19 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 |
| 19 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 |
| 19 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 |
| 21 | `column_major` | `iiiiiiiiijjjjjjjjj` | 43 | 4 | 0 |
| 22 | `row_major` | `jjjjjjjjjiiiiiiiii` | 63 | 4 | 0 |

## GESUMMV — N=1024

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced wave store |
| `wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `wave_lane_group.lane16.128B` | hypothesis | 128 | 0 | contiguous groups of 16 lanes |
| `wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `wave_lane_group.lane64.512B` | hypothesis | 512 | 0.5 | contiguous groups of 64 lanes |
| `lane_reuse.128B.window16` | hypothesis | 128 | 1 | sixteen consecutive inner-loop values used by one lane; a temporal-reuse neighborhood hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.5 | one wave's 64 FP64 matrix values in a broader locality region |
| `workgroup_step_panel.1024B` | hypothesis | 1024 | 0 | the 128-row A or B panel used by both waves at one loop step |
| `wave_phase.4096B` | hypothesis | 4096 | 4 | one wave's complete matrix-read phase in a cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 32768 | 7 | 14 | 4 | 0 |
| `tile8x16_column_major` | 32768 | 7 | 14 | 4 | 0 |
| `tile8x32_column_major` | 32768 | 7 | 14 | 4 | 0 |
| `tile16_interleaved` | 131072 | 7 | 10 | 16 | 0 |
| `tile16x8_row_major` | 262144 | 7 | 8 | 4 | 0 |
| `tile32x8_row_major` | 262144 | 7 | 8 | 4 | 0 |
| `tile8_row_major` | 262144 | 7 | 8 | 4 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 |
| 2 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 |
| 2 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 |
| 4.5 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 |
| 4.5 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 |
| 7 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 |
| 7 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 |
| 7 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 |
| 10 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 |
| 10 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 |
| 13 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 |
| 13 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 |
| 13 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 |
| 16 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 |
| 16 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 |
| 16 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 |
| 19 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 |
| 19 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 |
| 19 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 |
| 21 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 43 | 4 | 0 |
| 22 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 63 | 4 | 0 |

## MVT — N=256

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical A addresses issued by one traced row or transpose wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by a traced x1 or x2 wave store |
| `A.wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `A.wave_lane_group.lane16.128B` | hypothesis | 128 | 0 | contiguous groups of 16 lanes |
| `A.wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `A.wave_lane_group.lane64.512B` | hypothesis | 512 | 0.25 | contiguous groups of 64 lanes |
| `row_lane_stream.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive A[i,j] values used by one lane; a row-stream reuse hypothesis |
| `transpose_lane_stream.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive A[j,i] values used by one lane; a column-stream reuse hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one row or transpose wave load in a broader locality region |
| `transpose_wave_neighborhood.1024B` | hypothesis | 1024 | 0.0625 | one transpose-stream wave load in a 1024-byte cache neighborhood; an empirically calibrated hypothesis |
| `transpose_wave_neighborhood.4096B` | hypothesis | 4096 | 0.0625 | one transpose-stream wave load in a 4096-byte cache neighborhood; an empirically calibrated hypothesis |
| `transpose_wave_neighborhood.8192B` | hypothesis | 8192 | 0.0625 | one transpose-stream wave load in an 8192-byte cache neighborhood; an empirically calibrated hypothesis |
| `workgroup_step_cross.2048B` | hypothesis | 2048 | 0 | the row and column arms touched by a workgroup at one inner step; a cross-direction cache-reuse hypothesis |
| `wave_pattern_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive loads from one directional matrix stream |
| `wave_pattern_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete row or transpose stream in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16_interleaved` | 24576 | 7 | 3.75 | 8 | 0 |
| `tile8_column_major` | 36864 | 7 | 3.6875 | 2 | 0 |
| `tile8_row_major` | 36864 | 7 | 3.6875 | 2 | 0 |
| `tile8x16_column_major` | 36864 | 7 | 3.6875 | 2 | 0 |
| `tile8x32_column_major` | 36864 | 7 | 3.6875 | 2 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2.5 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 |
| 2.5 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 |
| 2.5 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 |
| 2.5 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 |
| 5 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 |
| 6 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 |
| 7 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 |
| 8 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 |
| 9 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 |
| 11 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 |
| 13 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 |
| 13 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 |
| 13 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 |
| 15 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 |
| 16 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 |
| 17 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 |
| 19 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 |
| 19 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 |
| 19 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 |
| 21 | `row_major` | `jjjjjjjjiiiiiiii` | 15.75 | 2 | 0 |
| 22 | `column_major` | `iiiiiiiijjjjjjjj` | 22.5625 | 2 | 0 |

## MVT — N=512

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical A addresses issued by one traced row or transpose wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by a traced x1 or x2 wave store |
| `A.wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `A.wave_lane_group.lane16.128B` | hypothesis | 128 | 0 | contiguous groups of 16 lanes |
| `A.wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `A.wave_lane_group.lane64.512B` | hypothesis | 512 | 0.25 | contiguous groups of 64 lanes |
| `row_lane_stream.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive A[i,j] values used by one lane; a row-stream reuse hypothesis |
| `transpose_lane_stream.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive A[j,i] values used by one lane; a column-stream reuse hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one row or transpose wave load in a broader locality region |
| `transpose_wave_neighborhood.1024B` | hypothesis | 1024 | 0.0625 | one transpose-stream wave load in a 1024-byte cache neighborhood; an empirically calibrated hypothesis |
| `transpose_wave_neighborhood.4096B` | hypothesis | 4096 | 0.0625 | one transpose-stream wave load in a 4096-byte cache neighborhood; an empirically calibrated hypothesis |
| `transpose_wave_neighborhood.8192B` | hypothesis | 8192 | 0.0625 | one transpose-stream wave load in an 8192-byte cache neighborhood; an empirically calibrated hypothesis |
| `workgroup_step_cross.2048B` | hypothesis | 2048 | 0 | the row and column arms touched by a workgroup at one inner step; a cross-direction cache-reuse hypothesis |
| `wave_pattern_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive loads from one directional matrix stream |
| `wave_pattern_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete row or transpose stream in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16_interleaved` | 49152 | 7 | 3.75 | 8 | 0 |
| `tile8_column_major` | 73728 | 7 | 3.6875 | 2 | 0 |
| `tile8_row_major` | 73728 | 7 | 3.6875 | 2 | 0 |
| `tile8x16_column_major` | 73728 | 7 | 3.6875 | 2 | 0 |
| `tile8x32_column_major` | 73728 | 7 | 3.6875 | 2 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2.5 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 |
| 2.5 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 |
| 2.5 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 |
| 2.5 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 |
| 5 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 |
| 6 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 |
| 7 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 |
| 8 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 |
| 9 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 |
| 11 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 |
| 13 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 |
| 13 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 |
| 13 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 |
| 15 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 |
| 16 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 |
| 17 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 |
| 19 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 |
| 19 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 |
| 19 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 |
| 21 | `row_major` | `jjjjjjjjjiiiiiiiii` | 15.75 | 2 | 0 |
| 22 | `column_major` | `iiiiiiiiijjjjjjjjj` | 25.5625 | 2 | 0 |

## MVT — N=1024

Workgroup: `128`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 0 | logical A addresses issued by one traced row or transpose wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by a traced x1 or x2 wave store |
| `A.wave_lane_group.lane8.64B` | hypothesis | 64 | 0 | contiguous groups of 8 lanes |
| `A.wave_lane_group.lane16.128B` | hypothesis | 128 | 0 | contiguous groups of 16 lanes |
| `A.wave_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `A.wave_lane_group.lane64.512B` | hypothesis | 512 | 0.25 | contiguous groups of 64 lanes |
| `row_lane_stream.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive A[i,j] values used by one lane; a row-stream reuse hypothesis |
| `transpose_lane_stream.128B.window16` | hypothesis | 128 | 0 | sixteen consecutive A[j,i] values used by one lane; a column-stream reuse hypothesis |
| `wave_neighborhood.512B` | hypothesis | 512 | 0.25 | one row or transpose wave load in a broader locality region |
| `transpose_wave_neighborhood.1024B` | hypothesis | 1024 | 0.0625 | one transpose-stream wave load in a 1024-byte cache neighborhood; an empirically calibrated hypothesis |
| `transpose_wave_neighborhood.4096B` | hypothesis | 4096 | 0.0625 | one transpose-stream wave load in a 4096-byte cache neighborhood; an empirically calibrated hypothesis |
| `transpose_wave_neighborhood.8192B` | hypothesis | 8192 | 0.0625 | one transpose-stream wave load in an 8192-byte cache neighborhood; an empirically calibrated hypothesis |
| `workgroup_step_cross.2048B` | hypothesis | 2048 | 0 | the row and column arms touched by a workgroup at one inner step; a cross-direction cache-reuse hypothesis |
| `wave_pattern_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive loads from one directional matrix stream |
| `wave_pattern_phase.32768B` | hypothesis | 32768 | 0 | one wave's complete row or transpose stream in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile16_interleaved` | 98304 | 7 | 3.75 | 8 | 0 |
| `tile8_column_major` | 147456 | 7 | 3.6875 | 2 | 0 |
| `tile8_row_major` | 147456 | 7 | 3.6875 | 2 | 0 |
| `tile8x16_column_major` | 147456 | 7 | 3.6875 | 2 | 0 |
| `tile8x32_column_major` | 147456 | 7 | 3.6875 | 2 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2.5 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 |
| 2.5 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 |
| 2.5 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 |
| 2.5 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 |
| 5 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 |
| 6 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 |
| 7 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 |
| 8 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 |
| 9 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 |
| 10 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 |
| 11 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 |
| 13 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 |
| 13 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 |
| 13 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 |
| 15 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 |
| 16 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 |
| 17 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 |
| 19 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 |
| 19 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 |
| 19 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 |
| 21 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 15.75 | 2 | 0 |
| 22 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 27.5625 | 2 | 0 |

## SYRK — N=256

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 1 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced C wave store |
| `A.row_j_lane_group.lane8.64B` | hypothesis | 64 | 4 | contiguous groups of 8 lanes |
| `A.row_j_lane_group.lane16.128B` | hypothesis | 128 | 0.25 | contiguous groups of 16 lanes |
| `A.row_j_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `A.row_j_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `A.paired_row_reuse.128B.window16` | hypothesis | 128 | 0.25 | eight consecutive k steps from both A row streams used by one lane; a temporal-reuse neighborhood hypothesis |
| `A.wave_neighborhood.512B` | hypothesis | 512 | 0 | one A wave load at a broader locality scale |
| `A.workgroup_k_column.256B` | hypothesis | 256 | 0 | unique A rows used across the workgroup at one k step |
| `A.wave_k_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive k steps from both A row streams in a cache-scale region |
| `A.wave_inner_phase.32768B` | hypothesis | 32768 | 1 | one wave's complete pair of A row streams in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 20992 | 6 | 1.76863 | 4 | 0 |
| `tile8x16_column_major` | 20992 | 6 | 1.76863 | 4 | 0 |
| `tile8x32_column_major` | 20992 | 6 | 1.76863 | 4 | 0 |
| `tile16_interleaved` | 69760 | 3 | 15.8226 | 16 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile8_column_major` | `iiijjj` | 1.76863 | 4 | 0 |
| 2 | `tile8x16_column_major` | `iiijjjj` | 1.76863 | 4 | 0 |
| 2 | `tile8x32_column_major` | `iiijjjjj` | 1.76863 | 4 | 0 |
| 6.5 | `tile16_column_major` | `iiiijjjj` | 2.76863 | 4 | 0 |
| 6.5 | `tile16x32_column_major` | `iiiijjjjj` | 2.76863 | 4 | 0 |
| 6.5 | `tile16x8_column_major` | `iiiijjj` | 2.76863 | 4 | 0 |
| 6.5 | `tile32_column_major` | `iiiiijjjjj` | 2.76863 | 4 | 0 |
| 6.5 | `tile32x16_column_major` | `iiiiijjjj` | 2.76863 | 4 | 0 |
| 6.5 | `tile32x8_column_major` | `iiiiijjj` | 2.76863 | 4 | 0 |
| 10 | `column_major` | `iiiiiiiijjjjjjjj` | 9.76863 | 4 | 0 |
| 11.5 | `tile16_interleaved` | `jijijiji` | 15.8226 | 16 | 0 |
| 11.5 | `tile32_interleaved` | `jijijijiji` | 15.8226 | 20 | 0 |
| 14 | `tile16x8_row_major` | `jjjiiii` | 35.7484 | 4 | 0 |
| 14 | `tile32x8_row_major` | `jjjiiiii` | 35.7484 | 4 | 0 |
| 14 | `tile8_row_major` | `jjjiii` | 35.7484 | 4 | 0 |
| 19 | `row_major` | `jjjjjjjjiiiiiiii` | 37.7562 | 4 | 0 |
| 19 | `tile16_row_major` | `jjjjiiii` | 37.7562 | 4 | 0 |
| 19 | `tile16x32_row_major` | `jjjjjiiii` | 37.7562 | 4 | 0 |
| 19 | `tile32_row_major` | `jjjjjiiiii` | 37.7562 | 4 | 0 |
| 19 | `tile32x16_row_major` | `jjjjiiiii` | 37.7562 | 4 | 0 |
| 19 | `tile8x16_row_major` | `jjjjiii` | 37.7562 | 4 | 0 |
| 19 | `tile8x32_row_major` | `jjjjjiii` | 37.7562 | 4 | 0 |

## SYRK — N=512

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 1 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced C wave store |
| `A.row_j_lane_group.lane8.64B` | hypothesis | 64 | 4 | contiguous groups of 8 lanes |
| `A.row_j_lane_group.lane16.128B` | hypothesis | 128 | 0.25 | contiguous groups of 16 lanes |
| `A.row_j_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `A.row_j_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `A.paired_row_reuse.128B.window16` | hypothesis | 128 | 0.25 | eight consecutive k steps from both A row streams used by one lane; a temporal-reuse neighborhood hypothesis |
| `A.wave_neighborhood.512B` | hypothesis | 512 | 0 | one A wave load at a broader locality scale |
| `A.workgroup_k_column.256B` | hypothesis | 256 | 0 | unique A rows used across the workgroup at one k step |
| `A.wave_k_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive k steps from both A row streams in a cache-scale region |
| `A.wave_inner_phase.32768B` | hypothesis | 32768 | 1 | one wave's complete pair of A row streams in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 41472 | 6 | 1.75935 | 4 | 0 |
| `tile8x16_column_major` | 41472 | 6 | 1.75935 | 4 | 0 |
| `tile8x32_column_major` | 41472 | 6 | 1.75935 | 4 | 0 |
| `tile16_interleaved` | 139392 | 3 | 15.83 | 16 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile8_column_major` | `iiijjj` | 1.75935 | 4 | 0 |
| 2 | `tile8x16_column_major` | `iiijjjj` | 1.75935 | 4 | 0 |
| 2 | `tile8x32_column_major` | `iiijjjjj` | 1.75935 | 4 | 0 |
| 6.5 | `tile16_column_major` | `iiiijjjj` | 2.75935 | 4 | 0 |
| 6.5 | `tile16x32_column_major` | `iiiijjjjj` | 2.75935 | 4 | 0 |
| 6.5 | `tile16x8_column_major` | `iiiijjj` | 2.75935 | 4 | 0 |
| 6.5 | `tile32_column_major` | `iiiiijjjjj` | 2.75935 | 4 | 0 |
| 6.5 | `tile32x16_column_major` | `iiiiijjjj` | 2.75935 | 4 | 0 |
| 6.5 | `tile32x8_column_major` | `iiiiijjj` | 2.75935 | 4 | 0 |
| 10.5 | `tile16_interleaved` | `jijijiji` | 15.83 | 16 | 0 |
| 10.5 | `tile32_interleaved` | `jijijijiji` | 15.83 | 20 | 0 |
| 12 | `column_major` | `iiiiiiiiijjjjjjjjj` | 17.7593 | 4 | 0 |
| 14 | `tile16x8_row_major` | `jjjiiii` | 35.7663 | 4 | 0 |
| 14 | `tile32x8_row_major` | `jjjiiiii` | 35.7663 | 4 | 0 |
| 14 | `tile8_row_major` | `jjjiii` | 35.7663 | 4 | 0 |
| 19 | `row_major` | `jjjjjjjjjiiiiiiiii` | 37.7741 | 4 | 0 |
| 19 | `tile16_row_major` | `jjjjiiii` | 37.7741 | 4 | 0 |
| 19 | `tile16x32_row_major` | `jjjjjiiii` | 37.7741 | 4 | 0 |
| 19 | `tile32_row_major` | `jjjjjiiiii` | 37.7741 | 4 | 0 |
| 19 | `tile32x16_row_major` | `jjjjiiiii` | 37.7741 | 4 | 0 |
| 19 | `tile8x16_row_major` | `jjjjiii` | 37.7741 | 4 | 0 |
| 19 | `tile8x32_row_major` | `jjjjjiii` | 37.7741 | 4 | 0 |

## SYRK — N=1024

Workgroup: `[32, 32, 1]`.

### Objective model

`grounded` scopes come from traced memory instructions. `hypothesis` scopes encode proposed reuse or cache-locality neighborhoods.

| Objective | Provenance | Region B | Tau | Meaning |
| --- | --- | --- | --- | --- |
| `wave_load.64B` | grounded | 64 | 1 | logical addresses issued by one traced wave load |
| `output_store.64B` | grounded | 64 | 0 | logical addresses issued by the traced C wave store |
| `A.row_j_lane_group.lane8.64B` | hypothesis | 64 | 4 | contiguous groups of 8 lanes |
| `A.row_j_lane_group.lane16.128B` | hypothesis | 128 | 0.25 | contiguous groups of 16 lanes |
| `A.row_j_lane_group.lane32.256B` | hypothesis | 256 | 0 | contiguous groups of 32 lanes |
| `A.row_j_lane_group.lane64.512B` | hypothesis | 512 | 0 | contiguous groups of 64 lanes |
| `A.paired_row_reuse.128B.window16` | hypothesis | 128 | 0.25 | eight consecutive k steps from both A row streams used by one lane; a temporal-reuse neighborhood hypothesis |
| `A.wave_neighborhood.512B` | hypothesis | 512 | 0 | one A wave load at a broader locality scale |
| `A.workgroup_k_column.256B` | hypothesis | 256 | 0 | unique A rows used across the workgroup at one k step |
| `A.wave_k_window.4096B` | hypothesis | 4096 | 0 | sixteen consecutive k steps from both A row streams in a cache-scale region |
| `A.wave_inner_phase.32768B` | hypothesis | 32768 | 1 | one wave's complete pair of A row streams in a broad cache-scale region |

### Score Pareto frontier

This is the exact non-dominated set over the notes-aligned locality vector plus separate codegen run and XOR costs. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| `tile8_column_major` | 82432 | 6 | 1.75468 | 4 | 0 |
| `tile8x16_column_major` | 82432 | 6 | 1.75468 | 4 | 0 |
| `tile8x32_column_major` | 82432 | 6 | 1.75468 | 4 | 0 |
| `tile16_interleaved` | 278656 | 3 | 15.8338 | 16 | 0 |

### Layout ranks

| Score rank | Layout | Word (low→high) | Score | Runs | XORs |
| --- | --- | --- | --- | --- | --- |
| 2 | `tile8_column_major` | `iiijjj` | 1.75468 | 4 | 0 |
| 2 | `tile8x16_column_major` | `iiijjjj` | 1.75468 | 4 | 0 |
| 2 | `tile8x32_column_major` | `iiijjjjj` | 1.75468 | 4 | 0 |
| 6.5 | `tile16_column_major` | `iiiijjjj` | 2.75468 | 4 | 0 |
| 6.5 | `tile16x32_column_major` | `iiiijjjjj` | 2.75468 | 4 | 0 |
| 6.5 | `tile16x8_column_major` | `iiiijjj` | 2.75468 | 4 | 0 |
| 6.5 | `tile32_column_major` | `iiiiijjjjj` | 2.75468 | 4 | 0 |
| 6.5 | `tile32x16_column_major` | `iiiiijjjj` | 2.75468 | 4 | 0 |
| 6.5 | `tile32x8_column_major` | `iiiiijjj` | 2.75468 | 4 | 0 |
| 10.5 | `tile16_interleaved` | `jijijiji` | 15.8338 | 16 | 0 |
| 10.5 | `tile32_interleaved` | `jijijijiji` | 15.8338 | 20 | 0 |
| 12 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 33.7547 | 4 | 0 |
| 14 | `tile16x8_row_major` | `jjjiiii` | 35.7753 | 4 | 0 |
| 14 | `tile32x8_row_major` | `jjjiiiii` | 35.7753 | 4 | 0 |
| 14 | `tile8_row_major` | `jjjiii` | 35.7753 | 4 | 0 |
| 19 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 37.7831 | 4 | 0 |
| 19 | `tile16_row_major` | `jjjjiiii` | 37.7831 | 4 | 0 |
| 19 | `tile16x32_row_major` | `jjjjjiiii` | 37.7831 | 4 | 0 |
| 19 | `tile32_row_major` | `jjjjjiiiii` | 37.7831 | 4 | 0 |
| 19 | `tile32x16_row_major` | `jjjjiiiii` | 37.7831 | 4 | 0 |
| 19 | `tile8x16_row_major` | `jjjjiii` | 37.7831 | 4 | 0 |
| 19 | `tile8x32_row_major` | `jjjjjiii` | 37.7831 | 4 | 0 |
