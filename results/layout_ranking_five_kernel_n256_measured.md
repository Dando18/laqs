# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runs and XORs are separate address-code generation costs. They are included in the Pareto frontier but are not folded into the scalar locality score or score rank.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| ATAX | 256 | 22 | 4 | 0.545 | 1.318 |
| GEMM | 256 | 22 | 4 | 0.864 | 0.250 |
| GESUMMV | 256 | 22 | 7 | 0.727 | 0.455 |
| MVT | 256 | 22 | 5 | 0.818 | 0.477 |
| SYRK | 256 | 22 | 4 | 0.636 | 0.500 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 | 0.060094 | 0.060048 | 0.000178 | 0.059827–0.060253 | 4.36 | +1.0 |
| 2 | 3 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 | 0.061453 | 0.061683 | 0.000344 | 0.061347–0.062160 | 4.27 | -1.0 |
| 2 | 2 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 | 0.060281 | 0.060235 | 0.000300 | 0.059894–0.060694 | 4.35 | +0.0 |
| 4.5 | 7 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 | 0.065080 | 0.065041 | 0.000151 | 0.064814–0.065201 | 4.03 | -2.5 |
| 4.5 | 13 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 | 0.071561 | 0.071550 | 0.000188 | 0.071307–0.071827 | 3.66 | -8.5 |
| 7 | 4 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 | 0.062107 | 0.062096 | 0.000216 | 0.061827–0.062347 | 4.22 | +3.0 |
| 7 | 5 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 | 0.062761 | 0.062787 | 0.000127 | 0.062627–0.063001 | 4.18 | +2.0 |
| 7 | 6 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 | 0.064413 | 0.064299 | 0.000163 | 0.064040–0.064440 | 4.07 | +1.0 |
| 10 | 12 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 | 0.071201 | 0.071192 | 0.000241 | 0.070854–0.071587 | 3.68 | -2.0 |
| 10 | 8 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 | 0.070534 | 0.070539 | 0.000229 | 0.070254–0.070934 | 3.72 | +2.0 |
| 10 | 14 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 | 0.071667 | 0.071569 | 0.000229 | 0.071121–0.071747 | 3.66 | -4.0 |
| 13 | 9 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 | 0.070600 | 0.070467 | 0.000374 | 0.069894–0.070987 | 3.71 | +4.0 |
| 13 | 18 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 | 0.072720 | 0.072584 | 0.000370 | 0.072054–0.073027 | 3.60 | -5.0 |
| 13 | 10 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 | 0.071187 | 0.070680 | 0.000876 | 0.069454–0.071587 | 3.68 | +3.0 |
| 15 | 21 | `column_major` | `iiiiiiiijjjjjjjj` | 33 | 2 | 0 | 0.084480 | 0.084475 | 0.000516 | 0.083614–0.085001 | 3.10 | -6.0 |
| 17 | 17 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 | 0.072467 | 0.072347 | 0.000296 | 0.071987–0.072680 | 3.62 | +0.0 |
| 17 | 20 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 | 0.073294 | 0.073174 | 0.000380 | 0.072427–0.073467 | 3.58 | -3.0 |
| 17 | 15 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 | 0.071707 | 0.071422 | 0.000361 | 0.070960–0.071734 | 3.66 | +2.0 |
| 20 | 11 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 | 0.071200 | 0.070787 | 0.000840 | 0.069693–0.071827 | 3.68 | +9.0 |
| 20 | 16 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 | 0.072120 | 0.071952 | 0.000613 | 0.071040–0.072574 | 3.63 | +4.0 |
| 20 | 19 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 | 0.073227 | 0.073056 | 0.000469 | 0.072334–0.073667 | 3.58 | +1.0 |
| 22 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 65.75 | 2 | 0 | 0.086108 | 0.086417 | 0.001080 | 0.085094–0.087827 | 3.04 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 10/22 | 0.455 | 1.864 | 9.000 |
| `peak-normalized-excess` | 10/22 | 0.455 | 1.364 | 9.000 |
| `weighted-normalized-excess` (selected) | 12/22 | 0.545 | 1.318 | 6.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.5 | 7 | `tile16x32_row_major` | `jjjjjiiii` | 8.29503 | 6 | 0 | 0.065894 | 0.065894 | 0.000037 | 0.065854–0.065947 | 509.22 | -3.5 |
| 3.5 | 5.5 | `tile16x8_row_major` | `jjjiiii` | 8.29503 | 6 | 0 | 0.065854 | 0.065867 | 0.000066 | 0.065787–0.065987 | 509.53 | -2.0 |
| 3.5 | 1 | `tile32_row_major` | `jjjjjiiiii` | 8.29503 | 6 | 0 | 0.065614 | 0.065630 | 0.000047 | 0.065587–0.065720 | 511.39 | +2.5 |
| 3.5 | 4 | `tile32x8_row_major` | `jjjiiiii` | 8.29503 | 6 | 0 | 0.065827 | 0.065979 | 0.000338 | 0.065774–0.066653 | 509.74 | -0.5 |
| 3.5 | 10 | `tile8_row_major` | `jjjiii` | 8.29503 | 6 | 0 | 0.065934 | 0.065920 | 0.000043 | 0.065867–0.065987 | 508.91 | -6.5 |
| 3.5 | 2 | `tile8x32_row_major` | `jjjjjiii` | 8.29503 | 6 | 0 | 0.065734 | 0.065769 | 0.000083 | 0.065681–0.065880 | 510.46 | +1.5 |
| 8.5 | 9 | `row_major` | `jjjjjjjjiiiiiiii` | 8.79503 | 6 | 0 | 0.065920 | 0.065910 | 0.000013 | 0.065894–0.065920 | 509.01 | -0.5 |
| 8.5 | 3 | `tile16_row_major` | `jjjjiiii` | 8.79503 | 6 | 0 | 0.065787 | 0.065758 | 0.000060 | 0.065667–0.065827 | 510.04 | +5.5 |
| 8.5 | 8 | `tile32x16_row_major` | `jjjjiiiii` | 8.79503 | 6 | 0 | 0.065907 | 0.065878 | 0.000129 | 0.065694–0.066054 | 509.12 | +0.5 |
| 8.5 | 5.5 | `tile8x16_row_major` | `jjjjiii` | 8.79503 | 6 | 0 | 0.065854 | 0.065998 | 0.000267 | 0.065814–0.066520 | 509.53 | +3.0 |
| 11.5 | 11 | `tile16_interleaved` | `jijijiji` | 15.6801 | 24 | 0 | 0.085947 | 0.085947 | 0.000125 | 0.085734–0.086120 | 390.41 | +0.5 |
| 11.5 | 15 | `tile32_interleaved` | `jijijijiji` | 15.6801 | 30 | 0 | 0.102734 | 0.102800 | 0.000231 | 0.102613–0.103253 | 326.62 | -3.5 |
| 14 | 13 | `tile8_column_major` | `iiijjj` | 55.8354 | 6 | 0 | 0.097014 | 0.097038 | 0.000039 | 0.097000–0.097107 | 345.87 | +1.0 |
| 14 | 12 | `tile8x16_column_major` | `iiijjjj` | 55.8354 | 6 | 0 | 0.096881 | 0.096945 | 0.000127 | 0.096787–0.097121 | 346.35 | +2.0 |
| 14 | 14 | `tile8x32_column_major` | `iiijjjjj` | 55.8354 | 6 | 0 | 0.097067 | 0.097150 | 0.000204 | 0.096948–0.097534 | 345.68 | +0.0 |
| 17 | 18 | `tile16_column_major` | `iiiijjjj` | 77.3354 | 6 | 0 | 0.159668 | 0.159673 | 0.000046 | 0.159601–0.159735 | 210.15 | -1.0 |
| 17 | 16 | `tile16x32_column_major` | `iiiijjjjj` | 77.3354 | 6 | 0 | 0.159588 | 0.159604 | 0.000037 | 0.159562–0.159655 | 210.26 | +1.0 |
| 17 | 17 | `tile16x8_column_major` | `iiiijjj` | 77.3354 | 6 | 0 | 0.159627 | 0.159681 | 0.000159 | 0.159494–0.159974 | 210.20 | +0.0 |
| 20 | 20 | `tile32_column_major` | `iiiiijjjjj` | 81.3354 | 6 | 0 | 0.159814 | 0.159822 | 0.000034 | 0.159774–0.159867 | 209.96 | +0.0 |
| 20 | 19 | `tile32x16_column_major` | `iiiiijjjj` | 81.3354 | 6 | 0 | 0.159787 | 0.159742 | 0.000066 | 0.159628–0.159801 | 209.99 | +1.0 |
| 20 | 21 | `tile32x8_column_major` | `iiiiijjj` | 81.3354 | 6 | 0 | 0.159867 | 0.159872 | 0.000056 | 0.159814–0.159974 | 209.89 | -1.0 |
| 22 | 22 | `column_major` | `iiiiiiiijjjjjjjj` | 89.3354 | 6 | 0 | 0.169321 | 0.169337 | 0.000098 | 0.169201–0.169508 | 198.17 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 17/22 | 0.773 | 0.341 | 3.500 |
| `peak-normalized-excess` | 9/22 | 0.409 | 2.841 | 13.500 |
| `weighted-normalized-excess` (selected) | 19/22 | 0.864 | 0.250 | 3.500 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 3 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 | 0.033787 | 0.033824 | 0.000102 | 0.033733–0.034014 | 7.78 | -1.0 |
| 2 | 1 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 | 0.032560 | 0.032525 | 0.000285 | 0.032080–0.032894 | 8.07 | +1.0 |
| 2 | 2 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 | 0.033587 | 0.033643 | 0.000167 | 0.033467–0.033907 | 7.83 | +0.0 |
| 4.5 | 4 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 | 0.034293 | 0.034320 | 0.000110 | 0.034173–0.034454 | 7.67 | +0.5 |
| 4.5 | 8 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 | 0.038547 | 0.038736 | 0.000280 | 0.038480–0.039147 | 6.82 | -3.5 |
| 7 | 5 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 | 0.034787 | 0.034787 | 0.000117 | 0.034587–0.034947 | 7.56 | +2.0 |
| 7 | 7 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 | 0.035347 | 0.035389 | 0.000316 | 0.035107–0.035987 | 7.44 | +0.0 |
| 7 | 6 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 | 0.035294 | 0.035358 | 0.000161 | 0.035214–0.035667 | 7.45 | +1.0 |
| 10 | 10 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 | 0.039560 | 0.039659 | 0.000161 | 0.039480–0.039853 | 6.65 | +0.0 |
| 10 | 9 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 | 0.039294 | 0.039288 | 0.000074 | 0.039187–0.039387 | 6.69 | +1.0 |
| 10 | 16 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 | 0.040387 | 0.040296 | 0.000171 | 0.040014–0.040454 | 6.51 | -6.0 |
| 13 | 14 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 | 0.039880 | 0.039776 | 0.000209 | 0.039373–0.039947 | 6.59 | -1.0 |
| 13 | 11 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 | 0.039640 | 0.039736 | 0.000200 | 0.039520–0.040054 | 6.63 | +2.0 |
| 13 | 12 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 | 0.039667 | 0.039803 | 0.000217 | 0.039587–0.040080 | 6.63 | +1.0 |
| 16 | 15 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 | 0.040040 | 0.040019 | 0.000152 | 0.039787–0.040240 | 6.57 | +1.0 |
| 16 | 13 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 | 0.039800 | 0.039752 | 0.000374 | 0.039347–0.040347 | 6.61 | +3.0 |
| 16 | 17 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 | 0.040440 | 0.040451 | 0.000197 | 0.040267–0.040814 | 6.50 | -1.0 |
| 18 | 18 | `column_major` | `iiiiiiiijjjjjjjj` | 27 | 4 | 0 | 0.041240 | 0.041320 | 0.000266 | 0.041027–0.041787 | 6.38 | +0.0 |
| 20 | 20 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 | 0.056600 | 0.056643 | 0.000200 | 0.056400–0.056894 | 4.65 | +0.0 |
| 20 | 21 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 | 0.056653 | 0.056699 | 0.000647 | 0.055867–0.057600 | 4.64 | -1.0 |
| 20 | 19 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 | 0.053894 | 0.054150 | 0.000562 | 0.053601–0.055068 | 4.88 | +1.0 |
| 22 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 63 | 4 | 0 | 0.063427 | 0.063123 | 0.000865 | 0.061827–0.064347 | 4.15 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 5/22 | 0.227 | 3.818 | 10.000 |
| `peak-normalized-excess` | 10/22 | 0.455 | 1.068 | 4.500 |
| `weighted-normalized-excess` (selected) | 16/22 | 0.727 | 0.455 | 3.500 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5 | 4 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 | 0.040253 | 0.040264 | 0.000108 | 0.040093–0.040427 | 6.55 | -1.5 |
| 2.5 | 1 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 | 0.039894 | 0.039856 | 0.000075 | 0.039707–0.039907 | 6.61 | +1.5 |
| 2.5 | 3 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 | 0.040054 | 0.040219 | 0.000257 | 0.040027–0.040707 | 6.58 | -0.5 |
| 2.5 | 2 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 | 0.040053 | 0.040141 | 0.000129 | 0.040014–0.040347 | 6.58 | +0.5 |
| 5 | 5 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 | 0.042427 | 0.042368 | 0.000195 | 0.042014–0.042547 | 6.21 | +0.0 |
| 6 | 11 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 | 0.048013 | 0.047987 | 0.000658 | 0.046867–0.048854 | 5.49 | -5.0 |
| 7 | 7 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 | 0.043347 | 0.043531 | 0.000291 | 0.043254–0.044014 | 6.08 | +0.0 |
| 8 | 6 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 | 0.042907 | 0.043038 | 0.000262 | 0.042827–0.043520 | 6.15 | +2.0 |
| 9 | 9 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 | 0.044987 | 0.044846 | 0.000254 | 0.044494–0.045107 | 5.86 | +0.0 |
| 10 | 10 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 | 0.045547 | 0.045571 | 0.000215 | 0.045241–0.045854 | 5.79 | +0.0 |
| 11 | 8 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 | 0.043573 | 0.043566 | 0.000172 | 0.043347–0.043787 | 6.05 | +3.0 |
| 13 | 14 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 | 0.050667 | 0.050603 | 0.000357 | 0.049934–0.051000 | 5.20 | -1.0 |
| 13 | 12 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 | 0.049147 | 0.049147 | 0.000311 | 0.048827–0.049694 | 5.37 | +1.0 |
| 13 | 13 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 | 0.049507 | 0.049435 | 0.000315 | 0.048840–0.049774 | 5.33 | +0.0 |
| 15 | 15 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 | 0.052294 | 0.052358 | 0.000310 | 0.052027–0.052934 | 5.04 | +0.0 |
| 16 | 16 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 | 0.054507 | 0.054537 | 0.000382 | 0.054000–0.054974 | 4.84 | +0.0 |
| 17 | 17 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 | 0.054826 | 0.054864 | 0.000520 | 0.054360–0.055800 | 4.81 | +0.0 |
| 19 | 21 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 | 0.055627 | 0.055598 | 0.000178 | 0.055387–0.055880 | 4.74 | -2.0 |
| 19 | 19 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 | 0.054920 | 0.054958 | 0.000120 | 0.054774–0.055120 | 4.80 | +0.0 |
| 19 | 18 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 | 0.054854 | 0.054816 | 0.000338 | 0.054373–0.055347 | 4.81 | +1.0 |
| 21 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 15.75 | 2 | 0 | 0.055654 | 0.055966 | 0.000516 | 0.055400–0.056774 | 4.74 | -1.0 |
| 22 | 20 | `column_major` | `iiiiiiiijjjjjjjj` | 22.5625 | 2 | 0 | 0.055373 | 0.055312 | 0.000324 | 0.054800–0.055640 | 4.76 | +2.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 18/22 | 0.818 | 0.477 | 5.000 |
| `peak-normalized-excess` | 8/22 | 0.364 | 1.318 | 6.500 |
| `weighted-normalized-excess` (selected) | 18/22 | 0.818 | 0.477 | 5.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | `tile8_column_major` | `iiijjj` | 1.76863 | 4 | 0 | 0.065920 | 0.065925 | 0.000040 | 0.065880–0.066000 | 509.02 | +0.0 |
| 2 | 3 | `tile8x16_column_major` | `iiijjjj` | 1.76863 | 4 | 0 | 0.065960 | 0.065971 | 0.000041 | 0.065907–0.066027 | 508.71 | -1.0 |
| 2 | 1 | `tile8x32_column_major` | `iiijjjjj` | 1.76863 | 4 | 0 | 0.065840 | 0.065875 | 0.000099 | 0.065801–0.066067 | 509.63 | +1.0 |
| 6.5 | 8 | `tile16_column_major` | `iiiijjjj` | 2.76863 | 4 | 0 | 0.066387 | 0.066371 | 0.000049 | 0.066280–0.066427 | 505.44 | -1.5 |
| 6.5 | 4 | `tile16x32_column_major` | `iiiijjjjj` | 2.76863 | 4 | 0 | 0.066120 | 0.066115 | 0.000020 | 0.066094–0.066147 | 507.48 | +2.5 |
| 6.5 | 6 | `tile16x8_column_major` | `iiiijjj` | 2.76863 | 4 | 0 | 0.066267 | 0.066267 | 0.000030 | 0.066227–0.066307 | 506.35 | +0.5 |
| 6.5 | 7 | `tile32_column_major` | `iiiiijjjjj` | 2.76863 | 4 | 0 | 0.066360 | 0.066405 | 0.000102 | 0.066294–0.066587 | 505.64 | -0.5 |
| 6.5 | 5 | `tile32x16_column_major` | `iiiiijjjj` | 2.76863 | 4 | 0 | 0.066241 | 0.066246 | 0.000047 | 0.066174–0.066321 | 506.55 | +1.5 |
| 6.5 | 9 | `tile32x8_column_major` | `iiiiijjj` | 2.76863 | 4 | 0 | 0.066400 | 0.066414 | 0.000030 | 0.066387–0.066467 | 505.34 | -2.5 |
| 10 | 10 | `column_major` | `iiiiiiiijjjjjjjj` | 9.76863 | 4 | 0 | 0.067267 | 0.067254 | 0.000043 | 0.067200–0.067320 | 498.82 | +0.0 |
| 11.5 | 11 | `tile16_interleaved` | `jijijiji` | 15.8226 | 16 | 0 | 0.069867 | 0.069899 | 0.000068 | 0.069827–0.069987 | 480.26 | +0.5 |
| 11.5 | 12 | `tile32_interleaved` | `jijijijiji` | 15.8226 | 20 | 0 | 0.070107 | 0.070213 | 0.000137 | 0.070093–0.070400 | 478.62 | -0.5 |
| 14 | 13 | `tile16x8_row_major` | `jjjiiii` | 35.7484 | 4 | 0 | 0.096654 | 0.096652 | 0.000023 | 0.096614–0.096681 | 347.16 | +1.0 |
| 14 | 14.5 | `tile32x8_row_major` | `jjjiiiii` | 35.7484 | 4 | 0 | 0.096787 | 0.096800 | 0.000028 | 0.096774–0.096854 | 346.68 | -0.5 |
| 14 | 14.5 | `tile8_row_major` | `jjjiii` | 35.7484 | 4 | 0 | 0.096787 | 0.096779 | 0.000034 | 0.096734–0.096827 | 346.68 | -0.5 |
| 19 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 37.7562 | 4 | 0 | 0.160468 | 0.160497 | 0.000066 | 0.160455–0.160628 | 209.10 | -3.0 |
| 19 | 16 | `tile16_row_major` | `jjjjiiii` | 37.7562 | 4 | 0 | 0.158774 | 0.158804 | 0.000094 | 0.158721–0.158988 | 211.33 | +3.0 |
| 19 | 17 | `tile16x32_row_major` | `jjjjjiiii` | 37.7562 | 4 | 0 | 0.158975 | 0.158953 | 0.000058 | 0.158868–0.159028 | 211.07 | +2.0 |
| 19 | 19 | `tile32_row_major` | `jjjjjiiiii` | 37.7562 | 4 | 0 | 0.159094 | 0.159089 | 0.000018 | 0.159054–0.159108 | 210.91 | +0.0 |
| 19 | 21 | `tile32x16_row_major` | `jjjjiiiii` | 37.7562 | 4 | 0 | 0.159121 | 0.159134 | 0.000071 | 0.159068–0.159267 | 210.87 | -2.0 |
| 19 | 18 | `tile8x16_row_major` | `jjjjiii` | 37.7562 | 4 | 0 | 0.159068 | 0.159105 | 0.000077 | 0.159041–0.159254 | 210.94 | +1.0 |
| 19 | 20 | `tile8x32_row_major` | `jjjjjiii` | 37.7562 | 4 | 0 | 0.159095 | 0.159124 | 0.000108 | 0.159028–0.159335 | 210.91 | -1.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 14/22 | 0.636 | 0.500 | 3.000 |
| `peak-normalized-excess` | 4/22 | 0.182 | 3.455 | 10.500 |
| `weighted-normalized-excess` (selected) | 14/22 | 0.636 | 0.500 | 3.000 |
