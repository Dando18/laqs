# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runs and XORs are separate address-code generation costs. They are included in the Pareto frontier but are not folded into the scalar locality score or score rank.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

Runtime samples were reused from `/g/g16/dnicho/record-replay/relay/results/layout_ranking_five_kernel_baseline.json`; objective scores and all rank metrics were recomputed for this report.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| ATAX | 256 | 22 | 4 | 0.500 | 1.773 |
| GEMM | 256 | 22 | 4 | 0.864 | 0.273 |
| GESUMMV | 256 | 22 | 7 | 0.773 | 0.341 |
| MVT | 256 | 22 | 5 | 0.909 | 0.364 |
| SYRK | 256 | 22 | 4 | 0.545 | 0.500 |

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
| 2 | 1 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 | 0.059667 | 0.059709 | 0.000190 | 0.059413–0.059947 | 4.39 | +1.0 |
| 2 | 5 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 | 0.063600 | 0.063557 | 0.000170 | 0.063293–0.063760 | 4.12 | -3.0 |
| 2 | 2 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 | 0.060227 | 0.060059 | 0.000270 | 0.059560–0.060280 | 4.35 | +0.0 |
| 4.5 | 7 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 | 0.065080 | 0.065091 | 0.000129 | 0.064934–0.065280 | 4.03 | -2.5 |
| 4.5 | 13 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 | 0.071240 | 0.071270 | 0.000139 | 0.071134–0.071534 | 3.68 | -8.5 |
| 7 | 3 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 | 0.062080 | 0.062067 | 0.000154 | 0.061800–0.062267 | 4.22 | +4.0 |
| 7 | 4 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 | 0.062814 | 0.062888 | 0.000192 | 0.062654–0.063214 | 4.17 | +3.0 |
| 7 | 6 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 | 0.064600 | 0.064731 | 0.000341 | 0.064307–0.065307 | 4.06 | +1.0 |
| 10 | 14 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 | 0.071307 | 0.071305 | 0.000401 | 0.070641–0.071761 | 3.68 | -4.0 |
| 10 | 9 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 | 0.070680 | 0.070659 | 0.000204 | 0.070294–0.070907 | 3.71 | +1.0 |
| 10 | 12 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 | 0.071094 | 0.071072 | 0.000281 | 0.070600–0.071454 | 3.69 | -2.0 |
| 13 | 16 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 | 0.072280 | 0.072352 | 0.000115 | 0.072240–0.072534 | 3.63 | -3.0 |
| 13 | 8 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 | 0.068947 | 0.068899 | 0.000202 | 0.068534–0.069094 | 3.80 | +5.0 |
| 13 | 11 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 | 0.071054 | 0.070875 | 0.000392 | 0.070347–0.071387 | 3.69 | +2.0 |
| 15 | 21 | `column_major` | `iiiiiiiijjjjjjjj` | 33 | 2 | 0 | 0.084174 | 0.083785 | 0.000750 | 0.082600–0.084481 | 3.11 | -6.0 |
| 17 | 18 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 | 0.072440 | 0.072603 | 0.000359 | 0.072227–0.073040 | 3.62 | -1.0 |
| 17 | 19 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 | 0.073280 | 0.073168 | 0.000493 | 0.072240–0.073654 | 3.58 | -2.0 |
| 17 | 15 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 | 0.071387 | 0.071352 | 0.000224 | 0.071081–0.071600 | 3.67 | +2.0 |
| 20 | 10 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 | 0.070867 | 0.070848 | 0.000410 | 0.070160–0.071307 | 3.70 | +10.0 |
| 20 | 17 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 | 0.072387 | 0.072352 | 0.000340 | 0.071840–0.072813 | 3.62 | +3.0 |
| 20 | 20 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 | 0.074747 | 0.074814 | 0.000268 | 0.074494–0.075227 | 3.51 | +0.0 |
| 22 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 65.75 | 2 | 0 | 0.086707 | 0.086731 | 0.000647 | 0.086067–0.087894 | 3.02 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 9/22 | 0.409 | 2.273 | 9.000 |
| `peak-normalized-excess` | 10/22 | 0.455 | 1.818 | 9.000 |
| `weighted-normalized-excess` (selected) | 11/22 | 0.500 | 1.773 | 6.000 |

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
| 3.5 | 3 | `tile16x32_row_major` | `jjjjjiiii` | 8.29503 | 6 | 0 | 0.065800 | 0.065782 | 0.000050 | 0.065707–0.065854 | 509.94 | +0.5 |
| 3.5 | 3 | `tile16x8_row_major` | `jjjiiii` | 8.29503 | 6 | 0 | 0.065800 | 0.065798 | 0.000052 | 0.065720–0.065880 | 509.94 | +0.5 |
| 3.5 | 3 | `tile32_row_major` | `jjjjjiiiii` | 8.29503 | 6 | 0 | 0.065800 | 0.065859 | 0.000122 | 0.065693–0.066014 | 509.94 | +0.5 |
| 3.5 | 1 | `tile32x8_row_major` | `jjjiiiii` | 8.29503 | 6 | 0 | 0.065720 | 0.065726 | 0.000039 | 0.065693–0.065800 | 510.56 | +2.5 |
| 3.5 | 10 | `tile8_row_major` | `jjjiii` | 8.29503 | 6 | 0 | 0.066174 | 0.066144 | 0.000136 | 0.065880–0.066254 | 507.07 | -6.5 |
| 3.5 | 5 | `tile8x32_row_major` | `jjjjjiii` | 8.29503 | 6 | 0 | 0.065880 | 0.065944 | 0.000117 | 0.065840–0.066160 | 509.32 | -1.5 |
| 8.5 | 9 | `row_major` | `jjjjjjjjiiiiiiii` | 8.79503 | 6 | 0 | 0.066040 | 0.066027 | 0.000066 | 0.065933–0.066120 | 508.09 | -0.5 |
| 8.5 | 7 | `tile16_row_major` | `jjjjiiii` | 8.79503 | 6 | 0 | 0.065920 | 0.066003 | 0.000113 | 0.065907–0.066160 | 509.01 | +1.5 |
| 8.5 | 6 | `tile32x16_row_major` | `jjjjiiiii` | 8.79503 | 6 | 0 | 0.065907 | 0.065920 | 0.000034 | 0.065880–0.065960 | 509.12 | +2.5 |
| 8.5 | 8 | `tile8x16_row_major` | `jjjjiii` | 8.79503 | 6 | 0 | 0.066001 | 0.066052 | 0.000084 | 0.065961–0.066188 | 508.39 | +0.5 |
| 11.5 | 11 | `tile16_interleaved` | `jijijiji` | 15.6801 | 24 | 0 | 0.085786 | 0.085826 | 0.000076 | 0.085760–0.085973 | 391.14 | +0.5 |
| 11.5 | 15 | `tile32_interleaved` | `jijijijiji` | 15.6801 | 30 | 0 | 0.102694 | 0.102707 | 0.000036 | 0.102667–0.102774 | 326.74 | -3.5 |
| 14 | 13.5 | `tile8_column_major` | `iiijjj` | 55.8354 | 6 | 0 | 0.097080 | 0.097070 | 0.000033 | 0.097027–0.097120 | 345.64 | +0.5 |
| 14 | 13.5 | `tile8x16_column_major` | `iiijjjj` | 55.8354 | 6 | 0 | 0.097080 | 0.097107 | 0.000099 | 0.096987–0.097254 | 345.64 | +0.5 |
| 14 | 12 | `tile8x32_column_major` | `iiijjjjj` | 55.8354 | 6 | 0 | 0.096907 | 0.096915 | 0.000036 | 0.096867–0.096973 | 346.25 | +2.0 |
| 17 | 16 | `tile16_column_major` | `iiiijjjj` | 77.3354 | 6 | 0 | 0.159441 | 0.159438 | 0.000042 | 0.159374–0.159495 | 210.45 | +1.0 |
| 17 | 20 | `tile16x32_column_major` | `iiiijjjjj` | 77.3354 | 6 | 0 | 0.159852 | 0.159738 | 0.000257 | 0.159372–0.160012 | 209.91 | -3.0 |
| 17 | 17 | `tile16x8_column_major` | `iiiijjj` | 77.3354 | 6 | 0 | 0.159654 | 0.159654 | 0.000115 | 0.159454–0.159801 | 210.17 | +0.0 |
| 20 | 18.5 | `tile32_column_major` | `iiiiijjjjj` | 81.3354 | 6 | 0 | 0.159814 | 0.159972 | 0.000315 | 0.159694–0.160534 | 209.96 | +1.5 |
| 20 | 21 | `tile32x16_column_major` | `iiiiijjjj` | 81.3354 | 6 | 0 | 0.159854 | 0.159854 | 0.000069 | 0.159734–0.159947 | 209.91 | -1.0 |
| 20 | 18.5 | `tile32x8_column_major` | `iiiiijjj` | 81.3354 | 6 | 0 | 0.159814 | 0.159865 | 0.000104 | 0.159787–0.160067 | 209.96 | +1.5 |
| 22 | 22 | `column_major` | `iiiiiiiijjjjjjjj` | 89.3354 | 6 | 0 | 0.169388 | 0.169375 | 0.000069 | 0.169295–0.169455 | 198.09 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 18/22 | 0.818 | 0.318 | 3.500 |
| `peak-normalized-excess` | 9/22 | 0.409 | 2.750 | 13.500 |
| `weighted-normalized-excess` (selected) | 19/22 | 0.864 | 0.273 | 3.500 |

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
| 2 | 2 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 | 0.033733 | 0.033717 | 0.000141 | 0.033534–0.033947 | 7.79 | +0.0 |
| 2 | 1 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 | 0.033160 | 0.033117 | 0.000108 | 0.032920–0.033240 | 7.93 | +1.0 |
| 2 | 3 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 | 0.034094 | 0.034038 | 0.000129 | 0.033827–0.034187 | 7.71 | -1.0 |
| 4.5 | 4 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 | 0.034173 | 0.034309 | 0.000287 | 0.034067–0.034867 | 7.69 | +0.5 |
| 4.5 | 8 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 | 0.038867 | 0.038827 | 0.000121 | 0.038680–0.038987 | 6.76 | -3.5 |
| 7 | 5 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 | 0.034853 | 0.034949 | 0.000189 | 0.034747–0.035254 | 7.54 | +2.0 |
| 7 | 6 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 | 0.035240 | 0.035299 | 0.000095 | 0.035227–0.035480 | 7.46 | +1.0 |
| 7 | 7 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 | 0.035414 | 0.035462 | 0.000061 | 0.035413–0.035560 | 7.42 | +0.0 |
| 10 | 10 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 | 0.039480 | 0.039488 | 0.000061 | 0.039400–0.039587 | 6.66 | +0.0 |
| 10 | 9 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 | 0.039293 | 0.039267 | 0.000087 | 0.039160–0.039387 | 6.69 | +1.0 |
| 10 | 15 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 | 0.040293 | 0.040376 | 0.000326 | 0.040014–0.040987 | 6.52 | -5.0 |
| 13 | 12 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 | 0.039854 | 0.039947 | 0.000451 | 0.039347–0.040667 | 6.60 | +1.0 |
| 13 | 14 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 | 0.040187 | 0.040160 | 0.000386 | 0.039507–0.040614 | 6.54 | -1.0 |
| 13 | 17 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 | 0.040587 | 0.040590 | 0.000122 | 0.040414–0.040787 | 6.48 | -4.0 |
| 16 | 11 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 | 0.039720 | 0.039851 | 0.000463 | 0.039467–0.040733 | 6.62 | +5.0 |
| 16 | 16 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 | 0.040334 | 0.040208 | 0.000254 | 0.039747–0.040453 | 6.52 | +0.0 |
| 16 | 13 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 | 0.040054 | 0.039912 | 0.000207 | 0.039587–0.040107 | 6.56 | +3.0 |
| 18 | 18 | `column_major` | `iiiiiiiijjjjjjjj` | 27 | 4 | 0 | 0.041160 | 0.041123 | 0.000254 | 0.040733–0.041507 | 6.39 | +0.0 |
| 20 | 21 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 | 0.056733 | 0.056901 | 0.000372 | 0.056400–0.057347 | 4.63 | -1.0 |
| 20 | 20 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 | 0.056694 | 0.056776 | 0.000443 | 0.056294–0.057560 | 4.64 | +0.0 |
| 20 | 19 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 | 0.054307 | 0.054152 | 0.000335 | 0.053694–0.054574 | 4.84 | +1.0 |
| 22 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 63 | 4 | 0 | 0.064667 | 0.064643 | 0.001209 | 0.063320–0.066627 | 4.07 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 6/22 | 0.273 | 3.591 | 10.000 |
| `peak-normalized-excess` | 13/22 | 0.591 | 0.773 | 3.500 |
| `weighted-normalized-excess` (selected) | 17/22 | 0.773 | 0.341 | 3.500 |

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
| 2.5 | 4 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 | 0.040134 | 0.040203 | 0.000151 | 0.040014–0.040440 | 6.57 | -1.5 |
| 2.5 | 2 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 | 0.039987 | 0.039912 | 0.000200 | 0.039547–0.040120 | 6.59 | +0.5 |
| 2.5 | 1 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 | 0.039707 | 0.039696 | 0.000132 | 0.039520–0.039893 | 6.64 | +1.5 |
| 2.5 | 3 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 | 0.040013 | 0.039984 | 0.000174 | 0.039760–0.040240 | 6.59 | -0.5 |
| 5 | 5 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 | 0.042173 | 0.042147 | 0.000204 | 0.041920–0.042480 | 6.25 | +0.0 |
| 6 | 11 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 | 0.047573 | 0.047557 | 0.000358 | 0.047013–0.048094 | 5.54 | -5.0 |
| 7 | 8 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 | 0.043307 | 0.043309 | 0.000190 | 0.043067–0.043547 | 6.09 | -1.0 |
| 8 | 6 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 | 0.043160 | 0.043120 | 0.000308 | 0.042693–0.043627 | 6.11 | +2.0 |
| 9 | 9 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 | 0.044973 | 0.044949 | 0.000207 | 0.044640–0.045267 | 5.86 | +0.0 |
| 10 | 10 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 | 0.045680 | 0.045512 | 0.000395 | 0.044774–0.045920 | 5.77 | +0.0 |
| 11 | 7 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 | 0.043267 | 0.043112 | 0.000534 | 0.042427–0.043933 | 6.09 | +4.0 |
| 13 | 14 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 | 0.049774 | 0.050051 | 0.000573 | 0.049454–0.051094 | 5.30 | -1.0 |
| 13 | 12 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 | 0.049494 | 0.049483 | 0.000139 | 0.049240–0.049667 | 5.33 | +1.0 |
| 13 | 13 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 | 0.049627 | 0.049592 | 0.000175 | 0.049334–0.049774 | 5.31 | +0.0 |
| 15 | 15 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 | 0.052494 | 0.052552 | 0.000141 | 0.052387–0.052774 | 5.02 | +0.0 |
| 16 | 16 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 | 0.054040 | 0.054158 | 0.000638 | 0.053454–0.055240 | 4.88 | +0.0 |
| 17 | 20 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 | 0.055414 | 0.055150 | 0.000430 | 0.054334–0.055480 | 4.76 | -3.0 |
| 19 | 19 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 | 0.055347 | 0.055414 | 0.000319 | 0.055120–0.056013 | 4.76 | +0.0 |
| 19 | 17 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 | 0.054293 | 0.054237 | 0.000147 | 0.053987–0.054414 | 4.86 | +2.0 |
| 19 | 18 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 | 0.054440 | 0.054491 | 0.000329 | 0.053960–0.054853 | 4.84 | +1.0 |
| 21 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 15.75 | 2 | 0 | 0.056587 | 0.056142 | 0.000833 | 0.054560–0.056840 | 4.66 | -1.0 |
| 22 | 21 | `column_major` | `iiiiiiiijjjjjjjj` | 22.5625 | 2 | 0 | 0.055960 | 0.055909 | 0.000471 | 0.055307–0.056667 | 4.71 | +1.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 20/22 | 0.909 | 0.364 | 5.000 |
| `peak-normalized-excess` | 6/22 | 0.273 | 1.091 | 6.500 |
| `weighted-normalized-excess` (selected) | 20/22 | 0.909 | 0.364 | 5.000 |

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
| 2 | 2 | `tile8_column_major` | `iiijjj` | 1.76863 | 4 | 0 | 0.065947 | 0.065936 | 0.000033 | 0.065894–0.065987 | 508.81 | +0.0 |
| 2 | 1 | `tile8x16_column_major` | `iiijjjj` | 1.76863 | 4 | 0 | 0.065920 | 0.065974 | 0.000092 | 0.065907–0.066147 | 509.01 | +1.0 |
| 2 | 3 | `tile8x32_column_major` | `iiijjjjj` | 1.76863 | 4 | 0 | 0.066054 | 0.066062 | 0.000023 | 0.066027–0.066094 | 507.98 | -1.0 |
| 6.5 | 4 | `tile16_column_major` | `iiiijjjj` | 2.76863 | 4 | 0 | 0.066200 | 0.066211 | 0.000027 | 0.066174–0.066254 | 506.86 | +2.5 |
| 6.5 | 5 | `tile16x32_column_major` | `iiiijjjjj` | 2.76863 | 4 | 0 | 0.066267 | 0.066259 | 0.000011 | 0.066241–0.066267 | 506.35 | +1.5 |
| 6.5 | 6 | `tile16x8_column_major` | `iiiijjj` | 2.76863 | 4 | 0 | 0.066280 | 0.066275 | 0.000038 | 0.066214–0.066320 | 506.25 | +0.5 |
| 6.5 | 9 | `tile32_column_major` | `iiiiijjjjj` | 2.76863 | 4 | 0 | 0.066627 | 0.066627 | 0.000123 | 0.066414–0.066760 | 503.62 | -2.5 |
| 6.5 | 7 | `tile32x16_column_major` | `iiiiijjjj` | 2.76863 | 4 | 0 | 0.066387 | 0.066393 | 0.000034 | 0.066347–0.066454 | 505.43 | -0.5 |
| 6.5 | 8 | `tile32x8_column_major` | `iiiiijjj` | 2.76863 | 4 | 0 | 0.066400 | 0.066536 | 0.000194 | 0.066347–0.066773 | 505.34 | -1.5 |
| 10 | 10 | `column_major` | `iiiiiiiijjjjjjjj` | 9.76863 | 4 | 0 | 0.067347 | 0.067347 | 0.000025 | 0.067320–0.067387 | 498.23 | +0.0 |
| 11.5 | 11 | `tile16_interleaved` | `jijijiji` | 15.8226 | 16 | 0 | 0.069947 | 0.069990 | 0.000188 | 0.069760–0.070240 | 479.71 | +0.5 |
| 11.5 | 12 | `tile32_interleaved` | `jijijijiji` | 15.8226 | 20 | 0 | 0.070240 | 0.070326 | 0.000195 | 0.070134–0.070627 | 477.71 | -0.5 |
| 14 | 14 | `tile16x8_row_major` | `jjjiiii` | 35.7484 | 4 | 0 | 0.096734 | 0.096763 | 0.000043 | 0.096721–0.096828 | 346.87 | +0.0 |
| 14 | 15 | `tile32x8_row_major` | `jjjiiiii` | 35.7484 | 4 | 0 | 0.096961 | 0.096969 | 0.000040 | 0.096921–0.097041 | 346.06 | -1.0 |
| 14 | 13 | `tile8_row_major` | `jjjiii` | 35.7484 | 4 | 0 | 0.096573 | 0.096595 | 0.000063 | 0.096520–0.096707 | 347.45 | +1.0 |
| 19 | 22 | `row_major` | `jjjjjjjjiiiiiiii` | 37.7562 | 4 | 0 | 0.160868 | 0.160895 | 0.000123 | 0.160695–0.161054 | 208.58 | -3.0 |
| 19 | 18 | `tile16_row_major` | `jjjjiiii` | 37.7562 | 4 | 0 | 0.159055 | 0.159273 | 0.000335 | 0.159028–0.159908 | 210.96 | +1.0 |
| 19 | 17 | `tile16x32_row_major` | `jjjjjiiii` | 37.7562 | 4 | 0 | 0.159028 | 0.159071 | 0.000083 | 0.159001–0.159228 | 211.00 | +2.0 |
| 19 | 21 | `tile32_row_major` | `jjjjjiiiii` | 37.7562 | 4 | 0 | 0.159321 | 0.159323 | 0.000124 | 0.159160–0.159481 | 210.61 | -2.0 |
| 19 | 19 | `tile32x16_row_major` | `jjjjiiiii` | 37.7562 | 4 | 0 | 0.159241 | 0.159227 | 0.000109 | 0.159041–0.159347 | 210.72 | +0.0 |
| 19 | 20 | `tile8x16_row_major` | `jjjjiii` | 37.7562 | 4 | 0 | 0.159294 | 0.159251 | 0.000129 | 0.159014–0.159400 | 210.65 | -1.0 |
| 19 | 16 | `tile8x32_row_major` | `jjjjjiii` | 37.7562 | 4 | 0 | 0.158788 | 0.158825 | 0.000063 | 0.158761–0.158934 | 211.32 | +3.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 12/22 | 0.545 | 0.500 | 3.000 |
| `peak-normalized-excess` | 5/22 | 0.227 | 3.591 | 9.500 |
| `weighted-normalized-excess` (selected) | 12/22 | 0.545 | 0.500 | 3.000 |
