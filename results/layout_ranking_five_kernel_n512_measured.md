# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runs and XORs are separate address-code generation costs. They are included in the Pareto frontier but are not folded into the scalar locality score or score rank.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| ATAX | 512 | 22 | 4 | 0.500 | 1.682 |
| GEMM | 512 | 22 | 4 | 0.455 | 1.273 |
| GESUMMV | 512 | 22 | 7 | 0.727 | 0.705 |
| MVT | 512 | 22 | 5 | 0.682 | 0.864 |
| SYRK | 512 | 22 | 4 | 0.227 | 1.091 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 | 0.119854 | 0.119830 | 0.000296 | 0.119280–0.120120 | 8.75 | +1.0 |
| 2 | 2 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 | 0.120081 | 0.120209 | 0.000388 | 0.119801–0.120947 | 8.73 | +0.0 |
| 2 | 3 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 | 0.121187 | 0.121168 | 0.000451 | 0.120507–0.121787 | 8.65 | -1.0 |
| 4.5 | 7 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 | 0.130707 | 0.130694 | 0.000149 | 0.130480–0.130854 | 8.02 | -2.5 |
| 4.5 | 10 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 | 0.139947 | 0.140038 | 0.000312 | 0.139720–0.140627 | 7.49 | -5.5 |
| 7 | 5 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 | 0.124427 | 0.124336 | 0.000512 | 0.123387–0.124867 | 8.43 | +2.0 |
| 7 | 6 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 | 0.125801 | 0.125782 | 0.000235 | 0.125347–0.126040 | 8.34 | +1.0 |
| 7 | 4 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 | 0.121960 | 0.122064 | 0.000368 | 0.121600–0.122574 | 8.60 | +3.0 |
| 10 | 12 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 | 0.140494 | 0.140547 | 0.000635 | 0.139613–0.141387 | 7.46 | -2.0 |
| 10 | 16 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 | 0.142014 | 0.141905 | 0.000554 | 0.140934–0.142547 | 7.38 | -6.0 |
| 10 | 13 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 | 0.140560 | 0.140390 | 0.000416 | 0.139720–0.140827 | 7.46 | -3.0 |
| 13 | 9 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 | 0.138040 | 0.137918 | 0.000246 | 0.137534–0.138200 | 7.60 | +4.0 |
| 13 | 8 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 | 0.137748 | 0.137694 | 0.000369 | 0.137067–0.138214 | 7.61 | +5.0 |
| 13 | 14 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 | 0.140667 | 0.140589 | 0.000295 | 0.140067–0.140880 | 7.45 | -1.0 |
| 15 | 21 | `column_major` | `iiiiiiiiijjjjjjjjj` | 37 | 2 | 0 | 0.166880 | 0.167451 | 0.000914 | 0.166534–0.169014 | 6.28 | -6.0 |
| 17 | 20 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 | 0.143841 | 0.143992 | 0.000522 | 0.143214–0.144600 | 7.29 | -3.0 |
| 17 | 11 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 | 0.140054 | 0.140006 | 0.000540 | 0.139041–0.140708 | 7.49 | +6.0 |
| 17 | 18 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 | 0.143027 | 0.142982 | 0.000322 | 0.142400–0.143347 | 7.33 | -1.0 |
| 20 | 15 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 | 0.141640 | 0.141736 | 0.000417 | 0.141307–0.142467 | 7.40 | +5.0 |
| 20 | 19 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 | 0.143281 | 0.143534 | 0.000567 | 0.142814–0.144267 | 7.32 | +1.0 |
| 20 | 17 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 | 0.143001 | 0.142907 | 0.000911 | 0.141507–0.144121 | 7.33 | +3.0 |
| 22 | 22 | `row_major` | `jjjjjjjjjiiiiiiiii` | 69.75 | 2 | 0 | 0.169107 | 0.169094 | 0.000436 | 0.168521–0.169574 | 6.20 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 9/22 | 0.409 | 2.318 | 9.000 |
| `peak-normalized-excess` | 9/22 | 0.409 | 1.455 | 9.000 |
| `weighted-normalized-excess` (selected) | 11/22 | 0.500 | 1.682 | 6.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.5 | 2 | `tile16x32_row_major` | `jjjjjiiii` | 8.29751 | 6 | 0 | 0.258882 | 0.258890 | 0.000029 | 0.258855–0.258935 | 1036.90 | +1.5 |
| 3.5 | 1 | `tile16x8_row_major` | `jjjiiii` | 8.29751 | 6 | 0 | 0.258575 | 0.258564 | 0.000113 | 0.258375–0.258722 | 1038.13 | +2.5 |
| 3.5 | 7 | `tile32_row_major` | `jjjjjiiiii` | 8.29751 | 6 | 0 | 0.286815 | 0.288943 | 0.006566 | 0.280775–0.299401 | 935.92 | -3.5 |
| 3.5 | 10 | `tile32x8_row_major` | `jjjiiiii` | 8.29751 | 6 | 0 | 0.310402 | 0.310488 | 0.000875 | 0.309136–0.311709 | 864.80 | -6.5 |
| 3.5 | 5 | `tile8_row_major` | `jjjiii` | 8.29751 | 6 | 0 | 0.259868 | 0.260020 | 0.000451 | 0.259468–0.260815 | 1032.97 | -1.5 |
| 3.5 | 6 | `tile8x32_row_major` | `jjjjjiii` | 8.29751 | 6 | 0 | 0.260121 | 0.260038 | 0.000585 | 0.259321–0.260975 | 1031.96 | -2.5 |
| 8.5 | 11 | `row_major` | `jjjjjjjjjiiiiiiiii` | 8.79751 | 6 | 0 | 0.320228 | 0.320247 | 0.000727 | 0.319188–0.321375 | 838.26 | -2.5 |
| 8.5 | 3 | `tile16_row_major` | `jjjjiiii` | 8.79751 | 6 | 0 | 0.259321 | 0.259340 | 0.000095 | 0.259241–0.259481 | 1035.15 | +5.5 |
| 8.5 | 8 | `tile32x16_row_major` | `jjjjiiiii` | 8.79751 | 6 | 0 | 0.290041 | 0.289761 | 0.001881 | 0.286841–0.292308 | 925.51 | +0.5 |
| 8.5 | 4 | `tile8x16_row_major` | `jjjjiii` | 8.79751 | 6 | 0 | 0.259720 | 0.259934 | 0.000352 | 0.259627–0.260574 | 1033.56 | +4.5 |
| 11.5 | 9 | `tile16_interleaved` | `jijijiji` | 15.69 | 24 | 0 | 0.303495 | 0.303719 | 0.000448 | 0.303375–0.304575 | 884.48 | +2.5 |
| 11.5 | 12 | `tile32_interleaved` | `jijijijiji` | 15.69 | 30 | 0 | 0.379027 | 0.379024 | 0.000538 | 0.378240–0.379854 | 708.22 | -0.5 |
| 14 | 15 | `tile8_column_major` | `iiijjj` | 55.8676 | 6 | 0 | 0.384174 | 0.384183 | 0.000071 | 0.384068–0.384281 | 698.73 | -1.0 |
| 14 | 13 | `tile8x16_column_major` | `iiijjjj` | 55.8676 | 6 | 0 | 0.383174 | 0.383134 | 0.000096 | 0.383014–0.383255 | 700.56 | +1.0 |
| 14 | 14 | `tile8x32_column_major` | `iiijjjjj` | 55.8676 | 6 | 0 | 0.384081 | 0.384038 | 0.000074 | 0.383895–0.384094 | 698.90 | +0.0 |
| 17 | 16 | `tile16_column_major` | `iiiijjjj` | 77.3676 | 6 | 0 | 0.632856 | 0.632822 | 0.000197 | 0.632550–0.633043 | 424.16 | +1.0 |
| 17 | 17 | `tile16x32_column_major` | `iiiijjjjj` | 77.3676 | 6 | 0 | 0.632881 | 0.632812 | 0.000250 | 0.632362–0.633122 | 424.15 | +0.0 |
| 17 | 18 | `tile16x8_column_major` | `iiiijjj` | 77.3676 | 6 | 0 | 0.633216 | 0.633123 | 0.000260 | 0.632683–0.633403 | 423.92 | -1.0 |
| 20 | 19 | `tile32_column_major` | `iiiiijjjjj` | 81.3676 | 6 | 0 | 0.634718 | 0.634715 | 0.000051 | 0.634625–0.634784 | 422.92 | +1.0 |
| 20 | 20 | `tile32x16_column_major` | `iiiiijjjj` | 81.3676 | 6 | 0 | 0.635002 | 0.635039 | 0.000073 | 0.634975–0.635175 | 422.73 | +0.0 |
| 20 | 21 | `tile32x8_column_major` | `iiiiijjj` | 81.3676 | 6 | 0 | 0.635162 | 0.635098 | 0.000150 | 0.634855–0.635295 | 422.63 | -1.0 |
| 22 | 22 | `column_major` | `iiiiiiiiijjjjjjjjj` | 89.3676 | 6 | 0 | 0.646683 | 0.647165 | 0.000999 | 0.645976–0.648376 | 415.10 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 9/22 | 0.409 | 1.318 | 8.000 |
| `peak-normalized-excess` | 2/22 | 0.091 | 4.091 | 10.500 |
| `weighted-normalized-excess` (selected) | 10/22 | 0.455 | 1.273 | 6.500 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 3 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 | 0.063813 | 0.063848 | 0.000541 | 0.063320–0.064827 | 16.46 | -1.0 |
| 2 | 1 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 | 0.063427 | 0.063512 | 0.000176 | 0.063293–0.063734 | 16.56 | +1.0 |
| 2 | 2 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 | 0.063693 | 0.063632 | 0.000239 | 0.063227–0.063960 | 16.49 | +0.0 |
| 4.5 | 4 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 | 0.067801 | 0.067849 | 0.000134 | 0.067707–0.068014 | 15.49 | +0.5 |
| 4.5 | 8 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 | 0.076267 | 0.076328 | 0.000146 | 0.076160–0.076587 | 13.77 | -3.5 |
| 7 | 7 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 | 0.068853 | 0.068741 | 0.000287 | 0.068214–0.069013 | 15.25 | +0.0 |
| 7 | 6 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 | 0.068534 | 0.068494 | 0.000247 | 0.068160–0.068880 | 15.32 | +1.0 |
| 7 | 5 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 | 0.068320 | 0.068368 | 0.000530 | 0.067867–0.069360 | 15.37 | +2.0 |
| 10 | 11 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 | 0.076693 | 0.076661 | 0.000080 | 0.076507–0.076733 | 13.69 | -1.0 |
| 10 | 9 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 | 0.076280 | 0.076272 | 0.000090 | 0.076134–0.076374 | 13.77 | +1.0 |
| 10 | 10 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 | 0.076347 | 0.076430 | 0.000198 | 0.076280–0.076814 | 13.75 | +0.0 |
| 13 | 18 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 | 0.081654 | 0.081523 | 0.000458 | 0.080760–0.082067 | 12.86 | -5.0 |
| 13 | 12 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 | 0.078133 | 0.078176 | 0.000231 | 0.077854–0.078574 | 13.44 | +1.0 |
| 13 | 13 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 | 0.078134 | 0.077990 | 0.000304 | 0.077427–0.078307 | 13.44 | +0.0 |
| 16 | 14 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 | 0.078254 | 0.078219 | 0.000499 | 0.077694–0.079040 | 13.42 | +2.0 |
| 16 | 17 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 | 0.081120 | 0.081035 | 0.000225 | 0.080747–0.081334 | 12.95 | -1.0 |
| 16 | 16 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 | 0.078907 | 0.078734 | 0.000493 | 0.078013–0.079347 | 13.31 | +0.0 |
| 19 | 20 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 | 0.113027 | 0.113257 | 0.000541 | 0.112614–0.114081 | 9.29 | -1.0 |
| 19 | 21 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 | 0.113107 | 0.113187 | 0.000433 | 0.112627–0.113920 | 9.28 | -2.0 |
| 19 | 19 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 | 0.109000 | 0.108960 | 0.000164 | 0.108720–0.109214 | 9.63 | +0.0 |
| 21 | 15 | `column_major` | `iiiiiiiiijjjjjjjjj` | 43 | 4 | 0 | 0.078294 | 0.078459 | 0.000295 | 0.078267–0.079041 | 13.41 | +6.0 |
| 22 | 22 | `row_major` | `jjjjjjjjjiiiiiiiii` | 63 | 4 | 0 | 0.124387 | 0.124883 | 0.001036 | 0.123640–0.126574 | 8.44 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 5/22 | 0.227 | 3.545 | 10.000 |
| `peak-normalized-excess` | 10/22 | 0.455 | 1.159 | 3.500 |
| `weighted-normalized-excess` (selected) | 16/22 | 0.727 | 0.705 | 5.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5 | 3 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 | 0.077840 | 0.077840 | 0.000301 | 0.077493–0.078307 | 13.51 | -0.5 |
| 2.5 | 2 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 | 0.077721 | 0.077699 | 0.000216 | 0.077414–0.078067 | 13.53 | +0.5 |
| 2.5 | 4 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 | 0.078267 | 0.078091 | 0.000308 | 0.077587–0.078373 | 13.44 | -1.5 |
| 2.5 | 1 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 | 0.077707 | 0.077688 | 0.000353 | 0.077147–0.078254 | 13.53 | +1.5 |
| 5 | 5 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 | 0.082147 | 0.082195 | 0.000120 | 0.082080–0.082414 | 12.80 | +0.0 |
| 6 | 11 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 | 0.095747 | 0.095830 | 0.000167 | 0.095667–0.096120 | 10.98 | -5.0 |
| 7 | 7 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 | 0.084187 | 0.084312 | 0.000267 | 0.083974–0.084654 | 12.49 | +0.0 |
| 8 | 6 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 | 0.083987 | 0.083845 | 0.000282 | 0.083387–0.084174 | 12.52 | +2.0 |
| 9 | 9 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 | 0.087160 | 0.087190 | 0.000219 | 0.086880–0.087467 | 12.07 | +0.0 |
| 10 | 10 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 | 0.089347 | 0.089144 | 0.000407 | 0.088494–0.089640 | 11.77 | +0.0 |
| 11 | 8 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 | 0.084947 | 0.084776 | 0.000478 | 0.083894–0.085320 | 12.38 | +3.0 |
| 13 | 14 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 | 0.097987 | 0.098001 | 0.000473 | 0.097254–0.098747 | 10.73 | -1.0 |
| 13 | 13 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 | 0.096227 | 0.096360 | 0.000221 | 0.096174–0.096760 | 10.93 | +0.0 |
| 13 | 12 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 | 0.096107 | 0.096217 | 0.000296 | 0.095827–0.096667 | 10.94 | +1.0 |
| 15 | 15 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 | 0.103200 | 0.103243 | 0.000461 | 0.102534–0.103947 | 10.19 | +0.0 |
| 16 | 19 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 | 0.107627 | 0.107643 | 0.000506 | 0.107094–0.108427 | 9.77 | -3.0 |
| 17 | 22 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 | 0.110147 | 0.109880 | 0.000691 | 0.108787–0.110814 | 9.55 | -5.0 |
| 19 | 20 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 | 0.109360 | 0.109342 | 0.000325 | 0.108760–0.109734 | 9.62 | -1.0 |
| 19 | 17 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 | 0.106707 | 0.106725 | 0.000344 | 0.106333–0.107267 | 9.86 | +2.0 |
| 19 | 18 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 | 0.107546 | 0.107402 | 0.000297 | 0.106973–0.107746 | 9.78 | +1.0 |
| 21 | 21 | `row_major` | `jjjjjjjjjiiiiiiiii` | 15.75 | 2 | 0 | 0.109854 | 0.109750 | 0.000596 | 0.108694–0.110454 | 9.57 | +0.0 |
| 22 | 16 | `column_major` | `iiiiiiiiijjjjjjjjj` | 25.5625 | 2 | 0 | 0.106374 | 0.106123 | 0.000570 | 0.105040–0.106641 | 9.89 | +6.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 15/22 | 0.682 | 0.864 | 5.000 |
| `peak-normalized-excess` | 5/22 | 0.227 | 1.591 | 6.500 |
| `weighted-normalized-excess` (selected) | 15/22 | 0.682 | 0.864 | 5.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 5 | `tile8_column_major` | `iiijjj` | 1.75935 | 4 | 0 | 0.259973 | 0.260052 | 0.000121 | 0.259933–0.260212 | 1032.55 | -3.0 |
| 2 | 1 | `tile8x16_column_major` | `iiijjjj` | 1.75935 | 4 | 0 | 0.258561 | 0.258582 | 0.000076 | 0.258481–0.258708 | 1038.19 | +1.0 |
| 2 | 3 | `tile8x32_column_major` | `iiijjjjj` | 1.75935 | 4 | 0 | 0.259388 | 0.259363 | 0.000091 | 0.259227–0.259481 | 1034.88 | -1.0 |
| 6.5 | 2 | `tile16_column_major` | `iiiijjjj` | 2.75935 | 4 | 0 | 0.259255 | 0.259252 | 0.000093 | 0.259108–0.259402 | 1035.41 | +4.5 |
| 6.5 | 4 | `tile16x32_column_major` | `iiiijjjjj` | 2.75935 | 4 | 0 | 0.259654 | 0.259726 | 0.000149 | 0.259614–0.260014 | 1033.82 | +2.5 |
| 6.5 | 7 | `tile16x8_column_major` | `iiiijjj` | 2.75935 | 4 | 0 | 0.260307 | 0.260603 | 0.000379 | 0.260280–0.261067 | 1031.23 | -0.5 |
| 6.5 | 9 | `tile32_column_major` | `iiiiijjjjj` | 2.75935 | 4 | 0 | 0.262173 | 0.261744 | 0.000643 | 0.260920–0.262360 | 1023.89 | -2.5 |
| 6.5 | 6 | `tile32x16_column_major` | `iiiiijjjj` | 2.75935 | 4 | 0 | 0.260268 | 0.260281 | 0.000204 | 0.259974–0.260535 | 1031.38 | +0.5 |
| 6.5 | 8 | `tile32x8_column_major` | `iiiiijjj` | 2.75935 | 4 | 0 | 0.261747 | 0.261520 | 0.000614 | 0.260773–0.262240 | 1025.55 | -1.5 |
| 10.5 | 11 | `tile16_interleaved` | `jijijiji` | 15.83 | 16 | 0 | 0.275842 | 0.275965 | 0.000345 | 0.275602–0.276615 | 973.15 | -0.5 |
| 10.5 | 12 | `tile32_interleaved` | `jijijijiji` | 15.83 | 20 | 0 | 0.282854 | 0.282854 | 0.001447 | 0.281200–0.284640 | 949.03 | -1.5 |
| 12 | 10 | `column_major` | `iiiiiiiiijjjjjjjjj` | 17.7593 | 4 | 0 | 0.262654 | 0.262702 | 0.000229 | 0.262440–0.263120 | 1022.01 | +2.0 |
| 14 | 14 | `tile16x8_row_major` | `jjjiiii` | 35.7663 | 4 | 0 | 0.382308 | 0.382318 | 0.000115 | 0.382174–0.382521 | 702.15 | +0.0 |
| 14 | 15 | `tile32x8_row_major` | `jjjiiiii` | 35.7663 | 4 | 0 | 0.383281 | 0.383302 | 0.000096 | 0.383187–0.383414 | 700.36 | -1.0 |
| 14 | 13 | `tile8_row_major` | `jjjiii` | 35.7663 | 4 | 0 | 0.381855 | 0.381837 | 0.000096 | 0.381709–0.381989 | 702.98 | +1.0 |
| 19 | 22 | `row_major` | `jjjjjjjjjiiiiiiiii` | 37.7741 | 4 | 0 | 0.638697 | 0.638899 | 0.000702 | 0.638150–0.640230 | 420.29 | -3.0 |
| 19 | 17 | `tile16_row_major` | `jjjjiiii` | 37.7741 | 4 | 0 | 0.631697 | 0.631732 | 0.000079 | 0.631630–0.631830 | 424.94 | +2.0 |
| 19 | 18 | `tile16x32_row_major` | `jjjjjiiii` | 37.7741 | 4 | 0 | 0.631723 | 0.631760 | 0.000146 | 0.631549–0.631990 | 424.93 | +1.0 |
| 19 | 20 | `tile32_row_major` | `jjjjjiiiii` | 37.7741 | 4 | 0 | 0.633198 | 0.633278 | 0.000181 | 0.633024–0.633518 | 423.94 | -1.0 |
| 19 | 19 | `tile32x16_row_major` | `jjjjiiiii` | 37.7741 | 4 | 0 | 0.632748 | 0.632711 | 0.000067 | 0.632601–0.632788 | 424.24 | +0.0 |
| 19 | 21 | `tile8x16_row_major` | `jjjjiii` | 37.7741 | 4 | 0 | 0.633309 | 0.633291 | 0.000041 | 0.633230–0.633336 | 423.86 | -2.0 |
| 19 | 16 | `tile8x32_row_major` | `jjjjjiii` | 37.7741 | 4 | 0 | 0.631176 | 0.631198 | 0.000150 | 0.631003–0.631416 | 425.29 | +3.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 6/22 | 0.273 | 0.955 | 3.500 |
| `peak-normalized-excess` | 1/22 | 0.045 | 4.091 | 10.500 |
| `weighted-normalized-excess` (selected) | 5/22 | 0.227 | 1.091 | 3.500 |
