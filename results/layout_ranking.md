# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| GEMM | 256 | 8 | 1 | 0.750 | 0.500 |
| GEMM | 512 | 8 | 2 | 0.500 | 0.500 |
| GEMM | 1024 | 8 | 2 | 1.000 | 0.000 |
| GESUMMV | 256 | 8 | 2 | 0.750 | 0.375 |
| GESUMMV | 512 | 8 | 2 | 0.750 | 0.250 |
| GESUMMV | 1024 | 8 | 2 | 0.500 | 0.750 |

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

This is the exact non-dominated set over the notes-aligned `(Q_fine, J_peak, J_area)` score vector. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area |
| --- | --- | --- | --- |
| `tile8_row_major` | 24704 | 5 | 14.2117 |

### Layout ranks

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | `tile8_row_major` | `jjjiii` | 14.2117 | 0.065644 | 0.065649 | 0.000076 | 0.065553–0.065769 | 511.15 | -1.0 |
| 2 | 3 | `tile16_row_major` | `jjjjiiii` | 21.3367 | 0.065760 | 0.065745 | 0.000064 | 0.065656–0.065832 | 510.26 | -1.0 |
| 3 | 4 | `tile32_row_major` | `jjjjjiiiii` | 24.1284 | 0.065804 | 0.065822 | 0.000061 | 0.065752–0.065912 | 509.91 | -1.0 |
| 4 | 1 | `row_major` | `jjjjjjjjiiiiiiii` | 28.3367 | 0.065336 | 0.065332 | 0.000026 | 0.065272–0.065368 | 513.57 | +3.0 |
| 5 | 5 | `tile8_column_major` | `iiijjj` | 37.8829 | 0.097001 | 0.097035 | 0.000162 | 0.096865–0.097473 | 345.92 | +0.0 |
| 6 | 6 | `tile16_column_major` | `iiiijjjj` | 48.8829 | 0.158904 | 0.158912 | 0.000057 | 0.158848–0.159032 | 211.16 | +0.0 |
| 7 | 7 | `tile32_column_major` | `iiiiijjjjj` | 56.5496 | 0.159128 | 0.159162 | 0.000124 | 0.159032–0.159512 | 210.86 | +0.0 |
| 8 | 8 | `column_major` | `iiiiiiiijjjjjjjj` | 71.7163 | 0.168764 | 0.168760 | 0.000105 | 0.168568–0.168944 | 198.82 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 6/8 | 0.750 | 0.500 | 3.000 |
| `peak-normalized-excess` | 1/8 | 0.125 | 1.625 | 4.500 |
| `weighted-normalized-excess` (selected) | 6/8 | 0.750 | 0.500 | 3.000 |

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

This is the exact non-dominated set over the notes-aligned `(Q_fine, J_peak, J_area)` score vector. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area |
| --- | --- | --- | --- |
| `tile16_row_major` | 49280 | 8 | 21.6225 |
| `tile8_row_major` | 49280 | 12 | 16.0475 |

### Layout ranks

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | `tile8_row_major` | `jjjiii` | 16.0475 | 0.258332 | 0.258352 | 0.000192 | 0.258088–0.258632 | 1039.11 | +0.0 |
| 2 | 2 | `tile16_row_major` | `jjjjiiii` | 21.6225 | 0.258572 | 0.258675 | 0.000383 | 0.258280–0.259408 | 1038.15 | +0.0 |
| 3 | 4 | `tile32_row_major` | `jjjjjiiiii` | 24.2975 | 0.337740 | 0.337367 | 0.001227 | 0.334104–0.338464 | 794.80 | -1.0 |
| 4 | 3 | `row_major` | `jjjjjjjjjiiiiiiiii` | 34.6725 | 0.319809 | 0.319945 | 0.000727 | 0.319162–0.321274 | 839.36 | +1.0 |
| 5 | 5 | `tile8_column_major` | `iiijjj` | 39.7455 | 0.381664 | 0.381671 | 0.000045 | 0.381616–0.381752 | 703.33 | +0.0 |
| 6 | 7 | `tile16_column_major` | `iiiijjjj` | 49.1955 | 0.635967 | 0.635829 | 0.000469 | 0.634996–0.636443 | 422.09 | -1.0 |
| 7 | 6 | `tile32_column_major` | `iiiiijjjjj` | 56.7455 | 0.632192 | 0.632189 | 0.000107 | 0.632040–0.632352 | 424.61 | +1.0 |
| 8 | 8 | `column_major` | `iiiiiiiiijjjjjjjjj` | 85.6455 | 0.659995 | 0.660010 | 0.002465 | 0.655555–0.663931 | 406.72 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 3/8 | 0.375 | 0.750 | 2.000 |
| `peak-normalized-excess` | 1/8 | 0.125 | 1.500 | 3.000 |
| `weighted-normalized-excess` (selected) | 4/8 | 0.500 | 0.500 | 1.000 |

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

This is the exact non-dominated set over the notes-aligned `(Q_fine, J_peak, J_area)` score vector. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area |
| --- | --- | --- | --- |
| `tile16_row_major` | 98432 | 8 | 21.8126 |
| `tile8_row_major` | 98432 | 13.4444 | 16.4099 |

### Layout ranks

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | `tile8_row_major` | `jjjiii` | 16.4099 | 1.437082 | 1.440802 | 0.016648 | 1.424294–1.476966 | 1494.34 | +0.0 |
| 2 | 2 | `tile16_row_major` | `jjjjiiii` | 21.8126 | 1.487038 | 1.492660 | 0.029117 | 1.451986–1.539290 | 1444.14 | +0.0 |
| 3 | 3 | `tile32_row_major` | `jjjjjiiiii` | 24.4099 | 1.488776 | 1.488282 | 0.032915 | 1.424608–1.552152 | 1442.45 | +0.0 |
| 4 | 4 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 38.5626 | 1.504150 | 1.500014 | 0.026068 | 1.450862–1.540798 | 1427.71 | +0.0 |
| 5 | 5 | `tile8_column_major` | `iiijjj` | 40.1214 | 2.109208 | 2.104028 | 0.104358 | 1.928690–2.296846 | 1018.15 | +0.0 |
| 6 | 6 | `tile16_column_major` | `iiiijjjj` | 49.3991 | 3.162923 | 3.163061 | 0.000848 | 3.161440–3.164509 | 678.96 | +0.0 |
| 7 | 7 | `tile32_column_major` | `iiiiijjjjj` | 56.8714 | 3.163693 | 3.163686 | 0.000452 | 3.162900–3.164604 | 678.79 | +0.0 |
| 8 | 8 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 89.5936 | 3.427574 | 3.435535 | 0.026231 | 3.404536–3.486387 | 626.53 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 8/8 | 1.000 | 0.000 | 0.000 |
| `peak-normalized-excess` | 2/8 | 0.250 | 1.062 | 2.500 |
| `weighted-normalized-excess` (selected) | 8/8 | 1.000 | 0.000 | 0.000 |

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

This is the exact non-dominated set over the notes-aligned `(Q_fine, J_peak, J_area)` score vector. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area |
| --- | --- | --- | --- |
| `tile8_column_major` | 8192 | 15 | 24.625 |
| `tile8_row_major` | 65536 | 15 | 21.75 |

### Layout ranks

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | `tile8_row_major` | `jjjiii` | 21.75 | 0.033212 | 0.033166 | 0.000252 | 0.032784–0.033552 | 7.92 | +0.0 |
| 2 | 2 | `tile8_column_major` | `iiijjj` | 24.625 | 0.034360 | 0.034373 | 0.000209 | 0.034112–0.034896 | 7.65 | +0.0 |
| 3 | 3 | `tile16_row_major` | `jjjjiiii` | 24.75 | 0.039068 | 0.039070 | 0.000061 | 0.038952–0.039160 | 6.73 | +0.0 |
| 4 | 6 | `column_major` | `iiiiiiiijjjjjjjj` | 31.5 | 0.041028 | 0.041054 | 0.000259 | 0.040552–0.041392 | 6.41 | -2.0 |
| 5 | 4 | `tile32_column_major` | `iiiiijjjjj` | 31.875 | 0.039640 | 0.039648 | 0.000229 | 0.039200–0.040080 | 6.63 | +1.0 |
| 6 | 5 | `tile16_column_major` | `iiiijjjj` | 34.75 | 0.039720 | 0.039678 | 0.000282 | 0.039256–0.040072 | 6.62 | +1.0 |
| 7 | 7 | `tile32_row_major` | `jjjjjiiiii` | 40.75 | 0.056440 | 0.056434 | 0.000220 | 0.056048–0.056728 | 4.66 | +0.0 |
| 8 | 8 | `row_major` | `jjjjjjjjiiiiiiii` | 100.75 | 0.063412 | 0.063388 | 0.000875 | 0.061960–0.064824 | 4.15 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 2/8 | 0.250 | 1.125 | 3.000 |
| `peak-normalized-excess` | 2/8 | 0.250 | 1.000 | 2.500 |
| `weighted-normalized-excess` (selected) | 6/8 | 0.750 | 0.375 | 2.000 |

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

This is the exact non-dominated set over the notes-aligned `(Q_fine, J_peak, J_area)` score vector. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area |
| --- | --- | --- | --- |
| `tile8_column_major` | 16384 | 15 | 24.625 |
| `tile8_row_major` | 131072 | 15 | 21.75 |

### Layout ranks

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | `tile8_row_major` | `jjjiii` | 21.75 | 0.063772 | 0.063749 | 0.000156 | 0.063400–0.063992 | 16.47 | +0.0 |
| 2 | 2 | `tile8_column_major` | `iiijjj` | 24.625 | 0.067020 | 0.066966 | 0.000267 | 0.066584–0.067288 | 15.67 | +0.0 |
| 3 | 3 | `tile16_row_major` | `jjjjiiii` | 24.75 | 0.075984 | 0.076006 | 0.000104 | 0.075888–0.076168 | 13.82 | +0.0 |
| 4 | 6 | `tile32_column_major` | `iiiiijjjjj` | 31.875 | 0.081212 | 0.081065 | 0.000535 | 0.080065–0.081721 | 12.93 | -2.0 |
| 5 | 4 | `column_major` | `iiiiiiiiijjjjjjjjj` | 33.5 | 0.078548 | 0.078494 | 0.000368 | 0.077872–0.079000 | 13.37 | +1.0 |
| 6 | 5 | `tile16_column_major` | `iiiijjjj` | 34.75 | 0.081120 | 0.081180 | 0.000349 | 0.080688–0.081600 | 12.95 | +1.0 |
| 7 | 7 | `tile32_row_major` | `jjjjjiiiii` | 40.75 | 0.112500 | 0.112457 | 0.000281 | 0.111896–0.112976 | 9.33 | +0.0 |
| 8 | 8 | `row_major` | `jjjjjjjjjiiiiiiiii` | 100.75 | 0.121316 | 0.121411 | 0.000638 | 0.120161–0.122680 | 8.66 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 3/8 | 0.375 | 0.875 | 3.000 |
| `peak-normalized-excess` | 2/8 | 0.250 | 1.000 | 2.500 |
| `weighted-normalized-excess` (selected) | 6/8 | 0.750 | 0.250 | 1.000 |

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

This is the exact non-dominated set over the notes-aligned `(Q_fine, J_peak, J_area)` score vector. Runtime is not a Pareto objective.

| Layout | Q fine | J peak | J area |
| --- | --- | --- | --- |
| `tile8_column_major` | 32768 | 15 | 24.625 |
| `tile8_row_major` | 262144 | 15 | 21.75 |

### Layout ranks

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | `tile8_row_major` | `jjjiii` | 21.75 | 0.134736 | 0.134672 | 0.000192 | 0.134312–0.134928 | 31.15 | +0.0 |
| 2 | 3 | `tile8_column_major` | `iiijjj` | 24.625 | 0.158113 | 0.158310 | 0.000982 | 0.156769–0.159689 | 26.55 | -1.0 |
| 3 | 2 | `tile16_row_major` | `jjjjiiii` | 24.75 | 0.157316 | 0.157323 | 0.000177 | 0.157112–0.157712 | 26.68 | +1.0 |
| 4 | 4 | `tile32_column_major` | `iiiiijjjjj` | 31.875 | 0.180356 | 0.180248 | 0.001723 | 0.177288–0.183296 | 23.27 | +0.0 |
| 5 | 8 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 33.5 | 0.302469 | 0.302797 | 0.000955 | 0.301297–0.304433 | 13.88 | -3.0 |
| 6 | 5 | `tile16_column_major` | `iiiijjjj` | 34.75 | 0.186101 | 0.186359 | 0.002187 | 0.182625–0.190249 | 22.55 | +1.0 |
| 7 | 6 | `tile32_row_major` | `jjjjjiiiii` | 40.75 | 0.216792 | 0.216353 | 0.000930 | 0.214312–0.217432 | 19.36 | +1.0 |
| 8 | 7 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 100.75 | 0.276354 | 0.275168 | 0.003424 | 0.267486–0.279030 | 15.19 | +1.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 1/8 | 0.125 | 1.375 | 3.000 |
| `peak-normalized-excess` | 0/8 | 0.000 | 1.375 | 4.500 |
| `weighted-normalized-excess` (selected) | 4/8 | 0.500 | 0.750 | 3.000 |
