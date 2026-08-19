# RELAY layout score/runtime experiment

All scores, runtimes, and ranks are ascending costs; lower is better. The displayed score uses `weighted-normalized-excess`.

Runs and XORs are separate address-code generation costs. They are included in the Pareto frontier but are not folded into the scalar locality score or score rank.

Runtime rank is the raw rank of the exact sample median. Score rank is the raw rank of the exact modeled score. Timing variation does not change either rank or any table value.

The variation-aware rank metric uses each layout's observed minimum-to-maximum sample interval. An overlapping competitor can appear on either side, producing a plausible runtime-rank range. A score rank is counted accurate when it lies inside that range. This is a conservative observed-sample check, not a confidence interval.

## Summary

| Kernel | N | Layouts | Pareto layouts | Variation-aware rank accuracy | Mean rank error |
| --- | --- | --- | --- | --- | --- |
| ATAX | 1024 | 22 | 4 | 0.182 | 4.136 |
| GEMM | 1024 | 22 | 4 | 0.864 | 0.091 |
| GESUMMV | 1024 | 22 | 7 | 0.364 | 1.227 |
| MVT | 1024 | 22 | 5 | 0.136 | 2.909 |
| SYRK | 1024 | 22 | 4 | 0.864 | 0.136 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | `tile8_column_major` | `iiijjj` | 19 | 2 | 0 | 0.242176 | 0.241888 | 0.000925 | 0.240149–0.242709 | 17.32 | +1.0 |
| 2 | 3 | `tile8x16_column_major` | `iiijjjj` | 19 | 2 | 0 | 0.244162 | 0.244175 | 0.000717 | 0.243442–0.245468 | 17.18 | -1.0 |
| 2 | 4 | `tile8x32_column_major` | `iiijjjjj` | 19 | 2 | 0 | 0.244309 | 0.244312 | 0.000772 | 0.242989–0.245349 | 17.17 | -2.0 |
| 4.5 | 10 | `tile16_interleaved` | `jijijiji` | 19.75 | 8 | 0 | 0.295173 | 0.295114 | 0.000290 | 0.294573–0.295400 | 14.21 | -5.5 |
| 4.5 | 11 | `tile32_interleaved` | `jijijijiji` | 19.75 | 10 | 0 | 0.312400 | 0.311989 | 0.000570 | 0.311173–0.312520 | 13.43 | -6.5 |
| 7 | 12 | `tile16x8_row_major` | `jjjiiii` | 24.75 | 2 | 0 | 0.312989 | 0.312962 | 0.000438 | 0.312296–0.313589 | 13.40 | -5.0 |
| 7 | 13 | `tile32x8_row_major` | `jjjiiiii` | 24.75 | 2 | 0 | 0.316362 | 0.316353 | 0.000854 | 0.315321–0.317482 | 13.26 | -6.0 |
| 7 | 2 | `tile8_row_major` | `jjjiii` | 24.75 | 2 | 0 | 0.244122 | 0.244122 | 0.000394 | 0.243429–0.244602 | 17.18 | +5.0 |
| 10 | 14 | `tile32_column_major` | `iiiiijjjjj` | 30 | 2 | 0 | 0.327560 | 0.327046 | 0.001679 | 0.324867–0.328960 | 12.80 | -4.0 |
| 10 | 15 | `tile32x16_column_major` | `iiiiijjjj` | 30 | 2 | 0 | 0.330922 | 0.330671 | 0.001082 | 0.329295–0.332082 | 12.67 | -5.0 |
| 10 | 16 | `tile32x8_column_major` | `iiiiijjj` | 30 | 2 | 0 | 0.332189 | 0.331800 | 0.001233 | 0.330149–0.333629 | 12.63 | -6.0 |
| 13 | 6 | `tile16_column_major` | `iiiijjjj` | 31 | 2 | 0 | 0.282176 | 0.282466 | 0.001291 | 0.280989–0.284882 | 14.86 | +7.0 |
| 13 | 17 | `tile16x32_column_major` | `iiiijjjjj` | 31 | 2 | 0 | 0.338562 | 0.339189 | 0.001904 | 0.336696–0.341496 | 12.39 | -4.0 |
| 13 | 5 | `tile16x8_column_major` | `iiiijjj` | 31 | 2 | 0 | 0.280761 | 0.280980 | 0.000423 | 0.280535–0.281681 | 14.94 | +8.0 |
| 15 | 21 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 37 | 2 | 0 | 0.431041 | 0.430881 | 0.002193 | 0.427641–0.433975 | 9.73 | -6.0 |
| 17 | 7 | `tile16_row_major` | `jjjjiiii` | 46.75 | 2 | 0 | 0.282455 | 0.282305 | 0.001415 | 0.280455–0.284561 | 14.85 | +10.0 |
| 17 | 19 | `tile32x16_row_major` | `jjjjiiiii` | 46.75 | 2 | 0 | 0.426297 | 0.426329 | 0.002354 | 0.422844–0.429484 | 9.84 | -2.0 |
| 17 | 8 | `tile8x16_row_major` | `jjjjiii` | 46.75 | 2 | 0 | 0.286656 | 0.286038 | 0.002063 | 0.282696–0.288963 | 14.63 | +9.0 |
| 20 | 20 | `tile16x32_row_major` | `jjjjjiiii` | 62.75 | 2 | 0 | 0.429444 | 0.429262 | 0.001176 | 0.427817–0.431164 | 9.77 | +0.0 |
| 20 | 18 | `tile32_row_major` | `jjjjjiiiii` | 62.75 | 2 | 0 | 0.409109 | 0.408818 | 0.001453 | 0.406975–0.410469 | 10.25 | +2.0 |
| 20 | 9 | `tile8x32_row_major` | `jjjjjiii` | 62.75 | 2 | 0 | 0.291162 | 0.291071 | 0.001919 | 0.288402–0.293682 | 14.41 | +11.0 |
| 22 | 22 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 69.75 | 2 | 0 | 0.431442 | 0.431234 | 0.001440 | 0.428655–0.432682 | 9.72 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 4/22 | 0.182 | 4.682 | 11.000 |
| `peak-normalized-excess` | 4/22 | 0.182 | 3.909 | 11.500 |
| `weighted-normalized-excess` (selected) | 4/22 | 0.182 | 4.136 | 11.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.5 | 4 | `tile16x32_row_major` | `jjjjjiiii` | 8.29875 | 6 | 0 | 1.500441 | 1.527058 | 0.047914 | 1.476628–1.611349 | 1431.23 | -0.5 |
| 3.5 | 3 | `tile16x8_row_major` | `jjjiiii` | 8.29875 | 6 | 0 | 1.479598 | 1.481179 | 0.040568 | 1.420117–1.547945 | 1451.40 | +0.5 |
| 3.5 | 6 | `tile32_row_major` | `jjjjjiiiii` | 8.29875 | 6 | 0 | 1.502328 | 1.527725 | 0.055677 | 1.470861–1.622008 | 1429.44 | -2.5 |
| 3.5 | 2 | `tile32x8_row_major` | `jjjiiiii` | 8.29875 | 6 | 0 | 1.474665 | 1.485404 | 0.041822 | 1.442372–1.559080 | 1456.25 | +1.5 |
| 3.5 | 1 | `tile8_row_major` | `jjjiii` | 8.29875 | 6 | 0 | 1.470879 | 1.478437 | 0.042057 | 1.420892–1.551853 | 1460.00 | +2.5 |
| 3.5 | 5 | `tile8x32_row_major` | `jjjjjiii` | 8.29875 | 6 | 0 | 1.502222 | 1.512731 | 0.046112 | 1.475715–1.601662 | 1429.54 | -1.5 |
| 8.5 | 11 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 8.79875 | 6 | 0 | 1.542228 | 1.533806 | 0.048121 | 1.461774–1.602282 | 1392.46 | -2.5 |
| 8.5 | 8 | `tile16_row_major` | `jjjjiiii` | 8.79875 | 6 | 0 | 1.516713 | 1.519966 | 0.053702 | 1.426339–1.584047 | 1415.88 | +0.5 |
| 8.5 | 7 | `tile32x16_row_major` | `jjjjiiiii` | 8.79875 | 6 | 0 | 1.515885 | 1.514610 | 0.064313 | 1.413138–1.603965 | 1416.65 | +1.5 |
| 8.5 | 10 | `tile8x16_row_major` | `jjjjiii` | 8.79875 | 6 | 0 | 1.534253 | 1.530496 | 0.051328 | 1.440439–1.597880 | 1399.69 | -1.5 |
| 11.5 | 9 | `tile16_interleaved` | `jijijiji` | 15.695 | 24 | 0 | 1.531586 | 1.546013 | 0.032645 | 1.513640–1.605440 | 1402.13 | +2.5 |
| 11.5 | 12 | `tile32_interleaved` | `jijijijiji` | 15.695 | 30 | 0 | 1.789896 | 1.797234 | 0.019603 | 1.780616–1.834816 | 1199.78 | -0.5 |
| 14 | 13 | `tile8_column_major` | `iiijjj` | 55.8838 | 6 | 0 | 1.924484 | 1.973996 | 0.096457 | 1.921084–2.166579 | 1115.88 | +1.0 |
| 14 | 14 | `tile8x16_column_major` | `iiijjjj` | 55.8838 | 6 | 0 | 2.043809 | 2.023654 | 0.052272 | 1.920569–2.066262 | 1050.73 | +0.0 |
| 14 | 15 | `tile8x32_column_major` | `iiijjjjj` | 55.8838 | 6 | 0 | 2.170326 | 2.190654 | 0.090148 | 2.042871–2.287100 | 989.48 | -1.0 |
| 17 | 18 | `tile16_column_major` | `iiiijjjj` | 77.3838 | 6 | 0 | 3.164535 | 3.164623 | 0.000439 | 3.164001–3.165282 | 678.61 | -1.0 |
| 17 | 16 | `tile16x32_column_major` | `iiiijjjjj` | 77.3838 | 6 | 0 | 3.164165 | 3.164122 | 0.000192 | 3.163884–3.164391 | 678.69 | +1.0 |
| 17 | 17 | `tile16x8_column_major` | `iiiijjj` | 77.3838 | 6 | 0 | 3.164506 | 3.164498 | 0.000291 | 3.164026–3.164866 | 678.62 | +0.0 |
| 20 | 21 | `tile32_column_major` | `iiiiijjjjj` | 81.3838 | 6 | 0 | 3.166775 | 3.166772 | 0.000266 | 3.166401–3.167108 | 678.13 | -1.0 |
| 20 | 20 | `tile32x16_column_major` | `iiiiijjjj` | 81.3838 | 6 | 0 | 3.165083 | 3.165110 | 0.000355 | 3.164643–3.165736 | 678.49 | +0.0 |
| 20 | 19 | `tile32x8_column_major` | `iiiiijjj` | 81.3838 | 6 | 0 | 3.165010 | 3.164954 | 0.000307 | 3.164543–3.165436 | 678.51 | +1.0 |
| 22 | 22 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 89.3838 | 6 | 0 | 3.449688 | 3.453203 | 0.013007 | 3.437101–3.476128 | 622.52 | +0.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 19/22 | 0.864 | 0.091 | 1.000 |
| `peak-normalized-excess` | 9/22 | 0.409 | 1.750 | 10.500 |
| `weighted-normalized-excess` (selected) | 19/22 | 0.864 | 0.091 | 1.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | `tile16x8_row_major` | `jjjiiii` | 8 | 4 | 0 | 0.129881 | 0.129807 | 0.000274 | 0.129295–0.130121 | 32.32 | +1.0 |
| 2 | 2 | `tile32x8_row_major` | `jjjiiiii` | 8 | 4 | 0 | 0.132214 | 0.132230 | 0.000585 | 0.131561–0.133121 | 31.75 | +0.0 |
| 2 | 3 | `tile8_row_major` | `jjjiii` | 8 | 4 | 0 | 0.136107 | 0.136046 | 0.000470 | 0.135387–0.136734 | 30.84 | -1.0 |
| 4.5 | 4 | `tile16_interleaved` | `jijijiji` | 10 | 16 | 0 | 0.141481 | 0.141505 | 0.000499 | 0.140921–0.142281 | 29.67 | +0.5 |
| 4.5 | 9 | `tile32_interleaved` | `jijijijiji` | 10 | 20 | 0 | 0.160121 | 0.160145 | 0.000099 | 0.160054–0.160321 | 26.21 | -4.5 |
| 7 | 10 | `tile8_column_major` | `iiijjj` | 14 | 4 | 0 | 0.163161 | 0.163313 | 0.001438 | 0.161521–0.165321 | 25.73 | -3.0 |
| 7 | 7 | `tile8x16_column_major` | `iiijjjj` | 14 | 4 | 0 | 0.158868 | 0.158710 | 0.000414 | 0.157921–0.159121 | 26.42 | +0.0 |
| 7 | 11 | `tile8x32_column_major` | `iiijjjjj` | 14 | 4 | 0 | 0.172521 | 0.172657 | 0.000572 | 0.172067–0.173548 | 24.33 | -4.0 |
| 10 | 6 | `tile16_row_major` | `jjjjiiii` | 15 | 4 | 0 | 0.154868 | 0.154881 | 0.000247 | 0.154455–0.155175 | 27.10 | +4.0 |
| 10 | 5 | `tile32x16_row_major` | `jjjjiiiii` | 15 | 4 | 0 | 0.154775 | 0.154807 | 0.000173 | 0.154601–0.155121 | 27.12 | +5.0 |
| 10 | 8 | `tile8x16_row_major` | `jjjjiii` | 15 | 4 | 0 | 0.159800 | 0.159838 | 0.000377 | 0.159320–0.160494 | 26.27 | +2.0 |
| 13 | 12 | `tile32_column_major` | `iiiiijjjjj` | 16 | 4 | 0 | 0.177388 | 0.177538 | 0.000610 | 0.177042–0.178722 | 23.66 | +1.0 |
| 13 | 13 | `tile32x16_column_major` | `iiiiijjjj` | 16 | 4 | 0 | 0.182227 | 0.182059 | 0.000574 | 0.181227–0.182694 | 23.03 | +0.0 |
| 13 | 14 | `tile32x8_column_major` | `iiiiijjj` | 16 | 4 | 0 | 0.183894 | 0.184046 | 0.001139 | 0.182840–0.186174 | 22.83 | -1.0 |
| 16 | 16 | `tile16_column_major` | `iiiijjjj` | 18 | 4 | 0 | 0.190762 | 0.190380 | 0.001726 | 0.188095–0.192842 | 22.00 | +0.0 |
| 16 | 15 | `tile16x32_column_major` | `iiiijjjjj` | 18 | 4 | 0 | 0.189242 | 0.189428 | 0.001273 | 0.187322–0.191122 | 22.18 | +1.0 |
| 16 | 17 | `tile16x8_column_major` | `iiiijjj` | 18 | 4 | 0 | 0.194854 | 0.194070 | 0.001690 | 0.191401–0.195974 | 21.54 | -1.0 |
| 19 | 18 | `tile16x32_row_major` | `jjjjjiiii` | 31 | 4 | 0 | 0.211309 | 0.211426 | 0.000891 | 0.210375–0.212869 | 19.86 | +1.0 |
| 19 | 20 | `tile32_row_major` | `jjjjjiiiii` | 31 | 4 | 0 | 0.213402 | 0.213498 | 0.000474 | 0.212882–0.214042 | 19.67 | -1.0 |
| 19 | 19 | `tile8x32_row_major` | `jjjjjiii` | 31 | 4 | 0 | 0.212148 | 0.212212 | 0.000307 | 0.211788–0.212668 | 19.79 | +0.0 |
| 21 | 22 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 43 | 4 | 0 | 0.298963 | 0.298672 | 0.000767 | 0.297189–0.299403 | 14.04 | -1.0 |
| 22 | 21 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 63 | 4 | 0 | 0.266695 | 0.264639 | 0.003878 | 0.258602–0.268949 | 15.74 | +1.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 4/22 | 0.182 | 2.955 | 10.000 |
| `peak-normalized-excess` | 1/22 | 0.045 | 2.955 | 8.500 |
| `weighted-normalized-excess` (selected) | 8/22 | 0.364 | 1.227 | 4.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.5 | 1 | `tile8_column_major` | `iiijjj` | 3.6875 | 2 | 0 | 0.155308 | 0.155327 | 0.000736 | 0.154095–0.156375 | 27.05 | +1.5 |
| 2.5 | 4 | `tile8_row_major` | `jjjiii` | 3.6875 | 2 | 0 | 0.162828 | 0.163025 | 0.000279 | 0.162761–0.163468 | 25.80 | -1.5 |
| 2.5 | 2 | `tile8x16_column_major` | `iiijjjj` | 3.6875 | 2 | 0 | 0.155761 | 0.155897 | 0.000527 | 0.155055–0.156481 | 26.97 | +0.5 |
| 2.5 | 7 | `tile8x32_column_major` | `iiijjjjj` | 3.6875 | 2 | 0 | 0.176948 | 0.177327 | 0.001313 | 0.176108–0.179828 | 23.74 | -4.5 |
| 5 | 3 | `tile16_interleaved` | `jijijiji` | 3.75 | 8 | 0 | 0.161441 | 0.161473 | 0.000288 | 0.161081–0.161908 | 26.02 | +2.0 |
| 6 | 10 | `tile32_interleaved` | `jijijijiji` | 3.8125 | 10 | 0 | 0.217495 | 0.217055 | 0.000622 | 0.216188–0.217655 | 19.31 | -4.0 |
| 7 | 5 | `tile16x8_row_major` | `jjjiiii` | 4 | 2 | 0 | 0.165347 | 0.165065 | 0.000605 | 0.163987–0.165707 | 25.40 | +2.0 |
| 8 | 11 | `tile32x8_row_major` | `jjjiiiii` | 4.1875 | 2 | 0 | 0.233522 | 0.233653 | 0.000537 | 0.232842–0.234402 | 17.99 | -3.0 |
| 9 | 6 | `tile8x16_row_major` | `jjjjiii` | 4.6875 | 2 | 0 | 0.173455 | 0.177612 | 0.008983 | 0.172148–0.195548 | 24.22 | +3.0 |
| 10 | 19 | `tile16_row_major` | `jjjjiiii` | 4.75 | 2 | 0 | 0.334522 | 0.333887 | 0.001088 | 0.332415–0.335148 | 12.56 | -9.0 |
| 11 | 18 | `tile32x16_row_major` | `jjjjiiiii` | 4.9375 | 2 | 0 | 0.329883 | 0.329403 | 0.001063 | 0.327336–0.330309 | 12.73 | -7.0 |
| 13 | 14 | `tile16_column_major` | `iiiijjjj` | 5 | 2 | 0 | 0.260067 | 0.259686 | 0.002464 | 0.256000–0.262747 | 16.15 | -1.0 |
| 13 | 8 | `tile16x32_column_major` | `iiiijjjjj` | 5 | 2 | 0 | 0.197294 | 0.197123 | 0.000730 | 0.196014–0.198067 | 21.29 | +5.0 |
| 13 | 13 | `tile16x8_column_major` | `iiiijjj` | 5 | 2 | 0 | 0.254654 | 0.254636 | 0.002194 | 0.251774–0.258028 | 16.49 | +0.0 |
| 15 | 9 | `tile8x32_row_major` | `jjjjjiii` | 8.0625 | 2 | 0 | 0.204907 | 0.204561 | 0.000709 | 0.203574–0.205321 | 20.50 | +6.0 |
| 16 | 20 | `tile16x32_row_major` | `jjjjjiiii` | 8.125 | 2 | 0 | 0.341908 | 0.342337 | 0.000875 | 0.341374–0.343668 | 12.29 | -4.0 |
| 17 | 17 | `tile32_row_major` | `jjjjjiiiii` | 8.1875 | 2 | 0 | 0.328214 | 0.327619 | 0.001611 | 0.325147–0.329587 | 12.80 | +0.0 |
| 19 | 12 | `tile32_column_major` | `iiiiijjjjj` | 9.1875 | 2 | 0 | 0.244416 | 0.244776 | 0.000590 | 0.244229–0.245736 | 17.19 | +7.0 |
| 19 | 16 | `tile32x16_column_major` | `iiiiijjjj` | 9.1875 | 2 | 0 | 0.264109 | 0.264085 | 0.001686 | 0.261456–0.266763 | 15.90 | +3.0 |
| 19 | 15 | `tile32x8_column_major` | `iiiiijjj` | 9.1875 | 2 | 0 | 0.261442 | 0.262311 | 0.002632 | 0.258695–0.266509 | 16.07 | +4.0 |
| 21 | 22 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 15.75 | 2 | 0 | 0.371387 | 0.371339 | 0.000539 | 0.370694–0.372268 | 11.31 | -1.0 |
| 22 | 21 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 27.5625 | 2 | 0 | 0.369683 | 0.368742 | 0.001262 | 0.366790–0.369817 | 11.36 | +1.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 3/22 | 0.136 | 2.909 | 9.000 |
| `peak-normalized-excess` | 1/22 | 0.045 | 3.023 | 8.500 |
| `weighted-normalized-excess` (selected) | 3/22 | 0.136 | 2.909 | 9.000 |

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

| Score rank | Runtime rank | Layout | Word (low→high) | Score | Runs | XORs | Median ms | Mean ms | SD ms | Observed range ms | GFLOP/s | Rank delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 3 | `tile8_column_major` | `iiijjj` | 1.75468 | 4 | 0 | 1.451772 | 1.464145 | 0.034972 | 1.426998–1.526799 | 1479.22 | -1.0 |
| 2 | 2 | `tile8x16_column_major` | `iiijjjj` | 1.75468 | 4 | 0 | 1.447306 | 1.466060 | 0.037485 | 1.436252–1.537573 | 1483.78 | +0.0 |
| 2 | 1 | `tile8x32_column_major` | `iiijjjjj` | 1.75468 | 4 | 0 | 1.446287 | 1.462303 | 0.027305 | 1.439980–1.512781 | 1484.83 | +1.0 |
| 6.5 | 10 | `tile16_column_major` | `iiiijjjj` | 2.75468 | 4 | 0 | 1.537720 | 1.524485 | 0.050581 | 1.468226–1.604254 | 1396.54 | -3.5 |
| 6.5 | 7 | `tile16x32_column_major` | `iiiijjjjj` | 2.75468 | 4 | 0 | 1.511713 | 1.514502 | 0.029814 | 1.471019–1.564380 | 1420.56 | -0.5 |
| 6.5 | 9 | `tile16x8_column_major` | `iiiijjj` | 2.75468 | 4 | 0 | 1.517269 | 1.517819 | 0.037958 | 1.469015–1.578683 | 1415.36 | -2.5 |
| 6.5 | 8 | `tile32_column_major` | `iiiiijjjjj` | 2.75468 | 4 | 0 | 1.512161 | 1.512030 | 0.045588 | 1.447640–1.588842 | 1420.14 | -1.5 |
| 6.5 | 6 | `tile32x16_column_major` | `iiiiijjjj` | 2.75468 | 4 | 0 | 1.509210 | 1.518456 | 0.047729 | 1.455597–1.591797 | 1422.92 | +0.5 |
| 6.5 | 11 | `tile32x8_column_major` | `iiiiijjj` | 2.75468 | 4 | 0 | 1.539654 | 1.530784 | 0.038644 | 1.485866–1.587201 | 1394.78 | -4.5 |
| 10.5 | 5 | `tile16_interleaved` | `jijijiji` | 15.8338 | 16 | 0 | 1.491786 | 1.504823 | 0.036892 | 1.468146–1.570853 | 1439.54 | +5.5 |
| 10.5 | 4 | `tile32_interleaved` | `jijijijiji` | 15.8338 | 20 | 0 | 1.483428 | 1.500725 | 0.044813 | 1.468002–1.588136 | 1447.65 | +6.5 |
| 12 | 12 | `column_major` | `iiiiiiiiiijjjjjjjjjj` | 33.7547 | 4 | 0 | 1.548802 | 1.529941 | 0.046756 | 1.461708–1.593736 | 1386.54 | +0.0 |
| 14 | 14 | `tile16x8_row_major` | `jjjiiii` | 35.7753 | 4 | 0 | 1.918150 | 1.967366 | 0.098966 | 1.917336–2.165297 | 1119.56 | +0.0 |
| 14 | 15 | `tile32x8_row_major` | `jjjiiiii` | 35.7753 | 4 | 0 | 1.928400 | 1.967425 | 0.058310 | 1.915734–2.038908 | 1113.61 | -1.0 |
| 14 | 13 | `tile8_row_major` | `jjjiii` | 35.7753 | 4 | 0 | 1.915254 | 1.917547 | 0.004824 | 1.914854–1.927187 | 1121.25 | +1.0 |
| 19 | 16 | `row_major` | `jjjjjjjjjjiiiiiiiiii` | 37.7831 | 4 | 0 | 3.284211 | 3.284870 | 0.008622 | 3.270318–3.293758 | 653.88 | +3.0 |
| 19 | 22 | `tile16_row_major` | `jjjjiiii` | 37.7831 | 4 | 0 | 3.783623 | 3.703076 | 0.165119 | 3.372914–3.790277 | 567.57 | -3.0 |
| 19 | 19 | `tile16x32_row_major` | `jjjjjiiii` | 37.7831 | 4 | 0 | 3.373616 | 3.373750 | 0.131741 | 3.165576–3.582177 | 636.55 | +0.0 |
| 19 | 18 | `tile32_row_major` | `jjjjjiiiii` | 37.7831 | 4 | 0 | 3.367745 | 3.447964 | 0.100797 | 3.362225–3.574253 | 637.66 | +1.0 |
| 19 | 21 | `tile32x16_row_major` | `jjjjiiiii` | 37.7831 | 4 | 0 | 3.781051 | 3.781227 | 0.001261 | 3.779838–3.783318 | 567.96 | -2.0 |
| 19 | 20 | `tile8x16_row_major` | `jjjjiii` | 37.7831 | 4 | 0 | 3.779926 | 3.780195 | 0.000537 | 3.779686–3.781179 | 568.13 | -1.0 |
| 19 | 17 | `tile8x32_row_major` | `jjjjjiii` | 37.7831 | 4 | 0 | 3.365267 | 3.365954 | 0.130221 | 3.160891–3.572681 | 638.13 | +2.0 |

### Variation-aware metrics

| Score mode | Ranks within variation | Accuracy | Mean rank error | Max rank error |
| --- | --- | --- | --- | --- |
| `weighted-region-count` | 19/22 | 0.864 | 0.136 | 1.000 |
| `peak-normalized-excess` | 16/22 | 0.727 | 1.455 | 10.000 |
| `weighted-normalized-excess` (selected) | 19/22 | 0.864 | 0.136 | 1.000 |
