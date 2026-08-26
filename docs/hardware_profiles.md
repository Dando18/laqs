# Universal access scopes and hardware profiles

RELAY separates the kernel's locality signature from the hardware response.
Kernel modules describe matrices, dynamic memory events, and schedule facts;
they do not choose objective names, byte scales, or weights. The universal
builder turns those facts into scale-free `EdgeFamily` objects, and a selected
`HardwareProfile` evaluates every realized family over one physical byte-scale
ladder.

For kernel instance `K`, layout `A`, access scope `s`, and byte scale `b`, the
hardware-area feature is

```text
x[s,b](K,A) = b * (Q[s,b](K,A) - LB[s,b](K)) / B[K]
```

where `B[K]` is the total dynamic useful bytes requested from target arrays.
This retains dynamic scope exposure while removing unavoidable packing cost.
The device score is

```text
J_area = sum(tau[s,b] * x[s,b])
J_peak = max(e[s,b] / kappa[s,b])
```

`tau` and `kappa` belong to the hardware profile. `e` is the existing relative
normalized excess and remains useful for diagnostics and peak constraints.

The profile may also contain score-only `ResourceMap` values. Each map defines
a transaction width, sparse byte-address parity masks, a cross-allocation
cohort family, and an allocation-phase policy. These maps produce the separate
placement coordinate `J_place`; they do not alter the low-address objectives
or the current layout search.

## Universal v1 schema

The v1 builder uses the same typed scope grammar for every event trace:

- `issue.g{8,16,32,64}.stream.{load,store,atomic}`;
- `lane_window.t{4,16}.{stream,array}.{load,store,atomic}`;
- `simd_window.t{4,16}.{stream,array}.{load,store,atomic}`;
- `workgroup_step.{stream,array}.{load,store,atomic}`;
- `workgroup_window.t{4,16}.{stream,array}.{load,store,atomic}`; and
- `phase.{lane,simd,workgroup}.{stream,array}.{load,store,atomic}`.

Only families with events in a particular kernel are materialized, but the
schema and construction rules are global. `stream` keeps static access sites
separate; `array` joins sites accessing the same allocation within the same
cohort and interval. Every realized scale-free family is crossed with the
profile's complete byte ladder. Affine cosets and non-affine sets of at most
256 points are compressed exactly under XOR translation, with their dynamic
multiplicity retained in the representative edge weight. Larger non-affine
working sets stay explicit so canonicalization cannot introduce quadratic
preprocessing.

The event trace includes the compiled execution schedule—lane identities,
SIMD/workgroup membership, steps, and phase boundaries. This is kernel/launch
input to the device-independent builder, not a source of objective weights.
The current five traces describe the 64-lane MI300A launch explicitly. Before
adding an H100 response, the same kernel modules must accept a 32-lane launch
schedule (or consume compiler-provided execution metadata); merely selecting a
new profile would not transform an existing 64-lane trace.

## MI300A proof-of-concept profile

`mi300a-gfx942-universal-v1-poc` uses the byte ladder from 16 B through 32 KiB
in powers of two. Its fitted area response has five empirically distinct
coordinates, represented by ten nonzero cells because proportional
stream/array coordinates split each fitted response equally:

The score-only resource model adds `hbm_stack_interleave_sketch`: eight colors
selected by byte-address bits 12--14, 64-byte deduplicated transactions, the
`simd_window.t4.cohort.load` family, and robust relative allocation phases.
AMD documents eight HBM stacks and a 4 KiB stack switch; inaccessible physical
page placement means this remains a sketch rather than an exact virtual-address
channel map.

| Area scope-scale response | tau per partition |
| --- | ---: |
| `lane_window.t4.{stream,array}.load.64B` | `0.00924` |
| `lane_window.t16.{stream,array}.load.256B` | `0.00157` |
| `simd_window.t4.{stream,array}.load.128B` | `0.018` |
| `simd_window.t4.{stream,array}.load.512B` | `1` |
| `simd_window.t4.{stream,array}.load.2048B` | `0.249` |

The independent peak support is:

| Peak scope-scale cell | kappa |
| --- | ---: |
| `issue.g8.stream.load.16B` | `1` |
| `issue.g16.stream.load.512B` | `15` |
| `issue.g64.stream.load.4096B` | `63` |
| `lane_window.t4.stream.load.64B` | `3` |
| `lane_window.t16.stream.load.256B` | `15` |
| `simd_window.t4.stream.load.128B` | `3` |

The checked tau is a hand-reviewed, solver-compatible feasible variant from
seeded sparse nonnegative fits; a common positive rescaling produces the same
frontier. The fit is underdetermined: the calibration command below also
finds passing supports (for example a phase-lane-256 response can replace the
checked lane-window-16 response), so the table is not presented as a unique
regression optimum. These are provisional empirical responses, not a claim
that they identify MI300A transaction or cache sizes. They were selected using
only ATAX and GESUMMV at `N=256` and `N=512`; the other kernels and sizes are a
provisional transfer check rather than additional fitting data.

SIMD-window-16 cells were left inactive because the correct global-slot
windows are not affine for GESUMMV and therefore are not currently supported
by the affine search grammar.
Consequently this first table is a solver-compatible candidate-generation
profile, not a pure unconstrained identification of every MI300A response.
Supporting arbitrary edge sets in the affine search cost is the clean fix;
the hardware table should then be recalibrated without that restriction.

On the 73-layout canonical corpus, the four calibration frontiers retain
`5, 5, 7, 7` candidates (8.22% on average). Both ATAX instances contain their
exact measured winner. GESUMMV regret is 0.966% and 0.737%, so all four are
within 1%; mean regret is 0.426%. The weight-free full universal basis contains
all four exact winners. Thus the remaining GESUMMV gap is a compression issue,
not evidence that these winners are dominated by the scope representation.

The complete calibration, unfitted-kernel transfer, and solver-search numbers
are collected in the
[MI300A proof-of-concept scorecard](../results/edge_construction_mi300a_poc.md).

Reproduce that timing-free rescore from the existing MI300A measurements:

```bash
.venv/bin/python experiments/layout_ranking.py \
  --kernel atax --kernel gesummv \
  --size 256 --size 512 \
  --samples 5 --iterations 3 --warmup 2 \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --hardware-profile mi300a \
  --tau-perturbation-trials 16 \
  --reuse-timings results/layout_ranking.json \
  --output results/edge_construction_mi300a_poc.json
```

No GPU is used by this command: it reconstructs objectives and attaches the
matching stored timings. The report includes aggregate, active-component,
full-basis, and dense-scale frontiers plus global profile perturbations.

The fitter is also available as a separate diagnostic. It reproduces the
frontier/regret constraints, but because several sparse supports are feasible
it is not a bit-for-bit generator for the reviewed table above. `--max-regret`
is a fraction, so `0.01` means one percent:

```bash
.venv/bin/python experiments/hardware_profile_calibration.py \
  results/edge_construction_mi300a_poc.json \
  --kernel atax --kernel gesummv \
  --size 256 --size 512 \
  --min-candidates 5 --max-candidates 7 \
  --max-regret 0.01 --search-support 8 \
  --seed 7 --iterations 20000 \
  --output results/edge_construction_mi300a_fit.json
```

The fitter reconstructs and validates `x` from `Q`, `LB`, byte scale, and
`B_K`; normalizes and deduplicates proportional coordinates; searches a shared
nonnegative response; and reports per-instance candidate/regret metrics and
full-basis winner-dominance certificates. It emits a recommendation only—it
never rewrites the checked-in profile. Solver applicability and transfer-set
results still need human review before a recommendation is promoted.

## Solver-frontier probe

With the frozen profile and an exact zero-slack fine gate, the standard and
affine solvers each complete exactly at `N=256` and `N=512`. Their raw union is
seven mappings per instance. Cross-grammar Pareto filtering leaves five ATAX
mappings and six GESUMMV mappings. The best measured union candidates improve
over the row-major baseline by 1.40x and 1.41x for ATAX, and 1.60x and 1.51x
for GESUMMV.

This probe also exposed a representation limit. Before the complete-sequence
contract corrected SIMD-window boundaries, the exploration measured a
GESUMMV independent-operand mapping that remains 13.0% and 13.4% faster than
the best candidate generated by the corrected model. That mapping is
dominated even in the corrected full universal additive feature basis, so no
nonnegative tau can recover it. The likely missing signal is a cross-array or
physical-address interaction (for example channel/set balance), not another
round of kernel-specific weight tweaking. These solver timings are therefore
a bounded proof of concept, not evidence of full-space regret below one
percent.

## Interpreting misses

Always inspect the full universal component frontier before changing `tau`.
If a measured winner is dominated there, no nonnegative profile can recover
it and the representation needs another justified scope or orthogonal feature.
If it is nondominated but absent from the compact frontier, improve the global
fit, retain another Pareto layer, or use a profile uncertainty ensemble.

The current calibration is deliberately small. A release-quality profile
should use independent operand layouts, hardware counters and targeted
microbenchmarks, then report leave-one-kernel-family and leave-one-size-out
frontier recall/regret with confidence intervals.
