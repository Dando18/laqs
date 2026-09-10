# Manual realization of frozen LAQS layouts

Manual LAQS storage gives a substantial row/column speedup: approximately 1.94×
at the original schedule and 1.55× against the independently tuned ordinary
baseline. The repaired automatic implementation is close. In the sum
diagnostic, manual indexing reproduces the automatic implementation's losses.
These cases support useful layout optimization, but do not support generic
address rewriting as the main explanation for the losing sum layouts.

## Implementation and measurement

The existing `triton/packet_layout/conventional.py` comparison now includes
explicit-source realizations of the exact saved split maps. Row/column uses its
existing blocked-storage address function. Sum and GEMM have source-specialized
copies in `packet_kernels.py`, changing pointer initialization and recurrence
while retaining the original arithmetic, masks and output semantics. Explicit
variants execute without the LAQS rewrite hook.

`manual_layout.py` recovers tile dimensions by matching the saved address matrix
rows, independently of candidate names. Non-split maps are rejected. Packing
uses reshape/permute, checks every packed element against the generic bit-matrix
packer, and checks a complete unpacking round trip. The automatic and explicit
versions share the same prepared bytes.

Each timing process measures all variants at three shared input addresses.
Complete outputs are validated before timing and at every captured placement.
Copies, warmups, packing and compilation are outside kernel timing. The report
averages log ratios within each placement and then across processes. Confidence
intervals use processes as replicates. One-use conversion costs include
synchronous packing and allocation wall time and are reported separately.

Row/column receives both a fixed-schedule comparison and the same six-schedule
budget for ordinary, column-major, tile16, explicit LAQS and automatic LAQS.
Fresh processes evaluate the frozen schedule winners. Sum and GEMM are
fixed-schedule diagnostic controls, with a manual ordinary-layout kernel as an
additional control. The historical FP64 HIP MVT/GESUMMV matched port is outside
this implementation; FP32 row/column is not presented as that port.

Validation: 56 CPU-discovered packet tests passed or skipped (15 GPU/environment
skips). A separate MI300A run passed packing and complete-output checks for all
three manual kernels, including tiled-pointer recurrence and reconstruction
cases. Every submitted allocation requested one task, one GPU and at most five
minutes.

## MI300A row/column: the earlier positive result survives stronger controls

Input: 1024×1024 FP32, computing `A @ x` and `A.T @ z` together. Fixed schedule:
`BLOCK_M=4`, `BLOCK_K=128`, four warps. The saved reference graph and large map
come from `packet-debug-20260908-215841`; the reference protocol also selects a
smaller equal-objective representative. The large storage tile is 1024×8; the
smaller tile is 4×8. Neither map changes during schedule tuning.

| Realization | Fixed-schedule speedup | Held-out speedup vs tuned ordinary | Held-out 95% interval |
| --- | ---: | ---: | ---: |
| Explicit large LAQS map | 1.9348× | 1.5115× | 1.4758–1.5480 |
| Repaired automatic large map | 1.9318× | 1.4674× | 1.4418–1.4934 |
| Explicit smaller LAQS map | 1.9430× | 1.5507× | 1.5074–1.5953 |
| Repaired automatic smaller map | 1.9173× | 1.5685× | 1.5353–1.6024 |
| Explicit conventional 16×16 tile | 1.8555× | 1.4491× | 1.4198–1.4790 |
| Explicit column-major | 1.0264× | 0.9886× | 0.9715–1.0059 |

Every representation received the same six schedules. Ordinary, column-major
and tile16 selected 16×128 work tiles; both LAQS maps and realizations selected
4×64 work tiles. Each comparison phase has three independent timing processes,
each with three placements, on `tuolumne1032`, GPU 0 with the same recorded UUID.
Fixed-phase identity/same-pointer deviations were at most 1.68%/0.96%; held-out
deviations were at most 2.03%/0.47%. Shared input addresses and saved executable
hashes were independently verified from the worker artifacts.

The fixed-schedule automatic/explicit runtime ratios are 1.0015× for large and
1.0134× for smaller. Per-site ownership contracts and memory primitives match.
The manual realization therefore preserves the useful storage benefit; it
does not reveal a large fixed-schedule compiler penalty. Tuned large shows a
modest 3.0% manual advantage, while tuned smaller slightly favors automatic.
These are much smaller effects than the benefit of changing storage.

The explicitly indexed smaller map beats the fixed conventional tile baseline
by 1.0701× in a direct paired held-out comparison (95% interval 1.0615–1.0789).
This is a comparison with the declared 16×16 storage tile and schedule budget,
not a claim of superiority over all conventional blocked layouts. The LAQS
split maps are themselves rectangular blocked layouts.

Packing dominates one-use cost: the smaller explicit map's conversion-inclusive
speedup is 0.0761× at the fixed schedule and 0.0553× after tuning. Its measured
break-even counts range from 23–30 uses at the fixed schedule and 41–58 uses
against tuned ordinary. These are warm synchronous allocation/packing estimates,
not a measured producer-to-consumer pipeline.

All variants pass complete numerical validation. Optional ownership inspection
is unavailable for column-major at the fixed schedule; its source kernel remains
included and correct. This is recorded separately from automatic primitive
admission. The final run does not count raw IR alias/snippet differences as
changed ownership.

Full data: [row/column report](../triton/experiments/results/manual-row-frozen-debug-v2/tuolumne/row_column--small/expert/conventional.md),
[JSON](../triton/experiments/results/manual-row-frozen-debug-v2/tuolumne/row_column--small/expert/conventional.json).

## MI300A sum: manual and repaired automatic agree

Input: 4096×1024 FP32, reduce dimension 1, 16×16 work tiles, two warps.
The reference graph and large map come from `packet-debug-split`; the existing
address-reference protocol supplies the smaller equal-objective representative.
Maps are frozen before any timings. Three independent processes ran on
`tuolumne1031`, GPU index 0 with the same recorded device UUID.

| Storage map | Automatic speedup | Explicit-source speedup | Explicit 95% interval | Automatic / explicit runtime |
| --- | ---: | ---: | ---: | ---: |
| Large, 4096×4 storage tile | 0.6726× | 0.6722× | 0.6318–0.7151 | 0.9994× |
| Smaller, 2×16 storage tile | 0.8768× | 0.8777× | 0.8568–0.8991 | 1.0011× |
| Ordinary source-copy control | — | 0.9996× | 0.9968–1.0023 | — |

Both changed explicit maps lose at every matched placement, and both
process-level intervals lie below parity. The automatic/explicit intervals are
0.99938–0.99946 for large and 0.99754–1.00471 for smaller. These results provide
no evidence of a useful remaining automatic-realization gap. The extremely
narrow large-map interval describes this three-process run, not reproducibility
across GPUs or days; the controls bound the practical precision more loosely.

Ordinary, manual ordinary, repaired automatic and explicit changed kernels all
use 11 registers, zero spills, 64 shared bytes, one `buffer_load_dwordx2` and one
`buffer_store_dword` in the static assembly. Inspected load ownership, vector
width and mask/native alignment contracts match between explicit and automatic
changed kernels. The separate raw IR text comparison flags differences:
that record includes repeated IR snippets and alias names, so its inequality
alone is **not evidence of changed lane ownership or memory primitives**. Consult
the per-site contracts and saved assembly before interpreting that flag.

Maximum identity-output deviation was 1.40%; maximum same-pointer deviation was
0.83%. All measured placements used the same input address across variants;
saved executable hashes were independently verified. Conversion-inclusive
speedups are only 0.0530× for large and 0.1526× for smaller;
neither has a positive kernel-time saving to amortize conversion.

Full data: [sum report](../triton/experiments/results/manual-sum-frozen-debug-v2/tuolumne/sum--small/expert/conventional.md),
[JSON](../triton/experiments/results/manual-sum-frozen-debug-v2/tuolumne/sum--small/expert/conventional.json).
Workers and actual graph-captured binaries are in `conventional-fixed/` beside
those reports.

## H100 GEMM: a small realization gap, but the large map still fails

The user completed the Matrix run. All variants passed numerical/packing
validation, with no excluded variants. Three timing processes ran on
`matrix10`, the same H100 80GB HBM3 device UUID, with three matched placements
per process. Input addresses matched across every variant at every placement;
saved cubin, PTX and SASS hashes were independently verified. No new GPU jobs
or counter collections were run for this analysis.

The operation is FP16 GEMM with M=512, N=4096, K=4096. The fixed launch uses
64×128×32 work tiles, four warps, four stages and GROUP_M=8. Only A is repacked:
the large storage tile is 512×8, and the smaller tile is 2×32. Both retain the
same saved objective value. There is no equal-budget GEMM schedule sweep here.

| Storage | Repaired automatic speedup | Manual speedup | Manual 95% interval | Automatic / manual runtime |
| --- | ---: | ---: | ---: | ---: |
| Large | 0.4776× | 0.4785× | 0.4750–0.4820 | 1.0019× |
| Smaller | 0.9760× | 1.0109× | 1.0010–1.0210 | 1.0358× |
| Manual ordinary control | — | 0.9912× | 0.9823–1.0002 | — |

The large map remains approximately 2.09 times slower than ordinary. Its manual
and automatic realizations agree: the direct ratio's interval is
0.9939–1.0099. This strongly reinforces the earlier copy-probe diagnosis that
the bad map is not rescued by straightforward address-code changes. The manual
kernel retains the asynchronous-copy path; it does not evade that diagnostic
by switching to synchronous copies. The earlier probe measured sector excess;
this run contains no new counters, so it does not independently remeasure that
mechanism in the manual full GEMM.

The smaller map does reveal a modest, repeatable implementation opportunity:
manual is faster than automatic at all nine placements, with process ratios
1.0362, 1.0340 and 1.0373. The paired ratio is 1.0358× with interval
1.0316–1.0401. This is stronger evidence than the manual kernel's net 1.1%
advantage over ordinary. That net gain is within the scale of the observed
identity-output and same-pointer deviations (2.43% and 1.60%), and manual loses
to ordinary in two of nine placements. Its interval narrowly clears one, but
these controls and three processes on one GPU do not establish a robust
deployment win. Relative to the manual ordinary source-copy control, the
smaller map improves runtime by 1.0199× (interval 1.0170–1.0228); that is a
separate comparison, not a replacement for the stronger original baseline.

Inspection finds matching per-site ownership and memory/matrix primitive
signatures. Actual binaries differ. For smaller, automatic has 610 static SASS
instructions and 96 reported registers; manual has 602 instructions and 118
registers. Both have zero spills and 49,152 shared bytes. Both contain 24 static
`LDGSTS.E.BYPASS.128`, eight `LDSM.16.M88.4`, eight `STG.E.128`, and two
`HGMMA.64x128x16.F32` instructions. PTX integer-address operations differ as well.
These observations make a narrow codegen investigation worthwhile, but do not
isolate its cause: static counts and signatures do not prove equal scheduling,
dependency chains, shared-memory mappings or dynamic service costs. The manual
kernel's zero `physical_recurrences` metadata count means it lacks the plugin's
annotation, not that its source pointer recurrence is absent.

Packing costs approximately 219 µs for large and 63.7 µs for smaller. One-use
conversion-inclusive speedups are 0.1296× and 0.3759× respectively. Smaller's
estimated break-even counts range from 55–351 uses in winning placements, with
no positive break-even in the two losing placements. There is no compelling
conversion-inclusive GEMM win in this experiment.

Across all three kernels, the main pattern is now clear: row/column benefits
substantially and survives automatic realization; sum's maps lose even with
manual realization; large GEMM loses outside the automatic backend, while small
GEMM offers a few-percent realization improvement and approximately parity with
ordinary storage. Equal-objective GEMM maps differ by about 2.11× in manual
runtime. The current locality objective and compatibility constraints therefore
remain insufficient for choosing profitable maps, even when address realization
is straightforward.

The next compiler experiment should be narrow: compare the smaller automatic
and manual pointer setup, scheduling and service counters at the same launch.
The large-map result supports retaining the previously proposed source-sector
compatibility restriction rather than attempting another generic address
rewrite. For positive end-to-end results, prioritize row/column and persistent
or producer-generated packed inputs; keep ordinary storage as the measured
fallback for this GEMM.

Full data: [Matrix report](../triton/experiments/results/manual-gemm-frozen-debug/matrix/gemm--asymmetric/expert/conventional.md),
[JSON](../triton/experiments/results/manual-gemm-frozen-debug/matrix/gemm--asymmetric/expert/conventional.json).

## Reproduce or continue in the debug queue

Run from the repository root. Each script uses a 4.5-minute internal budget and
retains completed stages and timing processes. Repeat an identical command to
resume; the completed sum command below verifies and reuses its checkpoints.

```bash
flux run -n1 -g1 -c8 -t 5m -q pdebug \
  bash triton/packet_layout/run-manual-tuolumne.bash \
  --root triton/experiments/results/manual-sum-frozen-debug-v2 \
  --cases sum--small \
  --address-reference triton/experiments/results/packet-debug-split

flux run -n1 -g1 -c8 -t 5m -q pdebug \
  bash triton/packet_layout/run-manual-tuolumne.bash \
  --root triton/experiments/results/manual-row-frozen-debug-v2 \
  --cases row_column--small \
  --address-reference triton/experiments/results/packet-debug-20260908-215841

srun -n1 -G1 -c8 -p pdebug -t 00:05:00 \
  bash triton/packet_layout/run-manual-matrix.bash \
  --root triton/experiments/results/manual-gemm-frozen-debug \
  --cases gemm--asymmetric \
  --address-reference triton/experiments/results/packet-debug-split
```

The wrappers without overrides use a fresh `manual-layout-debug` root and run
graph construction for both row/column sizes plus sum on Tuolumne or GEMM on
Matrix. The initial fresh Tuolumne panel exhausted its work limit in row/column
graph construction. Reusing a saved graph avoids that setup cost for this
manual-realization question. Old development roots without `-v2` predate the
inspection-reporting fix and should not be resumed with the final sources
(this warning concerns the earlier Tuolumne development roots, not the completed
Matrix `manual-gemm-frozen-debug` root). N=4096 row/column remains unmeasured in
this manual panel; the H100 measurements above are from the user's Matrix run.
