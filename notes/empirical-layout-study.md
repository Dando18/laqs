# Fixed-schedule empirical layout study

This implements sections 5, then 3–4 of the follow-up plan. Broad TritonBench
expansion remains deferred. The study asks whether the supported layout family
contains a useful map, whether LAQS selects it, and whether automatic realization
matches explicit indexing for that same map.

## Native H100 compatibility constraint

The packet capture examines final native PTX, matches `cp.async.cg.shared.global`
16-byte instructions to frontend load sites using source locations, and records a
per-site compatibility constraint. It applies only to an H100 with `sm_90`/`sm_90a`
PTX and the demonstrated two-byte element path. Those graph sites require four
protected low element-address bits, preserving original 32-byte source-sector
membership and within-sector order. The old per-thread packet constraint protected
three bits. The captured Matrix GEMM identifies both A and B loads; this study
changes A only.

This is an admission constraint, not a new transaction-count objective. It does
not regroup lanes, change component weights, or introduce an MI300A counterpart.
The historical address-ablation workflow still permits its explicitly frozen old
maps; the normal packet selector and this study enforce the new capture contract.

## Declared panels

All launches use their reference capture's frozen schedule. Conversion is outside
kernel timing, and measured packing costs and amortization are reported separately.

| Case | Varied matrix | Split domain, plus ordinary | Distinct physical maps |
| --- | --- | --- | ---: |
| MI300A row/column, small | A, 1024×1024 FP32 | TI=2^a, a=0…10; TJ=2^v, v=1…10 | 91 |
| MI300A sum, small | input, 4096×1024 FP32 | TI=2^a, a=0…12; TJ=2^v, v=1…10 | 109 |
| H100 GEMM, asymmetric | A, 512×4096 FP16; B ordinary | TI=2^a, a=1…9; TJ∈{16,32} | 19 |

TI=1 and full-width TJ can describe the same ordinary mapping; only identical
physical maps are removed. Physically different maps survive equal scores,
equal full component vectors, and equal partial flags. Exact quotient counts may
be reused during CPU scoring, but this never removes a timing candidate.

The H100 panel requires frozen BK=32. All 18 changed maps have scalar J=0 on the
saved graph, versus ordinary J=0.1166667; the analytical tie-break selects 2×32.
The H100 panel has 2 scalar-score classes, 5 full-feature classes, and 6 partial-flag
classes. This scalar tie alone does not establish equal features or flags. The sum panel
has 3 scalar-score classes and 15 full-feature classes; its analytical top-1 is
2×16. These are analytical observations, not new performance results.

## Measurement and independent confirmation

Tuning processes compare up to four changed maps with ordinary storage. Each
batch uses three shared input placements, rotated timing order, complete output
validation, exact packing/round-trip checks, primitive checks, and identity and
same-pointer controls. Candidates in different batches do not share a process or
necessarily the same addresses. Selection compares their geometric mean paired
ordinary/candidate ratios. Batch order rotates across process replicates.

The empirical choice includes ordinary and is frozen before evaluation. Fresh
evaluation processes compare ordinary, analytical top-1, and the empirical winner
together, along with their explicit source implementations. The largest training
contrast between equal full-feature vectors is also frozen and independently
remeasured when its training spread exceeds 5%. This additional contrast is a
diagnostic, not a second selection opportunity.

The final report uses held-out ratios:

- **H** = ordinary runtime / empirically selected runtime.
- **S** = analytical top-1 runtime / empirically selected runtime.
- **C** = automatic runtime / explicit runtime for the same map and schedule.

Process replicates, not placements or samples, are the independent units. These
ratios are estimates and need not satisfy algebraic bounds on held-out data.
Automatic rejections remain explicit failures. Every rejected map, and every map
whose tuning speedup is below the declared 0.9 threshold, gets a manual diagnostic
unless already included in evaluation. Manual audit winners do not silently
replace the empirical automatic choice. Ownership/primitive differences remain
visible in the code-generation comparisons.

`search.json` stores physical rows, full component vectors, separate scalar,
feature and flag classes, the protected-bit rule, domain, and schedule.
`study-choice.json` freezes the selection and audits. `study-partial.json` publishes
held-out results before lengthy manual audits finish. `study.json` and `study.md`
are the final reports. Raw timing and binary records live under `study-tune/`,
`study-evaluate/`, and `study-audit/`. There is no claim of a global layout oracle.

## Rerunnable debug jobs

Run from the repository root. Each script leaves cleanup margin inside a
five-minute, one-task, one-GPU allocation. Repeat an identical command until its
summary says complete. CPU component scores and GPU batches are checkpointed;
source, GPU/compiler, reference, panel, measurement, or frozen-choice changes
invalidate reuse. Use a fresh `--root` after code or experiment-setting changes.
`--retry-failed` retries a failed stage; ordinary budget expiration resumes without
that option. The default is three independent processes in each measurement phase.

```bash
flux run -n1 -g1 -c8 -t 5m -q pdebug \
  bash triton/packet_layout/run-study-tuolumne.bash row_column

flux run -n1 -g1 -c8 -t 5m -q pdebug \
  bash triton/packet_layout/run-study-tuolumne.bash sum

srun -n1 -G1 -c8 -p pdebug -t 00:05:00 \
  bash triton/packet_layout/run-study-matrix.bash
```

The scripts reuse existing automatic graphs after checking the frozen launch,
input identities, and access-manifest semantics (ignoring source locations);
they capture and validate the native kernel again. Default
roots are `triton/experiments/results/layout-study-{row_column,sum,gemm}-v1`.
On Matrix, try `pdebug` first; use `pbatch` if the debug queue remains occupied.
The complete sweeps and manual audits are expected to require multiple allocations.

After GEMM completes, this separate rerunnable counter script automatically skips
profiling unless held-out headroom clears the confidence interval and the existing
two-sided placement-control budget:

```bash
srun -n1 -G1 -c8 -p pdebug -t 00:05:00 \
  bash triton/packet_layout/run-study-counters-matrix.bash
```

It profiles only ordinary/top-1/winner, checks the profiled binary against the
timed binary, and collects a small L1/L2/DRAM/request/wavefront set. Registers,
spills and shared memory are retained with each binary. These are whole-kernel
counters, including B traffic; profiled duration is not a speedup measurement.
MI300A hardware counters are not implemented by this Nsight-specific stage.

## Implementation validation

CPU tests cover the evidence-bound guard, sector preservation, exact 19-map panel,
retention of feature/flag ties, count-cache correctness and restart, rejection
handling, and freezing selection before independent evaluation. An initial
two-minute MI300A row/column smoke job captured successfully and exposed the need
for component-level CPU checkpoints, which were added. A subsequent MI300A sum
smoke job scored all 109 maps and completed 11 timing batches covering 44 changed
maps plus ordinary. All 55 candidate validations passed, with no primitive
rejections. It paused cleanly at its 4.3-minute work budget. These development
artifacts are in `layout-study-sum-smoke-v2`; use the fresh default study roots for
the final code and three-process runs. The CPU packet suite passed 49 tests with
15 GPU-only tests skipped on the login node. A bounded 90-second row/column CPU
check completed 10 candidate scores and retained the next candidate's component
checkpoint at interruption.

Full study performance conclusions require the completed three-process runs above;
the smoke run has not selected and independently confirmed an empirical winner.

## Checkpoint restart repair

The first full runs exposed a bookkeeping bug: the worker hashed its batch after
the measurement helper added deterministic `tiles` diagnostics to the storage
records. The restart path hashed the original batch, so it rejected valid timing
checkpoints. The worker now hashes the immutable batch before measurement. A
regression test serializes a mutated worker result and resumes it, while still
rejecting altered timings or measurement settings. The packet suite now passes
50 CPU tests, with 15 GPU tests skipped on the login node.

The three `layout-study-*-v1` directories were repaired using
`bin/repair-study-batch-hashes.py`. The repair checks original record hashes,
reconstructs the exact tile annotations, and permits only the reviewed hash-timing
source change. It preserves all timing, validation and code-generation payloads,
rebinds metadata hashes, and retains GEMM's frozen choice. Original metadata and
an audit of the replacements are under each platform's `checkpoint-hash-repair/`.
The repair preserved 11 row/column, 14 sum, and 15 GEMM timing processes; none had
kernel-validation failures. Their original scripts resume without a fresh root or
`--retry-failed`.
