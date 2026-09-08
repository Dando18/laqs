# Experiments 4–6: repairs and rerun protocol

This follows [the September 5–7 results analysis](experiments-4-6-analysis.md).
The changes repair selection and measurement, and add a conservative compiler
contract. They do not establish a speedup on the full panel; that requires the
new measurements.

## Rerun

From the repository root, run the appropriate command on each cluster:

```bash
# Tuolumne / MI300A
triton/experiments/submit-experiments-4-6-tuolumne.bash

# Matrix / H100
triton/experiments/submit-experiments-4-6-matrix.bash
```

Each command rebuilds both plugins and submits Experiments 4, 5, and 6 under
all three frozen tau maps. Results go to
`triton/experiments/results/search-v2`, preserving the previous reports.
Use `--suite-root /shared/path` to select another shared directory.
[The experiment README](../triton/experiments/README.md) documents targeted runs,
queue and wall-time settings, outputs, and summary commands.

The combined submission has 118 jobs per platform. Each case has a GPU capture,
a CPU graph/search job, GPU preflight validation, and a GPU measurement job.
Two CPU jobs summarize preflight and final results. All GPU jobs request one
task and one GPU. Every case finishes preflight on its device before that
device's measurements begin. Failed dependencies are accounted for rather than
leaving the downstream workflow indefinitely waiting for a successful job.

Matrix's wrapper also reserves one GPU for the CPU stages via `SBATCH_GPUS=1`:
the initial submission accepted GPU capture but rejected the CPU-only search
request with “Requested node configuration is not available.” Those stages
still compute on CPU, leaving the reserved GPU idle. This is a scheduling
workaround; the partition policy has not been independently inspected.
Tuolumne continues to request no GPUs for CPU stages. The shell-only change
preserves the source fingerprints used by already-running suites.

## Implemented changes

| Finding | Repair |
| --- | --- |
| Ordinary layout lost on tile ties or regressions | Include the actual ordinary choice for every array; keep it on ties; require a strict improvement and a 1% margin with a unit score floor. Recompute and assert the final joint score does not exceed baseline. |
| Scalarized issue graph | Derive register groups from post-coalescing pointer contiguity, mask alignment, and element width. Preserve the resulting vector bits during search. |
| Workgroups represented as isolated waves | Retain complete workgroup traces and union all owners of a dynamic parent operation. Lane and SIMD temporal scopes remain separated by wave. |
| Undersized or misaligned natural tiles | Union all register slices and waves of each parent operation and use its aligned dyadic enclosing footprint. |
| Cheap predicted locality, expensive generated code | Admit only candidates without native-issue inflation. Compare saved final memory/matrix instruction forms, execution encodings, registers, spills, shared memory, and static instruction count. Reject missing final assembly and regressions outside the declared budget; fall back to ordinary storage. |
| Expensive Pareto construction in Experiment 6 | Use scalar inner search and scalar outer DP; exploit the separable objective across arrays. Skip inner maps invisible at every active byte scale. |
| Four-bit inner maps often cannot improve the score | Add a declared sparse neighborhood of one swap or XOR around the ordinary and canonical-optimum mappings, above protected vector bits. This changes the Experiment 6 search scope explicitly. |
| Graph bounds applied before compression | Compress complete workgroups online and bound retained events. Extend the aligned-translation proof to suitable matrix allocations and invariant addresses/masks. Perform graph construction on CPU. |
| Autotuning and graphs repeated across grammars/profiles | Capture one native configuration, resolved compiler defaults, and concrete index data per case/device; reuse one hashed graph for all nine selections. Lock shared preparation caches. |
| Python enqueue gaps in timing | Time captured CUDA/HIP graphs. Rotate ordinary, selected, and independent ordinary-identity controls. Retain separate profiler runs. |
| Correlated timing samples treated as replication | Report the geometric mean of paired process-median ratios and a log-ratio t interval using independent processes. Mark a gain as credible only if it also exceeds the declared identity-noise check. |
| Incomplete provenance and unsafe resume | Record source hashes, the actual editable Triton checkout, plugin/IR/assembly/binary hashes, GPU/runtime identity, seed, input byte probes, shapes, and strides. Bind checkpoints to the physical selection and measurement settings, excluding incidental elapsed search time. |
| Missing cases and one-sided regressions disappear | Enumerate all 29 cases in CSV/coverage output; plot every completed device case, with a complete status inventory. Report common-panel statistics separately and retain the twelve-family requirement. |
| Dropout rejected at a pointer bitcast | Support casts that preserve storage element width, including bool-to-byte pointers; reject width-changing reinterpretations. |
| Costly generic address realization | Use 32-bit offset/bit arithmetic when the packed envelope fits safely, after recovering the original pointer offset. Keep canonical bit-run lowering. Save generated code for inspection. |
| Non-power-of-two baseline ambiguity | Require a proof that native pitches and alignment induce the same region partition as the scoring envelope at every declared scale. Aligned non-power-of-two row pitches can pass; unproved operands remain fixed. Record native storage separately from the scoring map. |
| Broad numerical tolerance and NaN equality | Require finite outputs, use dtype-appropriate comparisons, and independently check the native operator with PyTorch references. Preserve initial output values when cloning read-write state. |
| Workload description mismatch | Correct the FP8 specification to the actual unscaled tutorial GEMM. The benchmark implementation is explicitly identified. |
| Setup costs absent | Record capture, graph/search, packing/allocation, compilation/first-launch, and validation costs, plus allocation expansion and a conservative packing/allocation amortization bound. |
| Profiler duration used to train speedup weights | Reject that training target. New speedup fits require unprofiled graph timings, the post-coalescing realization, and consistent source provenance. |

## Interpretation and remaining research work

The compiler contract is deliberately conservative: no additional registers,
spills, or shared memory; unchanged memory and matrix instruction forms and
execution encodings; at most 10% growth in final static instruction count.
These checks reduce the observed failure modes but cannot guarantee faster
execution. Timing remains an evaluation outcome and does not select layouts.

Experiment 6 is now **bounded scalar G_OC plus sparse neighbors**. Scalar
optima are checked against admission rules. It is neither an unrestricted G_OC
optimum nor an exact constrained optimizer over every admissible mapping.
Baseline abstentions should remain visible; they are not speedup successes.

The existing learned tau maps remain frozen **legacy pilot-weight ablations**.
They have not been recalibrated for the corrected graph and realization. New
calibration requires new independent pilot data. The fitter now prevents reuse
of profiler durations as speedup targets; no Experiment 4–6 measurements feed
training.

General structured loops and ragged launches still use bounded exact
interpretation when the translation proof does not apply. Streaming compression
and CPU staging remove the former memory/GPU-allocation failure modes; they do
not provide an unrestricted symbolic frontend. The preflight and final reports
must determine whether twelve common operator families actually pass.

The reserve audit found that the pinned Mamba operators import their Triton
kernels from optional `mamba_ssm`, which is absent from the tested Tuolumne
environment. The pinned jagged-layer-norm operator has no direct Triton JIT
backend. These are recorded as unavailable reserve implementations. They cannot
be counted as portable coverage without additional implementation/dependency
work. No replacement workload or external backend was silently introduced.

Joint schedule/storage tuning, producer–consumer optimization, and timed
shortlist selection remain alternative research directions from the analysis.
They would change the experimental question and need their own fair baselines;
this repair implements the conservative fixed-configuration direction.

## Validation

- All 36 CPU test modules passed in isolated processes. The expanded search and
  harness regression suite also covers baseline ties/regressions, exhaustive
  small-GL comparison, vector preservation, aligned footprints, native-pitch
  equivalence, cache identity, and invalid speedup training data.
- Eight compiled frontend/oracle tests and two code-generation tests passed in
  the ROCm/Triton environment. The GEMM oracle now independently represents all
  four workgroup waves rather than multiplying a single-wave trace.
- Both C++ plugins built successfully on Tuolumne.
- Reduced MI300A runs passed capture, shared graph construction, selection,
  numerical validation, graph timing, and report generation for vector-add and
  dropout across Experiments 4–6. All six selected ordinary storage.
- Direct transformed-pointer tests passed exact numerical comparison for FP32,
  dropout's boolean input, and non-power-of-two packing. The compiler contract
  rejected the deliberately scalarizing layout and detected its additional
  registers/instructions and changed memory instruction forms.
- Both scheduler command graphs were checked with dry runs, all experiment shell
  scripts passed syntax checks, and the report audit retained all 29 cases.

The full panel and H100 execution have not been rerun. A final separate-profiler
smoke test was canceled while queued after Flux estimated a 25-minute debug
wait. The completed GPU tests used one task, one GPU, and limits of five minutes
or less. Profiling and platform-wide coverage will be exercised by the staged
user-submitted reruns.
