# Final Experiments

This document is an overview of the final set of experiments that need to be scripted, run, and analyzed.

## Definitions

- Pilot Kernels: refers to the seven testing kernels we've been using -- bias+relu, softmax+bias, embedding bag, gemv, mvt, gesummv, 5-point stencil
- TritonBench: refers to TritonBench kernels
- Real Kernels: refers to the "real" kernels evaluated in the LinearLayouts paper bf16xint16_gemm, cross_entropy, embedding, flex_attention, fp8_gemm, fused_linear_cross_entropy, fused_linear_jsd, gather_gemv, geglu, gemm, grouped_gemm, int4_gemm, jsd, kl_div, swiglu, template_attention, welford, low_mem_dropout, rms_norm, rope, layer_norm

- G_C: refers to the canonical layout grammar; sometimes we search G_C over the whole tensor and sometimes over just the inner tile.
- G_OC: refers to the outer-canonical layout grammar where the tiles are in canonical order, but the inside of the tile is an arbitrary layout A in GL(p,2)

- MI300A: The MI300A AMD GPU on Tuolumne
- H100: The H100 NVIDIA GPU on Matrix


Unless otherwise stated all following experiments are to be run on both MI300A and H100.
Furthermore, where speedup is needed, we run the Triton (aka Triton+LinearLayouts) baseline in addition to Triton with our layout.

## Experiment List

### 1. Memory Counters Over G_C
Here we generate a seeded random pool of whole-tensor $G_C$ layouts and select a pre-counter stratified panel. We run three declared stratifications over the same pool: every automatic component, issue components only, and non-issue temporal components only. Each stratification balances the marginal amplification bands $Q_{s,b}/LB_{s,b}\in[1,1.5),[1.5,2),[2,4),[4,8),[8,\infty)$ and includes distinct row-major and column-major anchors before filling the remaining strata. We record all relevant memory counters and the Spearman correlation of every $Q_{s,b}$ and $J_{\mathrm{area}}$ with every counter. Counter-focused plots show the four byte scales for one fixed scope alongside $J_{\mathrm{area}}$. These pilot results are the tuning set for the device-specific tau values; TritonBench and the real kernels are excluded from tuning.

### 2. Memory Counters Over G_C Tiles
Here we generate $G_C$ layouts for tiles/blocks of the full tensor and apply the same all-component, issue-only, and temporal-only stratified-panel construction as Experiment 1. The random pool chooses among the kernel's declared tile hypotheses and canonical words; the selected layouts keep outer tiles row-major. We record the same automatic-component, byte-scale, and aggregate-score correlations as Experiment 1.

### 3. Memory Counters Over G_OC
Here we generate a seeded random pool of $G_{OC}$ layouts and apply the same three stratifications as Experiment 1. The pool chooses among declared tile hypotheses and draws the inner mapping from $GL(p,2)$, with outer tiles row-major. We record the same automatic-component, byte-scale, and aggregate-score correlations as Experiment 1.

### 4. DP G_C Search
For the kernels in TritonBench we run DP search on the whole tensor over G_C layouts. We record the runtime and in a separate run(s) we profile memory counters. This enables us to report speedup from the layout. It also allows us to report reduction % in L1 events, L2 events, etc. 

### 5. DP G_C Search Tiles
For the kernels in TritonBench we run DP search on tensor tiles over G_C layouts. We record the runtime and in a separate run(s) we profile memory counters. This enables us to report speedup from the layout. It also allows us to report reduction % in L1 events, L2 events, etc.

### 6. G_OC Search
For the kernels in TritonBench we run search over G_OC layouts. We record the runtime and in a separate run(s) we profile memory counters. This enables us to report speedup from the layout. It also allows us to report reduction % in L1 events, L2 events, etc.

#### TritonBench Workload Settings (Experiments 4--6)

Use a broad, cross-platform panel rather than TritonBench's complete default sweeps. The panel below emphasizes operator breadth while limiting each operator to two representative configurations (one smaller or latency-oriented case and one larger or throughput-oriented case). All cases are forward-only. Shapes are powers of two where practical so that non-power-of-two behavior remains a separate study in Experiment 13.

| Operator | Configurations | Triton implementation / purpose |
| --- | --- | --- |
| `vector_add` | $2^{18}$ and $2^{26}$ elements | `triton_add`; streaming no-harm control |
| `vector_exp` | $2^{18}$ and $2^{26}$ elements | `triton_exp`; compute-plus-streaming control, with its optional in-kernel profiling buffer disabled |
| `low_mem_dropout` | $2^{18}$ and $2^{26}$ elements | `triton_dropout`; two-input elementwise access |
| `softmax` | $(M,N)=(4096,1024)$ and $(4096,16384)$ | Triton fused softmax; row reduction |
| `sum` | $(4096,1024)$ and $(1024,16384)$ | `triton_sum`, 2-D input reduced along dimension 1 |
| `layer_norm` | $(4096,1024)$ and $(4096,16384)$ | `triton_layer_norm`; repeated reduction plus affine inputs |
| `gemm` | $(M,N,K)=(2048,2048,2048)$ and $(512,4096,4096)$ | tutorial Triton matmul; square and asymmetric reuse |
| `bf16xint16_gemm` | $(4096,1280,8192)$ and $(4096,8192,1024)$ | `bf16xint16`; mixed-width model projections |
| `int4_gemm` | $(B,L,N,K)=(1,1,1280,8192)$ and $(1,4096,1280,8192)$ | preprocessed Triton int4 GEMM; decode- and prefill-like cases |
| `fp8_gemm` | $(1024,1024,1024)$ and $(4096,4096,4096)$ | tensor-wise scaling and a Triton-generated GEMM choice |
| `gather_gemv` | $S=2048$ and $S=8192$ | `triton_gather_gemv`; indirect access plus reduction |
| `template_attention` | existing $(16,16,4096,64)$ case | `test_no_exp2`; tiled multi-array attention |
| `jagged_sum` | $(B,M,S,s)=(128,256,256,0.5)$ and $(512,128,1024,0.75)$ | simple-fused Triton kernel; ragged reduction |
| `jagged_mean` | same two jagged configurations | simple-fused Triton kernel; ragged normalized reduction |
| `jagged_softmax` | same two jagged configurations | simple-fused Triton kernel; ragged multi-pass reduction |

Here $s$ is the requested jagged sparsity and $S$ is maximum sequence length. Generate jagged offsets and gather indices from a fixed host seed, reuse the same concrete inputs for the ordinary and transformed layouts, and use the same logical cases on both machines. Use the operator's native datatype: fp16 for ordinary dense ML kernels, fp32 for the vector, sum, and jagged kernels, and the existing mixed datatypes for quantized GEMMs and `gather_gemv`.

Before the full run, compile, analyze, transform, and correctness-check every case once on both MI300A and H100. A case enters the reported cross-platform panel only if it produces an exact automatic trace on both targets, has at least one eligible read-only dense operand, and the transformed launch passes the numerical oracle. Record excluded cases and reasons. If fewer than twelve operator families pass, try `mamba2_chunk_state`, `mamba2_chunk_scan`, and then `jagged_layer_norm` as reserves; do not replace failures with architecture-specific Blackwell, CUDA-only, AITer, or optional-submodule kernels.

For an autotuned kernel, tune the ordinary Triton layout once per shape and platform, then freeze the selected block sizes, warp count, stage count, and other compile-time parameters for the baseline and all three searches. This deliberately allows MI300A and H100 to select different native configurations while preventing autotuning from confounding the within-platform layout comparison. Measure only the named Triton implementation, not every backend registered by TritonBench. Packing, search, compilation, autotuning, and correctness checks are excluded from kernel timing but their elapsed times are retained separately.

Experiment 4 searches the full logical tensor. Experiment 5 uses the frozen kernel configuration's natural power-of-two access footprint for each operand as its primary inner-tile hypothesis; include a second hypothesis only when the kernel has a distinct repeated access footprint. Experiment 6 declares every legal inner exponent tuple containing at most four bits and uses the exact bounded $G_{OC}$ solver. Outputs, read-write allocations, aliases, and unsupported views remain in the trace but keep their ordinary layout. Eligible input arrays are optimized jointly.

Use three fresh timing processes per case, with 10 warm-up rounds followed by 21 samples of 50-launch batches. Run counter collection separately for the ordinary and selected layouts, using three independent profiler launches with 5 warm-up and 20 measured dispatches each. Shard by operator and experiment so jobs can run concurrently and resume independently. The nominal 29 configurations produce 87 kernel/grammar cells per platform; budget approximately 8--14 MI300A GPU-hours and 10--18 H100 GPU-hours in aggregate, with most individual jobs expected to fit in two hours and heavier GEMM, attention, or $G_{OC}$ jobs allowed up to four hours.

### 7-9. Real Kernel Search
We repeat experiments 4, 5, and 6 on the "real" kernels. We only study runtime here.

### 10. (Appendix) Hardware Profile Sensitivity Study
Perturb the hardware profile tau's values and see how sensitive the speedups are to perturbations. This experiment can be relatively simple -- perturb tau randomly, by controlled magnitudes, and measure it's impact on speedups found.

### 11. (Appendix) Simple Hypergraphs
Construct Triton hypergraphs with only simple issue and temporal edges and run a subset of the above experiments.

### 12. (Appendix) Solve times.
Record solve times. We may already have this data from other experiments. Just report it here. If not, we can set up some simple re-runs. We essentially want to report the graph construction time, quotient score computation time (e.g. J_area), and solve time (e.g. the DP).

### 13. (Appendix) Non-power of two tensors.
Run benchmarks on non-power-of-two tensor sizes.

### 14. (Appendix) CUDA/HIP Kernels
Show results outside of Triton, perhaps on Rodinia.


## Quotient Scoring Function, Graph Construction, and Algorithms

### Graph Construction

All experiments in this document use the automatic Triton construction introduced by commit `8f40ee8e0a5528740ade0ba0f0ab229a27129215`, with exactly one exception: Experiment 11 deliberately uses the simple hand-built graph as an appendix ablation. For each concrete kernel configuration, we compile and launch the ordinary Triton kernel under the post-coalescing access-manifest plugin. The frontend reconstructs the exact active logical accesses for every memory operation from the manifest and Triton `LinearLayout`. It retains operation order, execution coordinates, boundary behavior, selected compile-time configuration, and dynamic multiplicity, then compresses only proven XOR-translation-equivalent traces while preserving their total weight.

An exact statically established program-ID period may be evaluated once and
recorded with its full launch multiplicity. This is a lossless trace
representation, not sampling: for example, MVT's 16 full power-of-two
workgroups have period one under aligned XOR translation, so the report keeps
one trace class with multiplicity 16. Data-dependent launches such as
`embedding_bag` are not assigned this shortcut.

From that trace, universal-v1 constructs every nonempty issue, lane-window, SIMD-window, workgroup-step/window, and phase family. The hardware profile crosses these scale-free families with its byte-scale ladder and supplies tau. Experiments 1--3 vary one designated persistent operand at a time: all kernel operations remain in the ordered trace, while score components are materialized for the varied operand and all other layouts stay fixed. Later multi-array search experiments jointly score all eligible arrays. The older manual Stage-1 `issue_events`, instruction/lane-cohort service model, and hand-authored temporal edges are not the graph used by Experiments 1--10 or 12--14.

### Quotient Score

For a layout $A$, hyperedge $E$, and aligned $b$-byte region, the primitive score is the number of regions touched,

$$
q_b(E;A)=\left|\left\{\left\lfloor \operatorname{byteaddr}_A(x)/b\right\rfloor:x\in E\right\}\right|.
$$

For every scope-scale component $(s,b)$, we sum this count over edges using their dynamic multiplicities to obtain $Q_{s,b}(A)$. We retain the complete component vector. On the pilot tuning set, counter-correlation experiments record the component that best matches each counter and fit nonnegative device-specific tau values to relevant L1/L2 work. Derived fields backed by the same native hardware counter contribute only once to the tuning target. A nonnegative ridge fit initializes a deterministic refinement that selects tau by training-set macro Spearman. The resulting MI300A and H100 profiles are frozen before they are used to select layouts for TritonBench or real kernels. The reports include both the counter-matched component and the aggregate selection score. Layout selection uses the hardware profile's fixed weighted excess score

$$
J_{\mathrm{area}}(A)=\sum_{s,b}\tau_{s,b}\,
\frac{b\left(Q_{s,b}(A)-LB_{s,b}\right)}{B_K},
$$

where $LB_{s,b}$ is the capacity-only packing lower bound and $B_K$ is the kernel's dynamic useful-byte exposure. All component scores, the fine quotient score, and the peak normalized excess are retained for analysis. Pilot counters are used only to tune the hardware profile; counters and runtimes from evaluation kernels are outcomes and are not used to choose their layouts. Exact score ties prefer the baseline layout; remaining ties are broken deterministically by address-code complexity and then the layout descriptor.

In these experiments we optimize J_area; we do not use the Pareto frontier. We are mainly focused on (1) correlating quotient score with memory counters, (2) providing >1 speedups in a non-neglible number of cases, and (3) not hurting performance in most if not all cases. We are not concerned with oracle regret anymore -- this was placing too much weight on locality being a strong predictor of performance, which it is not.

### Search Algorithms

For $G_C$, we use the exact count-grid DP. A state records how many low-address bits have been taken from each logical mode (and the last mode for the code-generation tie-break); a transition appends one legal mode bit and adds the quotient components whose region size occurs at that prefix rank. Whole-tensor search runs the DP over all tensor bits. Tile search runs it over each declared inner-tile shape with a fixed canonical outer-tile order, then compares the tile hypotheses under the same score.

For $G_{OC}$, each declared tile shape is handled by exact enumeration of its invertible inner maps $A_{\mathrm{inner}}\in GL(p,2)$, followed by the same canonical-suffix DP for the outer bits. Translation-equivalent edge patterns and score-equivalent inner candidates are collapsed before composing the inner and outer results. The current exact implementation is bounded to $p\leq4$; a larger bound will require a different exact method or an explicitly labeled approximate search. For kernels with multiple eligible arrays, per-array raw-score frontiers are joined before the hardware score is evaluated so that the reported result is a joint layout choice.


## Implementation

Please put all scripts in `trition/experiments/`, results in `triton/experiments/results/`, and plots in `triton/experiments/plots`.
