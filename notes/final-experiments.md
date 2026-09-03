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
Here we randomly sample whole tensor G_C layouts and run them with the pilot kernels. We record all the relevant memory counters and produce a plot comparing the complete automatic quotient-component vector and aggregate score with the memory counters. We also record the Spearman correlation. These pilot results are the tuning set for the device-specific tau values; TritonBench and the real kernels are excluded from tuning.

### 2. Memory Counters Over G_C Tiles
Here we randomly sample G_C layouts for tiles/blocks of the full tensor and run them with the pilot kernels. We record all the relevant memory counters and produce the same automatic-component and aggregate-score correlations as Experiment 1.

### 3. Memory Counters Over G_OC
Here we randomly sample G_OC layouts and run them with the pilot kernels. We record all the relevant memory counters and produce the same automatic-component and aggregate-score correlations as Experiment 1.

### 4. DP G_C Search
For the kernels in TritonBench we run DP search on the whole tensor over G_C layouts. We record the runtime and in a separate run(s) we profile memory counters. This enables us to report speedup from the layout. It also allows us to report reduction % in L1 events, L2 events, etc. 

### 5. DP G_C Search Tiles
For the kernels in TritonBench we run DP search on tensor tiles over G_C layouts. We record the runtime and in a separate run(s) we profile memory counters. This enables us to report speedup from the layout. It also allows us to report reduction % in L1 events, L2 events, etc.

### 6. G_OC Search
For the kernels in TritonBench we run search over G_OC layouts. We record the runtime and in a separate run(s) we profile memory counters. This enables us to report speedup from the layout. It also allows us to report reduction % in L1 events, L2 events, etc.

### 7-9. Real Kernel Search
We repeat experiments 4, 5, and 6 on the "real" kernels. We only study runtime here.

### 10. (Appendix) Hardware Profile Sensitivity Study
Perturb the hardware profile tau's values and see how sensitive the speedups are to perturbations.

### 11. (Appendix) Simple Hypergraphs
Construct Triton hypergraphs with only simple issue and temporal edges and run a subset of the above experiments.

### 12. (Appendix) Solve times.
Record solve times. We may already have this data from other experiments. Just report it here.

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
