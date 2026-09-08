## Yes—this is the direction I would pursue first

A more faithful hypergraph construction could plausibly recover much of the missing vector-L1 correlation. In fact, the pattern in your results points more strongly toward **incorrect access-event boundaries** than toward a fundamental failure of quotient locality:

* The quotient still ranks L1-to-L2 read requests extremely well.
* Changing the quotient scale from 32 to 128 bytes barely changes the vector-L1 correlation.
* Arbitrary \(A\) hurts the vector-L1 relationship much more than the downstream-request relationship.

That combination suggests that the missing variable is not primarily “which byte scale is correct.” It is likely **how one conceptual Triton access is partitioned into hardware service events**.

The qualification is that the graph must become more **lowering-faithful**, not simply contain more arbitrary edge families.

## Where the current construction may be wrong

The current issue construction effectively creates

$$
E_{r,w}
=
\left\{
x(\ell,r,w): \ell \in \text{lanes}
\right\},
$$

one hyperedge containing all lane accesses at a fixed register and warp coordinate.

The implicit hardware model is therefore

$$
C_{\mathrm{issue}}(A)
\approx
\sum_{r,w}
q_d(E_{r,w};A),
$$

where

$$
q_d(E;A)
=
\left|
\left\{
\left\lfloor
\operatorname{addr}_A(x)/2^d
\right\rfloor:
x\in E
\right\}
\right|.
$$

That is appropriate only when each \(E_{r,w}\) corresponds to one actual hardware event whose distinct \(2^d\)-byte regions are counted once.

But the lowered kernel may instead have service events such as

$$
E_{p,g},
$$

where \(p\) identifies a particular lowered memory instruction or vector component and \(g\) identifies some coalescer or lane-processing cohort. The vector-L1 counter could then behave more like

$$
C_{\mathrm{vL1}}(A)
\approx
C_{\mathrm{fixed}}
+
\sum_{p,g}
q_{64}(E_{p,g};A).
$$

AMD describes the vector-memory path as coalescing the lane requests associated with vector-memory instructions into cache-line requests, and the counter you are using is documented in terms of 64-byte vector-L1 accesses. That makes the actual lowered memory instruction a much more defensible primitive event than “all lanes at fixed abstract register coordinate.” ([ROCm Documentation][1])

Suppose several actual events have currently been merged:

$$
E=\bigcup_k E_k.
$$

Then LAQS currently measures

$$
q_d\left(\bigcup_k E_k;A\right),
$$

while the hardware may count

$$
\sum_k q_d(E_k;A).
$$

These are not equal:

$$
q_d\left(\bigcup_k E_k;A\right)
\leq
\sum_k q_d(E_k;A).
$$

The difference is the number of regions reused across events that the hardware services separately.

Canonical layouts may accidentally make this discrepancy regular: lane, register, instruction, and address order remain aligned, so the coarse quotient and the true event count rise together. A random dense \(A\) destroys that alignment. Two layouts can then have the same quotient for the union while causing very different sums over the actual subevents.

That would explain exactly why random \(A\) exposes the problem.

## “More faithful” may mean both splitting and merging

It is worth not framing this as purely finer-grained edges.

The current register fiber might be:

* **too coarse in lanes**, if one machine instruction is processed through multiple lane cohorts;
* **too fine in registers**, if several Triton register elements become one vector load such as a multi-dword instruction;
* **too coarse in program points**, if multiple lowered loads that happen to produce related register coordinates are combined;
* missing predicates, vector widths, masked lanes, or loop-stage identities.

So the goal should be:

> Construct one edge for each set of logical elements whose addresses are jointly coalesced or jointly serviced at a particular hardware stage.

That can produce smaller or larger edges than the current construction.

Your current structural validation does not establish this correspondence. It checks that the candidate and reference retain the same extracted execution layout, aggregate load/store opcode signature, and spill state. It does not determine which logical register elements participate in each individual lowered load instruction.

Thus, “same code structure” does not yet mean “the current hyperedges equal the machine issue events.”

## A hardware-conditioned hypergraph construction

I would represent each logical dynamic access as carrying hardware and schedule coordinates:

$$
a =
(p,t,b,w,r,\ell,v),
$$

where, for example:

* \(p\): memory operation or lowered instruction site;
* \(t\): loop iteration or temporal epoch;
* \(b,w\): block and warp;
* \(r\): per-thread register element;
* \(\ell\): lane;
* \(v\): vector component within the lowered instruction.

A hardware component \(s\) defines an equivalence relation over these accesses. Every equivalence class becomes a hyperedge:

$$
E_{s,c}
=
\left\{
x(a):a\in c
\right\}.
$$

Then the component score is

$$
J_{s,d}(A)
=
\sum_{c\in\mathcal C_s}
w_{s,c}\,
q_d(E_{s,c};A).
$$

This gives a **locality signature**

$$
\mathbf J(A)
=
\left(
J_{s,d}(A)
\right)_{s,d}
$$

instead of immediately compressing everything into one issue quotient.

A useful initial hierarchy would contain three stages.

### 1. Lowered-instruction events

Create one edge per actual global load instruction instance:

$$
E_{\mathrm{vmem}}
=
\{
\text{active logical elements participating in that instruction}
\}.
$$

This should encode:

* memory operation identity;
* vector width;
* active mask;
* register elements packed into the instruction;
* dynamic multiplicity.

The first experiment should compare

$$
\sum_{\mathrm{vmem}}
q_{64}(E_{\mathrm{vmem}};A)
$$

against `TCP_TOTAL_CACHE_ACCESSES_sum`.

This is the cleanest test because it introduces no fitted weights. If the counter really reflects distinct 64-byte line lookups generated by those instructions, the relationship should become close to affine after accounting for fixed whole-kernel traffic.

### 2. Lane-subspace fibers

If one machine instruction is internally processed in multiple cohorts, split its lane coordinate algebraically.

Rather than hard-coding “groups of 16 lanes,” define fibers using subsets of lane bits:

$$
E_{p,r,\ell_{\mathrm{fixed}}}^{(S)}
=
\left\{
x(p,r,\ell):
\ell_{\bar S}
=
\ell_{\mathrm{fixed}}
\right\},
$$

where \(S\) is the subset of lane bits allowed to vary.

This lets you test a small nested family such as:

$$
S_1 \subset S_2 \subset \cdots \subset S_{\mathrm{wave}},
$$

without assuming undocumented details about MI300A’s internal coalescer. It also handles interleaved cohorts that cannot be represented by merely taking contiguous lane-ID ranges.

### 3. Temporal/cache-service components

Keep issue formation separate from temporal reuse.

For vector-L1 accesses, the important object may be the exact instruction-level line lookups. For L1-to-L2 requests, accesses to the same region across nearby instructions can be absorbed by the cache or an outstanding request. A coarser space-time edge can represent that:

$$
E_{\mathrm{L2},h}
=
\bigcup_{t=t_0}^{t_0+h-1}
E_{\mathrm{vmem},t}.
$$

This naturally permits different components to predict different hardware counters:

$$
\begin{aligned}
J_{\mathrm{vmem},64}
&\longleftrightarrow
\text{vector-L1 line lookups},\\
J_{\mathrm{reuse},128}
&\longleftrightarrow
\text{L1-to-L2 read requests}.
\end{aligned}
$$

That would explain your current observation rather elegantly: the present full-issue quotient is already a reasonable approximation to the downstream request footprint but omits the internal decomposition needed to predict vector-L1 processing.

## Much of the software structure already exists

You would not need to replace the LAQS formulation.

The branch already has `linear_layout_resource_fiber`, which can vary selected Triton hardware dimensions while fixing others, and `execution_conditioned_hypergraph` can accept additional execution-fiber objective components.

The general access-scope code also already supports spatial groups, temporal windows, and contexts, while the hardware-profile abstraction supports weights that depend on both edge family and quotient level.

The main extensions appear to be:

1. **Instruction identity as an event coordinate.**
2. **Bit-level resource fibers**, rather than only varying or fixing an entire `lane`, `register`, or `warp` dimension.
3. **An extraction path from lowered Triton IR or ISA** that says which register elements and lanes belong to each memory instruction.
4. A small, architecturally motivated set of named service components.

The current production Stage-1 path does not supply these optional resource-fiber components; it primarily uses the issue construction and optional temporal edges.

## What hypergraph refinement cannot fix

There is an important mathematical boundary.

At quotient level \(d\), define the low-address subspace

$$
V_d(A)=A^{-1}U_d.
$$

For any candidate-independent logical edge \(E\),

$$
q_d(E;A)
=
\left|
(E+V_d(A))/V_d(A)
\right|.
$$

Therefore, every objective of the form

$$
J_d(A)
=
\sum_E w_E q_d(E;A)
$$

depends only on \(V_d(A)\), regardless of how many edge families you add.

Consequently:

* A refined graph **can** distinguish two layouts with the same current scalar quotient but different \(V_d\).
* It can also distinguish layouts whose total scores tie but whose per-component scores differ.
* It **cannot** distinguish two matrices \(A\) and \(A'\) with exactly the same \(V_d\) at every evaluated scale.

So the correct empirical sequence is not to assume the fiber realization matters, but also not to rule it out:

$$
\text{coarse score}
\;\rightarrow\;
\text{refined hardware components}
\;\rightarrow\;
\text{exact multiscale flag}
\;\rightarrow\;
\text{residual realization effects}.
$$

The current random experiment mostly compares layouts with the same **scalar score**, not necessarily the same subspace or complete flag. A scalar quotient sum is an enormous compression of the underlying locality structure. There is substantial room for better edge construction to resolve those apparent ties before introducing a realization model.

## The decisive experiment

I would reuse the existing random-layout profiles and recompute a sequence of predictors offline:

$$
\begin{array}{ll}
M_0:& \text{current fixed-register/full-lane issue quotient},\\
M_1:& \text{quotient per lowered memory instruction},\\
M_2:& \text{instruction quotient split by nested lane-bit fibers},\\
M_3:& \text{instruction and temporal/cache-context components}.
\end{array}
$$

For each model, report:

* tie-aware Spearman correlation;
* strict pairwise inversion rate;
* spread of the hardware counter among layouts with equal component signatures;
* correlation within each tile shape;
* incremental improvement relative to \(M_0\).

The strongest diagnostic is the residual spread within equivalence classes.

First group layouts by the current scalar quotient. Then group them by:

$$
\left(
J_{\mathrm{vmem},32},
J_{\mathrm{vmem},64},
J_{\mathrm{vmem},128},
J_{\mathrm{lane},32},
\ldots
\right).
$$

Finally group them by the exact subspaces

$$
\left(V_{32},V_{64},V_{128}\right).
$$

The outcomes have clean interpretations:

* If \(M_1\) nearly eliminates vector-L1 spread, the old edge did not correspond to an actual memory instruction.
* If \(M_2\) is required, the omitted information is an internal lane/coalescer partition.
* If \(M_3\) primarily improves L1-to-L2 prediction, temporal cache service is the missing distinction there.
* If substantial variation remains among layouts with the same exact multiscale flag and all refined component scores, then—and only then—is there strong evidence for flag-realization effects or some non-locality phenomenon.

## My recommendation

Make the next conceptual contribution **hardware-conditioned access-event induction**:

> Triton’s compiled execution layout determines not one issue hypergraph, but a hierarchy of service hypergraphs corresponding to memory-instruction formation, lane processing, and cache-level reuse.

This is preferable to immediately adding a flag-realization objective for three reasons.

First, it attacks a likely modeling mismatch directly. Second, it preserves the clean quotient-space theory: the improvement is in constructing the right \(E\), not modifying \(q(E;V_d)\). Third, it could turn the current seemingly conflicting observations into a stronger result:

> Fine-grained issue quotients explain vector-L1 work, while coarser space-time quotients explain requests propagated to L2.

The very high \(Q_{128}\)-to-L1→L2 correlation suggests that the quotient abstraction itself is capturing real address locality. The failure at vector L1 is more plausibly telling you that **one hyperedge currently spans the wrong hardware service cohort**. I would treat flag realization as the residual hypothesis after this richer construction has been tested, not as the immediate explanation.

[1]: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.0.2/conceptual/vector-l1-cache.html "https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.0.2/conceptual/vector-l1-cache.html"
