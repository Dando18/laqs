Q:
I have a question/thought about my current mapping. We use the layout L that maps from hardware to tensor elements, and then from that use a kernel specific P to find the per-program and iteration elements touched. This forms an edge. Why not retain the hardware elements that L maps to and use those to weight levels d during flag search? That is, we could retain elements mapped to registers and use that to weight lower d higher during our flag search. Does this make sense? Is there a simple, clean, and/or elegant way to go about this?



A:
## Yes—retain the **hardware fiber**, not merely the resulting elements

The idea makes sense. The clean formulation is to preserve which hardware-coordinate subspace produced each logical access set, then give each such **resource-fiber family** its own weights over address scales \(d\).

This fits the current theory unusually well. Your draft already separates the candidate memory layout \(A\) from execution-layout information and already allows a scope-and-scale weight \(\tau_{s,d}\). The main change is to let the scope \(s\) be induced systematically from Triton’s LinearLayout rather than defined only as an unlabeled logical edge. 

The current implementation is already close. `InducedMemoryEvent` retains the hardware location corresponding to every logical access, but its `hyperedge` projection keeps only the logical coordinates. Moreover, the current issue construction fixes register, warp, and block coordinates while varying lane, whereas the temporal construction fixes a complete hardware location while varying time.

## LinearLayout-induced resource fibers

Write Triton’s hardware-coordinate space as

$$
\mathcal H
=
\mathcal H_{\mathrm{reg}}
\oplus
\mathcal H_{\mathrm{lane}}
\oplus
\mathcal H_{\mathrm{warp}}
\oplus
\mathcal H_{\mathrm{block}}.
$$

For a resource family \(s\), choose a hardware subspace

$$
R_s \subseteq \mathcal H.
$$

For example,

$$
R_{\mathrm{reg}}=\mathcal H_{\mathrm{reg}},
\qquad
R_{\mathrm{lane}}=\mathcal H_{\mathrm{lane}},
\qquad
R_{\mathrm{reg}\times\mathrm{lane}}
=
\mathcal H_{\mathrm{reg}}\oplus\mathcal H_{\mathrm{lane}}.
$$

Fixing all complementary coordinates gives a hardware fiber

$$
F_{s,\bar h}=\bar h+R_s.
$$

For execution context \(c\), let

$$
X_c=P_c\circ L
$$

be the complete hardware-to-logical access map. The corresponding logical edge is

$$
E_{s,c,\bar h}
=
X_c(F_{s,\bar h})
=
P_c\!\left(L(\bar h+R_s)\right).
$$

This gives several natural edge families:

* **Lane issue fiber:** fix register, warp, and block; vary lane. This is the issue edge you already construct.
* **Register ownership fiber:** fix lane, warp, and block; vary register. This contains the values owned by one lane within the tensor operation.
* **Register–lane tile fiber:** fix warp and block; vary register and lane. This describes the full tensor fragment owned by one warp.
* **Temporal fiber:** fix the hardware location; vary an execution-step coordinate over a bounded window.

The last case can be made completely uniform by augmenting the execution space with a schedule coordinate,

$$
\widetilde{\mathcal H}=\mathcal H\oplus\mathcal H_{\mathrm{time}},
$$

and viewing both issue and temporal edges as images of different fibers of one execution map.

## The particularly elegant algebraic interpretation

Suppose \(P_c\) is affine over the fiber, with linear part \(M_c\). Then

$$
X_c(h)=a_c+M_cLh.
$$

The image of the resource fiber is an affine subspace

$$
E_{s,c,\bar h}
=
x_0+S_{s,c},
\qquad
S_{s,c}
=
M_cL(R_s).
$$

Its quotient cardinality has the closed form

$$
\left|
E_{s,c,\bar h}/V_d
\right|
=
2^{
    \dim S_{s,c}
    -
    \dim\left(S_{s,c}\cap V_d\right)
}.
$$

This is exactly your intuition stated in a basis-independent way.

For register-owned elements, weighting small \(d\) strongly encourages

$$
\dim\left(S_{\mathrm{reg},c}\cap V_d\right)
$$

to become large early in the flag. In other words, logical directions induced by register variation are encouraged to enter the low-address flag early.

Equivalently, a register fiber fits inside one \(2^d\)-element physical region precisely when

$$
S_{\mathrm{reg},c}\subseteq V_d,
$$

or, in terms of the complete composition,

$$
A\,M_c\,L(R_{\mathrm{reg}})
\subseteq U_d.
$$

This provides a direct mathematical bridge between LinearLayouts and LAQS:

$$
\boxed{
\text{hardware resource subspace}
\;\xrightarrow{\,L\,}\;
\text{tensor directions}
\;\xrightarrow{\,P_c\,}\;
\text{logical directions}
\;\xrightarrow{\,A\,}\;
\text{physical address bits}
}
$$

LAQS then asks how quickly each hardware-derived direction space is absorbed by the low-address flag.

Importantly, this is not really a new kind of score. It is the existing quotient score applied to LinearLayout-induced fibers.

## A small example

Consider four lanes and two register values per lane. Let

$$
r\in\Ftwo,\qquad
\ell\in\Ftwo^2,
$$

and suppose the execution layout maps

$$
L(r,\ell)=(\ell,r).
$$

Thus lanes select tensor rows and the register coordinate selects one of two tensor columns.

For fixed register \(r\), the lane issue fiber is

$$
E^{\mathrm{lane}}_r
=
\{
(0,r),(1,r),(2,r),(3,r)
\}.
$$

For fixed lane \(\ell\), the register ownership fiber is

$$
E^{\mathrm{reg}}_\ell
=
\{
(\ell,0),(\ell,1)
\}.
$$

The first edge says that the four elements requested simultaneously across lanes should be well packed. The second says that the two elements owned by one lane should be well packed.

A minimal objective could be

$$
J(V_\bullet)
=
\lambda_{\mathrm{issue}}
\sum_r
\left|E^{\mathrm{lane}}_r/V_{d_{\mathrm{txn}}}\right|
+
\lambda_{\mathrm{reg}}
\sum_\ell
\left|E^{\mathrm{reg}}_\ell/V_{d_{\mathrm{reg}}}\right|.
$$

Here \(d_{\mathrm{txn}}\) is the transaction-relevant scale, while \(d_{\mathrm{reg}}\) is a finer scale relevant to the register fragment. The flag search must balance putting lane-induced row directions and register-induced column directions into the low bits. Depending on the weights, this naturally produces row-major, column-major, blocked, or interleaved canonical words.

## How I would assign the scale weights

I would not infer the weights from \(L\) alone. There is a useful separation:

$$
\boxed{
L\text{ says which elements are related;}
\qquad
\tau_{s,d}\text{ says how valuable that relation is at scale }d.
}
$$

LinearLayout ownership does not by itself imply a memory transaction. In particular, several register values owned by one lane may be generated by separate load instructions. Therefore, “register-owned” does not automatically mean “must fit in the smallest possible region.”

A simple initial profile would be sparse:

$$
\tau_{\mathrm{lane},d}
\neq 0
\quad\text{near the transaction scale},
$$

$$
\tau_{\mathrm{reg},d}
\neq 0
\quad\text{near the footprint of the per-lane value fragment},
$$

$$
\tau_{\mathrm{reg}\times\mathrm{lane},d}
\neq 0
\quad\text{near cache-line or tile scales},
$$

with temporal fibers weighted near the scale corresponding to their reuse window.

For example, if a lane owns four FP32 values, the register fiber occupies 16 bytes. A reasonable first experiment would put its weight at 16 and perhaps 32 bytes, while retaining the lane-issue weight at 128 bytes. I would not monotonically increase the weight of every smaller \(d\). Below the fiber’s cardinality, multiple regions are unavoidable; above it, the score quickly becomes nondiscriminating. Your normalized excess \(e_{s,d}\) is well suited to handling this.

The repository’s hardware-profile abstraction already represents `tau` as responses over scope–byte-scale components, so this extension principally changes the scope basis, not the objective machinery.

## One important semantic constraint

The current issue construction loops over register coordinates and creates a separate lane edge for each register value. That accurately represents separate lane-wide issue cohorts.

Adding a register fiber introduces an additional assertion:

> Values associated with different register coordinates, but the same lane, should also be close in memory.

That is justified when those values belong to the same tensor operation or occur in a known bounded instruction window. It is less justified if they are accessed at unrelated points in the kernel. A precise version would therefore distinguish:

$$
\operatorname{RegisterFiber}
$$

from

$$
\operatorname{RegisterWindow}(q),
$$

where the latter groups only \(q\) register slots known to occur close together in the emitted schedule.

Thus, LinearLayout provides the ownership relation, but the operation or schedule determines whether that ownership relation should become a locality edge.

## Search complexity and DP

This extension should not require a new flag-search algorithm. The objective remains

$$
J_{\mathrm{area}}(V_\bullet)
=
\sum_s\sum_d
\tau_{s,d}\,
e_{s,d}(V_d),
$$

but now \(s\) includes LinearLayout-induced resource fibers.

For canonical DP, the cost at state \(V_d\) simply includes the register-, lane-, tile-, and temporal-fiber components at that depth. Since each contribution depends only on the current \(V_d\), the optimal-substructure argument remains unchanged.

For affine fibers, the intersection formula

$$
2^{\dim S-\dim(S\cap V_d)}
$$

may actually make scoring cheaper: translated instances with the same direction space \(S\) have identical quotient costs and can be represented once with a multiplicity.

## What this will and will not solve

This could improve flag selection substantially. In particular, it can distinguish layouts that have identical lane-issue scores but arrange per-lane register fragments differently. It gives the search more execution information without introducing an opaque performance model.

However, every such term still depends only on \(V_d\). Therefore, it cannot distinguish two layouts \(A_1\) and \(A_2\) that induce the same complete low-address flag:

$$
A_1^{-1}U_d=A_2^{-1}U_d
\qquad\text{for every }d.
$$

So this is a **better flag objective**, not a solution to same-flag realization selection. Distinguishing realizations would require scoring exact destination bits of

$$
A\,P_c\,L,
$$

or a resource-occupancy model over those bits, after the flag has been selected. That remains naturally separated as the later fiber/realization stage.

## Recommended formulation

I would call this **LinearLayout-induced execution fibers** and make it the central edge-construction abstraction:

$$
\boxed{
E_{s,c,\bar h}
=
(P_c\circ L)(\bar h+R_s)
}
$$

followed by

$$
\boxed{
J(V_\bullet)
=
\sum_{s,d}
\tau_{s,d}\,
e_{s,d}(V_d).
}
$$

Start with only three families:

$$
\text{lane issue},\qquad
\text{register ownership},\qquad
\text{per-location temporal}.
$$

That is small enough to be understandable and testable, directly connects LAQS to LinearLayouts, preserves the existing DP, and states your intuition—“register directions should enter the low-address flag earlier”—without introducing basis-dependent or per-element heuristics.
