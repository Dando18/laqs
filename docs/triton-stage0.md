# Triton integration stage 0: induced-hypergraph validation

Stage 0 validates the boundary between Triton's execution layout and RELAY's
logical access model. It does not search for a memory layout.

For one hardware issue cohort, the implementation checks the complete chain

\[
H_e \xrightarrow{L} E_e^L \xrightarrow{A_{\mathrm{default}}}
\text{byte offsets} \xrightarrow{\pi_d} \text{transactions}.
\]

`TritonLinearLayout` mirrors Triton's low-bit-first basis convention without
making Triton a dependency of the `relay` package. It can copy the `bases` and
`out_dims` properties of Triton's Python `LinearLayout` binding. The
`from_blocked` constructor also converts the common one-CTA blocked layout when
the register, lane, and warp factors exactly cover the tile. Repetition,
truncation, and CGA layouts must be supplied as explicit Triton bases rather
than inferred.

`induce_memory_event` enumerates an explicit cohort of register, lane, warp,
and block locations and maps it through `L`. Its result contains an ordinary
RELAY `MemoryEvent` for later stage-1 use and retains the aligned hardware
locations for validation.

`validate_induced_hypergraph` compares that induced event with an independent
trace containing, for every hardware location:

- the observed logical coordinate;
- the observed byte offset relative to the allocation; and
- its transaction ID at the requested aligned byte scale.

Validation requires the hardware-location sets, logical coordinates, byte
offsets, transaction IDs, transaction groups, and RELAY quotient count to
agree. Per-location comparison is intentional: a lane-bit permutation can
leave the quotient count unchanged and would otherwise hide a convention bug.
Multiple scalar elements owned by one lane are represented by distinct
`register` coordinates, so vector width is not folded into byte addressing.

## MI300A probe

The probe compiles and executes one 64-element int32 Triton tile load with one
64-lane wave. It reads AMD's local work-item ID through `__ockl_get_local_id`
(which lowers to the work-item intrinsic and equals the lane ID for this
one-wave probe), records the byte offset computed for each loaded coordinate,
and verifies the loaded values. It also reads the blocked layout from the
compiled TTGIR and passes an equivalent native Triton `LinearLayout` through
the RELAY adapter.

Run it from the repository root in a single-GPU Flux allocation:

```bash
module load rocm/7.2.1
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/validate-induced-hypergraph.py
```

The default 128-byte transaction scale should report two matching transaction
IDs for the 256-byte wave load. Use `--json PATH` to retain the compact result
or `--transaction-bytes` to validate another aligned power-of-two scale.

The library-side convention and failure cases require no GPU:

```bash
.venv/bin/python -m unittest tests.test_triton_hypergraph -v
```

Automatic extraction of arbitrary instruction cohorts is intentionally left
out of stage 0. Stage 1 can build directly on `InducedMemoryEvent.event` once
those cohorts are selected from a kernel.
