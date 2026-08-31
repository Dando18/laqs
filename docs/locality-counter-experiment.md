# LAQS locality hardware-counter experiment

`experiments/locality_counters.py` tests the paired question:

> When LAQS predicts lower quotient locality cost, does the generated program
> issue fewer memory requests at the modeled level?

For each of the five HIP test kernels (ATAX, GEMM, GESUMMV, MVT, and SYRK), the
experiment runs an exact canonical `G_C` count-grid DP for the minimum
`Q_fine`. Because the objective adds across separate allocations, each target
array can be solved independently without changing the exact joint minimum.
Code-generation runs and deterministic DP traversal break exact quotient ties;
hardware peak, area, placement, and runtime do not participate in selection.
The selected result is compared with the same kernel using full row-major
matrix layouts. Both binaries use identical problem data, launch geometry, and
timing settings. Every binary performs its existing layout-bijection and
numerical correctness checks before measurement.

The MI300A profile's fine component is `issue.g64.stream.load.64B`. The primary
hardware comparison is therefore `TCP_TCC_READ_REQ_sum`, the number of read
requests sent from TCP/L1 to TCC/L2. The experiment also records:

| Counter | Interpretation |
| --- | --- |
| `TCP_TOTAL_CACHE_ACCESSES_sum` | TCP cache-line tag accesses (hits and misses) |
| `TCP_TCC_READ_REQ_sum` | TCP/L1-to-TCC/L2 read requests |
| `TCP_TCC_WRITE_REQ_sum` | TCP/L1-to-TCC/L2 write requests |
| `TCC_REQ_sum` | requests processed by L2 tag blocks |
| `TCC_HIT_sum`, `TCC_MISS_sum` | L2 hits and misses |
| `FETCH_SIZE`, `WRITE_SIZE` | KiB transferred at the memory interface |

ROCprof cannot collect all of these counters in one compatible hardware pass.
`experiments/rocprof-locality.txt` uses three passes and ROCprof reruns the same
standalone executable once per pass. The merged CSV is parsed into logical
operations. ATAX's `tmp=A*x` and `y=A^T*tmp` dispatches are summed; each other
test kernel has one dispatch per operation.

Two cache regimes are retained in the JSON:

- `cold_first_operation` is the first target operation after allocations and
  host-to-device setup. It is a cold-start proxy, not a guaranteed flushed-L2
  measurement.
- `steady_state` is the median of the final `samples * iterations` operations,
  after the correctness launch and explicit warmups. This is the primary,
  reproducible comparison.

The counter totals cover the whole kernel operation. Fixed vectors and output
traffic are deliberately left in both sides of the pair, so a predicted matrix
request reduction need not appear as the same percentage reduction in the
whole-program counter.

## Run the experiment

The solver needs no GPU. For a larger size, prepare its resumable checkpoint on
the login node first:

```bash
.venv/bin/python experiments/locality_counters.py \
  --size 512 --prepare-only \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --output results/locality_counters_mi300a.json
```

Then compile, validate, time, and profile all ten layout/kernel pairs in one
short GPU allocation:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  .venv/bin/python experiments/locality_counters.py \
  --size 512 --resume \
  --compiler /opt/rocm-7.0.2/bin/hipcc --arch gfx942 \
  --output results/locality_counters_mi300a.json
```

Every completed pair is checkpointed. `--max-layouts COUNT` can split counter
collection across multiple allocations; repeat the same command with
`--resume`. Repeated `--kernel NAME` options select a subset. Raw merged ROCprof
CSVs are written to `results/locality_counters_mi300a_raw/` by default.

The JSON contains solver search statistics, both layouts, every modeled
quotient component, raw steady-operation counters, paired reductions, and an
aggregate count/correlation summary. Positive reductions mean the selected
layout issued fewer requests than row-major.

## Plot the result

```bash
.venv/bin/python experiments/plot_locality_counters.py \
  results/locality_counters_mi300a.json
```

This writes two PDF figures beside the report: an annotated reduction matrix
and a predicted-versus-measured request scatter. The plotting dependencies are
in the project's `experiments` optional dependency set.
