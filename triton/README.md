# Triton experiments

This directory pins the upstream Triton compiler and TritonBench as Git
submodules and contains the Tuolumne-specific setup for the initial RELAY
experiments.

## Repository setup

From the RELAY repository root, initialize the two top-level submodules with
shallow history:

```bash
git submodule update --init --depth 1 triton/triton-lang triton/tritonbench
```

TritonBench also declares optional kernel-library submodules. The baseline does
not initialize them because vector add, softmax, and GEMM use TritonBench's core
kernels. Initialize an individual nested submodule later if an experiment needs
it.

## Install on Tuolumne

Run the installer on a login node; it does not need a GPU:

```bash
bash triton/install-tuolumne.sh
```

The installer uses uv and installs into the environment reached through
`triton/.venv`, including when that path is a symlink to a workspace with more
capacity. It preserves an existing environment; if the path does not exist, it
uses `uv venv` with RELAY's `.venv/bin/python`. It then installs the ROCm
PyTorch nightly, replaces PyTorch's bundled Triton with the pinned editable
`triton-lang` checkout, and installs TritonBench's core AMD dependencies. It
leaves the RELAY environment unchanged and does not require pip in the Triton
environment.

The default is `rocm/7.2.1` with the PyTorch `rocm7.2` nightly channel. This
matches the HIP version recognized by the pinned TritonBench revision and is
available on Tuolumne. Settings can be overridden, for example:

```bash
RELAY_TRITON_ROCM_MODULE=rocm/7.0.2 \
RELAY_TRITON_TORCH_INDEX=rocm7.0 \
    bash triton/install-tuolumne.sh
```

`RELAY_TRITON_MAX_JOBS` controls Triton's parallel build width and defaults to
8. `RELAY_TRITON_VENV` and `RELAY_TRITON_CACHE_DIR` can override the install and
cache locations. `RELAY_TRITON_UV` can select a non-default uv executable.

The large CMake build defaults to `triton-lang-build` beside the resolved
virtual-environment target. Thus a `triton/.venv` symlink into `/usr/WS1` also
places the build in `/usr/WS1`. `RELAY_TRITON_BUILD_DIR` can select a different
location. If an earlier attempt left `triton/triton-lang/build` on the project
filesystem, the installer moves it beside the virtual environment before
starting a fresh external build. The archived partial build is not reused
because CMake caches absolute paths; it can be removed after installation
succeeds.

The installer also places the compiler cache and its temporary files beside the
resolved environment target. This prevents Tuolumne's ccache wrapper from
defaulting to `~/.ccache` on the project filesystem. Override these locations
with `RELAY_TRITON_CCACHE_DIR` and `RELAY_TRITON_CCACHE_TEMPDIR`.

Triton is built with clang and lld by default. Tuolumne's system `c++` is GCC 8
and cannot link the newer C++ compatibility symbols used by Triton's downloaded
LLVM archives. Set `RELAY_TRITON_BUILD_WITH_CLANG_LLD=OFF` only when supplying a
different compatible compiler toolchain.

uv uses copy mode by default in this script because its cache and the workspace
may not support hardlinks across their mount points. Override this with
`RELAY_TRITON_UV_LINK_MODE` if needed.

## Install and run on Matrix

Matrix uses an entirely separate CUDA installation under `triton/.venv-matrix`:

```bash
bash triton/install-matrix.sh
srun -n1 -G1 -p pdebug -t 00:05:00 bash triton/run-baseline-matrix.sh
```

The installer creates the environment in
`/usr/WS1/$USER/record-replay/relay/triton/.venv-matrix` and symlinks it into
the repository. It also uses Matrix-only build, ccache, uv-cache, and runtime-cache
paths. Because an editable Triton build places its platform-specific extension
in its source tree, the installer makes a commit-pinned source clone in the
same `/usr/WS1` directory. This prevents the CUDA build from replacing the
ROCm extension used by `triton/.venv` on Tuolumne.

The defaults are the `cuda/13.1.1` module and the PyTorch `cu130` nightly
channel. Override them with `RELAY_TRITON_MATRIX_CUDA_MODULE` and
`RELAY_TRITON_MATRIX_TORCH_INDEX`. Other Matrix-specific locations and build
settings use the `RELAY_TRITON_MATRIX_*` variables documented in
`install-matrix.sh`. The installer enables uv's native TLS mode by default so
package downloads use Matrix's system certificate store; set
`RELAY_TRITON_MATRIX_UV_NATIVE_TLS=false` to override it.

### Profile Stage 1 quotient levels on Matrix

Submit one five-minute, one-H100 Slurm job for each targeted Stage 1 kernel:

```bash
bash triton/submit-stage1-quotient-level-counters-matrix.sh
```

The Matrix jobs use Nsight Compute 2025.3 and the H100 counter-native quotient
scale, `Q_32B`. They write `*-q32b-matrix.json`, `*-q32b-matrix.csv`, profile,
and log paths, leaving both Tuolumne and the existing `Q_128B` Matrix reports
unchanged. The recorded counters are:

- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` for first-level read work;
- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` for global-load requests;
- `lts__t_sectors_srcunit_tex_op_read.sum` for both L1-to-L2 read traffic and
  L2 read work;
- `lts__t_sectors_op_read_lookup_miss.sum` for L2 read misses;
- `dram__bytes_read.sum` for HBM read volume; and
- `gpu__time_duration.sum` for kernel duration.

The JSON field `first_level_memory_accesses` records the first counter with
`native_counter` and `native_unit` set to the exact metric and
`32-byte global-load L1TEX sector`. `sectors_per_request` is derived per kernel
dispatch by dividing global-load sectors by global-load requests. Nsight
Compute cache flushing is disabled so the worker's explicit warmup controls
the warm-cache experiment.

The compiled-code guard parses PTX `ld.global*` and `st.global*` opcodes,
including predicated instructions. Candidate and reference load/store opcode
maps must match, so the H100 structural check is no longer vacuous.

Jobs are resumable from completed per-profile checkpoints. Run the submission
command again if a case reaches the five-minute limit; already completed
profiles are retained.

The submission defaults to `pdebug` for five minutes. If that partition stays
occupied, submit the same resumable jobs to `pbatch` with a longer user-run
allocation:

```bash
RELAY_TRITON_MATRIX_COUNTER_PARTITION=pbatch \
RELAY_TRITON_MATRIX_COUNTER_WALL_TIME=00:30:00 \
bash triton/submit-stage1-quotient-level-counters-matrix.sh
```

## Validate the induced RELAY hypergraph

The stage-0 probe checks that a compiled one-wave Triton tile load and RELAY's
copy of its LinearLayout agree on lane ownership, logical coordinates, byte
offsets, transaction grouping, and quotient count:

```bash
module load rocm/7.2.1
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/validate-induced-hypergraph.py
```

See [Triton integration stage 0](../docs/triton-stage0.md) for the validation
contract, library API, and current scope.

## Solve the execution-conditioned quotient problem

The Stage 1 experiment extracts the compiled one-wave layout for a symmetric
row/column load, induces the exact RELAY hyperedges, and solves for the best
canonical layout inside its 64x64 reuse tile. Tiles keep a fixed row-major
outer order. The selected prepacked layout is correctness-checked and timed
against Triton's default flat row-major matrix:

```bash
module load rocm/7.2.1
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1.py \
  --json triton/results/stage1.json
```

See [Triton integration stage 1](../docs/triton-stage1.md) for the objective,
experiment controls, JSON contents, and boundary with the later fiber search.

### Measure Stage 1 ranking quality

`run-stage1.py` benchmarks every candidate retained by the Stage 1 solver and
reports top-1 through top-5 regret, tie-aware Spearman correlation, equal-score
runtime spread, and same-flag spread when distinct realizations are present.
Top-k and correlation first deduplicate identical physical mappings while
preserving their first solver occurrence. It also records loaded-kernel
register and spill counts, shared memory, code-object and IR sizes, and
assembly opcode counts for every candidate.

Stage-1 workers search only canonical inner layouts and compose them with a
fixed row-major order across tiles. The kernel-breadth worker treats the inner
tile as an explicit hypothesis sweep, retains the best canonical result for
each shape plus flat row-major, and deduplicates physical mappings for ranking.
Full column-major is not searched as an outer layout. Results record
`inner_tile_shapes`, each candidate's `inner_tile_shape` and `inner_word`, the
`fixed_outer_order`, and the complete physical `word` and `a_rows` separately.

Use the sweep driver for the longer confirmation experiment across matrix sizes
and fresh Python processes:

```bash
module load rocm/7.2.1
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-ranking.py \
  --matrix-sizes 512 1024 2048 \
  --process-launches 3 \
  --json triton/results/stage1-ranking.json --quiet
```

The sweep defaults to 21 samples of 50-launch timing batches and 10 warmup
rounds per candidate. Each matrix-size/process pair is a new child process, and
the aggregate ranking uses the median of its per-process medians.

## Run the targeted Stage 1 kernel suite

The targeted suite exercises four access regimes without depending on broad
TritonBench operator coverage:

- contiguous vector addition as a negative control;
- a custom 32x32 tile using four warps and four register elements per lane;
- GESUMMV with A and B solved independently and measured separately and jointly;
- a fixed 32x32x32 FP16 GEMM configuration with only B prepacked.

Run three independent processes with longer timing batches:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-suite-sweep.py \
  --process-launches 3 \
  --json triton/results/stage1-suite-sweep.json --quiet
```

The worker can also be run directly for compilation or correctness probes with
`triton/run-stage1-suite.py`. Packing, reference construction, compilation, and
validation are excluded from every timing interval.

## Complete the Stage 1.5 GEMM ranking

Stage 1.5 benchmarks all eight retained B layouts for the fixed GEMM and then
profiles the default and LAQS-selected layouts. Run timing and counter
collection separately so both jobs remain below the five-minute limit:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --process-launches 3 \
  --json triton/results/stage15-gemm-ranking.json --quiet

flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --ranking-json triton/results/stage15-gemm-ranking.json \
  --profile --json triton/results/stage15-gemm.json --quiet
```

The result contains raw timing samples and full codegen statistics for every
candidate, aggregate top-1 through top-5 regret and rank correlation, and 20
raw steady-state counter dispatches for each profiled layout. Counter
collection uses the four-pass configuration in `triton/rocprof-stage15.txt`.

Add LinearLayout-induced per-lane register-ownership fibers at their exact
byte footprint with `--register-fibers`. The issue and register components are
combined as equal-weight normalized excesses so their different edge counts do
not set their relative importance. `--profile-all` profiles every retained
mapping and reports tie-aware correlations for the issue quotient, register
component, and combined score:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --register-fibers --process-launches 3 \
  --json triton/results/stage15-gemm-register-fibers-mi300a.json --quiet

flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --ranking-json triton/results/stage15-gemm-fresh-baseline-mi300a.json \
  --profile-all \
  --json triton/results/stage15-gemm-fiber-counter-panel-mi300a.json --quiet
```

The initial MI300A result is summarized in
[`results/stage15-gemm-register-fibers-mi300a.md`](results/stage15-gemm-register-fibers-mi300a.md).

Use `--hardware-hierarchy` to construct cumulative LinearLayout fibers for
register ownership, lane issues, per-warp fragments, and the full CTA
fragment. Each family is scored at several byte scales and combined as
`tau * normalized_excess`. The experiment also evaluates all permutations of
the nonempty register/lane/warp LinearLayout bases as low-to-high memory
directions, because those layouts are not all expressible by the canonical
word grammar:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --hardware-hierarchy --hardware-hierarchy-profile mi300a \
  --process-launches 3 \
  --json triton/results/stage15-gemm-hardware-hierarchy-mi300a.json --quiet

flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage15-gemm-sweep.py \
  --hardware-hierarchy --hardware-hierarchy-profile mi300a \
  --ranking-json triton/results/stage15-gemm-hardware-hierarchy-mi300a.json \
  --profile-all \
  --json triton/results/stage15-gemm-hardware-hierarchy-counter-panel-mi300a.json \
  --quiet
```

The MI300A hierarchy experiment and its tuned weights are summarized in
[`results/stage15-gemm-hardware-hierarchy-mi300a.md`](results/stage15-gemm-hardware-hierarchy-mi300a.md).

## Run the Stage 1 breadth experiments

The GEMM breadth driver ranks all eight retained B layouts across square,
skinny, warm-cache, cache-thrashed, transposed-storage, and fixed block/warp
regimes. It is resumable: divide the named cases across multiple five-minute
allocations and keep the same results directory.

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-gemm-breadth.py \
  --cases square_512_warm square_1024_warm square_2048_warm \
  --process-launches 3 \
  --json triton/results/stage1-gemm-breadth.json --quiet
```

The focused persistent-operand suite ranks bias-ReLU, biased softmax,
embedding bag, GEMV, MVT, GESUMMV, and five-point stencil cases:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-kernel-breadth.py \
  --process-launches 3 \
  --json triton/results/stage1-kernel-breadth.json --quiet
```

Both aggregate the median runtime from three independent process launches and
report the searched tile hypotheses and fixed outer order, default/selected
quotient, top-1 through top-5 regret, runtime, no-change status, rank
correlation, raw timing, and complete per-candidate codegen statistics.
See [Triton integration stage 1](../docs/triton-stage1.md) for the exact case
catalog and cache-control semantics.

### Add per-location temporal fibers

The kernel-breadth worker also supports a space-time quotient experiment.
`--temporal-mode union` adds issue and temporal hyperedges to one component;
`--temporal-mode split` reports them separately while minimizing their equal
raw-count sum. Both retain runs and XORs only as tie-breakers.

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-kernel-breadth.py \
  --temporal-mode split --process-launches 3 \
  --results-dir triton/results/stage1-kernel-temporal-split-cases \
  --json triton/results/stage1-kernel-temporal-split.json --quiet
```

GEMV, MVT, and GESUMMV use exact 32-step non-overlapping loop windows;
stencil neighborhoods are compressed into exact XOR-translation classes.
Results record issue and temporal scores separately for every candidate even
in union mode. See [Triton integration stage 1](../docs/triton-stage1.md) for
the schedule coverage, union/split equivalence check, and measured outcomes.

## Validate quotient locality with hardware counters

`run-stage1-locality-counters.py` reruns the inner-tile DP, profiles the
row-major and selected mappings in separate processes, and checkpoints every
layout profile. It collects three gfx942 replay passes covering TCP/L1 cache
accesses, TCP-to-TCC reads and writes, L2 requests/hits/misses, and HBM bytes.
Only the final target dispatches are summarized; solver, compilation,
correctness, warmup, and cache-thrash dispatches are excluded by selecting the
last target dispatches and by the rocprof kernel filter.

One independent default/selected pair in both cache regimes is a useful first
pass. Run a few cases per five-minute allocation; rerunning the same command
continues from the per-profile checkpoints:

```bash
module load rocm/7.0.2
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-locality-counters.py \
  --cases bias_relu softmax_bias embedding_bag \
  --cache-modes warm thrashed --max-profiles 6

flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-locality-counters.py \
  --cases gemv mvt gesummv stencil5 \
  --cache-modes warm thrashed --max-profiles 6
```

Repeat the commands until `missing_profiles` is empty. For stronger process
replication, add `--profile-launches 3`; already completed launch 1 profiles
are reused. The primary experiment uses `--temporal-mode issue`, because its
issue quotient is the direct 128-byte transaction predictor. Union and split
selections can be profiled into independent checkpoint namespaces by adding
`--temporal-mode union` or `--temporal-mode split` and choosing a distinct
aggregate `--json` path.

The aggregate JSON retains raw dispatch counters and writes a flat CSV beside
it. Render the paper-ready warm/cache-thrashed comparison heatmap without a
GPU:

```bash
.venv/bin/python triton/plot-stage1-locality-counters.py \
  triton/results/stage1-locality-counters.json \
  --output-dir triton/results/stage1-locality-counters-plots
```

The quotient counts transactions for only the persistent target operand,
whereas the counters cover the full kernel. Fixed input, output, and
instruction traffic is therefore expected to dilute the measured reduction.
Warm-cache profiles test request formation; cache-thrashed profiles additionally
make L2 misses and HBM traffic observable. Compare the issue quotient first
with total TCP cache-line accesses, the closest counter for transaction
formation. TCP-to-TCC reads, L2 misses, and HBM bytes are progressively
downstream and may stay flat when the new layout turns repeated line accesses
into TCP hits. None of these reductions is expected to be proportional to
whole-kernel runtime speedup.

### Sweep quotient levels and persistent tile layouts

Two resumable counter experiments test the quotient model directly. The first
fixes one persistent tile, exhaustively enumerates its canonical grammar,
deduplicates physical mappings, and retains the mapping with the fewest full
address-expression runs at every distinct issue-quotient level. The MI300A
runner defaults to the counter-native `Q_64B` scale; the H100 runner defaults
to `Q_32B`. For GESUMMV's 64x64 tile each still checks all 924 canonical
mappings. The default uses three independent warm-cache profiler launches per
mapping, with 20 final target dispatches and cyclically rotated mapping order.

Each profile contains four gfx942 counter passes. The worker
rejects a mapping unless its extracted execution LinearLayout and assembly load
opcode counts equal the row-major reference and it compiles without spills.
Run at most three profiles in each five-minute allocation and repeat the command
until the report says `complete: true`:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-quotient-level-counters.py \
  --case gesummv --tile-shape 64 64 --max-profiles 3
```

The MI300A profiles also collect `TCP_TOTAL_READ_sum` and `TCC_READ_sum` in a
fourth pass for paired first- and second-level read-activity figures. Recollect
the complete native-scale GEMV report on one exclusive node with:

```bash
flux run -N 1 -n 1 -g 1 -x -t 30m \
  triton/.venv/bin/python triton/run-stage1-quotient-level-counters.py \
  --case gemv --tile-shape 64 64 --transaction-bytes 64 \
  --profile-launches 3 --rerun
```

Render the paired plots from the resulting report with:

```bash
.venv/bin/python triton/plot-stage1-quotient-level-counters.py \
  triton/results/stage1-gemv-quotient-level-counters-q64b-mi300a.json \
  --metric TCP_TOTAL_READ_sum \
  --output triton/results/stage1-gemv-quotient-level-counters-mi300a-tcp-total-read.pdf

.venv/bin/python triton/plot-stage1-quotient-level-counters.py \
  triton/results/stage1-gemv-quotient-level-counters-q64b-mi300a.json \
  --metric TCC_READ_sum \
  --output triton/results/stage1-gemv-quotient-level-counters-mi300a-tcc-read.pdf
```

Then render the per-issue quotient/TCP relationship and Spearman correlation
without a GPU. Pass `--normalize-per-issue` to express TCP accesses per dynamic
issue cohort; omit it to retain raw per-kernel counter values. The completed
GESUMMV measurements were identical across all three profiler launches at every
quotient level, so their zero-height min--max ranges are omitted from the plot;
this is worth stating in the figure caption.

```bash
.venv/bin/python triton/plot-stage1-quotient-level-counters.py \
  triton/results/stage1-gesummv-quotient-level-counters-q64b-mi300a.json \
  --normalize-per-issue
```

Every retained candidate preserves its inner layout word from least- to
most-significant address bit. Add `--label-layout-words` to show those words
aligned beneath their points; the default output name gains a distinct
`-layout-words.pdf` suffix.

For the supplementary 128-byte view, pass the preserved
`stage1-gesummv-quotient-level-counters.json` report instead. A fresh explicit
`--transaction-bytes 128` MI300A run writes a separate `*-q128b-mi300a`
report. The corresponding preserved H100 report ends in `*-matrix.json`, and
a fresh explicit H100 run writes `*-q128b-matrix.json`.

Submit complete native-scale runs for all seven kernels on separate exclusive
nodes with a longer, non-`pdebug` allocation:

```bash
bash triton/submit-stage1-quotient-level-counters.sh
```

The submission script requests one task and one GPU on each exclusive node,
uses a 30-minute wall time, and writes resumable `*-q64b-mi300a` profiles and
reports. The existing `Q_128B` files are preserved for supplementary
correlations. Set `RELAY_QUOTIENT_COUNTER_WALL_TIME` to override the wall time.

The scale-selection rule is fixed by each measured counter's documented native
accounting unit: 64-byte vL1D accesses on MI300A and 32-byte global-load L1TEX
sectors on H100. Production layout searches may continue to use the independent
128-byte objective; the native-scale counter experiment asks whether quotient
cardinality predicts hardware-observable first-level work units.

The second experiment uses every inner-tile hypothesis already declared by
the selected kernel case. It retains one minimum-run representative per
tile/quotient pair and deduplicates identical full mappings across tiles. The
GESUMMV panel contains 27 mappings across 64x1 through 64x64 tiles. Its default
is one isolated profiler launch per mapping; repeat this command until complete:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-layout-counter-scatter.py \
  --case gesummv --max-profiles 6

.venv/bin/python triton/plot-stage1-layout-counter-scatter.py \
  triton/results/stage1-gesummv-layout-counter-scatter.json
```

Both runners accept any case in the seven-kernel suite. Use `--case` and, for
the fixed-level experiment, a rank-matching `--tile-shape`; case-specific
checkpoint and aggregate paths are selected automatically. `--metric` changes
the scatter plot's hardware measure, and `--profile-launches 3` adds independent
launch replication to the cross-tile experiment.

### Profile random tile layouts

The less restrictive random-layout experiment samples a tile shape uniformly
from the hypotheses declared by each kernel, then samples an invertible binary
inner address matrix uniformly for that tile. This includes non-canonical XOR
layouts, provides useful variation for one-dimensional kernels, and does not
select layouts by quotient score or address-expression run count. Sampling is
without replacement over complete physical mappings and is reproducible with
`--seed`.

Profile 100 GESUMMV layouts at `Q_64B` with:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage1-random-layout-counters.py \
  --kernel gesummv --byte-level 64 --layouts 100 --seed 0 \
  --max-profiles 6
```

The run is resumable, so repeat the command until the report is complete. The
JSON and CSV names include the kernel and quotient byte level. Each observation
records the sampled tile, inner matrix, complete address matrix, quotient score,
code-generation complexity, and collected counters. The aggregate statistics
include the tie-aware Spearman correlation between quotient score and
`TCP_TOTAL_CACHE_ACCESSES_sum`.

The existing scatter renderer accepts random-layout reports and recomputes
Spearman correlation for any selected metric:

```bash
.venv/bin/python triton/plot-stage1-layout-counter-scatter.py \
  triton/results/stage1-gesummv-random-layout-counters-q64b-mi300a.json
```

Submit one exclusive, one-task, one-GPU Flux job for every kernel at 32, 64,
and 128 bytes with:

```bash
bash triton/submit-stage1-random-layout-counters.sh
```

The launcher defaults to 100 layouts, seed 0, one profiler launch, and a
two-hour wall time. Override these with `RELAY_RANDOM_LAYOUT_COUNT`,
`RELAY_RANDOM_LAYOUT_SEED`, `RELAY_RANDOM_LAYOUT_PROFILE_LAUNCHES`, and
`RELAY_RANDOM_LAYOUT_WALL_TIME`. `RELAY_RANDOM_LAYOUT_BYTE_LEVELS` accepts a
space-separated list, and `RELAY_RANDOM_LAYOUT_QUEUE` selects a Flux queue.
Re-running the launcher resumes completed per-layout checkpoints.

## Run the controlled Stage-2 probe

The probe materializes 28 sparse shear realizations of the Stage-1-selected
2048-square GEMM B flag without adding fibers to the solver DP. It checks exact
flag and quotient invariance, records codegen and the MI300A resource-service
sketch separately, and benchmarks cache-thrashed GEMMs in three fresh
processes:

```bash
flux run -n1 -g1 -t 5m -q pdebug \
  triton/.venv/bin/python triton/run-stage2-probe-sweep.py \
  --process-launches 3 \
  --json triton/results/stage2-probe.json --quiet
```

The process checkpoint directory is reused, allowing the three launches to be
split across separate five-minute allocations. The aggregate reports runtime
spread, service/runtime and codegen/runtime rank correlations, service-optimal
regret, raw timings, and the explicit Stage-2 gate.

The current result preserves the flag and quotient for all 28 realizations and
shows a 3.11% runtime spread, but the service/runtime rank correlation is
-0.054 and the unsheared identity is fastest in the aggregate. The Stage-2
gate is therefore false; see the integration document and
`triton/results/stage2-probe.json` for the complete candidate data.

## Collect the initial baseline

Submit one task on one GPU for at most five minutes:

```bash
flux run -n1 -g1 -t 5m -q pdebug ./triton/run-baseline-tuolumne.sh
```

For isolated performance measurements, an exclusive node may be requested
instead. The runner masks the process to the first GPU exposed by Flux, or GPU
0 when the exclusive allocation exposes all GPUs:

```bash
flux run -N1 -x -t 5m -q pdebug ./triton/run-baseline-tuolumne.sh
```

Set `RELAY_TRITON_GPU` to select another physical GPU index or UUID. The runner
records both Flux's original device mask and the final `ROCR_VISIBLE_DEVICES`
selection in `environment.txt`.

The run compares PyTorch and Triton implementations for three representative
kernels:

- vector add at 1M, 2M, and 4M elements;
- FP16 softmax at 4096 x 1024;
- FP16 square GEMM at 1024, 2048, and 4096.

Each run creates `triton/results/baseline-<UTC timestamp>/` containing CSV and
JSON measurements, console logs, and `environment.txt` with the exact source
commits, package versions, loaded modules, and GPU identity. Structured results
are left visible to Git so they can be committed with the experiment; console
logs and the experiment virtual environment are ignored.
