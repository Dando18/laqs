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
row/column load, induces the exact RELAY hyperedges, solves for the best
canonical quotient-locality layout, and correctness-checks and times that
prepacked layout against Triton's default row-major matrix:

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
reports top-1/top-3 regret, tie-aware Spearman correlation, equal-score runtime
spread, and same-flag spread when distinct realizations are present. It also
records loaded-kernel register and spill counts, shared memory, code-object and
IR sizes, and assembly opcode counts for every candidate.

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
candidate, aggregate top-1/top-3 regret and rank correlation, and 20 raw
steady-state counter dispatches for each profiled layout. Counter collection
uses the four-pass configuration in `triton/rocprof-stage15.txt`.

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
report default/selected quotient, top-1/top-3 regret, runtime, no-change status,
rank correlation, raw timing, and complete per-candidate codegen statistics.
See [Triton integration stage 1](../docs/triton-stage1.md) for the exact case
catalog and cache-control semantics.

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
