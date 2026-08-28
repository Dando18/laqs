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
