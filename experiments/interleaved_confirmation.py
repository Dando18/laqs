#!/usr/bin/env python3
"""Confirm selected G_S timings with randomized paired measurement rounds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from relay.evaluator_layout import canonical_layout, layout_function


DEFAULT_INPUT = REPOSITORY_ROOT / "results" / "standard_scoring_mi300a.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "interleaved_confirmation_mi300a.json"
DEFAULT_INSTANCES = (("gemm", 512), ("gesummv", 1024))


@dataclass(frozen=True)
class Candidate:
    word: str
    roles: tuple[str, ...]
    sweep_time_ms: float


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_instance(value: str) -> tuple[str, int]:
    try:
        kernel, size = value.lower().split(":", 1)
        n = positive_integer(size)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected KERNEL:N") from error
    if kernel not in {"gemm", "gesummv"}:
        raise argparse.ArgumentTypeError(
            "the interleaved harness currently supports gemm and gesummv"
        )
    return kernel, n


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="completed exhaustive G_S summary JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="confirmation report (default: %(default)s)",
    )
    parser.add_argument(
        "--instance",
        action="append",
        type=parse_instance,
        default=None,
        metavar="KERNEL:N",
        help="instance to confirm; repeat as needed",
    )
    parser.add_argument(
        "--oracle-top",
        type=positive_integer,
        default=3,
        help="number of exhaustive-oracle leaders to include (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=positive_integer,
        default=24,
        help="paired randomized measurement rounds (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=positive_integer,
        default=50,
        help="kernel launches per timed burst (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=positive_integer,
        default=5,
        help="immediate warmup launches per timed burst (default: %(default)s)",
    )
    parser.add_argument(
        "--priming-cycles",
        type=positive_integer,
        default=2,
        help="untimed whole-panel cycles before measurement (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260825,
        help="balanced-order and bootstrap seed (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="HIP device ordinal (default: %(default)s)",
    )
    parser.add_argument(
        "--compiler",
        default="/opt/rocm-7.0.2/bin/hipcc",
        help="HIP compiler (default: %(default)s)",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        help="GPU offload architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="retain generated source and executables in this directory",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="generate and compile the harnesses without running them",
    )
    return parser.parse_args(argv)


def load_summaries(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    records = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if (
                record.get("grammar") != "G_S"
                or record.get("complete") is not True
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected a complete G_S record"
                )
            key = (str(record["kernel"]), int(record["matrix_size"]))
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate instance {key}")
            records[key] = record
    return records


def select_candidates(
    record: Mapping[str, object], oracle_top: int
) -> tuple[Candidate, ...]:
    oracle = record["oracle"]
    frontier = record["frontier"]
    assert isinstance(oracle, dict) and isinstance(frontier, dict)
    ordered: dict[str, dict[str, object]] = {}
    for rank, layout in enumerate(oracle["top_layouts"][:oracle_top], 1):
        word = str(layout["word"])
        ordered[word] = {
            "roles": [f"oracle_top_{rank}"],
            "time": float(layout["timing"]["median_ms"]),
        }
    base_frontier = record.get("base_colored_frontier")
    if isinstance(base_frontier, dict):
        for layout in base_frontier.get("layouts", ()):
            word = str(layout["word"])
            entry = ordered.setdefault(
                word,
                {
                    "roles": [],
                    "time": float(layout["timing"]["median_ms"]),
                },
            )
            entry["roles"].append("base_colored_frontier")
    for layout in frontier["layouts"]:
        word = str(layout["word"])
        entry = ordered.setdefault(
            word,
            {
                "roles": [],
                "time": float(layout["timing"]["median_ms"]),
            },
        )
        entry["roles"].append("analytical_frontier")
    return tuple(
        Candidate(
            word,
            tuple(str(role) for role in entry["roles"]),
            float(entry["time"]),
        )
        for word, entry in ordered.items()
    )


def _offset_functions(candidates: Sequence[Candidate]) -> str:
    return "\n\n".join(
        layout_function(f"offset_{index}", canonical_layout(item.word, item.word))
        for index, item in enumerate(candidates)
    )


def _kernel_switch(kernel: str, count: int) -> str:
    arguments = (
        "buffers[buffer_slot].c, buffers[buffer_slot].a, "
        "buffers[buffer_slot].b, 1.0, 0.0, n"
        if kernel == "gemm"
        else (
            "buffers[buffer_slot].a, buffers[buffer_slot].b, device_x, "
            "buffers[buffer_slot].y, 1.25, -0.75, n"
        )
    )
    return "\n".join(
        f"    case {index}: hipLaunchKernelGGL(kernel_{index}, grid, block, 0, 0, "
        f"{arguments}); break;"
        for index in range(count)
    )


def _host_offset_switch(count: int) -> str:
    cases = "\n".join(
        f"    case {index}: return offset_{index}(first, second, n);"
        for index in range(count)
    )
    return f"""static uint64_t host_offset(
    int id, uint32_t first, uint32_t second, uint32_t n) {{
  switch (id) {{
{cases}
    default: std::abort();
  }}
}}"""


def _measurement_driver(
    candidates: Sequence[Candidate], args: argparse.Namespace
) -> str:
    words = ",\n      ".join(f'"{item.word}"' for item in candidates)
    return f"""
  constexpr int rounds = {args.rounds};
  constexpr int iterations = {args.iterations};
  constexpr int warmup = {args.warmup};
  constexpr int priming_cycles = {args.priming_cycles};
  const char* words[case_count] = {{
      {words}
  }};

  for (int cycle = 0; cycle < priming_cycles; ++cycle) {{
    for (int id = 0; id < case_count; ++id) {{
      for (int launch = 0; launch < warmup; ++launch) launch_case(id, 0);
      HIP_CHECK(hipDeviceSynchronize());
    }}
  }}

  hipEvent_t start{{}}, stop{{}};
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<int> base(case_count);
  std::iota(base.begin(), base.end(), 0);
  std::mt19937 generator({args.seed}u);
  int round = 0;
  while (round < rounds) {{
    std::shuffle(base.begin(), base.end(), generator);
    for (int shift = 0; shift < case_count && round < rounds; ++shift, ++round) {{
      for (int position = 0; position < case_count; ++position) {{
        const int id = base[(position + shift) % case_count];
        for (int launch = 0; launch < warmup; ++launch) launch_case(id, 0);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start, 0));
        for (int launch = 0; launch < iterations; ++launch) launch_case(id, 0);
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        std::printf(
            "{{\\\"record_type\\\":\\\"measurement\\\","
            "\\\"round\\\":%d,\\\"position\\\":%d,\\\"case\\\":%d,"
            "\\\"word\\\":\\\"%s\\\",\\\"time_ms\\\":%.9g}}\\n",
            round, position, id, words[id], elapsed_ms / iterations);
      }}
      std::fflush(stdout);
    }}
  }}
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
"""


def _gemm_correctness_driver() -> str:
    return r"""
  const uint32_t check_i[] = {0, n / 2, n - 1};
  const uint32_t check_j[] = {0, n / 2, n - 1};
  for (int id = 0; id < case_count; ++id) {
    launch_case(id, id);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> host_c(elements);
    HIP_CHECK(hipMemcpy(
        host_c.data(), buffers[id].c, bytes, hipMemcpyDeviceToHost));
    for (int point = 0; point < 3; ++point) {
      const uint32_t i = check_i[point];
      const uint32_t j = check_j[point];
      double expected = 0.0;
      for (uint32_t k = 0; k < n; ++k) {
        const double a =
            static_cast<double>(
                static_cast<int>((i * 17u + k * 13u) % 101u) - 50) /
            101.0;
        const double b =
            static_cast<double>(
                static_cast<int>((k * 11u + j * 19u) % 103u) - 51) /
            103.0;
        expected = std::fma(a, b, expected);
      }
      const double observed = host_c[host_offset(id, i, j, n)];
      const double error = std::abs(observed - expected);
      if (error > 1.0e-10 * std::max(1.0, std::abs(expected))) {
        std::fprintf(stderr, "correctness failure for case %d\n", id);
        return EXIT_FAILURE;
      }
    }
  }
"""


def _gesummv_correctness_driver() -> str:
    return r"""
  const uint32_t check_i[] = {0, n / 2, n - 1};
  for (int id = 0; id < case_count; ++id) {
    launch_case(id, id);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> host_y(n);
    HIP_CHECK(hipMemcpy(
        host_y.data(), buffers[id].y, vector_bytes, hipMemcpyDeviceToHost));
    for (uint32_t i : check_i) {
      double sum_a = 0.0;
      double sum_b = 0.0;
      for (uint32_t j = 0; j < n; ++j) {
        const double a =
            static_cast<double>(
                static_cast<int>((i * 17u + j * 13u) % 101u) - 50) /
            101.0;
        const double b =
            static_cast<double>(
                static_cast<int>((i * 11u + j * 19u) % 103u) - 51) /
            103.0;
        sum_a = std::fma(a, host_x[j], sum_a);
        sum_b = std::fma(b, host_x[j], sum_b);
      }
      const double expected = 1.25 * sum_a - 0.75 * sum_b;
      const double error = std::abs(host_y[i] - expected);
      if (error > 1.0e-10 * std::max(1.0, std::abs(expected))) {
        std::fprintf(stderr, "correctness failure for case %d\n", id);
        return EXIT_FAILURE;
      }
    }
  }
"""


def _common_prefix(candidates: Sequence[Candidate]) -> str:
    return f"""#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define HIP_CHECK(command)                                                   \\
  do {{                                                                       \\
    const hipError_t error = (command);                                      \\
    if (error != hipSuccess) {{                                               \\
      std::fprintf(stderr, "HIP error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                   hipGetErrorString(error));                               \\
      std::exit(EXIT_FAILURE);                                               \\
    }}                                                                        \\
  }} while (0)

{_offset_functions(candidates)}

{_host_offset_switch(len(candidates))}
"""


def generate_gemm_source(
    n: int, candidates: Sequence[Candidate], args: argparse.Namespace
) -> str:
    kernels = "\n\n".join(
        f"""__global__ void kernel_{index}(
    double* __restrict__ c, const double* __restrict__ a,
    const double* __restrict__ b, double alpha, double beta, uint32_t n) {{
  const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n) return;
  double dot = 0.0;
  for (uint32_t k = 0; k < n; ++k) {{
    dot = fma(a[offset_{index}(i, k, n)],
              b[offset_{index}(k, j, n)], dot);
  }}
  const uint64_t c_index = offset_{index}(i, j, n);
  c[c_index] = fma(alpha, dot, beta * c[c_index]);
}}"""
        for index in range(len(candidates))
    )
    switch = _kernel_switch("gemm", len(candidates))
    correctness = _gemm_correctness_driver()
    measurements = _measurement_driver(candidates, args)
    return _common_prefix(candidates) + f"""

{kernels}

struct Buffers {{ double* a; double* b; double* c; }};

int main() {{
  constexpr uint32_t n = {n};
  constexpr int case_count = {len(candidates)};
  HIP_CHECK(hipSetDevice({args.device}));
  hipDeviceProp_t properties{{}};
  HIP_CHECK(hipGetDeviceProperties(&properties, {args.device}));
  std::printf(
      "{{\\\"record_type\\\":\\\"metadata\\\",\\\"device\\\":\\\"%s\\\","
      "\\\"kernel\\\":\\\"gemm\\\",\\\"matrix_size\\\":%u}}\\n",
      properties.name, n);

  const uint64_t elements = static_cast<uint64_t>(n) * n;
  const size_t bytes = static_cast<size_t>(elements * sizeof(double));
  std::vector<Buffers> buffers(case_count);
  for (int id = 0; id < case_count; ++id) {{
    std::vector<double> host_a(elements), host_b(elements), host_c(elements);
    for (uint32_t i = 0; i < n; ++i) {{
      for (uint32_t j = 0; j < n; ++j) {{
        const uint64_t physical = host_offset(id, i, j, n);
        host_a[physical] =
            static_cast<double>(static_cast<int>((i * 17u + j * 13u) % 101u) - 50) /
            101.0;
        host_b[physical] =
            static_cast<double>(static_cast<int>((i * 11u + j * 19u) % 103u) - 51) /
            103.0;
      }}
    }}
    HIP_CHECK(hipMalloc(&buffers[id].a, bytes));
    HIP_CHECK(hipMalloc(&buffers[id].b, bytes));
    HIP_CHECK(hipMalloc(&buffers[id].c, bytes));
    HIP_CHECK(hipMemcpy(buffers[id].a, host_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buffers[id].b, host_b.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buffers[id].c, host_c.data(), bytes, hipMemcpyHostToDevice));
  }}

  const dim3 block(32, 32);
  const dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
  auto launch_case = [&](int id, int buffer_slot) {{
    switch (id) {{
{switch}
      default: std::abort();
    }}
    HIP_CHECK(hipGetLastError());
  }};

{correctness}
{measurements}
  for (Buffers& item : buffers) {{
    HIP_CHECK(hipFree(item.a));
    HIP_CHECK(hipFree(item.b));
    HIP_CHECK(hipFree(item.c));
  }}
  return 0;
}}
"""


def generate_gesummv_source(
    n: int, candidates: Sequence[Candidate], args: argparse.Namespace
) -> str:
    kernels = "\n\n".join(
        f"""__global__ void kernel_{index}(
    const double* __restrict__ a, const double* __restrict__ b,
    const double* __restrict__ x, double* __restrict__ y,
    double alpha, double beta, uint32_t n) {{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double sum_a = 0.0;
  double sum_b = 0.0;
  for (uint32_t j = 0; j < n; ++j) {{
    const double xj = x[j];
    sum_a = fma(a[offset_{index}(i, j, n)], xj, sum_a);
    sum_b = fma(b[offset_{index}(i, j, n)], xj, sum_b);
  }}
  y[i] = alpha * sum_a + beta * sum_b;
}}"""
        for index in range(len(candidates))
    )
    switch = _kernel_switch("gesummv", len(candidates))
    correctness = _gesummv_correctness_driver()
    measurements = _measurement_driver(candidates, args)
    return _common_prefix(candidates) + f"""

{kernels}

struct Buffers {{ double* a; double* b; double* y; }};

int main() {{
  constexpr uint32_t n = {n};
  constexpr int case_count = {len(candidates)};
  HIP_CHECK(hipSetDevice({args.device}));
  hipDeviceProp_t properties{{}};
  HIP_CHECK(hipGetDeviceProperties(&properties, {args.device}));
  std::printf(
      "{{\\\"record_type\\\":\\\"metadata\\\",\\\"device\\\":\\\"%s\\\","
      "\\\"kernel\\\":\\\"gesummv\\\",\\\"matrix_size\\\":%u}}\\n",
      properties.name, n);

  const uint64_t elements = static_cast<uint64_t>(n) * n;
  const size_t matrix_bytes = static_cast<size_t>(elements * sizeof(double));
  const size_t vector_bytes = static_cast<size_t>(n * sizeof(double));
  std::vector<double> host_x(n);
  for (uint32_t j = 0; j < n; ++j) {{
    host_x[j] = static_cast<double>(static_cast<int>((j * 7u) % 37u) - 18) / 37.0;
  }}
  double* device_x = nullptr;
  HIP_CHECK(hipMalloc(&device_x, vector_bytes));
  HIP_CHECK(hipMemcpy(device_x, host_x.data(), vector_bytes, hipMemcpyHostToDevice));

  std::vector<Buffers> buffers(case_count);
  for (int id = 0; id < case_count; ++id) {{
    std::vector<double> host_a(elements), host_b(elements), host_y(n);
    for (uint32_t i = 0; i < n; ++i) {{
      for (uint32_t j = 0; j < n; ++j) {{
        const uint64_t physical = host_offset(id, i, j, n);
        host_a[physical] =
            static_cast<double>(static_cast<int>((i * 17u + j * 13u) % 101u) - 50) /
            101.0;
        host_b[physical] =
            static_cast<double>(static_cast<int>((i * 11u + j * 19u) % 103u) - 51) /
            103.0;
      }}
    }}
    HIP_CHECK(hipMalloc(&buffers[id].a, matrix_bytes));
    HIP_CHECK(hipMalloc(&buffers[id].b, matrix_bytes));
    HIP_CHECK(hipMalloc(&buffers[id].y, vector_bytes));
    HIP_CHECK(hipMemcpy(
        buffers[id].a, host_a.data(), matrix_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        buffers[id].b, host_b.data(), matrix_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        buffers[id].y, host_y.data(), vector_bytes, hipMemcpyHostToDevice));
  }}

  const dim3 block(128);
  const dim3 grid((n + block.x - 1) / block.x);
  auto launch_case = [&](int id, int buffer_slot) {{
    switch (id) {{
{switch}
      default: std::abort();
    }}
    HIP_CHECK(hipGetLastError());
  }};

{correctness}
{measurements}
  for (Buffers& item : buffers) {{
    HIP_CHECK(hipFree(item.a));
    HIP_CHECK(hipFree(item.b));
    HIP_CHECK(hipFree(item.y));
  }}
  HIP_CHECK(hipFree(device_x));
  return 0;
}}
"""


def compile_harness(
    kernel: str,
    n: int,
    candidates: Sequence[Candidate],
    args: argparse.Namespace,
    directory: Path,
) -> tuple[Path, Path, list[str]]:
    generator = generate_gemm_source if kernel == "gemm" else generate_gesummv_source
    source = directory / f"interleaved_{kernel}_n{n}.cpp"
    executable = directory / f"interleaved_{kernel}_n{n}"
    source.write_text(generator(n, candidates, args))
    command = [
        args.compiler,
        str(source),
        "-std=c++17",
        "-O3",
        f"--offload-arch={args.arch}",
        "-o",
        str(executable),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to compile {kernel} N={n}: {detail}")
    return source, executable, command


def run_harness(executable: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    completed = subprocess.run(
        [str(executable)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{executable.name} failed: {detail}")
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    metadata = [record for record in records if record["record_type"] == "metadata"]
    measurements = [
        record for record in records if record["record_type"] == "measurement"
    ]
    if len(metadata) != 1:
        raise ValueError(f"{executable.name}: expected one metadata record")
    return metadata[0], measurements


def bootstrap_mean_interval(
    values: Sequence[float], seed: int, trials: int = 10_000
) -> tuple[float, float]:
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(trials)
    )
    return means[int(0.025 * trials)], means[int(0.975 * trials)]


def analyze_measurements(
    candidates: Sequence[Candidate],
    measurements: Sequence[Mapping[str, object]],
    seed: int,
) -> dict[str, object]:
    by_word: dict[str, dict[int, float]] = {item.word: {} for item in candidates}
    for record in measurements:
        word = str(record["word"])
        round_index = int(record["round"])
        by_word[word][round_index] = float(record["time_ms"])
    round_sets = {tuple(sorted(values)) for values in by_word.values()}
    if len(round_sets) != 1:
        raise ValueError("measurements do not form complete paired rounds")

    summaries = []
    for item in candidates:
        values = list(by_word[item.word].values())
        summaries.append(
            {
                "word": item.word,
                "roles": list(item.roles),
                "sweep_time_ms": item.sweep_time_ms,
                "median_ms": statistics.median(values),
                "mean_ms": statistics.fmean(values),
                "sd_ms": statistics.pstdev(values),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
            }
        )
    summaries.sort(key=lambda item: (item["median_ms"], item["word"]))
    oracle_panel = [
        item
        for item in summaries
        if any(str(role).startswith("oracle_top_") for role in item["roles"])
    ]
    reference = min(oracle_panel, key=lambda item: item["median_ms"])
    reference_rounds = by_word[str(reference["word"])]
    for index, item in enumerate(summaries):
        ratios = [
            by_word[str(item["word"])][round_index]
            / reference_rounds[round_index]
            - 1.0
            for round_index in sorted(reference_rounds)
        ]
        lower, upper = bootstrap_mean_interval(ratios, seed + index)
        item["paired_vs_confirmed_oracle"] = {
            "mean_regret": statistics.fmean(ratios),
            "median_regret": statistics.median(ratios),
            "bootstrap_95pct_mean_regret": [lower, upper],
            "faster_rounds": sum(value < 0.0 for value in ratios),
            "tied_rounds": sum(value == 0.0 for value in ratios),
            "round_count": len(ratios),
        }

    frontier = [
        item for item in summaries if "analytical_frontier" in item["roles"]
    ]
    best_frontier = min(frontier, key=lambda item: item["median_ms"])
    base_frontier = [
        item
        for item in summaries
        if "base_colored_frontier" in item["roles"]
    ]
    sweep_oracle = min(
        item.sweep_time_ms
        for item in candidates
        if any(role.startswith("oracle_top_") for role in item.roles)
    )
    sweep_frontier = min(
        item.sweep_time_ms
        for item in candidates
        if "analytical_frontier" in item.roles
    )
    result = {
        "layout_summaries": summaries,
        "confirmed_oracle_word": reference["word"],
        "confirmed_oracle_median_ms": reference["median_ms"],
        "best_frontier_word": best_frontier["word"],
        "best_frontier_median_ms": best_frontier["median_ms"],
        "frontier_regret": best_frontier["median_ms"] / reference["median_ms"] - 1.0,
        "frontier_paired_vs_confirmed_oracle": best_frontier[
            "paired_vs_confirmed_oracle"
        ],
        "sweep_panel_oracle_ms": sweep_oracle,
        "sweep_frontier_best_ms": sweep_frontier,
        "sweep_frontier_regret": sweep_frontier / sweep_oracle - 1.0,
    }
    if base_frontier:
        best_base = min(base_frontier, key=lambda item: item["median_ms"])
        base_rounds = by_word[str(best_base["word"])]
        fiber_vs_base = [
            by_word[str(best_frontier["word"])][round_index]
            / base_rounds[round_index]
            - 1.0
            for round_index in sorted(base_rounds)
        ]
        lower, upper = bootstrap_mean_interval(fiber_vs_base, seed + len(summaries))
        result.update(
            {
                "best_base_frontier_word": best_base["word"],
                "best_base_frontier_median_ms": best_base["median_ms"],
                "fiber_paired_vs_base_frontier": {
                    "mean_change": statistics.fmean(fiber_vs_base),
                    "median_change": statistics.median(fiber_vs_base),
                    "bootstrap_95pct_mean_change": [lower, upper],
                    "faster_rounds": sum(value < 0.0 for value in fiber_vs_base),
                    "tied_rounds": sum(value == 0.0 for value in fiber_vs_base),
                    "round_count": len(fiber_vs_base),
                },
            }
        )
    return result


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    instances = tuple(dict.fromkeys(args.instance or DEFAULT_INSTANCES))
    summaries = load_summaries(args.input.expanduser().resolve())
    missing = [instance for instance in instances if instance not in summaries]
    if missing:
        raise ValueError(f"input has no completed records for {missing}")

    temporary = None
    if args.build_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="relay-interleaved-")
        build_directory = Path(temporary.name)
    else:
        build_directory = args.build_dir.expanduser().resolve()
        build_directory.mkdir(parents=True, exist_ok=True)

    compiled = []
    panels = {}
    for kernel, n in instances:
        candidates = select_candidates(summaries[(kernel, n)], args.oracle_top)
        source, executable, command = compile_harness(
            kernel, n, candidates, args, build_directory
        )
        compiled.append(
            {
                "kernel": kernel,
                "matrix_size": n,
                "source": str(source),
                "executable": str(executable),
                "command": command,
            }
        )
        panels[(kernel, n)] = candidates
        print(f"Compiled {kernel} N={n}: {len(candidates)} layouts", flush=True)

    if args.compile_only:
        if temporary is not None:
            print("Use --build-dir to retain compile-only artifacts")
        return 0

    groups = []
    for item in compiled:
        key = (str(item["kernel"]), int(item["matrix_size"]))
        metadata, measurements = run_harness(Path(str(item["executable"])))
        analysis = analyze_measurements(panels[key], measurements, args.seed)
        groups.append(
            {
                "kernel": key[0],
                "matrix_size": key[1],
                "device": metadata["device"],
                "candidates": [
                    {
                        "word": candidate.word,
                        "roles": list(candidate.roles),
                        "sweep_time_ms": candidate.sweep_time_ms,
                    }
                    for candidate in panels[key]
                ],
                "measurements": measurements,
                "analysis": analysis,
            }
        )
        paired = analysis["frontier_paired_vs_confirmed_oracle"]
        print(
            f"{key[0]} N={key[1]}: sweep frontier regret "
            f"{100 * analysis['sweep_frontier_regret']:.3f}%, interleaved paired "
            f"{100 * paired['mean_regret']:.3f}%"
            + (
                ", fiber vs base frontier "
                f"{100 * analysis['fiber_paired_vs_base_frontier']['mean_change']:.3f}%"
                if "fiber_paired_vs_base_frontier" in analysis
                else ""
            ),
            flush=True,
        )

    report = {
        "experiment": "interleaved-gs-confirmation",
        "schema_version": 1,
        "configuration": {
            "input": str(args.input.expanduser().resolve()),
            "instances": [list(instance) for instance in instances],
            "oracle_top": args.oracle_top,
            "rounds": args.rounds,
            "iterations_per_burst": args.iterations,
            "immediate_warmup_launches": args.warmup,
            "priming_cycles": args.priming_cycles,
            "timed_buffer_policy": "all layouts share buffer slot 0",
            "order": "seeded shuffled cyclic blocks balanced by position",
            "seed": args.seed,
            "compiler": args.compiler,
            "arch": args.arch,
            "device": args.device,
            "host": socket.gethostname(),
            "flux_job_id": os.environ.get("FLUX_JOB_ID"),
        },
        "groups": groups,
    }
    output = args.output.expanduser().resolve()
    atomic_write_json(output, report)
    print(output)
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
