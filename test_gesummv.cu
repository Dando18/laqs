// Benchmark GESUMMV with standard layouts and layouts emitted by
// examples/gesummv_multi.py.
//
// The generated candidates were synthesized for a 4096x4096 FP64 problem whose
// 128-thread workgroup maps lanes across i.  N may be changed, but must remain
// a multiple of 128 for every compiled layout to tile the matrix exactly.
//
// Build on an MI300A node:
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 test_gesummv.cu -o test_gesummv
//
// Run:
//   ./test_gesummv
//   ./test_gesummv --samples 20 --iterations 1000

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(command)                                                               \
  do {                                                                                   \
    const hipError_t error = (command);                                                   \
    if (error != hipSuccess) {                                                            \
      std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(error));                                             \
      std::exit(EXIT_FAILURE);                                                            \
    }                                                                                    \
  } while (0)

enum LayoutId {
  ROW_MAJOR = 0,
  COLUMN_MAJOR,
  TILE16_ROW,
  TILE16_COLUMN,
  C128X4_IIIIIIIJJ_OUTERJI,
  C64X1_IIIIII_OUTERJI,
  C64X16_JJJJIIIIII_OUTERJI,
  C128X16_JJJJIIIIIII_OUTERJI,
  LIN16X4_FLAG0_OUTERJI,
};

template <int Id>
struct Layout;

template <>
struct Layout<ROW_MAJOR> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    return static_cast<uint64_t>(i) * n + j;
  }
};

template <>
struct Layout<COLUMN_MAJOR> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    return static_cast<uint64_t>(j) * n + i;
  }
};

// Row-major 16x16 tiles, with tiles themselves in row-major order.
template <>
struct Layout<TILE16_ROW> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t tile =
        static_cast<uint64_t>(i >> 4) * (n >> 4) + (j >> 4);
    return (tile << 8) + static_cast<uint64_t>(i & 15u) * 16u + (j & 15u);
  }
};

// Column-major within each 16x16 tile.  The outer tile grid stays row-major.
template <>
struct Layout<TILE16_COLUMN> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t tile =
        static_cast<uint64_t>(i >> 4) * (n >> 4) + (j >> 4);
    return (tile << 8) + static_cast<uint64_t>(j & 15u) * 16u + (i & 15u);
  }
};

// RELAY canonical layout:
//   tile=128x4, word(low->high)=iiiiiiijj, outer_order=j,i.
template <>
struct Layout<C128X4_IIIIIIIJJ_OUTERJI> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t outer =
        static_cast<uint64_t>(i >> 7) * (n >> 2) + (j >> 2);
    const uint32_t inner = (i & 127u) | ((j & 3u) << 7);
    return (outer << 9) | inner;
  }
};

// RELAY canonical layout:
//   tile=64x1, word(low->high)=iiiiii, outer_order=j,i.
template <>
struct Layout<C64X1_IIIIII_OUTERJI> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t outer = static_cast<uint64_t>(i >> 6) * n + j;
    return (outer << 6) | (i & 63u);
  }
};

// RELAY canonical layout:
//   tile=64x16, word(low->high)=jjjjiiiiii, outer_order=j,i.
template <>
struct Layout<C64X16_JJJJIIIIII_OUTERJI> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t outer =
        static_cast<uint64_t>(i >> 6) * (n >> 4) + (j >> 4);
    const uint32_t inner = (j & 15u) | ((i & 63u) << 4);
    return (outer << 10) | inner;
  }
};

// RELAY canonical layout:
//   tile=128x16, word(low->high)=jjjjiiiiiii, outer_order=j,i.
template <>
struct Layout<C128X16_JJJJIIIIIII_OUTERJI> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t outer =
        static_cast<uint64_t>(i >> 7) * (n >> 4) + (j >> 4);
    const uint32_t inner = (j & 15u) | ((i & 127u) << 4);
    return (outer << 11) | inner;
  }
};

// RELAY linear-inner layout:
//   tile=16x4, outer_order=j,i
//   y0=j1, y1=i3, y2=i2, y3=j0, y4=i1, y5=i0.
template <>
struct Layout<LIN16X4_FLAG0_OUTERJI> {
  __host__ __device__ static __forceinline__ uint64_t offset(
      uint32_t i, uint32_t j, uint32_t n) {
    const uint64_t outer =
        static_cast<uint64_t>(i >> 4) * (n >> 2) + (j >> 2);
    const uint32_t inner = ((j & 2u) >> 1) |
                           ((i & 8u) >> 2) |
                           (i & 4u) |
                           ((j & 1u) << 3) |
                           ((i & 2u) << 3) |
                           ((i & 1u) << 5);
    return (outer << 6) | inner;
  }
};

template <int ALayout, int BLayout>
__global__ __launch_bounds__(128) void gesummv_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ x,
    double* __restrict__ y,
    uint32_t n,
    double alpha,
    double beta) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double sum_a = 0.0;
  double sum_b = 0.0;
  for (uint32_t j = 0; j < n; ++j) {
    const double xj = x[j];
    sum_a = fma(a[Layout<ALayout>::offset(i, j, n)], xj, sum_a);
    sum_b = fma(b[Layout<BLayout>::offset(i, j, n)], xj, sum_b);
  }
  y[i] = alpha * sum_a + beta * sum_b;
}

using LaunchFn = void (*)(const double*, const double*, const double*, double*,
                          uint32_t, double, double, hipStream_t);
using OffsetFn = uint64_t (*)(uint32_t, uint32_t, uint32_t);

template <int ALayout, int BLayout>
void launch_gesummv(const double* a, const double* b, const double* x, double* y,
                    uint32_t n, double alpha, double beta, hipStream_t stream) {
  constexpr uint32_t block_size = 128;
  const dim3 block(block_size);
  const dim3 grid((n + block_size - 1) / block_size);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gesummv_kernel<ALayout, BLayout>),
                     grid, block, 0, stream, a, b, x, y, n, alpha, beta);
}

template <int Id>
uint64_t host_offset(uint32_t i, uint32_t j, uint32_t n) {
  return Layout<Id>::offset(i, j, n);
}

struct Configuration {
  const char* name;
  const char* a_layout;
  const char* b_layout;
  LaunchFn launch;
  OffsetFn a_offset;
  OffsetFn b_offset;
};

static const Configuration kConfigurations[] = {
    {"row_major", "row_major", "row_major",
     &launch_gesummv<ROW_MAJOR, ROW_MAJOR>,
     &host_offset<ROW_MAJOR>, &host_offset<ROW_MAJOR>},
    {"column_major", "column_major", "column_major",
     &launch_gesummv<COLUMN_MAJOR, COLUMN_MAJOR>,
     &host_offset<COLUMN_MAJOR>, &host_offset<COLUMN_MAJOR>},
    {"tile16_row", "tile16_row", "tile16_row",
     &launch_gesummv<TILE16_ROW, TILE16_ROW>,
     &host_offset<TILE16_ROW>, &host_offset<TILE16_ROW>},
    {"tile16_column", "tile16_column", "tile16_column",
     &launch_gesummv<TILE16_COLUMN, TILE16_COLUMN>,
     &host_offset<TILE16_COLUMN>, &host_offset<TILE16_COLUMN>},
    {"c128x4/c128x4", "c128x4_iiiiiiijj_outerji",
     "c128x4_iiiiiiijj_outerji",
     &launch_gesummv<C128X4_IIIIIIIJJ_OUTERJI,
                      C128X4_IIIIIIIJJ_OUTERJI>,
     &host_offset<C128X4_IIIIIIIJJ_OUTERJI>,
     &host_offset<C128X4_IIIIIIIJJ_OUTERJI>},
    {"c128x4/c64x1", "c128x4_iiiiiiijj_outerji",
     "c64x1_iiiiii_outerji",
     &launch_gesummv<C128X4_IIIIIIIJJ_OUTERJI, C64X1_IIIIII_OUTERJI>,
     &host_offset<C128X4_IIIIIIIJJ_OUTERJI>,
     &host_offset<C64X1_IIIIII_OUTERJI>},
    {"c64x1/c128x4", "c64x1_iiiiii_outerji",
     "c128x4_iiiiiiijj_outerji",
     &launch_gesummv<C64X1_IIIIII_OUTERJI, C128X4_IIIIIIIJJ_OUTERJI>,
     &host_offset<C64X1_IIIIII_OUTERJI>,
     &host_offset<C128X4_IIIIIIIJJ_OUTERJI>},
    {"c64x1/c64x1", "c64x1_iiiiii_outerji",
     "c64x1_iiiiii_outerji",
     &launch_gesummv<C64X1_IIIIII_OUTERJI, C64X1_IIIIII_OUTERJI>,
     &host_offset<C64X1_IIIIII_OUTERJI>,
     &host_offset<C64X1_IIIIII_OUTERJI>},
    {"c64x16/c64x16", "c64x16_jjjjiiiiii_outerji",
     "c64x16_jjjjiiiiii_outerji",
     &launch_gesummv<C64X16_JJJJIIIIII_OUTERJI,
                      C64X16_JJJJIIIIII_OUTERJI>,
     &host_offset<C64X16_JJJJIIIIII_OUTERJI>,
     &host_offset<C64X16_JJJJIIIIII_OUTERJI>},
    {"c64x16/c128x16", "c64x16_jjjjiiiiii_outerji",
     "c128x16_jjjjiiiiiii_outerji",
     &launch_gesummv<C64X16_JJJJIIIIII_OUTERJI,
                      C128X16_JJJJIIIIIII_OUTERJI>,
     &host_offset<C64X16_JJJJIIIIII_OUTERJI>,
     &host_offset<C128X16_JJJJIIIIIII_OUTERJI>},
    {"c128x16/c64x16", "c128x16_jjjjiiiiiii_outerji",
     "c64x16_jjjjiiiiii_outerji",
     &launch_gesummv<C128X16_JJJJIIIIIII_OUTERJI,
                      C64X16_JJJJIIIIII_OUTERJI>,
     &host_offset<C128X16_JJJJIIIIIII_OUTERJI>,
     &host_offset<C64X16_JJJJIIIIII_OUTERJI>},
    {"c128x16/c128x16", "c128x16_jjjjiiiiiii_outerji",
     "c128x16_jjjjiiiiiii_outerji",
     &launch_gesummv<C128X16_JJJJIIIIIII_OUTERJI,
                      C128X16_JJJJIIIIIII_OUTERJI>,
     &host_offset<C128X16_JJJJIIIIIII_OUTERJI>,
     &host_offset<C128X16_JJJJIIIIIII_OUTERJI>},
    {"c64x16/lin16x4", "c64x16_jjjjiiiiii_outerji",
     "lin16x4_flag0_outerji",
     &launch_gesummv<C64X16_JJJJIIIIII_OUTERJI, LIN16X4_FLAG0_OUTERJI>,
     &host_offset<C64X16_JJJJIIIIII_OUTERJI>,
     &host_offset<LIN16X4_FLAG0_OUTERJI>},
    {"c128x16/lin16x4", "c128x16_jjjjiiiiiii_outerji",
     "lin16x4_flag0_outerji",
     &launch_gesummv<C128X16_JJJJIIIIIII_OUTERJI, LIN16X4_FLAG0_OUTERJI>,
     &host_offset<C128X16_JJJJIIIIIII_OUTERJI>,
     &host_offset<LIN16X4_FLAG0_OUTERJI>},
    {"lin16x4/c64x16", "lin16x4_flag0_outerji",
     "c64x16_jjjjiiiiii_outerji",
     &launch_gesummv<LIN16X4_FLAG0_OUTERJI, C64X16_JJJJIIIIII_OUTERJI>,
     &host_offset<LIN16X4_FLAG0_OUTERJI>,
     &host_offset<C64X16_JJJJIIIIII_OUTERJI>},
    {"lin16x4/c128x16", "lin16x4_flag0_outerji",
     "c128x16_jjjjiiiiiii_outerji",
     &launch_gesummv<LIN16X4_FLAG0_OUTERJI, C128X16_JJJJIIIIIII_OUTERJI>,
     &host_offset<LIN16X4_FLAG0_OUTERJI>,
     &host_offset<C128X16_JJJJIIIIIII_OUTERJI>},
    {"lin16x4/lin16x4", "lin16x4_flag0_outerji",
     "lin16x4_flag0_outerji",
     &launch_gesummv<LIN16X4_FLAG0_OUTERJI, LIN16X4_FLAG0_OUTERJI>,
     &host_offset<LIN16X4_FLAG0_OUTERJI>,
     &host_offset<LIN16X4_FLAG0_OUTERJI>},
};

constexpr size_t kConfigurationCount =
    sizeof(kConfigurations) / sizeof(kConfigurations[0]);

struct Options {
  uint32_t n = 4096;
  int samples = 12;
  int iterations = 20;
  int warmup = 5;
  int device = 0;
};

static void usage(const char* program) {
  std::printf(
      "Usage: %s [options]\n"
      "  --n N              matrix dimension (default 4096; multiple of 128)\n"
      "  --samples N        timing samples per configuration (default 12)\n"
      "  --iterations N     kernel launches per sample (default 20)\n"
      "  --warmup N         warmup launches per configuration (default 5)\n"
      "  --device N         HIP device ordinal (default 0)\n"
      "  --help             show this text\n",
      program);
}

static int parse_nonnegative(const char* value, const char* option) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!value[0] || *end != '\0' || parsed < 0 || parsed > 0x7fffffffL) {
    std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
}

static Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "Missing value for %s\n", argument.c_str());
      std::exit(EXIT_FAILURE);
    }
    const int value = parse_nonnegative(argv[++i], argument.c_str());
    if (argument == "--n") options.n = static_cast<uint32_t>(value);
    else if (argument == "--samples") options.samples = value;
    else if (argument == "--iterations") options.iterations = value;
    else if (argument == "--warmup") options.warmup = value;
    else if (argument == "--device") options.device = value;
    else {
      std::fprintf(stderr, "Unknown option: %s\n", argument.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

static void validate_layout(const char* name, OffsetFn offset, uint32_t n) {
  const uint64_t elements = static_cast<uint64_t>(n) * n;
  std::vector<unsigned char> seen(static_cast<size_t>(elements), 0);
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      const uint64_t physical = offset(i, j, n);
      if (physical >= elements || seen[static_cast<size_t>(physical)]) {
        std::fprintf(stderr,
                     "Layout %s is not bijective at (%u,%u): offset=%llu\n",
                     name, i, j, static_cast<unsigned long long>(physical));
        std::exit(EXIT_FAILURE);
      }
      seen[static_cast<size_t>(physical)] = 1;
    }
  }
}

static void pack_matrix(const std::vector<double>& logical,
                        std::vector<double>& physical,
                        OffsetFn offset,
                        uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      physical[static_cast<size_t>(offset(i, j, n))] =
          logical[static_cast<size_t>(i) * n + j];
    }
  }
}

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() & 1u) return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

struct DeviceConfiguration {
  double* a = nullptr;
  double* b = nullptr;
  std::vector<double> times_ms;
};

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (options.n < 128 || options.n % 128 != 0) {
    std::fprintf(stderr, "--n must be at least 128 and divisible by 128\n");
    return EXIT_FAILURE;
  }
  if (options.samples <= 0 || options.iterations <= 0 || options.warmup < 0) {
    std::fprintf(stderr, "samples and iterations must be positive; warmup cannot be negative\n");
    return EXIT_FAILURE;
  }

  HIP_CHECK(hipSetDevice(options.device));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, options.device));
  std::printf("Device: %s\n", properties.name);
  std::printf("GESUMMV FP64: N=%u, block=128, samples=%d, iterations=%d, warmup=%d\n",
              options.n, options.samples, options.iterations, options.warmup);

  const uint32_t n = options.n;
  const uint64_t elements = static_cast<uint64_t>(n) * n;
  const size_t matrix_bytes = static_cast<size_t>(elements * sizeof(double));
  const size_t vector_bytes = static_cast<size_t>(n) * sizeof(double);
  constexpr double alpha = 1.25;
  constexpr double beta = -0.75;

  std::vector<double> logical_a(static_cast<size_t>(elements));
  std::vector<double> logical_b(static_cast<size_t>(elements));
  std::vector<double> x(n);
  std::vector<double> reference(n);
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = static_cast<double>(static_cast<int>((i * 7u) % 37u) - 18) / 37.0;
    for (uint32_t j = 0; j < n; ++j) {
      logical_a[static_cast<size_t>(i) * n + j] =
          static_cast<double>(static_cast<int>((i * 17u + j * 13u) % 101u) - 50) /
          101.0;
      logical_b[static_cast<size_t>(i) * n + j] =
          static_cast<double>(static_cast<int>((i * 11u + j * 19u) % 103u) - 51) /
          103.0;
    }
  }
  for (uint32_t i = 0; i < n; ++i) {
    double sum_a = 0.0;
    double sum_b = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
      sum_a = std::fma(logical_a[static_cast<size_t>(i) * n + j], x[j], sum_a);
      sum_b = std::fma(logical_b[static_cast<size_t>(i) * n + j], x[j], sum_b);
    }
    reference[i] = alpha * sum_a + beta * sum_b;
  }

  // This catches mistakes in the handwritten layout formulas before any timing.
  validate_layout("row_major", &host_offset<ROW_MAJOR>, n);
  validate_layout("column_major", &host_offset<COLUMN_MAJOR>, n);
  validate_layout("tile16_row", &host_offset<TILE16_ROW>, n);
  validate_layout("tile16_column", &host_offset<TILE16_COLUMN>, n);
  validate_layout("c128x4_iiiiiiijj_outerji",
                  &host_offset<C128X4_IIIIIIIJJ_OUTERJI>, n);
  validate_layout("c64x1_iiiiii_outerji",
                  &host_offset<C64X1_IIIIII_OUTERJI>, n);
  validate_layout("c64x16_jjjjiiiiii_outerji",
                  &host_offset<C64X16_JJJJIIIIII_OUTERJI>, n);
  validate_layout("c128x16_jjjjiiiiiii_outerji",
                  &host_offset<C128X16_JJJJIIIIIII_OUTERJI>, n);
  validate_layout("lin16x4_flag0_outerji",
                  &host_offset<LIN16X4_FLAG0_OUTERJI>, n);

  double* device_x = nullptr;
  double* device_y = nullptr;
  HIP_CHECK(hipMalloc(&device_x, vector_bytes));
  HIP_CHECK(hipMalloc(&device_y, vector_bytes));
  HIP_CHECK(hipMemcpy(device_x, x.data(), vector_bytes, hipMemcpyHostToDevice));

  std::vector<DeviceConfiguration> device_configs(kConfigurationCount);
  std::vector<double> packed_a(static_cast<size_t>(elements));
  std::vector<double> packed_b(static_cast<size_t>(elements));
  std::vector<double> observed(n);

  std::puts("Packing and checking configurations:");
  for (size_t index = 0; index < kConfigurationCount; ++index) {
    const Configuration& config = kConfigurations[index];
    DeviceConfiguration& device = device_configs[index];
    pack_matrix(logical_a, packed_a, config.a_offset, n);
    pack_matrix(logical_b, packed_b, config.b_offset, n);
    HIP_CHECK(hipMalloc(&device.a, matrix_bytes));
    HIP_CHECK(hipMalloc(&device.b, matrix_bytes));
    HIP_CHECK(hipMemcpy(device.a, packed_a.data(), matrix_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device.b, packed_b.data(), matrix_bytes, hipMemcpyHostToDevice));

    config.launch(device.a, device.b, device_x, device_y, n, alpha, beta, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(observed.data(), device_y, vector_bytes, hipMemcpyDeviceToHost));
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
      const double absolute = std::abs(observed[i] - reference[i]);
      const double relative = absolute / std::max(1.0, std::abs(reference[i]));
      max_abs_error = std::max(max_abs_error, absolute);
      max_rel_error = std::max(max_rel_error, relative);
    }
    const bool correct = max_abs_error <= 5.0e-11 || max_rel_error <= 5.0e-11;
    std::printf("  %-20s A=%-31s B=%-31s %s (max abs %.3e)\n",
                config.name, config.a_layout, config.b_layout,
                correct ? "PASS" : "FAIL", max_abs_error);
    if (!correct) return EXIT_FAILURE;
  }

  for (size_t index = 0; index < kConfigurationCount; ++index) {
    const Configuration& config = kConfigurations[index];
    DeviceConfiguration& device = device_configs[index];
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      config.launch(device.a, device.b, device_x, device_y, n, alpha, beta, 0);
    }
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start{}, stop{};
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  std::vector<size_t> order(kConfigurationCount);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 random(0x4753554du);
  for (int sample = 0; sample < options.samples; ++sample) {
    std::shuffle(order.begin(), order.end(), random);
    for (size_t index : order) {
      const Configuration& config = kConfigurations[index];
      DeviceConfiguration& device = device_configs[index];
      HIP_CHECK(hipEventRecord(start, 0));
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        config.launch(device.a, device.b, device_x, device_y, n, alpha, beta, 0);
      }
      HIP_CHECK(hipEventRecord(stop, 0));
      HIP_CHECK(hipEventSynchronize(stop));
      HIP_CHECK(hipGetLastError());
      float elapsed_ms = 0.0f;
      HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
      device.times_ms.push_back(elapsed_ms / options.iterations);
    }
  }

  struct Result {
    size_t index;
    double median_ms;
    double mean_ms;
    double minimum_ms;
    double stddev_ms;
  };
  std::vector<Result> results;
  for (size_t index = 0; index < kConfigurationCount; ++index) {
    const std::vector<double>& times = device_configs[index].times_ms;
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double variance = 0.0;
    for (double time : times) variance += (time - mean) * (time - mean);
    variance /= times.size();
    results.push_back({index, median(times), mean,
                       *std::min_element(times.begin(), times.end()),
                       std::sqrt(variance)});
  }
  std::sort(results.begin(), results.end(), [](const Result& left, const Result& right) {
    return left.median_ms < right.median_ms;
  });

  const double row_major_ms = [&]() {
    for (const Result& result : results) {
      if (result.index == 0) return result.median_ms;
    }
    return 0.0;
  }();
  const double flops = 4.0 * static_cast<double>(n) * n + 3.0 * n;
  std::puts("\nResults (ranked by median kernel time; packing is excluded):");
  std::puts("rank  configuration          median_us  mean_us  min_us  sd_us  GFLOP/s  vs_row");
  for (size_t rank = 0; rank < results.size(); ++rank) {
    const Result& result = results[rank];
    const double gflops = flops / (result.median_ms * 1.0e6);
    std::printf("%4zu  %-21s %9.3f %8.3f %7.3f %6.3f %8.2f %7.3fx\n",
                rank + 1, kConfigurations[result.index].name,
                result.median_ms * 1000.0, result.mean_ms * 1000.0,
                result.minimum_ms * 1000.0, result.stddev_ms * 1000.0,
                gflops, row_major_ms / result.median_ms);
  }
  const Result& best = results.front();
  std::printf("\nBest: %s (A=%s, B=%s), %.3f us, %.3fx versus row-major.\n",
              kConfigurations[best.index].name,
              kConfigurations[best.index].a_layout,
              kConfigurations[best.index].b_layout,
              best.median_ms * 1000.0, row_major_ms / best.median_ms);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  for (DeviceConfiguration& device : device_configs) {
    HIP_CHECK(hipFree(device.a));
    HIP_CHECK(hipFree(device.b));
  }
  HIP_CHECK(hipFree(device_x));
  HIP_CHECK(hipFree(device_y));
  return EXIT_SUCCESS;
}
