// Isolate the asymmetric GEMM's A copy geometry, without matrix instructions.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)

constexpr int M = 512, K = 4096, Blocks = 256, Threads = 128, Trips = 128;

__host__ __device__ int offset(int layout, int i, int k) {
  if (layout == 1) return 4096 * (k / 8) + 8 * i + k % 8;
  if (layout == 2) return 8192 * (i / 2) + 64 * (k / 32) + 32 * (i % 2) + k % 32;
  return 4096 * i + k;
}

template<bool Async>
__device__ __forceinline__ void copy16(unsigned dst, const uint16_t *src) {
  if constexpr (Async) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
  } else {
    unsigned a, b, c, d;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(src) : "memory");
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "r"(dst), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
  }
}

template<int Layout, bool Async>
__global__ void copy_probe(const uint16_t *input, uint32_t *output) {
  __shared__ __align__(128) unsigned char shared[4096];
  int tid = threadIdx.x;
  int row = (blockIdx.x % 8) * 64 + tid / 4;
  int col = 8 * (tid % 4);
  const uint16_t *a = input + offset(Layout, row, col);
  const uint16_t *b = input + offset(Layout, row + 32, col);
  constexpr int step = Layout == 1 ? 16384 : Layout == 2 ? 64 : 32;
  // Saved native PTX: (16*tid & 2032) XOR (2*(tid & 24)).
  unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(shared)) + ((16 * tid & 2032) ^ (2 * (tid & 24)));
  uint32_t sum = 0;
  #pragma unroll 1
  for (int tile = 0; tile < Trips; ++tile) {
    copy16<Async>(dst, a);
    copy16<Async>(dst + 2048, b);
    if constexpr (Async) {
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
    #pragma unroll
    for (int packet = 0; packet < 2; ++packet) {
      unsigned x, y, z, w;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(dst + packet * 2048) : "memory");
      sum += x + y + z + w;
    }
    a += step;
    b += step;
  }
  output[blockIdx.x * Threads + tid] = sum;
}

void launch(int layout, bool async, const uint16_t *input, uint32_t *output, cudaStream_t stream) {
  #define RUN(L) if (async) copy_probe<L, true><<<Blocks, Threads, 0, stream>>>(input, output); \
                 else copy_probe<L, false><<<Blocks, Threads, 0, stream>>>(input, output)
  if (layout == 0) { RUN(0); }
  else if (layout == 1) { RUN(1); }
  else { RUN(2); }
  CUDA(cudaGetLastError());
}

uint16_t value(int i, int k) { return static_cast<uint16_t>((i * K + k) * 17 + 3); }

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: probe ordinary|large|smaller async|sync timing|profile output.json\n");
    return 2;
  }
  std::string name(argv[1]), mode(argv[2]), phase(argv[3]);
  int layout = name == "ordinary" ? 0 : name == "large" ? 1 : name == "smaller" ? 2 : -1;
  if (layout < 0 || (mode != "async" && mode != "sync") || (phase != "timing" && phase != "profile")) return 2;
  cudaDeviceProp device;
  CUDA(cudaGetDeviceProperties(&device, 0));
  if (device.major != 9 || device.minor != 0) {
    std::fprintf(stderr, "This diagnostic requires an H100 (compute capability 9.0).\n");
    return 2;
  }
  std::vector<uint16_t> packed(M * K);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) packed[offset(layout, i, k)] = value(i, k);
  uint16_t *input;
  uint32_t *output;
  CUDA(cudaMalloc(&input, packed.size() * sizeof(uint16_t)));
  CUDA(cudaMalloc(&output, Blocks * Threads * sizeof(uint32_t)));
  CUDA(cudaMemcpy(input, packed.data(), packed.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CUDA(cudaStreamCreate(&stream));
  launch(layout, mode == "async", input, output, stream);
  CUDA(cudaStreamSynchronize(stream));
  std::vector<uint32_t> observed(Blocks * Threads);
  CUDA(cudaMemcpy(observed.data(), output, observed.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (int block = 0; block < Blocks; ++block) {
    for (int tid = 0; tid < Threads; ++tid) {
      uint32_t expected = 0;
      for (int tile = 0; tile < Trips; ++tile)
        for (int packet = 0; packet < 2; ++packet)
          for (int word = 0; word < 4; ++word) {
            int i = (block % 8) * 64 + tid / 4 + packet * 32;
            int k = tile * 32 + 8 * (tid % 4) + word * 2;
            expected += uint32_t(value(i, k)) | (uint32_t(value(i, k + 1)) << 16);
          }
      if (observed[block * Threads + tid] != expected) {
        std::fprintf(stderr, "copy validation failed at block %d thread %d\n", block, tid);
        return 1;
      }
    }
  }
  std::vector<float> times;
  for (int i = 0; i < 500; ++i) launch(layout, mode == "async", input, output, stream);
  CUDA(cudaStreamSynchronize(stream));
  if (phase == "profile") {
    CUDA(cudaProfilerStart());
    launch(layout, mode == "async", input, output, stream);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaProfilerStop());
  } else {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 50; ++i) launch(layout, mode == "async", input, output, stream);
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 10; ++i) CUDA(cudaGraphLaunch(executable, stream));
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&end));
    for (int i = 0; i < 21; ++i) {
      CUDA(cudaEventRecord(start, stream));
      CUDA(cudaGraphLaunch(executable, stream));
      CUDA(cudaEventRecord(end, stream));
      CUDA(cudaEventSynchronize(end));
      float ms;
      CUDA(cudaEventElapsedTime(&ms, start, end));
      times.push_back(ms / 50);
    }
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(end));
    CUDA(cudaGraphExecDestroy(executable));
    CUDA(cudaGraphDestroy(graph));
  }
  std::ofstream result(argv[4]);
  result << "{\"layout\":\"" << name << "\",\"mode\":\"" << mode << "\",\"phase\":\"" << phase
         << "\",\"correct\":true,\"device\":\"" << device.name << "\",\"input_address\":"
         << reinterpret_cast<uintptr_t>(input) << ",\"samples_ms\":[";
  for (size_t i = 0; i < times.size(); ++i) result << (i ? "," : "") << times[i];
  result << "]}\n";
  if (!result) return 1;
  CUDA(cudaStreamDestroy(stream));
  CUDA(cudaFree(input));
  CUDA(cudaFree(output));
}
