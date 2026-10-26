// Smoke test: vector copy kernel, measures effective bandwidth vs the 360 GB/s peak.
#include <cstdio>
#include <vector>
#include "timing.h"

__global__ void copy_k(const float* in, float* out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

int main() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s, %d SMs, cc %d.%d\n", p.name, p.multiProcessorCount, p.major, p.minor);

  const size_t n = 64u << 20;  // 256 MB of floats
  float *a, *b;
  CUDA_CHECK(cudaMalloc(&a, n * 4));
  CUDA_CHECK(cudaMalloc(&b, n * 4));
  CUDA_CHECK(cudaMemset(a, 0, n * 4));

  Stats s = time_gpu([&] { copy_k<<<(n + 255) / 256, 256>>>(a, b, n); });
  double gbps = 2.0 * n * 4 / (s.median / 1000.0) / 1e9;
  printf("copy kernel: median %.3f ms, %.1f GB/s (%.0f%% of 360)\n", s.median, gbps, gbps / 3.6);

  std::vector<float> h(n);
  Stats t = time_cpu([&] { CUDA_CHECK(cudaMemcpy(h.data(), a, n * 4, cudaMemcpyDeviceToHost)); }, 5);
  printf("D2H pageable: median %.3f ms, %.1f GB/s\n", t.median, n * 4 / (t.median / 1000.0) / 1e9);

  // Practical PCIe peak: pinned host memory, both directions.
  float* hp;
  CUDA_CHECK(cudaMallocHost(&hp, n * 4));
  Stats ph = time_gpu([&] { CUDA_CHECK(cudaMemcpy(a, hp, n * 4, cudaMemcpyHostToDevice)); }, 10);
  Stats pd = time_gpu([&] { CUDA_CHECK(cudaMemcpy(hp, a, n * 4, cudaMemcpyDeviceToHost)); }, 10);
  printf("H2D pinned:   median %.3f ms, %.1f GB/s\n", ph.median, n * 4 / (ph.median / 1000.0) / 1e9);
  printf("D2H pinned:   median %.3f ms, %.1f GB/s\n", pd.median, n * 4 / (pd.median / 1000.0) / 1e9);
  cudaFreeHost(hp);
  cudaFree(a); cudaFree(b);
}
