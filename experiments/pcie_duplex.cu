// Does the PCIe link carry host-to-device and device-to-host copies at full speed at the same time?
// Times a pinned 256 MB copy in each direction alone, then both at once on two streams (the two
// copy engines of the RTX 3060). Explains how much copy/compute overlap Stage 11 can gain.
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(1); } } while (0)

static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main() {
  const size_t n = 256u << 20;
  void *h_a, *h_b, *d_a, *d_b;
  CK(cudaHostAlloc(&h_a, n, 0)); CK(cudaHostAlloc(&h_b, n, 0));
  CK(cudaMalloc(&d_a, n)); CK(cudaMalloc(&d_b, n));
  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time = [&](bool h2d, bool d2h) {
    std::vector<double> t;
    for (int i = 0; i < 12; i++) {
      CK(cudaEventRecord(e0, 0));
      if (h2d) CK(cudaMemcpyAsync(d_a, h_a, n, cudaMemcpyHostToDevice, s1));
      if (d2h) CK(cudaMemcpyAsync(h_b, d_b, n, cudaMemcpyDeviceToHost, s2));
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e1, 0));
      CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      if (i >= 2) t.push_back(ms);
    }
    return med(t);
  };
  time(true, true);  // warm-up: link idles at gen 1 and ramps under load
  const double a = time(true, false), b = time(false, true), both = time(true, true);
  printf("256 MB pinned copies, median of 10\n");
  printf("  H2D alone        %7.2f ms  %5.1f GB/s\n", a, n / (a / 1e3) / 1e9);
  printf("  D2H alone        %7.2f ms  %5.1f GB/s\n", b, n / (b / 1e3) / 1e9);
  printf("  both at once     %7.2f ms  %5.1f GB/s aggregate (%.1f H2D + %.1f D2H if shared evenly)\n", both, 2.0 * n / (both / 1e3) / 1e9,
         n / (both / 1e3) / 1e9, n / (both / 1e3) / 1e9);
  printf("  full duplex would take %.2f ms (the slower direction), serial would take %.2f ms\n", std::max(a, b), a + b);
  printf("  overlap achieved: %.0f%% of the serial-to-duplex gap\n", 100.0 * (a + b - both) / (a + b - std::max(a, b)));
}
