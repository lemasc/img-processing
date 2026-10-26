// Stage 1: color adjustment, CPU scalar reference vs CUDA kernel.
// Usage: stage1_color <input> <output> [runs]
#include <cstdio>
#include <cstdlib>
#include <string>
#include "color.h"
#include "image.h"
#include "timing.h"

// One thread per pixel. A warp reads 96 contiguous bytes, so loads and stores coalesce.
__global__ void color_kernel(const uint8_t* in, uint8_t* out, size_t npixels, ColorParams p) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < npixels) adjust_pixel(in + 3 * i, out + 3 * i, p);
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <input> <output> [runs]\n", argv[0]); return 2; }
  int runs = argc > 3 ? atoi(argv[3]) : 20;
  Image img;
  if (!load_image(argv[1], img)) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
  const size_t n = img.npixels(), bytes = img.bytes();
  const double mp = n / 1e6;
  printf("%s: %dx%d (%.1f MP)\n", argv[1], img.w, img.h, mp);

  ColorParams p;
  Image cpu_out = img, gpu_out = img;

  // CPU scalar reference.
  Stats cpu = time_cpu([&] { color_cpu(img.px.data(), cpu_out.px.data(), n, p); }, runs);

  // GPU: device buffers, then time each phase separately and end to end.
  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  const int block = 256;
  const int grid = (int)((n + block - 1) / block);
  auto h2d = [&] { CUDA_CHECK(cudaMemcpy(d_in, img.px.data(), bytes, cudaMemcpyHostToDevice)); };
  auto kern = [&] { color_kernel<<<grid, block>>>(d_in, d_out, n, p); };
  auto d2h = [&] { CUDA_CHECK(cudaMemcpy(gpu_out.px.data(), d_out, bytes, cudaMemcpyDeviceToHost)); };

  Stats s_h2d = time_gpu(h2d, runs);
  Stats s_kern = time_gpu(kern, runs);
  Stats s_d2h = time_gpu(d2h, runs);
  Stats s_e2e = time_cpu([&] { h2d(); kern(); d2h(); CUDA_CHECK(cudaDeviceSynchronize()); }, runs);

  // Quick sanity check only. Stage 3 builds the real correctness check.
  int maxdiff = 0;
  for (size_t i = 0; i < bytes; i++) maxdiff = std::max(maxdiff, abs((int)cpu_out.px[i] - (int)gpu_out.px[i]));

  double kern_gbs = 2.0 * bytes / (s_kern.median / 1e3) / 1e9;
  printf("CPU scalar      %9.3f ms  (%.1f MP/s)\n", cpu.median, mp / (cpu.median / 1e3));
  printf("GPU H2D         %9.3f ms\n", s_h2d.median);
  printf("GPU kernel      %9.3f ms  (%.1f GB/s, %.0f%% of 360 peak)\n", s_kern.median, kern_gbs, kern_gbs / 3.6);
  printf("GPU D2H         %9.3f ms\n", s_d2h.median);
  printf("GPU end-to-end  %9.3f ms\n", s_e2e.median);
  printf("speedup vs CPU: kernel %.0fx, end-to-end %.1fx\n", cpu.median / s_kern.median, cpu.median / s_e2e.median);
  printf("sanity: max |cpu-gpu| = %d\n", maxdiff);

  const char* cpus = getenv("BENCH_CPUS");
  std::string tag = std::string("cpus=") + (cpus ? cpus : "unpinned") + ";block=" + std::to_string(block);
  const std::string csv = "results/data/stage1.csv";
  csv_row(csv, "color", "cpu_scalar", mp, tag, cpu, 2.0 * bytes);
  csv_row(csv, "color", "cuda_h2d", mp, tag, s_h2d, bytes);
  csv_row(csv, "color", "cuda_kernel", mp, tag, s_kern, 2.0 * bytes);
  csv_row(csv, "color", "cuda_d2h", mp, tag, s_d2h, bytes);
  csv_row(csv, "color", "cuda_e2e", mp, tag, s_e2e, 2.0 * bytes);

  if (!save_image(argv[2], gpu_out)) { fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
  printf("wrote %s\n", argv[2]);
  cudaFree(d_in); cudaFree(d_out);
}
