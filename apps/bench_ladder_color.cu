// Stage 12: parallelism ladder on the color kernel. Same algorithm, image and parameters on every
// rung; each CPU rung is checked against the scalar rung (tolerance 1 level) before it is timed
// into results/data/ladder.csv. Thread count of the OpenMP rungs comes from OMP_NUM_THREADS
// (scripts/bench.sh sets it together with the CPU pinning).
// Usage: bench_ladder_color [-r runs] <image>...
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include "bench.h"
#include "color.h"
#include "compare.h"
#include "image.h"

using ColorFn = void (*)(const uint8_t*, uint8_t*, size_t, const ColorParams&);

struct Rung {
  const char* platform;  // CSV platform column
  const char* label;
  ColorFn fn;
  bool threaded;
};

static const Rung kRungs[] = {
    {"r1_scalar", "scalar", color_cpu, false},
    {"r2_autovec_avx2", "autovec AVX2", color_cpu_autovec, false},
    {"r2b_autovec_avx2_omp", "autovec AVX2 + OpenMP", color_cpu_autovec_omp, true},
    {"r6a_avx2_intrin", "AVX2 intrinsics", color_cpu_avx2, false},
    {"r6_avx2_intrin_omp", "AVX2 intrinsics + OpenMP", color_cpu_avx2_omp, true},
};

static bool run_one(const std::string& path, int runs) {
  Image img;
  if (!load_image(path, img)) { fprintf(stderr, "cannot load %s\n", path.c_str()); return false; }
  const size_t n = img.npixels(), bytes = img.bytes();
  const double mp = n / 1e6;
  const std::string name = std::filesystem::path(path).filename().string();
  const int threads = omp_get_max_threads();
  ColorParams p;
  Image ref = img, out = img;
  const std::string csv = "results/data/ladder.csv";

  printf("%s (%.1f MP), %d OpenMP threads, cpus=%s\n", name.c_str(), mp, threads,
         getenv("BENCH_CPUS") ? getenv("BENCH_CPUS") : "unpinned");
  double scalar_ms = 0;
  for (const Rung& r : kRungs) {
    Image& dst = (r.fn == color_cpu) ? ref : out;
    Stats s = time_cpu([&] { r.fn(img.px.data(), dst.px.data(), n, p); }, runs);
    DiffStats d{};
    if (r.fn != color_cpu) {
      d = compare_images(ref.px.data(), out.px.data(), n, 3, 1);
      log_diff("results/data/correctness.csv", "color", r.platform, path, 1, d);
      if (!passes(d)) { fprintf(stderr, "%s: %s differs from scalar: %s\n", name.c_str(), r.label, describe(d).c_str()); return false; }
    } else scalar_ms = s.median;
    const double gbs = 2.0 * bytes / (s.median / 1e3) / 1e9;
    printf("  %-26s %9.3f ms  %7.1f MP/s  %6.1f GB/s  x%-7.1f maxdiff %d\n", r.label, s.median, mp / (s.median / 1e3), gbs,
           scalar_ms / s.median, d.max_abs);
    csv_row(csv, "color", r.platform, mp, run_tag("img=" + name + ";threads=" + std::to_string(r.threaded ? threads : 1)), s,
            2.0 * bytes, kCpuPeakGBs);
  }

  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  GpuPhases g = measure_gpu_phases(
      [&] { CUDA_CHECK(cudaMemcpy(d_in, img.px.data(), bytes, cudaMemcpyHostToDevice)); },
      [&] { color_cuda_launch(d_in, d_out, n, p, 256); },
      [&] { CUDA_CHECK(cudaMemcpy(out.px.data(), d_out, bytes, cudaMemcpyDeviceToHost)); }, runs);
  cudaFree(d_in);
  cudaFree(d_out);
  DiffStats d = compare_images(ref.px.data(), out.px.data(), n, 3, 1);
  log_diff("results/data/correctness.csv", "color", "r7_cuda", path, 1, d);
  if (!passes(d)) { fprintf(stderr, "%s: CUDA differs from scalar: %s\n", name.c_str(), describe(d).c_str()); return false; }
  const std::string tag = run_tag("img=" + name + ";threads=1");
  const double both = 2.0 * bytes;
  csv_row(csv, "color", "r7_cuda_kernel", mp, tag, g.kernel, both, kGpuPeakGBs);
  csv_row(csv, "color", "r7_cuda_e2e", mp, tag, g.e2e, both);
  csv_row(csv, "color", "r7_cuda_h2d", mp, tag, g.h2d, (double)bytes, kPcieH2DGBs);
  csv_row(csv, "color", "r7_cuda_d2h", mp, tag, g.d2h, (double)bytes, kPcieD2HGBs);
  printf("  %-26s %9.3f ms  (compute only)           x%-7.1f maxdiff %d\n", "CUDA kernel", g.kernel.median, scalar_ms / g.kernel.median, d.max_abs);
  printf("  %-26s %9.3f ms  (h2d %.3f + kernel %.3f + d2h %.3f)  x%.1f\n", "CUDA end to end", g.e2e.median, g.h2d.median,
         g.kernel.median, g.d2h.median, scalar_ms / g.e2e.median);
  return true;
}

int main(int argc, char** argv) {
  int runs = 20, i = 1;
  if (argc > 2 && std::string(argv[1]) == "-r") { runs = atoi(argv[2]); i = 3; }
  if (i >= argc) { fprintf(stderr, "usage: %s [-r runs] <image>...\n", argv[0]); return 2; }
  printf("runs=%d, caches flushed before each run, medians\n", runs);
  for (; i < argc; i++)
    if (!run_one(argv[i], runs)) return 1;
}
