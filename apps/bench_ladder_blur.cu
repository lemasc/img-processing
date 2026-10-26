// Stage 12: parallelism ladder on the separable Gaussian blur (the compute-bound kernel). Same
// algorithm, image and weights on every rung; each rung is checked against the scalar rung
// (tolerance 1 level) before it is timed into results/data/ladder.csv. The OpenMP rungs use
// OMP_NUM_THREADS (scripts/bench.sh sets it together with the CPU pinning).
// Usage: bench_ladder_blur [-r gpu_runs] [-c cpu_runs] [-k 7,15,...] [-v rungs] <image>...
//   rungs: scalar,autovec,autovec_omp,avx2,avx2_omp,cuda (default all). Skipped single-thread rungs
//   are not timed again in the multi-thread modes; the scalar reference is still computed once.
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include "bench.h"
#include "blur.h"
#include "compare.h"
#include "image.h"

using SepFn = void (*)(const uint8_t*, uint8_t*, float*, int, int, const GaussKernel&);

struct Rung {
  const char* key;       // -v name
  const char* platform;  // CSV platform column
  const char* label;
  SepFn fn;
  bool threaded;
};

static const Rung kRungs[] = {
    {"scalar", "r1_scalar", "scalar", blur_sep_cpu, false},
    {"autovec", "r2_autovec_avx2", "autovec AVX2", blur_sep_autovec, false},
    {"autovec_omp", "r2b_autovec_avx2_omp", "autovec AVX2 + OpenMP", blur_sep_autovec_omp, true},
    {"avx2", "r6a_avx2_intrin", "AVX2 intrinsics", blur_sep_avx2, false},
    {"avx2_omp", "r6_avx2_intrin_omp", "AVX2 intrinsics + OpenMP", blur_sep_avx2_omp, true},
};

static bool want(const std::vector<std::string>& v, const char* key) {
  for (auto& s : v) if (s == key) return true;
  return false;
}

static bool run_one(const std::string& path, const std::vector<int>& ks, const std::vector<std::string>& rungs, int gpu_runs,
                    int cpu_runs) {
  Image img;
  if (!load_image(path, img)) { fprintf(stderr, "cannot load %s\n", path.c_str()); return false; }
  const size_t n = img.npixels(), bytes = img.bytes();
  const double mp = n / 1e6;
  const std::string name = std::filesystem::path(path).filename().string();
  const int threads = omp_get_max_threads();
  const std::string csv = "results/data/ladder.csv";
  Image ref = img, out = img;
  std::vector<float> tmp(bytes);
  uint8_t *d_in, *d_out;
  float* d_tmp;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMalloc(&d_tmp, bytes * sizeof(float)));

  for (int k : ks) {
    const GaussKernel g = make_gaussian(k);
    const std::string tag0 = "img=" + name + ";k=" + std::to_string(k);
    printf("%s (%.1f MP), k=%d (radius %d), %d OpenMP threads, cpus=%s\n", name.c_str(), mp, k, k / 2, threads,
           getenv("BENCH_CPUS") ? getenv("BENCH_CPUS") : "unpinned");
    const double gflop = 12.0 * k * n / 1e9;  // 6 flops per tap, 2k taps, same count as Stage 6
    double scalar_ms = 0;
    if (!want(rungs, "scalar")) blur_sep_cpu(img.px.data(), ref.px.data(), tmp.data(), img.w, img.h, g);  // untimed reference
    for (const Rung& r : kRungs) {
      if (!want(rungs, r.key)) continue;
      const bool is_ref = r.fn == blur_sep_cpu;
      Image& dst = is_ref ? ref : out;
      Stats s = time_cpu([&] { r.fn(img.px.data(), dst.px.data(), tmp.data(), img.w, img.h, g); }, cpu_runs);
      DiffStats d{};
      if (!is_ref) {
        d = compare_images(ref.px.data(), out.px.data(), n, 3, 1);
        log_diff("results/data/correctness.csv", "blur_sep", r.platform, name + " k=" + std::to_string(k), 1, d);
        if (!passes(d)) { fprintf(stderr, "%s k=%d: %s differs from scalar: %s\n", name.c_str(), k, r.label, describe(d).c_str()); return false; }
      } else scalar_ms = s.median;
      printf("  %-26s %10.2f ms  %7.1f MP/s  %7.1f GFLOP/s", r.label, s.median, mp / (s.median / 1e3), gflop / (s.median / 1e3));
      if (scalar_ms > 0) printf("  x%-7.1f", scalar_ms / s.median);
      printf(" maxdiff %d\n", d.max_abs);
      csv_row(csv, "blur_sep", r.platform, mp, run_tag(tag0 + ";threads=" + std::to_string(r.threaded ? threads : 1)), s, 10.0 * bytes,
              kCpuPeakGBs);
    }
    if (!want(rungs, "cuda")) continue;
    GpuPhases gp = measure_gpu_phases(
        [&] { CUDA_CHECK(cudaMemcpy(d_in, img.px.data(), bytes, cudaMemcpyHostToDevice)); },
        [&] { blur_sep_launch(d_in, d_tmp, d_out, img.w, img.h, g, 16, 16); },
        [&] { CUDA_CHECK(cudaMemcpy(out.px.data(), d_out, bytes, cudaMemcpyDeviceToHost)); }, gpu_runs);
    CUDA_CHECK(cudaGetLastError());
    DiffStats d = compare_images(ref.px.data(), out.px.data(), n, 3, 1);
    log_diff("results/data/correctness.csv", "blur_sep", "r7_cuda", name + " k=" + std::to_string(k), 1, d);
    if (!passes(d)) { fprintf(stderr, "%s k=%d: CUDA differs from scalar: %s\n", name.c_str(), k, describe(d).c_str()); return false; }
    const std::string tag = run_tag(tag0 + ";threads=1");
    csv_row(csv, "blur_sep", "r7_cuda_kernel", mp, tag, gp.kernel, 10.0 * bytes, kGpuPeakGBs);
    csv_row(csv, "blur_sep", "r7_cuda_e2e", mp, tag, gp.e2e, 2.0 * bytes);
    csv_row(csv, "blur_sep", "r7_cuda_h2d", mp, tag, gp.h2d, (double)bytes, kPcieH2DGBs);
    csv_row(csv, "blur_sep", "r7_cuda_d2h", mp, tag, gp.d2h, (double)bytes, kPcieD2HGBs);
    printf("  %-26s %10.2f ms  %7.1f MP/s  %7.1f GFLOP/s  (compute only)", "CUDA kernel", gp.kernel.median,
           mp / (gp.kernel.median / 1e3), gflop / (gp.kernel.median / 1e3));
    if (scalar_ms > 0) printf("  x%-7.1f", scalar_ms / gp.kernel.median);
    printf(" maxdiff %d\n  %-26s %10.2f ms  (h2d %.2f + kernel %.2f + d2h %.2f)", d.max_abs, "CUDA end to end", gp.e2e.median,
           gp.h2d.median, gp.kernel.median, gp.d2h.median);
    if (scalar_ms > 0) printf("  x%.1f", scalar_ms / gp.e2e.median);
    printf("\n");
  }
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_tmp);
  return true;
}

int main(int argc, char** argv) {
  int gpu_runs = 20, cpu_runs = 5, i = 1;
  std::vector<int> ks = {15};
  std::vector<std::string> rungs = {"scalar", "autovec", "autovec_omp", "avx2", "avx2_omp", "cuda"};
  for (; i + 1 < argc && argv[i][0] == '-'; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "-r") gpu_runs = atoi(v.c_str());
    else if (a == "-c") cpu_runs = atoi(v.c_str());
    else if (a == "-k" || a == "-v") {
      if (a == "-k") ks.clear(); else rungs.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) { if (a == "-k") ks.push_back(atoi(t.c_str())); else rungs.push_back(t); }
    } else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (i >= argc) { fprintf(stderr, "usage: %s [-r gpu_runs] [-c cpu_runs] [-k list] [-v rungs] <image>...\n", argv[0]); return 2; }
  printf("gpu runs=%d, cpu runs=%d, caches flushed before each run, medians\n", gpu_runs, cpu_runs);
  for (; i < argc; i++)
    if (!run_one(argv[i], ks, rungs, gpu_runs, cpu_runs)) return 1;
}
