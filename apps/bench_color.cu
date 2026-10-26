// Stages 1 and 2: color adjustment, CPU scalar reference vs CUDA, with the measurement harness.
// Usage: bench_color [-r runs] [-o outdir] <image>...
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include "bench.h"
#include "color.h"
#include "compare.h"
#include "image.h"

static bool run_one(const std::string& path, int runs, const std::string& outdir) {
  Image img;
  if (!load_image(path, img)) { fprintf(stderr, "cannot load %s\n", path.c_str()); return false; }
  const size_t n = img.npixels(), bytes = img.bytes();
  const double mp = n / 1e6;
  ColorParams p;
  Image cpu_out = img, gpu_out = img;

  Stats cpu = time_cpu([&] { color_cpu(img.px.data(), cpu_out.px.data(), n, p); }, runs);

  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  const int block = 256;
  GpuPhases g = measure_gpu_phases(
      [&] { CUDA_CHECK(cudaMemcpy(d_in, img.px.data(), bytes, cudaMemcpyHostToDevice)); },
      [&] { color_cuda_launch(d_in, d_out, n, p, block); },
      [&] { CUDA_CHECK(cudaMemcpy(gpu_out.px.data(), d_out, bytes, cudaMemcpyDeviceToHost)); }, runs);
  cudaFree(d_in);
  cudaFree(d_out);

  // Correctness gate (Stage 3): timings are only reported for output within tolerance.
  const int tol = 1;
  DiffStats d = compare_images(cpu_out.px.data(), gpu_out.px.data(), n, 3, tol);
  log_diff("results/data/correctness.csv", "color", "cuda", path, tol, d);
  if (!passes(d)) { fprintf(stderr, "%s: output differs from CPU reference: %s\n", path.c_str(), describe(d).c_str()); return false; }
  Image wrap_out = img;
  color_cuda(img.px.data(), wrap_out.px.data(), n, p);
  if (wrap_out.px != gpu_out.px) { fprintf(stderr, "color_cuda wrapper mismatch\n"); return false; }

  const double gbs = 2.0 * bytes / (g.kernel.median / 1e3) / 1e9;
  const std::string name = std::filesystem::path(path).filename().string();
  printf("%-26s %6.1f MP | cpu %8.2f | h2d %7.3f  kern %7.3f  d2h %7.3f  e2e %8.3f ms | kern %5.1f GB/s (%2.0f%%) | x%.0f kern, x%.1f e2e | diff max %d, %zu px\n",
         name.c_str(), mp, cpu.median, g.h2d.median, g.kernel.median, g.d2h.median, g.e2e.median, gbs,
         100 * gbs / kGpuPeakGBs, cpu.median / g.kernel.median, cpu.median / g.e2e.median, d.max_abs, d.mismatched);

  log_inout("results/data/color.csv", "color", "cpu_scalar", mp, run_tag("block=" + std::to_string(block) + ";img=" + name),
            cpu, g, bytes, bytes);

  if (!outdir.empty()) {
    std::filesystem::create_directories(outdir);
    std::string out = outdir + "/" + std::filesystem::path(path).stem().string() + "_color.png";
    if (!save_image(out, gpu_out)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return false; }
  }
  return true;
}

int main(int argc, char** argv) {
  int runs = 20;
  std::string outdir;
  int i = 1;
  for (; i < argc && argv[i][0] == '-'; i += 2) {
    std::string a = argv[i];
    if (i + 1 >= argc) break;
    if (a == "-r") runs = atoi(argv[i + 1]);
    else if (a == "-o") outdir = argv[i + 1];
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (i >= argc) { fprintf(stderr, "usage: %s [-r runs] [-o outdir] <image>...\n", argv[0]); return 2; }
  printf("runs=%d, caches flushed before each run, medians in ms\n", runs);
  for (; i < argc; i++)
    if (!run_one(argv[i], runs, outdir)) return 1;
}
