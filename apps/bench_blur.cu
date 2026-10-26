// Stages 4 and 5: Gaussian blur, CPU scalar vs CUDA naive, tiled (byte tile) and tiledf (float tile), over kernel size.
// Usage: bench_blur [-r gpu_runs] [-c cpu_runs] [-k 3,5,7,11] [-b BXxBY] [-v naive,tiled,tiledf] [-o outdir] <image>...
// CPU runs are few by default: scalar blur at 24 MP and k=11 takes seconds per run.
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

static const char* kCsv = "results/data/blur_conv.csv";

struct Opts {
  int gpu_runs = 20, cpu_runs = 3, bx = 16, by = 16;
  std::vector<int> ks = {3, 5, 7, 11};
  std::vector<std::string> variants = {"naive", "tiled", "tiledf"};
  std::string outdir;
};

static bool run_one(const std::string& path, const Opts& o) {
  Image img;
  if (!load_image(path, img)) { fprintf(stderr, "cannot load %s\n", path.c_str()); return false; }
  const size_t n = img.npixels(), bytes = img.bytes();
  const double mp = n / 1e6;
  const std::string name = std::filesystem::path(path).filename().string();
  Image cpu_out = img, gpu_out = img;

  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  for (int k : o.ks) {
    const GaussKernel g = make_gaussian(k);
    Stats cpu = time_cpu([&] { blur_cpu(img.px.data(), cpu_out.px.data(), img.w, img.h, g); }, o.cpu_runs);
    const std::string tag0 = "k=" + std::to_string(k) + ";block=" + std::to_string(o.bx) + "x" + std::to_string(o.by) +
                             ";img=" + name;
    csv_row(kCsv, "blur_gauss", "cpu_scalar", mp, run_tag(tag0 + ";v=cpu"), cpu, 2.0 * bytes, kCpuPeakGBs);
    double naive_kernel_ms = 0;

    for (const std::string& v : o.variants) {
      const bool ftile = v == "tiledf";
      const bool tiled = v == "tiled" || ftile;
      if (tiled && blur_tile_bytes(g, o.bx, o.by, ftile ? 4 : 1) > 48 * 1024) { printf("%-22s k=%-2d %s: tile exceeds 48 KB, skipped\n", name.c_str(), k, v.c_str()); continue; }
      auto launch = [&] {
        if (tiled) blur_tiled_launch(d_in, d_out, img.w, img.h, g, o.bx, o.by, ftile);
        else blur_naive_launch(d_in, d_out, img.w, img.h, g, o.bx, o.by);
      };
      GpuPhases gp = measure_gpu_phases(
          [&] { CUDA_CHECK(cudaMemcpy(d_in, img.px.data(), bytes, cudaMemcpyHostToDevice)); }, launch,
          [&] { CUDA_CHECK(cudaMemcpy(gpu_out.px.data(), d_out, bytes, cudaMemcpyDeviceToHost)); }, o.gpu_runs);
      CUDA_CHECK(cudaGetLastError());

      // Correctness gate (Stage 3 rule): timings are only reported for output within tolerance.
      const int tol = 1;
      DiffStats d = compare_images(cpu_out.px.data(), gpu_out.px.data(), n, 3, tol);
      log_diff("results/data/correctness.csv", "blur_gauss", "cuda_" + v, name + " k=" + std::to_string(k), tol, d);
      if (!passes(d)) { fprintf(stderr, "%s k=%d %s: output differs from CPU reference: %s\n", name.c_str(), k, v.c_str(), describe(d).c_str()); return false; }

      // Minimum traffic is one read and one write of the image. 7 flops per tap per pixel:
      // one weight product plus three multiply-adds.
      const double gbs = 2.0 * bytes / (gp.kernel.median / 1e3) / 1e9;
      const double gflops = 7.0 * k * k * n / (gp.kernel.median / 1e3) / 1e9;
      const double occ = tiled ? blur_tiled_occupancy(g, o.bx, o.by, ftile) : blur_naive_occupancy(o.bx, o.by);
      if (!tiled) naive_kernel_ms = gp.kernel.median;
      printf("%-18s %5.1f MP k=%-2d %-6s | cpu %8.1f | kern %8.3f  e2e %7.2f ms | %6.1f GB/s (%3.0f%%) %7.1f GFLOP/s | occ %3.0f%% | x%.0f vs cpu%s",
             name.c_str(), mp, k, v.c_str(), cpu.median, gp.kernel.median, gp.e2e.median, gbs, 100 * gbs / kGpuPeakGBs,
             gflops, 100 * occ, cpu.median / gp.kernel.median, "");
      if (tiled && naive_kernel_ms > 0) printf(" | x%.2f vs naive", naive_kernel_ms / gp.kernel.median);
      printf("\n");

      const std::string tag = run_tag(tag0 + ";v=" + v + ";occ=" + std::to_string((int)(100 * occ + 0.5)));
      const double both = 2.0 * bytes;
      csv_row(kCsv, "blur_gauss", "cuda_h2d", mp, tag, gp.h2d, (double)bytes, kPcieH2DGBs);
      csv_row(kCsv, "blur_gauss", "cuda_kernel", mp, tag, gp.kernel, both, kGpuPeakGBs);
      csv_row(kCsv, "blur_gauss", "cuda_d2h", mp, tag, gp.d2h, (double)bytes, kPcieD2HGBs);
      csv_row(kCsv, "blur_gauss", "cuda_e2e", mp, tag, gp.e2e, both);

      if (!o.outdir.empty()) {
        std::filesystem::create_directories(o.outdir);
        std::string out = o.outdir + "/" + std::filesystem::path(path).stem().string() + "_blur_k" + std::to_string(k) + "_" + v + ".png";
        if (!save_image(out, gpu_out)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return false; }
      }
    }
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return true;
}

int main(int argc, char** argv) {
  Opts o;
  int i = 1;
  for (; i < argc && argv[i][0] == '-'; i += 2) {
    std::string a = argv[i];
    if (i + 1 >= argc) break;
    std::string v = argv[i + 1];
    if (a == "-r") o.gpu_runs = atoi(v.c_str());
    else if (a == "-c") o.cpu_runs = atoi(v.c_str());
    else if (a == "-o") o.outdir = v;
    else if (a == "-b") { if (sscanf(v.c_str(), "%dx%d", &o.bx, &o.by) != 2) return 2; }
    else if (a == "-v") {
      o.variants.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) o.variants.push_back(t);
    } else if (a == "-k") {
      o.ks.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) o.ks.push_back(atoi(t.c_str()));
    } else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (i >= argc) { fprintf(stderr, "usage: %s [-r gpu_runs] [-c cpu_runs] [-k list] [-b BXxBY] [-v list] [-o outdir] <image>...\n", argv[0]); return 2; }
  printf("gpu runs=%d, cpu runs=%d, block=%dx%d, caches flushed before each run, medians\n", o.gpu_runs, o.cpu_runs, o.bx, o.by);
  for (; i < argc; i++)
    if (!run_one(argv[i], o)) return 1;
}
