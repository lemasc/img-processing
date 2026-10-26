// Stage 4: naive Gaussian blur, CPU scalar vs CUDA, over kernel size.
// Usage: bench_blur [-r gpu_runs] [-c cpu_runs] [-k 3,5,7,11] [-b BXxBY] [-o outdir] <image>...
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

struct Opts {
  int gpu_runs = 20, cpu_runs = 3, bx = 16, by = 16;
  std::vector<int> ks = {3, 5, 7, 11};
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
    GpuPhases gp = measure_gpu_phases(
        [&] { CUDA_CHECK(cudaMemcpy(d_in, img.px.data(), bytes, cudaMemcpyHostToDevice)); },
        [&] { blur_naive_launch(d_in, d_out, img.w, img.h, g, o.bx, o.by); },
        [&] { CUDA_CHECK(cudaMemcpy(gpu_out.px.data(), d_out, bytes, cudaMemcpyDeviceToHost)); }, o.gpu_runs);
    CUDA_CHECK(cudaGetLastError());

    // Correctness gate (Stage 3 rule): timings are only reported for output within tolerance.
    const int tol = 1;
    DiffStats d = compare_images(cpu_out.px.data(), gpu_out.px.data(), n, 3, tol);
    log_diff("results/data/correctness.csv", "blur_gauss", "cuda_naive", name + " k=" + std::to_string(k), tol, d);
    if (!passes(d)) { fprintf(stderr, "%s k=%d: output differs from CPU reference: %s\n", name.c_str(), k, describe(d).c_str()); return false; }

    // Minimum traffic is one read and one write of the image. 7 flops per tap per pixel:
    // one weight product plus three multiply-adds.
    const double gbs = 2.0 * bytes / (gp.kernel.median / 1e3) / 1e9;
    const double gflops = 7.0 * k * k * n / (gp.kernel.median / 1e3) / 1e9;
    const double ai = 7.0 * k * k / 6.0;
    printf("%-22s %5.1f MP k=%-2d | cpu %9.1f | h2d %6.2f  kern %8.3f  d2h %6.2f  e2e %8.2f ms | kern %6.1f GB/s (%3.0f%%) %7.1f GFLOP/s, AI %5.1f F/B | x%.0f kern, x%.1f e2e\n",
           name.c_str(), mp, k, cpu.median, gp.h2d.median, gp.kernel.median, gp.d2h.median, gp.e2e.median, gbs,
           100 * gbs / kGpuPeakGBs, gflops, ai, cpu.median / gp.kernel.median, cpu.median / gp.e2e.median);

    const std::string tag = run_tag("k=" + std::to_string(k) + ";block=" + std::to_string(o.bx) + "x" +
                                    std::to_string(o.by) + ";img=" + name);
    log_inout("results/data/blur_naive.csv", "blur_gauss", "cpu_scalar", mp, tag, cpu, gp, bytes, bytes);

    if (!o.outdir.empty()) {
      std::filesystem::create_directories(o.outdir);
      std::string out = o.outdir + "/" + std::filesystem::path(path).stem().string() + "_blur_k" + std::to_string(k) + ".png";
      if (!save_image(out, gpu_out)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return false; }
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
    else if (a == "-k") {
      o.ks.clear();
      std::stringstream ss(v);
      for (std::string t; std::getline(ss, t, ',');) o.ks.push_back(atoi(t.c_str()));
    } else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (i >= argc) { fprintf(stderr, "usage: %s [-r gpu_runs] [-c cpu_runs] [-k list] [-b BXxBY] [-o outdir] <image>...\n", argv[0]); return 2; }
  printf("gpu runs=%d, cpu runs=%d, block=%dx%d, caches flushed before each run, medians\n", o.gpu_runs, o.cpu_runs, o.bx, o.by);
  for (; i < argc; i++)
    if (!run_one(argv[i], o)) return 1;
}
