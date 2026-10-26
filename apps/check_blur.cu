// Stage 4 to 6 correctness check of the blur implementations (naive, tiled and separable).
// Usage: check_blur [image...]
// Always runs: CPU vs a double-precision reference, flat-image invariance, then CUDA vs CPU on random
// images with sizes smaller than, equal to and not dividing the block, for every kernel size and
// several block shapes. Tolerance is 1 level per channel (float rounding, as in Stage 3).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>
#include "blur.h"
#include "compare.h"
#include "image.h"

static bool failed = false;
static const std::string kCsv = "results/data/correctness.csv";
constexpr int kTol = 1;

static void report(const std::string& label, const std::string& impl, const DiffStats& d, int tol) {
  printf("%-44s %s  %s\n", label.c_str(), passes(d) ? "PASS" : "FAIL", describe(d).c_str());
  if (!passes(d)) failed = true;
  log_diff(kCsv, "blur_gauss", impl, label, tol, d);
}

static std::vector<uint8_t> random_image(int w, int h, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> v((size_t)w * h * 3);
  for (auto& b : v) b = rng() & 255;
  return v;
}

// Both CUDA kernels against the CPU reference (tolerance 1), and the tiled kernel against the naive
// one (tolerance 0: same taps, same order, so they should agree bit for bit).
static void check_cuda(const std::string& label, const uint8_t* in, int w, int h, int k, int bx, int by) {
  GaussKernel g = make_gaussian(k);
  std::vector<uint8_t> ref((size_t)w * h * 3), naive(ref.size()), sep(ref.size()), tiled(ref.size()), tiledf(ref.size());
  blur_cpu(in, ref.data(), w, h, g);
  blur_naive_cuda(in, naive.data(), w, h, g, bx, by);
  const std::string l = label + " k=" + std::to_string(k) + " blk=" + std::to_string(bx) + "x" + std::to_string(by);
  report(l, "cuda_naive", compare_images(ref.data(), naive.data(), (size_t)w * h, 3, kTol), kTol);
  // Separable against the 2D CPU reference: same blur up to float rounding, so the same tolerance.
  blur_sep_cuda(in, sep.data(), w, h, g, bx, by);
  report(l + " separable", "cuda_sep", compare_images(ref.data(), sep.data(), (size_t)w * h, 3, kTol), kTol);
  if (blur_tile_bytes(g, bx, by) > 48 * 1024) { printf("%-44s skipped (tile exceeds 48 KB)\n", l.c_str()); return; }
  blur_tiled_cuda(in, tiled.data(), w, h, g, bx, by);
  report(l, "cuda_tiled", compare_images(ref.data(), tiled.data(), (size_t)w * h, 3, kTol), kTol);
  report(l + " vs naive", "cuda_tiled_vs_naive", compare_images(naive.data(), tiled.data(), (size_t)w * h, 3, 0), 0);
  if (blur_tile_bytes(g, bx, by, 4) > 48 * 1024) { printf("%-44s float tile skipped (exceeds 48 KB)\n", l.c_str()); return; }
  blur_tiled_cuda(in, tiledf.data(), w, h, g, bx, by, true);
  report(l + " float tile", "cuda_tiledf", compare_images(ref.data(), tiledf.data(), (size_t)w * h, 3, kTol), kTol);
  report(l + " float tile vs naive", "cuda_tiledf_vs_naive", compare_images(naive.data(), tiledf.data(), (size_t)w * h, 3, 0), 0);
}

int main(int argc, char** argv) {
  const int ks[] = {3, 5, 7, 11};
  const int kall[] = {3, 5, 7, 11, 15, 31, 63, 127};  // Stage 6 radius sweep 3, 7, 15, 31, 63 is k = 2r + 1
  printf("tolerance: %d level(s) per channel (of 255)\n", kTol);

  for (int k : kall) {  // weights sum to 1
    GaussKernel g = make_gaussian(k);
    double s = 0;
    for (int i = 0; i < k; i++) s += g.w[i];
    if (std::abs(s - 1.0) > 1e-6) { printf("weights k=%d sum %.9f FAIL\n", k, s); failed = true; }
  }

  // The CPU reference against an independent double-precision loop (guards against a shared bug).
  for (int k : kall) {
    const int w = 37, h = 23;
    auto in = random_image(w, h, 1 + k);
    GaussKernel g = make_gaussian(k);
    std::vector<uint8_t> a(in.size()), b(in.size()), c(in.size());
    std::vector<float> tmp(in.size());
    blur_cpu(in.data(), a.data(), w, h, g);
    blur_ref_double(in.data(), b.data(), w, h, g);
    blur_sep_cpu(in.data(), c.data(), tmp.data(), w, h, g);
    report("cpu vs double ref 37x23 k=" + std::to_string(k), "cpu_vs_double", compare_images(b.data(), a.data(), (size_t)w * h, 3, kTol), kTol);
    report("cpu separable vs double ref 37x23 k=" + std::to_string(k), "cpu_sep_vs_double", compare_images(b.data(), c.data(), (size_t)w * h, 3, kTol), kTol);
  }

  // A flat image must come out unchanged, on both platforms (the weights sum to 1 within rounding).
  for (int k : ks) {
    const int w = 100, h = 60;
    std::vector<uint8_t> in((size_t)w * h * 3, 37), a(in.size()), b(in.size());
    GaussKernel g = make_gaussian(k);
    blur_cpu(in.data(), a.data(), w, h, g);
    blur_naive_cuda(in.data(), b.data(), w, h, g);
    std::vector<uint8_t> t(in.size());
    blur_tiled_cuda(in.data(), t.data(), w, h, g);
    report("flat 37: cuda tiled vs input k=" + std::to_string(k), "cuda_tiled_flat", compare_images(in.data(), t.data(), (size_t)w * h, 3, 0), 0);
    report("flat 37: cpu vs input k=" + std::to_string(k), "cpu_flat", compare_images(in.data(), a.data(), (size_t)w * h, 3, 0), 0);
    report("flat 37: cuda vs input k=" + std::to_string(k), "cuda_flat", compare_images(in.data(), b.data(), (size_t)w * h, 3, 0), 0);
  }

  // Flat image through the separable kernel for the whole sweep (the tiled kernel cannot hold k=127).
  for (int k : kall) {
    const int w = 100, h = 60;
    std::vector<uint8_t> in((size_t)w * h * 3, 37), s(in.size());
    blur_sep_cuda(in.data(), s.data(), w, h, make_gaussian(k));
    report("flat 37: cuda separable vs input k=" + std::to_string(k), "cuda_sep_flat", compare_images(in.data(), s.data(), (size_t)w * h, 3, 0), 0);
  }

  // Sizes: smaller than the kernel, one row or column, not a multiple of the block, exact multiple.
  const int sizes[][2] = {{1, 1}, {1, 9}, {9, 1}, {5, 5}, {17, 33}, {16, 16}, {250, 3}, {257, 129}};
  for (auto& s : sizes) {
    auto in = random_image(s[0], s[1], 7);
    const std::string base = "random " + std::to_string(s[0]) + "x" + std::to_string(s[1]);
    for (int k : kall) check_cuda(base, in.data(), s[0], s[1], k, 16, 16);
  }
  // Block shapes from the experiment matrix, on a size that divides none of them evenly.
  {
    auto in = random_image(301, 203, 9);
    for (auto b : {std::pair<int, int>{8, 8}, {16, 16}, {32, 32}, {32, 4}, {1, 1}})
      for (int k : {3, 11, 63}) check_cuda("random 301x203", in.data(), 301, 203, k, b.first, b.second);
  }

  for (int i = 1; i < argc; i++) {
    Image img;
    if (!load_image(argv[i], img)) { fprintf(stderr, "cannot load %s\n", argv[i]); return 2; }
    for (int k : ks) check_cuda(std::filesystem::path(argv[i]).filename().string(), img.px.data(), img.w, img.h, k, 16, 16);
  }
  printf(failed ? "RESULT: FAIL\n" : "RESULT: PASS\n");
  return failed ? 1 : 0;
}
