// Stage 4 correctness check of the blur implementations.
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

static void check_cuda(const std::string& label, const uint8_t* in, int w, int h, int k, int bx, int by) {
  GaussKernel g = make_gaussian(k);
  std::vector<uint8_t> ref((size_t)w * h * 3), out(ref.size());
  blur_cpu(in, ref.data(), w, h, g);
  blur_naive_cuda(in, out.data(), w, h, g, bx, by);
  report(label + " k=" + std::to_string(k) + " blk=" + std::to_string(bx) + "x" + std::to_string(by), "cuda_naive",
         compare_images(ref.data(), out.data(), (size_t)w * h, 3, kTol), kTol);
}

int main(int argc, char** argv) {
  const int ks[] = {3, 5, 7, 11};
  printf("tolerance: %d level(s) per channel (of 255)\n", kTol);

  for (int k : ks) {  // weights sum to 1
    GaussKernel g = make_gaussian(k);
    double s = 0;
    for (int i = 0; i < k; i++) s += g.w[i];
    if (std::abs(s - 1.0) > 1e-6) { printf("weights k=%d sum %.9f FAIL\n", k, s); failed = true; }
  }

  // The CPU reference against an independent double-precision loop (guards against a shared bug).
  for (int k : ks) {
    const int w = 37, h = 23;
    auto in = random_image(w, h, 1 + k);
    GaussKernel g = make_gaussian(k);
    std::vector<uint8_t> a(in.size()), b(in.size());
    blur_cpu(in.data(), a.data(), w, h, g);
    blur_ref_double(in.data(), b.data(), w, h, g);
    report("cpu vs double ref 37x23 k=" + std::to_string(k), "cpu_vs_double", compare_images(b.data(), a.data(), (size_t)w * h, 3, kTol), kTol);
  }

  // A flat image must come out unchanged, on both platforms (the weights sum to 1 within rounding).
  for (int k : ks) {
    const int w = 100, h = 60;
    std::vector<uint8_t> in((size_t)w * h * 3, 37), a(in.size()), b(in.size());
    GaussKernel g = make_gaussian(k);
    blur_cpu(in.data(), a.data(), w, h, g);
    blur_naive_cuda(in.data(), b.data(), w, h, g);
    report("flat 37: cpu vs input k=" + std::to_string(k), "cpu_flat", compare_images(in.data(), a.data(), (size_t)w * h, 3, 0), 0);
    report("flat 37: cuda vs input k=" + std::to_string(k), "cuda_flat", compare_images(in.data(), b.data(), (size_t)w * h, 3, 0), 0);
  }

  // Sizes: smaller than the kernel, one row or column, not a multiple of the block, exact multiple.
  const int sizes[][2] = {{1, 1}, {1, 9}, {9, 1}, {5, 5}, {17, 33}, {16, 16}, {250, 3}, {257, 129}};
  for (auto& s : sizes) {
    auto in = random_image(s[0], s[1], 7);
    const std::string base = "random " + std::to_string(s[0]) + "x" + std::to_string(s[1]);
    for (int k : ks) check_cuda(base, in.data(), s[0], s[1], k, 16, 16);
  }
  // Block shapes from the experiment matrix, on a size that divides none of them evenly.
  {
    auto in = random_image(301, 203, 9);
    for (auto b : {std::pair<int, int>{8, 8}, {16, 16}, {32, 32}, {32, 4}, {1, 1}})
      check_cuda("random 301x203", in.data(), 301, 203, 7, b.first, b.second);
  }

  for (int i = 1; i < argc; i++) {
    Image img;
    if (!load_image(argv[i], img)) { fprintf(stderr, "cannot load %s\n", argv[i]); return 2; }
    for (int k : ks) check_cuda(std::filesystem::path(argv[i]).filename().string(), img.px.data(), img.w, img.h, k, 16, 16);
  }
  printf(failed ? "RESULT: FAIL\n" : "RESULT: PASS\n");
  return failed ? 1 : 0;
}
