// Stage 3: correctness check of the CUDA color kernel against the CPU reference.
// Usage: check_color [--tol N] [image...]
// Always runs: every possible RGB color once, then pixel counts that do not divide the block size.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include "color.h"
#include "compare.h"
#include "image.h"

static bool failed = false;
static const std::string kCsv = "results/data/correctness.csv";

static void check(const std::string& label, const uint8_t* in, size_t n, int tol, const std::string& impl) {
  std::vector<uint8_t> ref(n * 3), out(n * 3);
  ColorParams p;
  color_cpu(in, ref.data(), n, p);
  color_cuda(in, out.data(), n, p);
  DiffStats d = compare_images(ref.data(), out.data(), n, 3, tol);
  printf("%-34s %s  %s  hist[0,1,2,3+]=%zu,%zu,%zu,%zu\n", label.c_str(), passes(d) ? "PASS" : "FAIL",
         describe(d).c_str(), d.hist[0], d.hist[1], d.hist[2], d.hist[3]);
  if (!passes(d)) {
    failed = true;
    printf("  first pixel over tolerance: %zu (rgb %d,%d,%d: cpu %d,%d,%d gpu %d,%d,%d)\n", d.first_bad,
           in[3 * d.first_bad], in[3 * d.first_bad + 1], in[3 * d.first_bad + 2], ref[3 * d.first_bad],
           ref[3 * d.first_bad + 1], ref[3 * d.first_bad + 2], out[3 * d.first_bad], out[3 * d.first_bad + 1],
           out[3 * d.first_bad + 2]);
  }
  log_diff(kCsv, "color", impl, label, tol, d);
}

int main(int argc, char** argv) {
  int tol = 1, i = 1;
  if (argc > 2 && std::string(argv[1]) == "--tol") { tol = atoi(argv[2]); i = 3; }
  const char* impl = getenv("CHECK_IMPL") ? getenv("CHECK_IMPL") : "cuda";
  printf("tolerance: %d level(s) per channel (of 255)\n", tol);

  // Every 24-bit RGB color exactly once: 4096 x 4096 = 16,777,216 pixels.
  const size_t all = 1u << 24;
  std::vector<uint8_t> colors(all * 3);
  for (size_t k = 0; k < all; k++) {
    colors[3 * k] = (k >> 16) & 255;
    colors[3 * k + 1] = (k >> 8) & 255;
    colors[3 * k + 2] = k & 255;
  }
  check("all 16.7M RGB colors", colors.data(), all, tol, impl);

  // Edge handling: the grid rounds up, so the last block has idle threads.
  for (size_t n : {1, 2, 255, 256, 257, 1023, 1025, 65537})
    check("first " + std::to_string(n) + " px of all-colors", colors.data(), n, tol, impl);

  for (; i < argc; i++) {
    Image img;
    if (!load_image(argv[i], img)) { fprintf(stderr, "cannot load %s\n", argv[i]); return 2; }
    check(std::filesystem::path(argv[i]).filename().string(), img.px.data(), img.npixels(), tol, impl);
  }
  printf(failed ? "RESULT: FAIL\n" : "RESULT: PASS\n");
  return failed ? 1 : 0;
}
