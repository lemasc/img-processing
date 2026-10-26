#include "compare.h"
#include <cstdio>
#include <cstdlib>

DiffStats compare_images(const uint8_t* a, const uint8_t* b, size_t pixels, int channels, int tol) {
  DiffStats d;
  d.pixels = pixels;
  unsigned long long sum = 0;
  for (size_t i = 0; i < pixels; i++) {
    int worst = 0;
    for (int c = 0; c < channels; c++) {
      int diff = abs((int)a[i * channels + c] - (int)b[i * channels + c]);
      d.hist[diff > 3 ? 3 : diff]++;
      sum += diff;
      if (diff > worst) worst = diff;
    }
    if (worst > 0) d.mismatched++;
    if (worst > tol) {
      d.over_tol++;
      if (d.first_bad == (size_t)-1) d.first_bad = i;
    }
    if (worst > d.max_abs) d.max_abs = worst;
  }
  d.mean_abs = pixels ? (double)sum / ((double)pixels * channels) : 0.0;
  return d;
}

std::string describe(const DiffStats& d) {
  char buf[256];
  snprintf(buf, sizeof buf, "max %d, mean %.6f, mismatched %zu/%zu px (%.4f%%), over tol %zu",
           d.max_abs, d.mean_abs, d.mismatched, d.pixels, d.pixels ? 100.0 * d.mismatched / d.pixels : 0.0, d.over_tol);
  return buf;
}

void log_diff(const std::string& csv, const std::string& kernel, const std::string& impl,
              const std::string& input, int tol, const DiffStats& d) {
  bool fresh = true;
  if (FILE* f = fopen(csv.c_str(), "r")) { fclose(f); fresh = false; }
  FILE* f = fopen(csv.c_str(), "a");
  if (fresh) fprintf(f, "kernel,impl,input,tol,pixels,mismatched,over_tol,max_abs,mean_abs,pass\n");
  fprintf(f, "%s,%s,%s,%d,%zu,%zu,%zu,%d,%.8f,%d\n", kernel.c_str(), impl.c_str(), input.c_str(), tol,
          d.pixels, d.mismatched, d.over_tol, d.max_abs, d.mean_abs, passes(d) ? 1 : 0);
  fclose(f);
}
