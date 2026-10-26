// Experiment: host DRAM bandwidth (STREAM-style copy and triad). Run via scripts/exp_peaks.sh.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
int main() {
  size_t n = 64u << 20;  // doubles per array = 512 MB
  double *a = (double*)aligned_alloc(64, n * 8), *b = (double*)aligned_alloc(64, n * 8), *c = (double*)aligned_alloc(64, n * 8);
  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) { a[i] = 1; b[i] = 2; c[i] = 0; }
  double best_copy = 0, best_triad = 0;
  for (int r = 0; r < 8; r++) {
    auto t0 = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) c[i] = a[i];
    auto t1 = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
    auto t2 = std::chrono::steady_clock::now();
    double tc = std::chrono::duration<double>(t1 - t0).count(), tt = std::chrono::duration<double>(t2 - t1).count();
    if (2.0 * n * 8 / tc / 1e9 > best_copy) best_copy = 2.0 * n * 8 / tc / 1e9;
    if (3.0 * n * 8 / tt / 1e9 > best_triad) best_triad = 3.0 * n * 8 / tt / 1e9;
  }
  printf("threads=%d  copy %.1f GB/s  triad %.1f GB/s   (check %.0f)\n", omp_get_max_threads(), best_copy, best_triad, a[n / 2]);
}
