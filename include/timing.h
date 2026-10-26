#pragma once
// Shared measurement harness: warm-up, cache flush, repeated runs, median/min/max, CSV output.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

struct Stats {
  double median, min, max;
  int runs;
};

inline Stats summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v.front(), v.back(), (int)v.size()};
}

// Evict the CPU caches (L3 is 20 MB on the i5-13400F) by writing a 64 MB buffer.
// Called before each timed run, outside the timed region, so small images are not served from cache.
inline void flush_cpu_cache() {
  static std::vector<char> buf(64u << 20);
  static char v = 0;
  memset(buf.data(), ++v, buf.size());
  asm volatile("" : : "r"(buf.data()) : "memory");  // keep the compiler from dropping the writes
}

// CPU timing in milliseconds. One untimed warm-up run, then `runs` timed runs.
inline Stats time_cpu(const std::function<void()>& fn, int runs = 20, bool flush = true) {
  fn();
  std::vector<double> t;
  for (int i = 0; i < runs; i++) {
    if (flush) flush_cpu_cache();
    auto a = std::chrono::steady_clock::now();
    fn();
    auto b = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
  }
  return summarize(t);
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t e_ = (x);                                                    \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),    \
              __FILE__, __LINE__);                                           \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// Evict the GPU L2 (3 MB on the RTX 3060) by writing 64 MB. Stream-ordered before the start
// event, so it is not part of the timed interval.
inline void flush_gpu_cache() {
  static void* buf = nullptr;
  if (!buf) CUDA_CHECK(cudaMalloc(&buf, 64u << 20));
  CUDA_CHECK(cudaMemset(buf, 0, 64u << 20));
}

// GPU timing in milliseconds using CUDA events. One untimed warm-up run.
inline Stats time_gpu(const std::function<void()>& fn, int runs = 20, bool flush = true) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> t;
  for (int i = 0; i < runs; i++) {
    if (flush) flush_gpu_cache();
    CUDA_CHECK(cudaEventRecord(s));
    fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    t.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return summarize(t);
}
#endif

// Append one row to a CSV file, writing the header if the file is new.
// bytes_moved: bytes read + written per run (0 = unknown). peak_gbs: peak to compare against
// (0 = no percentage, e.g. for PCIe transfers).
inline void csv_row(const std::string& path, const std::string& kernel,
                    const std::string& platform, double megapixels,
                    const std::string& params, const Stats& s,
                    double bytes_moved = 0, double peak_gbs = 0) {
  bool fresh = false;
  if (FILE* f = fopen(path.c_str(), "r")) fclose(f); else fresh = true;
  FILE* f = fopen(path.c_str(), "a");
  if (fresh)
    fprintf(f, "kernel,platform,megapixels,params,runs,median_ms,min_ms,max_ms,mpix_per_s,gb_per_s,pct_peak\n");
  double gbs = bytes_moved > 0 ? bytes_moved / (s.median / 1000.0) / 1e9 : 0.0;
  fprintf(f, "%s,%s,%.2f,%s,%d,%.4f,%.4f,%.4f,%.2f,%.2f,", kernel.c_str(), platform.c_str(),
          megapixels, params.c_str(), s.runs, s.median, s.min, s.max,
          megapixels / (s.median / 1000.0), gbs);
  if (gbs > 0 && peak_gbs > 0) fprintf(f, "%.1f\n", 100.0 * gbs / peak_gbs);
  else fprintf(f, "\n");
  fclose(f);
}
