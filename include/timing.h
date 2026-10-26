#pragma once
// Shared measurement harness: warm-up, repeated runs, median/min/max, CSV output.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct Stats {
  double median, min, max;
};

inline Stats summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v.front(), v.back()};
}

// CPU timing in milliseconds. One untimed warm-up run, then `runs` timed runs.
inline Stats time_cpu(const std::function<void()>& fn, int runs = 20) {
  fn();
  std::vector<double> t;
  for (int i = 0; i < runs; i++) {
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

// GPU timing in milliseconds using CUDA events. One untimed warm-up run.
inline Stats time_gpu(const std::function<void()>& fn, int runs = 20) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> t;
  for (int i = 0; i < runs; i++) {
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
inline void csv_row(const std::string& path, const std::string& kernel,
                    const std::string& platform, double megapixels,
                    const std::string& params, const Stats& s,
                    double bytes_moved = 0) {
  bool fresh = false;
  if (FILE* f = fopen(path.c_str(), "r")) fclose(f); else fresh = true;
  FILE* f = fopen(path.c_str(), "a");
  if (fresh)
    fprintf(f, "kernel,platform,megapixels,params,median_ms,min_ms,max_ms,mpix_per_s,gb_per_s\n");
  fprintf(f, "%s,%s,%.2f,%s,%.4f,%.4f,%.4f,%.2f,%.2f\n", kernel.c_str(),
          platform.c_str(), megapixels, params.c_str(), s.median, s.min, s.max,
          megapixels / (s.median / 1000.0),
          bytes_moved > 0 ? bytes_moved / (s.median / 1000.0) / 1e9 : 0.0);
  fclose(f);
}
