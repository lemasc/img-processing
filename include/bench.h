#pragma once
// Helpers shared by the stage drivers: GPU phase split (H2D / kernel / D2H / end-to-end),
// run tags, and CSV logging for pixel-in/pixel-out kernels. Needs nvcc (uses CUDA).
#include <cstdlib>
#include <string>
#include "timing.h"

constexpr double kGpuPeakGBs = 360.0;  // RTX 3060 memory bandwidth, theoretical
constexpr double kCpuPeakGBs = 76.8;   // dual-channel DDR5-4800, theoretical
// Measured practical PCIe ceiling with pinned memory (results/logs/smoke.txt). The link negotiates
// gen 4 x16 (31.5 GB/s theoretical) but this machine reaches only about 13 GB/s.
constexpr double kPcieH2DGBs = 13.2;
constexpr double kPcieD2HGBs = 11.8;

// Tag for the CSV params column: how the run was pinned, plus caller-supplied settings.
inline std::string run_tag(const std::string& extra = "") {
  const char* c = getenv("BENCH_CPUS");
  std::string t = std::string("cpus=") + (c ? c : "unpinned");
  return extra.empty() ? t : t + ";" + extra;
}

struct GpuPhases {
  Stats h2d, kernel, d2h, e2e;
};

// Time each phase on its own (CUDA events), then the three together as seen by the host.
inline GpuPhases measure_gpu_phases(const std::function<void()>& h2d, const std::function<void()>& kern,
                                    const std::function<void()>& d2h, int runs) {
  GpuPhases g;
  g.h2d = time_gpu(h2d, runs);
  g.kernel = time_gpu(kern, runs);
  g.d2h = time_gpu(d2h, runs);
  g.e2e = time_cpu([&] { h2d(); kern(); d2h(); CUDA_CHECK(cudaDeviceSynchronize()); }, runs);
  return g;
}

// Log one kernel's CPU and GPU results. in_bytes / out_bytes are the image bytes read and written.
inline void log_inout(const std::string& csv, const std::string& kernel, const std::string& cpu_platform,
                      double mp, const std::string& tag, const Stats& cpu, const GpuPhases& g,
                      size_t in_bytes, size_t out_bytes) {
  const double both = (double)(in_bytes + out_bytes);
  csv_row(csv, kernel, cpu_platform, mp, tag, cpu, both, kCpuPeakGBs);
  csv_row(csv, kernel, "cuda_h2d", mp, tag, g.h2d, (double)in_bytes, kPcieH2DGBs);
  csv_row(csv, kernel, "cuda_kernel", mp, tag, g.kernel, both, kGpuPeakGBs);
  csv_row(csv, kernel, "cuda_d2h", mp, tag, g.d2h, (double)out_bytes, kPcieD2HGBs);
  csv_row(csv, kernel, "cuda_e2e", mp, tag, g.e2e, both);
}
