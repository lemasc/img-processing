#pragma once
// Stage 4 Gaussian blur: k x k kernel on interleaved 8-bit RGB, edges clamped to the border.
// One definition of the per-pixel math is shared by the CPU reference and the CUDA kernels.
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

constexpr int kMaxTaps = 127;  // radius 63, the largest size in the Stage 6 sweep

// 1D Gaussian weights. The 2D weight of tap (j, i) is w[j] * w[i], so the 2D kernel is separable
// by construction and Stage 6 can reuse these weights.
struct GaussKernel {
  int k = 3;  // odd
  float w[kMaxTaps] = {};
};

// sigma follows the OpenCV default for a given size, so results compare with common tools.
inline GaussKernel make_gaussian(int k) {
  GaussKernel g;
  g.k = k;
  const double sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;
  const int r = k / 2;
  double sum = 0, tmp[kMaxTaps];
  for (int i = 0; i < k; i++) sum += tmp[i] = std::exp(-0.5 * (i - r) * (i - r) / (sigma * sigma));
  for (int i = 0; i < k; i++) g.w[i] = (float)(tmp[i] / sum);
  return g;
}

HD inline int clamp_idx(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// Blur one output pixel at (x, y) of a w x h image.
HD inline void blur_pixel(const uint8_t* in, int w, int h, int x, int y, const GaussKernel& g, uint8_t* out) {
  const int r = g.k / 2;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f;
  for (int j = 0; j < g.k; j++) {
    const size_t row = (size_t)clamp_idx(y + j - r, h - 1) * w;
    for (int i = 0; i < g.k; i++) {
      const uint8_t* p = in + 3 * (row + clamp_idx(x + i - r, w - 1));
      const float wt = g.w[j] * g.w[i];
      a0 += wt * p[0];
      a1 += wt * p[1];
      a2 += wt * p[2];
    }
  }
  const float acc[3] = {a0, a1, a2};
  for (int c = 0; c < 3; c++) {
    float v = acc[c] + 0.5f;
    out[c] = (uint8_t)(v > 255.f ? 255.f : v);
  }
}

// CPU reference, defined in src/blur_cpu.cpp (built scalar, no auto-vectorization).
void blur_cpu(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g);

// Slow double-precision reference, used only by the correctness check on small images.
void blur_ref_double(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g);

// CUDA naive version (Stage 4), defined in src/blur_cuda.cu. Global memory only, one thread per pixel.
// Kernel launch only: device pointers, asynchronous, no sync.
void blur_naive_launch(const uint8_t* d_in, uint8_t* d_out, int w, int h, const GaussKernel& g, int bx = 16,
                       int by = 16);
// Convenience: host pointers in and out.
void blur_naive_cuda(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g, int bx = 16, int by = 16);

// CUDA tiled version (Stage 5), defined in src/blur_cuda.cu. Each block loads its tile plus a
// (k-1)-wide halo into shared memory once, then every thread reads its taps from there. Same
// accumulation order as blur_pixel, so the result matches the naive kernel. Returns false (and
// launches nothing) if the tile does not fit in 48 KB of shared memory.
// float_tile keeps the tile as floats (converted once while loading) instead of bytes.
bool blur_tiled_launch(const uint8_t* d_in, uint8_t* d_out, int w, int h, const GaussKernel& g, int bx = 16,
                       int by = 16, bool float_tile = false);
void blur_tiled_cuda(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g, int bx = 16, int by = 16,
                     bool float_tile = false);
// Bytes of shared memory the tiled kernel needs for this kernel size, block shape and element size.
inline size_t blur_tile_bytes(const GaussKernel& g, int bx, int by, int elem = 1) {
  return (size_t)(bx + g.k - 1) * (by + g.k - 1) * 3 * elem;
}
// Theoretical occupancy (resident threads / 1536 on sm_86) for each kernel at this configuration.
double blur_naive_occupancy(int bx, int by);
double blur_tiled_occupancy(const GaussKernel& g, int bx, int by, bool float_tile = false);
