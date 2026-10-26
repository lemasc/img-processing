#pragma once
// Stage 1 color adjustment: per-pixel brightness, contrast, shadows, highlights.
// One definition shared by the CPU reference and the CUDA kernel so both run the same math.
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

struct ColorParams {
  float brightness = 0.05f;  // added to each channel, in [0,1] units
  float contrast = 1.2f;     // scale around mid-gray 0.5
  float shadows = 0.3f;      // lift, weighted toward dark pixels
  float highlights = -0.3f;  // pull down, weighted toward bright pixels
};

HD inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

// rgb points to 3 bytes. Reads and writes 3 bytes.
HD inline void adjust_pixel(const uint8_t* in, uint8_t* out, const ColorParams& p) {
  float c[3];
  for (int i = 0; i < 3; i++) {
    float x = in[i] * (1.f / 255.f);
    x += p.brightness;
    x = (x - 0.5f) * p.contrast + 0.5f;
    c[i] = clamp01(x);
  }
  float lum = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
  float ws = (1.f - lum) * (1.f - lum);  // shadow weight: strong in darks
  float wh = lum * lum;                  // highlight weight: strong in brights
  float delta = p.shadows * ws + p.highlights * wh;
  for (int i = 0; i < 3; i++) out[i] = (uint8_t)(clamp01(c[i] + delta) * 255.f + 0.5f);
}

// All implementations share this shape: (in, out, npixels, params), interleaved RGB bytes.

// CPU reference, defined in src/color_cpu.cpp (built scalar, no auto-vectorization).
void color_cpu(const uint8_t* in, uint8_t* out, size_t npixels, const ColorParams& p);

// CUDA, defined in src/color_cuda.cu.
// Kernel launch only: d_in and d_out are device pointers. Asynchronous, no sync.
void color_cuda_launch(const uint8_t* d_in, uint8_t* d_out, size_t npixels, const ColorParams& p, int block = 256);
// Convenience: host pointers in and out. Allocates, copies, runs, copies back, frees.
void color_cuda(const uint8_t* in, uint8_t* out, size_t npixels, const ColorParams& p);
