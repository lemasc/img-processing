// Ladder rungs 2 and 6 for the separable Gaussian blur, built with -O3 -mavx2 -mfma -fopenmp.
//   blur_sep_autovec[_omp]  the scalar pixel functions from blur.h, left to the auto-vectorizer
//   blur_sep_avx2[_omp]     hand-written AVX2: the row is treated as a flat array of 3w floats, so
//                           tap i of 8 neighbouring floats is one unaligned load at a shift of 3 * i
// Both are the same two-pass algorithm as the scalar rung, with the same float intermediate.
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include "blur.h"

namespace {

void sep_autovec(const uint8_t* __restrict in, uint8_t* __restrict out, float* __restrict tmp, int w, int h,
                 const GaussKernel& g, bool par) {
#pragma omp parallel for schedule(static) if (par)
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) blur_h_pixel(in, w, x, y, g, tmp + 3 * ((size_t)y * w + x));
#pragma omp parallel for schedule(static) if (par)
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) blur_v_pixel(tmp, w, h, x, y, g, out + 3 * ((size_t)y * w + x));
}

inline uint8_t to_byte(float acc) {
  float v = acc + 0.5f;
  return (uint8_t)(v > 255.f ? 255.f : v);
}

// Horizontal pass of one row. pad is scratch for 3 * (w + 2r) floats: the row as floats with r
// replicated pixels on each side, so taps need no clamping.
void h_row(const uint8_t* src, float* dst, float* pad, int w, const GaussKernel& g) {
  const int k = g.k, r = k / 2;
  const int L = 3 * w;
  float* mid = pad + 3 * r;
  int q = 0;
  for (; q + 8 <= L; q += 8)
    _mm256_storeu_ps(mid + q, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + q)))));
  for (; q < L; q++) mid[q] = src[q];
  for (int t = 0; t < r; t++)
    for (int c = 0; c < 3; c++) {
      pad[3 * t + c] = mid[c];
      mid[L + 3 * t + c] = mid[L - 3 + c];
    }
  int f = 0;
  for (; f + 32 <= L; f += 32) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < k; i++) {
      const __m256 wv = _mm256_set1_ps(g.w[i]);
      const float* p = pad + f + 3 * i;
      a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p), a0);
      a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p + 8), a1);
      a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p + 16), a2);
      a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p + 24), a3);
    }
    _mm256_storeu_ps(dst + f, a0);
    _mm256_storeu_ps(dst + f + 8, a1);
    _mm256_storeu_ps(dst + f + 16, a2);
    _mm256_storeu_ps(dst + f + 24, a3);
  }
  for (; f + 8 <= L; f += 8) {
    __m256 a = _mm256_setzero_ps();
    for (int i = 0; i < k; i++) a = _mm256_fmadd_ps(_mm256_set1_ps(g.w[i]), _mm256_loadu_ps(pad + f + 3 * i), a);
    _mm256_storeu_ps(dst + f, a);
  }
  for (; f < L; f++) {
    float a = 0.f;
    for (int i = 0; i < k; i++) a += g.w[i] * pad[f + 3 * i];
    dst[f] = a;
  }
}

constexpr int kStrip = 64;  // floats per column strip: 8 vectors held in registers

// Vertical pass over one column strip of every row. The k rows a window touches are 72 KB apart,
// so a strip keeps them to 64 floats each, small enough for the window to stay in L1/L2.
void v_strip(const float* tmp, uint8_t* out, long R, int h, const GaussKernel& g, long col, int nfloat) {
  const int k = g.k, r = k / 2;
  if (nfloat == kStrip) {
    for (int y = 0; y < h; y++) {
      __m256 a[8];
      for (int v = 0; v < 8; v++) a[v] = _mm256_setzero_ps();
      for (int j = 0; j < k; j++) {
        const float* p = tmp + (long)std::clamp(y + j - r, 0, h - 1) * R + col;
        const __m256 wv = _mm256_set1_ps(g.w[j]);
#pragma GCC unroll 8
        for (int v = 0; v < 8; v++) a[v] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(p + 8 * v), a[v]);
      }
      uint8_t* o = out + (long)y * R + col;
      const __m256 half = _mm256_set1_ps(0.5f), top = _mm256_set1_ps(255.f);
#pragma GCC unroll 8
      for (int v = 0; v < 8; v++) {
        __m256i i32 = _mm256_cvttps_epi32(_mm256_min_ps(_mm256_add_ps(a[v], half), top));
        __m256i p8 = _mm256_packus_epi16(_mm256_packus_epi32(i32, i32), _mm256_setzero_si256());
        uint32_t lo = (uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(p8));
        uint32_t hi = (uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(p8, 1));
        __builtin_memcpy(o + 8 * v, &lo, 4);
        __builtin_memcpy(o + 8 * v + 4, &hi, 4);
      }
    }
    return;
  }
  for (int y = 0; y < h; y++)  // partial strip at the end of the row
    for (int f = 0; f < nfloat; f++) {
      float a = 0.f;
      for (int j = 0; j < k; j++) a += g.w[j] * tmp[(long)std::clamp(y + j - r, 0, h - 1) * R + col + f];
      out[(long)y * R + col + f] = to_byte(a);
    }
}

void sep_avx2(const uint8_t* __restrict in, uint8_t* __restrict out, float* __restrict tmp, int w, int h,
              const GaussKernel& g, bool par) {
  const long R = 3L * w;
  const int r = g.k / 2;
  const int strips = (int)((R + kStrip - 1) / kStrip);
#pragma omp parallel if (par)
  {
    std::vector<float> pad(3 * ((size_t)w + 2 * r));
#pragma omp for schedule(static)
    for (int y = 0; y < h; y++) h_row(in + y * R, tmp + y * R, pad.data(), w, g);
#pragma omp for schedule(static)
    for (int s = 0; s < strips; s++) v_strip(tmp, out, R, h, g, (long)s * kStrip, (int)std::min<long>(kStrip, R - (long)s * kStrip));
  }
}

}  // namespace

void blur_sep_autovec(const uint8_t* in, uint8_t* out, float* tmp, int w, int h, const GaussKernel& g) {
  sep_autovec(in, out, tmp, w, h, g, false);
}
void blur_sep_autovec_omp(const uint8_t* in, uint8_t* out, float* tmp, int w, int h, const GaussKernel& g) {
  sep_autovec(in, out, tmp, w, h, g, true);
}
void blur_sep_avx2(const uint8_t* in, uint8_t* out, float* tmp, int w, int h, const GaussKernel& g) {
  sep_avx2(in, out, tmp, w, h, g, false);
}
void blur_sep_avx2_omp(const uint8_t* in, uint8_t* out, float* tmp, int w, int h, const GaussKernel& g) {
  sep_avx2(in, out, tmp, w, h, g, true);
}
