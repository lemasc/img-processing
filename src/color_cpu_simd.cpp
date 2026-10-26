// Ladder rungs 2 and 6 for the Stage 1 color adjustment, built with -O3 -mavx2 -mfma -fopenmp.
//   color_cpu_autovec      rung 2: the scalar source, left to the compiler's auto-vectorizer
//   color_cpu_autovec_omp  the same loop split across threads with OpenMP
//   color_cpu_avx2         hand-written AVX2 intrinsics, 16 pixels per iteration, one thread
//   color_cpu_avx2_omp     rung 6: the intrinsics version split across threads with OpenMP
#include <immintrin.h>
#include <omp.h>
#include "color.h"

void color_cpu_autovec(const uint8_t* __restrict in, uint8_t* __restrict out, size_t n, const ColorParams& p) {
  for (size_t i = 0; i < n; i++) adjust_pixel(in + 3 * i, out + 3 * i, p);
}

void color_cpu_autovec_omp(const uint8_t* __restrict in, uint8_t* __restrict out, size_t n, const ColorParams& p) {
#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)n; i++) adjust_pixel(in + 3 * i, out + 3 * i, p);
}

namespace {

// Byte shuffle masks that split 16 interleaved RGB pixels (three 16-byte loads, 48 bytes) into one
// register per channel, and merge them back. Built from the layout (byte 3p+c) instead of by hand.
struct Masks {
  __m128i split[3][3];  // [channel][source register]: bytes of that channel found in the register
  __m128i merge[3][3];  // [destination register][channel]: where the channel's lanes go
  Masks() {
    alignas(16) int8_t s[3][3][16], m[3][3][16];
    for (auto& a : s) for (auto& b : a) for (auto& v : b) v = -128;
    for (auto& a : m) for (auto& b : a) for (auto& v : b) v = -128;
    for (int p = 0; p < 16; p++)
      for (int c = 0; c < 3; c++) {
        int j = 3 * p + c;  // byte position in the 48-byte block
        s[c][j / 16][p] = (int8_t)(j % 16);
        m[j / 16][c][j % 16] = (int8_t)p;
      }
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        split[i][j] = _mm_load_si128((const __m128i*)s[i][j]);
        merge[i][j] = _mm_load_si128((const __m128i*)m[i][j]);
      }
  }
};
const Masks& masks() { static const Masks m; return m; }

struct Consts {
  __m256 inv255, bright, contrast, half, zero, one, sh, hi, wr, wg, wb, s255;
  explicit Consts(const ColorParams& p)
      : inv255(_mm256_set1_ps(1.f / 255.f)), bright(_mm256_set1_ps(p.brightness)),
        contrast(_mm256_set1_ps(p.contrast)), half(_mm256_set1_ps(0.5f)), zero(_mm256_setzero_ps()),
        one(_mm256_set1_ps(1.f)), sh(_mm256_set1_ps(p.shadows)), hi(_mm256_set1_ps(p.highlights)),
        wr(_mm256_set1_ps(0.2126f)), wg(_mm256_set1_ps(0.7152f)), wb(_mm256_set1_ps(0.0722f)),
        s255(_mm256_set1_ps(255.f)) {}
};

inline __m256 clamp01(__m256 x, const Consts& k) { return _mm256_min_ps(_mm256_max_ps(x, k.zero), k.one); }

// The same math as adjust_pixel, for 8 pixels of one channel each (values 0..255 as int32).
inline void adjust8(__m256i r8, __m256i g8, __m256i b8, const Consts& k, __m256i out[3]) {
  __m256 c[3] = {_mm256_cvtepi32_ps(r8), _mm256_cvtepi32_ps(g8), _mm256_cvtepi32_ps(b8)};
  for (int i = 0; i < 3; i++) {
    __m256 x = _mm256_add_ps(_mm256_mul_ps(c[i], k.inv255), k.bright);
    x = _mm256_fmadd_ps(_mm256_sub_ps(x, k.half), k.contrast, k.half);
    c[i] = clamp01(x, k);
  }
  __m256 lum = _mm256_fmadd_ps(k.wr, c[0], _mm256_fmadd_ps(k.wg, c[1], _mm256_mul_ps(k.wb, c[2])));
  __m256 il = _mm256_sub_ps(k.one, lum);
  __m256 delta = _mm256_fmadd_ps(k.sh, _mm256_mul_ps(il, il), _mm256_mul_ps(k.hi, _mm256_mul_ps(lum, lum)));
  for (int i = 0; i < 3; i++) {
    __m256 y = clamp01(_mm256_add_ps(c[i], delta), k);
    out[i] = _mm256_cvttps_epi32(_mm256_fmadd_ps(y, k.s255, k.half));
  }
}

// Widen 16 bytes to two vectors of 8 int32.
inline void widen(__m128i v, __m256i& lo, __m256i& hi) {
  lo = _mm256_cvtepu8_epi32(v);
  hi = _mm256_cvtepu8_epi32(_mm_srli_si128(v, 8));
}

// Narrow two vectors of 8 int32 (all in 0..255) to 16 bytes in order.
inline __m128i narrow(__m256i lo, __m256i hi) {
  __m256i w = _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8);  // 16 x u16 in order
  __m256i b = _mm256_packus_epi16(w, w);                                    // bytes, each half doubled
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1));
}

// 16 pixels (48 bytes) in, 48 bytes out.
inline void block16(const uint8_t* in, uint8_t* out, const Masks& m, const Consts& k) {
  __m128i src[3] = {_mm_loadu_si128((const __m128i*)in), _mm_loadu_si128((const __m128i*)(in + 16)),
                    _mm_loadu_si128((const __m128i*)(in + 32))};
  __m128i ch[3];
  for (int c = 0; c < 3; c++)
    ch[c] = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(src[0], m.split[c][0]), _mm_shuffle_epi8(src[1], m.split[c][1])),
                         _mm_shuffle_epi8(src[2], m.split[c][2]));
  __m256i lo[3], hi[3], olo[3], ohi[3];
  for (int c = 0; c < 3; c++) widen(ch[c], lo[c], hi[c]);
  adjust8(lo[0], lo[1], lo[2], k, olo);
  adjust8(hi[0], hi[1], hi[2], k, ohi);
  __m128i oc[3];
  for (int c = 0; c < 3; c++) oc[c] = narrow(olo[c], ohi[c]);
  for (int r = 0; r < 3; r++)
    _mm_storeu_si128((__m128i*)(out + 16 * r),
                     _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(oc[0], m.merge[r][0]), _mm_shuffle_epi8(oc[1], m.merge[r][1])),
                                  _mm_shuffle_epi8(oc[2], m.merge[r][2])));
}

}  // namespace

void color_cpu_avx2(const uint8_t* __restrict in, uint8_t* __restrict out, size_t n, const ColorParams& p) {
  const Masks& m = masks();
  const Consts k(p);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) block16(in + 3 * i, out + 3 * i, m, k);
  for (; i < n; i++) adjust_pixel(in + 3 * i, out + 3 * i, p);  // tail
}

void color_cpu_avx2_omp(const uint8_t* __restrict in, uint8_t* __restrict out, size_t n, const ColorParams& p) {
  const Masks& m = masks();
  const Consts k(p);
  const long long blocks = (long long)(n / 16);
#pragma omp parallel for schedule(static)
  for (long long b = 0; b < blocks; b++) block16(in + 48 * b, out + 48 * b, m, k);
  for (size_t i = (size_t)blocks * 16; i < n; i++) adjust_pixel(in + 3 * i, out + 3 * i, p);  // tail
}
