// Experiment: why does the scalar CPU color kernel run slower on noisy images?
// Hypothesis: the clamp compiles to compare-and-branch, and random data mispredicts the branches.
// Variants (set by -DVARIANT): 0 original ternary clamp (branches unless built with
// -ffinite-math-only), 1 fminf/fmaxf (libm calls), 3 std::min/std::max (still branches),
// 4 SSE min/max intrinsics (branch-free). Run via scripts/exp_clamp.sh.
#include <cstdio>
#include <cmath>
#include "color.h"
#include "image.h"
#include "timing.h"

#include <algorithm>
#include <xmmintrin.h>
static inline float sse_clamp(float x) { return _mm_cvtss_f32(_mm_min_ss(_mm_max_ss(_mm_set_ss(x), _mm_setzero_ps()), _mm_set_ss(1.f))); }
#if VARIANT == 4
#define CLAMP(x) sse_clamp(x)
#elif VARIANT == 3
#define CLAMP(x) std::min(std::max((x), 0.f), 1.f)
#elif VARIANT == 1  // fminf/fmaxf
#define CLAMP(x) fminf(fmaxf((x), 0.f), 1.f)
#elif VARIANT == 2  // ternary written to force select, via integer-free min/max
#define CLAMP(x) ((x) > 1.f ? 1.f : ((x) < 0.f ? 0.f : (x)))
#else
#define CLAMP(x) clamp01(x)
#endif

static void run(const uint8_t* in, uint8_t* out, size_t n, const ColorParams& p) {
  for (size_t i = 0; i < n; i++) {
    float c[3];
    for (int k = 0; k < 3; k++) {
      float x = in[3*i+k] * (1.f / 255.f);
      x += p.brightness;
      x = (x - 0.5f) * p.contrast + 0.5f;
      c[k] = CLAMP(x);
    }
    float lum = 0.2126f*c[0] + 0.7152f*c[1] + 0.0722f*c[2];
    float ws = (1.f-lum)*(1.f-lum), wh = lum*lum;
    float d = p.shadows*ws + p.highlights*wh;
    for (int k = 0; k < 3; k++) out[3*i+k] = (uint8_t)(CLAMP(c[k] + d) * 255.f + 0.5f);
  }
}

int main(int argc, char** argv) {
  ColorParams p;
  for (int a = 1; a < argc; a++) {
    Image img; load_image(argv[a], img);
    Image out = img;
    Stats s = time_cpu([&] { run(img.px.data(), out.px.data(), img.npixels(), p); }, 10);
    printf("variant %d  %-24s %8.2f ms\n", VARIANT, argv[a], s.median);
  }
}
