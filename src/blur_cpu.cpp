// CPU references for Stage 4. blur_cpu is built scalar (ladder rung 1).
#include "blur.h"

void blur_cpu(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g) {
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) blur_pixel(in, w, h, x, y, g, out + 3 * ((size_t)y * w + x));
}

void blur_ref_double(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g) {
  const int r = g.k / 2;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      for (int c = 0; c < 3; c++) {
        double acc = 0;
        for (int j = 0; j < g.k; j++)
          for (int i = 0; i < g.k; i++)
            acc += (double)g.w[j] * g.w[i] * in[3 * ((size_t)clamp_idx(y + j - r, h - 1) * w + clamp_idx(x + i - r, w - 1)) + c];
        double v = acc + 0.5;
        out[3 * ((size_t)y * w + x) + c] = (uint8_t)(v > 255.0 ? 255.0 : v);
      }
}
