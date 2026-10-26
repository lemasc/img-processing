// CPU reference for Stage 1. Compiled with -fno-tree-vectorize (ladder rung 1: scalar).
#include "color.h"

void color_cpu(const uint8_t* in, uint8_t* out, size_t npixels, const ColorParams& p) {
  for (size_t i = 0; i < npixels; i++) adjust_pixel(in + 3 * i, out + 3 * i, p);
}
