// CUDA implementation of the Stage 1 color adjustment.
#include <cstdio>
#include <cstdlib>
#include "color.h"
#include "timing.h"  // CUDA_CHECK

// One thread per pixel. A warp reads 96 contiguous bytes, so loads and stores coalesce.
__global__ void color_kernel(const uint8_t* in, uint8_t* out, size_t npixels, ColorParams p) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < npixels) adjust_pixel(in + 3 * i, out + 3 * i, p);
}

void color_cuda_launch(const uint8_t* d_in, uint8_t* d_out, size_t npixels, const ColorParams& p, int block) {
  int grid = (int)((npixels + block - 1) / block);
  color_kernel<<<grid, block>>>(d_in, d_out, npixels, p);
}

void color_cuda(const uint8_t* in, uint8_t* out, size_t npixels, const ColorParams& p) {
  const size_t bytes = npixels * 3;
  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
  color_cuda_launch(d_in, d_out, npixels, p);
  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
}
