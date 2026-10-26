// CUDA implementation of the Stage 4 naive Gaussian blur.
#include "blur.h"
#include "timing.h"  // CUDA_CHECK

// One thread per output pixel. Every tap is a separate global-memory load, with no shared memory
// and no reuse between neighbouring threads except what the L1/L2 caches give. The weights travel
// as a kernel argument (constant bank, broadcast to the warp). Threads in a warp are adjacent in x,
// so loads of one tap row are contiguous.
__global__ void blur_naive_kernel(const uint8_t* in, uint8_t* out, int w, int h, GaussKernel g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) blur_pixel(in, w, h, x, y, g, out + 3 * ((size_t)y * w + x));
}

void blur_naive_launch(const uint8_t* d_in, uint8_t* d_out, int w, int h, const GaussKernel& g, int bx, int by) {
  dim3 block(bx, by), grid((w + bx - 1) / bx, (h + by - 1) / by);
  blur_naive_kernel<<<grid, block>>>(d_in, d_out, w, h, g);
}

void blur_naive_cuda(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g, int bx, int by) {
  const size_t bytes = (size_t)w * h * 3;
  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
  blur_naive_launch(d_in, d_out, w, h, g, bx, by);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
}
