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

// Tiled version. The block cooperatively copies (bx + k - 1) x (by + k - 1) pixels into shared memory
// (border-clamped, so the halo needs no special cases later), then each thread blurs from there.
// Global loads fall from k*k*3 per pixel to about 3 * (1 + (k-1)/bx) * (1 + (k-1)/by).
// Shared memory is dynamic because the tile size depends on k and the block shape.
// T = uint8_t keeps the tile as bytes, so every tap converts 3 bytes to float. T = float converts
// once per tile pixel while loading, and taps then read floats directly (4x the shared memory).
template <typename T>
__global__ void blur_tiled_kernel(const uint8_t* in, uint8_t* out, int w, int h, GaussKernel g) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* tile = reinterpret_cast<T*>(smem_raw);
  const int r = g.k / 2;
  const int tw = blockDim.x + g.k - 1, th = blockDim.y + g.k - 1;
  const int x0 = blockIdx.x * blockDim.x - r, y0 = blockIdx.y * blockDim.y - r;
  const int nt = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int idx = tid; idx < tw * th; idx += nt) {
    const int ty = idx / tw, tx = idx - ty * tw;
    const uint8_t* p = in + 3 * ((size_t)clamp_idx(y0 + ty, h - 1) * w + clamp_idx(x0 + tx, w - 1));
    T* s = tile + 3 * idx;
    s[0] = p[0];
    s[1] = p[1];
    s[2] = p[2];
  }
  __syncthreads();  // every thread reaches this, including those outside the image

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f;
  for (int j = 0; j < g.k; j++) {
    const T* row = tile + 3 * ((threadIdx.y + j) * tw + threadIdx.x);
    for (int i = 0; i < g.k; i++) {
      const T* p = row + 3 * i;
      const float wt = g.w[j] * g.w[i];
      a0 += wt * p[0];
      a1 += wt * p[1];
      a2 += wt * p[2];
    }
  }
  const float acc[3] = {a0, a1, a2};
  uint8_t* o = out + 3 * ((size_t)y * w + x);
  for (int c = 0; c < 3; c++) {
    float v = acc[c] + 0.5f;
    o[c] = (uint8_t)(v > 255.f ? 255.f : v);
  }
}

bool blur_tiled_launch(const uint8_t* d_in, uint8_t* d_out, int w, int h, const GaussKernel& g, int bx, int by,
                       bool float_tile) {
  const size_t smem = blur_tile_bytes(g, bx, by, float_tile ? 4 : 1);
  if (smem > 48 * 1024) return false;
  dim3 block(bx, by), grid((w + bx - 1) / bx, (h + by - 1) / by);
  if (float_tile) blur_tiled_kernel<float><<<grid, block, smem>>>(d_in, d_out, w, h, g);
  else blur_tiled_kernel<uint8_t><<<grid, block, smem>>>(d_in, d_out, w, h, g);
  return true;
}

void blur_tiled_cuda(const uint8_t* in, uint8_t* out, int w, int h, const GaussKernel& g, int bx, int by,
                     bool float_tile) {
  const size_t bytes = (size_t)w * h * 3;
  uint8_t *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
  if (!blur_tiled_launch(d_in, d_out, w, h, g, bx, by, float_tile)) { fprintf(stderr, "tile does not fit in shared memory\n"); exit(1); }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
}

static double occupancy(const void* kernel, int bx, int by, size_t smem) {
  int blocks = 0, dev = 0, max_threads = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, bx * by, smem));
  return (double)blocks * bx * by / max_threads;
}
double blur_naive_occupancy(int bx, int by) { return occupancy((const void*)blur_naive_kernel, bx, by, 0); }
double blur_tiled_occupancy(const GaussKernel& g, int bx, int by, bool float_tile) {
  return float_tile ? occupancy((const void*)blur_tiled_kernel<float>, bx, by, blur_tile_bytes(g, bx, by, 4))
                    : occupancy((const void*)blur_tiled_kernel<uint8_t>, bx, by, blur_tile_bytes(g, bx, by, 1));
}
