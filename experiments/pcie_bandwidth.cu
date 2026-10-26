// Experiment: PCIe copy-engine and zero-copy bandwidth by transfer size. Run via scripts/exp_peaks.sh.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
__global__ void sum_k(const float4* p, size_t n, float* out) {
  float s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = p[i]; s += v.x + v.y + v.z + v.w; }
  if (s == 123456.f) *out = s;
}
static float ms(cudaEvent_t a, cudaEvent_t b) { float m; cudaEventElapsedTime(&m, a, b); return m; }
int main() {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  void *d; CK(cudaMalloc(&d, 1u << 30));
  printf("-- copy engine, pinned, by size (H2D / D2H GB/s)\n");
  for (size_t mb : {1, 4, 16, 64, 256, 1024}) {
    size_t n = mb << 20; void* h; CK(cudaMallocHost(&h, n));
    for (int w = 0; w < 3; w++) cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
    int it = mb >= 256 ? 5 : 20; float t1 = 0, t2 = 0;
    for (int i = 0; i < it; i++) {
      cudaEventRecord(a); cudaMemcpy(d, h, n, cudaMemcpyHostToDevice); cudaEventRecord(b); cudaEventSynchronize(b); t1 += ms(a, b);
      cudaEventRecord(a); cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost); cudaEventRecord(b); cudaEventSynchronize(b); t2 += ms(a, b);
    }
    printf("%5zu MB: %5.1f / %5.1f\n", mb, n / (t1 / it / 1e3) / 1e9, n / (t2 / it / 1e3) / 1e9);
    cudaFreeHost(h);
  }
  printf("-- zero-copy: kernel reads 256 MB of mapped pinned host memory directly\n");
  size_t n = 256u << 20; float4* hm; float* o; CK(cudaMalloc(&o, 4));
  CK(cudaHostAlloc((void**)&hm, n, cudaHostAllocMapped)); for (size_t i = 0; i < n / 4; i++) ((float*)hm)[i] = 1.f;
  float4* dm; CK(cudaHostGetDevicePointer((void**)&dm, hm, 0));
  sum_k<<<1024, 256>>>(dm, n / 16, o); cudaDeviceSynchronize();
  float t = 0; for (int i = 0; i < 5; i++) { cudaEventRecord(a); sum_k<<<1024, 256>>>(dm, n / 16, o); cudaEventRecord(b); cudaEventSynchronize(b); t += ms(a, b); }
  printf("zero-copy read: %.1f GB/s\n", n / (t / 5 / 1e3) / 1e9);
}
