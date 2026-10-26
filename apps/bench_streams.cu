// Stage 11: pinned host memory and chunked copy/compute overlap with CUDA streams.
// For each workload (color, separable blur) the same image goes through the GPU as:
//   pageable, 1 chunk   the Stage 1/6 end-to-end path (H2D, kernel, D2H one after another)
//   pinned,   1 chunk   same, from page-locked host buffers
//   pinned,   N chunks  one stream per chunk: H2D, kernel, D2H of different chunks overlap
//   pageable, N chunks  control: async copies from pageable memory do not overlap
// Every variant's output must equal the pageable 1-chunk output exactly (tolerance 0).
// Also logs transfer time pageable vs pinned, the host memcpy needed to stage pageable data into
// pinned buffers, and the cost of allocating pinned memory.
// Usage: bench_streams [-r runs] [-k 15,63,127] [-n 2,4,8,16] [-w color,blur] <image>...
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include "bench.h"
#include "blur.h"
#include "color.h"
#include "compare.h"
#include "image.h"

static const char* kCsv = "results/data/streams.csv";

// One workload: device buffers and the per-chunk work for a given chunk count.
struct Workload {
  std::string name;
  std::function<void(int nc)> setup;
  std::function<void()> teardown;
  // Issue chunk c of nc on stream s: H2D from h_in, kernel(s), D2H into h_out. Asynchronous.
  std::function<void(int c, int nc, const uint8_t* h_in, uint8_t* h_out, cudaStream_t s)> issue;
  std::function<void()> kernel_full;  // all kernels of the whole image, default stream (nc = 1 setup)
};

static Workload make_color(size_t n) {
  auto st = std::make_shared<std::pair<uint8_t*, uint8_t*>>(nullptr, nullptr);
  Workload w;
  w.name = "color";
  const size_t bytes = n * 3;
  w.setup = [=](int) { CUDA_CHECK(cudaMalloc(&st->first, bytes)); CUDA_CHECK(cudaMalloc(&st->second, bytes)); };
  w.teardown = [=] { cudaFree(st->first); cudaFree(st->second); };
  w.issue = [=](int c, int nc, const uint8_t* h_in, uint8_t* h_out, cudaStream_t s) {
    const size_t p0 = (size_t)c * n / nc, p1 = (size_t)(c + 1) * n / nc;
    CUDA_CHECK(cudaMemcpyAsync(st->first + 3 * p0, h_in + 3 * p0, 3 * (p1 - p0), cudaMemcpyHostToDevice, s));
    color_cuda_launch(st->first + 3 * p0, st->second + 3 * p0, p1 - p0, ColorParams{}, 256, s);
    CUDA_CHECK(cudaMemcpyAsync(h_out + 3 * p0, st->second + 3 * p0, 3 * (p1 - p0), cudaMemcpyDeviceToHost, s));
  };
  w.kernel_full = [=] { color_cuda_launch(st->first, st->second, n, ColorParams{}, 256, nullptr); };
  return w;
}

struct BlurState {
  std::vector<uint8_t*> in, out;
  std::vector<float*> tmp;
};

static Workload make_blur(int wd, int ht, int k) {
  auto st = std::make_shared<BlurState>();
  const GaussKernel g = make_gaussian(k);
  Workload w;
  w.name = "blur_sep_k" + std::to_string(k);
  w.setup = [=](int nc) {
    st->in.assign(nc, nullptr); st->out.assign(nc, nullptr); st->tmp.assign(nc, nullptr);
    for (int c = 0; c < nc; c++) {
      const int y0 = (int)((long)c * ht / nc), y1 = (int)((long)(c + 1) * ht / nc);
      int ya, yb;
      band_rows(ht, k, y0, y1, &ya, &yb);
      CUDA_CHECK(cudaMalloc(&st->in[c], 3 * (size_t)wd * (yb - ya)));
      CUDA_CHECK(cudaMalloc(&st->tmp[c], 3 * sizeof(float) * (size_t)wd * (yb - ya)));
      CUDA_CHECK(cudaMalloc(&st->out[c], 3 * (size_t)wd * (y1 - y0)));
    }
  };
  w.teardown = [=] {
    for (size_t c = 0; c < st->in.size(); c++) { cudaFree(st->in[c]); cudaFree(st->tmp[c]); cudaFree(st->out[c]); }
  };
  w.issue = [=](int c, int nc, const uint8_t* h_in, uint8_t* h_out, cudaStream_t s) {
    const int y0 = (int)((long)c * ht / nc), y1 = (int)((long)(c + 1) * ht / nc);
    int ya, yb;
    band_rows(ht, k, y0, y1, &ya, &yb);
    const size_t row = 3 * (size_t)wd;
    CUDA_CHECK(cudaMemcpyAsync(st->in[c], h_in + row * ya, row * (yb - ya), cudaMemcpyHostToDevice, s));
    blur_sep_band_launch(st->in[c], st->tmp[c], st->out[c], wd, ht, y0, y1, g, s);
    CUDA_CHECK(cudaMemcpyAsync(h_out + row * y0, st->out[c], row * (y1 - y0), cudaMemcpyDeviceToHost, s));
  };
  w.kernel_full = [=] { blur_sep_band_launch(st->in[0], st->tmp[0], st->out[0], wd, ht, 0, ht, g, nullptr); };
  return w;
}

struct Opts {
  int runs = 20;
  std::vector<int> ks = {15, 63, 127}, ns = {2, 4, 8, 16};
  std::vector<std::string> workloads = {"color", "blur"};
};

static bool run_workload(Workload& w, const Image& img, const std::string& name, const Opts& o, uint8_t* pin_in,
                         uint8_t* pin_out) {
  const size_t n = img.npixels(), bytes = img.bytes();
  const double mp = n / 1e6;
  const std::string tag0 = "img=" + name;
  Image pageable_out = img, ref = img;

  struct Variant { bool pinned; int nc; };
  std::vector<Variant> vs = {{false, 1}, {true, 1}};
  for (int nc : o.ns) vs.push_back({true, nc});
  vs.push_back({false, 4});

  printf("%s, %s (%.1f MP)\n", w.name.c_str(), name.c_str(), mp);
  double base_ms = 0;
  for (const Variant& v : vs) {
    w.setup(v.nc);
    std::vector<cudaStream_t> ss(v.nc);
    for (auto& s : ss) CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const uint8_t* h_in = v.pinned ? pin_in : img.px.data();
    uint8_t* h_out = v.pinned ? pin_out : pageable_out.px.data();
    auto run = [&] {
      for (int c = 0; c < v.nc; c++) w.issue(c, v.nc, h_in, h_out, ss[c]);
      for (auto s : ss) CUDA_CHECK(cudaStreamSynchronize(s));
    };
    Stats s = time_cpu(run, o.runs);
    CUDA_CHECK(cudaGetLastError());
    // Correctness: clear the output, run once, compare with the pageable 1-chunk result, tolerance 0.
    memset(h_out, 0, bytes);
    run();
    if (!v.pinned && v.nc == 1) ref.px.assign(h_out, h_out + bytes);
    DiffStats d = compare_images(ref.px.data(), h_out, n, 3, 0);
    const std::string plat = std::string("gpu_") + (v.pinned ? "pinned" : "pageable") + "_n" + std::to_string(v.nc);
    log_diff("results/data/correctness.csv", w.name, plat, name, 0, d);
    bool ok = passes(d);
    if (!ok) fprintf(stderr, "%s %s: output differs from the pageable 1-chunk output: %s\n", w.name.c_str(), plat.c_str(), describe(d).c_str());
    if (!v.pinned && v.nc == 1) base_ms = s.median;
    printf("  %-8s %2d chunk%s  %8.3f ms  (min %.3f, max %.3f)  x%.2f vs pageable/1  maxdiff %d\n", v.pinned ? "pinned" : "pageable",
           v.nc, v.nc == 1 ? " " : "s", s.median, s.min, s.max, base_ms / s.median, d.max_abs);
    csv_row(kCsv, w.name, plat, mp, run_tag(tag0 + ";chunks=" + std::to_string(v.nc) + ";mem=" + (v.pinned ? "pinned" : "pageable")), s);
    for (auto st : ss) cudaStreamDestroy(st);
    if (v.pinned && v.nc == 1) {  // kernel-only time, for the overlap lower bound
      Stats k = time_gpu(w.kernel_full, o.runs);
      printf("  kernel only (whole image)  %8.3f ms\n", k.median);
      csv_row(kCsv, w.name, "gpu_kernel", mp, run_tag(tag0), k);
    }
    w.teardown();
    if (!ok) return false;
  }
  return true;
}

// Transfer time pageable vs pinned, host staging cost, and pinned allocation cost.
static void run_transfers(const Image& img, const std::string& name, int runs, uint8_t* pin_in, uint8_t* pin_out) {
  const size_t bytes = img.bytes();
  const double mp = img.npixels() / 1e6;
  const std::string tag = run_tag("img=" + name);
  std::vector<uint8_t> pg_out(bytes);
  uint8_t* d;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  struct T { const char* plat; std::function<void()> fn; double peak; };
  T ts[] = {
      {"h2d_pageable", [&] { CUDA_CHECK(cudaMemcpy(d, img.px.data(), bytes, cudaMemcpyHostToDevice)); }, kPcieH2DGBs},
      {"h2d_pinned", [&] { CUDA_CHECK(cudaMemcpy(d, pin_in, bytes, cudaMemcpyHostToDevice)); }, kPcieH2DGBs},
      {"d2h_pageable", [&] { CUDA_CHECK(cudaMemcpy(pg_out.data(), d, bytes, cudaMemcpyDeviceToHost)); }, kPcieD2HGBs},
      {"d2h_pinned", [&] { CUDA_CHECK(cudaMemcpy(pin_out, d, bytes, cudaMemcpyDeviceToHost)); }, kPcieD2HGBs},
  };
  printf("transfers, %s (%.1f MP, %.0f MB)\n", name.c_str(), mp, bytes / 1e6);
  for (auto& t : ts) {
    Stats s = time_gpu(t.fn, runs);
    printf("  %-14s %8.3f ms  %5.1f GB/s\n", t.plat, s.median, bytes / (s.median / 1e3) / 1e9);
    csv_row(kCsv, "transfer", t.plat, mp, tag, s, (double)bytes, t.peak);
  }
  Stats stage = time_cpu([&] { memcpy(pin_in, img.px.data(), bytes); memcpy(pg_out.data(), pin_out, bytes); }, runs);
  printf("  %-14s %8.3f ms  (memcpy pageable -> pinned for the input and pinned -> pageable for the output)\n", "host_staging", stage.median);
  csv_row(kCsv, "transfer", "host_staging", mp, tag, stage, 4.0 * bytes);
  Stats alloc = time_cpu([&] { void *a, *b; CUDA_CHECK(cudaHostAlloc(&a, bytes, cudaHostAllocDefault)); CUDA_CHECK(cudaHostAlloc(&b, bytes, cudaHostAllocDefault)); cudaFreeHost(a); cudaFreeHost(b); }, 5);
  printf("  %-14s %8.3f ms  (cudaHostAlloc + cudaFreeHost of two %.0f MB buffers)\n", "pinned_alloc", alloc.median, bytes / 1e6);
  csv_row(kCsv, "transfer", "pinned_alloc", mp, tag, alloc);
  cudaFree(d);
}

static std::vector<std::string> split(const std::string& v) {
  std::vector<std::string> r;
  std::stringstream ss(v);
  for (std::string t; std::getline(ss, t, ',');) r.push_back(t);
  return r;
}

int main(int argc, char** argv) {
  Opts o;
  int i = 1;
  for (; i + 1 < argc && argv[i][0] == '-'; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "-r") o.runs = atoi(v.c_str());
    else if (a == "-k") { o.ks.clear(); for (auto& t : split(v)) o.ks.push_back(atoi(t.c_str())); }
    else if (a == "-n") { o.ns.clear(); for (auto& t : split(v)) o.ns.push_back(atoi(t.c_str())); }
    else if (a == "-w") o.workloads = split(v);
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (i >= argc) { fprintf(stderr, "usage: %s [-r runs] [-k list] [-n chunk list] [-w color,blur] <image>...\n", argv[0]); return 2; }
  printf("runs=%d, caches flushed before each run, medians; pinned buffers hold the image already (staging is timed separately)\n", o.runs);
  for (; i < argc; i++) {
    Image img;
    if (!load_image(argv[i], img)) { fprintf(stderr, "cannot load %s\n", argv[i]); return 1; }
    const std::string name = std::filesystem::path(argv[i]).filename().string();
    uint8_t *pin_in, *pin_out;
    CUDA_CHECK(cudaHostAlloc(&pin_in, img.bytes(), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&pin_out, img.bytes(), cudaHostAllocDefault));
    memcpy(pin_in, img.px.data(), img.bytes());
    run_transfers(img, name, o.runs, pin_in, pin_out);
    bool ok = true;
    for (const std::string& wl : o.workloads) {
      if (wl == "color") { Workload w = make_color(img.npixels()); ok = run_workload(w, img, name, o, pin_in, pin_out) && ok; }
      else if (wl == "blur")
        for (int k : o.ks) { Workload w = make_blur(img.w, img.h, k); ok = run_workload(w, img, name, o, pin_in, pin_out) && ok; }
    }
    cudaFreeHost(pin_in);
    cudaFreeHost(pin_out);
    if (!ok) return 1;
  }
}
