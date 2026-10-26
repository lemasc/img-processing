# HCT Assignment: Project Plan

**Title:** Performance Analysis of Image Color Processing Across Parallel Computing Platforms
**Course:** Hybrid Computing Technology (HCT), Computing Code Development and Process
**Author:** Sakditach Thanasodsai, 67070173
**Deadline:** Before the final exam, submitted via Microsoft Teams Assignment (PDF or Word)

---

## 1. Goal

Build a set of image processing programs and measure how different kinds of parallelism perform on them. The focus is **performance analysis**: where time goes, what the bottleneck is, and when each platform wins.

**Central question:** For image processing workloads, when does GPU parallelism (CUDA) beat CPU parallelism (SIMD, multi-core, WASM), and why?

**Central concept:** Arithmetic intensity (operations per byte moved) decides whether a workload is memory-bound or compute-bound, and therefore which platform wins.

---

## 2. Hardware Under Test

| Component | Spec | Why it matters |
|---|---|---|
| GPU | NVIDIA RTX 3060 12 GB | 360 GB/s memory bandwidth, about 12.7 TFLOPS FP32 (theoretical peaks) |
| CPU | Intel Core i5-13400F | 6 P-cores + 4 E-cores, 16 threads, AVX2 (no AVX-512) |
| OS | Linux | |
| Host RAM | 2 x 8 GB Kingston DDR5-4800 (KF552C40-8), dual-channel, 16 GB total | About 76.8 GB/s theoretical peak, the CPU-side bandwidth ceiling |
| PCIe | Gen 4 x16 capable (idles at gen 1, verify link under load) | Limits host-to-device transfers |

Installed software: driver 595.71.05, CUDA Toolkit 13.2 (target `sm_86`), gcc 13.3, CMake 3.28, Ninja 1.11, Node.js 24.19, Nsight Compute and Nsight Systems, uv 0.11 (Python 3.12). Emscripten via `emsdk` is installed at Stage 12. Final versions are recorded by `scripts/env_report.sh` into `results/env.txt`.

**Measured ceilings (use these in the roofline, with theoretical peaks shown dashed).** From `scripts/exp_peaks.sh`, log in `results/logs/exp_peaks.txt`:

| Resource | Theoretical | Measured |
|---|---|---|
| GPU memory | 360 GB/s | 331 GB/s (copy kernel, `results/logs/smoke.txt`) |
| Host DRAM, 12 threads | 76.8 GB/s | 51.4 GB/s (STREAM triad) |
| Host DRAM, 1 thread | n/a | 27.9 GB/s (triad), 23.3 GB/s (copy) |
| PCIe H2D / D2H, pinned | 31.5 GB/s (gen 4 x16) | 13.2 / 11.8 GB/s |
| PCIe kernel reading host memory (zero-copy) | | 10.6 GB/s |

PCIe finding: the link negotiates 16 GT/s x16 (gen 4) under load, yet copies of 4 MB and larger cap at about 13 GB/s in both directions (1 MB copies reach 20 GB/s, probably from CPU cache). Host DRAM is not the limit (51 GB/s). Cause not found without root (candidates: IOMMU in lazy-translation mode, BIOS or root-port settings, max payload size); reported as a platform limitation. The link idles at gen 1 and ramps under load, so benchmarks must warm up.

**Machine setup for benchmarking:** CPU governor set to `performance`. Check PCIe link generation under load. Pin CPU threads to P-cores for the main runs and report 6-thread and 16-thread results separately (hybrid P/E-core CPU).

---

## 2a. Test Images

Real photos are 24 to 32 MP at most.

- **Natural content:** real photos at 24 to 32 MP. Smaller sizes (1, 6 MP) are made by area-average downscaling.
- **48 MP:** made by upscaling a real photo and labelled "upscaled" in the report. Per-MP time for the real 32 MP photo matches the upscaled photos, and the color kernel is content-independent on both CPU and GPU (24 MP: flat, photo, noise all 164 ms on CPU, 0.47 to 0.48 ms on GPU), so upscaled images are safe for it.
- **Content-dependent kernels (histogram, median):** use real photos plus generated flat and noise images, never upscaled ones.
- **Flat and noise images:** generated at every size.

---

## 2b. Environment and Safety (remote machine)

The machine is remote with 16 GB RAM, and an out-of-memory event has taken it down before. Precautions:

- Run builds and benchmarks inside a memory-capped cgroup, for example `systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 <cmd>`, so the OOM killer hits the job and not the machine.
- Pinned host memory cannot be paged out and counts against the cgroup cap. Stage 11 pins at most two images (in and out), 288 MB at 48 MP, allocated once per run.
- Limit build parallelism (`ninja -j4`). CUDA compiles are memory-heavy.
- Check available memory before large runs. Budget buffers explicitly (a 48 MP float RGB image is about 576 MB, and blur needs ping-pong copies).
- Cap repetitions for slow CPU rungs (for example 3 runs for scalar 63-radius 2D blur at 24 MP and above).
- Run long jobs under `tmux` so an SSH drop does not kill them.
- Python tooling is managed with `uv` (project venv, `uv sync`), not system pip.

---

## 3. Scope Tiers

Each stage is a working checkpoint. The report is valid at any tier boundary.

| Tier | Stages | Outcome |
|---|---|---|
| **Submission scope (decided)** | 1 to 6 (done), 11 (pinned memory and streams), 12 (ladder on color and blur), roofline and charts | Complete, gradable report |
| **Optional inside scope** | WASM rungs 3 and 4 (scalar, SIMD128) in Node | Extra platform angle, drop if setup runs long |
| **Deferred (personal project, future work in the report)** | 7, 8, 9, 10, median filter, WASM threads (rung 5) | Not needed for the rubric |

Scope decision (revised after finishing Stage 6): stages 1 to 6 already cover source code, development steps and experiment summary for the rubric. What is missing is the CPU-vs-GPU comparison (Stage 12), the PCIe follow-up (Stage 11), and the write-up. The deferred stages stay in the roadmap below with their descriptions, and the report lists them as future work.

Rule: do not start a stage until the previous one is measured and logged (command and output saved under `results/logs/`).

---

## 4. Stage Roadmap

### Stage 1: Baseline color processing
- **Build:** CPU reference and CUDA kernel for brightness, contrast, highlights, shadows (per-pixel, luminance-weighted).
- **Input:** full-size camera image. **Output:** adjusted image.
- **Measure:** kernel time, total time, speedup.

### Stage 2: Measurement harness
- **Build:** timing infrastructure shared by all later stages, with warm-up run, repeated runs, and CSV output.
- **Measure:** median of about 20 runs, time split (H2D copy, kernel, D2H copy), effective bandwidth as % of the 360 GB/s peak.
- **Finding (clamp branches):** the first scalar baseline took 169 / 195 / 290 ms on flat / photo / noise 24 MP, because GCC compiled the clamp to compare-and-branch and random data mispredicts. A branch-free clamp (`-ffinite-math-only -fno-signed-zeros`, giving `minss`/`maxss`) takes 161 ms on all three. The baseline uses that, since it is the fairer scalar rung (the GPU clamp is branch-free too). Reproduce with `scripts/exp_clamp.sh`; log in `results/logs/exp_clamp.txt`. `fminf`/`fmaxf` without those flags are 3x slower (libm calls). The median filter experiment is where data-dependent branching matters on purpose.
- **Built:** `include/timing.h` (warm-up, cache flush before every timed run, median/min/max, CSV), `include/bench.h` (phase split, peaks). Caches are flushed because the 3060 has only 3 MB of L2 and small images would otherwise report impossible bandwidth.

### Stage 3: Correctness check
- **Build:** pixel-by-pixel comparison of GPU and CPU output.
- **Measure:** max difference, mean difference, count of mismatched pixels.
- **Purpose:** every later optimization must pass this check.
- **Tolerance:** at most 1 level (of 255) per channel for float implementations; fixed-point ladder rungs get their own tolerance and report the error. Counts are reported with tolerance 0 as well.
- **Built:** `include/compare.h`, `src/compare.cpp` (self-tested in `apps/test_compare.cpp`), `apps/check_color.cu`, run by `ctest`. Checks every possible RGB color once (16.7M pixels), pixel counts that do not divide the block size, and real photos. `bench_color` refuses to report timings for output outside tolerance.
- **Result:** CPU and GPU differ by at most 1 level, on 224 of 16.7M colors (0.0013%), and 39 of 24M pixels on a night photo. Cause: nvcc fuses multiply-adds (one rounding), the CPU does not (two). With `--fmad=false` the GPU is bit-exact on all inputs (`scripts/exp_fma.sh`, `results/logs/exp_fma.txt`). Float rounding differences are therefore expected between platforms, not bugs.

### Stage 4: Convolution, naive
- **Build:** 2D blur using global memory only.
- **Measure:** time vs kernel size (3, 5, 7, 11).
- **Built:** `include/blur.h` (Gaussian weights, OpenCV sigma, 1D weights so 2D weight is `w[j]*w[i]`; shared per-pixel math, edges clamped), `src/blur_cpu.cpp` (scalar reference plus a double-precision reference for checking), `src/blur_cuda.cu` (one thread per pixel, 16x16 blocks, every tap a global load, weights as kernel argument), `apps/bench_blur.cu`, `apps/check_blur.cu` (in `ctest`). Logs: `results/logs/stage4_check_blur.txt`, `results/logs/stage4_blur_naive.txt`, data in `results/data/blur_naive.csv`. CPU runs are capped at 3 (scalar k=11 at 24 MP takes 3.7 s).
- **Result (24 MP photo, 16x16 blocks):** kernel 1.46 / 3.24 / 5.98 / 13.2 ms for k = 3 / 5 / 7 / 11, 252x to 282x faster than scalar CPU (367 / 871 / 1669 / 3737 ms). End-to-end speedup grows with k, 23x to 136x, because transfers are a fixed 14 ms. The kernel reaches only 11 to 99 GB/s of useful traffic (3 to 27% of peak) and 1.0 to 1.5 TFLOP/s (about 8 to 12% of 12.7 peak, counting 7 flops per tap). By the minimum-traffic arithmetic intensity (10 to 141 flop/byte) it should be compute-bound, but it is limited by neither DRAM nor FP32 throughput. Hypothesis: load issue and L1 throughput (3 byte-loads per tap), which Stage 5 tiling should relieve. Confirm with Nsight Compute before claiming it.
- **Block size sweep (24 MP photo, `results/logs/stage4_block_sweep.txt`):** kernel ms for k = 3 / 5 / 7 / 11 is 1.73 / 3.48 / 6.10 / 13.88 at 8x8, 1.36 / 3.11 / 5.98 / 13.30 at 16x16, 1.92 / 3.89 / 6.57 / 15.56 at 32x32, 1.31 / 3.23 / 5.59 / 13.15 at 32x4. 32x32 is slowest at every k (10 to 17% behind the best). 8x8 is worst only at k=3 (27% behind). 16x16 and 32x4 are tied within run-to-run noise (the 16x16 k=3 time moved 7% between two runs). Block shape changes kernel time by at most about 1.4x, so it does not explain the low bandwidth and FLOP rates, which fits the load-issue hypothesis. The sweep used `-c 1`, so its CPU rows in `blur_naive.csv` are single runs and the 16x16 rows repeat the main benchmark. Use the `-c 3` rows for CPU time.

### Stage 5: Convolution, tiled
- **Build:** shared-memory version with halo loading.
- **Measure:** speedup over naive, effect of block size, achieved occupancy.
- **Built:** `blur_tiled_kernel<T>` in `src/blur_cuda.cu`: the block copies its tile plus a (k-1)-wide border-clamped halo into dynamic shared memory, then every thread blurs from shared memory. Two variants: byte tile (`tiled`) and float tile (`tiledf`, converts once at load, 4x the shared memory). Same taps in the same order as the naive kernel, so both are bit-identical to it (`check_blur`, 227 checks, tolerance 0 against naive and 1 against the CPU). `bench_blur` runs all three kernels and reports theoretical occupancy from the CUDA occupancy API. Logs: `results/logs/stage5_tiled_vs_naive.txt`, `results/logs/stage5_block_sweep.txt`, data in `results/data/blur_conv.csv`.
- **Result (24 MP, 16x16):** tiled over naive is only 1.02x / 1.23x / 1.25x / 1.27x for k = 3 / 5 / 7 / 11 (kernel 1.34 / 2.53 / 4.52 / 10.5 ms). Tiling cuts global loads per pixel from 3k^2 to about 3x1.3 (k=3) or 3x2.9 (k=11) bytes, so load count alone was not the limit, and the Stage 4 hypothesis that it was is only weakly supported (hypothesis 4 holds, but the gap is small). The float tile is no faster than the byte tile (within 1%), which rules out the per-tap byte-to-float conversion as the limit. The kernel still reaches only 4% of DRAM bandwidth and about 1.9 TFLOP/s (15% of the 12.7 peak) at k=11, so what limits it is not yet known.
- **Block size (tiled, 24 MP, kernel ms for k=3 / k=11):** 8x8 1.65 / 11.20, 16x16 1.34 / 10.48, 32x32 2.21 / 11.42, 32x4 1.38 / 10.84. Theoretical occupancy is 100% for 16x16 and 32x4 and 67% for 8x8 (16 blocks/SM limit, 1024 of 1536 threads) and 32x32 (one 1024-thread block per SM), which matches those two being slowest. At k=3 the 32x32 tiled kernel is slower than naive (x0.87), because the halo load and sync are not repaid at small k.
- **Profile (Nsight Compute, 24 MP photo, 16x16, one launch each, `--set full`):** reports in `results/profiler/stage5_{naive,tiled}_k{3,11}.ncu-rep` (66 MB, git-ignored, regenerate with `ncu --set full -s 1 -c 1 -o ... build/bench_blur -r 2 -c 1 -k K -v VARIANT images/photo_24mp.png`), full details in `results/logs/stage5_ncu_details.txt`. Counters were blocked by `ERR_NVGPUCTRPERM` (`RmProfilingAdminOnly=1`) until `NVreg_RestrictProfilingToAdminUsers=0` was set in `/etc/modprobe.d/nvidia-profiling.conf` and the machine rebooted.

  | | naive k=3 | naive k=11 | tiled k=3 | tiled k=11 |
  |---|---|---|---|---|
  | kernel time (ms, under ncu) | 1.45 | 14.2 | 1.43 | 11.2 |
  | LSU pipe utilization (% of peak) | 64 | 76 | 82 | **99.7** |
  | L1/TEX throughput (% of peak) | 64 | 76 | 82 | **99.7** |
  | FMA pipe active (%) | 33 | 31 | 32 | 23 |
  | global load instrs (M) | 20.3 | 272 | 3.1 | 6.2 |
  | shared load instrs (M) | 0 | 0 | 20.3 | 272 |
  | DRAM read (MB) | 72 | 72 | 72 | 72 |
  | achieved occupancy (% of peak warps) | 77 | 96 | 94 | 98 |

  **Finding: the limiter is the load/store instruction pipe (LSU/L1), not DRAM and not FP32.** DRAM traffic is the 72 MB minimum (24 MP x 3 bytes) for every variant, and the FMA pipe is at most 33% busy. Tiling only moves each per-tap load from a global load to a shared load, one for one (272.25 M at k=11 in both kernels), so the LSU instruction count does not drop and the speedup is small. The tiled k=11 kernel saturates the LSU pipe (99.7%), and its top stall is the MIO queue (7.7 of 18.3 cycles between issues, 42%). This confirms the Stage 4 hypothesis with a correction: the limit is load issue, and tiling does not relieve it, because it removes cache and DRAM traffic, not load instructions. A fix has to cut load instructions per output pixel: vector loads (load 4 bytes or a `uchar4` per instruction), several outputs per thread with register reuse, or the separable blur in Stage 6 (2k taps instead of k^2). Occupancy is 94 to 98% achieved, so it does not explain the gap. The naive k=3 value (77%) is a short-kernel tail effect.
- **Caveat:** one profiled launch per configuration, under ncu (clocks locked, so times differ slightly from the benchmark). The report quotes the text output in `results/logs/stage5_ncu_details.txt`, no GUI screenshots.

### Stage 6: Separable and large-radius blur
- **Build:** separable Gaussian (horizontal then vertical pass) and a radius sweep (3, 7, 15, 31, 63).
- **Measure:** time vs radius for naive 2D, tiled 2D, and separable. Work grows with radius² for 2D and with radius for separable.
- **Purpose:** produces a smooth curve of increasing arithmetic intensity for the roofline chart.
- **Built:** `blur_h_pixel` and `blur_v_pixel` in `include/blur.h` (shared by CPU and GPU), `blur_sep_cpu` (scalar, caller-owned scratch buffer), `blur_sep_h_kernel` and `blur_sep_v_kernel` in `src/blur_cuda.cu` (one thread per pixel, global loads, 16x16 blocks). The intermediate between the passes is float (12 bytes per pixel), so the result matches 2D to float rounding and passes the same 1-level tolerance. Rounding it to 8 bits would add up to one more level of error. Radius r is k = 2r + 1, so the sweep 3, 7, 15, 31, 63 is `-k 7,15,31,63,127`. `check_blur` now covers the separable kernels and k up to 127 (455 checks, all pass, `results/logs/stage6_check_blur.txt`). `bench_blur` has a `sep` variant and `-m` (largest k that gets a scalar 2D CPU run, default 31). Log: `results/logs/stage6_radius_sweep.txt`, data in `results/data/blur_conv.csv`.
- **Result (24 MP photo, 16x16, kernel ms):**

  | radius (k) | naive 2D | tiled 2D | separable | sep over naive | sep GB/s (% of 360) | scalar CPU 2D / separable (ms) |
  |---|---|---|---|---|---|---|
  | 3 (7) | 5.72 | 4.56 | 2.43 | 2.35x | 296 (82%) | 1603 / 433 |
  | 7 (15) | 24.4 | 19.2 | 3.58 | 6.8x | 201 (56%) | 6943 / 855 |
  | 15 (31) | 101.8 | 80.0 | 6.78 | 15.0x | 106 (30%) | 28991 / 1700 |
  | 31 (63) | 419.3 | 329.3 | 13.3 | 31.5x | 54 (15%) | not run / 3521 |
  | 63 (127) | 1705.6 | skipped | 26.7 | 63.9x | 27 (7%) | not run / 7449 |

  - **Work scaling:** naive time grows by 4.1x to 4.3x for each doubling of k (k² work), separable by 1.5x to 2.0x (about linear in k). Separable beats naive by 63.9x at k=127, which equals the tap-count ratio k²/2k = 63.5. At k=7 the ratio of work is only 3.5 and the measured speedup 2.35x, because the float intermediate makes the small-k separable kernel memory-bound: it moves 30 bytes per pixel (3 + 12 + 12 + 3) and reaches 82% of the 360 GB/s peak (89% of the measured 331 GB/s).
  - **Tiled 2D:** a constant 1.26x to 1.27x over naive at every k it fits (k=7 to 63). The Stage 5 finding holds across the sweep: tiling swaps global loads for shared loads one for one. The byte tile is skipped at k=127 and the float tile at k=63 and above because the tile exceeds 48 KB.
  - **Compute rate:** every kernel stays at 1.4 to 2.0 TFLOP/s (11 to 16% of the 12.7 TFLOP/s FP32 peak) at large k, separable included (1.2 to 1.4). Separable is far above the ridge point by arithmetic intensity (12k/30 = 2.8 flop/byte at k=7 up to 51 at k=127, against a ridge of about 38) yet does not approach the FP32 peak. Each tap is 3 loads for 3 multiply-adds, so load issue is the likely limit here too. This was not profiled for the separable kernels, so it stays a hypothesis. GFLOP/s counts 12k flops per pixel for separable and 7k² for 2D, so the two are not the same flop count.
  - **CPU:** scalar 2D is capped at k=31 (29 s per run at 24 MP, 3 runs). Above that the 2D GPU kernels have no CPU speedup figure and the GPU outputs are checked against the CPU separable result. Like for like (separable on both), the GPU kernel is 178x to 279x faster than scalar CPU. The scalar CPU separable is 17x faster than scalar CPU 2D at k=31 (1.7 s vs 29 s). End to end, separable at k=127 takes 40 ms against 1730 ms for naive 2D, with 14 ms of that fixed PCIe transfer.
- **Caveats:** one image (24 MP photo), one block shape (16x16), medians of 20 GPU runs and 3 CPU runs. No Nsight Compute profile of the separable kernels yet, and no separable sweep across block sizes or image sizes.

### Stages 7 to 10 (deferred)
Not in the submission scope. Kept here for the personal project and for the report's future-work section.

### Stage 7: Histogram
- **Build:** global-atomics version, then shared-memory privatized version.
- **Measure:** time of each, on a flat-color image (maximum contention) vs a noisy image.

### Stage 8: Auto-contrast
- **Build:** pipeline of histogram, then contrast stretch.
- **Measure:** time breakdown per pipeline stage.

### Stage 9: Bilateral filter
- **Build:** edge-preserving filter with per-pixel exponentials over a neighborhood.
- **Measure:** time vs radius, GPU vs CPU. Expected to be strongly compute-bound.

### Stage 10: Local highlights/shadows
- **Build:** blurred-mask method that reuses the blur and color kernels so adjustments follow local regions.
- **Measure:** pipeline time breakdown, and a visual before/after comparison against the per-pixel version from Stage 1.

### Stage 11: Memory optimization (in scope)
- **Build:** pinned host memory, then chunked processing with CUDA streams.
- **Measure:** transfer time pageable vs pinned, percentage saved by copy/compute overlap.
- **Why it is in scope:** the color kernel is transfer-dominated (14 ms of PCIe against about 0.5 ms of kernel at 24 MP), and the measured PCIe cap (13 GB/s of 31.5 theoretical) is the main end-to-end limit. This stage shows how much of that can be hidden.
- **Apply to:** color (transfer-bound) and separable blur (kernel 13 to 27 ms at large k, so overlap helps there). Expect overlap to save at most the smaller of transfer and compute time per chunk.
- **Checks:** output identical to the unchunked version (Stage 3 check), chunk count swept (for example 1, 2, 4, 8, 16), and the transfer-time numbers compared against the measured 13.2 / 11.8 GB/s ceilings.
- **Done when:** pageable vs pinned table, a streams-vs-chunks table, and a timeline from Nsight Systems (text summary is enough) are logged.
- **Built:** `apps/bench_streams.cu` (variants: pageable 1 chunk, pinned 1 chunk, pinned N chunks with one stream per chunk, pageable 4 chunks as a control), `blur_sep_band_launch` in `src/blur_cuda.cu` (blur of a row band from its input rows plus an r-row halo, `blur_v_pixel_band` in `include/blur.h`), a stream argument on `color_cuda_launch`, and `experiments/pcie_duplex.cu`. `check_blur` gained 240 band checks (bands must join to the whole-image result with tolerance 0, including kernels taller than the image). Every variant's output is compared with the pageable 1-chunk output at tolerance 0. Logs: `results/logs/stage11_streams_{sizes,radius}.txt`, `stage11_check_blur.txt`, `exp_pcie_duplex.txt`, data in `results/data/streams.csv`. Run pinned to the 6 P-cores (`-t cores`), 20 runs.
- **Result (24 MP, end to end ms):**

  | | pageable 1 chunk | pinned 1 chunk | pinned 2 / 4 / 8 / 16 chunks | best vs pageable |
  |---|---|---|---|---|
  | color | 15.0 | 11.9 | 10.6 / 10.0 / 9.8 / 9.6 | 1.56x |
  | blur k=15 | 17.9 | 14.8 | 11.1 / 10.4 / 10.1 / 10.3 | 1.77x |
  | blur k=63 | 27.0 | 24.0 | 18.5 / 15.8 / 14.9 / 14.9 | 1.82x |
  | blur k=127 | 41.4 | 38.4 | 33.3 / 31.4 / 31.7 / 34.0 | 1.32x |

  - **Pinned memory:** H2D 13.3 GB/s and D2H 12.1 GB/s against 10.3 and 9.8 pageable (the Stage 2 ceilings of 13.2 and 11.8). That alone saves 1.4 to 3.1 ms at 24 MP.
  - **Streams need pinned memory:** pageable with 4 chunks is no faster than pageable with 1 chunk (0.92x to 1.0x), and slower at 1 MP.
  - **Overlap is limited by the link, not by the code.** `pcie_duplex` shows H2D alone 13.2 GB/s, D2H alone 11.8, both at once only 15.3 GB/s aggregate (full duplex would be about 25). Color at 24 MP moves 144 MB, so the link needs 9.4 ms at 15.3 GB/s, and the pipelined run takes 9.6 ms: it is at the measured aggregate limit, 98%. The same platform limit that caps each direction at about 13 GB/s caps the sum. Streams hide the kernel and the gaps, but cannot hide the transfer itself.
  - **Chunk count:** returns flatten at 4 to 8 chunks. Blur k=127 gets worse at 16 chunks (34.0 vs 31.4 ms at 4) because each band re-reads r=63 halo rows on each side, so the extra input grows with the chunk count and a band carries 126 halo rows.
  - **Costs not in the table:** `cudaHostAlloc` of in plus out takes 36 ms at 24 MP (72 ms at 48 MP), so buffers must be pinned once and reused. If the image arrives in pageable memory, staging it into pinned buffers costs a 6.9 ms host memcpy (24 MP, input plus output), which cancels the gain (pinned 1 chunk 11.9 + 6.9 is worse than pageable 15.0). The timings assume the image is decoded straight into pinned memory.
  - **Against the CPU ladder:** best GPU color is 9.6 ms against 4.0 ms for AVX2 + OpenMP on the CPU, so the CPU still wins color end to end even after pinning and streams (hypothesis 2 stands). Blur k=15 is 10.1 ms against 30 ms CPU, and k=127 is 31.4 against 126 ms, so the GPU wins blur by 3x to 4x end to end.
- **Caveat:** one image per size, one thread mode (6 P-cores), medians of 20, one stream per chunk with no stream-count sweep. The Nsight Systems timeline is not captured.

### Stage 12: Parallelism ladder (in scope)
Run the same algorithm and the same image on every rung, on one memory-bound kernel (color) and one compute-bound kernel (separable blur, Stage 6; bilateral is deferred).

**Build order:** native rungs first (1, 2, 6, 7), since they answer the central question. Then WASM rungs 3 and 4 in Node only if emsdk setup is quick. Rung 5 (WASM threads) is deferred.

**Report both end-to-end and compute-only time per rung.** The GPU rung includes the PCIe transfer in end-to-end, the CPU rungs do not, and hypothesis 2 depends on that difference. Pin threads to P-cores for the main rung 6 runs and report 6 and 16 threads separately (Section 2).

**Reuse:** the shared pixel math in `include/` (`blur.h`, color math), the Stage 2 harness, and the Stage 3 compare, with the fixed-point rung getting its own tolerance.

| Rung | Parallelism type | Approach |
|---|---|---|
| 1 | None (baseline) | Scalar C++, vectorization disabled |
| 2 | Compiler SIMD | Auto-vectorized C++ with AVX2 |
| 3 | None, sandboxed | WASM scalar |
| 4 | 128-bit SIMD | WASM SIMD128 |
| 5 | SIMD + multi-core | WASM SIMD with threads |
| 6 | 256-bit SIMD + multi-core | Native AVX2 with OpenMP |
| 7 | Massive SIMT | CUDA on RTX 3060 |

Notes:
- Benchmark WASM in Node.js for cleaner timings.
- Compare a float implementation against a fixed-point integer one on SIMD, and use the Stage 3 check to report the precision cost.
- Rung 6 is now required (it is the strongest CPU rung, and the fair opponent for the GPU). Rungs 3 and 4 are optional, rung 5 is deferred.

### Optional: Median filter
A branch-heavy counterexample showing warp divergence, and a case where the GPU advantage shrinks.

---

## 5. Experiments Matrix

| Variable | Values | Applies to |
|---|---|---|
| Image size | 1, 6, 24, 48 MP | All stages |
| Thread block size | 8x8, 16x16, 32x32, 32x4 | All CUDA kernels |
| Blur radius / kernel size | 3, 7, 15, 31, 63 | Stages 4 to 6, 9 |
| Image content | flat color, natural photo, noise | Histogram, median |
| Memory type | pageable, pinned | Stage 11 |
| Platform | ladder rungs 1 to 7 | Stage 12 |

---

## 6. Metrics and Methodology

**Metrics collected for every kernel**
- Kernel time and end-to-end time (separately)
- Time split: H2D copy, compute, D2H copy
- Effective bandwidth = (bytes read + bytes written) / kernel time, reported as % of peak
- Throughput in megapixels per second
- Speedup vs scalar CPU baseline
- Arithmetic intensity (estimated FLOPs per byte) for the roofline chart

**Rules for valid measurements**
1. Always do a warm-up run before timing (first CUDA call includes context initialization).
2. Report the median of many runs, plus min and max.
3. Time GPU work with CUDA events, CPU work with a high-resolution CPU clock, and never time GPU code with a CPU clock without synchronizing.
4. Use identical algorithm, image, and precision across platforms.
5. Log every result to CSV so every number in the report is reproducible.
6. Record hardware and software versions once, at the top of the results.
7. Keep the machine quiet during runs (close other apps, note GPU and CPU clock behavior).
8. Run the correctness check after every optimization.

**Profiling:** use Nsight Compute on one or two key kernels (naive vs tiled convolution is a good pair) and capture memory throughput and occupancy.

---

## 7. Planned Charts and Tables

1. Time breakdown (H2D, kernel, D2H) for the color kernel, showing transfer dominance
2. Speedup vs image size
3. Effective bandwidth as % of peak, per kernel
4. Block size comparison
5. Naive vs tiled vs separable blur, time vs radius
6. (Deferred) Histogram contention, flat vs noisy image
7. (Deferred) Pipeline stage breakdown for auto-contrast and local highlights/shadows
8. Pinned vs pageable, and streams overlap gain
9. **Parallelism ladder:** one chart for the memory-bound kernel, one for the compute-bound kernel, side by side
10. **Roofline chart** with CPU and GPU ceilings and each kernel plotted by arithmetic intensity
11. Before/after image comparisons (original, color adjustment output, blur output). The local-adjustment version is deferred with Stage 10.

---

## 8. Expected Findings (hypotheses to confirm or refute)

1. The color kernel is **memory-bound**. The GPU kernel itself is very fast, but PCIe transfer dominates end-to-end time.
2. Multi-core CPU SIMD can be **competitive end-to-end** on simple per-pixel work, because it skips the PCIe transfer.
3. Blur is **compute-bound** by arithmetic intensity, so the GPU wins by a large margin even including transfers. Stage 4 to 6 data already complicates this: the kernels stay at 11 to 16% of FP32 peak and are limited by load issue, so report "compute-bound by intensity, load-issue-bound in practice".
4. Tiled convolution beats naive, and separable beats both, with the gap growing with radius. Measured: tiled is only 1.02x to 1.27x over naive, separable up to 63.9x.
5. (Deferred with Stage 7) Histogram atomics suffer heavy contention on flat images, and privatization largely fixes it.
6. (Optional, only if WASM rungs are built) WASM SIMD sits between scalar and native AVX2, limited by its 128-bit width and sandbox overhead.
7. (Deferred with the median filter) The median filter narrows the GPU advantage because of divergence.
8. (Stage 11) Pinned memory raises PCIe throughput toward the measured 13 GB/s cap, and chunked streams hide at most the smaller of transfer and compute time.

Any hypothesis that fails to match the data is a stronger discussion point than one that matches, so report it honestly.

---

## 9. Project Layout (planned)

```
project/
  src/          reusable library code: CUDA kernels, CPU reference, SIMD versions
  apps/         thin drivers, one per stage (load image, call library, time, write CSV)
  wasm/         WASM sources and benchmark runner
  include/      image I/O and timing headers
  images/       test images at several sizes
  results/      CSV data, charts, output images, logs, profiler reports
  scripts/      run wrappers (capped, bench, logrun), chart generation, aggregation
  report/       final report and log excerpts
  Makefile
  README.md
```

---

## 10. Mapping to the Grading Rubric

| Rubric item (20% each) | Where it comes from in this plan |
|---|---|
| **Source code with explanation** | Stage 1 kernel plus one short explanation per later stage (what each kernel does, how threads map to pixels) |
| **Tools used with explanation** | Hardware table (Section 2), plus compiler, profiler, and runtime tools, each with a one-line role |
| **Development steps with real screenshots** | Per-stage record: input, command, output. Evidence is the command and its saved output (`results/logs/`), set in the PDF as monospaced terminal-style blocks with the command as caption, plus charts generated from the CSVs and before/after output images. No GUI screenshots. |
| **Installation and execution** | Tool installation steps, build steps, run steps, and the hardware/software required |
| **Experiment summary** | Resources used (hardware and software), findings from Section 8, charts from Section 7 |

**Report must include:** name and student ID, source code with brief explanation, tools (hardware and software) with brief roles, numbered development steps (input, command, output from real runs, as saved logs), installation and execution instructions, and a conclusion listing the hardware and software resources required.

---

## 11. Report Outline

1. Cover: name, student ID, title
2. Introduction: goal, hybrid computing context, central question
3. Hardware and software environment
4. Source code overview and explanation
5. Development and compile steps (per stage: input, command, output)
6. Installation and execution
7. Experiments and results (charts, tables, analysis)
8. Parallelism ladder comparison
9. Discussion: bottlenecks, when each platform wins, limitations
10. Conclusion: resources required, key findings
11. Appendix: full timing tables, extra images

---

## 12. Suggested Timeline

Adjust to the actual deadline. Work backward from the exam date.

| Phase | Work | Target |
|---|---|---|
| A | Stages 1 to 3 (baseline, harness, correctness) | Done |
| B | Stages 4 to 6 (convolution, separable blur) | Done |
| C | Stage 12 ladder, native rungs 1, 2, 6, 7 on color and blur | Next |
| D | Stage 11 (pinned memory, streams) | After C |
| E | Optional: WASM rungs 3 and 4 in Node | Only if time allows, cap the setup effort |
| F | Chart scripts, roofline, Installation and Execution section, draft sections that do not depend on C to E (environment, source overview, steps) | In parallel with C to E |
| G | Finish report, proofread, export PDF, submit | Reserve at least 5 days before the deadline |

Writing the report is not an afterthought. Draft each section right after its stage while the logs and numbers are fresh.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scope too large | Tiered scope, stop at any finished stage |
| Inconsistent measurements | Shared harness, warm-up, median of runs, quiet machine |
| Unfair platform comparison | Same algorithm, image, and precision on every rung |
| WASM threading setup trouble | Use Node.js, and drop to WASM SIMD without threads if needed |
| Results contradict expectations | Report honestly and explain, since this counts as analysis |
| Missing evidence at the end | Every run saves its command and output under `results/logs/`, so the report only has to excerpt them |
| Floating-point differences between platforms | Define a tolerance in Stage 3 and report max error |

---

## 14. Master Checklist

- [x] Hardware and software versions recorded
- [x] Test images prepared (1, 6, 24, 48 MP; flat, photo, noise)
- [x] Stage 1: baseline color
- [x] Stage 2: measurement harness
- [x] Stage 3: correctness check
- [x] Stage 4: convolution naive
- [x] Stage 5: convolution tiled
- [x] Stage 6: separable and radius sweep
- [x] Stage 11: pinned memory and streams
- [x] Stage 12: parallelism ladder (color), native rungs 1, 2, 6, 7
- [x] Stage 12: parallelism ladder (separable blur), native rungs 1, 2, 6, 7
- [ ] Stage 12 (optional): WASM rungs 3 and 4 in Node
- [ ] Chart scripts (`scripts/`) for the in-scope charts
- [ ] Roofline chart
- [ ] Installation and execution section (commands, memory cap, profiling permission fix)
- [ ] Optional: `ncu` run on the separable kernel (settles the load-issue hypothesis)
- [ ] Report written, name and student ID included
- Deferred: Stage 7 histogram, Stage 8 auto-contrast, Stage 9 bilateral, Stage 10 local highlights/shadows, median filter, WASM threads
- [ ] Exported as PDF or Word
- [ ] Submitted via Microsoft Teams before the final exam
