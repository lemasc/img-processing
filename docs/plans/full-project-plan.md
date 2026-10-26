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
| **Minimum (must finish)** | 1 to 5, plus the ladder (Stage 12) on color and blur only | Complete, gradable report |
| **Target** | Add 6, 9, 10 | Compute-bound analysis, roofline chart |
| **Stretch** | Add 7, 8, 11 and the median filter | Full coverage, extra depth |

Rule: do not start a stage until the previous one is measured, screenshotted, and logged.

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

### Stage 5: Convolution, tiled
- **Build:** shared-memory version with halo loading.
- **Measure:** speedup over naive, effect of block size, achieved occupancy.

### Stage 6: Separable and large-radius blur
- **Build:** separable Gaussian (horizontal then vertical pass) and a radius sweep (3, 7, 15, 31, 63).
- **Measure:** time vs radius for naive 2D, tiled 2D, and separable. Work grows with radius² for 2D and with radius for separable.
- **Purpose:** produces a smooth curve of increasing arithmetic intensity for the roofline chart.

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

### Stage 11: Memory optimization
- **Build:** pinned host memory, then chunked processing with CUDA streams.
- **Measure:** transfer time pageable vs pinned, percentage saved by copy/compute overlap.

### Stage 12: Parallelism ladder
Run the same algorithm and the same image on every rung, on at least one memory-bound kernel (color) and one compute-bound kernel (blur or bilateral).

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
- Rung 6 may be skipped if time is short, but it shows what the 128-bit WASM limit costs compared with native AVX2.

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
6. Histogram contention, flat vs noisy image
7. Pipeline stage breakdown for auto-contrast and local highlights/shadows
8. Pinned vs pageable, and streams overlap gain
9. **Parallelism ladder:** one chart for the memory-bound kernel, one for the compute-bound kernel, side by side
10. **Roofline chart** with CPU and GPU ceilings and each kernel plotted by arithmetic intensity
11. Before/after image comparisons (original, global adjustment, local adjustment)

---

## 8. Expected Findings (hypotheses to confirm or refute)

1. The color kernel is **memory-bound**. The GPU kernel itself is very fast, but PCIe transfer dominates end-to-end time.
2. Multi-core CPU SIMD can be **competitive end-to-end** on simple per-pixel work, because it skips the PCIe transfer.
3. Blur and bilateral are **compute-bound**, so the GPU wins by a large margin even including transfers.
4. Tiled convolution beats naive, and separable beats both, with the gap growing with radius.
5. Histogram atomics suffer heavy contention on flat images, and privatization largely fixes it.
6. WASM SIMD sits between scalar and native AVX2, limited by its 128-bit width and sandbox overhead.
7. The median filter narrows the GPU advantage because of divergence.

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
  results/      CSV data, charts, output images, profiler screenshots
  scripts/      run wrappers (capped, bench, logrun), chart generation, aggregation
  report/       final report and screenshots
  Makefile
  README.md
```

---

## 10. Mapping to the Grading Rubric

| Rubric item (20% each) | Where it comes from in this plan |
|---|---|
| **Source code with explanation** | Stage 1 kernel plus one short explanation per later stage (what each kernel does, how threads map to pixels) |
| **Tools used with explanation** | Hardware table (Section 2), plus compiler, profiler, and runtime tools, each with a one-line role |
| **Development steps with real screenshots** | Per-stage record: input, command, output, screenshot. Capture as you go. |
| **Installation and execution** | Tool installation steps, build steps, run steps, and the hardware/software required |
| **Experiment summary** | Resources used (hardware and software), findings from Section 8, charts from Section 7 |

**Report must include:** name and student ID, source code with brief explanation, tools (hardware and software) with brief roles, numbered development steps (input, output, screenshots from real runs), installation and execution instructions, and a conclusion listing the hardware and software resources required.

---

## 11. Report Outline

1. Cover: name, student ID, title
2. Introduction: goal, hybrid computing context, central question
3. Hardware and software environment
4. Source code overview and explanation
5. Development and compile steps (per stage: input, output, screenshots)
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
| A | Stages 1 to 3 (baseline, harness, correctness) | Week 1 |
| B | Stages 4 to 6 (convolution, separable blur) | Week 2 |
| C | Stage 12 ladder on color and blur (minimum tier complete here) | Week 3 |
| D | Stages 9 and 10 (bilateral, local highlights/shadows) | Week 4 |
| E | Stages 7, 8, 11 and median filter (stretch) | Week 5, only if time allows |
| F | Charts, writing, proofreading, submission | Final week, reserve at least 5 days |

Writing the report is not an afterthought. Draft each section right after its stage while the screenshots and numbers are fresh.

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scope too large | Tiered scope, stop at any finished stage |
| Inconsistent measurements | Shared harness, warm-up, median of runs, quiet machine |
| Unfair platform comparison | Same algorithm, image, and precision on every rung |
| WASM threading setup trouble | Use Node.js, and drop to WASM SIMD without threads if needed |
| Results contradict expectations | Report honestly and explain, since this counts as analysis |
| Missing screenshots at the end | Capture at every stage, not afterward |
| Floating-point differences between platforms | Define a tolerance in Stage 3 and report max error |

---

## 14. Master Checklist

- [x] Hardware and software versions recorded
- [x] Test images prepared (1, 6, 24, 48 MP; flat, photo, noise)
- [x] Stage 1: baseline color
- [x] Stage 2: measurement harness
- [x] Stage 3: correctness check
- [ ] Stage 4: convolution naive
- [ ] Stage 5: convolution tiled
- [ ] Stage 6: separable and radius sweep
- [ ] Stage 7: histogram
- [ ] Stage 8: auto-contrast
- [ ] Stage 9: bilateral filter
- [ ] Stage 10: local highlights/shadows
- [ ] Stage 11: pinned memory and streams
- [ ] Stage 12: parallelism ladder (color)
- [ ] Stage 12: parallelism ladder (blur or bilateral)
- [ ] Roofline chart
- [ ] Nsight Compute screenshots
- [ ] Report written, name and student ID included
- [ ] Exported as PDF or Word
- [ ] Submitted via Microsoft Teams before the final exam
