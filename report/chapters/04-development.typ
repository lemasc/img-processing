#import "../template.typ": *

= ขั้นตอนการพัฒนา

บทนี้อธิบายขั้นตอนการพัฒนาตามลำดับที่ทำจริง เริ่มจากการคอมไพล์ Source Code แล้วตามด้วยขั้นการพัฒนาแต่ละขั้น แต่ละขั้นระบุสิ่งที่ทำ Input คำสั่งที่ใช้ และ Output ที่ได้ ภาพประกอบทุกภาพในบทนี้เป็นผลลัพธ์จากเทอร์มินัลที่บันทึกไว้ระหว่างทดลองจริงใน `results/logs/` แต่ละไฟล์บันทึกคำสั่ง เวลา และ Commit ของ Git ที่ใช้ จึงย้อนกลับไปทำซ้ำได้

หลักการที่ใช้ตลอดการพัฒนาคือ จะไม่เริ่มขั้นถัดไปจนกว่าขั้นปัจจุบันจะผ่านการตรวจความถูกต้อง วัดผล และบันทึก Log เรียบร้อยแล้ว

#fig-placeholder(
  [ภาพรวมลำดับขั้นการพัฒนา],
  [แผนภาพลำดับขั้น: เตรียมเครื่องและ Smoke Test → การปรับสีพื้นฐาน → ชุดวัดเวลา → ตรวจความถูกต้อง → เบลอ Naive → เบลอ Tiled + Profiler → เบลอ Separable → Pinned Memory + Streams → บันไดการประมวลผลแบบขนาน],
  label: <fig-dev-flow>,
)

== การคอมไพล์ Source Code

โปรเจกต์ใช้ CMake อธิบายการ Build และใช้ Ninja สั่งคอมไพล์ กระบวนการแปลงจาก Source Code เป็นโปรแกรมที่ใช้งานได้มีขั้นตอนดังนี้ (@fig-compile-flow)

#fig-placeholder(
  [ขั้นตอนการแปลง Source Code เป็นโปรแกรมที่ทำงานได้],
  [แผนภาพ: `CMakeLists.txt` → cmake → `build/build.ninja` ; ไฟล์ `.cu` → nvcc (ส่วน Device → PTX/SASS sm_86, ส่วน Host → g++) → `.o` ; ไฟล์ `.cpp` → g++ (Flag ตามระดับ) → `.o` ; `.o` → ar → ไลบรารี `.a` ; ไลบรารี + `apps/*.cu` → nvcc/ลิงก์ → โปรแกรมใน `build/`],
  label: <fig-compile-flow>,
)

=== ขั้นที่ 1: กำหนดค่าการ Build ด้วย CMake

CMake อ่านไฟล์ `CMakeLists.txt` ซึ่งกำหนดว่าโปรเจกต์ใช้ภาษา C++ และ CUDA มาตรฐาน C++17 สถาปัตยกรรม GPU เป้าหมายคือ `sm_86` (RTX 3060) และค้นหาไลบรารี OpenMP จากนั้นสร้างไฟล์ Build ของ Ninja

#step-io(
  input: [`CMakeLists.txt` และ Source Code ทั้งหมด],
  command: [`cmake -S . -B build -G Ninja`],
  output: [ไดเรกทอรี `build/` ที่มีไฟล์ `build.ninja`],
)

#term-log([ผลการกำหนดค่าการ Build ด้วย CMake], "results/logs/stage1_cmake_configure.txt", label: <term-cmake>)

=== ขั้นที่ 2: คอมไพล์และลิงก์ด้วย Ninja

Ninja สั่งคอมไพล์ทุกไฟล์ตามที่ CMake กำหนด ไฟล์ `.cu` ถูกแปลด้วย `nvcc` ซึ่งแยกโค้ดเป็นสองส่วน คือส่วน Device (Kernel) ที่แปลเป็นรหัสเครื่องของ GPU และส่วน Host ที่ส่งต่อให้ g++ แปล ไฟล์ `.cpp` ถูกแปลด้วย g++ โดยแต่ละไลบรารีใช้ Flag ต่างกันตามระดับของบันได (@tab-flags) แล้วรวมเป็นไลบรารีแบบ Static และลิงก์เป็นโปรแกรม การ Build จำกัดไว้ 4 งานพร้อมกัน (`-j4`) เพราะการคอมไพล์ CUDA ใช้หน่วยความจำมาก

#step-io(
  input: [ไดเรกทอรี `build/` จากขั้นที่ 1],
  command: [`cmake --build build -j4`],
  output: [ไลบรารี `.a` และโปรแกรม เช่น `build/bench_color`, `build/bench_blur`, `build/check_blur`, `build/bench_ladder_color`, `build/bench_streams`],
)

#data-table([Flag การคอมไพล์ของแต่ละไลบรารี], ("ไลบรารี", "ระดับ", "Flag", "เหตุผล"), (
  ([`color_cpu`, `blur_cpu`], [Scalar], [`-O2 -fno-tree-vectorize -fno-tree-slp-vectorize -ffinite-math-only -fno-signed-zeros`], [ปิดการ Vectorize อัตโนมัติเพื่อเป็นเส้นฐาน ส่วน `-ffinite-math-only` ทำให้ Clamp ใช้ `minss`/`maxss` แทน Branch]),
  ([`color_cpu_simd`, `blur_cpu_simd`], [Autovec และ AVX2 (1 Thread และ OpenMP)], [`-O3 -mavx2 -mfma -ffinite-math-only -fno-signed-zeros` และ OpenMP], [Flag Clamp เหมือนระดับ Scalar จึงต่างกันเฉพาะการ Vectorize และ Thread]),
  ([`color_cuda`, `blur_cuda`], [CUDA], [`nvcc -arch=sm_86` (Release) และเลือก `--fmad=false` ได้ด้วย `-DCOLOR_CUDA_FMAD=OFF` (เฉพาะ `color_cuda`)], [`nvcc` รวมการคูณและบวกเป็น FMA ตามค่าเริ่มต้น ใช้ปิดเพื่ออธิบายผลต่าง 1 ระดับ]),
), align: left, widths: (1fr, 0.9fr, 2.3fr, 1.5fr), label: <tab-flags>)

#term-log([ผลการคอมไพล์และลิงก์ด้วย Ninja], "results/logs/stage1_ninja_build.txt", ranges: ((1, 8), (33, 37)), label: <term-ninja>)

== ขั้นการพัฒนาโปรแกรม

=== ขั้นที่ 3: ตรวจเครื่องมือและวัดขีดจำกัดของเครื่อง

ก่อนเขียน Kernel จริง ได้เขียนโปรแกรม `apps/smoke.cu` เพื่อยืนยันว่า Toolchain, GPU และชุดจับเวลาทำงานได้ โดยรัน Kernel คัดลอกข้อมูลอย่างง่ายและวัดแบนด์วิดท์หน่วยความจำของ GPU จากนั้นวัดขีดจำกัดของ DRAM และ PCIe ด้วยโปรแกรมใน `experiments/`

#step-io(
  input: [โปรแกรม `build/smoke`, `experiments/host_bandwidth.cpp`, `experiments/pcie_bandwidth.cu`],
  command: [`scripts/bench.sh -t single build/smoke` และ `scripts/exp_peaks.sh`],
  output: [แบนด์วิดท์ GPU 331 GB/s, DRAM 51.4 GB/s, PCIe 13.2 / 11.8 GB/s (@tab-ceilings)],
)

#term-log([ผลการทดสอบ Toolchain และ GPU], "results/logs/smoke.txt", label: <term-smoke>)
#term-log([ผลการวัดขีดจำกัด DRAM และ PCIe], "results/logs/exp_peaks.txt", label: <term-peaks>)

=== ขั้นที่ 4: การปรับสีพื้นฐานบน CPU และ GPU

เขียนสูตรปรับสีใน `include/color.h` (@lst-adjust) เวอร์ชัน CPU แบบ Scalar ใน `src/color_cpu.cpp` และ Kernel ของ CUDA ใน `src/color_cuda.cu` (@lst-color-kernel) แล้วรันกับภาพถ่ายจริงขนาด 24 MP เพื่อดูผลลัพธ์และเวลาเบื้องต้น

#step-io(
  input: [ภาพถ่าย `images/photos/24mp_complex_colors.jpg` (6000×4000)],
  command: [`scripts/bench.sh -t single build/stage1_color <input> <output> 10`],
  output: [ภาพที่ปรับสีแล้ว `results/output/stage1_24mp_complex_colors.png` และเวลาของ CPU และ GPU],
)

#term-log([ผลการปรับสีพื้นฐาน], "results/logs/stage1_24mp_complex_colors.txt", label: <term-stage1>)

#labelled(figure(image("/results/charts/stage1_before_after.png", width: 100%), kind: image, caption: [ภาพก่อนและหลังการปรับสี]), <fig-before-after-color>)

=== ขั้นที่ 5: ชุดวัดเวลา

เขียนชุดวัดเวลาใน `include/timing.h` และ `include/bench.h` (@lst-timing) ซึ่งทำ Warm-up ล้าง Cache ก่อนทุกครั้ง รันซ้ำ 20 ครั้ง และแยกเวลาเป็น H2D, Kernel และ D2H พร้อมคำนวณแบนด์วิดท์ที่ได้เทียบกับค่าสูงสุด แล้ววัดการปรับสีที่ภาพทุกขนาดและทุกชนิดเนื้อหา

ระหว่างขั้นนี้พบว่า CPU แบบ Scalar ใช้เวลาต่างกันตามเนื้อหาภาพ (169, 195 และ 290 ms สำหรับภาพสีเรียบ ภาพถ่าย และภาพสัญญาณรบกวน) เพราะ GCC แปลการจำกัดค่า (Clamp) เป็นคำสั่งเปรียบเทียบและกระโดด ซึ่งทายผิดบ่อยกับข้อมูลสุ่ม เมื่อเพิ่ม Flag `-ffinite-math-only -fno-signed-zeros` GCC ใช้คำสั่ง `minss`/`maxss` แทน ทำให้ใช้เวลา 161 ms เท่ากันทุกภาพ จึงใช้ Flag นี้เป็นเส้นฐาน

#step-io(
  input: [ภาพ photo ขนาด 1, 6, 24, 48 MP, ภาพถ่าย 32 MP, ภาพ flat และ noise 24 MP],
  command: [`scripts/bench.sh -t single build/bench_color -r 20 -o results/output <images...>` และ `scripts/exp_clamp.sh`],
  output: [`results/data/color.csv` และภาพผลลัพธ์ใน `results/output/`],
)

#term-log([ผลการวัดเวลาการปรับสีทุกขนาดภาพ], "results/logs/stage2_color_sweep.txt", label: <term-stage2>)
#term-log([ผลการทดลองคำสั่ง Clamp แบบมีและไม่มี Branch], "results/logs/exp_clamp.txt", label: <term-clamp>)

=== ขั้นที่ 6: ตรวจความถูกต้อง

เขียนชุดเปรียบเทียบใน `include/compare.h` และ `src/compare.cpp` พร้อมโปรแกรมทดสอบตัวเอง แล้วเขียน `apps/check_color.cu` เพื่อเทียบผล GPU กับ CPU ทุกสีที่เป็นไปได้ (16.7 ล้านสี) และกับภาพจริง ผลคือต่างกันไม่เกิน 1 ระดับใน 224 สี (0.0013%) สาเหตุคือ nvcc รวมการคูณและบวกเป็นคำสั่ง FMA ซึ่งปัดเศษครั้งเดียว ขณะที่ CPU ปัดเศษสองครั้ง เมื่อคอมไพล์ด้วย `--fmad=false` ผล GPU ตรงกับ CPU ทุกบิต

#step-io(
  input: [ภาพทุกสี (สร้างในโปรแกรม), ภาพถ่าย 24 และ 32 MP, ภาพ flat และ noise],
  command: [`build/check_color <images...>`, `scripts/exp_fma.sh` และ `ctest --test-dir build`],
  output: [รายงาน PASS/FAIL พร้อมผลต่างสูงสุด ผลต่างเฉลี่ย และจำนวนพิกเซลที่ต่างกัน],
)

#term-log([ผลการตรวจความถูกต้องของการปรับสี], "results/logs/stage3_check_color.txt", ranges: ((1, 4), (13, 19)), label: <term-check-color>)
#term-log([ผลการทดลองปิด FMA บน GPU], "results/logs/exp_fma.txt", label: <term-fma>)

=== ขั้นที่ 7: การเบลอแบบ Naive

เขียนสูตรเบลอใน `include/blur.h` เวอร์ชัน CPU ใน `src/blur_cpu.cpp` (รวมเวอร์ชันอ้างอิงแบบ Double Precision) และ Kernel แบบ Naive ใน `src/blur_cuda.cu` (@lst-naive) จากนั้นตรวจความถูกต้อง วัดเวลาที่ขนาด Kernel 3, 5, 7 และ 11 และทดลองขนาด Block 4 แบบ

#step-io(
  input: [ภาพ photo 1, 6, 24 MP, ภาพถ่ายกลางคืน 24 MP, ภาพ noise 6 MP],
  command: [`build/check_blur <images...>`, `scripts/bench.sh -t single -m 8G build/bench_blur -o results/output <images...>` และการวน `-b 8x8,16x16,32x32,32x4`],
  output: [`results/data/blur_naive.csv` และภาพเบลอใน `results/output/`],
)

#term-log([ผลการตรวจความถูกต้องของการเบลอ], "results/logs/stage4_check_blur.txt", ranges: ((1, 8), (59, 66)), label: <term-check-blur>)
#term-log([ผลการวัดเวลาการเบลอแบบ Naive], "results/logs/stage4_blur_naive.txt", label: <term-naive>)

=== ขั้นที่ 8: การเบลอแบบ Tiled และการวิเคราะห์ด้วย Profiler

เพิ่ม Kernel แบบ Tiled (@lst-tiled) ทั้งแบบ Tile เป็นไบต์และเป็นทศนิยม ตรวจว่าผลตรงกับ Naive ทุกบิต แล้ววัดเวลาเทียบกัน ผลที่ได้คือเร็วขึ้นเพียง 1.02 ถึง 1.27 เท่า ซึ่งน้อยกว่าที่คาด จึงใช้ Nsight Compute เก็บค่า Hardware Counter ของ Kernel ทั้งสองแบบ เพื่อหาคอขวดที่แท้จริง

การใช้ Nsight Compute ครั้งแรกล้มเหลวด้วยข้อผิดพลาด `ERR_NVGPUCTRPERM` เพราะ Driver อนุญาตให้อ่าน Counter เฉพาะผู้ดูแลระบบ จึงต้องตั้งค่า `NVreg_RestrictProfilingToAdminUsers=0` แล้ว Reboot (รายละเอียดในบทที่ 5)

#step-io(
  input: [ภาพ photo 24 MP, Kernel ขนาด 3 และ 11],
  command: [`scripts/bench.sh -t single -m 8G build/bench_blur -c 1 <images...>` และ `ncu --set full -s 1 -c 1 -o <report> build/bench_blur -r 2 -c 1 -k K -v VARIANT images/photo_24mp.png`],
  output: [`results/data/blur_conv.csv`, รายงาน `.ncu-rep` และสรุปเป็นข้อความใน `results/logs/stage5_ncu_details.txt`],
)

#term-log([ผลการวัดเวลาแบบ Tiled เทียบกับ Naive], "results/logs/stage5_tiled_vs_naive.txt", ranges: ((1, 4), (29, 40)), label: <term-tiled>)
#term-log([ผลการวิเคราะห์ด้วย Nsight Compute], "results/logs/stage5_ncu_details.txt", cmd: "ncu --set full -s 1 -c 1 -o results/profiler/stage5_tiled_k11 build/bench_blur -r 2 -c 1 -k 11 -v tiled images/photo_24mp.png", ranges: ((1, 16), (207, 221), (418, 432), (616, 630)), label: <term-ncu>)

=== ขั้นที่ 9: การเบลอแบบ Separable และรัศมีขนาดใหญ่

เพิ่มสูตรแนวนอนและแนวตั้งใน `include/blur.h` เวอร์ชัน CPU แบบ Separable และ Kernel สองรอบ (@lst-sep) เพิ่มการตรวจความถูกต้องจนถึง $k = 127$ (454 กรณี) แล้ววัดเวลาที่รัศมี 3, 7, 15, 31 และ 63 เทียบกับแบบ Naive และ Tiled

#step-io(
  input: [ภาพ photo 24 MP, $k$ = 7, 15, 31, 63, 127],
  command: [`build/check_blur <images...>` และ `build/bench_blur -k 7,15,31,63,127 -m 31 -c 3 -r 20 images/photo_24mp.png`],
  output: [`results/data/blur_conv.csv` (เพิ่มแถวของ Separable)],
)

#term-log([ผลการตรวจความถูกต้องของการเบลอแบบ Separable], "results/logs/stage6_check_blur.txt", cmd: "build/check_blur", ranges: ((1, 5), (470, 476)), label: <term-check-sep>)
#term-log([ผลการวัดเวลาตามรัศมี], "results/logs/stage6_radius_sweep.txt", label: <term-radius>)

=== ขั้นที่ 10: หน่วยความจำแบบ Pinned และ CUDA Streams

เขียนโปรแกรม `apps/bench_streams.cu` (@lst-streams) ที่เปรียบเทียบ 4 แบบ คือ Pageable 1 Chunk, Pinned 1 Chunk, Pinned $N$ Chunk ($N$ = 2, 4, 8, 16) และ Pageable 4 Chunk เป็นกลุ่มควบคุม ผลลัพธ์ทุกแบบต้องตรงกับแบบ Pageable 1 Chunk ทุกบิต และเขียน `experiments/pcie_duplex.cu` เพื่อวัดว่า PCIe คัดลอกสองทิศทางพร้อมกันได้เร็วเท่าใด

#step-io(
  input: [ภาพ photo 1, 6, 24, 48 MP, การปรับสี และการเบลอ $k$ = 15, 63, 127],
  command: [`scripts/bench.sh -t cores build/bench_streams -r 20 -k 15 <images...>` และ `scripts/bench.sh -t single build/exp/pcie_duplex`],
  output: [`results/data/streams.csv`],
)

#term-log([ผลการทดลอง Pinned Memory และ Streams ตามขนาดภาพ], "results/logs/stage11_streams_sizes.txt", ranges: ((1, 4), (55, 79)), label: <term-streams>)
#term-log([ผลการวัด PCIe สองทิศทางพร้อมกัน], "results/logs/exp_pcie_duplex.txt", label: <term-duplex>)

=== ขั้นที่ 11: บันไดการประมวลผลแบบขนาน

เขียนเวอร์ชัน CPU ที่ Vectorize อัตโนมัติ และเวอร์ชัน AVX2 Intrinsics ทั้งแบบ Thread เดียวและแบบ OpenMP ใน `src/color_cpu_simd.cpp` (@lst-block16) และ `src/blur_cpu_simd.cpp` (@lst-vstrip) พร้อมโปรแกรม `bench_ladder_color` และ `bench_ladder_blur` ที่รันทุกระดับกับภาพเดียวกัน ตรวจผลกับเวอร์ชัน Scalar ก่อนจับเวลา และรันซ้ำใน 4 โหมด Thread (`single`, `cores`, `multi`, `all`)

#step-io(
  input: [ภาพ photo 1, 6, 24, 48 MP, ภาพ flat และ noise 24 MP],
  command: [`scripts/bench.sh -t <mode> build/bench_ladder_color -r 20 <images...>` และ `scripts/bench.sh -t <mode> build/bench_ladder_blur -r 20 -c 5 -v autovec_omp,avx2_omp,cuda -k 7,15,31,63,127 images/photo_24mp.png`],
  output: [`results/data/ladder.csv`],
)

#term-log([ผลบันไดการปรับสี 12 Thread], "results/logs/stage12_color_ladder_multi.txt", ranges: ((1, 4), (21, 28)), label: <term-ladder-color>)
#term-log([ผลบันไดการเบลอ 12 Thread ตามรัศมี], "results/logs/stage12_blur_ladder_multi_radius.txt", label: <term-ladder-blur>)
