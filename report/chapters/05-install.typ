#import "../template.typ": *

= การติดตั้งและสั่งให้ทำงาน

บทนี้อธิบายการติดตั้งเครื่องมือ การ Build โปรแกรม และการสั่งให้โปรแกรมทำงาน บนเครื่องที่ใช้ระบบปฏิบัติการ Ubuntu 24.04 และ GPU ของ NVIDIA ทุกขั้นตอนในบทนี้ได้ทดลองติดตั้งและรันจริงบนเครื่องที่ใช้ทดลอง (@tab-hw)

== ความต้องการของระบบ

#table-placeholder(
  [ความต้องการขั้นต่ำของระบบ],
  [GPU NVIDIA ที่รองรับ CUDA (โปรเจกต์ตั้งค่าไว้สำหรับ sm_86 ต้องแก้ `CMAKE_CUDA_ARCHITECTURES` หากใช้ GPU รุ่นอื่น) ; CPU x86-64 ที่รองรับ AVX2 และ FMA ; RAM อย่างน้อย 8 GB (ภาพ 48 MP ต้องการ 10 GB) ; พื้นที่ดิสก์สำหรับภาพทดสอบ ; Ubuntu 24.04 ; NVIDIA Driver ; CUDA Toolkit 13.x ; GCC 13 ; CMake ≥ 3.24 ; Ninja ; uv],
  columns: ("รายการ", "ขั้นต่ำ", "ที่ใช้ทดลอง"),
  label: <tab-requirements>,
)

== การติดตั้งเครื่องมือ

=== NVIDIA Driver และ CUDA Toolkit

ติดตั้งจาก Repository ของ NVIDIA ซึ่งได้ทั้ง Driver, ตัวแปลภาษา `nvcc`, CUDA Runtime, Nsight Compute และ Nsight Systems จากนั้นเพิ่มตำแหน่งของ CUDA ใน `PATH`

#cmd[
  ```
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update
  sudo apt install cuda-toolkit-13-2 cuda-drivers
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  ```
]

#todo[ยืนยันชื่อแพ็กเกจ Driver ที่ติดตั้งจริง (เช่น `cuda-drivers` หรือ `nvidia-driver-595-open`)] หลังติดตั้งแล้ว ให้ Reboot และตรวจสอบด้วยคำสั่งต่อไปนี้ ซึ่งควรแสดงชื่อ GPU รุ่นของ Driver และรุ่นของ `nvcc`

#cmd[
  ```
  nvidia-smi
  nvcc --version
  ```
]

#term-placeholder([ผลการตรวจสอบ Driver และ CUDA Toolkit], "nvidia-smi && nvcc --version", "results/env.txt", label: <term-nvsmi>)

=== ตัวแปลภาษา C++ และเครื่องมือ Build

#cmd[
  ```
  sudo apt install build-essential cmake ninja-build git
  ```
]

แพ็กเกจ `build-essential` มี GCC 13 และ `libgomp` ซึ่งเป็นไลบรารี OpenMP ของ GCC อยู่แล้ว CMake ของ Ubuntu 24.04 คือรุ่น 3.28 ซึ่งรองรับ CUDA ตามที่โปรเจกต์ต้องการ (3.24 ขึ้นไป)

=== การอนุญาตให้ Nsight Compute อ่าน Hardware Counter

โดยค่าเริ่มต้น Driver ของ NVIDIA อนุญาตให้อ่าน Hardware Counter ของ GPU เฉพาะผู้ดูแลระบบ หากไม่ตั้งค่า `ncu` จะแสดงข้อผิดพลาด `ERR_NVGPUCTRPERM` ให้สร้างไฟล์ตั้งค่าของ Kernel Module แล้ว Reboot

#cmd[
  ```
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
    sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u
  sudo reboot
  ```
]

ขั้นตอนนี้จำเป็นเฉพาะเมื่อต้องการใช้ Nsight Compute ไม่จำเป็นสำหรับการ Build และรันโปรแกรม

=== Python และ uv

สคริปต์สร้างภาพทดสอบและกราฟเขียนด้วย Python และจัดการด้วย uv ซึ่งสร้าง Virtual Environment ของโปรเจกต์และติดตั้ง NumPy, pandas, Matplotlib และ Pillow ตาม `pyproject.toml` และ `uv.lock`

#cmd[
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
]

== การ Build โปรแกรม

=== ดาวน์โหลด Source Code และติดตั้งแพ็กเกจ Python

#cmd[
  ```
  git clone <repository-url> img-processing
  cd img-processing
  uv sync
  scripts/env_report.sh          # บันทึกรุ่นของฮาร์ดแวร์และซอฟต์แวร์ลง results/env.txt
  ```
]

=== คอมไพล์

ขั้นตอนนี้คือขั้นที่ 1 และ 2 ในบทที่ 4 คำสั่งแรกกำหนดค่าการ Build คำสั่งที่สองคอมไพล์ทุกโปรแกรม

#cmd[
  ```
  cmake -S . -B build -G Ninja
  cmake --build build -j4
  ```
]

โปรแกรมที่ได้อยู่ในไดเรกทอรี `build/` ดังตาราง @tab-programs โปรแกรมเหล่านี้ทำงานได้ทันทีโดยไม่ต้องติดตั้งเพิ่ม เพราะไลบรารีของโปรเจกต์ลิงก์แบบ Static ไว้แล้ว ต้องการเพียง Driver ของ NVIDIA และ CUDA Runtime ที่มากับ Toolkit

#table-placeholder(
  [โปรแกรมที่ได้จากการ Build และหน้าที่],
  [smoke, bench_color, check_color, test_compare, bench_blur, check_blur, bench_ladder_color, bench_ladder_blur, bench_streams],
  columns: ("โปรแกรม", "หน้าที่", "ตัวเลือกสำคัญ"),
  source: "CMakeLists.txt, apps/",
  label: <tab-programs>,
)

== การเตรียมภาพทดสอบ

สคริปต์ `make_test_images.py` สร้างภาพสีเรียบและภาพสัญญาณรบกวนทุกขนาด และถ้าระบุภาพถ่ายต้นฉบับ จะย่อหรือขยายภาพนั้นเป็นภาพ photo ทุกขนาด ภาพถ่ายจริงให้วางไว้ใน `images/photos/`

#cmd[
  ```
  uv run scripts/make_test_images.py images/photos/24mp_complex_colors.jpg
  ```
]

== การสั่งให้โปรแกรมทำงาน

=== ตรวจความถูกต้อง

ก่อนวัดเวลา ให้รันชุดตรวจความถูกต้องทั้งหมดผ่าน CTest ซึ่งรันการทดสอบตัวเองของชุดเปรียบเทียบ การเทียบผลการปรับสีและการเบลอบน GPU กับ CPU ทุกกรณี

#cmd[
  ```
  ctest --test-dir build --output-on-failure
  ```
]

#term-placeholder([ผลการรันชุดตรวจความถูกต้องด้วย CTest], "ctest --test-dir build --output-on-failure", "ยังไม่มี (ต้องรันและบันทึกใหม่)", label: <term-ctest>)

=== รันการวัดประสิทธิภาพ

ทุกการวัดรันผ่าน `scripts/bench.sh` ซึ่งตรึง Thread ไว้กับแกนที่กำหนด ตัวอย่างการรันแต่ละการทดลอง

#cmd[
  ```
  # การปรับสี: CPU Scalar เทียบ GPU
  scripts/bench.sh -t single build/bench_color -r 20 images/photo_24mp.png

  # การเบลอ: Naive, Tiled, Separable ตามรัศมี
  build/bench_blur -k 7,15,31,63,127 -c 3 -r 20 images/photo_24mp.png

  # Pinned Memory และ CUDA Streams บน 6 P-core
  scripts/bench.sh -t cores build/bench_streams -r 20 images/photo_24mp.png

  # บันไดการประมวลผลแบบขนาน (โหมด single, cores, multi, all)
  scripts/bench.sh -t multi build/bench_ladder_color -r 20 images/photo_24mp.png
  ```
]

=== วิเคราะห์ด้วย Nsight Compute

#cmd[
  ```
  ncu --set full -s 1 -c 1 -o results/profiler/tiled_k11 \
      build/bench_blur -r 2 -c 1 -k 11 -v tiled images/photo_24mp.png
  ncu --import results/profiler/tiled_k11.ncu-rep --page details
  ```
]

ตัวเลือก `-s 1 -c 1` ข้าม Kernel แรก (Warm-up) และเก็บข้อมูลเพียง 1 Kernel เพราะการเก็บแบบ `--set full` ต้องรัน Kernel ซ้ำหลายสิบรอบ

== ข้อควรระวังในการรัน

- ภาพ 48 MP แบบทศนิยมใช้หน่วยความจำประมาณ 576 MB ต่อสำเนา ควรเผื่อหน่วยความจำให้พอ
- หน่วยความจำแบบ Pinned ไม่สามารถย้ายออกได้ จึงจองไว้เพียงสองบัฟเฟอร์และใช้ซ้ำ
- ตั้ง CPU Governor เป็น `performance` และให้เครื่องไม่มีงานอื่นระหว่างวัด `bench.sh` จะเตือนหากเงื่อนไขไม่ครบ
- บัส PCIe ทำงานที่ Gen 1 เมื่อว่าง และเพิ่มความเร็วเมื่อมีโหลด การวัดจึงต้อง Warm-up ก่อนเสมอ
