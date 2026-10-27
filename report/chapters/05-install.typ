#import "../template.typ": *

= การติดตั้งและสั่งให้ทำงาน

บทนี้อธิบายการติดตั้งเครื่องมือ การ Build โปรแกรม และการสั่งให้โปรแกรมทำงาน บนเครื่องที่ใช้ระบบปฏิบัติการ Ubuntu 24.04 และ GPU ของ NVIDIA ทุกขั้นตอนในบทนี้ได้ทดลองติดตั้งและรันจริงบนเครื่องที่ใช้ทดลอง (@tab-hw)

== ความต้องการของระบบ

#data-table([ความต้องการขั้นต่ำของระบบ], ("รายการ", "ขั้นต่ำ", "ที่ใช้ทดลอง"), (
  ([GPU], [NVIDIA ที่รองรับ CUDA (ตั้งค่าไว้สำหรับ sm_86 หากใช้รุ่นอื่นต้องแก้ `CMAKE_CUDA_ARCHITECTURES`)], [RTX 3060 12 GB (sm_86)]),
  ([CPU], [x86-64 ที่รองรับ AVX2 และ FMA], [Core i5-13400F]),
  ([RAM], [16 GB (ภาพ 48 MP ใช้ประมาณ 10 GB)], [16 GB]),
  ([พื้นที่ดิสก์], [พอสำหรับภาพทดสอบ (ประมาณ 400 MB)], [–]),
  ([ระบบปฏิบัติการ], [Ubuntu 24.04], [Ubuntu 24.04 (Linux 6.8)]),
  ([NVIDIA Driver], [รุ่นที่รองรับ CUDA 13.x], [595.71.05]),
  ([CUDA Toolkit], [13.x], [13.2]),
  ([GCC], [13], [13.3.0]),
  ([CMake], [3.24 ขึ้นไป], [3.28.3]),
  ([Ninja], [ทุกรุ่น], [1.11.1]),
  ([uv], [ใช้เมื่อสร้างภาพทดสอบและกราฟ], [0.11.3]),
), align: left, widths: (1fr, 2fr, 1.3fr), label: <tab-requirements>)

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

หลังติดตั้งแล้ว ให้ Reboot และตรวจสอบด้วยคำสั่งต่อไปนี้ ซึ่งควรแสดงชื่อ GPU รุ่นของ Driver และรุ่นของ `nvcc`

#cmd[
  ```
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  nvcc --version
  ```
]

#term-log([ผลการตรวจสอบ Driver และ CUDA Toolkit], "results/logs/env_nvidia.txt", label: <term-nvsmi>)

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

#data-table([โปรแกรมที่ได้จากการ Build และหน้าที่], ("โปรแกรม", "หน้าที่", "ตัวเลือกสำคัญ"), (
  ([`smoke`], [ตรวจ Toolchain และ GPU วัดแบนด์วิดท์ของ Copy Kernel และ PCIe], [–]),
  ([`bench_color`], [วัดเวลาการปรับสีของ CPU เทียบ GPU], [`-r` จำนวนรอบ, `-o` ไดเรกทอรีเก็บภาพผลลัพธ์]),
  ([`check_color`], [ตรวจผลการปรับสีของ GPU เทียบ CPU ทั้ง 16.7 ล้านสีและภาพจริง], [`--tol` ผลต่างที่ยอมรับได้]),
  ([`test_compare`], [ทดสอบชุดเปรียบเทียบผลลัพธ์เอง], [–]),
  ([`bench_blur`], [วัดเวลาการเบลอแบบ Naive, Tiled และ Separable], [`-r`, `-c` จำนวนรอบ GPU และ CPU, `-k` ขนาดเคอร์เนล, `-b` ขนาด Block, `-v` เวอร์ชัน, `-m` $k$ สูงสุดของ CPU 2D, `-o`]),
  ([`check_blur`], [ตรวจความถูกต้องของการเบลอทุกเวอร์ชัน], [รายชื่อภาพ (ไม่ระบุจะใช้ภาพสังเคราะห์)]),
  ([`bench_ladder_color`], [วัดบันไดการประมวลผลแบบขนานของการปรับสี], [`-r`]),
  ([`bench_ladder_blur`], [วัดบันไดการประมวลผลแบบขนานของการเบลอแบบ Separable], [`-r`, `-c`, `-k`, `-v` ระดับของบันได]),
  ([`bench_streams`], [วัด Pinned Memory และ CUDA Streams], [`-r`, `-k`, `-n` จำนวน Chunk, `-w` งาน (color, blur)]),
), align: left, widths: (3.9cm, 1fr, 1fr), label: <tab-programs>)

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

#term-log([ผลการรันชุดตรวจความถูกต้องด้วย CTest], "results/logs/ctest.txt", label: <term-ctest>)

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
