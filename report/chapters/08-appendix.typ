#import "../template.typ": *

// Appendix headings are unnumbered; the file contents are read from the repository at build time.
#set heading(numbering: none)

// Scripts have long unbroken lines (flag lists, paths): use a smaller size and allow breaks after
// separators so nothing runs past the box.
#show raw.where(block: true): set text(size: 8pt)
#show raw.where(block: true): it => {
  show regex("[,/=:\\-]"): m => m + sym.zws
  it
}

= ภาคผนวก ก Source Code ของสคริปต์และไฟล์ Build

ภาคผนวกนี้รวบรวม Source Code ของสคริปต์ที่ใช้ควบคุมการทดลอง บันทึกผล สร้างภาพทดสอบ รวมถึงไฟล์ Build ของโปรเจกต์ ซึ่งอ้างถึงในบทก่อนหน้าแต่ไม่ได้แสดงเต็ม โค้ดทุกส่วนดึงจากไฟล์จริงในโปรเจกต์โดยตรง ส่วน Source Code ของ Kernel และโปรแกรมหลักอยู่ในไดเรกทอรี `src/`, `apps/`, `include/` และ `experiments/` ของโปรเจกต์ โดยส่วนสำคัญยกมาแสดงแล้วในบทที่ 3

#let appendix-file(title, desc, path, lang) = [
  == #title

  #desc

  #par(first-line-indent: 0pt)[ไฟล์: #raw(path)]

  #code-file("/" + path, lang: lang)
]

#appendix-file(
  [`bench.sh` สคริปต์รันการวัดผลภายใต้เงื่อนไขควบคุม],
  [ตรึง Thread ไว้กับแกนที่กำหนดด้วย `taskset` และกำหนดค่า OpenMP ให้สอดคล้องกัน เพื่อให้การวัดบันไดการประมวลผลแบบขนานทำซ้ำได้],
  "scripts/bench.sh", "bash",
)

#appendix-file(
  [`run-capped.sh` สคริปต์จำกัดทรัพยากรของงานที่รัน],
  [รันคำสั่งใน cgroup Scope ที่จำกัดหน่วยความจำและโควตา CPU เพื่อไม่ให้งานที่ผิดพลาดทำให้เครื่องล่ม],
  "scripts/run-capped.sh", "bash",
)

#appendix-file(
  [`logrun.sh` สคริปต์บันทึกผลลัพธ์จากเทอร์มินัล],
  [รันคำสั่งแล้วบันทึกคำสั่ง เวลา และผลลัพธ์ลง `results/logs/` เพื่อใช้อ้างอิงในรายงาน],
  "scripts/logrun.sh", "bash",
)

#appendix-file(
  [`env_report.sh` สคริปต์บันทึกสภาพแวดล้อมของเครื่อง],
  [บันทึกรุ่นของฮาร์ดแวร์และซอฟต์แวร์ที่ใช้ในการทดลองลง `results/env.txt`],
  "scripts/env_report.sh", "bash",
)

#appendix-file(
  [`exp_peaks.sh` สคริปต์วัดขีดจำกัดของเครื่อง],
  [วัดแบนด์วิดท์ DRAM และ PCIe ที่ทำได้จริงเพื่อใช้เป็นเพดานในแบบจำลอง Roofline],
  "scripts/exp_peaks.sh", "bash",
)

#appendix-file(
  [`exp_clamp.sh` สคริปต์ทดลอง Branch ของ Clamp],
  [คอมไพล์และรันการทดลองเปรียบเทียบวิธี Clamp หลายแบบบนภาพ 24 MP],
  "scripts/exp_clamp.sh", "bash",
)

#appendix-file(
  [`exp_fma.sh` สคริปต์ทดลองผลของ FMA],
  [เปรียบเทียบผลลัพธ์ที่ความคลาดเคลื่อน 0 เมื่อเปิดและปิดการรวมคำสั่ง FMA บน GPU],
  "scripts/exp_fma.sh", "bash",
)

#appendix-file(
  [`make_test_images.py` สคริปต์สร้างภาพทดสอบ],
  [สร้างภาพเรียบ ภาพสัญญาณรบกวน และภาพที่ปรับขนาดสำหรับการทดสอบ],
  "scripts/make_test_images.py", "python",
)

#appendix-file(
  [`CMakeLists.txt` ไฟล์ Build ของโปรเจกต์],
  [กำหนดไลบรารี โปรแกรม และ Flag ของตัวแปลภาษาสำหรับทั้ง C++ และ CUDA],
  "CMakeLists.txt", "cmake",
)
