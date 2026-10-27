// Build the PDF from the repository root (the report reads files from results/ and images/):
//   typst compile --root . report/main.typ
// The output is report/main.pdf. Use `typst watch --root . report/main.typ` to rebuild on save.

#import "template.typ": *

#show: report

#cover(
  title-th: [การวิเคราะห์ประสิทธิภาพการประมวลผลภาพ \ บนแพลตฟอร์มการประมวลผลแบบขนาน],
  title-en: [Performance Analysis of Image Color Processing \ Across Parallel Computing Platforms],
  authors: ([#todo[ชื่อ-นามสกุล ภาษาไทย] รหัสนักศึกษา 67070173],),
  advisor: [#todo[ชื่ออาจารย์ผู้สอน]],
  course: [Hybrid Computing Technology (HCT) #todo[รหัสวิชา]],
  term: [#todo[ภาคเรียนที่ … ปีการศึกษา …]],
  faculty: [#todo[คณะ / สาขาวิชา]],
  institute: [สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง],
)

#show: front-matter

#outline(title: [สารบัญ], indent: auto)
#pagebreak()
#outline(title: [สารบัญตาราง], target: figure.where(kind: table))
#pagebreak()
#outline(title: [สารบัญภาพ], target: figure.where(kind: image))

#show: main-matter

#include "chapters/01-intro.typ"
#include "chapters/02-tools.typ"
#include "chapters/03-source.typ"
#include "chapters/04-development.typ"
#include "chapters/05-install.typ"
#include "chapters/06-results.typ"
#include "chapters/07-conclusion.typ"
#include "chapters/08-appendix.typ"
