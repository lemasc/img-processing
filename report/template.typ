// Thai report template: TH SarabunPSK, Thai thesis margins, "บทที่ N" chapter headings,
// chapter-relative figure and table numbers, and placeholder blocks for content not yet filled in.
// Build from the repository root: typst compile --root . report/main.typ

#let font-body = "TH SarabunPSK"
#let font-mono = ("DejaVu Sans Mono", "TH SarabunPSK")
#let indent = 1.25cm

// Thai letters for front-matter page numbers (ก ข ค ...). ฃ and ฅ are obsolete and skipped.
#let thai-letters = "กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ".clusters()
#let thai-page(..n) = thai-letters.at(calc.rem(n.pos().first() - 1, thai-letters.len()))

// Chapter-relative number for figures and tables: 3.2 = second item of chapter 3.
#let chapter-num(n) = numbering("1.1", counter(heading).get().first(), n)

// ---------- placeholders ----------
// Every chart, table and terminal output starts as one of these and is replaced once confirmed.

#let placeholder-box(kind, body, height: auto) = block(
  width: 100%,
  height: height,
  inset: 10pt,
  radius: 3pt,
  fill: rgb("#fff7e0"),
  stroke: (paint: rgb("#d9a400"), dash: "dashed", thickness: 0.8pt),
  breakable: false,
)[
  #set par(first-line-indent: 0pt, justify: false)
  #set align(left)
  #text(weight: "bold", fill: rgb("#8a6400"))[\[#kind\]]
  #body
]

// Attach a label straight to the figure (not to a surrounding sequence) so @refs resolve.
#let labelled(f, label) = if label == none { f } else { [#f#label] }

// A chart or image. `source` names the data or file it will come from.
#let fig-placeholder(caption, desc, source: none, label: none) = labelled(figure(
  placeholder-box("ภาพ", height: 6cm)[
    #desc
    #if source != none [ \ แหล่งข้อมูล: #raw(source)]
  ],
  kind: image,
  caption: caption,
), label)

// A table. `columns` lists the intended column headings.
#let table-placeholder(caption, desc, columns: (), source: none, label: none) = labelled(figure(
  placeholder-box("ตาราง")[
    #desc
    #if columns.len() > 0 [ \ คอลัมน์: #columns.join(" | ")]
    #if source != none [ \ แหล่งข้อมูล: #raw(source)]
  ],
  kind: table,
  caption: caption,
), label)

// A terminal capture from a saved log. Counted as a figure (ภาพประกอบจากการทดลองจริง).
#let term-placeholder(caption, cmd, log, desc: none, label: none) = labelled(figure(
  placeholder-box("ผลลัพธ์จากเทอร์มินัล")[
    #text(font: font-mono, size: 9pt)[\$ #cmd]
    \ log: #raw(log)
    #if desc != none [ \ #desc]
  ],
  kind: image,
  caption: caption,
), label)

// Input / command / output summary of one development step.
#let step-io(input: [], command: [], output: []) = block(
  width: 100%,
  inset: (left: indent),
  grid(
    columns: (auto, 1fr),
    column-gutter: 0.8em,
    row-gutter: 0.5em,
    [*Input*], input,
    [*คำสั่ง*], command,
    [*Output*], output,
  ),
)

// Inline note for prose or values still to be confirmed.
#let todo(body) = text(fill: rgb("#b00020"))[\[#body\]]

// ---------- code ----------

// Excerpt of a source file, lines a..b inclusive (1-based), shown as a numbered listing.
#let code-file(path, lang: "cpp", from: 1, to: none) = {
  let lines = read(path).split("\n")
  let to = if to == none { lines.len() } else { to }
  raw(lines.slice(from - 1, to).join("\n"), lang: lang, block: true)
}

#let listing(caption, body, label: none) = labelled(figure(body, kind: raw, caption: caption), label)

// Command block for instructions the reader types (not a captured output).
#let cmd(body) = block(
  width: 100%,
  inset: 8pt,
  radius: 3pt,
  fill: luma(245),
  stroke: luma(200) + 0.5pt,
  text(font: font-mono, size: 9pt, body),
)

// ---------- cover ----------

#let cover(
  title-th: [],
  title-en: [],
  authors: (),
  advisor: [],
  course: [],
  faculty: [],
  institute: [],
  term: [],
) = page(numbering: none, header: none)[
  #set align(center)
  #set par(first-line-indent: 0pt, justify: false)
  #set text(weight: "bold", size: 20pt)
  #v(1fr)
  #text(size: 24pt, title-th) \
  #text(size: 20pt, title-en)
  #v(1fr)
  จัดทำโดย \
  #for a in authors [#a \ ]
  #v(1fr)
  เสนอ \
  #advisor
  #v(1fr)
  #text(size: 18pt)[
    รายงานนี้เป็นส่วนหนึ่งของรายวิชา #course \
    #term \
    #faculty \
    #institute
  ]
  #v(0.5fr)
]

// ---------- document ----------

#let page-number-header(fmt) = context {
  set text(size: 16pt)
  align(right, fmt(counter(page).get().first()))
}

#let report(body) = {
  set document(title: "HCT Assignment Report")
  set page(
    paper: "a4",
    margin: (top: 3.81cm, left: 3.81cm, right: 2.54cm, bottom: 2.54cm),
  )
  set text(font: font-body, size: 16pt, lang: "th")
  set par(first-line-indent: (amount: indent, all: true), justify: true, leading: 0.6em, spacing: 0.6em)
  show raw: set text(font: font-mono, size: 9pt)
  show raw.where(block: true): set par(justify: false)
  show raw.where(block: true): block.with(width: 100%, inset: 8pt, radius: 3pt, fill: luma(245), stroke: luma(200) + 0.5pt)

  set list(indent: indent, body-indent: 0.5em)
  set enum(indent: indent, body-indent: 0.5em)

  // Headings: level 1 is "บทที่ N" over the title, centered, new page. Lower levels left-aligned.
  set heading(numbering: "1.1")
  show heading: set text(weight: "bold")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    // Reset chapter-relative figure counters.
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: raw)).update(0)
    set align(center)
    set text(size: 20pt)
    block(below: 1.2em)[
      #if it.numbering != none [บทที่ #counter(heading).display("1") \ ]
      #it.body
    ]
  }
  show heading.where(level: 2): set text(size: 18pt)
  show heading.where(level: 2): set block(above: 1.2em, below: 0.8em)
  show heading.where(level: 3): set text(size: 16pt)
  show heading.where(level: 3): set block(above: 1em, below: 0.6em)

  // Figures: Thai supplements, chapter numbers, table captions on top.
  set figure(numbering: chapter-num)
  show figure.where(kind: image): set figure(supplement: [ภาพที่])
  show figure.where(kind: table): set figure(supplement: [ตารางที่])
  show figure.where(kind: raw): set figure(supplement: [โปรแกรมที่])
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: raw): set figure.caption(position: top)
  show figure.caption: set par(first-line-indent: 0pt)
  show figure.caption: it => [*#it.supplement #context it.counter.display(it.numbering)* #it.body]
  show figure: set block(breakable: true)

  // Table of contents: chapters show as "บทที่ N  title".
  show outline.entry.where(level: 1): it => {
    set text(weight: "bold")
    let el = it.element
    if el.func() == heading and el.numbering != none {
      let n = numbering("1", ..counter(heading).at(el.location()))
      link(el.location(), it.indented([บทที่ #n], it.inner()))
    } else { it }
  }

  body
}

// Front matter: Thai-letter page numbers in the top right.
#let front-matter(body) = {
  set page(numbering: thai-page, header: page-number-header(thai-page), footer: none)
  counter(page).update(1)
  body
}

// Body: Arabic page numbers in the top right, restarting at 1.
#let main-matter(body) = {
  set page(numbering: "1", header: page-number-header(n => str(n)), footer: none)
  counter(page).update(1)
  body
}

