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

// A generated chart or diagram from results/charts (made by scripts/make_charts.py).
#let chart-fig(path, caption, width: 80%, label: none) = labelled(figure(
  image(path, width: width),
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

// ---------- saved logs and data ----------
// Logs and CSVs are read straight from results/ (paths are relative to the repository root).

// Show a saved log as a terminal block. `ranges` picks line ranges (1-based, as numbered in the
// file) and marks each gap; `cmd` replaces the command line, or adds one to a log without it.
// The `# date host commit` and `# exit` bookkeeping lines are dropped.
#let term-log(caption, log, cmd: none, ranges: none, label: none) = {
  let all = read("/" + log).split("\n")
  if all.last() == "" { all = all.slice(0, -1) }
  let picked = ()
  if ranges == none {
    picked = all
  } else {
    for (i, r) in ranges.enumerate() {
      if i > 0 {
        picked.push("[ ... ละไว้ " + str(r.at(0) - ranges.at(i - 1).at(1) - 1) + " บรรทัด ... ]")
      }
      picked += all.slice(r.at(0) - 1, r.at(1))
    }
  }
  let is-meta(l) = l.starts-with("# ") and (l.contains("host=") or l.starts-with("# exit="))
  let body = picked.filter(l => not is-meta(l)).map(l => l.replace(regex("scripts/run-capped\\.sh( -m \\S+)? "), ""))
  if cmd != none {
    if body.len() > 0 and body.first().starts-with("$ ") { body = body.slice(1) }
    body = ("$ " + cmd,) + body
  }
  labelled(figure(raw(body.join("\n"), block: true), kind: image, caption: caption), label)
}

// A results table: header row, booktabs-style rules, numbers right-aligned unless `align` is given.
#let data-table(caption, columns, rows, align: none, widths: none, label: none) = {
  let n = columns.len()
  let al = if align == none { (left,) + (right,) * (n - 1) } else { align }
  labelled(figure({
    set text(size: 14pt)
    set par(first-line-indent: 0pt, justify: false, leading: 0.5em)
    table(
      columns: if widths == none { n } else { widths },
      align: al,
      inset: (x: 6pt, y: 5pt),
      stroke: none,
      table.hline(stroke: 0.8pt),
      table.header(..columns.map(strong)),
      table.hline(stroke: 0.5pt),
      ..rows.flatten(),
      table.hline(stroke: 0.8pt),
    )
  }, kind: table, caption: caption), label)
}

// Fixed-decimal number as a string: fx(15, d: 1) = "15.0".
#let fx(x, d: 1) = {
  if x == none { return "–" }
  if d == 0 { return str(int(calc.round(x))) }
  let parts = str(calc.round(x, digits: d)).split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  while frac.len() < d { frac += "0" }
  parts.at(0) + "." + frac
}

// Rows of a measurement CSV from results/data. Some `params` hold commas (cpu lists), so the first
// three and last seven fields are taken by position and the middle is the params.
#let measures(path) = read("/" + path).split("\n").slice(1).filter(l => l.trim() != "").map(l => {
  let f = l.split(",")
  let n = f.len()
  let kv = f.slice(3, n - 7).join(",").split(";").map(p => p.split("=")).filter(p => p.len() == 2)
  (
    kernel: f.at(0),
    platform: f.at(1),
    params: kv.map(p => (p.at(0), p.at(1))).to-dict(),
    median: float(f.at(n - 6)),
  )
})

// Median (ms) of the last row matching kernel, platform and the given params, or none.
#let pick(rows, kernel, platform, ..want) = {
  let m = rows.filter(r => (
    r.kernel == kernel and r.platform == platform
      and want.named().pairs().all(((k, v)) => r.params.at(k, default: none) == v)
  ))
  if m.len() == 0 { none } else { m.last().median }
}

// Counts from a check_* log: (cases, failed, largest max-difference). The RESULT line is not a case.
#let check-stats(log, only: none) = {
  let lines = read("/" + log).split("\n").filter(l => only == none or l.starts-with(only))
  let cases = lines.map(l => l.match(regex("^\\S.*?\\s+(PASS|FAIL)\\s+max (\\d+),"))).filter(m => m != none)
  (
    cases: cases.len(),
    failed: cases.filter(m => m.captures.at(0) == "FAIL").len(),
    max: calc.max(..cases.map(m => int(m.captures.at(1)))),
  )
}

// Kernel times of the Gaussian blur block-size sweep: ("naive"|"tiled", block, k) -> ms,
// and block -> theoretical occupancy (%). Naive comes from stage 4 (k = 3, 5, 7, 11), tiled and
// occupancy from stage 5 (k = 3, 11).
#let block-sweep() = {
  let ms = (:)
  let occ = (:)
  for path in ("results/logs/stage4_block_sweep.txt", "results/logs/stage5_block_sweep.txt") {
    let stage5 = path.contains("stage5")
    let blk = none
    for l in read("/" + path).split("\n") {
      let b = l.match(regex("block=(\\d+x\\d+)"))
      if b != none { blk = b.captures.at(0); continue }
      let m = l.match(regex("k=(\\d+)\\s+(?:(naive|tiled)\\s+)?\\|.*?kern\\s+([\\d.]+)"))
      if m == none { continue }
      let k = m.captures.at(0)
      let v = if m.captures.at(1) == none { "naive" } else { m.captures.at(1) }
      if not stage5 or v == "tiled" { ms.insert(v + "|" + blk + "|" + k, float(m.captures.at(2))) }
      if stage5 and v == "naive" { occ.insert(blk, int(l.match(regex("occ\\s+(\\d+)%")).captures.at(0))) }
    }
  }
  (ms: ms, occ: occ)
}

// The radius sweep of stage 6: k -> (variant -> (kernel ms, GB/s, % of peak)), plus the CPU times.
#let radius-sweep() = {
  let t = (:)
  for l in read("/results/logs/stage6_radius_sweep.txt").split("\n") {
    let m = l.match(regex("k=(\\d+)\\s+(naive|tiled|sep)\\s+\\|\\s+cpu2d\\s+([\\d.]+)\\s+cpusep\\s+([\\d.]+)\\s+\\|\\s+kern\\s+([\\d.]+).*?\\|\\s+([\\d.]+) GB/s \\(\\s*(\\d+)%\\)"))
    if m == none { continue }
    let c = m.captures
    let e = t.at(c.at(0), default: (:))
    e.insert(c.at(1), (ms: float(c.at(4)), gbs: float(c.at(5)), pct: int(c.at(6))))
    e.insert("cpu2d", float(c.at(2)))
    e.insert("cpusep", float(c.at(3)))
    t.insert(c.at(0), e)
  }
  t
}

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
  // Inline code has no break points inside file names; allow a break after these characters.
  show raw.where(block: false): it => {
    show regex("[_/.,\\-]"): m => m + sym.zws
    it
  }
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
    let el = it.element
    if el.func() != heading { return it }
    set text(weight: "bold")
    if el.numbering != none {
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


// Kernel time in ms with two decimals below 10 and one above.
#let ms(x) = if x == none { "–" } else { fx(x, d: if x < 10 { 2 } else { 1 }) }
