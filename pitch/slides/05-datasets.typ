#import "../theme.typ": *

= TEST DATA
== What we tested on — 5,562 held-out cases

#let disguise(label) = box(
  inset: (x: 7pt, y: 3pt), radius: 4pt, fill: soft,
  text(size: 11pt, weight: "semibold", fill: hr, label),
)
#let src(name, n, mark: none) = block(
  width: 100%, inset: (x: 7pt, y: 6pt), radius: 7pt, fill: white,
  stroke: 0.7pt + rgb("#D0DCDE"), breakable: false,
)[
  #set par(leading: 2.2pt, spacing: 0pt)
  #stack(spacing: 3pt,
    stack(dir: ltr, spacing: 4pt,
      text(size: 11pt, weight: "bold", fill: ink, name),
      if mark != none { mark } else { [] },
    ),
    text(size: 15pt, weight: "bold", fill: hr, n),
  )
]

#slide[
  #set par(leading: 3pt, spacing: 0pt)

  #grid(columns: (1fr, 1.15fr), column-gutter: 14pt,
    // Tile 1 — NVIDIA held-out test
    block(
      width: 100%, height: 268pt, inset: 14pt, radius: 10pt, fill: soft,
      stroke: (top: 3.5pt + nv), breakable: false,
    )[
      #grid(columns: (1fr, auto), align: horizon,
        text(size: 16pt, weight: "bold", fill: ink)[NVIDIA Nemotron Content Safety v3.5 test],
        logo-nv(12pt),
      )
      #v(18pt)
      #text(size: 42pt, weight: "bold", fill: nv, tracking: -1pt)[3,340]
      #v(6pt)
      #text(size: 15pt, fill: muted)[held-out cases]
      #v(18pt)
      #text(size: 15pt, fill: ink)[23 harm categories · 12 languages]
    ],

    // Tile 2 — Disguised attacks synthetic set
    block(
      width: 100%, height: 268pt, inset: 14pt, radius: 10pt, fill: soft,
      stroke: (top: 3.5pt + hr), breakable: false,
    )[
      #grid(columns: (1fr, auto), align: horizon,
        text(size: 16pt, weight: "bold", fill: ink)[Disguised attacks],
        chip("OUR SYNTHETIC SET", hr),
      )
      #v(8pt)
      #grid(columns: (auto, 1fr), column-gutter: 10pt, align: horizon,
        text(size: 36pt, weight: "bold", fill: hr, tracking: -0.8pt)[2,222],
        text(size: 14pt, fill: muted)[300 messages × 8 disguises],
      )
      #v(10pt)
      #text(size: 12pt, weight: "bold", fill: muted, tracking: 0.4pt)[BUILT FROM]
      #v(6pt)
      #grid(columns: (1fr, 1fr, 1fr), column-gutter: 5pt, row-gutter: 5pt,
        src("OpenAI Moderation", "446"),
        src("WildGuard", "320"),
        src("XSTest", "107"),
        src("Aegis 2.0", "382", mark: logo-nv(9pt)),
        src("HatemojiCheck", "633"),
        src("NVIDIA", "334", mark: logo-nv(9pt)),
      )
    ],
  )

  #v(10pt)
  #align(center, stack(dir: ltr, spacing: 5pt,
    disguise([plain]),
    disguise([leetspeak]),
    disguise([look-alike letters]),
    disguise([spaced]),
    disguise([invisible]),
    disguise([mixed case]),
    disguise([emoji]),
    disguise([base64]),
  ))

  #sources[Training data never overlaps these tests (170 overlapping rows removed)]
]
