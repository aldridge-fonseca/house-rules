#import "../theme.typ": *

= TEST DATA
== What we tested on

#let danger = rgb("#AC3546")
#let strip-item(value, label) = stack(spacing: 1pt,
  text(size: 24pt, weight: "bold", fill: hr, tracking: -0.5pt, value),
  text(size: 10pt, fill: muted, label),
)
#let card(accent: hr, body) = block(
  width: 100%, height: 218pt, inset: (x: 12pt, y: 10pt),
  radius: 9pt, fill: soft, stroke: (top: 3pt + accent), breakable: false, body,
)
#let src-row(name, n) = grid(columns: (1fr, auto),
  text(size: 10.5pt, name),
  text(size: 10.5pt, weight: "bold", fill: ink, n),
)
#let lang(name, n) = grid(columns: (1fr, auto),
  text(size: 9.5pt, fill: muted, name),
  text(size: 9.5pt, weight: "semibold", fill: ink, n),
)

#slide[
  #set par(leading: 2.5pt, spacing: 0pt)

  // Big strip — four headline facts
  #block(width: 100%, inset: (x: 6pt, y: 7pt), radius: 7pt, fill: soft)[
    #grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 12pt, align: center,
      strip-item("5,562", "evaluated rows"),
      strip-item("23", "categories"),
      strip-item("12", "languages"),
      strip-item("8", "attack forms"),
    )
  ]

  #v(8pt)

  #grid(columns: (1fr, 1fr), column-gutter: 12pt,
    // Card 1 — NVIDIA policy test
    card(accent: nv)[
      #grid(columns: (1fr, auto), align: horizon,
        text(size: 14pt, weight: "bold", fill: ink)[NVIDIA policy test],
        logo-nv(10pt),
      )
      #v(2pt)
      #text(size: 10pt, fill: muted)[Nemotron Content-Safety v3.5 · held-out]
      #v(6pt)
      #grid(columns: (auto, 1fr), column-gutter: 8pt, align: horizon,
        text(size: 28pt, weight: "bold", fill: nv, tracking: -0.6pt)[3,340],
        text(size: 11pt, fill: muted)[rows · follows THIS company's policy],
      )
      #v(6pt)
      #grid(columns: (1fr, 1fr), column-gutter: 8pt,
        stack(spacing: 0pt,
          text(size: 17pt, weight: "bold", fill: danger)[2,699],
          text(size: 10pt, fill: muted)[unsafe],
        ),
        stack(spacing: 0pt,
          text(size: 17pt, weight: "bold", fill: hr)[641],
          text(size: 10pt, fill: muted)[safe],
        ),
      )
      #v(7pt)
      #text(size: 10pt, weight: "bold", fill: ink)[Languages]
      #v(3pt)
      #grid(columns: (1fr, auto),
        text(size: 10pt, weight: "semibold")[English],
        text(size: 10pt, weight: "bold")[590],
      )
      #v(3pt)
      #grid(columns: (1fr, 1fr), column-gutter: 8pt, row-gutter: 2pt,
        lang("French", "250"), lang("German", "250"),
        lang("Arabic", "250"), lang("Spanish", "250"),
        lang("Dutch", "250"), lang("Korean", "250"),
        lang("Hindi", "250"), lang("Chinese", "250"),
        lang("Japanese", "250"), lang("Thai", "250"),
        lang("Italian", "250"), [],
      )
    ],

    // Card 2 — Disguised attacks
    card(accent: hr)[
      #grid(columns: (1fr, auto), align: horizon,
        text(size: 14pt, weight: "bold", fill: ink)[Disguised attacks],
        chip("OUR SYNTHETIC SET", hr),
      )
      #v(2pt)
      #text(size: 10pt, fill: muted)[1,119 unsafe / 1,103 safe · 21 categories]
      #v(6pt)
      #grid(columns: (auto, 1fr), column-gutter: 8pt, align: horizon,
        text(size: 28pt, weight: "bold", fill: hr, tracking: -0.6pt)[2,222],
        text(size: 11pt, fill: muted)[rows = 300 messages × 8 forms],
      )
      #v(7pt)
      #text(size: 10pt, weight: "bold", fill: ink)[Base messages from 6 public sets]
      #v(4pt)
      #stack(spacing: 3pt,
        src-row("HatemojiCheck", "633"),
        src-row("OpenAI Moderation", "446"),
        src-row("Aegis 2.0", "382"),
        src-row("NVIDIA", "334"),
        src-row("WildGuard", "320"),
        src-row("XSTest", "107"),
      )
    ],
  )

  #v(6pt)
  #text(size: 10.5pt, fill: muted)[Images: Super VL lane wired · not benchmarked (future work)]

  #sources[Evaluated text only: 3,340 + 2,222 = 5,562 · results/nvidia-v1/final.jsonl]
]
