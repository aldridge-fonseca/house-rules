#import "../theme.typ": *

= WHAT WE BUILT
== Our moderation harness

#let edge = rgb("#D8E3E4")
#let arrow = align(horizon + center, text(size: 24pt, fill: muted)[→])

#let step(icon, title, sub) = block(
  width: 100%, breakable: false,
  inset: (x: 8pt, y: 14pt), radius: 10pt, fill: soft,
)[
  #set align(center)
  #set par(leading: 3.2pt, spacing: 0pt)
  #text(size: 38pt, fill: hr, icon)
  #v(10pt)
  #text(size: 16pt, weight: "bold", fill: ink, title)
  #v(6pt)
  #text(size: 15pt, fill: muted, sub)
]

#let model-tile(name, note, main: false) = block(
  width: 100%, height: 112pt, breakable: false,
  inset: 14pt, radius: 10pt,
  fill: if main { soft } else { white },
  stroke: if main { 2pt + hr } else { 0.8pt + edge },
)[
  #set par(leading: 3.2pt, spacing: 0pt)
  #stack(dir: ltr, spacing: 6pt,
    logo-nv(12pt),
    text(size: 12pt, fill: muted)[+],
    logo-fw(12pt),
  )
  #v(10pt)
  #text(size: 15pt, weight: "bold", fill: if main { hr } else { ink }, name)
  #v(8pt)
  #text(size: 15pt, fill: muted, note)
]

#slide(setting: body => body)[
  #set par(leading: 3pt, spacing: 0pt)
  #grid(
    columns: (1fr, 14pt, 1fr, 14pt, 1fr, 14pt, 1fr, 14pt, 1fr),
    align: horizon,
    step([✨], [Clean up], [undo disguises]),
    arrow,
    step([⛔], [Rules], [instant block]),
    arrow,
    step([⤴], [Router], [text · language · image]),
    arrow,
    step([⚖], [Decision step], [unsure? ask a bigger model]),
    arrow,
    step([✔], [Verdict], [\+ category, cost]),
  )
  #v(20pt)
  #grid(
    columns: (1.35fr, 1fr, 1fr),
    column-gutter: 14pt,
    model-tile(
      [NVIDIA Nemotron Nano 3 30B — fine-tuned on Fireworks],
      [handles \~95% of messages],
      main: true,
    ),
    model-tile(
      [NVIDIA Nemotron 3 Ultra 550B — Fireworks serverless],
      [second opinion, 5.8%],
    ),
    model-tile(
      [NVIDIA Nemotron 3.5 Super VL — images],
      [vision lane],
    ),
  )
]
