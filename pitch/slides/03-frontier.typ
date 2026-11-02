#import "../theme.typ": *

= FRONTIER MODELS FALL SHORT
== Why not just use a frontier model?

#let big-card(num, label, body, color) = block(
  width: 100%, height: 210pt, breakable: false,
  inset: (x: 16pt, y: 22pt), radius: 10pt, fill: soft,
  stroke: (top: 3pt + color),
)[
  #set par(leading: 4pt, spacing: 0pt)
  #stack(spacing: 16pt,
    text(size: 40pt, weight: "bold", fill: color, tracking: -1pt, num),
    text(size: 16pt, weight: "bold", fill: ink, tracking: 0.8pt, label),
    text(size: 16pt, fill: muted, body),
  )
]

#slide[
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 16pt,
    big-card([\$182K / month], [COST], [GPT-6.1 Sol at 100M msgs], c-gpt),
    big-card([\~30% missed], [WRONG RULEBOOK], [Follows its lab’s policy, not yours.], ink),
    big-card([13% refused], [REFUSES], [Claude Opus 5.5], c-opus),
  )
  #sources[Held-out NVIDIA Nemotron Content-Safety · n = 3,340 · same rows · refusals = wrong. Bills at 100M msgs/mo.]
]
