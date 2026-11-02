#import "../theme.typ": *

= WHY NOT JUST USE A FRONTIER MODEL?
== Smarter model. Wrong tool.

// Frontier-only argument; model colors follow the shared series palette.
#let frontier-card(title, body) = block(
  width: 100%, height: 261pt, breakable: false,
  inset: 16pt, radius: 10pt, fill: rgb("#F3F4F5"),
  stroke: (top: 3pt + c-opus),
)[
  #text(size: 12pt, weight: "bold", tracking: 0.6pt, fill: ink, title)
  #v(15pt)
  #body
]
#let model-label(name, color) = text(size: 12pt, weight: "semibold", fill: color, name)
#let metric-row(name, value, color) = grid(
  columns: (1fr, auto), align: horizon, column-gutter: 5pt,
  model-label(name, color), text(size: 19pt, weight: "bold", fill: color, value),
)

#slide(setting: body => body)[
  #set par(leading: 3pt, spacing: 0pt)
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 14pt,
    frontier-card([COST], [
      #model-label([GPT-6.1 Sol], c-gpt)
      #v(6pt)
      #text(size: 44pt, weight: "bold", fill: c-gpt)[\$182K]
      #v(17pt)
      #model-label([Opus 5.5], c-opus)
      #v(6pt)
      #text(size: 44pt, weight: "bold", fill: c-opus)[\$605K]
      #v(17pt)
      #text(size: 12pt, fill: muted)[always-on · 100M msgs/mo]
    ]),
    frontier-card([WRONG RULEBOOK], [
      #text(size: 48pt, weight: "bold", fill: ink)[\~30%]
      #v(3pt)
      #text(size: 15pt, weight: "semibold")[harmful missed]
      #v(12pt)
      #text(size: 14pt, fill: muted)[Misses this policy’s unsafe cases.]
      #v(26pt)
      #text(size: 11pt, fill: muted)[HARMFUL CAUGHT]
      #v(6pt)
      #metric-row([GPT-6.1 Sol], [70.0%], c-gpt)
      #v(7pt)
      #metric-row([Opus 5.5], [67.6%], c-opus)
    ]),
    frontier-card([REFUSES TO CLASSIFY], [
      #text(size: 48pt, weight: "bold", fill: c-opus)[13%]
      #v(3pt)
      #model-label([Opus 5.5 · 434 of 3,340], c-opus)
      #v(26pt)
      #metric-row([GPT-6.1 Sol], [1.9%], c-gpt)
      #v(28pt)
      #text(size: 15pt, weight: "semibold")[A moderator that refuses\ is useless.]
    ]),
  )
  #sources[Held-out NVIDIA Nemotron Content-Safety policy test · n = 3,340 · same rows · refusals counted as wrong. Monthly = always-on bills at 100M.]
]
