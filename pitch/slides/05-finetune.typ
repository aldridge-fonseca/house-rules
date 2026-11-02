#import "../theme.typ": *

= FINE-TUNING ON FIREWORKS
== How we fine-tuned

#let recipe(n, title, body, accent: hr) = block(
  width: 100%, height: 248pt, breakable: false,
  inset: 10pt, radius: 9pt, fill: soft,
  stroke: (top: 3pt + accent),
)[
  #set align(top + left)
  #set par(leading: 3pt, spacing: 0pt)
  #text(size: 11pt, weight: "bold", fill: accent, tracking: 0.7pt)[#n / #title]
  #v(8pt)
  #body
]
#let arrow = align(horizon, text(size: 16pt, fill: muted)[→])
#let loss-row(label, value, width-frac, color) = grid(
  columns: (38pt, 1fr, 34pt), column-gutter: 4pt, align: horizon,
  text(size: 11pt, fill: muted, label),
  block(width: 100%, height: 9pt, radius: 2pt, fill: rgb("#D9E6E4"), clip: true)[
    #place(left + horizon, rect(width: 100% * width-frac, height: 9pt, radius: 2pt, fill: color, stroke: none))
  ],
  text(size: 11pt, weight: "bold", fill: color, value),
)

#slide(setting: body => body)[
  #set par(leading: 3pt, spacing: 0pt)
  #grid(
    columns: (1fr, 11pt, 1.05fr, 11pt, 1.15fr, 11pt, 1.1fr),
    align: horizon,
    recipe([01], [DATA], [
      #brand-nv(height: 11pt)
      #v(6pt)
      #text(size: 11pt, weight: "semibold")[Nemotron Content\ Safety Dataset v3.5]
      #v(8pt)
      #text(size: 28pt, weight: "bold", fill: hr)[19,830]
      #v(1pt)
      #text(size: 11pt, fill: muted)[rows after removing\ test overlaps]
      #v(8pt)
      #text(size: 11pt, weight: "semibold")[19,806 consumed]
      #v(1pt)
      #text(size: 11pt, fill: muted)[24 dropped by trainer filter]
      #v(8pt)
      #text(size: 11pt, weight: "semibold")[12 languages]
      #v(1pt)
      #text(size: 11pt, fill: muted)[23 categories]
    ], accent: nv),
    arrow,
    recipe([02], [FORMAT], [
      #text(size: 11pt, weight: "semibold")[Chat SFT]
      #v(8pt)
      #set text(size: 8pt, font: "DejaVu Sans Mono")
      #stack(spacing: 8pt,
        stack(spacing: 2pt,
          text(fill: hr, weight: "bold")[system],
          [You are a content-safety\
           classifier. Reply only\
           with a JSON object\
           \{label, categories\}],
        ),
        stack(spacing: 2pt,
          text(fill: hr, weight: "bold")[user],
          raw("<message>"),
        ),
        stack(spacing: 2pt,
          text(fill: hr, weight: "bold")[assistant],
          raw("{\"label\":\"unsafe\",\n \"categories\":\n [\"criminal_planning\"]}"),
        ),
      )
    ]),
    arrow,
    recipe([03], [TRAIN], [
      #brand-fw(height: 11pt)
      #v(5pt)
      #text(size: 11pt, weight: "semibold", fill: fw)[Training API]
      #v(5pt)
      #text(size: 11pt, weight: "semibold")[NVIDIA Nemotron\ Nano 3 30B-A3B]
      #v(7pt)
      #grid(columns: (1fr, auto), row-gutter: 4pt,
        ..(("LoRA", "r8 / α32"),
           ("Learning rate", "1e-4"),
           ("Batch / max seq", "32 / 2,048"),
           ("1 epoch", "620 steps")).map(pair => (
          text(size: 11pt, fill: muted, pair.at(0)),
          text(size: 11pt, weight: "semibold", pair.at(1)),
        )).flatten(),
      )
      #v(9pt)
      #text(size: 11pt, weight: "bold", fill: fw)[2× NVIDIA B200]
      #v(1pt)
      #text(size: 11pt, fill: muted)[19-min trainer job (incl. evals)]
    ], accent: fw),
    arrow,
    recipe([04], [RESULT], [
      #text(size: 11pt, fill: muted)[HELD-OUT EVAL LOSS]
      #v(5pt)
      #loss-row([before], [0.753], 1.0, c-nano)
      #v(3pt)
      #loss-row([after], [0.104], 0.14, c-ours)
      #v(12pt)
      #text(size: 11pt, fill: muted)[MACRO-F1 · NVIDIA TEST]
      #v(6pt)
      #grid(columns: (auto, 1fr), column-gutter: 6pt, align: horizon,
        text(size: 22pt, weight: "bold", fill: c-nano)[82.7],
        text(size: 11pt, fill: muted)[untuned],
      )
      #v(2pt)
      #align(left, text(size: 14pt, fill: hr)[↓])
      #v(1pt)
      #text(size: 38pt, weight: "bold", fill: c-ours)[95.3]
      #v(2pt)
      #text(size: 11pt, weight: "semibold", fill: c-ours)[tuned · House Rules]
    ]),
  )
  #v(8pt)
  #text(size: 11pt, fill: muted)[
    LoRA rank/alpha match #brand-nv(height: 11pt)'s published Nemotron Safety Guard recipe
  ]
  #sources[results/training/ (raw training log) · results/nvidia-v1/ours_summary.json]
]
