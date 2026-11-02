#import "../theme.typ": *

= FINE-TUNING
== One Fireworks training job

#let arrow = align(horizon + center, text(size: 32pt, fill: muted)[→])

#let stage(accent, header, body) = block(
  width: 100%, height: 268pt, breakable: false,
  inset: 18pt, radius: 10pt, fill: soft,
  stroke: (top: 4pt + accent),
)[
  #set align(horizon + left)
  #set par(leading: 5pt, spacing: 0pt)
  #stack(spacing: 18pt, header, body)
]

#slide(setting: body => body)[
  #set par(leading: 4pt, spacing: 0pt)
  #grid(
    columns: (1fr, 26pt, 1fr, 26pt, 1.2fr),
    align: horizon,
    stage(nv,
      stack(spacing: 10pt,
        logo-nv(18pt),
        text(size: 18pt, weight: "bold", fill: nv, tracking: 0.6pt)[DATA],
      ),
      text(size: 16pt, fill: ink)[
        Nemotron Content Safety\
        · 19,806 examples\
        · 12 languages
      ],
    ),
    arrow,
    stage(fw,
      stack(spacing: 10pt,
        logo-fw(18pt),
        text(size: 15pt, weight: "bold", fill: fw, tracking: 0.3pt)[TRAINING API],
      ),
      text(size: 15pt, fill: ink)[
        LoRA on NVIDIA Nemotron Nano 3 30B\
        · 1 epoch\
        · 2× NVIDIA B200\
        · 19 min
      ],
    ),
    arrow,
    stage(hr,
      text(size: 18pt, weight: "bold", fill: hr, tracking: 0.6pt)[RESULT],
      [
        #text(size: 40pt, weight: "bold", fill: c-ours, tracking: -1pt)[82.7 → 95.3]
        #v(8pt)
        #text(size: 16pt, weight: "semibold", fill: ink)[macro-F1]
        #v(2pt)
        #text(size: 15pt, fill: muted)[untuned → tuned]
        #v(16pt)
        #text(size: 15pt, fill: muted)[held-out loss 0.753 → 0.104]
      ],
    ),
  )

  #sources[results/training (raw training log)]
]
