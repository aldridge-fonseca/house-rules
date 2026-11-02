#import "../theme.typ": *

= RESULTS · QUALITY
== Better on your policy

#slide[
  #set par(leading: 3pt, spacing: 0pt)

  #grid(columns: (2.55fr, 1fr), column-gutter: 22pt, align: top,
    [
      #align(right, text(size: 13pt, weight: "semibold", fill: muted)[MACRO-F1 ↑])
      #v(10pt)
      #hbars((
        ("House Rules", 95.3, c-ours, "95.3"),
        ("Nano 30B · untuned", 82.7, c-nano, "82.7"),
        ("Ultra 550B · untuned", 84.5, c-ultra, "84.5"),
        ("GPT-6.1 Sol", 71.5, c-gpt, "71.5"),
        ("Claude Opus 5.5", 69.0, c-opus, "69.0"),
      ), lo: 50, hi: 100, label-w: 178pt, bar-h: 40pt, size: 17pt)
    ],
    [
      #v(6pt)
      #stack(spacing: 30pt,
        stack(spacing: 4pt,
          text(size: 54pt, weight: "bold", fill: c-ours, tracking: -1.5pt)[+23.8],
          text(size: 16pt, weight: "semibold", fill: ink)[vs GPT-6.1 Sol],
          text(size: 11pt, fill: muted)[95% CI +22.1, +25.5],
        ),
        stack(spacing: 4pt,
          text(size: 54pt, weight: "bold", fill: c-ours, tracking: -1.5pt)[+26.3],
          text(size: 16pt, weight: "semibold", fill: ink)[vs Opus 5.5],
          text(size: 11pt, fill: muted)[95% CI +24.6, +28.0],
        ),
      )
    ],
  )

  #v(22pt)
  #text(size: 16pt, weight: "semibold", fill: ink)[0% refusals · 5× faster than GPT]

  #sources[Same 3,340 held-out cases · refusals count as wrong]
]
