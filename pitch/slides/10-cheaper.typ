#import "../theme.typ": *

= RESULTS · COST
== And 32× cheaper

#let cost-col(name, dollars, bar-frac, color) = align(center, stack(spacing: 10pt,
  text(size: 48pt, weight: "bold", fill: color, tracking: -1pt, dollars),
  text(size: 15pt, weight: "semibold", fill: ink, name),
  v(4pt),
  block(width: 100%, height: 18pt, radius: 4pt, fill: rgb("#E8EEEE"), clip: true)[
    #place(left + horizon,
      rect(width: 100% * bar-frac, height: 18pt, radius: 4pt, fill: color, stroke: none))
  ],
))

#slide[
  #set par(leading: 3pt, spacing: 0pt)

  #align(center, text(size: 15pt, weight: "semibold", fill: muted)[COST PER 1M MESSAGES])
  #v(20pt)

  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 28pt, align: top,
    cost-col([House Rules], [\$56], 0.05, c-ours),
    cost-col([GPT-6.1 Sol], [\$1,820], 0.30, c-gpt),
    cost-col([Claude Opus 5.5], [\$6,045], 1.0, c-opus),
  )

  #v(28pt)
  #align(center, text(size: 18pt, weight: "bold", fill: ink)[
    At 100M messages/month: \$9.8K vs \$182K vs \$605K
  ])

  #v(12pt)
  #align(center, text(size: 13pt, fill: muted)[
    Below \~5.2M messages/month, pay-per-token GPT is cheaper.
  ])

  #v(20pt)
  #align(center, text(size: 16pt, weight: "semibold", fill: hr)[
    github.com/aldridge-fonseca/house-rules
  ])
]
