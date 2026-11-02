#import "../theme.typ": *

= WHAT WE BUILT
== House Rules

#slide[
  #set par(leading: 0pt, spacing: 0pt)
  // Coordinates share one canvas: every branch starts at the router and
  // rejoins before the verdict. Logos identify the model and serving partners.
  #let wire(x, y, dx, dy: 0pt, arrow: false, color: muted) = place(
    top + left, dx: x, dy: y,
    line(start: (0pt, 0pt), end: (dx, dy), stroke: 1pt + color)
  ) + if arrow {
    place(top + left, dx: x + dx - 4pt, dy: y + dy - 3pt,
      polygon(fill: color, stroke: none, (0pt, 0pt), (4pt, 3pt), (0pt, 6pt)))
  }
  #let stage(x, width, title, detail: none) = place(top + left, dx: x, dy: 62pt,
    block(width: width, height: 54pt, inset: 8pt, radius: 7pt,
      fill: soft, stroke: none)[
      #align(center + horizon)[
        #text(size: 13pt, weight: "bold")[#title]
        #if detail != none { [#v(5pt)#text(size: 10pt, fill: muted)[#detail]] }
      ]
    ])
  #let lane(y, label, model, main: false, fireworks: false) = place(
    top + left, dx: 321pt, dy: y,
    block(width: 299pt, height: 54pt, inset: 10pt, radius: 7pt,
      fill: if main { soft } else { rgb("#F5F7F7") },
      stroke: if main { 1.5pt + hr } else { 0.6pt + rgb("#DCE4E5") })[
      #place(top + right, dx: 0pt, dy: 0pt,
        stack(dir: ltr, spacing: 8pt, logo-nv(8pt),
          if fireworks { logo-fw(7pt) }))
      #text(size: 9pt, weight: "bold", fill: if main { hr } else { muted })[#label]
      #v(8pt)
      #text(size: 14pt, weight: "semibold")[#model]
    ])

  #block(width: 100%, height: 182pt)[
    #stage(0pt, 65pt, [Message])
    #wire(65pt, 89pt, 19pt, arrow: true)
    #stage(84pt, 113pt, [Clean-up], detail: [Undo disguises])
    #wire(197pt, 89pt, 19pt, arrow: true)
    #stage(216pt, 69pt, [Router])
    #wire(285pt, 89pt, 17pt)
    #wire(302pt, 27pt, 0pt, dy: 124pt)
    #wire(302pt, 27pt, 19pt, arrow: true, color: hr)
    #wire(302pt, 89pt, 19pt, arrow: true)
    #wire(302pt, 151pt, 19pt, arrow: true)
    #lane(0pt, [TEXT · FINE-TUNED], [Nemotron Nano 3 30B-A3B], main: true, fireworks: true)
    #lane(62pt, [IMAGES], [Nemotron 3.5 Super VL 120B], fireworks: true)
    #lane(124pt, [UNSURE · 3%], [Nemotron 3 Ultra 550B], fireworks: true)
    #wire(620pt, 27pt, 18pt, color: hr)
    #wire(620pt, 89pt, 18pt)
    #wire(620pt, 151pt, 18pt)
    #wire(638pt, 27pt, 0pt, dy: 124pt)
    #wire(638pt, 89pt, 18pt, arrow: true)
    #place(top + left, dx: 656pt, dy: 62pt,
      block(width: 97pt, height: 54pt, inset: 7pt, radius: 7pt, fill: hr)[
        #align(center + horizon)[
          #text(size: 14pt, weight: "bold", fill: white)[Verdict]
          #v(5pt)
          #text(size: 11pt, fill: white)[\+ category]
        ]
      ])
  ]
  #v(10pt)
  #grid(columns: (302pt, 1fr), column-gutter: 12pt,
    block(height: 75pt, width: 100%, inset: 11pt, radius: 7pt,
      stroke: 0.6pt + rgb("#DCE4E5"))[
      #text(size: 9pt, weight: "bold", fill: muted, tracking: 0.7pt)[TRAINED ON]
      #v(7pt)
      #text(size: 12pt, weight: "semibold")[NVIDIA Nemotron Content Safety Dataset v3.5]
      #v(6pt)
      #text(size: 11pt, fill: hr, weight: "bold")[19,806 rows · leakage-filtered]
    ],
    block(height: 75pt, width: 100%, inset: 11pt, radius: 7pt,
      stroke: 0.6pt + rgb("#DCE4E5"))[
      #text(size: 9pt, weight: "bold", fill: fw, tracking: 0.7pt)[HOW · FIREWORKS TRAINING API]
      #v(7pt)
      #text(size: 12pt, weight: "semibold")[LoRA r8 · 1 epoch · \~15 min · 2× NVIDIA B200]
      #v(7pt)
      #text(size: 10.5pt, fill: muted)[Served on Fireworks: dedicated + serverless · firectl end to end]
    ],
  )
]
