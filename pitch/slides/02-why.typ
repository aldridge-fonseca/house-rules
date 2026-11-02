#import "../theme.typ": *

= WHY THIS IS NEEDED
== One AI feature. Eleven days.

#slide[
  #grid(columns: (1.2fr, 1fr), column-gutter: 32pt,
    [
      #text(size: 12pt, weight: "semibold", fill: muted)[GROK ON X · DEC 29 – JAN 8]
      #v(12pt)
      #stack(spacing: 25pt,
        stat([\~3 million], [sexualized images of real people · researchers estimated]),
        stat([\~23,000], [appeared to depict children · same estimate], color: ink),
      )
    ],
    [
      #block(width: 100%, breakable: false, inset: 20pt, radius: 10pt, fill: soft)[
        #stack(spacing: 19pt,
          text(size: 21pt, weight: "bold", fill: hr)[AI agents share our platforms.],
          stack(spacing: 7pt,
            text(size: 14pt, fill: muted)[Moderate every interaction.],
            text(size: 21pt, weight: "semibold")[Agent → human \ Agent → agent],
          ),
          stack(spacing: 5pt,
            text(size: 39pt, weight: "bold", fill: hr)[51%],
            text(size: 14pt, fill: muted)[automated web traffic · 2024],
          ),
        )
      ]
    ],
  )
  #sources[
    #link("https://www.theguardian.com/technology/2026/jan/22/grok-ai-generated-millions-sexualised-images-in-month-research-says")[CCDH estimate via The Guardian (Jan 22, 2026)] ·
    #link("https://www.imperva.com/resources/wp-content/uploads/sites/6/reports/2025-Bad-Bot-Report.pdf")[Imperva/Thales Bad Bot Report (2025)].
 Estimates only.
  ]
]
