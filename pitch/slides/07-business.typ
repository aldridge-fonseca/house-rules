#import "../theme.typ": *

= BUSINESS VALUE
== Moderation is a billion-dollar line item

#let spend-card(who, dollars, people) = block(
  width: 100%, breakable: false,
  inset: (x: 12pt, y: 10pt), radius: 10pt, fill: soft,
  stroke: (top: 3pt + muted),
)[
  #set par(leading: 2.5pt, spacing: 0pt)
  #text(size: 15pt, weight: "bold", fill: muted, who)
  #v(4pt)
  #text(size: 28pt, weight: "bold", fill: ink, tracking: -0.8pt, dollars)
  #v(3pt)
  #text(size: 15pt, fill: muted, people)
]

#let trust-chip(icon, label) = block(
  width: 100%, breakable: false,
  inset: (x: 6pt, y: 7pt), radius: 8pt, fill: soft,
)[
  #set align(center)
  #set par(leading: 2pt, spacing: 0pt)
  #text(size: 16pt, fill: hr, icon)
  #v(3pt)
  #text(size: 15pt, weight: "semibold", fill: ink, label)
]

#slide[
  #set par(leading: 3pt, spacing: 0pt)

  #grid(columns: (1fr, 1.1fr), column-gutter: 20pt, align: top,
    [
      #text(size: 15pt, weight: "bold", fill: muted, tracking: 0.8pt)[WHAT BIG TECH SPENDS]
      #v(6pt)
      #spend-card([Meta], [\~\$5B \/ year], [\~40,000 people on safety & security])
      #v(8pt)
      #spend-card([TikTok], [\$2B+ in 2024], [40,000+ trust & safety staff])
      #v(8pt)
      #text(size: 15pt, fill: muted)[
        Moderation market: \~\$13B (2026) → \~\$26B (2031), industry estimate
      ]
    ],
    [
      #text(size: 15pt, weight: "bold", fill: muted, tracking: 0.8pt)[WHAT HOUSE RULES SAVES]
      #v(6pt)
      #block(
        width: 100%, height: 188pt, breakable: false,
        inset: (x: 14pt, y: 14pt), radius: 10pt, fill: soft,
        stroke: (top: 3pt + hr),
      )[
        #set par(leading: 2.5pt, spacing: 0pt)
        #text(size: 46pt, weight: "bold", fill: hr, tracking: -1.5pt)[\$2.1M]
        #v(1pt)
        #text(size: 18pt, weight: "bold", fill: ink)[\/ year saved]
        #v(5pt)
        #text(size: 15pt, fill: muted)[vs GPT-6.1 Sol · 100M msgs/month]
        #v(14pt)
        #text(size: 28pt, weight: "bold", fill: hr, tracking: -1pt)[\~\$21M \/ year]
        #v(3pt)
        #text(size: 15pt, fill: muted)[at 1B msgs/month · at marginal rate]
      ]
    ],
  )

  #v(10pt)
  #text(size: 15pt, weight: "bold", fill: muted, tracking: 0.8pt)[TRUST]
  #v(6pt)
  #grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 10pt,
    trust-chip([✔], [0% refusals]),
    trust-chip([✎], [Your policy, not a vendor’s]),
    trust-chip([☰], [Audit trail on every verdict]),
    trust-chip([◆], [Open NVIDIA on your Fireworks]),
  )

  #sources[Senate testimony (Meta, Jan 2024) · TikTok Senate QFR responses (2024) · Mordor Intelligence]
]
