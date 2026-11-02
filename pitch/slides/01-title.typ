#import "../theme.typ": *

#slide(config: utils.merge-dicts(
  config-page(header: none, margin: (left: 44pt, right: 44pt, top: 35pt, bottom: 49pt)),
))[
  #box(width: 1pt, height: 1pt)
  #place(top + right, dx: 44pt, dy: -35pt,
    rect(width: 12pt, height: 100%, fill: hr, stroke: none))
  #place(top + left, dy: 54pt, text(size: 76pt, weight: "bold", fill: hr, tracking: -2pt)[House Rules])
  #place(top + left, dy: 151pt, text(size: 30pt, weight: "medium", fill: ink)[Your safety policy, enforced.])
  #place(top + left, dy: 210pt, rect(width: 64pt, height: 5pt, fill: hr, stroke: none))
  #place(top + left, dy: 244pt, text(size: 18pt, weight: "semibold")[Aldridge Fonseca · Gitika Rath])
  #place(top + left, dy: 295pt, block(width: 100%, fill: soft, radius: 8pt, inset: 15pt)[
    #stack(spacing: 10pt,
      text(size: 14pt, fill: ink, weight: "medium")[Built on Fireworks AI · NVIDIA Nemotron],
      grid(columns: (auto, auto), column-gutter: 28pt, align: horizon,
        brand-nv(height: 14pt), brand-fw(height: 14pt)),
    )
  ])
  #place(top + left, dy: 388pt, text(size: 10pt, fill: muted)[Fireworks: Deploy · with Microsoft & NVIDIA · Nov 2, 2026])
]
