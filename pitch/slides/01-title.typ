#import "../theme.typ": *

#slide(config: utils.merge-dicts(
  config-page(header: none, margin: (left: 44pt, right: 44pt, top: 35pt, bottom: 49pt)),
))[
  #box(width: 1pt, height: 1pt)
  #place(top + right, dx: 44pt, dy: -35pt,
    rect(width: 12pt, height: 100%, fill: hr, stroke: none))
  #place(top + left, dy: 72pt, text(size: 76pt, weight: "bold", fill: hr, tracking: -2pt)[House Rules])
  #place(top + left, dy: 172pt, text(size: 30pt, weight: "medium", fill: ink)[Your safety policy, enforced.])
  #place(top + left, dy: 230pt, rect(width: 64pt, height: 5pt, fill: hr, stroke: none))
  #place(top + left, dy: 268pt, text(size: 18pt, weight: "semibold")[Aldridge Fonseca · Gitika Rath])
  #place(top + left, dy: 330pt, stack(dir: ltr, spacing: 14pt,
    text(size: 16pt, fill: ink, weight: "medium")[Built on],
    logo-nv(16pt),
    text(size: 16pt, fill: ink, weight: "medium")[Nemotron +],
    logo-fw(16pt),
  ))
]
