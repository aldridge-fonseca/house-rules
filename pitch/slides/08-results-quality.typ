#import "../theme.typ": *

= RESULTS · QUALITY
== Quality on the company’s own policy

#slide[
  #set par(leading: 3pt, spacing: 0pt)
  #grid(columns: (1fr, auto), align: horizon,
    text(size: 11pt, weight: "semibold", fill: muted)[HELD-OUT TEST · n = 3,340 · identical rows · zero-shot baselines had no written policy],
    [#chip("OURS", c-ours) #chip("BASELINE", c-ultra) #chip("FRONTIER", c-opus)],
  )
  #v(14pt)
  #grid(columns: (2.3fr, 1fr), column-gutter: 22pt,
    [
      #grid(columns: (1fr, 102pt), column-gutter: 14pt,
        align(right, text(size: 10pt, weight: "semibold", fill: muted)[MACRO-F1 ↑]),
        align(right, text(size: 10pt, weight: "semibold", fill: muted)[\$ / 1M messages]),
      )
      #v(8pt)
      #grid(columns: (1fr, 102pt), column-gutter: 14pt,
        hbars((
          ("House Rules", 95.3, c-ours, "95.3"),
          ("Nano 30B · untuned", 82.7, c-nano, "82.7"),
          ("Ultra 550B · untuned", 84.5, c-ultra, "84.5"),
          ("GPT-6.1 Sol", 71.5, c-gpt, "71.5"),
          ("Claude Opus 5.5", 69.0, c-opus, "69.0"),
        ), lo: 50, hi: 100, label-w: 151pt, bar-h: 22pt, size: 12.5pt),
        grid(columns: (1fr,), rows: (22pt,) * 5, row-gutter: 8pt,
          ..(("$56", c-ours), ("$64", c-nano), ("$108", c-ultra),
             ("$1,820", c-gpt), ("$6,045", c-opus)).map(r =>
            block(width: 100%, height: 22pt, inset: (left: 10pt),
              stroke: (left: 2pt + r.at(1)))[
              #align(right + horizon, text(size: 13pt, weight: "bold", fill: ink, r.at(0)))
            ]
          ),
        ),
      )
      #v(5pt)
      #align(right, text(size: 9pt, fill: muted)[Bar range: 50–100 · marginal \$/1M])
    ],
    block(width: 100%, inset: 11pt, radius: 8pt, fill: soft)[
      #stack(spacing: 5pt,
        text(size: 28pt, weight: "bold", fill: c-ours)[+23.8],
        text(size: 11pt)[vs GPT-6.1 Sol],
        text(size: 9pt, fill: muted)[95% CI +22.1 to +25.5],
        line(length: 100%, stroke: 0.6pt + c-ours.lighten(65%)),
        text(size: 28pt, weight: "bold", fill: c-ours)[+26.3],
        text(size: 11pt)[vs Opus 5.5],
        text(size: 9pt, fill: muted)[95% CI +24.6 to +28.0],
        line(length: 100%, stroke: 0.6pt + c-ours.lighten(65%)),
        text(size: 28pt, weight: "bold", fill: c-ours)[+10.8],
        text(size: 11pt)[vs Ultra 550B],
        text(size: 9pt, fill: muted)[95% CI +9.4 to +12.4],
      )
    ],
  )
  #v(10pt)
  #grid(columns: (1.6fr, 1fr), column-gutter: 26pt, align: horizon,
    [
      #text(size: 25pt, weight: "bold", fill: c-ours)[97.6%]
      #text(size: 12pt)[harmful caught]
      #text(size: 12pt, fill: muted)[vs #text(fill: c-gpt, weight: "bold")[70.0%] GPT]
    ],
    [#text(size: 25pt, weight: "bold", fill: c-ours)[5.5%] #text(size: 12pt)[safe wrongly blocked]],
  )
  #v(8pt)
  #text(size: 9.5pt, fill: muted)[Disguised-set frontier receipts not retained · unverifiable here.]
  #sources[NVIDIA policy n = 3,340 · same rows · refusals = wrong. Cost = Fireworks GPU/API receipts. Paired CIs: results/stats/vs_frontier.json.]
]
