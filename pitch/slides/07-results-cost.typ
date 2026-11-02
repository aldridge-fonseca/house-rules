#import "../theme.typ": *

= RESULTS · COST, SPEED, COVERAGE
== Cost, speed, coverage

#slide[
  #set par(leading: 3pt, spacing: 0pt)
  #align(right)[#chip("OURS", c-ours) #chip("BASELINE", c-ultra) #chip("FRONTIER", c-opus)]
  #v(13pt)
  #let panel(headline, caption, metric, rows, hi: 100) = block(
    width: 100%, inset: 12pt, radius: 8pt, fill: soft,
  )[
    #text(size: 33pt, weight: "bold", fill: c-ours, headline)
    #v(3pt)
    #text(size: 12pt, fill: ink, caption)
    #v(16pt)
    #text(size: 10pt, weight: "semibold", fill: muted, metric)
    #v(12pt)
    #hbars(rows, hi: hi, label-w: 78pt, bar-h: 16pt, size: 9.5pt)
  ]
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 12pt,
    panel([33x cheaper], [than GPT-6.1 Sol], [\$/MONTH · 100M MESSAGES¹], (
      ("House Rules", calc.ln(5.6), c-ours, "$5.6K"),
      ("Nano untuned", calc.ln(6.4), c-nano, "$6.4K"),
      ("Ultra untuned", calc.ln(10.8), c-ultra, "$10.8K"),
      ("GPT-6.1 Sol", calc.ln(182), c-gpt, "$182K"),
      ("Opus 5.5", calc.ln(605), c-opus, "$605K"),
    ), hi: calc.ln(605) * 1.12),
    panel([5x faster], [than GPT-6.1 Sol], [P50 LATENCY · SECONDS], (
      ("House Rules", 0.64, c-ours, "0.64"),
      ("Nano untuned", 0.80, c-nano, "0.80"),
      ("Ultra untuned", 0.71, c-ultra, "0.71"),
      ("GPT-6.1 Sol", 3.2, c-gpt, "3.2"),
      ("Opus 5.5", 3.0, c-opus, "3.0"),
    ), hi: 3.6),
    panel([0% refusals], [100% answered], [REFUSED · %], (
      ("House Rules", 0, c-ours, "0%"),
      ("Nano untuned", 0, c-nano, "0%"),
      ("Ultra untuned", 0, c-ultra, "0%"),
      ("GPT-6.1 Sol", 1.9, c-gpt, "1.9%"),
      ("Opus 5.5", 13, c-opus, "13%"),
    ), hi: 15),
  )
  #v(9pt)
  #grid(columns: (1fr, 2fr), column-gutter: 12pt,
    text(size: 9pt, fill: muted)[¹ Cost bars use a log scale.],
    [],
  )
  #sources[Held-out NVIDIA Nemotron Content-Safety test, n = 3,340; same rows for every system; refusals scored as wrong. Cost = Fireworks GPU time / API receipts.]
]
