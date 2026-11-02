#import "../theme.typ": *

= RESULTS · COST, SPEED, COVERAGE
== Always-on economics at 100M / month

#slide[
  #set par(leading: 3pt, spacing: 0pt)
  #grid(columns: (1fr, auto), align: horizon,
    text(size: 11pt, weight: "semibold", fill: muted)[32× lower marginal cost than GPT · \$56 vs \$1,820 per 1M msgs],
    [#chip("OURS", c-ours) #chip("BASELINE", c-ultra) #chip("FRONTIER", c-opus)],
  )
  #v(10pt)
  #let panel(headline, caption, metric, rows, hi: 100) = block(
    width: 100%, inset: 11pt, radius: 8pt, fill: soft,
  )[
    #text(size: 28pt, weight: "bold", fill: c-ours, headline)
    #v(2pt)
    #text(size: 11pt, fill: ink, caption)
    #v(12pt)
    #text(size: 9.5pt, weight: "semibold", fill: muted, metric)
    #v(10pt)
    #hbars(rows, hi: hi, label-w: 78pt, bar-h: 15pt, size: 9pt)
  ]
  #grid(columns: (1fr, 1fr, 1fr), column-gutter: 12pt,
    panel([18.6×], [cheaper than GPT · 61.8× vs Opus], [ALWAYS-ON \$/MO · 100M MSGS¹], (
      ("House Rules", calc.ln(9.777), c-ours, "$9,777"),
      ("Nano untuned", calc.ln(9.490), c-nano, "$9,490"),
      ("Ultra untuned", calc.ln(10.790), c-ultra, "$10,790"),
      ("GPT-6.1 Sol", calc.ln(181.964), c-gpt, "$181,964"),
      ("Opus 5.5", calc.ln(604.524), c-opus, "$604,524"),
    ), hi: calc.ln(604.524) * 1.12),
    panel([5×], [faster p50 than GPT], [P50 LATENCY · SECONDS], (
      ("House Rules", 0.64, c-ours, "0.64"),
      ("Nano untuned", 0.80, c-nano, "0.80"),
      ("Ultra untuned", 0.71, c-ultra, "0.71"),
      ("GPT-6.1 Sol", 3.2, c-gpt, "3.2"),
      ("Opus 5.5", 3.0, c-opus, "3.0"),
    ), hi: 3.6),
    panel([0%], [refusals · 100% answered], [REFUSED · %], (
      ("House Rules", 0, c-ours, "0%"),
      ("Nano untuned", 0, c-nano, "0%"),
      ("Ultra untuned", 0, c-ultra, "0%"),
      ("GPT-6.1 Sol", 1.9, c-gpt, "1.9%"),
      ("Opus 5.5", 13, c-opus, "13%"),
    ), hi: 15),
  )
  #v(8pt)
  #text(size: 9pt, fill: muted)[¹ Log scale · always-on \$9,777 vs \$181,964 / \$604,524. Below ~5.2M msgs/month, pay-per-token GPT is cheaper.]
  #sources[NVIDIA policy n = 3,340 · always-on bills from README monthly table · latency/refusals from shipped receipts.]
]
