#import "../theme.typ": *

= WHAT WE BUILT
== A moderation harness — every message, every time

#let edge = rgb("#D8E3E4")
#let soft2 = rgb("#F6F8F8")
#let mono(body) = text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK JP"), size: 9pt, fill: muted, body)
#let brands(h: 10pt) = stack(dir: ltr, spacing: 4pt,
  brand-nv(height: h), text(size: h - 1pt, fill: muted)[+], brand-fw(height: h))
#let arrow = align(horizon + center, text(size: 16pt, fill: muted)[→])

#let stage(n, icon, title, body, fill: soft2, stroke: 0.7pt + edge) = block(
  width: 100%, height: 248pt, breakable: false,
  inset: (x: 9pt, y: 9pt), radius: 8pt, fill: fill, stroke: stroke,
)[
  #set par(leading: 3.2pt, spacing: 0pt)
  #text(size: 12pt, weight: "bold", fill: hr)[#n  #icon  #title]
  #v(8pt)
  #body
]

#let lane(icon, title, count, model, note, main: false, future: false) = block(
  width: 100%, height: 64pt, breakable: false,
  inset: (x: 9pt, y: 7pt), radius: 7pt,
  fill: if main { soft } else { white },
  stroke: if main { 1.6pt + hr } else if future { 0.6pt + edge } else { 0.7pt + edge },
)[
  #set par(leading: 2.8pt, spacing: 0pt)
  #grid(columns: (16pt, 1fr, auto), column-gutter: 5pt, align: horizon,
    text(size: 14pt, fill: if future { muted } else { hr }, icon),
    text(size: 11pt, weight: "bold", fill: if future { muted } else { ink })[#title · #count],
    brands(h: 9pt),
  )
  #v(5pt)
  #text(size: 11pt, weight: "semibold", fill: if future { muted } else { ink }, model)
  #v(3pt)
  #text(size: 11pt, fill: muted, note)
]

#slide(setting: body => body)[
  #set par(leading: 3pt, spacing: 0pt)
  #grid(
    columns: (56pt, 10pt, 96pt, 10pt, 100pt, 10pt, 1fr, 10pt, 100pt, 10pt, 92pt),
    align: horizon,
    // Message
    block(width: 100%, height: 248pt, breakable: false, inset: 8pt, radius: 8pt,
      fill: soft, stroke: 1.4pt + hr,
    )[
      #set align(center + horizon)
      #stack(spacing: 10pt,
        text(size: 28pt)[✉],
        text(size: 11pt, weight: "bold", fill: hr)[Message],
      )
    ],
    arrow,
    // 0 Clean-up
    stage([0], [✨], [Clean-up], [
      #text(size: 11pt, fill: ink)[Always]
      #v(3pt)
      #text(size: 11pt, fill: muted)[~0 ms · \$0]
      #v(12pt)
      #text(size: 11pt, fill: muted)[NFKC · ZW\ look-alikes\ leetspeak\ base64\ emoji → text]
      #v(14pt)
      #mono([Wh47…dru9\ → cleaned])
    ], fill: soft),
    arrow,
    // 1 Rule gate
    stage([1], [⛔], [Rule gate], [
      #text(size: 11pt, fill: ink)[Luhn cards /\ jailbreaks]
      #v(10pt)
      #block(width: 100%, inset: 8pt, radius: 6pt, fill: white, stroke: 1.2pt + hr)[
        #text(size: 11pt, weight: "bold", fill: hr)[↗ side exit]
        #v(5pt)
        #text(size: 11pt, fill: ink)[instant verdict]
        #v(4pt)
        #text(size: 11pt, fill: muted)[0 ms · \$0]
        #v(3pt)
        #text(size: 11pt, fill: muted)[0.1% · 6 rows]
      ]
      #v(12pt)
      #mono([DAN → blocked])
    ]),
    arrow,
    // 2 Router
    block(width: 100%, height: 248pt, breakable: false,
      inset: (x: 9pt, y: 8pt), radius: 8pt, fill: soft2, stroke: 0.7pt + edge,
    )[
      #set par(leading: 2.8pt, spacing: 0pt)
      #text(size: 12pt, weight: "bold", fill: hr)[2  ⤴  Router]
      #h(6pt)
      #text(size: 11pt, fill: muted)[image? language?]
      #v(6pt)
      #stack(spacing: 5pt,
        lane([🖼], [Image], [wired · not benchmarked],
          [NVIDIA Nemotron 3.5 Super VL 120B],
          [on Fireworks · future work], future: true),
        lane([🌐], [Non-English], [2,967 rows],
          [fine-tuned NVIDIA Nemotron Nano 3 30B],
          [Fireworks · live-merge LoRA], main: true),
        lane([Aa], [English], [2,589 rows],
          [fine-tuned NVIDIA Nemotron Nano 3 30B],
          [same Fireworks deployment], main: true),
      )
    ],
    arrow,
    // 3 Confidence gate
    stage([3], [⚖], [Conf. gate], [
      #text(size: 11pt, fill: ink)[P(unsafe)]
      #v(3pt)
      #text(size: 13pt, weight: "bold", fill: c-ultra)[0.35–0.65]
      #v(10pt)
      #brands(h: 10pt)
      #v(6pt)
      #text(size: 11pt, weight: "semibold")[NVIDIA\ Nemotron 3\ Ultra 550B]
      #v(4pt)
      #text(size: 11pt, fill: muted)[Fireworks\ serverless]
      #v(12pt)
      #text(size: 18pt, weight: "bold", fill: c-ultra)[5.8%]
      #v(2pt)
      #text(size: 11pt, fill: muted)[325 / 5,562]
    ]),
    arrow,
    // 4 Verdict
    stage([4], [✔], [Verdict], [
      #text(size: 11pt, weight: "semibold", fill: hr)[JSON · every msg]
      #v(12pt)
      #stack(spacing: 7pt,
        text(size: 11pt, fill: ink)[label],
        text(size: 11pt, fill: ink)[categories],
        text(size: 11pt, fill: ink)[route],
        text(size: 11pt, fill: ink)[latency],
        text(size: 11pt, fill: ink)[cost],
      )
      #v(14pt)
      #mono([{"label":\ "unsafe"}])
    ], fill: soft, stroke: 1.4pt + hr),
  )
  #sources[Route shares on 5,562 evaluated rows · results/nvidia-v1/final.jsonl]
]
