// House Rules / Touying 0.8.0 (latest registry release verified 2026-11-02).
// Content-slide pattern: = COLORED KICKER, then == Bold title, then body.
// The section heading is a kicker only: it never creates an extra page.
#import "@preview/touying:0.8.0": *
#import themes.simple: slide

#let hr = rgb("#007F86")
#let nv = rgb("#76B900")
// Fireworks live brand palette verified 2026-11-02 at https://fireworks.ai/.
// CSS: https://fireworks.ai/_next/static/css/8f752097094d0865.css
// :root --color-brand-bloom-800:#451234 (also bloom-950:#280016,
// bloom-100:#ffc0e7). Use the verified plum token, not the old guessed #6720FF.
#let fw = rgb("#451234")
#let ink = rgb("#102F38")
#let muted = rgb("#587078")
#let soft = rgb("#EAF5F4")

#let stat(big, label, color: hr) = stack(spacing: 12pt,
  text(size: 54pt, weight: "bold", fill: color, tracking: -1pt, big),
  text(size: 18pt, fill: ink, label),
)

#let point(icon, title, body, color: hr) = block(
  width: 100%, breakable: false, inset: 16pt, radius: 10pt, fill: soft,
  stroke: (top: 3pt + color),
)[
  #set par(leading: 3pt, spacing: 4pt)
  #stack(spacing: 12pt,
    text(size: 30pt, weight: "bold", fill: color, icon),
    text(size: 18pt, weight: "bold", fill: ink, title),
    text(size: 15pt, fill: muted, body),
  )
]

// The lower source band is reserved by the page margin; no flow displacement.
#let sources(body) = place(bottom + left, dy: 19pt,
  block(width: 100%, text(size: 7.5pt, fill: muted, body)))
// Text branding only — SVG/PNG logos retained under assets/ but unused in slides
// until written authorization is recorded (see assets/LOGOS.md).
#let brand-nv(height: 11pt) = text(size: height, weight: "bold", fill: nv, tracking: 0.4pt)[NVIDIA]
#let brand-fw(height: 11pt) = text(size: height, weight: "bold", fill: fw, tracking: 0.3pt)[Fireworks AI]
#let logo-nv(height) = brand-nv(height: height)
#let logo-fw(height) = brand-fw(height: height)

#let house-theme(body) = {
  set text(font: ("Noto Sans", "DejaVu Sans"), size: 21pt, fill: ink)
  set par(leading: 0.55em)
  show: themes.simple.simple-theme.with(
    aspect-ratio: "16-9",
    primary: hr,
    header: self => [
      #rect(width: 48pt, height: 4pt, fill: hr, stroke: none)
      #v(9pt)
      #text(size: 12pt, weight: "bold", fill: hr, tracking: 1.2pt,
        utils.display-current-heading(level: 1))
    ],
    header-right: none,
    footer: text(size: 9pt, fill: muted, weight: "semibold")[House Rules],
    footer-right: context text(size: 9pt, fill: muted, utils.slide-counter.display()),
    subslide-preamble: block(below: 22pt,
      text(size: 34pt, weight: "bold", fill: ink,
        utils.display-current-heading(level: 2))),
    config-page(
      margin: (left: 44pt, right: 44pt, top: 76pt, bottom: 59pt),
      header-ascent: 20pt,
      footer-descent: 24pt,
      fill: white,
    ),
    config-common(new-section-slide-fn: none, breakable: false, detect-overflow: true),
    config-colors(neutral-light: muted, neutral-darkest: ink),
    config-methods(init: (self: none, body) => {
      set text(font: ("Noto Sans", "DejaVu Sans"), size: 21pt, fill: ink)
      set par(leading: 0.55em)
      body
    }),
  )
  body
}

// ---- Series colors: use these EVERYWHERE a system appears (charts, tables, chips) ----
#let c-ours = hr                    // House Rules (ours)
#let c-nano = rgb("#B9D98A")        // baseline: NVIDIA Nemotron Nano, untuned (small)
#let c-ultra = rgb("#5E9400")       // baseline: NVIDIA Nemotron 3 Ultra 550B, untuned (large)
#let c-gpt = rgb("#9AA0A6")         // frontier: GPT-6.1 Sol
#let c-opus = rgb("#5F6368")        // frontier: Claude Opus 5.5
#let systems = (
  (name: "House Rules (ours)", color: c-ours),
  (name: "Nemotron Nano · untuned", color: c-nano),
  (name: "Nemotron Ultra 550B · untuned", color: c-ultra),
  (name: "GPT-6.1 Sol", color: c-gpt),
  (name: "Claude Opus 5.5", color: c-opus),
)

// Small colored label chip, e.g. #chip("FRONTIER", c-gpt)
#let chip(label, color) = box(inset: (x: 7pt, y: 3pt), radius: 4pt, fill: color,
  text(size: 11pt, weight: "bold", fill: white, label))

// Horizontal bar chart. rows: array of (label, value, color, note); value in [lo, hi].
// Example: #hbars(((“House Rules”, 95.3, c-ours, "95.3"), ...), lo: 50, hi: 100)
#let hbars(rows, lo: 0, hi: 100, width: 100%, bar-h: 22pt, label-w: 210pt, size: 14pt) = {
  grid(columns: (label-w, 1fr), row-gutter: 8pt, column-gutter: 10pt, align: horizon,
    ..rows.map(r => {
      let (label, value, color, note) = r
      let frac = calc.max(0.02, calc.min(1.0, (value - lo) / (hi - lo)))
      (
        align(right, text(size: size, fill: ink, label)),
        stack(dir: ltr, spacing: 6pt,
          rect(width: frac * 78%, height: bar-h, radius: 3pt, fill: color, stroke: none),
          align(horizon, text(size: size, weight: "bold", fill: ink, note))),
      )
    }).flatten()
  )
}
