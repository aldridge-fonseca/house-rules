# Logo assets — sources and brand-usage notes

**Deck policy (2026-11-02):** slides use text branding (`NVIDIA` / `Fireworks AI`) via
`theme.typ` helpers. These SVG/PNG files are kept for reference only and are **not**
embedded in `pitch.pdf` until written authorization is recorded.

All files pulled 2026-11-02 from official sources only. Verified: both SVGs parse as valid
XML; PNG checked (3840×2160, RGBA transparent).

## fireworks-logo.svg
- File: horizontal lockup (spark mark + FIREWORKS wordmark), `fill="currentColor"`
  (renders black; recolor via CSS/color context).
- Source: official Fireworks AI website, https://fireworks.ai/ — path artwork extracted
  verbatim from the inline SVGs the site's own navbar serves (icon viewBox 0 0 41 20 +
  wordmark viewBox 50 0 110 20, combined into one 160×20 lockup).
- No public brand/press kit located. Press contact listed on their newsroom page
  (https://fireworks.ai/news): press@fireworks.ai — request written permission for
  submission-repo use.
- Brand color: identity is "Fireworks Purple" on a dark ground per the studio that designed
  it (https://monogram.io/work/fireworks-ai); official hex value unverified — confirm with
  Fireworks before recoloring.

## nvidia-logo.svg + nvidia-logo.png
- `nvidia-logo.svg`: horizontal lockup (eye mark + NVIDIA wordmark). Path artwork extracted
  verbatim from NVIDIA's official site (SVG symbol sprite served on nvidia.com, e.g.
  https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/). The eye-mark
  path carries NVIDIA's own embedded green fill `#74b71b`; wordmark inherits (black).
- `nvidia-logo.png`: official download from NVIDIA Newsroom media assets —
  https://nvidianews.nvidia.com/multimedia/corporate/nvidia-logos (asset "NVIDIA Logo
  horizontal, black, 16x9"), 3840×2160 RGBA transparent PNG.
- Brand guidance (NVIDIA Logo and Brand Guidelines,
  https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/ and
  https://international.download.nvidia.com/partnerforce-us/Brand-Guidelines/NVIDIA_LogoGuidelines.pdf):
  - Assets "may not be used in any manner that isn't expressly authorized in writing by
    NVIDIA." Approved examples: sponsor/co-branding, logo gardens, educational/research use
    of NVIDIA tech.
  - Clear space on all sides ≥ height of the "n"; minimum sizes (horizontal: 0.35in / 9mm /
    38px tall). Prefer the two-color version; one-color allowed on NVIDIA-green or
    low-reproduction contexts. Min contrast 1:10.
  - Do NOT recolor, distort, add effects, separate eye from wordmark, or resize elements
    independently. One logo per page/view. "NVIDIA" always uppercase.
  - Palette: NVIDIA Green PMS 376C / RGB 118,185,0 / HEX 76B900; dark gray 1E1E1E.

## nemotron-logo — NOT FOUND (none saved)
- No official standalone NVIDIA Nemotron product mark was located on official properties.
  NVIDIA typesets the name as text ("NVIDIA Nemotron™") on the product page
  (https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) and newsroom;
  research.nvidia.com/labs/nemotron serves only favicons (403-restricted), and
  developer.nvidia.com/nemotron exposes no logo asset.
- Recommendation: use the NVIDIA corporate logo + "Nemotron" in text (per NVIDIA trademark
  guidelines, ™ on first use). Third-party "Nemotron" SVGs (e.g. thesvg.org) exist but are
  not official — do not use.
