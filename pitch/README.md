# House Rules — pitch deck

Typst + [Touying](https://typst.app/universe/package/touying) (16:9). Seven slides, one file each.

## Build
```bash
cd pitch
typst compile pitch.typ pitch.pdf                                 # PDF
typst compile --format png --ppi 70 pitch.typ 'preview-{p}.png'   # PNG previews
```
Requires Typst ≥ 0.15 (Touying 0.8.0 is fetched automatically on first compile).

## Layout
| File | Slide |
|---|---|
| `slides/01-title.typ` | Title |
| `slides/02-why.typ` | Why moderation is needed |
| `slides/03-frontier.typ` | Why a frontier model fails (frontier numbers only) |
| `slides/04-built.typ` | What we built (pipeline + training) |
| `slides/05-data.typ` | Test data (6,220 cases) |
| `slides/06-results-quality.typ` | Results — quality |
| `slides/07-results-cost.typ` | Results — cost, speed, coverage |

- `theme.typ` — colors and helpers. Brand colors: House Rules teal `hr`, NVIDIA green `nv`, Fireworks plum `fw`.
  **Series colors (use everywhere a system appears):** `c-ours`, `c-nano`, `c-ultra`, `c-gpt`, `c-opus`.
  Helpers: `stat`, `point`, `chip`, `hbars`, `sources`, `logo-nv`, `logo-fw`.
- `CONTRACT.md` — deck rules and the only allowed numbers (all from `results/`).
- `assets/` — official NVIDIA / Fireworks logos and source notes.

Rules: keep each slide skimmable (≤ ~15 words of body text), never let content overflow (page count must stay 7),
and only use numbers that appear in `CONTRACT.md` / `results/`.
