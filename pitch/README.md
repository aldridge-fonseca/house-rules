# House Rules — pitch deck

Typst + [Touying](https://typst.app/universe/package/touying) (16:9). Twelve slides, one file each.

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
| `slides/02-why.typ` | Why moderation is needed (Grok incident, AI agents) |
| `slides/03-frontier.typ` | Why not just use a frontier model |
| `slides/04-results.typ` | Results — better on your policy |
| `slides/05-cheaper.typ` | Results — 32× cheaper |
| `slides/06-datasets.typ` | What we tested on (5,562 held-out cases) |
| `slides/07-business.typ` | Business value (big-tech spend, savings, trust) |
| `slides/08-harness.typ` | Our moderation harness |
| `slides/09-finetune.typ` | One Fireworks training job |
| `slides/10-examples-a.typ` | Examples — where frontier models get it wrong |
| `slides/11-examples-b.typ` | Examples — tricks and gray areas |

- `theme.typ` — colors and helpers. Brand colors: House Rules teal `hr`, NVIDIA green `nv`, Fireworks plum `fw`.
  **Series colors:** `c-ours`, `c-nano`, `c-ultra`, `c-gpt`, `c-opus`.
  Helpers: `stat`, `point`, `chip`, `hbars`, `sources`, `brand-nv`, `brand-fw` (text branding).
- `FACTS.md` — finished evidence map (only allowed numbers/claims).
- `assets/` — optional logo files kept for reference; slides use text branding.
  `assets/grok-incident.md` — concise public-source citations for slide 2.

Rules: keep slides skimmable, never let content overflow (page count must stay 11),
and only use numbers/claims that appear in `FACTS.md` / shipped `results/`.
