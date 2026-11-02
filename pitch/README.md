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
| `slides/02-why.typ` | Why moderation is needed |
| `slides/03-frontier.typ` | Why a frontier model fails (frontier numbers only) |
| `slides/04-harness.typ` | The harness: clean-up → rule gate → router → confidence gate → verdict |
| `slides/05-finetune.typ` | How we fine-tuned (Fireworks Training API, receipts in `results/training/`) |
| `slides/06-datasets.typ` | Test data (5,562 evaluated text rows; our synthetic disguised set) |
| `slides/07-attacks.typ` | Every attack form tested (8 forms, clean-up on/off) |
| `slides/08-results-quality.typ` | Results — quality (+ $/1M, paired CIs) |
| `slides/09-results-cost.typ` | Results — always-on cost, speed, coverage |
| `slides/10-ex-policy.typ` | Examples — your rulebook, not theirs |
| `slides/11-ex-disguise.typ` | Examples — disguised attacks |
| `slides/12-ex-harness.typ` | Examples — second opinion & fewer false alarms |

- `theme.typ` — colors and helpers. Brand colors: House Rules teal `hr`, NVIDIA green `nv`, Fireworks plum `fw`.
  **Series colors:** `c-ours`, `c-nano`, `c-ultra`, `c-gpt`, `c-opus`.
  Helpers: `stat`, `point`, `chip`, `hbars`, `sources`, `brand-nv`, `brand-fw` (text branding).
- `FACTS.md` — finished evidence map (only allowed numbers/claims).
- `assets/` — optional logo files kept for reference; slides use text branding.
  `assets/examples/meme1.svg` — original unevaluated vision illustration.
  `assets/grok-incident.md` — concise public-source citations for slide 2.

Rules: keep slides skimmable, never let content overflow (page count must stay 12),
and only use numbers/claims that appear in `FACTS.md` / shipped `results/`.
