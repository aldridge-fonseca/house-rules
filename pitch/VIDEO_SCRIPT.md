# House Rules — 120-second video script (11 slides)

Pace ≈ 2.3 words/second. **Bold** = say it with emphasis (NVIDIA / Fireworks AI / headline numbers).

| # | Slide | Sec | Say |
|---|---|---:|---|
| 1 | Title | 9 | This is House Rules: your safety policy, enforced. A fine-tuned **NVIDIA Nemotron** model, trained and served on **Fireworks AI**. |
| 2 | Why this is needed | 10 | AI content moves fast. In eleven days, Grok produced an estimated three million sexualized images. And AI agents now talk to people, and to each other. |
| 3 | Why not a frontier model | 10 | Why not just use a frontier model? It costs **$182K a month** at scale, follows its lab's rules instead of yours, missing about 30% of harmful content, and Opus refused 13% of requests. |
| 4 | Results | 15 | We tested on 3,340 held-out cases from NVIDIA's safety policy, same cases for every model. House Rules scores **95.3**, versus 71.5 for GPT-6.1 Sol and 69 for Claude Opus. That's **plus 23.8 points**, with zero refusals. |
| 5 | 32× cheaper | 10 | And it's **32 times cheaper** per message: $56 per million, versus $1,820 for GPT. Below about five million messages a month, GPT is still cheaper. |
| 6 | Test data | 10 | 5,562 test cases: **NVIDIA's** policy test in 12 languages, plus our own synthetic set of 2,222 disguised attacks. |
| 7 | Business value | 11 | Meta spends about $5 billion a year on safety; TikTok over $2 billion. At 100 million messages a month, House Rules saves **$2.1 million a year**, with an audit trail on every verdict. |
| 8 | Harness | 13 | Every message runs through our harness. Clean-up undoes disguises, rules block known attacks instantly, a router picks the model, and when **Nemotron Nano** is unsure, **Nemotron Ultra on Fireworks serverless** gives a second opinion. |
| 9 | Fine-tuning | 12 | We fine-tuned **NVIDIA Nemotron Nano** in one **Fireworks Training API** job: NVIDIA's safety data, LoRA, one epoch on two **NVIDIA B200s**, 19 minutes. Accuracy went from **82.7 to 95.3**. |
| 10 | Examples A | 9 | Legal advice on a divorce, or a stereotype asked in Japanese: frontier models say "safe". Trained on your policy, we flag both. |
| 11 | Examples B | 11 | Base64-hidden attacks get decoded, and jailbreaks are blocked instantly at zero cost. Code and results: **github.com/aldridge-fonseca/house-rules**. |
| | **Total** | **120** | |

Recording tips: export slides from `pitch.pdf` at 1080p, hold each slide for its seconds, and record the voice-over in one take.
