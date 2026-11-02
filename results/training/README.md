# Training receipts — House Rules specialist (NVIDIA Nemotron Nano 3 30B-A3B LoRA)

Run on 2026-11-02 with the Fireworks Training API, `scripts/train_api_sft.py --job nano`.

| Field | Value |
|---|---|
| Base model | `accounts/fireworks/models/nemotron-nano-3-30b-a3b` |
| Training shape | `nemotron-nano-3-30b-a3b-262k-b200-lora` (dedicated, 2x NVIDIA B200) |
| Method | supervised fine-tuning, LoRA rank 8 / alpha 32, lr 1e-4, batch 32, max seq 2,048, 1 epoch, thinking disabled |
| Input file | `data/nvidia/filtered/train_sft.jsonl` — 19,830 rows (NVIDIA Nemotron Content Safety v3.5 after test-overlap filtering) |
| Rows consumed | 19,806 (24 removed by the trainer's sequence-length/format filter) |
| Steps | 620 optimizer steps |
| Held-out eval (2,475 dev rows) | loss **0.753 before training → 0.104 after** (perplexity 2.12 → 1.11) |
| Wall clock | trainer job created 10:44:17, dataset loaded 10:46:27, training done 10:59:16, final eval 11:02:29, trainer deleted 11:03:13 (PT) |
| Output | LoRA model `safety-one-nano-safety`, served as a live-merge LoRA on a Fireworks on-demand deployment (1x B200) |

Files: `nano_sft_events.log` (lifecycle events, account id redacted), `nano_sft_metrics.jsonl` (per-step train loss / tokens per second, plus eval loss at step 0 and 620).
Billed training cost was not separately metered and is **not recorded**; inference GPU pricing is not a training invoice. The logged trainer lifecycle (creation through deletion) is **18 min 56 s**, including setup and evaluations; it is not a measurement of optimizer time alone. The exact historical cookbook/trainer revision is not recorded.

Provenance: the 622 metric JSON records were copied verbatim from the original trainer log and independently compared byte-for-byte. The lifecycle extract preserves logged event values, with local paths and account references redacted. The original local log is not published.
