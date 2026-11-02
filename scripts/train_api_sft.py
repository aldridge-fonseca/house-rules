#!/usr/bin/env python3
"""House Rules — Fireworks Training API LoRA SFT launcher.

Thin reproducible wrapper over the cookbook ``recipes/sft_loop.py``. Two jobs:

  --job nano     Dedicated trainer. NVIDIA Nemotron Nano 3 30B-A3B on the
                 2xB200 LoRA training shape. Paid GPU time ($13/GPU-h B200,
                 deployment rate verified via firectl; dedicated
                 training rate not separately published -> 85 min wall-clock
                 kill is enforced by the launch wrapper, not this script).
  --job glimmer  Serverless pooled trainer. Muse Glimmer 30B LoRA SFT is free
                 via the Serverless Training API until 2026-11-30 (verified on
                 docs.fireworks.ai/fine-tuning/training-api/cookbook/muse-glimmer
                 2026-11-02). In-session generation sampling would be billed,
                 so generation_eval stays off; eval = forward_backward loss on
                 the dev split, which is part of training.

Data: data/nvidia/filtered/train_sft.jsonl (19,830 rows),
      data/nvidia/filtered/dev_sft.jsonl (2,479 rows) — chat JSONL.

Defaults for the shipped Nano recipe: rank 8, alpha 32, 1 epoch, lr 1e-4,
batch 32, ``--max-examples`` None (all 19,830 rows). Trainer job id, wall-clock
and training cost: **not recorded**.

``config/train.yaml`` is a draft for the unused managed-SFT (``firectl sftj``)
path and must not be read as this Training API recipe.

The recipe streams rows from the local file; it never calls the disabled
dataset-upload endpoint. The final checkpoint is saved resumable+promotable and
promoted to accounts/<acct>/models/<output_model_id>, which is what
`firectl deployment create` serves as a live-merge LoRA.

Env: FIREWORKS_API_KEY must be set (load ``.env`` from the repo root). Never log it.

Optional dependency: the ``training`` package from the public Fireworks AI
cookbook is required only to actually launch a job (``--help`` works without
it). It is intentionally not in requirements.txt because it pulls heavyweight
training dependencies. Install once, pinned for reproducibility:

  git clone https://github.com/fw-ai/cookbook.git cookbook
  git -C cookbook checkout 57968a2237b663be33078d2747a28463983ae134
  pip install -e cookbook/training

Run (from repo root):
  python scripts/train_api_sft.py --job nano
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

COOKBOOK_REPO = "https://github.com/fw-ai/cookbook"
COOKBOOK_COMMIT = "57968a2237b663be33078d2747a28463983ae134"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_JSONL = os.path.join(REPO, "data/nvidia/filtered/train_sft.jsonl")
EVAL_JSONL = os.path.join(REPO, "data/nvidia/filtered/dev_sft.jsonl")

NANO_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/nemotron-nano-3-30b-a3b-262k-b200-lora"

PRESETS = {
    # Job A — headline. NVIDIA base, dedicated 2x B200 trainer.
    "nano": dict(
        serverless=False,
        base_model="accounts/fireworks/models/nemotron-nano-3-30b-a3b",
        tokenizer_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        # Strict JSON classifier: train in the no-thinking format so the served
        # LoRA (reasoning_effort=none) emits the label directly.
        renderer_name="nemotron3_disable_thinking",
        output_model_id="safety-one-nano-safety",
        log_path="/tmp/so-train-logs/nano-run",
        training_shape_id=NANO_TRAINING_SHAPE,
    ),
    # Job B — hedge. Free serverless LoRA SFT promo (until 11/30).
    "glimmer": dict(
        serverless=True,
        base_model="accounts/fireworks/models/muse-glimmer-30b",
        tokenizer_model="meta-models/Muse-Glimmer-30B",
        renderer_name="muse_glimmer",
        output_model_id="safety-one-glimmer-safety",
        log_path="/tmp/so-train-logs/glimmer-run",
        close_session_on_exit=True,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safety One Training API LoRA SFT")
    p.add_argument("--job", choices=sorted(PRESETS), required=True)
    p.add_argument("--train-path", default=TRAIN_JSONL)
    p.add_argument("--eval-path", default=EVAL_JSONL)
    p.add_argument("--output-model-id", default=None, help="Override preset output model id.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="Cookbook sft_loop default; no Nano-specific LR is published.")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--dcp-save-interval", type=int, default=150,
                   help="Periodic resumable checkpoint every N steps (0=off).")
    p.add_argument("--render-workers", type=int, default=2,
                   help="Local render workers; keep low on small hosts.")
    p.add_argument("--log-path", default=None, help="Override preset log dir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train_api_sft")

    if not os.environ.get("FIREWORKS_API_KEY"):
        log.error("FIREWORKS_API_KEY is not set; source .env from the repo root first")
        return 2
    for path in (args.train_path, args.eval_path):
        if not os.path.exists(path):
            log.error("missing dataset: %s", path)
            return 2

    try:
        import training.recipes.sft_loop as sft_loop
        from training.utils import RunnerConfig, TrainerConfig, WandBConfig
    except ImportError:
        log.error(
            "optional dependency `training` (Fireworks AI cookbook) is not "
            "installed; it is needed only to launch a training job. Install:\n"
            "  git clone %s cookbook\n"
            "  git -C cookbook checkout %s\n"
            "  %s -m pip install -e cookbook/training",
            COOKBOOK_REPO, COOKBOOK_COMMIT, sys.executable)
        return 2

    preset = PRESETS[args.job]
    log_path = args.log_path or preset["log_path"]
    os.makedirs(log_path, exist_ok=True)
    output_model_id = args.output_model_id or preset["output_model_id"]

    cfg = sft_loop.Config(
        log_path=log_path,
        base_model=preset["base_model"],
        tokenizer_model=preset["tokenizer_model"],
        renderer_name=preset["renderer_name"],
        dataset=args.train_path,
        evaluation_dataset=args.eval_path,
        serverless=preset["serverless"],
        close_session_on_exit=preset.get("close_session_on_exit", False),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_examples=args.max_examples,
        dcp_save_interval=args.dcp_save_interval,
        render_workers=args.render_workers,
        output_model_id=output_model_id,
        trainer=(TrainerConfig(training_shape_id=preset["training_shape_id"])
                 if preset.get("training_shape_id") else TrainerConfig()),
        wandb=WandBConfig(project="safety-one-sft"),
        runner=RunnerConfig(
            status_file=os.path.join(log_path, "status.json"),
            metrics_file=os.path.join(log_path, "metrics.jsonl"),
            output_model_path=os.path.join(log_path, "output_model.json"),
        ),
    )

    log.info(
        "launch job=%s base=%s shape=%s serverless=%s lora=%d/%d lr=%g epochs=%d "
        "batch=%d max_seq=%d out_model=%s",
        args.job, cfg.base_model,
        getattr(cfg.trainer, "training_shape_id", None) or "pool",
        cfg.serverless, cfg.lora_rank, cfg.lora_alpha, cfg.learning_rate,
        cfg.epochs, cfg.batch_size, cfg.max_seq_len, cfg.output_model_id,
    )
    metrics = sft_loop.main(cfg)
    log.info("DONE job=%s final=%s", args.job, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
